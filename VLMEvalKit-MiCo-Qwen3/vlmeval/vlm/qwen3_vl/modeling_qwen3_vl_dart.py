# coding=utf-8
"""DART (EMNLP'25) in-LLM visual-token pruning for Qwen3-VL.

Ported from qwen2_5_vl/modeling_qwen2_5_vl_dart.py, but rebuilt on the Qwen3-VL FastV skeleton
(modeling_qwen3_vl_fastv.py) so it inherits the same hard-won Qwen3-specific correctness fixes:

  * P2 -- recompute RoPE from the pruned position_ids; never gather Qwen3-VL cos/sin on dim 2
    (they are [B, S, head_dim], so dim 2 is head_dim, not the sequence).
  * P3 -- rebuild the post-prune causal mask with attention_mask / past_key_values / position_ids
    dropped TOGETHER (the packed-sequence poisoning guard).
  * P5 -- accept, inject and re-prune the DeepStack side-features in sync with the prune mask.
  * P6 -- explicit set_visual_token_positions() index vector instead of the Qwen2.5-VL
    `system_prompt_length=15` offset window (Qwen3-VL's first image token sits at index 4, and
    multi-image prompts are not a contiguous run at all).

Selection criterion (unchanged from the DART reference):
  dart          -- pivot = top-`pivot_image_token` visual tokens by K-state L1 norm, then each
                   pivot pulls in the visual tokens LEAST cosine-similar to it (anti-cosine
                   diversity) until `target` tokens are kept.
  dart_coverage -- same pivots, but the fill switches to SCOPE marginal-coverage gain (from TOPS).
  cover_llm     -- pure coverage greedy (NO DART pivots); the layer is set by COVER_LLM_LAYER.

Like the Qwen2.5-VL port, the DART reference modeling is NOT copied wholesale (it depends on newer
masking_utils APIs); only the "which tokens to keep" rule is DART's. Everything structural comes
from the verified FastV path.
"""

import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    from transformers.models.qwen3_vl.modeling_qwen3_vl import create_causal_mask as _create_causal_mask_impl
    _BASE_MODEL_CLS = Qwen3VLModel
    _USING_QWEN3 = True
except ImportError:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel as _BASE_MODEL_CLS
    _create_causal_mask_impl = None
    _USING_QWEN3 = False

logger = logging.get_logger(__name__)

DART_REV = "qwen3vl-dart-20260826/P2P3P5P6"

_FALSEY = ('', '0', 'false', 'False', 'FALSE', 'no', 'off')


def _env_on(name: str, default: str = '0') -> bool:
    return str(os.environ.get(name, default)).strip() not in _FALSEY


def _past_length(cache) -> int:
    if cache is None:
        return 0
    try:
        return int(cache.get_seq_length())
    except Exception:
        return 0


class DARTQwen3VLModel(_BASE_MODEL_CLS):
    """DART in-LLM pruning for Qwen3-VL. Runs K layers at full length, selects visual tokens with
    DART's pivot+diversity (or coverage) rule at layer K, prunes, then continues -- DeepStack-aware."""

    def __init__(self, config, visual_token_num: int, use_coverage: bool = False, pure_cover: bool = False):
        super().__init__(config)
        lm = self.language_model
        self.embed_tokens = lm.embed_tokens
        self.layers = lm.layers
        self.norm = lm.norm
        self.rotary_emb = lm.rotary_emb
        self._lm_update_causal_mask = getattr(lm, '_update_causal_mask', None)

        # P6: explicit visual-position window (no system_prompt_length offset).
        self.visual_token_positions: Optional[torch.LongTensor] = None
        self._visual_positions_fresh = False

        self.visual_token_length = 1296   # nominal; logging/back-compat only
        self.visual_token_num = 0.0
        self.visual_token_len = 0.0

        self.target = max(1, int(visual_token_num))
        self.use_coverage = use_coverage
        self.pure_cover = pure_cover
        self.pivot_image_token = 4        # DART image pivot count

        # DART prunes AFTER decoder layer (self.K - 1). layer_idx below is 1-based, so the prune
        # fires when layer_idx == self.K. dart/dart_coverage use K=2 (the reference default); cover_llm
        # exposes the layer via COVER_LLM_LAYER (="prune after layer L" -> self.K = L + 1).
        self.K = int(os.environ.get('COVER_LLM_LAYER', '2')) + 1 if pure_cover else 2
        self.dart_debug = _env_on('ENABLE_DEBUG', '0')

        tag = 'cover_llm' if pure_cover else ('dart_coverage' if use_coverage else 'dart')
        print(f"[DART Init] rev={DART_REV} variant={tag} Qwen3-VL K={self.K} "
              f"target T={self.target} pivots={self.pivot_image_token} "
              f"(L={len(self.layers)} text layers)")

        self.reset_state()

    def reset_state(self):
        self.prefill_done = False
        self.current_visual_length: Optional[int] = None
        self.original_visual_length: Optional[int] = None
        self.retained_visual_length: Optional[int] = None
        self.retain_token_indices: Optional[torch.LongTensor] = None
        self.visual_token_num = 0.0
        self.visual_token_len = 0.0

    def set_visual_token_positions(self, positions: Optional[torch.LongTensor]) -> None:
        """Absolute positions (post-stage-1 sequence) of the surviving image tokens. Shared window
        API with fastv / tops_variant3 / tops -- an explicit index vector, correct for multi-image prompts."""
        if positions is None:
            self.visual_token_positions = None
            self._visual_positions_fresh = False
            return
        self.visual_token_positions = positions.detach().reshape(-1).to(torch.long)
        self._visual_positions_fresh = True

    def _get_causal_mask(self, input_tensor, attention_mask, cache_position, past_key_values, position_ids=None):
        # P3b: drop position_ids on exactly the packed-sequence trigger (position_ids != None,
        # attention_mask None, past_key_values None); gapped post-prune positions would otherwise be
        # read as document boundaries and blind the answer tokens to the retained visual tokens.
        if position_ids is not None and attention_mask is None and past_key_values is None:
            position_ids = None
        if _create_causal_mask_impl is not None:
            return _create_causal_mask_impl(
                self.config, input_tensor, attention_mask, cache_position,
                past_key_values=past_key_values, position_ids=position_ids
            )
        elif self._lm_update_causal_mask is not None:
            return self._lm_update_causal_mask(
                attention_mask, input_tensor, cache_position, past_key_values, False
            )
        return None

    def _dart_select(self, hidden_states: torch.Tensor, decoder_layer: nn.Module,
                     visual_positions: torch.LongTensor) -> torch.LongTensor:
        """Return LOCAL indices (into visual_positions, 0..Nv-1) of the visual tokens to KEEP.

        DART: pivots = top-`pivot_image_token` visual tokens by K-state (raw k_proj) L1 norm; then
        each pivot absorbs the visual tokens least cosine-similar to it, until `target` are kept.
        dart_coverage / cover_llm swap the fill for SCOPE coverage gain (with / without pivots).
        """
        self_attn = decoder_layer.self_attn
        attn_input = hidden_states
        ln = getattr(decoder_layer, 'input_layernorm', None)
        if ln is not None:
            attn_input = ln(attn_input)

        # K states (raw projection, pre-RoPE and pre-k_norm): DART's pivot criterion is key-state
        # magnitude, so k_norm is deliberately NOT applied (it would normalise the magnitude away).
        key_states = self_attn.k_proj(attn_input)[0]              # [seq, nkv*hd]
        k_img = key_states.index_select(0, visual_positions)      # [Nv, nkv*hd]

        feat = self.norm(hidden_states)[0].index_select(0, visual_positions)  # [Nv, D]
        Nv = feat.shape[0]
        device = feat.device
        K_target = min(self.target, Nv)

        featn = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        sim = torch.matmul(featn, featn.t())                      # [Nv, Nv] cosine

        if self.pure_cover:
            # Pure coverage greedy (== divprune_coverage), no DART pivots.
            simc = torch.clamp(sim, min=0.0)
            avail = torch.ones(Nv, dtype=torch.bool, device=device)
            g = torch.zeros(Nv, device=device)
            selc = []
            i0 = int(torch.argmax(simc.sum(dim=1)).item())
            selc.append(i0)
            avail[i0] = False
            g = torch.maximum(g, simc[i0])
            while len(selc) < K_target and avail.any():
                gain = torch.clamp(simc - g.unsqueeze(0), min=0.0).sum(dim=1)
                gain[~avail] = -float('inf')
                j = int(torch.argmax(gain).item())
                selc.append(j)
                avail[j] = False
                g = torch.maximum(g, simc[j])
            return torch.tensor(sorted(selc)[:K_target], device=device, dtype=torch.long)

        # ── DART pivots: top-`pivot_image_token` by K-state L1 norm ──
        k_l1 = torch.norm(k_img.float(), p=1, dim=-1)             # [Nv]
        npv = min(self.pivot_image_token, Nv)
        pivots = k_l1.topk(npv).indices
        chosen = set(pivots.tolist())

        if not self.use_coverage:
            # DART: each already-chosen token pulls in its LEAST similar (most diverse) neighbours.
            per = max(1, (K_target - len(chosen)) // max(1, len(chosen)))
            valid = set(range(Nv)) - chosen
            for it in list(chosen):
                if len(chosen) >= K_target or not valid:
                    break
                vlist = list(valid)
                cs = -sim[it, torch.tensor(vlist, device=device)]   # anti-similarity
                take = min(per, len(vlist))
                topk = cs.topk(take).indices
                add = [vlist[i] for i in topk.tolist()]
                chosen.update(add)
                valid.difference_update(add)
        else:
            # dart_coverage: fill with SCOPE marginal-coverage gain.
            avail = torch.ones(Nv, dtype=torch.bool, device=device)
            for i in chosen:
                avail[i] = False
            sel = list(chosen)
            if sel:
                g = sim[torch.tensor(sel, device=device)].max(dim=0).values
            else:
                g = torch.zeros(Nv, device=device)
            while len(chosen) < K_target and avail.any():
                gain = torch.clamp(sim - g.unsqueeze(0), min=0.0).sum(dim=1)
                gain[~avail] = -float('inf')
                j = int(torch.argmax(gain).item())
                chosen.add(j)
                avail[j] = False
                g = torch.maximum(g, sim[j])

        return torch.tensor(sorted(chosen)[:K_target], device=device, dtype=torch.long)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        past_seen_tokens = _past_length(past_key_values)
        if past_seen_tokens == 0:
            self.reset_state()

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else getattr(self.config, 'output_hidden_states', False)
        )
        use_cache = use_cache if use_cache is not None else getattr(self.config, 'use_cache', True)
        return_dict = return_dict if return_dict is not None else getattr(self.config, 'use_return_dict', True)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if self.is_gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        seq_length = inputs_embeds.shape[1]
        is_prefill = (past_seen_tokens == 0) and (seq_length > 1)

        # Qwen3-VL 4D position_ids: [4, B, S] where [0]=text, [1:]=mrope T/H/W
        text_position_ids = None
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        elif position_ids.dim() == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        # else: already [3, B, S]

        causal_mask = self._get_causal_mask(
            inputs_embeds, attention_mask, cache_position, past_key_values, position_ids=text_position_ids
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None

        # ── Resolve this sample's visual window ──
        visual_positions: Optional[torch.LongTensor] = None
        n_visual = 0
        do_prune = False

        if is_prefill:
            self.prefill_done = True
            pushed = self.visual_token_positions if self._visual_positions_fresh else None
            self._visual_positions_fresh = False
            if pushed is None:
                logger.warning_once(
                    "[DART] no visual token positions pushed for this sample; this prefill runs "
                    "UNPRUNED (expected only for text-only prompts)."
                )
            else:
                pushed = pushed.reshape(-1).to(device=hidden_states.device, dtype=torch.long)
                if pushed.numel() and int(pushed.max()) >= seq_length:
                    raise RuntimeError(
                        f"[DART] stale visual_token_positions: max index {int(pushed.max())} >= "
                        f"seq_len {seq_length}."
                    )
                visual_positions = pushed
                n_visual = int(pushed.numel())

            if visual_positions is not None and visual_pos_masks is not None and self.dart_debug:
                _pm = visual_pos_masks[0].nonzero(as_tuple=False).squeeze(-1)
                assert _pm.numel() == n_visual and bool(
                    torch.equal(_pm.to(visual_positions.device), visual_positions)
                ), "visual_pos_masks and visual_token_positions disagree (DeepStack would misalign)"

            if n_visual > 0:
                self.original_visual_length = n_visual
                self.current_visual_length = n_visual
                do_prune = (self.target > 0) and (n_visual > self.target)
                if self.dart_debug:
                    tag = 'cover_llm' if self.pure_cover else ('dart_coverage' if self.use_coverage else 'dart')
                    print(f"\n[DART Qwen3-VL] rev={DART_REV} variant={tag} prefill seq={seq_length} "
                          f"visual N={n_visual} K={self.K} target={self.target} "
                          f"prune={'yes' if do_prune else 'no'} "
                          f"deepstack={0 if deepstack_visual_embeds is None else len(deepstack_visual_embeds)}",
                          flush=True)

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_idx = decoder_layer.self_attn.layer_idx + 1   # 1-BASED

            # Layer runs at full length; DART selects on its OUTPUT (like the Qwen2.5 reference).
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            # ── P5a: DeepStack injection after text layers 0/1/2 (upstream UP:929) ──
            _ds = layer_idx - 1
            if (
                deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and 0 <= _ds < len(deepstack_visual_embeds)
            ):
                assert deepstack_visual_embeds[_ds].shape[0] == int(visual_pos_masks.sum()), \
                    "deepstack rows out of sync with visual_pos_masks"
                hidden_states = self.language_model._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[_ds]
                )

            # ── Prune AFTER layer K ──
            if is_prefill and do_prune and layer_idx == self.K and visual_positions is not None:
                device = hidden_states.device
                orig_seq_len = hidden_states.shape[1]

                keep_local = self._dart_select(hidden_states, decoder_layer, visual_positions)
                kept_visual_abs = visual_positions.index_select(0, keep_local.to(visual_positions.device))

                retain_mask = torch.ones(orig_seq_len, dtype=torch.bool, device=device)
                retain_mask[visual_positions] = False
                retain_mask[kept_visual_abs] = True
                retain_token_indices = retain_mask.nonzero(as_tuple=False).squeeze(-1)

                hidden_states = hidden_states.index_select(1, retain_token_indices).contiguous()
                new_len = hidden_states.shape[1]

                position_ids = position_ids[:, :, retain_token_indices]
                if text_position_ids is not None:
                    text_position_ids = text_position_ids[:, retain_token_indices]
                cache_position = cache_position[:new_len]

                # P2: recompute RoPE from the pruned position_ids (never gather cos/sin on dim 2).
                position_embeddings = self.rotary_emb(hidden_states, position_ids)

                # P3: rebuild causal mask with attention_mask / past_key_values / position_ids all None.
                causal_mask = self._get_causal_mask(
                    hidden_states, None, cache_position, None, position_ids=None
                )
                if _env_on('ENABLE_DEBUG') and causal_mask is not None and causal_mask.dtype == torch.bool:
                    assert bool(causal_mask[0, 0, -1].sum() == hidden_states.shape[1]), \
                        "post-prune causal mask is not plain-causal (packed-sequence poisoning)"

                # P5b: re-prune the DeepStack side-channel in sync (mask-driven).
                if visual_pos_masks is not None:
                    _vis_pos = visual_pos_masks[0].nonzero(as_tuple=True)[0]
                    _surv = torch.isin(_vis_pos, retain_token_indices)
                    if deepstack_visual_embeds is not None:
                        deepstack_visual_embeds = [d[_surv] for d in deepstack_visual_embeds]
                    visual_pos_masks = visual_pos_masks[:, retain_token_indices]

                visual_positions = torch.searchsorted(retain_token_indices, kept_visual_abs)
                self.visual_token_positions = visual_positions
                self.retain_token_indices = retain_token_indices
                self.current_visual_length = int(kept_visual_abs.numel())
                self.retained_visual_length = self.current_visual_length
                self.visual_token_num = float(self.current_visual_length)
                self.visual_token_len = float(self.current_visual_length)

                if self.dart_debug:
                    print(f"[DART PRUNE] layer {layer_idx}: visual {n_visual} -> "
                          f"{self.current_visual_length} (target={self.target}); "
                          f"seq {orig_seq_len} -> {new_len}", flush=True)

        hidden_states = self.norm(hidden_states)

        if is_prefill and not do_prune and n_visual > 0:
            self.visual_token_num = float(n_visual)
            self.visual_token_len = float(n_visual)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=None,
        )
