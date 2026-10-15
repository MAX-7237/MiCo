# coding=utf-8
"""DUET-VLM Stage 2 (PyramidDrop) for Qwen3-VL.

DUET-VLM (CVPR2026) = two complementary visual-token reduction stages:
  * Stage 1 (V2V, VisionZip): dominant top-k + contextual cluster-merge, done in the wrapper's
    Stage-1 selection branch (modeling_qwen3_vl.py, `pruning_method == 'duet'`). The wrapper then
    compacts the sequence and hands this decoder the surviving visual positions + DeepStack embeds.
  * Stage 2 (T2V, PyramidDrop, THIS file): progressively drop visual tokens at a schedule of LLM
    decoder layers, ranking each token by the attention it RECEIVES from the trailing text
    (question) tokens, so that the per-layer AVERAGE visual token count equals the target `_t`.

This file mirrors `modeling_qwen3_vl_fastv.py` (same Qwen3-VL plumbing: q_norm/k_norm probe,
MRoPE recompute, causal-mask rebuild, DeepStack inject/re-prune, explicit non-contiguous visual
window for multi-image prompts). The ONLY behavioural differences vs FastV are:
  * prune at EVERY layer in `self.layer_list` (default [round(0.5L), round(0.75L)] = [18, 27] for
    L=36), not once at K;
  * the retained count per layer comes from an absolute schedule calibrated to the layer-average
    budget (port of qwen2_5vl_duet_official.py:_build_pdrop_token_schedule), not FastV's R;
  * the importance probe uses ALL trailing text tokens as queries (PyramidDrop T2V), not only the
    last query row.
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
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb as _apply_rotary_pos_emb_impl
    from transformers.models.qwen3_vl.modeling_qwen3_vl import create_causal_mask as _create_causal_mask_impl
    from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv as _repeat_kv
    _BASE_MODEL_CLS = Qwen3VLModel
    _USING_QWEN3 = True
except ImportError:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel as _BASE_MODEL_CLS
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv as _repeat_kv
    _apply_rotary_pos_emb_impl = None
    _create_causal_mask_impl = None
    _USING_QWEN3 = False

logger = logging.get_logger(__name__)

DUET_REV = "qwen3vl-rebuild-20260724/duet-pdrop"

_FALSEY = ('', '0', 'false', 'False', 'FALSE', 'no', 'off')


def _env_on(name: str, default: str = '0') -> bool:
    return str(os.environ.get(name, default)).strip() not in _FALSEY


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin, unsqueeze_dim: int = 1):
    if _apply_rotary_pos_emb_impl is not None:
        return _apply_rotary_pos_emb_impl(q, k, cos, sin)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _past_length(cache) -> int:
    if cache is None:
        return 0
    try:
        return int(cache.get_seq_length())
    except Exception:
        return 0


class DuetQwen3VLModel(_BASE_MODEL_CLS):
    """DUET-VLM PyramidDrop stage-2 for Qwen3-VL."""

    def __init__(self, config, visual_token_num: int):
        super().__init__(config)
        lm = self.language_model
        self.embed_tokens = lm.embed_tokens
        self.layers = lm.layers
        self.norm = lm.norm
        self.rotary_emb = lm.rotary_emb
        self._lm_update_causal_mask = getattr(lm, '_update_causal_mask', None)

        self.visual_token_positions: Optional[torch.LongTensor] = None
        self._visual_positions_fresh = False

        self.visual_token_num = 0.0
        self.visual_token_len = 0.0
        self.duet_debug = _env_on('ENABLE_DEBUG', '1')

        try:
            visual_token_num = int(visual_token_num)
        except (TypeError, ValueError):
            raise ValueError(f"visual_token_num must be an integer, got {visual_token_num!r}")
        self.target_visual_tokens = visual_token_num

        n_layers = getattr(config, 'num_hidden_layers', None) or config.text_config.num_hidden_layers
        self.num_text_layers = len(self.layers) if len(self.layers) else int(n_layers)

        # PyramidDrop layer boundaries at 0.5 / 0.75 of the depth (Qwen2.5-VL used [14,21] on 28
        # layers; the same fractions give [18,27] on Qwen3-VL-8B's 36 layers). Cumulative keep
        # ratios [1.0, 0.5, 0.0] reproduce the repo's duet_640 preset (model.py:376).
        L = self.num_text_layers
        env_layers = os.environ.get('DUET_LAYER_LIST', '')
        if env_layers.strip():
            self.layer_list = [int(x) for x in env_layers.replace(',', ' ').split()]
        else:
            self.layer_list = sorted({max(1, min(L - 1, int(round(f * L)))) for f in (0.5, 0.75)})
        env_ratios = os.environ.get('DUET_RATIO_LIST', '')
        if env_ratios.strip():
            tail = [float(x) for x in env_ratios.replace(',', ' ').split()]
        else:
            tail = [0.5, 0.0]
        self.image_token_ratio_list = [1.0] + tail
        assert len(self.image_token_ratio_list) == len(self.layer_list) + 1, (
            f"DUET needs one initial ratio + one per prune layer; "
            f"ratios={self.image_token_ratio_list} layers={self.layer_list}"
        )

        print(f"[DUET Init] rev={DUET_REV} Qwen3-VL (L={self.num_text_layers} text layers), "
              f"layer_list={self.layer_list}, ratios={self.image_token_ratio_list}, "
              f"target_avg T={self.target_visual_tokens}", flush=True)

        self.reset_state()

    def reset_state(self):
        self.prefill_done = False
        self.current_visual_length: Optional[int] = None
        self.original_visual_length: Optional[int] = None
        self.retain_token_indices: Optional[torch.LongTensor] = None
        self.visual_token_sum = 0.0
        self.visual_token_num = 0.0
        self.visual_token_len = 0.0
        self._schedule: Optional[List[int]] = None

    def set_visual_token_positions(self, positions: Optional[torch.LongTensor]) -> None:
        if positions is None:
            self.visual_token_positions = None
            self._visual_positions_fresh = False
            return
        self.visual_token_positions = positions.detach().reshape(-1).to(torch.long)
        self._visual_positions_fresh = True

    def _get_causal_mask(self, input_tensor, attention_mask, cache_position, past_key_values, position_ids=None):
        # Same scoped guard as FastV: gapped post-prune positions must not trigger masking_utils'
        # packed-sequence (block-diagonal) branch, which would blind answer tokens to visual tokens.
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

    def _build_schedule(self, n_visual: int) -> List[int]:
        """Absolute per-stage keep targets calibrated so the LAYER-AVERAGE equals the budget.

        Port of qwen2_5vl_duet_official.py:_build_pdrop_token_schedule for batch=1. Segments are
        [layer_list[0], gaps..., L - layer_list[-1]] layers long; the residual correction nudges an
        intermediate stage by the smallest integer that hits target_avg*L exactly.
        """
        ratios = self.image_token_ratio_list
        schedule = [int(n_visual)] + [int(n_visual * float(r)) for r in ratios[1:]]
        T, L = self.target_visual_tokens, self.num_text_layers
        if T is None or T <= 0 or not self.layer_list:
            return schedule
        segs = [self.layer_list[0]]
        segs += [self.layer_list[i] - self.layer_list[i - 1] for i in range(1, len(self.layer_list))]
        segs += [L - self.layer_list[-1]]
        if len(segs) != len(schedule) or any(s <= 0 for s in segs):
            logger.warning_once(f"[DUET] invalid schedule layers={self.layer_list} L={L}; no calibration")
            return schedule
        target_total = int(round(float(T) * L))
        measured = sum(segs[s] * schedule[s] for s in range(len(schedule)))
        residual = target_total - measured
        if residual != 0:
            for stage in range(len(schedule) - 2, 0, -1):
                width = segs[stage]
                if residual % width != 0:
                    continue
                candidate = schedule[stage] + residual // width
                if schedule[stage + 1] <= candidate <= schedule[stage - 1]:
                    schedule[stage] = candidate
                    break
        return schedule

    def _text_to_visual_probe(
        self,
        hidden_states: torch.Tensor,
        decoder_layer: nn.Module,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        visual_positions: torch.LongTensor,
    ) -> torch.Tensor:
        """PyramidDrop T2V importance = attention RECEIVED by each visual token from the trailing
        text (question) tokens, measured with this layer's own projections.

        Replicates Qwen3VLTextAttention (q_norm/k_norm before RoPE; head counts from config since
        Qwen3VLTextAttention exposes no num_heads). Query rows = every position AFTER the last
        visual token (the question); softmax over ALL keys (DUET full form), then slice the visual
        columns. Robust to multi-image (visual_positions need not be contiguous).
        """
        self_attn = decoder_layer.self_attn
        attn_input = hidden_states
        input_layernorm = getattr(decoder_layer, 'input_layernorm', None)
        if input_layernorm is not None:
            attn_input = input_layernorm(attn_input)

        bsz, seq_len, _ = attn_input.shape
        head_dim = self_attn.head_dim
        text_config = getattr(self.config, 'text_config', self.config)
        num_heads = getattr(text_config, 'num_attention_heads', None) or (self_attn.q_proj.out_features // head_dim)
        num_kv_heads = getattr(text_config, 'num_key_value_heads', None) or (self_attn.k_proj.out_features // head_dim)
        num_groups = getattr(self_attn, 'num_key_value_groups', None) or max(1, num_heads // num_kv_heads)

        q_norm = getattr(self_attn, 'q_norm', None)
        k_norm = getattr(self_attn, 'k_norm', None)
        if _USING_QWEN3 and (q_norm is None or k_norm is None):
            raise RuntimeError(
                "Qwen3VLTextAttention is expected to expose q_norm/k_norm; without them the DUET "
                "probe measures a different function than the model's own attention."
            )

        query_states = self_attn.q_proj(attn_input).view(bsz, seq_len, num_heads, head_dim)
        key_states = self_attn.k_proj(attn_input).view(bsz, seq_len, num_kv_heads, head_dim)
        if q_norm is not None:
            query_states = q_norm(query_states)
        if k_norm is not None:
            key_states = k_norm(key_states)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rope(query_states, key_states, cos, sin)
        key_states = _repeat_kv(key_states, num_groups)

        scaling = getattr(self_attn, 'scaling', head_dim ** -0.5)

        # Rater rows: text tokens after the LAST visual token (the trailing question). Fall back to
        # the last row if there is no trailing text.
        last_vis = int(visual_positions.max())
        if last_vis + 1 < seq_len:
            rater_idx = torch.arange(last_vis + 1, seq_len, device=hidden_states.device)
        else:
            rater_idx = torch.arange(seq_len - 1, seq_len, device=hidden_states.device)

        q_raters = query_states[:, :, rater_idx, :]                      # [B, H, R, D]
        attn_scores = torch.matmul(q_raters, key_states.transpose(-2, -1)) * scaling  # [B, H, R, S]
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
        vis_weights = attn_weights[..., visual_positions]               # [B, H, R, V]
        # mean over heads, then over rater rows, then over batch -> [V]
        visual_importance = vis_weights.mean(dim=1).mean(dim=1).mean(dim=0)
        return visual_importance.float()

    def _prune_visual(
        self,
        hidden_states,
        importance,
        visual_positions,
        retained_num,
        position_ids,
        text_position_ids,
        cache_position,
        visual_pos_masks,
        deepstack_visual_embeds,
    ):
        """Drop all visual tokens except the top-`retained_num` by importance, then rebuild
        position_ids / RoPE / causal mask / DeepStack in sync. Mirrors FastV's prune block."""
        device = hidden_states.device
        orig_seq_len = hidden_states.shape[1]
        n_visual = int(visual_positions.numel())
        retained_num = int(max(0, min(retained_num, n_visual)))

        if retained_num > 0:
            selected_t = torch.sort(importance.topk(retained_num).indices).values
        else:
            selected_t = torch.empty(0, dtype=torch.long, device=importance.device)
        selected_t = selected_t.to(device=device, dtype=torch.long)

        kept_visual_abs = visual_positions.index_select(0, selected_t)
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

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        causal_mask = self._get_causal_mask(hidden_states, None, cache_position, None, position_ids=None)

        if visual_pos_masks is not None:
            _vis_pos = visual_pos_masks[0].nonzero(as_tuple=True)[0]
            _surv = torch.isin(_vis_pos, retain_token_indices)
            if deepstack_visual_embeds is not None:
                deepstack_visual_embeds = [d[_surv] for d in deepstack_visual_embeds]
            visual_pos_masks = visual_pos_masks[:, retain_token_indices]

        new_visual_positions = torch.searchsorted(retain_token_indices, kept_visual_abs)
        return (
            hidden_states, new_visual_positions, position_ids, text_position_ids,
            cache_position, position_embeddings, causal_mask, visual_pos_masks,
            deepstack_visual_embeds,
        )

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

        text_position_ids = None
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        elif position_ids.dim() == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]

        causal_mask = self._get_causal_mask(
            inputs_embeds, attention_mask, cache_position, past_key_values, position_ids=text_position_ids
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None

        visual_positions: Optional[torch.LongTensor] = None
        n_visual = 0
        do_prune = False
        schedule: Optional[List[int]] = None

        if is_prefill:
            self.prefill_done = True
            pushed = self.visual_token_positions if self._visual_positions_fresh else None
            self._visual_positions_fresh = False
            if pushed is None:
                logger.warning_once(
                    "[DUET] no visual token positions pushed for this sample; prefill runs UNPRUNED "
                    "(expected only for text-only prompts)."
                )
            else:
                pushed = pushed.reshape(-1).to(device=hidden_states.device, dtype=torch.long)
                if pushed.numel() and int(pushed.max()) >= seq_length:
                    raise RuntimeError(
                        f"[DUET] stale visual_token_positions: max {int(pushed.max())} >= seq_len {seq_length}."
                    )
                visual_positions = pushed
                n_visual = int(pushed.numel())

            if n_visual > 0:
                self.original_visual_length = n_visual
                self.current_visual_length = n_visual
                schedule = self._build_schedule(n_visual)
                self._schedule = schedule
                do_prune = (self.target_visual_tokens > 0) and (len(self.layer_list) > 0)
                if self.duet_debug:
                    segs = [self.layer_list[0]] + \
                           [self.layer_list[i] - self.layer_list[i - 1] for i in range(1, len(self.layer_list))] + \
                           [self.num_text_layers - self.layer_list[-1]]
                    avg = sum(segs[s] * schedule[s] for s in range(len(schedule))) / max(1, self.num_text_layers)
                    print("\n" + "=" * 80)
                    print(f"[DUET Qwen3-VL DEBUG] rev={DUET_REV} prefill")
                    print(f"  seq_len={seq_length}  stage-1 visual N={n_visual}  "
                          f"first/last visual idx={int(visual_positions[0])}/{int(visual_positions[-1])}")
                    print(f"  layer_list={self.layer_list}  schedule={schedule}  "
                          f"target_avg T={self.target_visual_tokens}  layer-avg={avg:.2f}")
                    print(f"  deepstack levels={0 if deepstack_visual_embeds is None else len(deepstack_visual_embeds)}")
                    print("=" * 80 + "\n", flush=True)

        visual_token_accumulator = 0.0

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_idx = decoder_layer.self_attn.layer_idx + 1   # 1-based

            if is_prefill and self.current_visual_length is not None:
                visual_token_accumulator += float(self.current_visual_length)

            prune_here = do_prune and (layer_idx in self.layer_list) and (visual_positions is not None) \
                and (int(visual_positions.numel()) > 0)

            importance = None
            if prune_here:
                importance = self._text_to_visual_probe(
                    hidden_states, decoder_layer, position_embeddings, visual_positions
                )

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            # DeepStack injection after decoder layers 0,1,2 (0-indexed). DUET prunes at much later
            # layers, so counts always match here, but keep the guarded call identical to FastV.
            _ds = layer_idx - 1
            if (
                deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and 0 <= _ds < len(deepstack_visual_embeds)
            ):
                if deepstack_visual_embeds[_ds].shape[0] == int(visual_pos_masks.sum()):
                    hidden_states = self.language_model._deepstack_process(
                        hidden_states, visual_pos_masks, deepstack_visual_embeds[_ds]
                    )

            if prune_here:
                stage = self.layer_list.index(layer_idx)
                retained_num = int(schedule[stage + 1])
                (
                    hidden_states, visual_positions, position_ids, text_position_ids,
                    cache_position, position_embeddings, causal_mask, visual_pos_masks,
                    deepstack_visual_embeds,
                ) = self._prune_visual(
                    hidden_states, importance, visual_positions, retained_num,
                    position_ids, text_position_ids, cache_position,
                    visual_pos_masks, deepstack_visual_embeds,
                )
                self.visual_token_positions = visual_positions
                self.current_visual_length = int(visual_positions.numel())
                if self.duet_debug:
                    print(f"[DUET PRUNE] layer {layer_idx} (stage {stage}): visual -> "
                          f"{self.current_visual_length} (target {retained_num}); "
                          f"seq_len={hidden_states.shape[1]}", flush=True)

        hidden_states = self.norm(hidden_states)

        if is_prefill:
            n_run = self.num_text_layers if self.num_text_layers else 1
            average_tokens = visual_token_accumulator / n_run
            self.visual_token_sum = float(visual_token_accumulator)
            self.visual_token_num = float(average_tokens)
            self.visual_token_len = float(average_tokens)
            if self.duet_debug:
                print(f"[DUET DEBUG] visual token layer-average: {average_tokens:.2f} "
                      f"(N={n_visual}, T={self.target_visual_tokens})", flush=True)

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
