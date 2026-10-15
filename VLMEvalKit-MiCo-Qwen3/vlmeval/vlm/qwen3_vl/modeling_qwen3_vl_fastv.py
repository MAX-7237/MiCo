# coding=utf-8
"""True FastV Stage 2 for Qwen3-VL.

Rebuilt 2026-07-24 (rev=qwen3vl-rebuild-20260724 / P10). The previous TOPS2 port in this file was
a *fake* FastV with three independent structural defects; every one of them is fixed here:

  1. It scored visual tokens by ``hidden_states.norm(dim=-1)`` (an L2-norm heuristic). There was no
     attention probe in the file at all. FastV's definition is "attention received from the last
     query token at layer K", so the old file measured a different quantity entirely.
     -> ``_visual_attention_probe`` below replicates ``Qwen3VLTextAttention.forward`` (UP:459-500):
        input_layernorm -> q_proj/k_proj -> view(B,S,H,D) -> **q_norm/k_norm on the head dim** ->
        transpose -> apply_rotary_pos_emb (interleaved MRoPE already folded into cos/sin at UP:355)
        -> repeat_kv for GQA -> scaled dot product -> full-row softmax.
  2. It pruned *before* running layer K, so only ``K-1 = 1`` layer ever saw the full sequence and
     the probe (had there been one) would have read a post-prune state.
     -> the loop now runs the first K layers at FULL length, probes the INPUT of layer K, runs
        layer K, and prunes AFTERWARDS (the reference order, qwen2_5_vl/..._fastv.py:284-327).
  3. ``R_dict["8b"]`` was a byte-identical copy of the 28-layer Qwen2.5-VL-7B table, and it was
     keyed on the *nominal* target instead of the actual visual token count.
     -> ``R_dict`` is deleted. R is derived per sample from L=len(self.layers)=36 and the real N.

Other rebuild items landed in this file:
  * P2 -- recompute RoPE from the pruned ``position_ids``; never gather ``position_embeddings`` on
    dim 2 (Qwen3-VL cos/sin are ``[B, S, head_dim]``, so dim 2 is head_dim=128, not the sequence).
  * P3 -- the post-prune causal mask is rebuilt with ``attention_mask=None``, ``past_key_values=None``
    and ``position_ids=None`` *together*; plus a scoped hardening in ``_get_causal_mask``.
  * P4/P5 -- accept, inject and re-prune the DeepStack kwargs.
  * P6 -- explicit ``set_visual_token_positions`` API; ``system_prompt_length`` is gone.
  * P8-adjacent -- ``self_attn.num_heads`` does not exist in transformers 5.2 (UP:431-457 defines
    only head_dim / scaling / num_key_value_groups); heads are read from the config.
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

# Self-identifying banner so a log line can be traced back to this patch.
FASTV_REV = "qwen3vl-rebuild-20260724/P10-true-fastv"

_FALSEY = ('', '0', 'false', 'False', 'FALSE', 'no', 'off')


def _env_on(name: str, default: str = '0') -> bool:
    return str(os.environ.get(name, default)).strip() not in _FALSEY


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1):
    """Prefer upstream's own ``apply_rotary_pos_emb`` so the probe cannot drift from the model."""
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


class FastVQwen3VLModel(_BASE_MODEL_CLS):
    """FastV stage-2 for Qwen3-VL: run K layers at full length, probe attention, prune, continue."""

    def __init__(self, config, visual_token_num: int):
        super().__init__(config)
        lm = self.language_model
        self.embed_tokens = lm.embed_tokens
        self.layers = lm.layers
        self.norm = lm.norm
        self.rotary_emb = lm.rotary_emb
        self._lm_update_causal_mask = getattr(lm, '_update_causal_mask', None)

        # P6: NO system_prompt_length. Qwen3-VL has no system prompt; the first image token sits at
        # index 4 under the shipped chat template, not 15 as in Qwen2.5-VL. The old
        # `seq_length - 15 - 10` window was an 11-token shift that both protected 11 real image
        # tokens from pruning and made 11 real TEXT tokens deletable. The window is now pushed in
        # explicitly by the wrapper; see the reference rationale in
        # qwen2_5_vl/modeling_qwen2_5_vl_tops_variant3.py:61-65.
        self.visual_token_positions: Optional[torch.LongTensor] = None
        # Freshness flag, consumed by each prefill. Guards the one hazard the explicit-window design
        # introduces: a text-only sample that follows an image sample would otherwise inherit the
        # previous sample's positions and happily prune TEXT tokens.
        self._visual_positions_fresh = False

        self.visual_token_length = 1296   # nominal full-res token count; logging/back-compat only
        self.visual_token_num = 0.0
        self.visual_token_len = 0.0
        self.fastv_debug = _env_on('ENABLE_DEBUG', '1')

        try:
            visual_token_num = int(visual_token_num)
        except (TypeError, ValueError):
            raise ValueError(f"visual_token_num must be an integer, got {visual_token_num!r}")

        n_layers = getattr(config, 'num_hidden_layers', None) or config.text_config.num_hidden_layers
        if n_layers <= 22:
            self.scale = "4b"
        elif n_layers <= 30:
            self.scale = "7b"   # Qwen2.5-VL-7B (28 layers)
        elif n_layers <= 40:
            self.scale = "8b"   # Qwen3-VL-8B (36 layers)
        else:
            self.scale = "unknown"

        # L in the budget identity. len(self.layers) is authoritative -- config.num_hidden_layers
        # lives on text_config in tf>=4.52 and the `or` fallback above has bitten this file before.
        self.num_text_layers = len(self.layers) if len(self.layers) else int(n_layers)

        self.target_visual_tokens = visual_token_num
        self.K = 2                      # FastV pruning layer, 1-based (prune AFTER this layer runs)
        self.pruning_layer = self.K     # back-compat alias
        self.R: Optional[int] = None    # per-sample, derived in forward(); NOT a table lookup

        print(f"[FastV Init] rev={FASTV_REV} Qwen3-VL scale={self.scale} "
              f"(L={self.num_text_layers} text layers), K={self.K}, target T={self.target_visual_tokens}, "
              f"budget={os.environ.get('FASTV_BUDGET', 'avg')} "
              f"(R is derived per sample as (L*T - K*N)/(L-K); the 28-layer R_dict is gone)")

        self.reset_state()

    def reset_state(self):
        # NOTE (P6b): `visual_token_positions` is deliberately NOT cleared here. reset_state() runs
        # at the top of forward(), i.e. AFTER the wrapper has already pushed this sample's positions.
        # Clearing it would delete the very thing stage 2 needs. Staleness is caught instead by the
        # range check in forward().
        self.prefill_done = False
        self.current_visual_length: Optional[int] = None
        self.original_visual_length: Optional[int] = None
        self.retained_visual_length: Optional[int] = None
        self.retain_token_indices: Optional[torch.LongTensor] = None
        self.visual_token_sum = 0.0
        self.visual_token_num = 0.0
        self.visual_token_len = 0.0
        self.R = None

    def set_visual_token_positions(self, positions: Optional[torch.LongTensor]) -> None:
        """Absolute positions (in the post-stage-1 sequence) of the surviving image tokens.

        One window API shared by fastv / tops_variant3 / tops. An explicit index vector -- not a
        `(start, length)` offset -- because Qwen prompts do not guarantee that visual tokens begin
        at a fixed offset, and a multi-image prompt (model.py allows up to 24 images) is not a
        contiguous run at all.
        """
        if positions is None:
            self.visual_token_positions = None
            self._visual_positions_fresh = False
            return
        self.visual_token_positions = positions.detach().reshape(-1).to(torch.long)
        self._visual_positions_fresh = True

    def _get_causal_mask(self, input_tensor, attention_mask, cache_position, past_key_values, position_ids=None):
        # P3b (scoped): masking_utils' packed-sequence branch (MU:823) fires on exactly
        #     position_ids is not None and attention_mask is None and past_key_values is None
        # Post-prune position_ids are GAPPED, so that branch would read every prune gap as a
        # document boundary, build a block-diagonal mask (MU:732 + MU:928-930) and blind the answer
        # tokens to every retained visual token (RCA: POPE-120 acc 37.5 vs 86.7 fixed).
        # The guard is written as "drop position_ids on exactly that combination" rather than the
        # plan's unconditional `position_ids = None`, so the INITIAL mask site -- which always
        # passes a non-None past_key_values -- stays bit-identical to today's verified-correct
        # behaviour, while any future caller that reproduces the dangerous triple is corrected and
        # told about it.
        if position_ids is not None and attention_mask is None and past_key_values is None:
            logger.warning_once(
                "[FastV] dropping position_ids for causal-mask construction: "
                "(position_ids != None, attention_mask == None, past_key_values == None) is the "
                "packed-sequence trigger in masking_utils and gapped post-prune positions would be "
                "read as document boundaries."
            )
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

    def _compute_R(self, n_visual: int) -> int:
        """FastV average budget: K layers hold N, (L-K) hold R, and the layer average must equal T:

              (K*N + (L-K)*R) / L == T   =>   R = (L*T - K*N) // (L - K)

        L = 36 for Qwen3-VL-8B, K = 2. Example: T=128, N=1320 -> R=57, layer-avg 127.17.
        The deleted 28-layer table said 38, which would have spent a layer-average of ~109.
        N is per sample (it depends on grid_thw), which is exactly why a static table cannot work.

        'avg' vs 'flat' is a genuine human decision (spec section 4.1): the encoder-side baselines
        (CDPruner / DivPrune) keep a flat T in *every* layer, so at the same nominal T=128 the two
        families differ 57 vs 128 tokens per layer. The convention is switchable via FASTV_BUDGET
        and is printed on every prefill so no run's convention is ever ambiguous.
        """
        T, L, K = self.target_visual_tokens, self.num_text_layers, self.K
        if T is None or T <= 0 or L <= K or n_visual <= 0:
            return int(n_visual)
        if str(os.environ.get('FASTV_BUDGET', 'avg')).strip().lower() == 'flat':
            return int(min(T, n_visual))
        return int(max(0, min((L * T - K * n_visual) // (L - K), n_visual)))

    def _visual_attention_probe(
        self,
        hidden_states: torch.Tensor,
        decoder_layer: nn.Module,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        visual_positions: torch.LongTensor,
    ) -> torch.Tensor:
        """FastV importance = attention paid by the LAST query token to each visual token, measured
        with layer K's own projections.

        Replicates UP:459-500 exactly. Two Qwen3-specific details the Qwen2.5-VL reference does not
        have and that silently change the measured function if omitted:
          * q_norm / k_norm -- per-head RMSNorm applied BEFORE RoPE and before the transpose
            (UP:471-472).
          * `self_attn.num_heads` / `num_key_value_heads` do NOT exist on Qwen3VLTextAttention
            (UP:431-457 defines head_dim, scaling and num_key_value_groups only). Copying the
            reference's lines 119-120 raises AttributeError here, so the head counts come from the
            config, with the projection shapes as a cross-check.
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
                "Qwen3VLTextAttention is expected to expose q_norm/k_norm (UP:471-472); without them "
                "the FastV probe measures a different function than the model's own attention."
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
        # Last query row only: [B, H, 1, S]. No attention mask is applied -- this decoder is
        # batch-1 and the last row of a causal prefill mask is all-visible by construction.
        # FASTV_PROBE_NORM selects the softmax normalisation scope:
        #   full   -> softmax over ALL keys, then slice visual columns (FastV paper form)
        #   visual -> softmax over the visual keys ONLY (qwen2_5_vl reference form)
        # These pick DIFFERENT tokens: full-row normalisation down-weights heads that attend
        # mostly to text, changing the head-average and hence the top-k.
        _probe_norm = str(os.environ.get('FASTV_PROBE_NORM', 'visual')).strip().lower()
        if _probe_norm == 'visual':
            key_visual = key_states[..., visual_positions, :]
            attn_scores = torch.matmul(query_states[:, :, -1:, :], key_visual.transpose(-2, -1)) * scaling
            attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
            visual_importance = attn_weights.mean(dim=1).squeeze(1).mean(dim=0)
        else:
            attn_scores = torch.matmul(query_states[:, :, -1:, :], key_states.transpose(-2, -1)) * scaling
            attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
            # [B, H, 1, V] -> mean over heads -> [B, 1, V] -> [B, V] -> mean over batch -> [V]
            visual_importance = attn_weights[..., visual_positions].mean(dim=1).squeeze(1).mean(dim=0)
        return visual_importance.float()

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
        # P4: `visual_pos_masks` / `deepstack_visual_embeds` default to None so every pre-existing
        # call site keeps working, and `**kwargs` absorbs anything else the wrapper or generate()
        # hands down. Extra kwargs are deliberately NOT forwarded to the decoder layers: attn
        # implementation is pinned to sdpa on this path, so there are no FlashAttentionKwargs to
        # relay, and silently forwarding unknown keys is how the old port grew its bugs.

        # Read the cache length BEFORE a DynamicCache is created, so a chunked-prefill second chunk
        # (q_len > 1 against a non-empty cache) is not mistaken for a fresh prefill the way the old
        # `seq_length > 1` test was.
        past_seen_tokens = _past_length(past_key_values)

        if past_seen_tokens == 0:
            self.reset_state()

        output_attentions = (
            output_attentions if output_attentions is not None
            else getattr(self.config, 'output_attentions', False)
        )
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
            text_position_ids = position_ids[0]  # [B, S]
            position_ids = position_ids[1:]      # [3, B, S]
        # else: already [3, B, S]

        # INITIAL causal mask -- correct as-is, do not touch. The cache is empty at prefill so
        # masking_utils sizes it from inputs_embeds, and past_key_values is non-None so the
        # packed-sequence branch (MU:823) is short-circuited regardless of text_position_ids.
        causal_mask = self._get_causal_mask(
            inputs_embeds, attention_mask, cache_position, past_key_values, position_ids=text_position_ids
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None

        # ── Resolve this sample's visual window and its per-sample R ──────────────────────────
        visual_positions: Optional[torch.LongTensor] = None
        n_visual = 0
        do_prune = False

        if is_prefill:
            self.prefill_done = True
            # Consume the freshness flag: positions pushed for a PREVIOUS sample must never prune
            # this one. The vector itself is left in place for external inspection.
            pushed = self.visual_token_positions if self._visual_positions_fresh else None
            self._visual_positions_fresh = False
            if pushed is None:
                logger.warning_once(
                    "[FastV] no visual token positions were pushed for this sample; this prefill runs "
                    "UNPRUNED. Expected only for text-only prompts. The old "
                    "`seq_length - system_prompt_length - 10` heuristic is off by 11 tokens on "
                    "Qwen3-VL and is deliberately not used as a fallback."
                )
            else:
                pushed = pushed.reshape(-1).to(device=hidden_states.device, dtype=torch.long)
                if pushed.numel() and int(pushed.max()) >= seq_length:
                    raise RuntimeError(
                        f"[FastV] stale visual_token_positions: max index {int(pushed.max())} >= "
                        f"seq_len {seq_length}. set_visual_token_positions() must be called with the "
                        f"positions in the POST-stage-1 sequence of THIS sample."
                    )
                visual_positions = pushed
                n_visual = int(pushed.numel())

            if visual_positions is not None and visual_pos_masks is not None and self.fastv_debug:
                _pushed_mask_pos = visual_pos_masks[0].nonzero(as_tuple=False).squeeze(-1)
                assert _pushed_mask_pos.numel() == n_visual and bool(
                    torch.equal(_pushed_mask_pos.to(visual_positions.device), visual_positions)
                ), "visual_pos_masks and visual_token_positions disagree (DeepStack would be misaligned)"

            if n_visual > 0:
                self.original_visual_length = n_visual
                self.current_visual_length = n_visual
                self.R = self._compute_R(n_visual)
                do_prune = (self.target_visual_tokens > 0) and (self.R < n_visual)
                if self.R == 0 and do_prune:
                    logger.warning_once(
                        "[FastV] budget produced R=0: every visual token will be dropped at layer K. "
                        "Check target_visual_tokens against the real visual token count."
                    )
                if self.fastv_debug:
                    _conv = str(os.environ.get('FASTV_BUDGET', 'avg')).strip().lower()
                    _r_eff = self.R if do_prune else n_visual
                    _avg = (self.K * n_visual + (self.num_text_layers - self.K) * _r_eff) / self.num_text_layers
                    print("\n" + "=" * 80)
                    print(f"[FastV Qwen3-VL DEBUG] rev={FASTV_REV} prefill")
                    print(f"  seq_len={seq_length}  visual tokens N={n_visual}  "
                          f"first/last visual idx={int(visual_positions[0])}/{int(visual_positions[-1])}")
                    print(f"  K={self.K} (layers 1..{self.K} run at FULL length, prune AFTER layer {self.K})")
                    print(f"  target T={self.target_visual_tokens}  budget={_conv}  "
                          f"L={self.num_text_layers}  ->  R={self.R}  layer-avg={_avg:.2f}  "
                          f"prune={'yes' if do_prune else 'no (R >= N or T <= 0)'}")
                    print(f"  deepstack levels={0 if deepstack_visual_embeds is None else len(deepstack_visual_embeds)}")
                    print("=" * 80 + "\n", flush=True)

        visual_token_accumulator = 0.0

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_idx = decoder_layer.self_attn.layer_idx + 1   # 1-BASED (upstream's is 0-based)

            # A layer executes at its PRE-prune count, so charge it before the body runs.
            if is_prefill and self.current_visual_length is not None:
                visual_token_accumulator += float(self.current_visual_length)

            probe_here = do_prune and (layer_idx == self.K)

            importance = None
            if probe_here:
                # Probe the INPUT of layer K (reference qwen2_5_vl/..._fastv.py:284-286).
                importance = self._visual_attention_probe(
                    hidden_states, decoder_layer, position_embeddings, visual_positions
                )
                if self.fastv_debug:
                    print(f"[FastV PROBE] layer {layer_idx}: importance over {importance.numel()} visual tokens, "
                          f"min={float(importance.min()):.3e} max={float(importance.max()):.3e} "
                          f"mean={float(importance.mean()):.3e}", flush=True)

            # Layer K RUNS AT FULL LENGTH (reference :300-310) -- the prune happens below.
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            # ── P5a: DeepStack injection ─────────────────────────────────────────────────────
            # UP:929 injects AFTER decoder layers 0,1,2 (0-indexed) via
            # `layer_idx in range(len(deepstack_visual_embeds))`. The [8,16,24] in the config are
            # the *ViT* blocks the features are extracted from (UP:807), NOT text layers.
            # `layer_idx` here is 1-BASED, upstream's is 0-based, hence the -1.
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

            # ── Prune AFTER layer K (reference :312-327) ─────────────────────────────────────
            if probe_here:
                device = hidden_states.device
                orig_seq_len = hidden_states.shape[1]
                retained_num = int(max(0, min(self.R, n_visual)))

                if retained_num > 0:
                    selected_t = torch.sort(importance.topk(retained_num).indices).values
                else:
                    selected_t = torch.empty(0, dtype=torch.long, device=importance.device)
                selected_t = selected_t.to(device=device, dtype=torch.long)

                # Build the retain set from the explicit index vector: drop every visual position,
                # then put the selected ones back. Text tokens are never touched, and this is
                # correct for multi-image prompts where the visual tokens are not one contiguous run.
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
                # Contiguous truncation, NOT a gather: post-prune the DynamicCache holds exactly
                # new_len entries at slots 0..new_len-1.
                cache_position = cache_position[:new_len]

                # ── P2: recompute RoPE, never gather position_embeddings on dim 2 ───────────
                # Qwen3-VL's rotary_emb returns cos/sin of shape [B, S, head_dim] --
                # apply_interleaved_mrope (UP:355) collapses the leading 3-axis, unlike
                # Qwen2.5-VL's [3, B, S, head_dim]. Gathering dim 2 therefore indexes head_dim
                # (=128), not the sequence: IndexError on any real prompt. Recompute from the
                # already-pruned position_ids instead. Do NOT renumber the retained positions
                # contiguously -- that would corrupt MRoPE.
                position_embeddings = self.rotary_emb(hidden_states, position_ids)

                # ── P3: rebuild the causal mask with all three arguments dropped together ────
                #  * attention_mask=None   -- the prompt mask no longer matches the pruned length.
                #  * past_key_values=None  -- layers 0..K-1 hold an S-long cache, so masking_utils
                #    would set kv_length = new_len + S and build a [1,1,new_len,new_len+S] mask;
                #    sdpa_attention_forward does NOT slice it, so SDPA raises a shape error.
                #  * position_ids=None     -- pruning leaves GAPPED absolute positions; MU:823 would
                #    read them as packed-sequence boundaries, turn every prune gap into a
                #    pseudo-document and AND a block-diagonal mask onto the causal one, blinding the
                #    answer tokens to every retained visual token.
                # With all three None, MU:810 sets kv_length = new_len and _ignore_causal_mask_sdpa
                # returns None => plain is_causal. Landing only some of them is worse than none.
                causal_mask = self._get_causal_mask(
                    hidden_states, None, cache_position, None, position_ids=None
                )

                # P3c invariant. Only meaningful under sdpa, where the mask is torch.bool; under
                # 'eager' it is an additive float and the row sum is not the visible-token count.
                if _env_on('ENABLE_DEBUG') and causal_mask is not None and causal_mask.dtype == torch.bool:
                    assert bool(causal_mask[0, 0, -1].sum() == hidden_states.shape[1]), \
                        "post-prune causal mask is not plain-causal (packed-sequence poisoning)"

                # ── P5b: re-prune the DeepStack side-channel in sync ─────────────────────────
                # Mask-driven, so it is correct regardless of where the visual tokens sit. With
                # K=2 the injections for layers 0 and 1 are already done; injection 2 lands AFTER
                # this prune, so without this block UP:949 raises a shape mismatch.
                if visual_pos_masks is not None:
                    _vis_pos = visual_pos_masks[0].nonzero(as_tuple=True)[0]
                    _surv = torch.isin(_vis_pos, retain_token_indices)
                    if deepstack_visual_embeds is not None:
                        deepstack_visual_embeds = [d[_surv] for d in deepstack_visual_embeds]
                    visual_pos_masks = visual_pos_masks[:, retain_token_indices]

                # Re-map the window into the compacted sequence (reference qwen2_5_vl/..._tops_variant3.py:510-512).
                visual_positions = torch.searchsorted(retain_token_indices, kept_visual_abs)
                self.visual_token_positions = visual_positions
                self.retain_token_indices = retain_token_indices
                self.current_visual_length = int(kept_visual_abs.numel())
                self.retained_visual_length = self.current_visual_length

                if self.fastv_debug:
                    print(f"[FastV PRUNE] layer {layer_idx}: visual {n_visual} -> "
                          f"{self.current_visual_length} (R={self.R}); "
                          f"seq {orig_seq_len} -> {new_len}; mask="
                          f"{'None (is_causal)' if causal_mask is None else tuple(causal_mask.shape)}",
                          flush=True)

        hidden_states = self.norm(hidden_states)

        if is_prefill:
            n_run = self.num_text_layers if self.num_text_layers else 1
            average_tokens = visual_token_accumulator / n_run
            self.visual_token_sum = float(visual_token_accumulator)
            self.visual_token_num = float(average_tokens)
            self.visual_token_len = float(average_tokens)
            if self.fastv_debug:
                print(f"[FastV DEBUG] visual token layer-average: {average_tokens:.2f} "
                      f"(N={n_visual}, R={self.R}, T={self.target_visual_tokens}, "
                      f"budget={os.environ.get('FASTV_BUDGET', 'avg')})", flush=True)

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
