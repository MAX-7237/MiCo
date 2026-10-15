# coding=utf-8
"""
TOPS Stage 2: Progressive visual token pruning with RDC (Relevance-Diversity-Coverage)
greedy selection for Qwen3-VL, ported from modeling_qwen2_5_vl_tops.py.

Key design:
  - Relevance (R): attention importance computed via QKV probe on PRE-layer hidden_states
  - Diversity (D): cosine dissimilarity computed on POST-layer hidden_states
  - Coverage (C): representativeness computed on POST-layer hidden_states
  - Greedy selection jointly optimizing R + alpha*D + lambda*C

Rebuild 2026-07-24 (branch qwen3vl-rebuild-20260724), REBUILD_SPEC patches applied to this file:
  P2  RoPE recompute after prune (the dim-2 gather indexed head_dim, not the sequence)
  P3  post-prune causal mask: attention_mask/past_key_values/position_ids all None
  P4  DeepStack kwargs on forward (import-safe no-op when None)
  P5  DeepStack injection after text layers 0/1/2 + in-sync re-prune
  P6  explicit visual-position API (replaces the system_prompt_length=15 offset window)
  P8  probe reads num_heads/num_kv_heads from the projections (tf 5.2 removed the attrs)
  P9  36-layer TOPS_SCHEDULE["8b"] average-budget rows
Deliberately NOT applied here: P8's q_norm/k_norm sub-patch (this file already applies them,
see `_compute_visual_token_importance`) and P9's `_compute_schedule_average` reporting fix
(this file already charges the pre-prune count; only the star_v3 twin was wrong).
"""

import os
import math
from typing import Dict, List, Optional, Tuple, Union, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

# Try Qwen3-VL imports (transformers 5.7+), fall back to Qwen2.5-VL as structural reference
try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    from transformers.models.qwen3_vl.modeling_qwen3_vl import create_causal_mask as _create_causal_mask_impl
    _BASE_MODEL_CLS = Qwen3VLModel
    _USING_QWEN3 = True
except ImportError:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb as apply_rotary_pos_emb
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel as _BASE_MODEL_CLS
    _USING_QWEN3 = False
    _create_causal_mask_impl = None

logger = logging.get_logger(__name__)

# Self-identifying banner so a log tells you which build produced it (starpro-experiments-always-debug).
_REBUILD_REV = "qwen3vl-coverprune2-20260827 (S1 inside-weighted cover x2T; S2 all-text-attn + inside-weighted coverage greedy; sched L14/L24)"

# TOPS pruning schedule for Qwen3-VL
# Format: {scale: {target_tokens: [(layer_idx_1indexed, keep_tokens), ...]}}
# "8b": 36 layers (Qwen3-VL-8B). The comment that used to sit here claimed 28 -- copied from the
#       Qwen2.5-VL-7B depth -- and the "8b" rows below were derived from that wrong layer count.
# "4b": ~22 layers (estimate, update once 4B config is known)
# "2b": ~28 layers (estimate)
TOPS_SCHEDULE: Dict[str, Dict[int, List[Tuple[int, int]]]] = {
    "7b": {
        # Qwen2.5-VL-7B (28 layers): prune at layers 12 and 16
        512: [(12, 281), (16, 77)],
        256: [(12, 139), (16, 39)],
        128: [(12, 71),  (16, 19)],
    },
    "8b": {
        # Qwen3-VL-8B, 36 layers. Stage-1 keeps 2T; prune at 1-indexed layers 14 and 24.
        # A layer executes at its PRE-prune count, so the stages are 14 | 10 | 12 layers:
        #   T=256: 14*512 +10*128 +12*64 = 7168+1280+768 = 9216 = 36*256  avg 256.0
        #   T=128: 14*256 +10*64  +12*32 = 3584+ 640+384 = 4608 = 36*128  avg 128.0
        #   T=512: 14*1024+10*256 +12*128=14336+2560+1536=18432 = 36*512  avg 512.0
        # (user-specified schedule; the identity assumes N >= 2T, else it spends strictly less.)
        512: [(14, 256), (24, 128)],
        256: [(14, 128), (24, 64)],
        128: [(14, 64),  (24, 32)],
    },
    "4b": {
        512: [(8, 281), (12, 77)],
        256: [(8, 139), (12, 39)],
        128: [(8, 71),  (12, 19)],
    },
    "2b": {
        512: [(12, 281), (16, 77)],
        256: [(12, 139), (16, 39)],
        128: [(12, 71),  (16, 19)],
    },
}


class CoverPrune2Qwen3VLModel(_BASE_MODEL_CLS):
    """
    TOPS Stage-2 for Qwen3-VL: progressive RDC greedy pruning inside LLM layers.
    Ported from modeling_qwen2_5_vl_tops.py.

    Assumes Stage-1 has already reduced visual tokens before the LLM.
    """

    def __init__(self, config, visual_token_num: int):
        super().__init__(config)
        # transformers 4.52+: transformer layers live inside self.language_model
        lm = self.language_model
        self.embed_tokens = lm.embed_tokens
        self.layers = lm.layers
        self.norm = lm.norm
        self.rotary_emb = lm.rotary_emb
        self._lm_update_causal_mask = getattr(lm, '_update_causal_mask', None)

        # NO system_prompt_length. Qwen3-VL has no system prompt; the first image token is at index 4
        # (measured against the shipped chat template), not 15 as in Qwen2.5-VL -- an 11-token window
        # shift that both protects 11 real image tokens from pruning and makes 11 real TEXT tokens
        # deletable. The window is pushed in explicitly by the wrapper; see the reference rationale in
        # qwen2_5_vl/modeling_qwen2_5_vl_starv3.py:61-65.
        self.visual_token_positions: Optional[torch.LongTensor] = None
        # Retained: `visual_token_length` is NOT part of the removed offset heuristic. It is the
        # Stage-0 token count used by the three budget lines below and by the stage summary.
        self.visual_token_length = 1296  # default Stage-0 tokens
        self.visual_token_num = 0.0
        self.visual_token_len = 0.0

        try:
            visual_token_num = int(visual_token_num)
        except (TypeError, ValueError):
            raise ValueError(f"visual_token_num must be an integer, got {visual_token_num!r}")

        n_layers = getattr(config, 'num_hidden_layers', None) or config.text_config.num_hidden_layers
        if n_layers <= 16:
            self.scale = "2b"
        elif n_layers <= 22:
            self.scale = "4b"
        elif n_layers <= 30:
            self.scale = "7b"   # Qwen2.5-VL-7B (28 layers)
        elif n_layers <= 40:
            self.scale = "8b"   # Qwen3-VL-8B (36 layers)
        else:
            self.scale = "unknown"

        self.target_visual_tokens = visual_token_num
        self.stage1_keep_tokens = max(1, visual_token_num) * 2 if visual_token_num > 0 else self.visual_token_length
        self.stage0_visual_tokens = self.visual_token_length
        self.stage1_expected_tokens = min(self.visual_token_length, self.stage1_keep_tokens)

        schedule = TOPS_SCHEDULE.get(self.scale, {}).get(self.target_visual_tokens)
        if schedule is None:
            self.pruning_layers: Dict[int, int] = {}
            if visual_token_num > 0:
                logger.warning_once(
                    "TOPS schedule missing for target %s on scale %s. Progressive pruning disabled.",
                    visual_token_num, self.scale,
                )
        else:
            self.pruning_layers = {layer_idx: target for layer_idx, target in schedule}

        print(f"[TOPS Init] Qwen3-VL  rev={_REBUILD_REV}")
        print(f"  Scale: {self.scale} ({n_layers} layers)")
        print(f"  Target visual tokens: {self.target_visual_tokens}")
        print(f"  Pruning schedule (1-indexed): {self.pruning_layers}")

        self.reset_state()

    def _get_causal_mask(self, input_tensor, attention_mask, cache_position, past_key_values, position_ids=None):
        # This decoder is always batch-1, single-document: packed-sequence semantics are never wanted.
        # Hard-drop position_ids so no call site can re-trigger masking_utils.py:823
        # (`position_ids is not None and attention_mask is None and past_key_values is None`).
        # NOTE: this helper serves BOTH mask sites. At the INITIAL site in `forward`
        # (inputs_embeds, attention_mask, cache_position, past_key_values, position_ids=text_position_ids)
        # this drop is a no-op today, because MU:823 also requires `past_key_values is None` and that
        # site always passes a live cache. It is kept as defence in depth: should `attention_mask`
        # ever be None there while a post-stage-1 *gapped* text_position_ids is in play, the packed
        # branch would otherwise fire and blind the text tokens to the visual tokens.
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

    def set_visual_token_positions(self, positions: torch.LongTensor) -> None:
        """Absolute positions (in the post-stage-1 sequence) of the surviving image tokens."""
        self.visual_token_positions = positions.detach()

    def reset_state(self):
        # NOTE: `visual_token_positions` is deliberately NOT cleared here. `reset_state` runs at the
        # top of `forward`, i.e. AFTER the wrapper has already pushed the positions for this prefill.
        self.prefill_done = False
        self.current_visual_length: Optional[int] = None
        self.original_visual_length: Optional[int] = None
        self.visual_token_sum = 0.0
        self.stage1_actual_tokens: Optional[int] = None
        self.stage2_events: List[Tuple[int, int, int]] = []
        self.stage2_skip_events: List[Tuple[int, int, int]] = []
        self.stage2_reported_layers: Set[int] = set()
        self._stage_summary_emitted = False
        self.visual_token_num = 0.0
        self.visual_token_len = 0.0

    # ─────────── helpers ───────────

    def _select_text_raters(
        self,
        hidden_states: torch.Tensor,
        visual_positions: torch.LongTensor,
    ) -> torch.Tensor:
        """Select informative text tokens as raters via normalized text-visual similarity.

        Returns ABSOLUTE positions (not offsets), per qwen2_5_vl/modeling_qwen2_5_vl_starv3.py:191-221.
        """
        seq_len = hidden_states.shape[1]
        non_visual = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)
        non_visual[visual_positions] = False
        text_positions = non_visual.nonzero(as_tuple=False).squeeze(-1)
        text_hidden = hidden_states.index_select(1, text_positions)
        if text_hidden.size(1) == 0:
            return hidden_states.new_empty((0,), dtype=torch.long)

        visual_hidden = hidden_states.index_select(1, visual_positions)
        text_norm = F.normalize(text_hidden, dim=-1)
        visual_norm = F.normalize(visual_hidden, dim=-1)

        text_visual_sim = torch.matmul(text_norm, visual_norm.transpose(1, 2))
        text_importance = text_visual_sim.mean(dim=-1)
        if text_importance.dim() == 2:
            text_importance = text_importance.mean(dim=0)

        mean_importance = text_importance.mean()
        selected = torch.nonzero(text_importance > mean_importance, as_tuple=False).squeeze(-1)
        if selected.numel() == 0:
            keep = max(1, text_importance.numel() // 2)
            selected = text_importance.topk(keep).indices
        return text_positions.index_select(0, selected)

    def _compute_visual_token_importance(
        self,
        hidden_states: torch.Tensor,
        decoder_layer: nn.Module,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        visual_positions: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Compute attention-based relevance via QKV probe on PRE-layer hidden_states.
        Uses dynamic text rater tokens as queries, visual tokens as keys.
        Returns: (V,) importance scores for each visual token.
        """
        self_attn = decoder_layer.self_attn

        attn_input = hidden_states
        input_layernorm = getattr(decoder_layer, "input_layernorm", None)
        if input_layernorm is not None:
            attn_input = input_layernorm(attn_input)

        bsz, seq_len, _ = attn_input.shape
        # Qwen3VLTextAttention (transformers 5.2) defines head_dim, scaling and num_key_value_groups
        # but NEITHER num_heads NOR num_key_value_heads -- reading them is an AttributeError.
        # `head_dim` must be bound BEFORE the two lines below, which divide by it.
        head_dim = self_attn.head_dim
        num_heads = self_attn.q_proj.out_features // head_dim
        num_kv_heads = self_attn.k_proj.out_features // head_dim

        query_states = self_attn.q_proj(attn_input)
        key_states = self_attn.k_proj(attn_input)

        query_states = query_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        # Apply q_norm/k_norm if present (Qwen3-VL LLM uses RMSNorm on q and k).
        # Upstream normalizes on [B, S, H, D] before the transpose; RMSNorm reduces over the last
        # dim (= head_dim) either way, so applying it here on [B, H, S, D] is equivalent.
        # DO NOT add a second application -- it is already here.
        if hasattr(self_attn, 'q_norm'):
            query_states = self_attn.q_norm(query_states)
        if hasattr(self_attn, 'k_norm'):
            key_states = self_attn.k_norm(key_states)

        if num_kv_heads != num_heads:
            repeat_factor = num_heads // num_kv_heads
            key_states = key_states.repeat_interleave(repeat_factor, dim=1)

        # Apply RoPE (Qwen3-VL uses apply_rotary_pos_emb, not the multimodal variant; the 3-axis is
        # already folded into cos/sin by apply_interleaved_mrope).
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # COVER_Prune_2: use ALL text tokens as queries (not a dynamic rater subset) --
        # matches qwen2_5_vl/modeling_qwen2_5_vl_coverprune2.py:196-201 (all_text_attention).
        non_visual = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)
        non_visual[visual_positions] = False
        text_positions = non_visual.nonzero(as_tuple=False).squeeze(-1)
        if text_positions.numel() == 0:
            text_positions = torch.tensor([seq_len - 1], device=hidden_states.device, dtype=torch.long)

        query_raters = torch.index_select(query_states, 2, text_positions)
        key_visual = torch.index_select(key_states, 2, visual_positions)

        attn_scores = torch.matmul(query_raters, key_visual.transpose(-2, -1))
        attn_scores = attn_scores * self_attn.scaling  # == / sqrt(head_dim); tracks upstream
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Average over raters, heads, batch → (V,)
        visual_importance = attn_weights.mean(dim=2).mean(dim=1).mean(dim=0)

        print(f"[TOPS S2] text_rater positions numel: {text_positions.numel()}, "
              f"importance: min={visual_importance.min():.4f}, max={visual_importance.max():.4f}")
        return visual_importance

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _rdc_greedy_select(
        self,
        attn: torch.Tensor,
        post_hidden: torch.Tensor,
        retained_num: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        RDC greedy selection on POST-layer hidden states.
        attn: (V,) normalized relevance scores
        post_hidden: (B, V, D) POST-layer visual token features
        Returns: (retained_num,) sorted selected indices (batch=1 path)
        """
        a2 = float(os.environ.get("A2", "1.0"))
        alpha_2 = float(os.environ.get("ALPHA_2", "1.0"))
        lambda_2 = float(os.environ.get("LAMBDA_2", "1.0"))

        B, V, D = post_hidden.shape

        attn = attn.detach()
        if attn.dim() == 1:
            attn = attn.unsqueeze(0)  # (1, V)
        attn = (attn - attn.min(dim=1, keepdim=True).values) / (
            attn.max(dim=1, keepdim=True).values -
            attn.min(dim=1, keepdim=True).values + 1e-8
        )

        vis_norm = F.normalize(post_hidden, p=2, dim=-1)  # (B, V, D)

        all_selected = []
        for b in range(B):
            a = attn[b]        # (V,)
            feat = vis_norm[b]  # (V, D)

            available = torch.ones(V, dtype=torch.bool, device=device)
            selected = []

            full_sim = feat @ feat.t()  # (V, V)
            # COVER_Prune_2 seed: max weighted coverage  Σ_v a_v·max(0, sim(v,i))  (g=0)
            cov0 = (a.unsqueeze(0) * torch.clamp(full_sim, min=0.0)).sum(dim=1)
            i0 = torch.argmax(cov0).item()
            selected.append(i0)
            available[i0] = False

            max_sim_to_S = full_sim[i0].clone()  # (V,)
            max_sim_to_S[i0] = 1.0

            for _ in range(1, retained_num):
                if not available.any():
                    break

                # COVER_Prune_2: attn weighted INTO each coverage term (w_v = covered token v's attn),
                # no outer multiply by the candidate's own attn:
                #   gain[i] = Σ_v a_v · max(0, sim(v,i) - g_v)
                gain = (a.unsqueeze(0) * torch.clamp(full_sim - max_sim_to_S.unsqueeze(0), min=0.0)).sum(dim=1)
                scores = gain
                scores[~available] = -float("inf")
                idx = torch.argmax(scores).item()

                selected.append(idx)
                available[idx] = False

                sim_vec = full_sim[idx]
                max_sim_to_S = torch.maximum(max_sim_to_S, sim_vec)
                max_sim_to_S[idx] = 1.0

            selected_t = torch.tensor(selected, device=device, dtype=torch.long)
            selected_t = torch.sort(selected_t).values
            all_selected.append(selected_t)

        a_debug = attn[0]
        feat_debug = vis_norm[0]
        sel = all_selected[0]
        avg_attn = a_debug[sel].mean().item()
        if sel.numel() > 1:
            sel_feat = feat_debug[sel]
            sel_sim = sel_feat @ sel_feat.t()
            diag_mask = ~torch.eye(sel_sim.size(0), dtype=torch.bool, device=sel_sim.device)
            avg_sim = sel_sim[diag_mask].mean().item()
        else:
            avg_sim = 0.0
        print(f"[TOPS S2] RDC: retained={sel.numel()}, a2={a2}, alpha_2={alpha_2}, lambda_2={lambda_2}, "
              f"avg_attn={avg_attn:.4f}, avg_sim={avg_sim:.4f}")

        return all_selected[0]

    def _compute_schedule_average(self, stage1_tokens, total_layers):
        if stage1_tokens is None or total_layers <= 0:
            return None
        sorted_layers = sorted(self.pruning_layers.items())
        total_tokens = 0.0
        current_tokens = float(stage1_tokens)
        current_layer = 1
        for layer_idx, target in sorted_layers:
            if layer_idx < current_layer or layer_idx > total_layers:
                continue
            if layer_idx > current_layer:
                total_tokens += (layer_idx - current_layer) * current_tokens
            # The prune layer RUNS BEFORE its prune, so it is charged its PRE-prune count.
            # (Already correct here; it is the star_v3 twin that charged `target`.)
            total_tokens += current_tokens
            current_tokens = float(min(current_tokens, target))
            current_layer = layer_idx + 1
        if current_layer <= total_layers:
            total_tokens += (total_layers - current_layer + 1) * current_tokens
        return total_tokens / total_layers

    def _compute_observed_average(self, total_layers):
        if not self.prefill_done or self.stage1_actual_tokens is None or total_layers <= 0:
            return None
        total_tokens = 0.0
        current_tokens = float(self.stage1_actual_tokens)
        current_layer = 1
        for layer_idx, before_tokens, after_tokens in sorted(self.stage2_events):
            if layer_idx < current_layer or layer_idx > total_layers:
                continue
            if layer_idx > current_layer:
                total_tokens += (layer_idx - current_layer) * current_tokens
            total_tokens += float(before_tokens) if before_tokens is not None else float(after_tokens)
            current_tokens = float(after_tokens)
            current_layer = layer_idx + 1
        if current_layer <= total_layers:
            total_tokens += (total_layers - current_layer + 1) * current_tokens
        return total_tokens / total_layers

    # ─────────── forward ───────────

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
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

        def _is_empty_cache(cache):
            if cache is None:
                return True
            if isinstance(cache, DynamicCache):
                return cache.get_seq_length() == 0
            return False

        if _is_empty_cache(past_key_values):
            self.reset_state()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.is_gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # Qwen3-VL 4D position_ids: [4, B, S] where [0]=text, [1:]=mrope T/H/W
        # rotary_emb expects [3, B, S]; decoder layers expect [B, S] text positions
        text_position_ids = None
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        elif position_ids.dim() == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]  # [B, S]
            position_ids = position_ids[1:]      # [3, B, S]
        # else: already [3, B, S]

        # INITIAL mask site -- CORRECT AS-IS, do not change. `past_key_values` is a live cache here
        # (empty at prefill), which sizes kv_length correctly and short-circuits the packed branch.
        causal_mask = self._get_causal_mask(
            inputs_embeds, attention_mask, cache_position, past_key_values, position_ids=text_position_ids
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None

        seq_length = inputs_embeds.shape[1]

        # ── Prefill phase init ──
        if seq_length > 1 and not self.prefill_done:
            if self.visual_token_positions is None:
                raise RuntimeError(
                    "TOPS stage-2: visual token positions were not pushed by the wrapper. "
                    "The 'seq_length - 15 - 10' heuristic is off by 11 tokens on Qwen3-VL and must "
                    "not be used as a fallback."
                )
            detected_visual_len = int(self.visual_token_positions.numel())

            self.current_visual_length = detected_visual_len
            self.original_visual_length = detected_visual_len
            self.stage1_actual_tokens = detected_visual_len
            self.prefill_done = True
            self.visual_token_sum = 0.0
            print(f"[TOPS] Prefill: seq_length={seq_length}, detected_visual_len={detected_visual_len}, "
                  f"stage1_keep_tokens={self.stage1_keep_tokens}")

        tops_enabled = (
            self.prefill_done
            and self.current_visual_length is not None
            and self.current_visual_length > 0
            and self.visual_token_positions is not None
            and len(self.pruning_layers) > 0
        )
        visual_token_accumulator = 0.0

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_idx = decoder_layer.self_attn.layer_idx + 1

            if seq_length > 1 and self.current_visual_length is not None:
                visual_token_accumulator += float(self.current_visual_length)

            should_prune = (
                tops_enabled
                # Prefill only. `visual_token_positions` addresses the PREFILL sequence; at a decode
                # step hidden_states is 1 token wide and index_select would raise. Unreachable in
                # practice (after prefill current_visual_length <= every remaining target), but the
                # index-vector window turns the old silent corruption into a hard IndexError, so the
                # guard is cheap insurance. FastV carries the same test.
                and seq_length > 1
                and layer_idx in self.pruning_layers
                and self.current_visual_length is not None
                and self.current_visual_length > self.pruning_layers[layer_idx]
            )

            if should_prune:
                target = self.pruning_layers[layer_idx]
                device = hidden_states.device
                visual_positions = self.visual_token_positions.to(device)

                print(f"\n[TOPS S2] Layer {layer_idx}: {self.current_visual_length} → {target} tokens")

                # Step 1: Attention relevance on PRE-layer hidden_states
                attn = self._compute_visual_token_importance(
                    hidden_states, decoder_layer, position_embeddings, visual_positions,
                )

                # Step 2: Run layer normally → POST-layer hidden_states
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

                # Step 3: RDC greedy selection on POST-layer visual tokens
                post_visual = hidden_states.index_select(1, visual_positions)  # (B, V, D)
                retained_num = min(int(target), post_visual.shape[1])

                selected_t = self._rdc_greedy_select(attn, post_visual, retained_num, device)

                # Step 4: Rebuild hidden_states from an explicit retain mask.
                # (qwen2_5_vl/modeling_qwen2_5_vl_starv3.py:494-499 -- correct for any visual layout,
                # unlike the old `cat(prefix, kept, suffix)` which assumed one contiguous window.)
                orig_seq_len = hidden_states.shape[1]
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

                # Qwen3-VL rotary_emb returns cos/sin of shape [B, S, head_dim] -- apply_interleaved_mrope
                # collapses the leading 3-axis, unlike Qwen2.5-VL's [3, B, S, head_dim].
                # Gathering dim 2 therefore indexes head_dim (=128), not the sequence: IndexError on any
                # real prompt. Recompute from the already-pruned position_ids instead.
                position_embeddings = self.rotary_emb(hidden_states, position_ids)

                # POST-PRUNE mask site. All THREE arguments matter and must change together:
                #  * past_key_values=None -- layers 0..K-1 hold an S-long cache, so masking_utils:803-804
                #    would set kv_length = new_len + S and build an over-wide mask; sdpa_attention_forward
                #    does NOT slice it (verified: no `attention_mask[...]` anywhere in that file), so
                #    SDPA raises a shape error against new_len-long keys.
                #  * position_ids=None    -- pruning leaves gapped absolute positions; MU:823 would read
                #    them as packed-sequence boundaries, MU:732 turns every prune gap into a pseudo-
                #    document and MU:928-930 ANDs a block-diagonal mask onto the causal one, blinding the
                #    answer tokens to every retained visual token (RCA: POPE-120 acc 37.5 vs 86.7 fixed).
                #    (Also enforced inside `_get_causal_mask`; passed here too so the call site reads true.)
                #  * attention_mask=None  -- with the two above, MU:809-810 sets kv_length = new_len and
                #    _ignore_causal_mask_sdpa (MU:233) returns None => plain is_causal. Correct.
                # Do NOT renumber the retained positions contiguously: that corrupts MRoPE.
                causal_mask = self._get_causal_mask(
                    hidden_states, None, cache_position, None, position_ids=None
                )
                if os.environ.get("ENABLE_DEBUG", "0") not in ("", "0", "false", "False"):
                    # Only valid at the POST-PRUNE site and only under sdpa (bool mask). At the initial
                    # site with left padding this would fire spuriously; under 'eager' the mask is an
                    # additive float and .sum() would not equal seq_len.
                    assert causal_mask is None or bool(
                        causal_mask[0, 0, -1].sum() == hidden_states.shape[1]
                    ), "post-prune causal mask is not plain-causal (packed-sequence poisoning)"

                # DeepStack rows must shrink with the sequence, or a later injection raises on the
                # `hidden_states[visual_pos_masks, :] + visual_embeds` add. Mask-driven, so it stays
                # correct regardless of where the visual tokens sit.
                if visual_pos_masks is not None:
                    _vis_pos = visual_pos_masks[0].nonzero(as_tuple=True)[0]
                    _surv = torch.isin(_vis_pos, retain_token_indices.to(_vis_pos.device))
                    if deepstack_visual_embeds is not None:
                        deepstack_visual_embeds = [d[_surv] for d in deepstack_visual_embeds]
                    visual_pos_masks = visual_pos_masks[:, retain_token_indices.to(visual_pos_masks.device)]

                # Re-map the surviving visual tokens into the NEW index space, so the second prune
                # point (layer 21) addresses the right rows. qwen2_5_vl/..._starv3.py:510-512.
                self.visual_token_positions = torch.searchsorted(retain_token_indices, kept_visual_abs)

                prev_length = self.current_visual_length
                self.current_visual_length = selected_t.numel()
                tops_enabled = self.current_visual_length > 0
                self.stage2_events.append((layer_idx, prev_length, self.current_visual_length))
                self.stage2_reported_layers.add(layer_idx)

                print(f"[TOPS S2] Layer {layer_idx} done: "
                      f"{prev_length} → {self.current_visual_length} tokens, "
                      f"total_seq={hidden_states.shape[1]}")

            else:
                if (
                    tops_enabled
                    and self.current_visual_length is not None
                    and layer_idx in self.pruning_layers
                    and layer_idx not in self.stage2_reported_layers
                ):
                    target = self.pruning_layers[layer_idx]
                    print(f"[TOPS S2] Layer {layer_idx}: skip (current {self.current_visual_length} ≤ target {target})")
                    self.stage2_skip_events.append((layer_idx, self.current_visual_length, target))
                    self.stage2_reported_layers.add(layer_idx)

                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            # DeepStack: upstream injects AFTER decoder layers 0,1,2 (0-indexed) --
            # `layer_idx in range(len(deepstack_visual_embeds))`. The config's [8,16,24] are the
            # *ViT* blocks the features are extracted from, NOT text layers.
            # NOTE: `layer_idx` here is 1-BASED (self_attn.layer_idx + 1), upstream's is 0-based.
            # Placed at loop-body level so it runs in BOTH branches above.
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

        hidden_states = self.norm(hidden_states)

        if seq_length > 1 and self.current_visual_length is not None:
            average_tokens = visual_token_accumulator / len(self.layers) if self.layers else 0.0
            self.visual_token_num = float(average_tokens)
            self.visual_token_len = float(average_tokens)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None

        if seq_length > 1 and self.prefill_done and not self._stage_summary_emitted:
            total_layers = len(self.layers)
            observed_avg = self._compute_observed_average(total_layers)
            schedule_avg = self._compute_schedule_average(self.stage1_expected_tokens, total_layers)
            print("\n[TOPS] Stage summary:")
            if self.stage1_actual_tokens is not None:
                print(f"  Stage 1: {self.stage0_visual_tokens} → {self.stage1_actual_tokens} "
                      f"(expected cap {self.stage1_expected_tokens})")
            if self.stage2_events:
                for layer_idx, before, after in self.stage2_events:
                    print(f"  Stage 2 Layer {layer_idx}: {before} → {after}")
                print(f"  Final visual tokens: {self.current_visual_length}")
            else:
                print("  Stage 2: no pruning applied")
            if observed_avg is not None:
                print(f"  Layer-avg tokens (observed): {observed_avg:.2f} (target {self.target_visual_tokens})")
            if schedule_avg is not None:
                print(f"  Layer-avg tokens (schedule): {schedule_avg:.2f}")
            self._stage_summary_emitted = True

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=None,
        )
