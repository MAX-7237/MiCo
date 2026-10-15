# coding=utf-8
"""
TOPSVariant3 Stage 2 for Qwen3-VL: Progressive three-step pruning with text-guided RDC selection.
Adapted from modeling_qwen2_5_vl_tops_variant3.py (the algorithm ground truth).

REBUILD 2026-07-24 (branch qwen3vl-rebuild-20260724).  Every patch below was located by
unique text, never by line number (the rebuild plan's line numbers are known-wrong).

  P2  RoPE is RECOMPUTED after pruning.  Qwen3-VL's `rotary_emb` returns cos/sin of shape
      [B, S, head_dim] (`apply_interleaved_mrope` collapses the leading 3-axis), unlike
      Qwen2.5-VL's [3, B, S, head_dim], so the old `pos_emb[:, :, retain_token_indices]`
      gather indexed head_dim (=128), not the sequence -> IndexError on any real prompt.
  P3  Post-prune causal mask is rebuilt with past_key_values=None AND position_ids=None
      (attention_mask was already None).  The INITIAL mask site is deliberately left
      untouched -- see `_get_causal_mask`.
  P4  `visual_pos_masks` / `deepstack_visual_embeds` / `**kwargs` accepted (None defaults).
  P5  DeepStack injection after decoder layers 0,1,2 (0-indexed), plus in-sync re-pruning
      of the DeepStack side-features whenever the sequence is pruned.
  P6  Explicit visual-position API `set_visual_token_positions(index_vector)`.  The
      `system_prompt_length = 15` window is GONE: Qwen3-VL has no system prompt and the
      first image token sits at index 4, so the old slice protected 11 image tokens from
      pruning and made 11 real text tokens deletable.  Rationale for an index vector over
      a (start, length) offset is the reference's own, verbatim:
      qwen2_5_vl/modeling_qwen2_5_vl_tops_variant3.py -- "Qwen prompts do not guarantee that
      visual tokens begin at a fixed offset".
  P8  The importance probe is now Qwen3's attention: q_norm/k_norm applied before RoPE,
      head/kv-head counts derived from the projections, `self_attn.scaling` for the scale.
  P9  36-layer budget schedule (see TOPS_VARIANT3_SCHEDULE below) + the `_compute_schedule_average`
      accounting fix (a prune layer executes at its PRE-prune count).
  P12 `seq_length > 1` prefill guard on `should_prune`.
"""

import math
import os
from typing import Dict, List, Optional, Set, Tuple, Union

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
except ImportError:  # documented Qwen2.5-VL fallback
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel as _BASE_MODEL_CLS
    _create_causal_mask_impl = None
    _USING_QWEN3 = False

logger = logging.get_logger(__name__)


def _debug_enabled() -> bool:
    """ENABLE_DEBUG gate (project rule: experiments always run with ENABLE_DEBUG on)."""
    return os.environ.get("ENABLE_DEBUG", "0") not in ("", "0", "false", "False")


# ======================================================================================
# P9 -- budget schedule
# ======================================================================================
#
# BUDGET CONVENTION -- this is an OPEN HUMAN DECISION (rebuild plan §4.1) and is NOT
# decided here.  The default is and stays "avg" (average-over-layers), which is the
# convention the Qwen2.5-VL ground-truth reference uses -- its own 28-layer comments show
# the identical accounting, e.g. `# initial=1024, (1024x12+281x4+77x12)/28=512`.
#
# To evaluate a different convention later, set the env var named by
# TOPS_VARIANT3_BUDGET_CONVENTION_ENV (or pass `convention=` to `get_tops_variant3_schedule`).  No
# decoder code has to change: only the table lookup below is affected.
#
TOPS_VARIANT3_BUDGET_CONVENTION_DEFAULT = "avg"
TOPS_VARIANT3_BUDGET_CONVENTION_ENV = "TOPS_EXPERIMENTAL_TOPSVariant3_BUDGET"

# --------------------------------------------------------------------------------------
# DEFAULT table -- "avg" (average-over-layers) convention.
# --------------------------------------------------------------------------------------
# Rows are [(prune_at_1based_layer, target_tokens), ...]; a layer executes at its
# PRE-prune count (probe -> run layer -> prune), and `layer_idx` is 1-based
# (`decoder_layer.self_attn.layer_idx + 1`).
TOPS_VARIANT3_SCHEDULE: Dict[str, Dict[int, List[Tuple[int, int]]]] = {
    "7b": {
        # Qwen2.5-VL-7B (28 layers) -- unchanged, mirrors the ground-truth reference.
        # Stages 12 | 4 | 12  =>  12*2T + 4a + 12b = 28T  <=>  a + 3b = T
        512: [(12, 281), (16, 77)],   # initial=1024, (1024*12+281*4+77*12)/28 = 512
        256: [(12, 139), (16, 39)],   # initial=512,  (512*12+139*4+39*12)/28  = 256
        128: [(12, 71),  (16, 19)],   # initial=256,  (256*12+71*4+19*12)/28   = 128
    },
    "8b": {
        # Qwen3-VL-8B, 36 text layers, prune at 1-indexed layers 15 and 21
        #   (15 = round(12/28*36) = 15.43 -> 15, 21 = round(16/28*36) = 20.57 -> 21,
        #    proportional to the 28-layer parent's 12/16).
        # Partition is 15 | 6 | 15: layers 1-15 hold 2T, layers 16-21 hold a,
        # layers 22-36 hold b.  A layer executes at its PRE-prune count, so
        #     15*2T + 6a + 15b = 36T   <=>   6a + 15b = 6T   <=>   2a + 5b = 2T
        #   T=128: 15*256  + 6*78  + 15*20 =  3840 +  468 +  300 =  4608 = 36*128  avg 128.0
        #   T=256: 15*512  + 6*151 + 15*42 =  7680 +  906 +  630 =  9216 = 36*256  avg 256.0
        #   T=512: 15*1024 + 6*307 + 15*82 = 15360 + 1842 + 1230 = 18432 = 36*512  avg 512.0
        # Integrality: 2a = 2T - 5b forces b even and a == T (mod 5); all three rows comply.
        # (a, b) is under-determined by one equation in two unknowns; these triples preserve
        # the reference's a/b ratio (~3.6).  See rebuild plan §4.2 -- also a human decision.
        # Conditional on stage 1 actually attaining 2T, i.e. N >= 2T; with N < 2T the run
        # spends strictly LESS than T per layer (safe, under budget).
        512: [(15, 307), (21, 82)],
        256: [(15, 151), (21, 42)],
        128: [(15, 78),  (21, 20)],
    },
    "4b": {
        # Not re-derived for the 4B geometry; inherited from the previous table.
        512: [(8, 281), (12, 77)],
        256: [(8, 139), (12, 39)],
        128: [(8, 71),  (12, 19)],
    },
    "2b": {
        # Not re-derived for the 2B geometry; inherited from the previous table.
        512: [(12, 281), (16, 77)],
        256: [(12, 139), (16, 39)],
        128: [(12, 71),  (16, 19)],
    },
}

# --------------------------------------------------------------------------------------
# ALTERNATE table -- "flat_final" convention.  NOT the default; nothing selects it unless
# a human sets the env var.  These are the PREVIOUS 8b rows, kept verbatim: their second
# entry is exactly T (512/256/128), i.e. they encode "the final stage holds a flat T per
# layer" rather than "the layer average is T".  Under the avg convention they spend
# ~1.45T per layer (2a+5b = 3760/1880/960 against the required 1024/512/256).  Recorded
# so the old numbers stay reproducible, not because either convention is endorsed here.
#
# NOTE: a *strictly* flat T-in-every-layer schedule is not expressible for TOPSPrototype-Pro at
# all, because stage 1 keeps min(2T, N) for the first 15 layers by construction; making
# it flat would be a method change, not a reporting change (rebuild plan §4.1).
# --------------------------------------------------------------------------------------
TOPS_VARIANT3_SCHEDULE_FLAT_FINAL: Dict[str, Dict[int, List[Tuple[int, int]]]] = {
    "8b": {
        512: [(15, 600), (21, 512)],
        256: [(15, 300), (21, 256)],
        128: [(15, 160), (21, 128)],
    },
}

TOPS_VARIANT3_SCHEDULE_TABLES: Dict[str, Dict[str, Dict[int, List[Tuple[int, int]]]]] = {
    "avg": TOPS_VARIANT3_SCHEDULE,
    "flat_final": TOPS_VARIANT3_SCHEDULE_FLAT_FINAL,
}

# Schedule-only ablation for Qwen3-VL-8B. Keep the same alpha=2 global QR
# stage 1 and TOPSPrototype-Pro average_all stage 2, but delay the first decoder prune
# from layer 12 to layer 14. The 2:1 post-prune ratio and exact layer-average
# budget are both preserved:
#   (14*256 + 10*64 + 12*32) / 36 = 128.
LLAVA_ALIGNED_SCHEDULE: Dict[str, Dict[int, List[Tuple[int, int]]]] = {
    "8b": {
        128: [(14, 64), (24, 32)],
    },
}


def resolve_budget_convention() -> str:
    """Which schedule table to use.  Default never changes without a human setting the env var."""
    convention = os.environ.get(TOPS_VARIANT3_BUDGET_CONVENTION_ENV, TOPS_VARIANT3_BUDGET_CONVENTION_DEFAULT)
    convention = (convention or TOPS_VARIANT3_BUDGET_CONVENTION_DEFAULT).strip().lower()
    if convention not in TOPS_VARIANT3_SCHEDULE_TABLES:
        logger.warning_once(
            "Unknown %s=%r; falling back to %r. Known conventions: %s",
            TOPS_VARIANT3_BUDGET_CONVENTION_ENV, convention,
            TOPS_VARIANT3_BUDGET_CONVENTION_DEFAULT, sorted(TOPS_VARIANT3_SCHEDULE_TABLES),
        )
        convention = TOPS_VARIANT3_BUDGET_CONVENTION_DEFAULT
    return convention


def get_tops_variant3_schedule(
    scale: str,
    target_visual_tokens: int,
    convention: Optional[str] = None,
) -> Tuple[Optional[List[Tuple[int, int]]], str]:
    """Return (schedule_rows, convention_actually_used) for `scale` / `target_visual_tokens`."""
    convention = convention or resolve_budget_convention()
    schedule = TOPS_VARIANT3_SCHEDULE_TABLES[convention].get(scale, {}).get(target_visual_tokens)
    if schedule is None and convention != TOPS_VARIANT3_BUDGET_CONVENTION_DEFAULT:
        logger.warning_once(
            "No %r schedule for scale=%s target=%s; falling back to the %r table.",
            convention, scale, target_visual_tokens, TOPS_VARIANT3_BUDGET_CONVENTION_DEFAULT,
        )
        convention = TOPS_VARIANT3_BUDGET_CONVENTION_DEFAULT
        schedule = TOPS_VARIANT3_SCHEDULE_TABLES[convention].get(scale, {}).get(target_visual_tokens)
    return schedule, convention


class TOPSVariant3Qwen3VLModel(_BASE_MODEL_CLS):
    """
    TOPS-Variant3 stage-2 for Qwen3-VL: progressive text-guided token pruning inside LLM layers.
    Adapted from TOPSVariant3Qwen2_5_VLModel.  Assumes Stage-1 has already reduced the visual
    tokens to about `2 x target_visual_tokens` and has pushed their absolute positions via
    `set_visual_token_positions`.
    """

    def __init__(
        self,
        config,
        visual_token_num: int,
        schedule_variant: str = "native",
        text_agg_mode: str = "multi_token",
    ):
        super().__init__(config)
        lm = self.language_model
        self.embed_tokens = lm.embed_tokens
        self.layers = lm.layers
        self.norm = lm.norm
        self.rotary_emb = lm.rotary_emb
        self._lm_update_causal_mask = getattr(lm, '_update_causal_mask', None)

        # P6: NO system_prompt_length.  Qwen3-VL has no system prompt and the first image
        # token is at index 4 (measured against the shipped chat template), not 15 as in
        # Qwen2.5-VL -- an 11-token window shift that both protected 11 real image tokens
        # from pruning and made 11 real TEXT tokens deletable.  The window is pushed in
        # explicitly by the wrapper instead; see the reference rationale in
        # qwen2_5_vl/modeling_qwen2_5_vl_tops_variant3.py.
        self.visual_token_positions: Optional[torch.LongTensor] = None

        # Nominal Stage-0 token count, used ONLY for budget bookkeeping before the first
        # prefill.  This is NOT a window offset and must never be used to slice a sequence.
        self.visual_token_length = 1296
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

        self.schedule_variant = (schedule_variant or "native").strip().lower()
        self.text_agg_mode = (text_agg_mode or "multi_token").strip().lower()
        if self.text_agg_mode not in ("multi_token", "average_all"):
            raise ValueError(
                f"Unsupported TOPS-Variant3 text_agg_mode={self.text_agg_mode!r}; "
                "expected 'multi_token' or 'average_all'."
            )

        if self.schedule_variant == "llava_aligned":
            schedule = LLAVA_ALIGNED_SCHEDULE.get(self.scale, {}).get(self.target_visual_tokens)
            self.budget_convention = "avg"
            if schedule is None:
                raise ValueError(
                    "LLaVA-aligned TOPSPrototype-Pro schedule is only calibrated for "
                    f"Qwen3-VL-8B T=128, got scale={self.scale!r} "
                    f"target={self.target_visual_tokens}."
                )
        elif self.schedule_variant == "native":
            schedule, self.budget_convention = get_tops_variant3_schedule(
                self.scale, self.target_visual_tokens
            )
        else:
            raise ValueError(
                f"Unsupported TOPS-Variant3 schedule_variant={self.schedule_variant!r}."
            )

        if schedule is None:
            self.pruning_layers: Dict[int, int] = {}
            if visual_token_num > 0:
                logger.warning_once(
                    "TOPS-Variant3 schedule missing for target %s on scale %s. Progressive pruning disabled.",
                    visual_token_num, self.scale,
                )
        else:
            self.pruning_layers = {layer_idx: target for layer_idx, target in schedule}

        print(f"[TOPS-Variant3 Init] Qwen3-VL scale={self.scale} ({n_layers} layers), "
              f"target={self.target_visual_tokens}, budget_convention={self.budget_convention} "
              f"(env {TOPS_VARIANT3_BUDGET_CONVENTION_ENV}, default {TOPS_VARIANT3_BUDGET_CONVENTION_DEFAULT}), "
              f"schedule_variant={self.schedule_variant}, text_agg_mode={self.text_agg_mode}, "
              f"pruning_layers={self.pruning_layers}, deepstack={'on' if _USING_QWEN3 else 'n/a'}")

        self.reset_state()

    # ----------------------------------------------------------------------------------
    # causal mask
    # ----------------------------------------------------------------------------------
    def _get_causal_mask(self, input_tensor, attention_mask, cache_position, past_key_values, position_ids=None):
        """
        Shared by the two -- and only two -- mask sites in this file.  They are DIFFERENT
        and must stay different:

          * INITIAL site (`inputs_embeds, attention_mask, cache_position, past_key_values,
            position_ids=text_position_ids`) is CORRECT AS-IS.  The cache is empty there,
            so it sizes the mask correctly, and the packed branch (masking_utils' three-way
            `position_ids is not None and attention_mask is None and past_key_values is
            None`) is already dead because past_key_values is non-None.

          * POST-PRUNE site passes `None` for attention_mask, past_key_values AND
            position_ids -- see the comment at that call site.

        The rebuild plan's P3b proposed hard-dropping `position_ids` inside this helper.
        That is NOT done here: it would silently change the initial site as well, and the
        initial site is correct as-is.  The fix is landed at the one call site that needs it.
        """
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

    # ----------------------------------------------------------------------------------
    # state / visual-window API (P6)
    # ----------------------------------------------------------------------------------
    def set_visual_token_positions(self, positions: torch.LongTensor) -> None:
        """
        Absolute positions (in the post-stage-1 sequence) of the surviving image tokens.
        Deliberately NOT cleared by `reset_state()`: `reset_state` runs at the top of
        `forward`, i.e. AFTER the wrapper has pushed the positions for this prefill.
        """
        self.visual_token_positions = positions.detach()

    def reset_state(self):
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

    # ----------------------------------------------------------------------------------
    # importance probe
    # ----------------------------------------------------------------------------------
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        # Byte-identical to transformers' qwen3_vl.rotate_half; kept local so the probe
        # works on both the Qwen3 and the Qwen2.5 fallback import path.
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _select_text_raters(
        self,
        hidden_states: torch.Tensor,
        visual_positions: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Return the ABSOLUTE positions of the text tokens that rate the visual tokens.
        Text = everything that is not in `visual_positions` (P6: no `hidden_states[:,
        visual_end:]` tail slice -- text also precedes the image).
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

        # [batch, text_len, visual_len]
        text_visual_sim = torch.matmul(text_norm, visual_norm.transpose(1, 2))
        text_importance = text_visual_sim.mean(dim=-1)   # average over visual tokens
        if text_importance.dim() == 2:
            text_importance = text_importance.mean(dim=0)  # aggregate across batch

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
        self_attn = decoder_layer.self_attn

        attn_input = hidden_states
        input_layernorm = getattr(decoder_layer, "input_layernorm", None)
        if input_layernorm is not None:
            attn_input = input_layernorm(attn_input)

        bsz, seq_len, _ = attn_input.shape

        # P8: Qwen3VLTextAttention defines head_dim / num_key_value_groups / scaling but
        # NEITHER num_heads NOR num_key_value_heads -- reading them is an AttributeError.
        # `head_dim` must be bound BEFORE the two lines that use it (in the pre-rebuild
        # file it sat one line BELOW them).
        head_dim = self_attn.head_dim
        num_heads = self_attn.q_proj.out_features // head_dim
        num_kv_heads = self_attn.k_proj.out_features // head_dim

        if _debug_enabled():
            print(f"    [TOPS-Variant3 probe] bsz={bsz} seq_len={seq_len} heads={num_heads} "
                  f"kv_heads={num_kv_heads} head_dim={head_dim} visual={visual_positions.numel()}")

        # P8: Qwen3 applies per-head RMSNorm to q and k BEFORE RoPE and before the
        # transpose.  Omitting it makes the probe measure a different function than the
        # model's own attention.  `getattr` keeps the Qwen2.5-VL fallback path working
        # (that attention has no q_norm/k_norm) and never double-applies.
        query_states = self_attn.q_proj(attn_input).view(bsz, seq_len, num_heads, head_dim)
        key_states = self_attn.k_proj(attn_input).view(bsz, seq_len, num_kv_heads, head_dim)
        q_norm = getattr(self_attn, "q_norm", None)
        k_norm = getattr(self_attn, "k_norm", None)
        if q_norm is not None:
            query_states = q_norm(query_states)
        if k_norm is not None:
            key_states = k_norm(key_states)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        if num_kv_heads != num_heads:
            repeat_factor = num_heads // num_kv_heads
            key_states = key_states.repeat_interleave(repeat_factor, dim=1)

        # Qwen3-VL's rotary_emb returns [B, S, head_dim]; unsqueeze the head axis so it
        # broadcasts against [B, H, S, head_dim].  Equivalent to upstream
        # apply_rotary_pos_emb(..., unsqueeze_dim=1).
        cos, sin = position_embeddings
        if cos.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        query_states = (query_states * cos) + (self._rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)

        if self.text_agg_mode == "average_all":
            # Match LLaVA TOPSPrototype-Pro's TEXT_AGG_MODE=average_all exactly: every
            # text token after the visual block rates every visual token. Prefix
            # template tokens before the image are not question tokens.
            last_visual = int(visual_positions.max())
            text_positions = torch.arange(
                last_visual + 1,
                seq_len,
                device=hidden_states.device,
                dtype=torch.long,
            )
        else:
            text_positions = self._select_text_raters(hidden_states, visual_positions)

        if text_positions.numel() == 0:
            text_positions = torch.tensor(
                [seq_len - 1], device=hidden_states.device, dtype=torch.long
            )

        query_raters = torch.index_select(query_states, 2, text_positions)

        # P8: use the module's own scale factor (numerically 1/sqrt(head_dim), but tracks upstream).
        scaling = getattr(self_attn, "scaling", 1.0 / math.sqrt(head_dim))

        if self.text_agg_mode == "average_all":
            # LLaVA averages the decoder layer's actual causal attention after
            # slicing text-query rows and visual-key columns. Reconstruct that
            # same full-softmax denominator here instead of normalising over
            # visual keys only.
            attn_scores = torch.matmul(
                query_raters, key_states.transpose(-2, -1)
            ) * scaling
            key_positions = torch.arange(
                seq_len, device=hidden_states.device, dtype=torch.long
            )
            causal = key_positions.unsqueeze(0) <= text_positions.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(
                ~causal.unsqueeze(0).unsqueeze(0),
                torch.finfo(attn_scores.dtype).min,
            )
            attn_weights = F.softmax(
                attn_scores, dim=-1, dtype=torch.float32
            )
            visual_weights = attn_weights[..., visual_positions]
            visual_importance = (
                visual_weights.mean(dim=2).mean(dim=1).mean(dim=0)
            )
        else:
            key_visual = torch.index_select(key_states, 2, visual_positions)
            attn_scores = (
                torch.matmul(query_raters, key_visual.transpose(-2, -1))
                * scaling
            )
            attn_weights = F.softmax(
                attn_scores, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_weights = attn_weights.mean(dim=2).mean(dim=1)
            visual_importance = attn_weights.mean(dim=0)

        return visual_importance.float()

    # ----------------------------------------------------------------------------------
    # budget accounting (P9)
    # ----------------------------------------------------------------------------------
    def _compute_schedule_average(
        self,
        stage1_tokens: Optional[int],
        total_layers: int,
    ) -> Optional[float]:
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
                span = layer_idx - current_layer
                total_tokens += span * current_tokens

            # P9: the prune layer runs BEFORE its prune, so charge it its PRE-prune count.
            # Charging `target` here under-counted the true spend of the schedule.
            total_tokens += current_tokens
            new_tokens = float(min(current_tokens, target))
            current_tokens = new_tokens
            current_layer = layer_idx + 1

        if current_layer <= total_layers:
            total_tokens += (total_layers - current_layer + 1) * current_tokens

        return total_tokens / total_layers

    def _compute_observed_average(self, total_layers: int) -> Optional[float]:
        if not self.prefill_done or self.stage1_actual_tokens is None or total_layers <= 0:
            return None

        total_tokens = 0.0
        current_tokens = float(self.stage1_actual_tokens)
        current_layer = 1

        for layer_idx, before_tokens, after_tokens in sorted(self.stage2_events):
            if layer_idx < current_layer or layer_idx > total_layers:
                continue

            if before_tokens is not None and abs(before_tokens - current_tokens) > 1e-5:
                logger.debug(
                    "TOPS-Variant3 average mismatch at layer %s: expected %.2f, recorded %.2f",
                    layer_idx, current_tokens, before_tokens,
                )

            if layer_idx > current_layer:
                span = layer_idx - current_layer
                total_tokens += span * current_tokens

            # Same convention as _compute_schedule_average: the prune layer executes at
            # its pre-prune count.
            total_tokens += current_tokens
            current_tokens = float(after_tokens)
            current_layer = layer_idx + 1

        if current_layer <= total_layers:
            total_tokens += (total_layers - current_layer + 1) * current_tokens

        return total_tokens / total_layers

    # ----------------------------------------------------------------------------------
    # forward
    # ----------------------------------------------------------------------------------
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
        # P4: DeepStack side-channel (import-safe no-op when the wrapper does not push it)
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

        is_prefill = _is_empty_cache(past_key_values)
        if is_prefill:
            self.reset_state()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if self.is_gradient_checkpointing and self.training and use_cache:
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
        text_position_ids = None
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        elif position_ids.dim() == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]  # [B, S]
            position_ids = position_ids[1:]      # [3, B, S]
        # else: already [3, B, S]

        # INITIAL mask site -- CORRECT AS-IS, do not change (see `_get_causal_mask`).
        causal_mask = self._get_causal_mask(
            inputs_embeds, attention_mask, cache_position, past_key_values, position_ids=text_position_ids
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        seq_length = inputs_embeds.shape[1]

        if seq_length > 1 and not self.prefill_done:
            # P6c: the window comes from the wrapper, never from a heuristic.
            if self.visual_token_positions is None:
                raise RuntimeError(
                    "TOPS-Variant3 stage-2: visual token positions were not pushed by the wrapper. "
                    "The 'seq_length - 15 - 10' heuristic is off by 11 tokens on Qwen3-VL and must "
                    "not be used as a fallback. Call model.set_visual_token_positions(...) first."
                )
            detected_visual_len = int(self.visual_token_positions.numel())

            self.current_visual_length = detected_visual_len
            self.original_visual_length = detected_visual_len
            self.stage1_expected_tokens = detected_visual_len
            self.stage1_actual_tokens = detected_visual_len
            self.prefill_done = True
            self.visual_token_sum = 0.0
            print(f"[TOPS-Variant3] Prefill: seq_length={seq_length}, visual_len={detected_visual_len} "
                  f"(stage-1 cap {self.stage1_keep_tokens}, target {self.target_visual_tokens}, "
                  f"convention {self.budget_convention})")

        star_enabled = (
            is_prefill
            and self.prefill_done
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

            # Accumulate BEFORE the layer runs: a layer executes at its PRE-prune count.
            if seq_length > 1 and self.current_visual_length is not None:
                visual_token_accumulator += float(self.current_visual_length)

            should_prune = (
                star_enabled
                and seq_length > 1  # P12: prefill guard -- never prune on a decode step
                and layer_idx in self.pruning_layers
                and self.current_visual_length is not None
                and self.current_visual_length > self.pruning_layers[layer_idx]
            )

            visual_positions = None
            selected_t = None

            if should_prune:
                target = self.pruning_layers[layer_idx]
                visual_positions = self.visual_token_positions
                print(f"\n[TOPS-Variant3 S2] Layer {layer_idx}: {self.current_visual_length} → {target} tokens")

                visual_importance = self._compute_visual_token_importance(
                    hidden_states, decoder_layer, position_embeddings, visual_positions,
                )

                keep = max(0, min(int(target), int(visual_importance.shape[0])))
                if keep == 0:
                    selected_t = visual_importance.new_empty((0,), dtype=torch.long)
                else:
                    selected_t = torch.sort(visual_importance.topk(keep).indices).values

                if _debug_enabled():
                    print(f"    [TOPS-Variant3 S2] importance: min={visual_importance.min():.6f} "
                          f"max={visual_importance.max():.6f} mean={visual_importance.mean():.6f} "
                          f"keep={selected_t.numel()}")
            else:
                if (
                    star_enabled
                    and self.current_visual_length is not None
                    and layer_idx in self.pruning_layers
                    and layer_idx not in self.stage2_reported_layers
                ):
                    target = self.pruning_layers[layer_idx]
                    print(f"[TOPS-Variant3 S2] Layer {layer_idx}: skip (current {self.current_visual_length} ≤ target {target})")
                    self.stage2_skip_events.append((layer_idx, self.current_visual_length, target))
                    self.stage2_reported_layers.add(layer_idx)

            # The layer always runs on the UNPRUNED sequence: TOPS-Variant3 prunes AFTER the layer
            # (probe -> run layer -> prune), the opposite of FastV.
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            if should_prune:
                device = hidden_states.device
                orig_seq_len = hidden_states.shape[1]
                prev_length = self.current_visual_length

                # P6: retain-mask build over the explicit index vector -- no visual_start /
                # visual_end slicing anywhere.
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

                # P2: Qwen3-VL's rotary_emb returns cos/sin of shape [B, S, head_dim]
                # (apply_interleaved_mrope collapses the leading 3-axis), unlike
                # Qwen2.5-VL's [3, B, S, head_dim].  Gathering dim 2 therefore indexes
                # head_dim (=128), not the sequence: IndexError on any real prompt.
                # Recompute from the already-pruned position_ids instead.
                position_embeddings = self.rotary_emb(hidden_states, position_ids)

                # P3: BOTH remaining arguments matter and must change together.
                #  * past_key_values=None -- layers 0..K-1 still hold an S-long cache, so
                #    masking_utils would set kv_length = new_len + S and build an over-wide
                #    [1, 1, new_len, new_len+S] mask; sdpa_attention_forward does NOT slice
                #    it (verified: no `attention_mask[..., :key_len]` anywhere), so SDPA
                #    raises a shape error against the new_len-long keys.
                #  * position_ids=None -- pruning leaves gapped absolute positions; with
                #    attention_mask None and past_key_values None, masking_utils would read
                #    them as packed-sequence boundaries, turn every prune gap into a
                #    pseudo-document and AND a block-diagonal mask onto the causal one,
                #    blinding the answer tokens to every retained visual token
                #    (measured: POPE-120 acc 37.5 broken vs 86.7 fixed).
                #  * attention_mask is already the literal None below.
                # With all three None, kv_length = new_len and the sdpa skip returns None,
                # i.e. plain is_causal -- correct.
                # Do NOT renumber the retained positions contiguously: that corrupts MRoPE.
                causal_mask = self._get_causal_mask(
                    hidden_states, None, cache_position, None, position_ids=None
                )

                if _debug_enabled():
                    # Only meaningful under sdpa, where the mask is boolean (P0 pins sdpa
                    # on this load path).  Post-prune site only: at the initial site with
                    # left-padding this would fire spuriously.
                    assert (
                        causal_mask is None
                        or causal_mask.dtype != torch.bool
                        or bool(causal_mask[0, 0, -1].sum() == hidden_states.shape[1])
                    ), "post-prune causal mask is not plain-causal (packed-sequence poisoning)"

                # P5b: re-prune the DeepStack side-features in sync.  Mask-driven, so it is
                # correct regardless of where the visual tokens sit (batch-1 prefill).
                if visual_pos_masks is not None:
                    _vis_pos = visual_pos_masks[0].nonzero(as_tuple=True)[0]
                    _surv = torch.isin(_vis_pos, retain_token_indices)
                    if deepstack_visual_embeds is not None:
                        deepstack_visual_embeds = [d[_surv] for d in deepstack_visual_embeds]
                    visual_pos_masks = visual_pos_masks[:, retain_token_indices]

                # Re-map the window into the compacted sequence for the NEXT prune point.
                self.visual_token_positions = torch.searchsorted(retain_token_indices, kept_visual_abs)
                self.current_visual_length = int(selected_t.numel())
                star_enabled = self.current_visual_length > 0
                self.stage2_events.append((layer_idx, prev_length, self.current_visual_length))
                self.stage2_reported_layers.add(layer_idx)

                print(f"[TOPS-Variant3 S2] Layer {layer_idx} done: {prev_length} → {self.current_visual_length} "
                      f"(seq {orig_seq_len} → {new_len}, text {new_len - self.current_visual_length})")

            # P5a: DeepStack injects AFTER decoder layers 0,1,2 (0-indexed) --
            # `layer_idx in range(len(deepstack_visual_embeds))` upstream.  [8,16,24] are
            # the *ViT* blocks the features come from, NOT text layers.
            # NOTE: layer_idx here is 1-BASED, upstream's is 0-based.
            if _USING_QWEN3 and deepstack_visual_embeds is not None and visual_pos_masks is not None:
                _ds = layer_idx - 1
                if 0 <= _ds < len(deepstack_visual_embeds):
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
            print("\n[TOPS-Variant3] Stage summary:")
            print(f"  Budget convention: {self.budget_convention} "
                  f"(target {self.target_visual_tokens}, {total_layers} layers)")
            if self.stage1_actual_tokens is not None:
                print(f"  Stage 1 (pushed by wrapper): {self.stage1_actual_tokens} visual tokens")
            if self.stage2_events:
                for layer_idx, before, after in self.stage2_events:
                    print(f"  Layer {layer_idx}: {before} → {after}")
                print(f"  Final visual tokens: {self.current_visual_length}")
            else:
                print("  Stage 2: no pruning applied (schedule missing or already at target)")
            for layer_idx, current_len, target in self.stage2_skip_events:
                print(f"  Layer {layer_idx}: skipped (current {current_len} ≤ target {target})")
            if observed_avg is not None:
                print(f"  Layer-avg tokens (observed): {observed_avg:.2f}")
            if schedule_avg is not None:
                print(f"  Layer-avg tokens (schedule, stage1={self.stage1_expected_tokens}): {schedule_avg:.2f}")
            self._stage_summary_emitted = True

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=None,
        )
