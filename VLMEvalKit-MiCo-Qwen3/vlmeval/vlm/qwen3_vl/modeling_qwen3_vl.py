# coding=utf-8
"""
Qwen3-VL pruning wrapper: Stage 1 (vision-side attention-guided token selection)
and dispatch to Stage 2 pruning models (TOPS, TOPSVariant3, FastV).

Adapted from modeling_qwen2_5_vl.py for Qwen3-VL.

Key differences from Qwen2.5-VL:
  - patch_size=16 (vs 28), vision depth=27, spatial_merge_size=2 (same)
  - image_token_id=151655, video_token_id=151656
  - DeepStack visual indexes at blocks [8, 16, 24] (multi-scale features)
  - mrope_interleaved=True in the LLM text model
  - Requires transformers 4.57+ for Qwen3VLForConditionalGeneration
"""

import math
import os
import types
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss

from transformers.utils import logging

# Attempt Qwen3-VL imports (transformers 5.7+)
try:
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel,
        apply_rotary_pos_emb_vision,
    )
    _CausalLMOutputCls = None
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast
        _CausalLMOutputCls = Qwen3VLCausalLMOutputWithPast
    except ImportError:
        from transformers.modeling_outputs import CausalLMOutputWithPast as _CausalLMOutputCls
    _BASE_CONDITIONAL_GEN = Qwen3VLForConditionalGeneration
    _BASE_MODEL_CLS = Qwen3VLModel
    _USING_QWEN3 = True
    print("[modeling_qwen3_vl] Using native Qwen3-VL from transformers")
except ImportError:
    from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration as _BASE_CONDITIONAL_GEN
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLModel as _BASE_MODEL_CLS,
        Qwen2_5_VLCausalLMOutputWithPast as _CausalLMOutputCls,
        apply_rotary_pos_emb_vision,
    )
    _USING_QWEN3 = False
    print("[modeling_qwen3_vl] WARNING: Qwen3-VL not found in transformers. "
          "Please upgrade to transformers>=5.7. Falling back to Qwen2.5-VL structure (may not work).")

from .modeling_qwen3_vl_fastv import FastVQwen3VLModel
from .modeling_qwen3_vl_tops_variant3 import TOPSVariant3Qwen3VLModel
from .modeling_qwen3_vl_tops import TOPSQwen3VLModel
from .modeling_qwen3_vl_duet import DuetQwen3VLModel
from .modeling_qwen3_vl_dart import DARTQwen3VLModel
from .modeling_qwen3_vl_sparsevlm import SparseVLMQwen3VLModel
from .modeling_qwen3_vl_apet import ApETQwen3VLModel, apet_stage1, apet_budget
from .modeling_qwen3_vl_coverprune2 import CoverPrune2Qwen3VLModel
from .modeling_qwen3_vl_mico_coverage_vit_proj import (
    MiCoCoverageVitProjLlm12Qwen3VLModel,
    inner_weighted_raw_cls_select,
)

logger = logging.get_logger(__name__)


def _is_mico(method: str) -> bool:
    prefix = "mico_"
    return isinstance(method, str) and method.startswith(prefix) and method[len(prefix):].isdigit()


def _is_mico_s1(method: str) -> bool:
    return method == "mico_s1" or _is_mico(method)


def _qr_residual_select_exact(
    features: torch.Tensor,
    keep: int,
    forced_indices: Optional[List[int]] = None,
) -> torch.LongTensor:
    """Pure pivoted-QR token selection, optionally conditioned on forced pivots."""
    flat = features.detach().reshape(features.shape[0], -1).to(torch.float32)
    n_tokens = int(flat.shape[0])
    keep = min(max(1, int(keep)), n_tokens)
    initial_norms = torch.square(flat).sum(dim=1)
    if not torch.isfinite(initial_norms).all():
        raise ValueError('[QR-STAGE1] non-finite token norms')

    residual = flat.clone()
    available = torch.ones(n_tokens, dtype=torch.bool, device=flat.device)
    selected: List[int] = []
    eps = 1e-8

    for idx in forced_indices or []:
        idx = int(idx)
        if idx < 0 or idx >= n_tokens:
            raise ValueError(f'[QR-STAGE1] forced index out of range: {idx}/{n_tokens}')
        if not bool(available[idx]):
            continue
        selected.append(idx)
        available[idx] = False
        pivot_norm = residual[idx].norm()
        if torch.isfinite(pivot_norm) and pivot_norm > eps:
            basis = residual[idx] / pivot_norm
            residual = residual - torch.outer(residual @ basis, basis)
        if len(selected) == keep:
            break

    while len(selected) < keep:
        residual_norms = torch.square(residual).sum(dim=1)
        residual_norms = torch.where(
            available, residual_norms, torch.full_like(residual_norms, -1.0)
        )
        pivot = int(torch.argmax(residual_norms).item())
        if residual_norms[pivot] <= eps:
            remaining = torch.nonzero(available).flatten()
            remaining = remaining[
                torch.argsort(initial_norms[remaining], descending=True, stable=True)
            ]
            selected.extend(
                int(i) for i in remaining[:keep - len(selected)].tolist()
            )
            break
        selected.append(pivot)
        available[pivot] = False
        basis = residual[pivot] / (residual[pivot].norm() + eps)
        residual = residual - torch.outer(residual @ basis, basis)

    result = torch.tensor(selected, dtype=torch.long, device=flat.device)
    if result.numel() != keep or torch.unique(result).numel() != keep:
        raise RuntimeError(
            f'[QR-STAGE1] exact unique selection failed: '
            f'count={result.numel()} unique={torch.unique(result).numel()} expected={keep}'
        )
    return result


def _split_spatial_blocks(height: int, width: int, count: int) -> List[Tuple[int, int, int, int]]:
    """Split a 2-D grid into exactly ``count`` non-empty rectangular blocks."""
    count = min(max(1, int(count)), int(height * width))
    blocks = [(0, int(height), 0, int(width))]
    while len(blocks) < count:
        candidates = [
            (-(r1 - r0) * (c1 - c0), idx, r0, r1, c0, c1)
            for idx, (r0, r1, c0, c1) in enumerate(blocks)
            if (r1 - r0) > 1 or (c1 - c0) > 1
        ]
        if not candidates:
            raise RuntimeError(
                f'[QWEN-BLOCK-QR] cannot create {count} blocks for {height}x{width} grid'
            )
        _, idx, r0, r1, c0, c1 = min(candidates)
        row_span, col_span = r1 - r0, c1 - c0
        if row_span >= col_span and row_span > 1:
            mid = r0 + row_span // 2
            pair = [(r0, mid, c0, c1), (mid, r1, c0, c1)]
        else:
            mid = c0 + col_span // 2
            pair = [(r0, r1, c0, mid), (r0, r1, mid, c1)]
        blocks[idx:idx + 1] = pair
    return blocks


def _allocate_block_counts(capacities: List[int], total: int) -> List[int]:
    """Deterministically allocate a bounded block budget proportional to grid area."""
    if not capacities or any(capacity < 1 for capacity in capacities):
        raise ValueError(f'[QWEN-BLOCK-QR] invalid capacities: {capacities}')
    total = min(int(total), sum(capacities))
    if total < len(capacities):
        raise ValueError(
            f'[QWEN-BLOCK-QR] coverage={total} cannot cover {len(capacities)} image planes'
        )
    allocation = [1 for _ in capacities]
    while sum(allocation) < total:
        candidates = [
            (capacities[idx] / allocation[idx], capacities[idx], -idx, idx)
            for idx in range(len(capacities))
            if allocation[idx] < capacities[idx]
        ]
        if not candidates:
            break
        allocation[max(candidates)[-1]] += 1
    if sum(allocation) != total:
        raise RuntimeError(
            f'[QWEN-BLOCK-QR] allocation mismatch: {sum(allocation)} != {total}'
        )
    return allocation


def _qwen_block_global_qr_indices(
    image_embeds: torch.Tensor,
    image_grid_thw: torch.Tensor,
    keep: int,
    merge_size: int,
    coverage_ratio: float,
) -> Tuple[torch.LongTensor, Dict[str, Any]]:
    """Qwen-aware hierarchical QR: spatial coverage pivots, then global residual QR."""
    if image_grid_thw is None:
        raise ValueError('[QWEN-BLOCK-QR] image_grid_thw is required')
    grids = [[int(value) for value in row] for row in image_grid_thw.detach().cpu().tolist()]
    planes: List[Tuple[int, int, int]] = []
    offset = 0
    per_image_shapes: List[Tuple[int, int, int]] = []
    for temporal, height, width in grids:
        if height % merge_size or width % merge_size:
            raise ValueError(
                f'[QWEN-BLOCK-QR] grid {temporal}x{height}x{width} '
                f'is not divisible by spatial_merge_size={merge_size}'
            )
        merged_h, merged_w = height // merge_size, width // merge_size
        per_image_shapes.append((temporal, merged_h, merged_w))
        for _ in range(temporal):
            planes.append((offset, merged_h, merged_w))
            offset += merged_h * merged_w
    if offset != int(image_embeds.shape[0]):
        raise RuntimeError(
            f'[QWEN-BLOCK-QR] grid/token mismatch: grid={offset} embeds={image_embeds.shape[0]}'
        )

    keep = min(max(1, int(keep)), offset)
    requested_coverage = max(len(planes), int(round(keep * float(coverage_ratio))))
    coverage_keep = min(keep, requested_coverage)
    block_counts = _allocate_block_counts(
        [height * width for _, height, width in planes], coverage_keep
    )
    token_norms = torch.square(
        image_embeds.detach().reshape(offset, -1).to(torch.float32)
    ).sum(dim=1)
    coverage_indices: List[int] = []
    for (plane_offset, height, width), block_count in zip(planes, block_counts):
        for r0, r1, c0, c1 in _split_spatial_blocks(height, width, block_count):
            rows = torch.arange(r0, r1, device=image_embeds.device)
            cols = torch.arange(c0, c1, device=image_embeds.device)
            positions = (
                plane_offset + rows[:, None] * width + cols[None, :]
            ).flatten()
            pivot = positions[torch.argmax(token_norms[positions])]
            coverage_indices.append(int(pivot.item()))

    selected = _qr_residual_select_exact(
        image_embeds, keep=keep, forced_indices=coverage_indices
    )
    metadata: Dict[str, Any] = {
        'coverage': len(coverage_indices),
        'blocks': block_counts,
        'grids': per_image_shapes,
    }
    return selected, metadata


def _dpp_greedy_select(kernel, similarity, k, device, tag="dpp"):
    """P11: numerically-guarded DPP (determinantal point process) greedy MAP selection.

    Ports the loop body of the Qwen2.5-VL reference `cdpruner_mean`
    (qwen2_5_vl/modeling_qwen2_5_vl.py:996-1006) but hardens it against the underflow that makes
    the raw reference degenerate: `image_embeds` are bf16, so `di2s -= eis**2` in bf16 drives the
    residual negative within ~30 iterations, after which `sqrt(negative) -> NaN`, `argmax(NaN)`
    returns a constant index, and the selection collapses to one token repeated k times. Fixes:
      * accumulate in fp32;
      * clamp `di2s` at 0 and guard `sqrt` with a floor;
      * when the greedy pick is already chosen / non-finite / underflowed, refill by the
        minimum-correlation token (the standard DPP fallback) and WARN once.

    `similarity` is only used for the reference's i==0 seed (`argmin(max(similarity,-1))`) and for
    the correlation refill; `kernel` is the (relevance-weighted) similarity kernel to diagonalise.
    Returns a LongTensor of selected indices (length <= k; == k unless the pool is exhausted).
    """
    eps = 1e-10
    N = kernel.shape[0]
    k = int(min(max(1, k), N))
    kernel = kernel.float()
    similarity_f = similarity.float()
    cis = kernel.new_zeros((k, N))
    di2s = torch.diag(kernel).clone()
    selected_indices = torch.empty(k, dtype=torch.long, device=device)
    chosen = torch.zeros(N, dtype=torch.bool, device=device)
    degenerate_from = None
    n_sel = k
    for i in range(k):
        # Reference seed (REF:1000): diag(similarity)~1.0 so this argmin is bf16 noise, kept verbatim
        # for parity with the Qwen2.5-VL table. Official CDPruner uses argmax(di2s) for all i.
        j = torch.argmin(torch.max(similarity_f, dim=-1).values) if i == 0 else torch.argmax(di2s)
        j_int = int(j)
        if chosen[j_int] or not torch.isfinite(di2s[j_int]) or di2s[j_int] <= eps:
            degenerate_from = i if degenerate_from is None else degenerate_from
            sel = selected_indices[:i]
            max_sim = similarity_f[:, sel].max(dim=1).values if i > 0 else torch.zeros(N, device=device)
            max_sim = torch.nan_to_num(max_sim, nan=1.0, posinf=1.0, neginf=1.0)
            max_sim[chosen] = float('inf')
            j_int = int(torch.argmin(max_sim))
            if chosen[j_int]:
                remaining = (~chosen).nonzero(as_tuple=True)[0]
                if remaining.numel() == 0:
                    n_sel = i
                    break
                j_int = int(remaining[0])
            selected_indices[i] = j_int
            chosen[j_int] = True
            di2s[j_int] = -float('inf')
            cis[i, :] = 0.0
            continue
        selected_indices[i] = j_int
        chosen[j_int] = True
        eis = (kernel[j_int, :] - cis[:i, j_int] @ cis[:i, :]) / torch.sqrt(torch.clamp(di2s[j_int], min=eps))
        eis = torch.nan_to_num(eis, nan=0.0, posinf=0.0, neginf=0.0)
        cis[i, :] = eis
        di2s = torch.clamp(di2s - torch.square(eis), min=0.0)
        di2s[chosen] = -float('inf')
    selected_indices = selected_indices[:n_sel]
    if degenerate_from is not None:
        logger.warning(
            f"[{tag}] DPP loop degenerated at iter {degenerate_from}/{k}; correlation refill used. "
            f"{int(chosen.sum())} distinct tokens selected."
        )
    return selected_indices


class Qwen3VLForConditionalGenerationPruning(_BASE_CONDITIONAL_GEN):
    """
    Wrapper around Qwen3VLForConditionalGeneration that applies visual token pruning.
    Stage 1: vision-side attention-guided token selection (here in forward).
    Stage 2: LLM-side progressive pruning (delegated to pruning model classes).
    """

    _supports_flash_attn_2 = False

    def __init__(self, config, pruning_method='vanilla', visual_token_num=-1):
        super().__init__(config)
        if pruning_method == 'fastv':
            self.model = FastVQwen3VLModel(config, visual_token_num)
        elif pruning_method == 'tops_variant3':
            self.model = TOPSVariant3Qwen3VLModel(config, visual_token_num)
        elif pruning_method in ('mico_coverage_variant', 'qr_block_star_avg'):
            self.model = TOPSVariant3Qwen3VLModel(
                config,
                visual_token_num,
                schedule_variant='llava_aligned',
                text_agg_mode='average_all',
            )
        elif pruning_method == 'sparsevlm':
            # SparseVLM (text-guided, decoder-internal, multi-stage). Unlike VisionZip/HoloV this
            # keeps the FULL visual window through stage 1 and prunes progressively inside the
            # decoder at 3 depths (text-rater cross-attention + cluster-and-merge).
            self.model = SparseVLMQwen3VLModel(config, visual_token_num)
        elif pruning_method == 'apet':
            # ApET (two-stage FPS-basis reconstruction-residual pruning). S1 selects+merges at the
            # ViT side (in the selection block below, keeping 3T/2), S2 prunes 3T/2 -> T/2 inside
            # the decoder at layer L//2. Layer-average visual tokens == T.
            self.model = ApETQwen3VLModel(config, visual_token_num)
        elif pruning_method == 'tops':
            self.model = TOPSQwen3VLModel(config, visual_token_num)
        elif pruning_method in ('duet', 'duet_faithful', 'qr_duet'):
            self.model = DuetQwen3VLModel(config, visual_token_num)
        elif pruning_method in ('dart', 'dart_coverage', 'cover_llm'):
            # DART (EMNLP'25) in-LLM pruning at layer K, ported to Qwen3-VL (DeepStack-aware).
            #   dart          -> pivot(K-state L1) + diversity (anti-cosine) fill
            #   dart_coverage -> pivot + SCOPE coverage fill
            #   cover_llm     -> pure coverage greedy (no pivots), layer via COVER_LLM_LAYER
            self.model = DARTQwen3VLModel(
                config, visual_token_num,
                use_coverage=(pruning_method == 'dart_coverage'),
                pure_cover=(pruning_method == 'cover_llm'),
            )
        elif pruning_method == 'cover_prune_2':
            # COVER_Prune_2: two-stage. S1 (vision-side) = inside-weighted coverage -> 2T;
            # S2 (in-LLM, DeepStack-aware) = all-text-attn relevance + inside-weighted coverage
            # greedy, schedule L14/L24 (avg = target). cover_prune_1_proj (=SCOPE) and
            # cover_prune_2_proj are SINGLE-stage (base model + stock path), handled in Stage-1 only.
            self.model = CoverPrune2Qwen3VLModel(config, visual_token_num)
        elif pruning_method in (
            'mico_coverage_vit_proj_llm_12', 'mico_coverage_vit_proj_llm_6',
            'mico_coverage_vit_proj_llm_8', 'mico_coverage_vit_proj_llm_10',
            'mico_coverage_proj_llm_12',
            'mico_coverage_proj_llm_6', 'mico_coverage_proj_llm_8', 'mico_coverage_proj_llm_10',
            'mico_coverage_proj_llm_11', 'mico_coverage_proj_llm_13',
        ) or _is_mico(pruning_method):
            # New names encode the actual 1-based post-layer cut; the legacy _12 alias stays at L15.
            mico = _is_mico(pruning_method)
            k_layer = None if (not mico and pruning_method.endswith('_llm_12')) else int(pruning_method.rsplit('_', 1)[1])
            self.model = MiCoCoverageVitProjLlm12Qwen3VLModel(
                config, visual_token_num, k_layer=k_layer, method_name=pruning_method, mico=mico
            )
        elif pruning_method in ('mico_coverage_vit_proj', 'mico_coverage_proj', 'mico_s1'):
            # SINGLE-stage twins: S1 prunes straight to T, no in-LLM cut. Kept on the same class so the
            # S1 selector, DeepStack routing and sdpa decode path stay byte-identical to the two-stage
            # arms -- only pruning_layers={} differs. Name carries no K, so none is inferred.
            self.model = MiCoCoverageVitProjLlm12Qwen3VLModel(
                config, visual_token_num, method_name=pruning_method,
                mico=_is_mico_s1(pruning_method), single_stage=True,
            )
        else:
            self.model = _BASE_MODEL_CLS(config)

        self.pruning_method = pruning_method
        self.target_visual_token_num = visual_token_num
        self.visual_token_num = visual_token_num
        self.rope_deltas = None
        self._duet_faithful_stage1_logged = False
        self._qr_duet_stage1_logged = False
        self._qr_block_stage1_logged = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # P7a: hoist the stage-1 state so the re-gated prune block (below) and the decode branch can
        # read them unconditionally. Without this, `selected_masks` / `image_token_positions` are only
        # bound inside `if pixel_values is not None:` inside `if inputs_embeds is None:`, and the moment
        # the prune gate goes live they raise NameError on the first text-only or decode-step call.
        selected_masks = None
        image_token_positions = None      # [N] absolute indices of image tokens in the prompt
        deepstack_image_embeds = None     # list[Tensor], one [N_i, D] per DeepStack level (Qwen3 only)
        visual_pos_masks = None           # bool [B, S'] over the POST-stage-1 sequence
        deepstack_visual_embeds = None    # list[Tensor] aligned to visual_pos_masks

        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.model.visual.dtype)

                # ── Stage 1: hook last vision block attention for importance scoring ──
                if self.pruning_method in [
                    'fastervlm', 'prumerge', 'prumerge+', 'visionzip', 'visionzip+',
                    'vispruner', 'tops_variant3', 'tops', 'tops_2', 'tops_plus',
                    'holov', 'duet', 'duet_faithful',
                    'tsvol_w', 'tsvol_w_anti',
                    'cover_prune_1_proj', 'cover_prune_2', 'cover_prune_2_proj',
                    'scope_vit_origin', 'cover_prune_2_projector_vit', 'scope_raw', 'cover_prune_2_projector_vit_raw',
                    'cover_prune_2_projector_vit_raw_merge_first',
                    'cover_prune_2_projector_merge_first',
                    'mico_coverage_vit_proj_llm_12', 'mico_coverage_vit_proj_llm_6',
                    'mico_coverage_vit_proj_llm_8', 'mico_coverage_vit_proj_llm_10',
                    'mico_coverage_vit_proj',
                    'mico_coverage_proj_llm_12', 'mico_coverage_proj',
                    'mico_coverage_proj_llm_6', 'mico_coverage_proj_llm_8', 'mico_coverage_proj_llm_10',
                    'mico_coverage_proj_llm_11', 'mico_coverage_proj_llm_13',
                ] or _is_mico_s1(self.pruning_method):
                    def forward_with_attn(
                        self_attn,
                        hidden_states: torch.Tensor,
                        cu_seqlens: torch.Tensor,
                        rotary_pos_emb: Optional[torch.Tensor] = None,
                        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                        **kwargs,
                    ) -> torch.Tensor:
                        seq_length = hidden_states.shape[0]
                        q, k, v = (
                            self_attn.qkv(hidden_states)
                            .reshape(seq_length, 3, self_attn.num_heads, -1)
                            .permute(1, 0, 2, 3)
                            .unbind(0)
                        )
                        # q, k, v: (seq_len, num_heads, head_dim)

                        cos, sin = position_embeddings
                        q_rot, k_rot = apply_rotary_pos_emb_vision(q, k, cos, sin)

                        # Save pre-rotation k for VisionZip/PruMerge key-based methods
                        self_attn.q = q.clone()
                        self_attn.k = k.clone()

                        # Run attention per-chunk using standard SDPA
                        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
                        q_chunks = torch.split(q_rot, lengths, dim=0)
                        k_chunks = torch.split(k_rot, lengths, dim=0)
                        v_chunks = torch.split(v, lengths, dim=0)

                        attn_chunk_outputs = []
                        for qi, ki, vi in zip(q_chunks, k_chunks, v_chunks):
                            # (Li, H, D) -> (1, H, Li, D)
                            qi_t = qi.transpose(0, 1).unsqueeze(0)
                            ki_t = ki.transpose(0, 1).unsqueeze(0)
                            vi_t = vi.transpose(0, 1).unsqueeze(0)
                            out_i = torch.nn.functional.scaled_dot_product_attention(
                                qi_t, ki_t, vi_t, attn_mask=None, is_causal=False
                            )
                            attn_chunk_outputs.append(out_i.squeeze(0).transpose(0, 1))

                        attn_output = torch.cat(attn_chunk_outputs, dim=0)
                        attn_output = attn_output.reshape(seq_length, -1)
                        attn_output = self_attn.proj(attn_output)

                        # Compute attention weights for pruning importance scoring.
                        # The number of images is the GROUND TRUTH from cu_seqlens (`lengths` above),
                        # NOT a seq_length heuristic. The old `is_likely_multi_image` guess split a
                        # single LARGE image (e.g. wide ChartQA charts, ~5000-5184 ViT patches) into
                        # fake sub-images, so the captured attention matrix had length actual_tpi > N
                        # and `selected_masks[topk_idx] = True` (visionzip) indexed out of bounds ->
                        # CUDA device-side assert that killed job-zxctxymzz2p6. HoloV was silently
                        # scoring on the same fake split. cu_seqlens delimits per-image segments in
                        # Qwen3-VL vision (no window attention), so len(lengths) is the real count.
                        scaling = 1.0 / math.sqrt(q_rot.shape[-1])
                        if len(lengths) > 1:
                            # One attention block per ACTUAL image segment (real cu_seqlens bounds).
                            all_attn_weights = []
                            offset = 0
                            for Li in lengths:
                                q_img = q_rot[offset:offset + Li].transpose(0, 1)  # (H, Li, D)
                                k_img = k_rot[offset:offset + Li].transpose(0, 1)
                                attn_img = torch.matmul(q_img, k_img.transpose(-2, -1)) * scaling
                                attn_img = torch.nn.functional.softmax(attn_img.float(), dim=-1).to(q_rot.dtype)
                                all_attn_weights.append(attn_img)
                                offset += Li

                            num_heads = all_attn_weights[0].shape[0]
                            max_dim = max(w.shape[-1] for w in all_attn_weights)
                            avg_attn = torch.zeros(num_heads, max_dim, max_dim, device=q_rot.device, dtype=q_rot.dtype)
                            for attn_block in all_attn_weights:
                                s = attn_block.shape[-1]
                                avg_attn[:, :s, :s] += attn_block
                            avg_attn = avg_attn / len(all_attn_weights)
                            self_attn.attn_weights = avg_attn
                            self_attn.attn_blocks = all_attn_weights
                            self_attn.num_images = len(lengths)
                        else:
                            try:
                                q2 = q_rot.transpose(0, 1)  # (H, seq, D)
                                k2 = k_rot.transpose(0, 1)
                                attn_weights = torch.matmul(q2, k2.transpose(-2, -1)) * scaling
                                attn_weights = torch.nn.functional.softmax(attn_weights.float(), dim=-1).to(q2.dtype)
                                self_attn.attn_weights = attn_weights
                            except torch.cuda.OutOfMemoryError:
                                q_probe = q_rot.mean(dim=0, keepdim=True).transpose(0, 1)  # (H, 1, D)
                                k2 = k_rot.transpose(0, 1)
                                attn_scores = torch.matmul(q_probe, k2.transpose(-2, -1)) * scaling
                                attn_weights = torch.nn.functional.softmax(attn_scores.float(), dim=-1).to(q_probe.dtype)
                                self_attn.attn_weights = attn_weights.expand(-1, seq_length, -1)

                        return attn_output

                    self.model.visual.blocks[-1].attn.forward = types.MethodType(
                        forward_with_attn, self.model.visual.blocks[-1].attn
                    )

                # ViT-space coverage variants (scope_vit_origin / cover_prune_2_projector_vit): capture
                # the patch-merger INPUT (pre-merge ViT features, [4N, vit_dim]) via a forward_pre_hook.
                # The merger does `view(-1, hidden*4)`, i.e. consecutive-4 rows -> 1 merged token, so a
                # reshape(N,4,-1).mean(1) recovers a per-merged-token ViT feature. Qwen3-VL has no window
                # attention, so (unlike Qwen2.5-VL) NO get_window_index reorder is needed.
                if (self.pruning_method in ('scope_vit_origin', 'cover_prune_2_projector_vit',
                                           'mmtok', 'mmtok_origin', 'scope_raw', 'cover_prune_2_projector_vit_raw',
                                           'cover_prune_2_projector_vit_raw_merge_first',
                                           'mico_coverage_vit_proj_llm_12', 'mico_coverage_vit_proj_llm_6',
                                           'mico_coverage_vit_proj_llm_8', 'mico_coverage_vit_proj_llm_10',
                                           'mico_coverage_vit_proj') or _is_mico_s1(self.pruning_method)) \
                        and not getattr(self, '_mmtok_hook_set', False):
                    def _cap_merger_in(mod, inp):
                        self._mmtok_merger_in = inp[0]
                    self.model.visual.merger.register_forward_pre_hook(_cap_merger_in)
                    self._mmtok_hook_set = True

                image_embeds_out = self.model.visual(pixel_values, grid_thw=image_grid_thw)
                # Qwen3-VL returns BaseModelOutputWithDeepstackFeatures; extract merged tokens
                if hasattr(image_embeds_out, 'pooler_output') and image_embeds_out.pooler_output is not None:
                    image_embeds = image_embeds_out.pooler_output
                elif hasattr(image_embeds_out, 'last_hidden_state'):
                    image_embeds = image_embeds_out.last_hidden_state
                else:
                    image_embeds = image_embeds_out

                # P7b: capture the DeepStack multi-scale features (UP:815-818). Dropping them makes
                # EVERY pruned run "Qwen3-VL without DeepStack", not comparable to the vanilla branch
                # (MP:157-161) which keeps them. Rows are 1:1 with the merged image tokens above.
                deepstack_image_embeds = getattr(image_embeds_out, 'deepstack_features', None)
                if deepstack_image_embeds is not None and len(deepstack_image_embeds) == 0:
                    deepstack_image_embeds = None

                # ── Extract attention info after vision encoder ──
                attn_last = self.model.visual.blocks[-1].attn

                if self.pruning_method in ['tops', 'tops_2', 'tops_plus', 'tops_variant3', 'holov', 'duet', 'duet_faithful',
                                           'tsvol_w', 'tsvol_w_anti',
                                           'cover_prune_1_proj', 'cover_prune_2', 'cover_prune_2_proj',
                                           'scope_vit_origin', 'cover_prune_2_projector_vit', 'scope_raw', 'cover_prune_2_projector_vit_raw',
                                           'cover_prune_2_projector_vit_raw_merge_first',
                                           'cover_prune_2_projector_merge_first',
                                           'mico_coverage_vit_proj_llm_12', 'mico_coverage_vit_proj_llm_6',
                                           'mico_coverage_vit_proj_llm_8', 'mico_coverage_vit_proj_llm_10',
                                           'mico_coverage_vit_proj',
                                           'mico_coverage_proj_llm_12', 'mico_coverage_proj',
                                           'mico_coverage_proj_llm_6', 'mico_coverage_proj_llm_8', 'mico_coverage_proj_llm_10',
                                           'mico_coverage_proj_llm_11', 'mico_coverage_proj_llm_13'] or _is_mico_s1(self.pruning_method):
                    has_window_index = hasattr(self.model.visual, 'get_window_index')
                    if hasattr(attn_last, 'num_images') and attn_last.num_images > 1:
                        all_patch_scores = []
                        for attn_block in attn_last.attn_blocks:
                            all_patch_scores.append(attn_block.mean(dim=0).mean(dim=0))
                        all_patch_scores = torch.cat(all_patch_scores)

                        token_scores = []
                        offset = 0
                        for thw in image_grid_thw:
                            _, H, W = thw[0].item(), thw[1].item(), thw[2].item()
                            t = H * W
                            img_scores = all_patch_scores[offset:offset + t]
                            img_token_scores = img_scores.view(-1, 4).mean(dim=1)
                            if has_window_index:
                                win_idx, _ = self.model.visual.get_window_index(thw.unsqueeze(0))
                                rev_idx = torch.argsort(win_idx)
                                img_token_scores = img_token_scores[rev_idx]
                            token_scores.append(img_token_scores)
                            offset += t
                        image_attns = torch.cat(token_scores)
                    else:
                        seq_length = pixel_values.shape[0]
                        attn_weights = attn_last.attn_weights
                        if has_window_index:
                            window_index, _ = self.model.visual.get_window_index(image_grid_thw)
                            reverse_indices = torch.argsort(window_index)
                            image_attns = attn_weights.view(-1, seq_length // 4, 4, seq_length // 4, 4).mean(dim=(2, 4))
                            image_attns = image_attns[:, reverse_indices, :][:, :, reverse_indices]
                        else:
                            # Qwen3-VL: no windowed attention, patches already in spatial order
                            image_attns = attn_weights.view(-1, seq_length // 4, 4, seq_length // 4, 4).mean(dim=(2, 4))

                elif self.pruning_method in ['prumerge', 'prumerge+', 'visionzip', 'visionzip+']:
                    attn_keys = attn_last.k
                    attn_weights = attn_last.attn_weights
                    has_window_index = hasattr(self.model.visual, 'get_window_index')
                    if hasattr(attn_last, 'num_images') and attn_last.num_images > 1:
                        avg_seq_length = attn_weights.shape[-1]
                        image_attns = attn_weights.view(-1, avg_seq_length // 4, 4, avg_seq_length // 4, 4).mean(dim=(2, 4)) \
                            if avg_seq_length % 4 == 0 else attn_weights
                        image_keys = attn_keys[:avg_seq_length:4, :, :] if attn_keys.shape[0] > avg_seq_length \
                            else attn_keys[::4, :, :]
                    else:
                        seq_length = pixel_values.shape[0]
                        image_attns = attn_weights.view(-1, seq_length // 4, 4, seq_length // 4, 4).mean(dim=(2, 4))
                        image_keys = attn_keys.view(seq_length // 4, 4, attn_keys.shape[1], attn_keys.shape[2]).mean(dim=1)
                        if has_window_index:
                            window_index, _ = self.model.visual.get_window_index(image_grid_thw)
                            reverse_indices = torch.argsort(window_index)
                            image_attns = image_attns[:, reverse_indices, :][:, :, reverse_indices]
                            image_keys = image_keys[reverse_indices, :, :]

                elif self.pruning_method in ['fastervlm', 'vispruner']:
                    attn_weights = attn_last.attn_weights
                    seq_length = pixel_values.shape[0]
                    image_attns = attn_weights.view(-1, seq_length // 4, 4, seq_length // 4, 4).mean(dim=(2, 4))
                    if hasattr(self.model.visual, 'get_window_index'):
                        window_index, _ = self.model.visual.get_window_index(image_grid_thw)
                        reverse_indices = torch.argsort(window_index)
                        image_attns = image_attns[:, reverse_indices, :][:, :, reverse_indices]

                elif self.pruning_method in ['trim_mean', 'trim_max', 'cdpruner_mean', 'cdpruner_max']:
                    image_token_indices = torch.where(input_ids == self.config.image_token_id)[1]

                # ── Generate selection mask ──
                N, D = image_embeds.shape
                device = image_embeds.device
                selected_masks = torch.ones(N, dtype=torch.bool, device=device)

                if self.pruning_method == 'sparsevlm':
                    # SparseVLM stage 1 is a DELIBERATE PASS-THROUGH: it keeps all N visual tokens
                    # here and lets SparseVLMQwen3VLModel prune progressively inside the decoder
                    # (3 depths, text-rater + cluster-and-merge). Left implicit this would read as
                    # the P11 "unmatched method ran UNPRUNED" bug, so it is spelled out.
                    selected_masks = torch.ones(N, dtype=torch.bool, device=device)

                elif self.pruning_method == 'apet':
                    # This suite's existing ApET S1 acts on post-merger image embeddings.
                    # Keep 3T/2 tokens by FPS-basis reconstruction residual,
                    # cosine-merge the discarded remainder into the kept non-basis tokens. The merged
                    # FP features overwrite image_embeds at the kept rows; the kept indices become the
                    # stage-1 keep mask (compaction to 3T/2). Decoder S2 later prunes 3T/2 -> T/2.
                    stage1, _stage2, _pa = apet_budget(self.target_visual_token_num, len(self.model.layers))
                    stage1 = max(1, min(stage1, N))
                    if stage1 >= N:
                        # image already at/below the S1 budget -> pass through (S2 still runs).
                        selected_masks = torch.ones(N, dtype=torch.bool, device=device)
                    else:
                        merged_full, kept_idx = apet_stage1(image_embeds, stage1, self.model.basis)
                        image_embeds[kept_idx] = merged_full.index_select(0, kept_idx).to(image_embeds.dtype)
                        selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                        selected_masks[kept_idx] = True
                    print(f"[ApET Q3 S1] N={N} keep={stage1} "
                          f"space=post-merger-llm-adaptation precision=fp32-merge basis={self.model.basis}",
                          flush=True)

                elif self.pruning_method in ('random', 'rand'):
                    random_scores = torch.rand(N, device=device)
                    selected_indices = random_scores.topk(self.target_visual_token_num).indices
                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[selected_indices] = True

                elif self.pruning_method == 'fastervlm':
                    scores = image_attns.mean(dim=1).mean(dim=0)
                    selected_indices = scores.topk(self.target_visual_token_num).indices
                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[selected_indices] = True

                elif self.pruning_method == 'vispruner':
                    # VisPruner: attention-important top-k + iterative diverse bipartite merge.
                    # important_ratio=0.5 is the AUTHOR DEFAULT, matching the curated reference
                    # /path/to/local-assets
                    # vispruner.py:29,69 (Theia-4869/VisPruner). image_attns is (H,N,N) and
                    # image_embeds is [N,D], so the CLS-attn reduction + index bookkeeping carry over.
                    # Selection-only (no embed merge), so it rides the P7 stage-1 compaction path.
                    important_token_num = int(self.target_visual_token_num * 0.5)
                    diverse_token_num = self.target_visual_token_num - important_token_num

                    image_attns = image_attns.mean(dim=1).mean(dim=0)
                    token_indices = image_attns.argsort(dim=-1, descending=True)

                    important_indices = token_indices[:important_token_num]
                    residual_indices = token_indices[important_token_num:]

                    image_normalized = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                    while diverse_token_num > 0:
                        residual_embeds = image_normalized[residual_indices]
                        r = min(8, residual_embeds.shape[0] - diverse_token_num)
                        if r <= 0:
                            break

                        a, b = residual_embeds[::2], residual_embeds[1::2]
                        scores = a @ b.t()
                        scores = scores.max(dim=-1).values

                        distinct_indices = scores.argsort(dim=-1, descending=True)[r:]
                        residual_indices = torch.cat([residual_indices[::2][distinct_indices], residual_indices[1::2]])

                    selected_indices = torch.cat([important_indices, residual_indices])
                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[selected_indices] = True

                elif self.pruning_method in ['prumerge', 'prumerge+']:
                    image_keys = image_keys.mean(dim=1)
                    image_attns = image_attns.mean(dim=1).mean(dim=0)
                    selected_indices = image_attns.topk(
                        self.target_visual_token_num if self.pruning_method == 'prumerge'
                        else self.target_visual_token_num // 2
                    ).indices
                    if self.pruning_method == 'prumerge+':
                        arithmetic_step = int(N / self.target_visual_token_num * 2)
                        arithmetic_indices = torch.linspace(
                            arithmetic_step // 2, N - 1 - arithmetic_step // 2,
                            self.target_visual_token_num // 2, dtype=torch.long, device=device
                        )
                        image_attns[arithmetic_indices] = 0
                        selected_indices = torch.cat([selected_indices, arithmetic_indices])

                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[selected_indices] = True
                    pruned_indices = selected_masks.float().topk(N - self.target_visual_token_num, largest=False).indices

                    selected_embeds = image_embeds[selected_indices]
                    selected_keys = image_keys[selected_indices]
                    selected_attns = image_attns[selected_indices]
                    pruned_embeds = image_embeds[pruned_indices]
                    pruned_keys = image_keys[pruned_indices]
                    pruned_attns = image_attns[pruned_indices]

                    selected_keys_norm = selected_keys / selected_keys.norm(dim=-1, keepdim=True)
                    pruned_keys_norm = pruned_keys / pruned_keys.norm(dim=-1, keepdim=True)

                    updated_embeds = torch.zeros_like(selected_embeds)
                    for i in range(updated_embeds.shape[0]):
                        center_key_norm = selected_keys_norm[i]
                        other_keys_norm = torch.cat([
                            selected_keys_norm[:i], selected_keys_norm[i+1:], pruned_keys_norm
                        ])
                        similarity = torch.matmul(center_key_norm, other_keys_norm.t())
                        cluster_indices = similarity.topk(32).indices
                        other_embeds = torch.cat([selected_embeds[:i], selected_embeds[i+1:], pruned_embeds])
                        other_attns = torch.cat([selected_attns[:i], selected_attns[i+1:], pruned_attns])
                        cluster_embeds = other_embeds[cluster_indices]
                        cluster_weights = other_attns[cluster_indices].unsqueeze(-1)
                        updated_embeds[i] = selected_embeds[i] + torch.sum(cluster_embeds * cluster_weights, dim=0)
                    image_embeds[selected_indices] = updated_embeds

                elif self.pruning_method in ['visionzip', 'visionzip+']:
                    if self.pruning_method == 'visionzip':
                        dominant_token_num = int(self.target_visual_token_num * 27 / 32)
                    else:
                        dominant_token_num = self.target_visual_token_num - int(N * 5 / 100)
                    contextual_token_num = self.target_visual_token_num - dominant_token_num
                    dominant_token_num = max(1, min(dominant_token_num, N))
                    contextual_token_num = max(0, min(contextual_token_num, N - dominant_token_num))

                    image_attns = image_attns.mean(dim=1).mean(dim=0)
                    dominant_indices = image_attns.topk(dominant_token_num).indices
                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[dominant_indices] = True

                    if contextual_token_num > 0:
                        filtered_embeds = image_embeds[~selected_masks]
                        if filtered_embeds.shape[0] > 0:
                            normalized_embeds = filtered_embeds / filtered_embeds.norm(dim=-1, keepdim=True)
                            contextual_token_num = min(contextual_token_num, normalized_embeds.shape[0])
                            if contextual_token_num > 0:
                                target_step = max(1, normalized_embeds.shape[0] // contextual_token_num)
                                target_indices = torch.arange(0, normalized_embeds.shape[0], target_step, device=device)[:contextual_token_num]
                                target_embeds = normalized_embeds[target_indices]
                                source_mask = ~torch.isin(torch.arange(normalized_embeds.shape[0], device=device), target_indices)
                                source_embeds = normalized_embeds[source_mask]
                                if source_embeds.shape[0] > 0:
                                    similarity = torch.matmul(source_embeds, target_embeds.t())
                                    assign_one_hot = image_embeds.new_zeros(source_embeds.shape[0], contextual_token_num)
                                    assign_one_hot.scatter_(1, similarity.argmax(dim=-1).unsqueeze(-1), 1)
                                    counts = assign_one_hot.sum(dim=0).clamp(min=1).unsqueeze(-1)
                                    contextual_positions = (~selected_masks).nonzero(as_tuple=True)[0]
                                    target_positions = contextual_positions[target_indices]
                                    source_positions = contextual_positions[source_mask]
                                    selected_masks[target_positions] = True
                                    aggregated_embeds = torch.matmul(assign_one_hot.t(), source_embeds) / counts
                                    image_embeds[target_positions] += aggregated_embeds

                elif self.pruning_method == 'qr_block_star_avg':
                    # Pure selection only. Qwen's post-merger token sequence is restored to
                    # its native-aspect 2-D grid. A small coverage budget chooses one QR pivot
                    # per adaptive rectangular block; global residual QR, conditioned on those
                    # pivots, freely fills the remaining Stage-1 budget. No embeddings are merged.
                    _qr_mult = float(os.environ.get('STAGE1_MULT', '2'))
                    _coverage_ratio = float(
                        os.environ.get('STAGE1_BLOCK_COVERAGE_RATIO', '0.25')
                    )
                    if not (0.0 < _coverage_ratio <= 1.0):
                        raise ValueError(
                            f'[QWEN-BLOCK-QR] invalid coverage ratio: {_coverage_ratio}'
                        )
                    stage1_keep_num = min(
                        max(1, int(self.target_visual_token_num * _qr_mult)), N
                    )
                    merge_size = int(
                        getattr(self.config.vision_config, 'spatial_merge_size', 2)
                    )
                    selected_indices, block_meta = _qwen_block_global_qr_indices(
                        image_embeds=image_embeds,
                        image_grid_thw=image_grid_thw,
                        keep=stage1_keep_num,
                        merge_size=merge_size,
                        coverage_ratio=_coverage_ratio,
                    )
                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[selected_indices] = True
                    if os.environ.get('ENABLE_DEBUG', '0') == '1' and not self._qr_block_stage1_logged:
                        print(
                            f'[QWEN-BLOCK-QR Stage1] selected={selected_indices.numel()}/{N} '
                            f'keep={stage1_keep_num} target={self.target_visual_token_num} '
                            f'mult={_qr_mult:g} coverage={block_meta["coverage"]} '
                            f'blocks={block_meta["blocks"]} grids={block_meta["grids"]} '
                            f'merge=off residual=global',
                            flush=True,
                        )
                        self._qr_block_stage1_logged = True

                elif self.pruning_method in ('qr_duet', 'mico_coverage_variant'):
                    # Query-agnostic residual-QR Stage 1 shared by two controlled
                    # Stage-2 experiments. mico_coverage_variant keeps TOPSPrototype-Pro's own decoder
                    # and uses the LLaVA-aligned average-all-text variant.
                    # This is pure selection: no contextual token aggregation and no
                    # dense vision-attention matrix is materialized.
                    import os as _qr_os
                    _qr_mult = float(_qr_os.environ.get('STAGE1_MULT', '2'))
                    stage1_keep_num = min(
                        max(1, int(self.target_visual_token_num * _qr_mult)), N
                    )
                    _qr_flat = image_embeds.detach().reshape(N, -1).to(torch.float32)
                    _qr_init = (_qr_flat ** 2).sum(dim=1)
                    if not torch.isfinite(_qr_init).all():
                        raise ValueError('[QR-STAGE1] non-finite token norms')
                    _qr_res = _qr_flat.clone()
                    _qr_avail = torch.ones(N, dtype=torch.bool, device=device)
                    _qr_sel: List[int] = []
                    _qr_eps = 1e-8
                    while len(_qr_sel) < stage1_keep_num:
                        _qr_rn = (_qr_res ** 2).sum(dim=1)
                        _qr_rn = torch.where(
                            _qr_avail, _qr_rn, torch.full_like(_qr_rn, -1.0)
                        )
                        _qr_j = int(torch.argmax(_qr_rn).item())
                        if _qr_rn[_qr_j] <= _qr_eps:
                            _qr_rem = torch.nonzero(_qr_avail).flatten()
                            _qr_rem = _qr_rem[
                                torch.argsort(_qr_init[_qr_rem], descending=True, stable=True)
                            ]
                            _qr_sel.extend(
                                int(i)
                                for i in _qr_rem[
                                    :stage1_keep_num - len(_qr_sel)
                                ].tolist()
                            )
                            break
                        _qr_sel.append(_qr_j)
                        _qr_avail[_qr_j] = False
                        _qr_basis = _qr_res[_qr_j] / (_qr_res[_qr_j].norm() + _qr_eps)
                        _qr_res = _qr_res - torch.outer(_qr_res @ _qr_basis, _qr_basis)
                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[torch.tensor(_qr_sel, device=device, dtype=torch.long)] = True
                    if _qr_os.environ.get('ENABLE_DEBUG', '0') == '1' and not self._qr_duet_stage1_logged:
                        _qr_tag = 'QR-TOPSPrototype-AVG' if self.pruning_method == 'mico_coverage_variant' else 'QR-DUET'
                        print(f'[{_qr_tag} Stage1] selected={len(_qr_sel)}/{N} '
                              f'keep={stage1_keep_num} target={self.target_visual_token_num} '
                              f'mult={_qr_mult:g}', flush=True)
                        self._qr_duet_stage1_logged = True

                elif self.pruning_method == 'tops_variant3':
                    # ---- TOPSPrototype-Pro-QR Stage 1 (STAGE1_SCORER=qr): query-agnostic residual QR ----
                    # Ported from InternVL3 TOPSPrototype-Pro-QR. Iteratively pick the max residual-norm
                    # visual token, then Gram-Schmidt project it out. No text participates; result
                    # is deterministic and prompt-independent. Stage-2 (in-decoder attention) is
                    # unchanged. Default path (env unset) is the original text-relevance greedy.
                    import os as _qr_os
                    if _qr_os.environ.get('STAGE1_SCORER', '').strip().lower() == 'qr':
                        _qr_mult = float(_qr_os.environ.get('STAGE1_MULT', '2'))
                        stage1_keep_num = min(max(1, int(self.target_visual_token_num * _qr_mult)), N)
                        _qr_flat = image_embeds.detach().reshape(N, -1).to(torch.float32)
                        _qr_init = (_qr_flat ** 2).sum(dim=1)
                        if not torch.isfinite(_qr_init).all():
                            raise ValueError('[TOPSPrototype-QWEN3-QR] non-finite token norms')
                        _qr_res = _qr_flat.clone()
                        _qr_avail = torch.ones(N, dtype=torch.bool, device=device)
                        _qr_sel: List[int] = []
                        _qr_eps = 1e-8
                        while len(_qr_sel) < stage1_keep_num:
                            _qr_rn = (_qr_res ** 2).sum(dim=1)
                            _qr_rn = torch.where(_qr_avail, _qr_rn, torch.full_like(_qr_rn, -1.0))
                            _qr_j = int(torch.argmax(_qr_rn).item())
                            if _qr_rn[_qr_j] <= _qr_eps:
                                _qr_rem = torch.nonzero(_qr_avail).flatten()
                                _qr_rem = _qr_rem[torch.argsort(_qr_init[_qr_rem], descending=True, stable=True)]
                                _qr_sel.extend(int(i) for i in _qr_rem[:stage1_keep_num - len(_qr_sel)].tolist())
                                break
                            _qr_sel.append(_qr_j)
                            _qr_avail[_qr_j] = False
                            _qr_basis = _qr_res[_qr_j] / (_qr_res[_qr_j].norm() + _qr_eps)
                            _qr_res = _qr_res - torch.outer(_qr_res @ _qr_basis, _qr_basis)
                        _qr_t = torch.tensor(_qr_sel, device=device, dtype=torch.long)
                        selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                        selected_masks[_qr_t] = True
                        if _qr_os.environ.get('ENABLE_DEBUG', '0') == '1':
                            print(f'[TOPSPrototype-QWEN3-QR S1] scorer=qr selected={len(_qr_sel)}/{N} '
                                  f'keep={stage1_keep_num} target={self.target_visual_token_num}', flush=True)
                    else:
                        # TOPS-Variant3 Stage 1: THCP — relevance + diversity greedy selection
                        image_token_indices_v = (input_ids == self.config.image_token_id).nonzero(as_tuple=True)[1]
                        if len(image_token_indices_v) > 0:
                            text_start_idx = image_token_indices_v[-1].item() + 1
                            text_end_idx = input_ids.shape[1] - 5
                            if text_start_idx < text_end_idx:
                                text_embeds = inputs_embeds.squeeze(0)[text_start_idx:text_end_idx]
                            else:
                                text_embeds = torch.tensor([], device=device)
                        else:
                            text_embeds = torch.tensor([], device=device)

                        M = text_embeds.shape[0]
                        stage1_keep_num = min(max(1, self.target_visual_token_num * 2), N)

                        if M == 0:
                            if image_attns.dim() == 3:
                                scores = image_attns.mean(dim=1).mean(dim=0)
                            elif image_attns.dim() == 2:
                                scores = image_attns.mean(dim=0)
                            else:
                                scores = image_attns
                            selected_indices = scores.topk(stage1_keep_num).indices
                            selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                            selected_masks[selected_indices] = True
                        else:
                            image_normalized = image_embeds / (image_embeds.norm(dim=-1, keepdim=True) + 1e-8)
                            text_normalized = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-8)
                            response_matrix = torch.matmul(image_normalized, text_normalized.t())
                            text_relevance = response_matrix.mean(dim=-1) if M > 1 else response_matrix.squeeze(-1)
                            text_relevance = (text_relevance - text_relevance.min()) / (
                                text_relevance.max() - text_relevance.min() + 1e-6
                            )
                            visual_similarity = torch.matmul(image_normalized, image_normalized.t())
                            available_mask = torch.ones(N, dtype=torch.bool, device=device)
                            selected_indices_list: List[int] = []
                            for step in range(min(stage1_keep_num, N)):
                                if not available_mask.any():
                                    break
                                scores = text_relevance.clone()
                                if step > 0:
                                    sel_t = torch.tensor(selected_indices_list, device=device, dtype=torch.long)
                                    max_sim = visual_similarity[:, sel_t].max(dim=1).values
                                    scores = 0.5 * scores + 0.5 * (1 - max_sim)
                                scores[~available_mask] = -float('inf')
                                selected_idx = scores.argmax().item()
                                selected_indices_list.append(selected_idx)
                                available_mask[selected_idx] = False
                            if len(selected_indices_list) == 0:
                                selected_indices_list = list(range(stage1_keep_num))
                            sel_tensor = torch.tensor(selected_indices_list, device=device, dtype=torch.long)
                            selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                            selected_masks[sel_tensor] = True

                elif self.pruning_method == 'tops':
                    # TOPS Stage 1: Vision-side RDC greedy selection
                    a1 = float(os.environ.get("A1", "1.0"))
                    alpha_1 = float(os.environ.get("ALPHA_1", "1.0"))
                    lambda_1 = float(os.environ.get("LAMBDA_1", "1.0"))
                    coverage_method = os.environ.get("COVERAGE", "MEAN")

                    stage1_keep_num = min(max(1, self.target_visual_token_num * 2), N)

                    print(f"\n{'='*80}")
                    print(f"[TOPS Stage 1] Qwen3-VL Vision-Side RDC Greedy Selection")
                    print(f"  Visual tokens: {N}, Stage1 target: {stage1_keep_num} (target×2)")
                    print(f"  a1={a1}, alpha_1={alpha_1}, lambda_1={lambda_1}")

                    # Relevance: CLS attention from vision encoder
                    if image_attns is not None:
                        if image_attns.dim() == 3:
                            cls_attn = image_attns.mean(dim=1).mean(dim=0)
                        elif image_attns.dim() == 2:
                            cls_attn = image_attns.mean(dim=0)
                        else:
                            cls_attn = image_attns
                    else:
                        cls_attn = image_embeds.norm(dim=-1)

                    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

                    feat_n = image_embeds / (image_embeds.norm(dim=-1, keepdim=True) + 1e-8)
                    sim = torch.matmul(feat_n, feat_n.t())

                    avail = torch.ones(N, dtype=torch.bool, device=device)
                    selected = []
                    i0 = torch.argmax(cls_attn).item()
                    selected.append(i0)
                    avail[i0] = False
                    max_sim_to_S = sim[i0].clone()
                    max_sim_to_S[i0] = 1.0

                    for _ in range(1, stage1_keep_num):
                        if not avail.any():
                            break
                        diversity = 1.0 - max_sim_to_S
                        if coverage_method == "SCOPE":
                            coverage = torch.clamp(sim - max_sim_to_S.unsqueeze(0), min=0.0).sum(dim=1)
                            coverage[~avail] = 0.0
                        else:
                            sim_to_avail = sim[:, avail].sum(dim=1)
                            n_avail = avail.sum().item()
                            coverage = (sim_to_avail - 1.0) / (n_avail - 1) if n_avail > 1 else torch.zeros(N, device=device)

                        a_mean = cls_attn.mean().clamp(min=1e-8)
                        div_mean = diversity.mean().clamp(min=1e-8)
                        cov_mean = coverage.mean().clamp(min=1e-8)
                        scores = a1 * (cls_attn / a_mean) + alpha_1 * (diversity / div_mean) + lambda_1 * (coverage / cov_mean)
                        scores[~avail] = -float("inf")
                        idx = torch.argmax(scores).item()
                        selected.append(idx)
                        avail[idx] = False
                        max_sim_to_S = torch.maximum(max_sim_to_S, sim[idx])
                        max_sim_to_S[idx] = 1.0

                    selected_tensor = torch.tensor(selected, device=device, dtype=torch.long)
                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[selected_tensor] = True

                    print(f"  Selected {selected_masks.sum().item()} tokens")
                    print(f"{'='*80}\n")

                # ── P11: selection branches ported from qwen2_5_vl/modeling_qwen2_5_vl.py ─────────
                # Before this port, divprune / dppruner / trim_* / cdpruner_* had NO selection branch
                # in this module: control fell through to `self.visual_token_len = ...` with
                # `selected_masks` still all-ones, so those methods silently pruned NOTHING (the
                # `selected_masks.sum() < selected_masks.numel()` gate below is False). `image_token_indices`
                # was extracted at W:299 (now above) for the trim_*/cdpruner_* family.
                elif self.pruning_method == 'divprune':
                    # DivPrune: diversity-based greedy selection (REF:913-943; no text, pure visual).
                    image_normalized = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                    cosine_matrix = 1.0 - torch.matmul(image_normalized, image_normalized.t())  # [N, N]
                    k = int(min(max(1, self.target_visual_token_num), N))
                    selected_indices = torch.empty(k, dtype=torch.long, device=device)
                    for i in range(k):
                        if i == 0:
                            scores = torch.topk(cosine_matrix, 2, dim=0, largest=False).values[1]  # [N]
                        else:
                            m2 = cosine_matrix[selected_indices[:i], :]  # [i, N]
                            scores = torch.min(m2, dim=0).values  # [N]
                            scores[selected_indices[:i]] = -float('inf')
                        selected_indices[i] = torch.argmax(scores)
                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[selected_indices] = True

                elif self.pruning_method == 'dppruner':
                    # DPP MAP on the raw visual similarity kernel (REF:944-964), guarded loop.
                    image_normalized = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                    similarity = torch.matmul(image_normalized, image_normalized.t())
                    selected_indices = _dpp_greedy_select(
                        similarity, similarity, self.target_visual_token_num, device, tag='dppruner'
                    )
                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[selected_indices] = True

                elif self.pruning_method in ['trim_mean', 'trim_max']:
                    # TRIM: image-text relevance top-k (REF:965-984). The +2/-5 text slice offsets are
                    # verified correct for Qwen3-VL (spec §0); guard the empty slice the reference lacks.
                    image_normalized = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                    if image_token_indices.numel() > 0:
                        text_embeds = inputs_embeds.squeeze(0)[image_token_indices[-1] + 2:-5]
                    else:
                        text_embeds = inputs_embeds.new_zeros((0, D))
                    k = int(min(max(1, self.target_visual_token_num), N))
                    if text_embeds.shape[0] == 0:
                        # No text to score against: fall back to uniform stride (deterministic).
                        selected_indices = torch.linspace(0, N - 1, k, device=device).round().long().unique()
                    else:
                        text_normalized = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                        sim = torch.matmul(image_normalized, text_normalized.t())
                        similarity = sim.mean(dim=-1) if self.pruning_method == 'trim_mean' else sim.max(dim=-1).values
                        selected_indices = similarity.topk(k, dim=-1).indices
                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[selected_indices] = True

                elif self.pruning_method in ['cdpruner_mean', 'cdpruner_max']:
                    # CDPruner: relevance-conditioned DPP (REF:985-1034). BOTH variants use the SAME
                    # (correct) mean-variant loop body -- REF:1030-1031 is a real indentation bug in
                    # `cdpruner_max` (its di2s/di2s[j] updates sit OUTSIDE the loop, selecting only 2
                    # tokens); the variants differ ONLY in mean-vs-max relevance, which is applied here.
                    image_normalized = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                    similarity = torch.matmul(image_normalized, image_normalized.t())
                    if image_token_indices.numel() > 0:
                        text_embeds = inputs_embeds.squeeze(0)[image_token_indices[-1] + 2:-5]
                    else:
                        text_embeds = inputs_embeds.new_zeros((0, D))
                    if text_embeds.shape[0] == 0:
                        # No text: uniform relevance (the reference would divide by a zero range).
                        relevance = torch.ones(N, device=device, dtype=similarity.dtype)
                    else:
                        text_normalized = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                        rel = torch.matmul(image_normalized, text_normalized.t())
                        relevance = rel.mean(dim=-1) if self.pruning_method == 'cdpruner_mean' else rel.max(dim=-1).values
                        relevance = (relevance - relevance.min() + 0.01) / (relevance.max() - relevance.min() + 1e-6)
                    kernel = relevance.unsqueeze(1) * similarity * relevance.unsqueeze(0)
                    selected_indices = _dpp_greedy_select(
                        kernel, similarity, self.target_visual_token_num, device, tag=self.pruning_method
                    )
                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[selected_indices] = True

                elif self.pruning_method == 'holov':
                    # HoloV (NeurIPS25): region-balanced holistic selection. Faithful port of the
                    # Qwen-HoloV signal used in qwen2_5_vl/modeling_qwen2_5_vl.py:582-605 -- last-ViT
                    # received attention + heterogeneous spatial bands. Pure selection (no merge);
                    # keeps ~target tokens, stock Qwen3-VL decoder + DeepStack.
                    from .holov_utils import HoloV as _HoloV
                    if image_attns.dim() == 3:
                        holov_attn = image_attns.mean(dim=1).mean(dim=0)
                    elif image_attns.dim() == 2:
                        holov_attn = image_attns.mean(dim=0)
                    else:
                        holov_attn = image_attns
                    keep_num = int(min(max(1, self.target_visual_token_num), N))
                    num_patches = max(1, int((1024.0 / 576.0) * N / max(1, keep_num)))
                    _, final_position = _HoloV(
                        image_embeds.unsqueeze(0), holov_attn, num_patches, keep_num
                    )
                    final_position = final_position.reshape(-1).to(device).long()
                    final_position = final_position[final_position >= 0]
                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[final_position] = True

                elif self.pruning_method == 'duet':
                    # DUET-VLM (CVPR2026) Stage-1 = VisionZip V2V: dominant top-k + contextual
                    # cluster-merge. Budget mirrors the repo's Qwen2.5 duet_official (model.py:376):
                    # dominant=round(_t/128*170), contextual=round(_t/128*35). Stage-2 PyramidDrop
                    # runs in DuetQwen3VLModel and brings the LAYER-AVERAGE down to _t.
                    _t = int(self.target_visual_token_num)
                    dominant_token_num = max(1, int(round(_t / 128.0 * 170)))
                    contextual_token_num = max(0, int(round(_t / 128.0 * 35)))
                    dominant_token_num = min(dominant_token_num, N)
                    contextual_token_num = min(contextual_token_num, N - dominant_token_num)

                    if image_attns.dim() == 3:
                        duet_attn = image_attns.mean(dim=1).mean(dim=0)
                    elif image_attns.dim() == 2:
                        duet_attn = image_attns.mean(dim=0)
                    else:
                        duet_attn = image_attns
                    dominant_indices = duet_attn.topk(dominant_token_num).indices
                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[dominant_indices] = True

                    if contextual_token_num > 0:
                        remaining_positions = (~selected_masks).nonzero(as_tuple=True)[0]
                        if remaining_positions.numel() > 0:
                            filtered_embeds = image_embeds[remaining_positions]
                            normalized_embeds = filtered_embeds / (filtered_embeds.norm(dim=-1, keepdim=True) + 1e-8)
                            contextual_token_num = min(contextual_token_num, normalized_embeds.shape[0])
                            target_step = max(1, normalized_embeds.shape[0] // contextual_token_num)
                            target_local = torch.arange(
                                0, normalized_embeds.shape[0], target_step, device=device
                            )[:contextual_token_num]
                            target_embeds = normalized_embeds[target_local]
                            source_mask = torch.ones(normalized_embeds.shape[0], dtype=torch.bool, device=device)
                            source_mask[target_local] = False
                            source_embeds = normalized_embeds[source_mask]
                            target_positions = remaining_positions[target_local]
                            selected_masks[target_positions] = True
                            if source_embeds.shape[0] > 0:
                                similarity = torch.matmul(source_embeds, target_embeds.t())
                                assign_one_hot = image_embeds.new_zeros(source_embeds.shape[0], target_local.numel())
                                assign_one_hot.scatter_(1, similarity.argmax(dim=-1).unsqueeze(-1), 1)
                                counts = assign_one_hot.sum(dim=0).clamp(min=1).unsqueeze(-1)
                                aggregated_embeds = torch.matmul(assign_one_hot.t(), source_embeds) / counts
                                image_embeds[target_positions] += aggregated_embeds

                elif self.pruning_method == 'duet_faithful':
                    # Faithful Qwen3 port of AMD-AGI/DUET-VLM Qwen2.5 Stage 1 at
                    # cd69e489f39a: dominant attention top-k, then the published
                    # C-variant's attention preselection (cluster_width=4), anchor
                    # sampling inside that candidate pool, cosine assignment, and
                    # aggregation of RAW hidden states. The same assignment is
                    # applied to every Qwen3 DeepStack stream; merely subsetting
                    # DeepStack would give contextual tokens two inconsistent
                    # meanings and can bias the comparison.
                    _t = int(self.target_visual_token_num)
                    dominant_token_num = max(1, int(round(_t / 128.0 * 170)))
                    contextual_token_num = max(0, int(round(_t / 128.0 * 35)))
                    dominant_token_num = min(dominant_token_num, N)
                    contextual_token_num = min(contextual_token_num, N - dominant_token_num)

                    if image_attns.dim() == 3:
                        duet_attn = image_attns.mean(dim=1).mean(dim=0)
                    elif image_attns.dim() == 2:
                        duet_attn = image_attns.mean(dim=0)
                    else:
                        duet_attn = image_attns
                    duet_attn = duet_attn.reshape(-1)
                    if duet_attn.numel() != N:
                        raise RuntimeError(
                            f"[DUET-Faithful] attention/token mismatch: "
                            f"scores={duet_attn.numel()} visual_tokens={N}"
                        )

                    dominant_indices = duet_attn.topk(dominant_token_num).indices
                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[dominant_indices] = True

                    if contextual_token_num > 0:
                        remaining_positions = (~selected_masks).nonzero(as_tuple=True)[0]
                        contextual_token_num = min(contextual_token_num, remaining_positions.numel())
                        contextual_scores = duet_attn[remaining_positions]

                        # AMD-AGI's C variant: keep a 4x attention-ranked candidate
                        # pool, sample contextual anchors from that ordered pool,
                        # and merge the other candidates into their nearest anchor.
                        cluster_width = 4
                        preselect_num = min(
                            max(1, cluster_width * contextual_token_num),
                            remaining_positions.numel(),
                        )
                        preselect_local = contextual_scores.topk(preselect_num).indices
                        anchor_step = max(1, preselect_num // max(1, contextual_token_num))
                        anchor_pos = torch.arange(
                            0, preselect_num, anchor_step, device=device
                        )[:contextual_token_num]
                        if anchor_pos.numel() != contextual_token_num:
                            raise RuntimeError(
                                f"[DUET-Faithful] anchor count mismatch: "
                                f"anchors={anchor_pos.numel()} expected={contextual_token_num}"
                            )

                        target_local = preselect_local[anchor_pos]
                        target_positions = remaining_positions[target_local]
                        selected_masks[target_positions] = True

                        candidate_merge_mask = torch.ones(
                            preselect_num, dtype=torch.bool, device=device
                        )
                        candidate_merge_mask[anchor_pos] = False
                        merge_local = preselect_local[candidate_merge_mask]
                        source_positions = remaining_positions[merge_local]

                        if source_positions.numel() > 0:
                            normalized_remaining = image_embeds[remaining_positions]
                            normalized_remaining = normalized_remaining / (
                                normalized_remaining.norm(dim=-1, keepdim=True) + 1e-8
                            )
                            target_metric = normalized_remaining[target_local]
                            source_metric = normalized_remaining[merge_local]
                            similarity = torch.matmul(source_metric, target_metric.t())
                            assignment = image_embeds.new_zeros(
                                source_positions.numel(), target_positions.numel()
                            )
                            assignment.scatter_(
                                1, similarity.argmax(dim=-1).unsqueeze(-1), 1
                            )
                            counts = assignment.sum(dim=0).clamp(min=1).unsqueeze(-1)

                            def merge_stream(stream):
                                if stream.ndim != 2 or stream.shape[0] != N:
                                    raise RuntimeError(
                                        f"[DUET-Faithful] DeepStack alignment mismatch: "
                                        f"stream_shape={tuple(stream.shape)} expected_tokens={N}"
                                    )
                                stream = stream.clone()
                                target_idx = target_positions.to(stream.device)
                                source_idx = source_positions.to(stream.device)
                                weights = assignment.to(stream.device, stream.dtype)
                                denom = counts.to(stream.device, stream.dtype)
                                target_hidden = stream[target_idx]
                                source_hidden = stream[source_idx]
                                aggregated_hidden = torch.matmul(
                                    weights.t(), source_hidden
                                ) / denom
                                stream[target_idx] = target_hidden + aggregated_hidden
                                return stream

                            image_embeds = merge_stream(image_embeds)
                            if deepstack_image_embeds is not None:
                                deepstack_image_embeds = [
                                    merge_stream(stream)
                                    for stream in deepstack_image_embeds
                                ]

                    expected_keep = dominant_token_num + contextual_token_num
                    observed_keep = int(selected_masks.sum().item())
                    if observed_keep != expected_keep:
                        raise RuntimeError(
                            f"[DUET-Faithful] Stage1 keep mismatch: "
                            f"observed={observed_keep} expected={expected_keep}"
                        )
                    if (
                        os.environ.get('ENABLE_DEBUG', '0') == '1'
                        and not self._duet_faithful_stage1_logged
                    ):
                        print(
                            f"[DUET-Faithful Stage1] official_sha=cd69e489f39a "
                            f"variant=C cluster_width=4 aggregation=raw "
                            f"deepstack_sync={'on' if deepstack_image_embeds is not None else 'off'} "
                            f"input={N} dominant={dominant_token_num} "
                            f"contextual={contextual_token_num} keep={observed_keep}",
                            flush=True,
                        )
                        self._duet_faithful_stage1_logged = True

                elif self.pruning_method in ('tops_2', 'tops_plus'):
                    # TOPS single-stage (ported from qwen2_5_vl:1177-1409). tops_2 / tops_plus prune
                    # directly to target in ONE vision-side RDC pass (NO Stage-2 LLM schedule -- they
                    # use the stock Qwen3-VL decoder + DeepStack, so no custom model is mounted).
                    # tops_plus additionally does a training-free ToMe-style merge (pre and/or post).
                    a1 = float(os.environ.get("A1", "1.0"))
                    alpha_1 = float(os.environ.get("ALPHA_1", "1.0"))
                    lambda_1 = float(os.environ.get("LAMBDA_1", "1.0"))
                    coverage_method = os.environ.get("COVERAGE", "MEAN")

                    stage1_keep_num = min(max(1, self.target_visual_token_num), N)

                    _img_attns = locals().get('image_attns', None)
                    if _img_attns is not None:
                        if _img_attns.dim() == 3:
                            cls_attn = _img_attns.mean(dim=1).mean(dim=0)
                        elif _img_attns.dim() == 2:
                            cls_attn = _img_attns.mean(dim=0)
                        else:
                            cls_attn = _img_attns
                    else:
                        cls_attn = image_embeds.norm(dim=-1)
                    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

                    # tops+ merge-first: adaptive multi-round bipartite near-duplicate merge (param-free)
                    if self.pruning_method == 'tops_plus' and os.environ.get("TOPS_PLUS_FIRST", "0") == "1":
                        import math as _math
                        tgt = max(1, self.target_visual_token_num)
                        R_env = os.environ.get("TOPS_PLUS_ROUNDS", "")
                        if R_env != "":
                            R = int(R_env)
                        else:
                            R = max(1, int(round(_math.log2(max(2.0, N / max(1.0, float(tgt)))))))
                        image_embeds = image_embeds.clone()
                        for _r in range(R):
                            fn = image_embeds / (image_embeds.norm(dim=-1, keepdim=True) + 1e-8)
                            simfull = torch.matmul(fn, fn.t())
                            idxA = torch.arange(0, N, 2, device=device)
                            idxB = torch.arange(1, N, 2, device=device)
                            if idxA.numel() == 0 or idxB.numel() == 0:
                                break
                            simAB = simfull[idxA][:, idxB]
                            bestB = simAB.argmax(dim=1)
                            aff = simAB.max(dim=1).values.clamp(min=0.0).float()
                            w = aff * aff
                            newB = image_embeds[idxB].clone().float()
                            wsumB = torch.ones(idxB.shape[0], device=device)
                            newB.index_add_(0, bestB, w.unsqueeze(-1) * image_embeds[idxA].float())
                            wsumB.index_add_(0, bestB, w)
                            newB = newB / wsumB.unsqueeze(-1).clamp(min=1e-6)
                            image_embeds[idxB] = newB.to(image_embeds.dtype)
                            image_embeds[idxA] = newB[bestB].to(image_embeds.dtype)
                        print(f"[tops+FIRST] {R}-round bipartite pre-merge: N={N} -> ~{N // (2 ** R)} distinct")

                    feat_n = image_embeds / (image_embeds.norm(dim=-1, keepdim=True) + 1e-8)
                    sim = torch.matmul(feat_n, feat_n.t())

                    avail = torch.ones(N, dtype=torch.bool, device=device)
                    selected = []
                    i0 = torch.argmax(cls_attn).item()
                    selected.append(i0)
                    avail[i0] = False
                    max_sim_to_S = sim[i0].clone()
                    max_sim_to_S[i0] = 1.0
                    for _ in range(1, stage1_keep_num):
                        if not avail.any():
                            break
                        diversity = 1.0 - max_sim_to_S
                        if coverage_method == "SCOPE":
                            coverage = torch.clamp(sim - max_sim_to_S.unsqueeze(0), min=0.0).sum(dim=1)
                            coverage[~avail] = 0.0
                        else:
                            sim_to_avail = sim[:, avail].sum(dim=1)
                            n_avail = avail.sum().item()
                            if n_avail > 1:
                                coverage = (sim_to_avail - 1.0) / (n_avail - 1)
                            else:
                                coverage = torch.zeros(N, device=device)
                        a_mean = cls_attn.mean().clamp(min=1e-8)
                        div_mean = diversity.mean().clamp(min=1e-8)
                        cov_mean = coverage.mean().clamp(min=1e-8)
                        scores = a1 * (cls_attn / a_mean) + \
                            alpha_1 * (diversity / div_mean) + \
                            lambda_1 * (coverage / cov_mean)
                        scores[~avail] = -float("inf")
                        idx = torch.argmax(scores).item()
                        selected.append(idx)
                        avail[idx] = False
                        max_sim_to_S = torch.maximum(max_sim_to_S, sim[idx])
                        max_sim_to_S[idx] = 1.0
                    selected_tensor = torch.tensor(selected, device=device, dtype=torch.long)
                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[selected_tensor] = True

                    # tops+ merge-back (training-free): pruned tokens fold into their most-similar kept
                    # token. Operates on image_embeds only; the DeepStack side-features are pruned by the
                    # same selected_masks downstream (P5), so the merged primary embeds carry the signal.
                    if self.pruning_method == 'tops_plus' and os.environ.get("TOPS_PLUS_MERGE", "1") == "1":
                        if os.environ.get("TOPS_PLUS_CRM", "0") == "1":
                            sel_idx = selected_tensor
                            pruned_idx = (~selected_masks).nonzero().squeeze(-1)
                            if pruned_idx.numel() > 0:
                                sim_ps = sim[pruned_idx][:, sel_idx]
                                assign = sim_ps.argmax(dim=1)
                                aff = sim_ps.max(dim=1).values.clamp(min=0.0).float()
                                pe = image_embeds[pruned_idx].float()
                                anc = image_embeds[sel_idx][assign].float()
                                anc_hat = anc / (anc.norm(dim=-1, keepdim=True) + 1e-8)
                                proj = (pe * anc_hat).sum(dim=-1, keepdim=True) * anc_hat
                                resid = pe - proj
                                contrib = aff.unsqueeze(-1) * resid
                                Kk = sel_idx.shape[0]
                                acc = torch.zeros(Kk, pe.shape[1], device=device, dtype=torch.float32)
                                acc.index_add_(0, assign, contrib)
                                orig = image_embeds[sel_idx].float()
                                v = orig + acc
                                v = v * (orig.norm(dim=-1, keepdim=True) / (v.norm(dim=-1, keepdim=True) + 1e-8))
                                image_embeds = image_embeds.clone()
                                image_embeds[sel_idx] = v.to(image_embeds.dtype)
                                print(f"[tops+CRM] residual-injected {pruned_idx.numel()} pruned -> {Kk} kept")
                        else:
                            pf = os.environ.get("TOPS_PLUS_PF", "1") == "1"
                            sel_idx = selected_tensor
                            pruned_idx = (~selected_masks).nonzero().squeeze(-1)
                            if pf and pruned_idx.numel() > 0:
                                sim_ps = sim[pruned_idx][:, sel_idx]
                                assign = sim_ps.argmax(dim=1)
                                aff = sim_ps.max(dim=1).values.clamp(min=0.0).float()
                                w = aff * aff
                                merged = image_embeds[sel_idx].clone().float()
                                wsum = torch.ones(sel_idx.shape[0], device=device)
                                pe = image_embeds[pruned_idx].float()
                                merged.index_add_(0, assign, w.unsqueeze(-1) * pe)
                                wsum.index_add_(0, assign, w)
                                merged = merged / wsum.unsqueeze(-1).clamp(min=1e-6)
                                image_embeds = image_embeds.clone()
                                image_embeds[sel_idx] = merged.to(image_embeds.dtype)
                                print(f"[tops+PF] merged {pruned_idx.numel()} pruned -> {sel_idx.shape[0]} kept (aff^2)")
                    print(f"[TOPS-{self.pruning_method}] single-stage selected {int(selected_masks.sum().item())} tokens")

                elif self.pruning_method in ('tsvol', 'tsvol_w', 'tsvol_anti', 'tsvol_w_anti'):
                    # TSVOL (Text-Subspace VOLume), ported from qwen2_5_vl:939-1012. Project visual tokens
                    # into (normal) / orthogonal-complement of (anti) the prompt-text SVD subspace, then
                    # QR max-volume greedy for K diverse tokens; _w variants row-weight by cls_attn^0.5.
                    import os as _os
                    weighted = self.pruning_method in ('tsvol_w', 'tsvol_w_anti')
                    anti = self.pruning_method in ('tsvol_anti', 'tsvol_w_anti')
                    tol = float(_os.environ.get('TSVOL_TOL', '1e-3'))
                    K = int(min(max(1, self.target_visual_token_num), N))
                    ie = image_embeds.float()
                    Nv, Cf = ie.shape
                    if _os.environ.get('TSVOL_VNORM', '0') == '1':
                        ie = ie / (ie.norm(dim=-1, keepdim=True) + 1e-8)
                    image_token_indices = torch.where(input_ids == self.config.image_token_id)[1]
                    if image_token_indices.numel() > 0:
                        te = inputs_embeds.squeeze(0)[image_token_indices[-1] + 2:-5].float()
                    else:
                        te = inputs_embeds.new_zeros((0, D)).float()
                    if te.shape[0] == 0:
                        # No text query: fall back to uniform stride (deterministic), matching the
                        # empty-text guards the qwen3 trim/cdpruner ports use.
                        selected_indices = torch.linspace(0, N - 1, K, device=device).round().long().unique()
                        selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                        selected_masks[selected_indices] = True
                        print(f"[TSVOL] method={self.pruning_method} no-text fallback K={K}")
                    else:
                        _, s_t, vh_t = torch.linalg.svd(te, full_matrices=False)
                        r = int((s_t > tol * s_t[0]).sum().item()) if s_t.numel() > 0 else 1
                        r = max(r, 1)
                        Q_T = vh_t[:r].transpose(0, 1)
                        w_pow = float(_os.environ.get('TSVOL_W_POW', '0.5' if weighted else '0'))
                        _img_attns = locals().get('image_attns', None)
                        use_w = (w_pow > 0) and (_img_attns is not None)
                        if use_w:
                            if _img_attns.dim() == 3:
                                cls_w = _img_attns.mean(dim=1).mean(dim=0)
                            elif _img_attns.dim() == 2:
                                cls_w = _img_attns.mean(dim=0)
                            else:
                                cls_w = _img_attns
                            w_vec = torch.pow(torch.clamp(cls_w.float(), min=0.0) + 1e-8, w_pow)
                        if anti:
                            coeff = ie @ Q_T
                            U = ie - coeff @ Q_T.transpose(0, 1)
                        else:
                            U = ie @ Q_T
                        if use_w:
                            U = U * w_vec.unsqueeze(-1)
                        nrm2 = (U * U).sum(dim=1)
                        chosen = []
                        avail = torch.ones(Nv, dtype=torch.bool, device=device)
                        i0 = int(torch.argmax(nrm2).item())
                        chosen.append(i0)
                        avail[i0] = False
                        Qbasis = (U[i0] / (U[i0].norm() + 1e-8)).unsqueeze(1)
                        steps = min(K, Nv) if anti else min(K, r, Nv)
                        for _ in range(1, steps):
                            proj = U @ Qbasis
                            resid = U - proj @ Qbasis.transpose(0, 1)
                            res_norm = resid.norm(dim=1)
                            res_norm[~avail] = -float('inf')
                            j = int(torch.argmax(res_norm).item())
                            if res_norm[j].item() <= 1e-6:
                                break
                            chosen.append(j)
                            avail[j] = False
                            qj = resid[j] / (resid[j].norm() + 1e-8)
                            Qbasis = torch.cat([Qbasis, qj.unsqueeze(1)], dim=1)
                        if len(chosen) < K:
                            rem = nrm2.clone()
                            rem[~avail] = -float('inf')
                            extra = torch.topk(rem, K - len(chosen)).indices
                            chosen.extend(extra.tolist())
                        selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                        selected_masks[torch.tensor(chosen[:K], device=device, dtype=torch.long)] = True
                        print(f"[TSVOL] method={self.pruning_method} K={K} r={r} anti={anti} w={use_w} "
                              f"selected={int(selected_masks.sum().item())}")

                elif self.pruning_method == 'divprune_coverage':
                    # DivPrune-Coverage (qwen2_5_vl:798-829): DivPrune greedy frame, but the criterion is
                    # SCOPE marginal-coverage gain instead of max-min diversity. No relevance, no weights.
                    image_normalized = image_embeds / (image_embeds.norm(dim=-1, keepdim=True) + 1e-8)
                    sim = torch.matmul(image_normalized, image_normalized.t())
                    K = int(min(max(1, self.target_visual_token_num), N))
                    avail = torch.ones(N, dtype=torch.bool, device=device)
                    selected = []
                    g = torch.zeros(N, device=device)
                    cov0 = torch.clamp(sim, min=0.0).sum(dim=1)
                    i0 = int(torch.argmax(cov0).item())
                    selected.append(i0)
                    avail[i0] = False
                    g = torch.maximum(g, sim[i0])
                    for _ in range(1, K):
                        if not avail.any():
                            break
                        gain = torch.clamp(sim - g.unsqueeze(0), min=0.0).sum(dim=1)
                        gain[~avail] = -float('inf')
                        j = int(torch.argmax(gain).item())
                        selected.append(j)
                        avail[j] = False
                        g = torch.maximum(g, sim[j])
                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[torch.tensor(selected, device=device, dtype=torch.long)] = True
                    print(f"[divprune_coverage] K={K} selected={int(selected_masks.sum().item())}")

                elif self.pruning_method == 'divprune_coverage_to_others_only':
                    # Ablation (qwen2_5_vl:830-859): coverage over OTHERS only (diagonal self-term removed).
                    image_normalized = image_embeds / (image_embeds.norm(dim=-1, keepdim=True) + 1e-8)
                    sim = torch.matmul(image_normalized, image_normalized.t())
                    sim_off = sim.clone()
                    sim_off.fill_diagonal_(0.0)
                    K = int(min(max(1, self.target_visual_token_num), N))
                    avail = torch.ones(N, dtype=torch.bool, device=device)
                    selected = []
                    g = torch.zeros(N, device=device)
                    cov0 = torch.clamp(sim_off, min=0.0).sum(dim=1)
                    i0 = int(torch.argmax(cov0).item())
                    selected.append(i0)
                    avail[i0] = False
                    g = torch.maximum(g, sim[i0])
                    for _ in range(1, K):
                        if not avail.any():
                            break
                        gain = torch.clamp(sim_off - g.unsqueeze(0), min=0.0).sum(dim=1)
                        gain[~avail] = -float('inf')
                        j = int(torch.argmax(gain).item())
                        selected.append(j)
                        avail[j] = False
                        g = torch.maximum(g, sim[j])
                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[torch.tensor(selected, device=device, dtype=torch.long)] = True
                    print(f"[divprune_coverage_to_others_only] K={K} selected={int(selected_masks.sum().item())}")

                elif self.pruning_method in ('cover_qe', 'cover_prune', 'cover_e', 'cover_q', 'cover_u'):
                    # COVER-Prune (qwen2_5_vl:860-917): weighted facility-location coverage.
                    #   w_v = energy_v * task_v; energy=||x_v||^2 (norm to mean 1);
                    #   task_v = (1-lam)+lam*N*softmax(r_v/tau), r_v(Q)=max_t cos(visual v, text query t).
                    # Ablations: cover_u(w=1)/cover_e(energy)/cover_q(task)/cover_qe==cover_prune(both).
                    import os as _os
                    lam = float(_os.environ.get('COVER_LAMBDA', '0.5'))
                    tau = float(_os.environ.get('COVER_TAU', '0.1'))
                    use_e = self.pruning_method in ('cover_qe', 'cover_prune', 'cover_e')
                    use_q = self.pruning_method in ('cover_qe', 'cover_prune', 'cover_q')
                    K = int(min(max(1, self.target_visual_token_num), N))
                    image_normalized = image_embeds / (image_embeds.norm(dim=-1, keepdim=True) + 1e-8)
                    sim = torch.clamp(torch.matmul(image_normalized, image_normalized.t()), min=0.0)
                    if use_e:
                        energy = image_embeds.float().pow(2).sum(dim=-1)
                        energy = energy / (energy.mean() + 1e-6)
                    else:
                        energy = torch.ones(N, device=device)
                    if use_q:
                        img_idx = torch.where(input_ids == self.config.image_token_id)[1]
                        if img_idx.numel() > 0:
                            te = inputs_embeds.squeeze(0)[img_idx[-1] + 2:-5].float()
                        else:
                            te = inputs_embeds.new_zeros((0, D)).float()
                        if te.shape[0] > 0:
                            ie = image_embeds.float()
                            ien = ie / (ie.norm(dim=-1, keepdim=True) + 1e-8)
                            ten = te / (te.norm(dim=-1, keepdim=True) + 1e-8)
                            r_v = torch.matmul(ien, ten.t()).max(dim=1).values
                            pi = torch.softmax(r_v / tau, dim=0)
                            task = (1.0 - lam) + lam * N * pi
                        else:
                            task = torch.ones(N, device=device)
                    else:
                        task = torch.ones(N, device=device)
                    w = (energy * task).to(sim.dtype)
                    w = w / (w.mean() + 1e-6)
                    avail = torch.ones(N, dtype=torch.bool, device=device)
                    selected = []
                    g = torch.zeros(N, device=device, dtype=sim.dtype)
                    cov0 = (w.unsqueeze(0) * sim).sum(dim=1)
                    i0 = int(torch.argmax(cov0).item())
                    selected.append(i0)
                    avail[i0] = False
                    g = torch.maximum(g, sim[i0])
                    for _ in range(1, K):
                        if not avail.any():
                            break
                        gain = (w.unsqueeze(0) * torch.clamp(sim - g.unsqueeze(0), min=0.0)).sum(dim=1)
                        gain[~avail] = -float('inf')
                        j = int(torch.argmax(gain).item())
                        selected.append(j)
                        avail[j] = False
                        g = torch.maximum(g, sim[j])
                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[torch.tensor(selected, device=device, dtype=torch.long)] = True
                    print(f"[{self.pruning_method}] K={K} lam={lam} tau={tau} e={use_e} q={use_q} "
                          f"selected={int(selected_masks.sum().item())}")

                elif self.pruning_method in ('cover_prune_1', 'cover_prune_1_proj', 'scope_vit_origin', 'scope_raw'):
                    # SCOPE OUTSIDE weighting: A_v^1 * SCOPE-coverage-gain (official Algorithm 1:
                    #   v* = argmax_v Δ(v;S)·A_v^α, α=1; Δ(v;S)=Σ_u max(0, sim(u,v)-c_u)).
                    #   cover_prune_1_proj = single-stage, coverage on projector/embed (merged) space
                    #   scope_vit_origin   = single-stage, coverage on PRE-merge ViT features, A_v min-max normalized
                    #   scope_raw          = single-stage, coverage on PRE-merge ViT features, A_v RAW (no norm)
                    #                        -- faithful to the paper, which does NOT normalize the attention.
                    #   cover_prune_1      = two-stage (->2T)
                    _proj = self.pruning_method != 'cover_prune_1'   # single-stage unless the 2-stage cp1
                    stage1_keep_num = min(max(1, self.target_visual_token_num * (1 if _proj else 2)), N)
                    _img_attns = locals().get('image_attns', None)
                    if _img_attns is not None:
                        cls_attn = (_img_attns.mean(dim=1).mean(dim=0) if _img_attns.dim() == 3
                                    else (_img_attns.mean(dim=0) if _img_attns.dim() == 2 else _img_attns))
                    else:
                        cls_attn = image_embeds.norm(dim=-1)
                    # scope_raw keeps A_v RAW (paper does no normalization); all others min-max to [0,1].
                    if self.pruning_method != 'scope_raw':
                        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
                    if self.pruning_method in ('scope_vit_origin', 'scope_raw'):
                        _vf = getattr(self, '_mmtok_merger_in', None)
                        if _vf is not None and _vf.shape[0] // 4 == N:
                            cov_feat = _vf.float().reshape(N, 4, -1).mean(dim=1); _covsrc = 'vit'
                        else:
                            cov_feat = image_embeds.float(); _covsrc = 'proj(fallback)'
                    else:
                        cov_feat = image_embeds.float(); _covsrc = 'proj'
                    image_normalized = cov_feat / (cov_feat.norm(dim=-1, keepdim=True) + 1e-8)
                    sim = torch.matmul(image_normalized, image_normalized.t())
                    avail = torch.ones(N, dtype=torch.bool, device=device)
                    selected = []
                    g = torch.zeros(N, device=device)
                    cov0 = torch.clamp(sim, min=0.0).sum(dim=1)
                    i0 = int(torch.argmax(cls_attn * cov0).item())
                    selected.append(i0); avail[i0] = False; g = torch.maximum(g, sim[i0])
                    for _ in range(1, stage1_keep_num):
                        if not avail.any():
                            break
                        coverage = torch.clamp(sim - g.unsqueeze(0), min=0.0).sum(dim=1)
                        scores = cls_attn * coverage
                        scores[~avail] = -float('inf')
                        j = int(torch.argmax(scores).item())
                        selected.append(j); avail[j] = False; g = torch.maximum(g, sim[j])
                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[torch.tensor(selected, device=device, dtype=torch.long)] = True
                    print(f"[{self.pruning_method} S1] cls_attn x coverage cov_space={_covsrc}: N={N} -> keep "
                          f"{int(selected_masks.sum().item())} ({'target' if _proj else 'target*2'}={stage1_keep_num})")

                elif self.pruning_method in ('cover_prune_2', 'cover_prune_2_proj', 'cover_prune_2_projector_vit', 'cover_prune_2_projector_vit_raw', 'cover_prune_2_projector_vit_raw_merge_first', 'cover_prune_2_projector_merge_first'):
                    # cover_prune_2 (two-stage ->2T, in-LLM S2) / cover_prune_2_proj (single-stage ->T,
                    # coverage on projector/embed space) / cover_prune_2_projector_vit (single-stage ->T,
                    # coverage on PRE-merge ViT features, A_v min-max normalized) / cover_prune_2_projector_vit_raw
                    # (same but A_v RAW, no normalization) / cover_prune_2_projector_vit_raw_merge_first
                    # (raw + ViT space + tops+-style bipartite aff^2 near-dup pre-merge on cov_feat) /
                    # cover_prune_2_projector_merge_first (projector/embed space + A_v MIN-MAX + pre-merge).
                    # merge-first: does NOT drop tokens, only makes near-duplicate features converge, THEN
                    # inside-weighted coverage greedy (cls_attn multiplied into each coverage term).
                    # Ported from qwen2_5_vl:1228-1261 + cover_prune_2_projector_vit:1302-1345 + merge_first:1356-1380.
                    _mergefirst = 'merge_first' in self.pruning_method
                    _raw = '_raw' in self.pruning_method
                    _vitspace = self.pruning_method in ('cover_prune_2_projector_vit', 'cover_prune_2_projector_vit_raw', 'cover_prune_2_projector_vit_raw_merge_first')
                    _proj = self.pruning_method != 'cover_prune_2'  # single-stage unless the 2-stage cp2
                    stage1_keep_num = min(max(1, self.target_visual_token_num * (1 if _proj else 2)), N)
                    _img_attns = locals().get('image_attns', None)
                    if _img_attns is not None:
                        cls_attn = (_img_attns.mean(dim=1).mean(dim=0) if _img_attns.dim() == 3
                                    else (_img_attns.mean(dim=0) if _img_attns.dim() == 2 else _img_attns))
                    else:
                        cls_attn = image_embeds.norm(dim=-1)
                    # *_raw keeps A_v RAW (no normalization); others min-max to [0,1].
                    if not _raw:
                        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
                    if _vitspace:
                        _vf = getattr(self, '_mmtok_merger_in', None)
                        if _vf is not None and _vf.shape[0] // 4 == N:
                            cov_feat = _vf.float().reshape(N, 4, -1).mean(dim=1); _covsrc = 'vit'
                        else:
                            cov_feat = image_embeds.float(); _covsrc = 'proj(fallback)'
                    else:
                        cov_feat = image_embeds.float(); _covsrc = 'proj'
                    # merge-first: R-round bipartite aff^2 near-dup pre-merge on cov_feat (tops+ style;
                    # does NOT drop tokens, only makes near-dup features converge). R = round(log2(N/target)).
                    if _mergefirst:
                        import math as _math
                        tgt = max(1, self.target_visual_token_num)
                        R = max(1, int(round(_math.log2(max(2.0, N / max(1.0, float(tgt)))))))
                        cov_feat = cov_feat.clone()
                        for _r in range(R):
                            fn = cov_feat / (cov_feat.norm(dim=-1, keepdim=True) + 1e-8)
                            simfull = torch.matmul(fn, fn.t())
                            idxA = torch.arange(0, N, 2, device=device)
                            idxB = torch.arange(1, N, 2, device=device)
                            if idxA.numel() == 0 or idxB.numel() == 0:
                                break
                            simAB = simfull[idxA][:, idxB]
                            bestB = simAB.argmax(dim=1)
                            aff = simAB.max(dim=1).values.clamp(min=0.0)
                            w = aff * aff
                            newB = cov_feat[idxB].clone()
                            wsumB = torch.ones(idxB.shape[0], device=device)
                            newB.index_add_(0, bestB, w.unsqueeze(-1) * cov_feat[idxA])
                            wsumB.index_add_(0, bestB, w)
                            newB = newB / wsumB.unsqueeze(-1).clamp(min=1e-6)
                            cov_feat[idxB] = newB
                            cov_feat[idxA] = newB[bestB]
                        print(f"[cp2projvit_raw_merge_first] {R}-round pre-merge on cov_feat(vit): N={N} target={tgt}")
                    image_normalized = cov_feat / (cov_feat.norm(dim=-1, keepdim=True) + 1e-8)
                    sim = torch.clamp(torch.matmul(image_normalized, image_normalized.t()), min=0.0)
                    avail = torch.ones(N, dtype=torch.bool, device=device)
                    selected = []
                    g = torch.zeros(N, device=device)
                    cov0 = (cls_attn.unsqueeze(0) * sim).sum(dim=1)
                    i0 = int(torch.argmax(cov0).item())
                    selected.append(i0); avail[i0] = False; g = torch.maximum(g, sim[i0])
                    for _ in range(1, stage1_keep_num):
                        if not avail.any():
                            break
                        gain = (cls_attn.unsqueeze(0) * torch.clamp(sim - g.unsqueeze(0), min=0.0)).sum(dim=1)
                        gain[~avail] = -float('inf')
                        j = int(torch.argmax(gain).item())
                        selected.append(j); avail[j] = False; g = torch.maximum(g, sim[j])
                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[torch.tensor(selected, device=device, dtype=torch.long)] = True
                    print(f"[{self.pruning_method} S1] sum cls_attn(v)*cov cov_space={_covsrc}: N={N} -> keep "
                          f"{int(selected_masks.sum().item())} ({'target' if _proj else 'target*2'}={stage1_keep_num})")

                elif self.pruning_method in (
                    'mico_coverage_vit_proj_llm_12', 'mico_coverage_vit_proj_llm_6',
                    'mico_coverage_vit_proj_llm_8', 'mico_coverage_vit_proj_llm_10',
                    'mico_coverage_vit_proj',
                    'mico_coverage_proj_llm_12', 'mico_coverage_proj',
                    'mico_coverage_proj_llm_6', 'mico_coverage_proj_llm_8', 'mico_coverage_proj_llm_10',
                    'mico_coverage_proj_llm_11', 'mico_coverage_proj_llm_13',
                ) or _is_mico_s1(self.pruning_method):
                    # Same inner raw-attn×coverage as LLaVA/InternVL. Weight is last-ViT
                    # attention (Qwen3 has no CLS; InternVL uses CLS query row). Cosine
                    # space: vit_proj* = pre-merger ViT; mico_coverage_proj* = merger/projector out.
                    keep_2t = '_llm_' in self.pruning_method
                    use_vit = 'vit_proj' in self.pruning_method
                    stage1_multiplier = self.model.stage1_multiplier
                    stage1_keep_num = min(max(1, self.target_visual_token_num * stage1_multiplier), N)
                    _img_attns = locals().get('image_attns', None)
                    if _img_attns is not None:
                        cls_attn = (
                            _img_attns.mean(dim=1).mean(dim=0) if _img_attns.dim() == 3
                            else (_img_attns.mean(dim=0) if _img_attns.dim() == 2 else _img_attns)
                        )
                    else:
                        cls_attn = image_embeds.norm(dim=-1)
                    if use_vit:
                        _vf = getattr(self, '_mmtok_merger_in', None)
                        if _vf is not None and _vf.shape[0] // 4 == N:
                            cov_feat = _vf.float().reshape(N, 4, -1).mean(dim=1)
                            _covsrc = 'vit'
                        else:
                            cov_feat = image_embeds.float()
                            _covsrc = 'proj(fallback)'
                    else:
                        cov_feat = image_embeds.float()
                        _covsrc = 'proj'
                    if cov_feat.shape[0] != N:
                        raise ValueError(
                            f"{self.pruning_method} S1 feature rows {cov_feat.shape[0]} != N={N}"
                        )
                    mico = _is_mico_s1(self.pruning_method)
                    selected_masks = inner_weighted_raw_cls_select(
                        cov_feat, cls_attn, stage1_keep_num, mico=mico
                    )
                    print(
                        f"[{self.pruning_method}] sample S1={_covsrc} raw_visual_attn"
                        f"{'*|h|' if mico else ''}*coverage "
                        f"{str(stage1_multiplier) + 'T' if keep_2t else 'T'}: N={N} keep={int(selected_masks.sum().item())} "
                        f"({'S2=post-layer L' + str(self.model.K) if keep_2t else 'no LLM prune'})",
                        flush=True,
                    )

                elif self.pruning_method in ('mmtok', 'mmtok_origin'):
                    # MMTok (ICLR2026, arXiv 2508.18264): multimodal max-coverage, single-stage,
                    # after-projector selection. Ported from qwen2_5_vl/modeling_qwen2_5_vl.py:942-1005.
                    #   P = softmax(z_norm @ x_norm.T / tv)/M   text->vision coverage [M,N] (x = projector out)
                    #   Q = softmax(qn @ qn.T / vv)/N           vision->vision coverage [N,N] (qn = PRE-merge ViT)
                    #   Combined=[P; alpha*Q], greedy max-coverage: delta_j = Σ_row max(0, C[row,j]-best[row]).
                    # mmtok: text = truncated prompt text in inputs_embeds; mmtok_origin: re-encode
                    #   "Question: {q}" (split-into-words, drop BOS) via embed_tokens. Q uses ViT space both.
                    import os as _os
                    tv = float(_os.environ.get('MMTOK_TV_TEMP', '0.02'))
                    vv = float(_os.environ.get('MMTOK_VV_TEMP', '0.2'))
                    alpha = float(_os.environ.get('MMTOK_ALPHA', '0.5'))
                    device_ = image_embeds.device
                    K = min(self.target_visual_token_num, N)
                    x = image_embeds.float()
                    x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
                    # text query tokens (truncated / re-encoded)
                    if self.pruning_method == 'mmtok_origin':
                        te = x.new_zeros((0, x.shape[1]))
                        q_text = getattr(self, '_mmtok_question', None)
                        tok = getattr(self, '_mmtok_tokenizer', None)
                        if q_text and tok is not None:
                            enc = tok(("Question: " + q_text).split(), is_split_into_words=True,
                                      return_tensors="pt", padding=True, truncation=True)
                            qids = enc["input_ids"].to(device_)
                            emb = self.model.get_input_embeddings()(qids)[0]
                            _bos = getattr(tok, 'bos_token_id', None)
                            st = 1 if (qids.shape[1] > 1 and _bos is not None and int(qids[0, 0]) == _bos) else 0
                            te = emb[st:].float()
                    else:
                        img_idx = torch.where(input_ids == self.config.image_token_id)[1]
                        te = inputs_embeds.squeeze(0)[img_idx[-1] + 2:-5].float() if img_idx.numel() > 0 \
                            else x.new_zeros((0, x.shape[1]))
                    # Q on PRE-merge ViT features (reshape N,4,-1 mean; NO window reorder in Qwen3-VL)
                    _vf = getattr(self, '_mmtok_merger_in', None)
                    if _vf is not None and _vf.shape[0] // 4 == N:
                        q_feat = _vf.float().reshape(N, 4, -1).mean(dim=1); q_src = 'vit'
                    else:
                        q_feat = x; q_src = 'proj(fallback)'
                    qn = q_feat / (q_feat.norm(dim=-1, keepdim=True) + 1e-8)
                    Q = torch.softmax((qn @ qn.t()) / vv, dim=1) / N
                    if te.shape[0] > 0:
                        z_norm = te / (te.norm(dim=-1, keepdim=True) + 1e-8)
                        M = z_norm.shape[0]
                        P = torch.softmax((z_norm @ x_norm.t()) / tv, dim=1) / M
                        Combined = torch.cat([P, alpha * Q], dim=0)
                    else:
                        Combined = alpha * Q
                    best = torch.zeros(Combined.shape[0], device=device_, dtype=Combined.dtype)
                    score_mask = torch.zeros(N, device=device_, dtype=Combined.dtype)
                    sel = torch.empty(K, dtype=torch.long, device=device_)
                    for i in range(K):
                        delta = (Combined - best.unsqueeze(1)).clamp_min(0.0).sum(dim=0) + score_mask
                        j = int(torch.argmax(delta).item())
                        sel[i] = j
                        best = torch.maximum(best, Combined[:, j])
                        score_mask[j] = -float('inf')
                    selected_masks = torch.zeros(N, dtype=torch.bool, device=device)
                    selected_masks[sel] = True
                    print(f"[{self.pruning_method}] K={K} tv={tv} vv={vv} alpha={alpha} Q_space={q_src} "
                          f"M_text={te.shape[0]} selected={int(selected_masks.sum().item())}")

                elif self.pruning_method not in [
                    'vanilla',
                    'fastervlm', 'prumerge', 'prumerge+', 'visionzip', 'visionzip+', 'vispruner',
                    'tops_variant3', 'mico_coverage_variant', 'qr_block_star_avg',
                    'tops', 'fastv', 'random', 'duet_faithful', 'qr_duet',
                    'dart', 'dart_coverage', 'cover_llm',
                ]:
                    # Guard: an unmatched method would otherwise run UNPRUNED with a fictional token
                    # count (the exact silent-no-op bug P11 fixes). Fail loudly instead.
                    raise ValueError(
                        f"pruning_method={self.pruning_method!r} has no Stage-1 selection branch and "
                        f"would silently run UNPRUNED. Add a branch or use a supported method."
                    )

                self.visual_token_len = selected_masks.sum().item()

                # ── Validate token count ──
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: "
                        f"tokens: {n_image_tokens}, features: {n_image_features}"
                    )

                # ── Scatter image_embeds into inputs_embeds ──
                mask = input_ids == self.config.image_token_id
                # P7c: record the absolute positions of the image tokens NOW. `mask` is rebound to the
                # video mask at the video branch below, and W:585's old `int(mask.max(dim=-1).indices)`
                # (a scalar "system prompt length") is exactly the 15-vs-4 window bug -- capture the
                # full index vector instead.
                image_token_positions = mask[0].nonzero(as_tuple=False).squeeze(-1)
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.model.visual.dtype)
                video_embeds_out = self.model.visual(pixel_values_videos, grid_thw=video_grid_thw)
                if hasattr(video_embeds_out, 'pooler_output') and video_embeds_out.pooler_output is not None:
                    video_embeds = video_embeds_out.pooler_output
                elif hasattr(video_embeds_out, 'last_hidden_state'):
                    video_embeds = video_embeds_out.last_hidden_state
                else:
                    video_embeds = video_embeds_out
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: "
                        f"tokens: {n_video_tokens}, features: {n_video_features}"
                    )
                mask = input_ids == self.config.video_token_id
                mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # ── RoPE index. Only reachable OUTSIDE generate(): transformers 5.2 generate() precomputes a
        #    4D [4,B,S] position_ids in prepare_inputs_for_generation (UP:1512-1546) and always passes
        #    it, so `position_ids is None` is False in the VLMEvalKit path and this block is dormant. ──
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                # P1: Qwen3-VL's get_rope_index signature is
                #   (input_ids, image_grid_thw, video_grid_thw, attention_mask, **kwargs)  -- UP:978-985.
                # It removed Qwen2.5-VL's `second_per_grid_ts`. Passing it (the old 5-positional call)
                # is a TypeError on every image prefill, and **kwargs does NOT absorb extra POSITIONAL
                # args. Keyword form also survives a future parameter reorder.
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas

        # ── P7: Stage-1 compaction ──────────────────────────────────────────────────────────────
        # The OLD prune lived under `if position_ids is None:` (W:568). Under generate() position_ids
        # is never None, so that gate was ALWAYS False: stage-1 selection was computed and then
        # silently discarded, and every tops_variant3 / tops / visionzip / prumerge / divprune / cdpruner
        # run in this tree was UNPRUNED while the logs reported a fictional token count. Dead-code
        # claim AST-verified: the If at W:568 tested `position_ids is None`, and its entire body
        # (the prune at 583-598 and the decode arm at 599-617) was unreachable when position_ids was
        # a real tensor. Re-gate on a robust prefill test that does not depend on position_ids.
        _past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        _is_prefill = (_past_len == 0) and (cache_position is None or int(cache_position[0]) == 0)

        if _is_prefill:
            self._orig_prompt_len = inputs_embeds.shape[1]
            self.retain_token_indices = None      # reset: stale indices must not leak across samples

            new_visual_positions = None
            if (
                selected_masks is not None
                and image_token_positions is not None
                and bool(selected_masks.sum() < selected_masks.numel())
            ):
                dev = inputs_embeds.device
                ipos = image_token_positions.to(dev)
                sel = selected_masks.to(dev)
                selected_image_positions = ipos[sel]
                retain_mask = torch.ones(inputs_embeds.shape[1], dtype=torch.bool, device=dev)
                retain_mask[ipos] = False
                retain_mask[selected_image_positions] = True
                retain_token_indices = retain_mask.nonzero(as_tuple=False).squeeze(-1)
                self.retain_token_indices = retain_token_indices

                inputs_embeds = inputs_embeds[:, retain_token_indices]
                if position_ids is not None:
                    # `[..., idx]` is correct for BOTH [4,B,S] (generate) and [3,B,S] (get_rope_index).
                    position_ids = position_ids[..., retain_token_indices]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, retain_token_indices]
                if cache_position is not None:
                    # Contiguous truncation, NOT cache_position[retain_token_indices]: post-prune the
                    # DynamicCache holds exactly new_len entries at slots 0..new_len-1; gathered values
                    # would exceed kv_length and make prefill silently BIDIRECTIONAL.
                    cache_position = cache_position[: retain_token_indices.shape[0]]

                # Positions of the surviving image tokens in the COMPACTED sequence.
                new_visual_positions = torch.searchsorted(retain_token_indices, selected_image_positions)
                # P5: prune the DeepStack side-features in sync with the stage-1 keep mask.
                if deepstack_image_embeds is not None:
                    deepstack_visual_embeds = [d[sel.to(d.device)] for d in deepstack_image_embeds]
                    visual_pos_masks = torch.zeros(inputs_embeds.shape[:2], dtype=torch.bool, device=dev)
                    visual_pos_masks[0, new_visual_positions] = True
            elif image_token_positions is not None:
                # No stage-1 pruning (e.g. fastv, or T>=N): keep the full visual window but STILL route
                # DeepStack so the unpruned run is Qwen3-VL WITH DeepStack, comparable to vanilla.
                dev = inputs_embeds.device
                new_visual_positions = image_token_positions.to(dev)
                if deepstack_image_embeds is not None:
                    deepstack_visual_embeds = list(deepstack_image_embeds)
                    visual_pos_masks = torch.zeros(inputs_embeds.shape[:2], dtype=torch.bool, device=dev)
                    visual_pos_masks[0, new_visual_positions] = True

            # P6: hand the custom decoders (fastv / tops_variant3 / tops) the explicit post-stage-1
            # visual-position index vector. Their setters reject None, so push only when present; a
            # text-only prompt simply leaves the previous vector in place (fastv self-guards via a
            # freshness flag; tops_variant3/tops never see a text-only prompt in this image benchmark suite).
            if (self.pruning_method in (
                'fastv', 'tops_variant3', 'mico_coverage_variant', 'qr_block_star_avg',
                'tops', 'duet', 'duet_faithful', 'qr_duet',
                'dart', 'dart_coverage', 'cover_llm', 'sparsevlm', 'apet',
                'cover_prune_2', 'mico_coverage_vit_proj_llm_12', 'mico_coverage_vit_proj_llm_6',
                'mico_coverage_vit_proj_llm_8', 'mico_coverage_vit_proj_llm_10', 'mico_coverage_proj_llm_12',
                'mico_coverage_proj_llm_6', 'mico_coverage_proj_llm_8', 'mico_coverage_proj_llm_10',
                'mico_coverage_proj_llm_11', 'mico_coverage_proj_llm_13',
                'mico_coverage_vit_proj', 'mico_coverage_proj', 'mico_s1',
            ) or _is_mico_s1(self.pruning_method)) and new_visual_positions is not None:
                self.model.set_visual_token_positions(new_visual_positions)
        else:
            # Decode step. generate() grows attention_mask against the UNPRUNED prompt
            # (generation/utils.py:1068-1072), so at decode step t it is S_orig+t wide while the cache
            # holds new_len+t. Narrow it to the retained prefix + the decoded tail. Idempotent, unlike
            # the in-place growth of self.retain_token_indices in the deleted W:612-617.
            ri = getattr(self, 'retain_token_indices', None)
            if attention_mask is not None and ri is not None:
                tail = torch.arange(
                    self._orig_prompt_len, attention_mask.shape[1], device=ri.device, dtype=ri.dtype
                )
                attention_mask = attention_mask[:, torch.cat((ri, tail))]

        # ── P7e: dispatch. Custom decoders take the DeepStack kwargs directly; stock-decoder methods
        #    must go through self.model.language_model(...) or DeepStack is silently OFF (UP:1242-1243
        #    forces visual_pos_masks/deepstack_visual_embeds to None inside Qwen3VLModel.forward). ────
        if self.pruning_method in (
            'fastv', 'tops_variant3', 'mico_coverage_variant', 'qr_block_star_avg',
            'tops', 'duet', 'duet_faithful', 'qr_duet',
            'dart', 'dart_coverage', 'cover_llm', 'sparsevlm', 'apet',
            'cover_prune_2', 'mico_coverage_vit_proj_llm_12', 'mico_coverage_vit_proj_llm_6',
            'mico_coverage_vit_proj_llm_8', 'mico_coverage_vit_proj_llm_10', 'mico_coverage_proj_llm_12',
            'mico_coverage_proj_llm_6', 'mico_coverage_proj_llm_8', 'mico_coverage_proj_llm_10',
            'mico_coverage_proj_llm_11', 'mico_coverage_proj_llm_13',
            'mico_coverage_vit_proj', 'mico_coverage_proj', 'mico_s1',
        ) or _is_mico_s1(self.pruning_method):
            outputs = self.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )
        else:
            # Stock path. Qwen3VLTextModel.forward (UP:851-863) does NOT declare output_attentions /
            # output_hidden_states / return_dict -- do not pass them. It returns a BaseModelOutputWithPast,
            # so outputs[0], .past_key_values, .hidden_states, .attentions below all resolve.
            lm_kwargs = dict(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            if _USING_QWEN3:
                # Only Qwen3's text model accepts these; the Qwen2.5-VL fallback would forward them into
                # its decoder layers via **kwargs (C2). DeepStack is a Qwen3-only concept anyway.
                if position_ids is None:
                    # Bypassing Qwen3VLModel.forward skips compute_3d_position_ids (UP:1150), so the text
                    # model would fall back to plain 1-D positions and silently lose MRoPE. Recover it.
                    # (Only reachable outside generate(); generate() supplies a 4D position_ids.)
                    position_ids = self.model.compute_3d_position_ids(
                        input_ids=input_ids, inputs_embeds=inputs_embeds,
                        image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                        attention_mask=attention_mask, past_key_values=past_key_values,
                    )
                    lm_kwargs['position_ids'] = position_ids
                lm_kwargs['visual_pos_masks'] = visual_pos_masks
                lm_kwargs['deepstack_visual_embeds'] = deepstack_visual_embeds
            outputs = self.model.language_model(**lm_kwargs)

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return _CausalLMOutputCls(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
