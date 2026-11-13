"""MiCo vision-token selection and decoder refinement for Qwen3-VL."""

import math
import os
import types
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.nn import CrossEntropyLoss
from transformers.utils import logging

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
    from transformers.models.qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration as _BASE_CONDITIONAL_GEN,
    )
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLModel as _BASE_MODEL_CLS,
        Qwen2_5_VLCausalLMOutputWithPast as _CausalLMOutputCls,
        apply_rotary_pos_emb_vision,
    )

    _USING_QWEN3 = False
    print(
        "[modeling_qwen3_vl] WARNING: Qwen3-VL not found in transformers. Please upgrade to transformers>=5.7. Falling back to Qwen2.5-VL structure (may not work)."
    )
from .mico_decoder import MiCoQwen3VLDecoder, inner_weighted_raw_cls_select

logger = logging.get_logger(__name__)


def _is_mico(method: str) -> bool:
    prefix = "mico_"
    return isinstance(method, str) and method.startswith(prefix) and method[len(prefix) :].isdigit()


def _is_mico_s1(method: str) -> bool:
    return method == "mico_s1" or _is_mico(method)


def _qr_residual_select_exact(
    features: torch.Tensor, keep: int, forced_indices: Optional[List[int]] = None
) -> torch.LongTensor:
    """Pure pivoted-QR token selection, optionally conditioned on forced pivots."""
    flat = features.detach().reshape(features.shape[0], -1).to(torch.float32)
    n_tokens = int(flat.shape[0])
    keep = min(max(1, int(keep)), n_tokens)
    initial_norms = torch.square(flat).sum(dim=1)
    if not torch.isfinite(initial_norms).all():
        raise ValueError("[QR-STAGE1] non-finite token norms")
    residual = flat.clone()
    available = torch.ones(n_tokens, dtype=torch.bool, device=flat.device)
    selected: List[int] = []
    eps = 1e-08
    for idx in forced_indices or []:
        idx = int(idx)
        if idx < 0 or idx >= n_tokens:
            raise ValueError(f"[QR-STAGE1] forced index out of range: {idx}/{n_tokens}")
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
            selected.extend((int(i) for i in remaining[: keep - len(selected)].tolist()))
            break
        selected.append(pivot)
        available[pivot] = False
        basis = residual[pivot] / (residual[pivot].norm() + eps)
        residual = residual - torch.outer(residual @ basis, basis)
    result = torch.tensor(selected, dtype=torch.long, device=flat.device)
    if result.numel() != keep or torch.unique(result).numel() != keep:
        raise RuntimeError(
            f"[QR-STAGE1] exact unique selection failed: count={result.numel()} unique={torch.unique(result).numel()} expected={keep}"
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
            if r1 - r0 > 1 or c1 - c0 > 1
        ]
        if not candidates:
            raise RuntimeError(
                f"[QWEN-BLOCK-QR] cannot create {count} blocks for {height}x{width} grid"
            )
        _, idx, r0, r1, c0, c1 = min(candidates)
        row_span, col_span = (r1 - r0, c1 - c0)
        if row_span >= col_span and row_span > 1:
            mid = r0 + row_span // 2
            pair = [(r0, mid, c0, c1), (mid, r1, c0, c1)]
        else:
            mid = c0 + col_span // 2
            pair = [(r0, r1, c0, mid), (r0, r1, mid, c1)]
        blocks[idx : idx + 1] = pair
    return blocks


def _allocate_block_counts(capacities: List[int], total: int) -> List[int]:
    """Deterministically allocate a bounded block budget proportional to grid area."""
    if not capacities or any((capacity < 1 for capacity in capacities)):
        raise ValueError(f"[QWEN-BLOCK-QR] invalid capacities: {capacities}")
    total = min(int(total), sum(capacities))
    if total < len(capacities):
        raise ValueError(
            f"[QWEN-BLOCK-QR] coverage={total} cannot cover {len(capacities)} image planes"
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
        raise RuntimeError(f"[QWEN-BLOCK-QR] allocation mismatch: {sum(allocation)} != {total}")
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
        raise ValueError("[QWEN-BLOCK-QR] image_grid_thw is required")
    grids = [[int(value) for value in row] for row in image_grid_thw.detach().cpu().tolist()]
    planes: List[Tuple[int, int, int]] = []
    offset = 0
    per_image_shapes: List[Tuple[int, int, int]] = []
    for temporal, height, width in grids:
        if height % merge_size or width % merge_size:
            raise ValueError(
                f"[QWEN-BLOCK-QR] grid {temporal}x{height}x{width} is not divisible by spatial_merge_size={merge_size}"
            )
        merged_h, merged_w = (height // merge_size, width // merge_size)
        per_image_shapes.append((temporal, merged_h, merged_w))
        for _ in range(temporal):
            planes.append((offset, merged_h, merged_w))
            offset += merged_h * merged_w
    if offset != int(image_embeds.shape[0]):
        raise RuntimeError(
            f"[QWEN-BLOCK-QR] grid/token mismatch: grid={offset} embeds={image_embeds.shape[0]}"
        )
    keep = min(max(1, int(keep)), offset)
    requested_coverage = max(len(planes), int(round(keep * float(coverage_ratio))))
    coverage_keep = min(keep, requested_coverage)
    block_counts = _allocate_block_counts(
        [height * width for _, height, width in planes], coverage_keep
    )
    token_norms = torch.square(image_embeds.detach().reshape(offset, -1).to(torch.float32)).sum(
        dim=1
    )
    coverage_indices: List[int] = []
    for (plane_offset, height, width), block_count in zip(planes, block_counts):
        for r0, r1, c0, c1 in _split_spatial_blocks(height, width, block_count):
            rows = torch.arange(r0, r1, device=image_embeds.device)
            cols = torch.arange(c0, c1, device=image_embeds.device)
            positions = (plane_offset + rows[:, None] * width + cols[None, :]).flatten()
            pivot = positions[torch.argmax(token_norms[positions])]
            coverage_indices.append(int(pivot.item()))
    selected = _qr_residual_select_exact(image_embeds, keep=keep, forced_indices=coverage_indices)
    metadata: Dict[str, Any] = {
        "coverage": len(coverage_indices),
        "blocks": block_counts,
        "grids": per_image_shapes,
    }
    return (selected, metadata)


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
        j = torch.argmin(torch.max(similarity_f, dim=-1).values) if i == 0 else torch.argmax(di2s)
        j_int = int(j)
        if chosen[j_int] or not torch.isfinite(di2s[j_int]) or di2s[j_int] <= eps:
            degenerate_from = i if degenerate_from is None else degenerate_from
            sel = selected_indices[:i]
            max_sim = (
                similarity_f[:, sel].max(dim=1).values if i > 0 else torch.zeros(N, device=device)
            )
            max_sim = torch.nan_to_num(max_sim, nan=1.0, posinf=1.0, neginf=1.0)
            max_sim[chosen] = float("inf")
            j_int = int(torch.argmin(max_sim))
            if chosen[j_int]:
                remaining = (~chosen).nonzero(as_tuple=True)[0]
                if remaining.numel() == 0:
                    n_sel = i
                    break
                j_int = int(remaining[0])
            selected_indices[i] = j_int
            chosen[j_int] = True
            di2s[j_int] = -float("inf")
            cis[i, :] = 0.0
            continue
        selected_indices[i] = j_int
        chosen[j_int] = True
        eis = (kernel[j_int, :] - cis[:i, j_int] @ cis[:i, :]) / torch.sqrt(
            torch.clamp(di2s[j_int], min=eps)
        )
        eis = torch.nan_to_num(eis, nan=0.0, posinf=0.0, neginf=0.0)
        cis[i, :] = eis
        di2s = torch.clamp(di2s - torch.square(eis), min=0.0)
        di2s[chosen] = -float("inf")
    selected_indices = selected_indices[:n_sel]
    if degenerate_from is not None:
        logger.warning(
            f"[{tag}] DPP loop degenerated at iter {degenerate_from}/{k}; correlation refill used. {int(chosen.sum())} distinct tokens selected."
        )
    return selected_indices


class Qwen3VLForConditionalGenerationPruning(_BASE_CONDITIONAL_GEN):
    """
    Wrapper around Qwen3VLForConditionalGeneration that applies visual token pruning.
    Stage 1: vision-side attention-guided token selection (here in forward).
    Stage 2: LLM-side progressive pruning (delegated to pruning model classes).
    """

    _supports_flash_attn_2 = False

    def __init__(self, config, pruning_method="vanilla", visual_token_num=-1):
        super().__init__(config)
        if pruning_method in ("mico_2", "mico_4", "mico_6"):
            mico = pruning_method in ("mico_2", "mico_4", "mico_6")
            k_layer = (
                None
                if not mico and pruning_method.endswith("_llm_12")
                else int(pruning_method.rsplit("_", 1)[1])
            )
            self.model = MiCoQwen3VLDecoder(
                config, visual_token_num, k_layer=k_layer, method_name=pruning_method, mico=mico
            )
        elif pruning_method in ("mico_s1",):
            self.model = MiCoQwen3VLDecoder(
                config,
                visual_token_num,
                method_name=pruning_method,
                mico=pruning_method in ("mico_s1", "mico_2", "mico_4", "mico_6"),
                single_stage=True,
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
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        selected_masks = None
        image_token_positions = None
        deepstack_image_embeds = None
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.model.visual.dtype)
                if self.pruning_method in ("mico_s1", "mico_2", "mico_4", "mico_6"):

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
                        cos, sin = position_embeddings
                        q_rot, k_rot = apply_rotary_pos_emb_vision(q, k, cos, sin)
                        self_attn.q = q.clone()
                        self_attn.k = k.clone()
                        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
                        q_chunks = torch.split(q_rot, lengths, dim=0)
                        k_chunks = torch.split(k_rot, lengths, dim=0)
                        v_chunks = torch.split(v, lengths, dim=0)
                        attn_chunk_outputs = []
                        for qi, ki, vi in zip(q_chunks, k_chunks, v_chunks):
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
                        scaling = 1.0 / math.sqrt(q_rot.shape[-1])
                        if len(lengths) > 1:
                            all_attn_weights = []
                            offset = 0
                            for Li in lengths:
                                q_img = q_rot[offset : offset + Li].transpose(0, 1)
                                k_img = k_rot[offset : offset + Li].transpose(0, 1)
                                attn_img = torch.matmul(q_img, k_img.transpose(-2, -1)) * scaling
                                attn_img = torch.nn.functional.softmax(attn_img.float(), dim=-1).to(
                                    q_rot.dtype
                                )
                                all_attn_weights.append(attn_img)
                                offset += Li
                            num_heads = all_attn_weights[0].shape[0]
                            max_dim = max((w.shape[-1] for w in all_attn_weights))
                            avg_attn = torch.zeros(
                                num_heads, max_dim, max_dim, device=q_rot.device, dtype=q_rot.dtype
                            )
                            for attn_block in all_attn_weights:
                                s = attn_block.shape[-1]
                                avg_attn[:, :s, :s] += attn_block
                            avg_attn = avg_attn / len(all_attn_weights)
                            self_attn.attn_weights = avg_attn
                            self_attn.attn_blocks = all_attn_weights
                            self_attn.num_images = len(lengths)
                        else:
                            try:
                                q2 = q_rot.transpose(0, 1)
                                k2 = k_rot.transpose(0, 1)
                                attn_weights = torch.matmul(q2, k2.transpose(-2, -1)) * scaling
                                attn_weights = torch.nn.functional.softmax(
                                    attn_weights.float(), dim=-1
                                ).to(q2.dtype)
                                self_attn.attn_weights = attn_weights
                            except torch.cuda.OutOfMemoryError:
                                q_probe = q_rot.mean(dim=0, keepdim=True).transpose(0, 1)
                                k2 = k_rot.transpose(0, 1)
                                attn_scores = torch.matmul(q_probe, k2.transpose(-2, -1)) * scaling
                                attn_weights = torch.nn.functional.softmax(
                                    attn_scores.float(), dim=-1
                                ).to(q_probe.dtype)
                                self_attn.attn_weights = attn_weights.expand(-1, seq_length, -1)
                        return attn_output

                    self.model.visual.blocks[-1].attn.forward = types.MethodType(
                        forward_with_attn, self.model.visual.blocks[-1].attn
                    )
                if self.pruning_method in ("mico_s1", "mico_2", "mico_4", "mico_6") and (
                    not getattr(self, "_mico_merger_hook_set", False)
                ):

                    def _cap_merger_in(mod, inp):
                        self._mico_merger_input = inp[0]

                    self.model.visual.merger.register_forward_pre_hook(_cap_merger_in)
                    self._mico_merger_hook_set = True
                image_embeds_out = self.model.visual(pixel_values, grid_thw=image_grid_thw)
                if (
                    hasattr(image_embeds_out, "pooler_output")
                    and image_embeds_out.pooler_output is not None
                ):
                    image_embeds = image_embeds_out.pooler_output
                elif hasattr(image_embeds_out, "last_hidden_state"):
                    image_embeds = image_embeds_out.last_hidden_state
                else:
                    image_embeds = image_embeds_out
                deepstack_image_embeds = getattr(image_embeds_out, "deepstack_features", None)
                if deepstack_image_embeds is not None and len(deepstack_image_embeds) == 0:
                    deepstack_image_embeds = None
                attn_last = self.model.visual.blocks[-1].attn
                if self.pruning_method in ("mico_s1", "mico_2", "mico_4", "mico_6"):
                    has_window_index = hasattr(self.model.visual, "get_window_index")
                    if hasattr(attn_last, "num_images") and attn_last.num_images > 1:
                        all_patch_scores = []
                        for attn_block in attn_last.attn_blocks:
                            all_patch_scores.append(attn_block.mean(dim=0).mean(dim=0))
                        all_patch_scores = torch.cat(all_patch_scores)
                        token_scores = []
                        offset = 0
                        for thw in image_grid_thw:
                            _, H, W = (thw[0].item(), thw[1].item(), thw[2].item())
                            t = H * W
                            img_scores = all_patch_scores[offset : offset + t]
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
                            image_attns = attn_weights.view(
                                -1, seq_length // 4, 4, seq_length // 4, 4
                            ).mean(dim=(2, 4))
                            image_attns = image_attns[:, reverse_indices, :][:, :, reverse_indices]
                        else:
                            image_attns = attn_weights.view(
                                -1, seq_length // 4, 4, seq_length // 4, 4
                            ).mean(dim=(2, 4))
                N, D = image_embeds.shape
                device = image_embeds.device
                selected_masks = torch.ones(N, dtype=torch.bool, device=device)
                if self.pruning_method in ("mico_s1", "mico_2", "mico_4", "mico_6"):
                    keep_2t = self.pruning_method in ("mico_2", "mico_4", "mico_6")
                    use_vit = False
                    stage1_multiplier = self.model.stage1_multiplier
                    stage1_keep_num = min(
                        max(1, self.target_visual_token_num * stage1_multiplier), N
                    )
                    _img_attns = locals().get("image_attns", None)
                    if _img_attns is not None:
                        cls_attn = (
                            _img_attns.mean(dim=1).mean(dim=0)
                            if _img_attns.dim() == 3
                            else _img_attns.mean(dim=0) if _img_attns.dim() == 2 else _img_attns
                        )
                    else:
                        cls_attn = image_embeds.norm(dim=-1)
                    if use_vit:
                        _vf = getattr(self, "_mico_merger_input", None)
                        if _vf is not None and _vf.shape[0] // 4 == N:
                            cov_feat = _vf.float().reshape(N, 4, -1).mean(dim=1)
                            _covsrc = "vit"
                        else:
                            cov_feat = image_embeds.float()
                            _covsrc = "proj(fallback)"
                    else:
                        cov_feat = image_embeds.float()
                        _covsrc = "proj"
                    if cov_feat.shape[0] != N:
                        raise ValueError(
                            f"{self.pruning_method} S1 feature rows {cov_feat.shape[0]} != N={N}"
                        )
                    mico = self.pruning_method in ("mico_s1", "mico_2", "mico_4", "mico_6")
                    selected_masks = inner_weighted_raw_cls_select(
                        cov_feat, cls_attn, stage1_keep_num, mico=mico
                    )
                    print(
                        f"[{self.pruning_method}] sample S1={_covsrc} raw_visual_attn{'*|h|'}*coverage {(str(stage1_multiplier) + 'T' if keep_2t else 'T')}: N={N} keep={int(selected_masks.sum().item())} ({('S2=post-layer L' + str(self.model.K) if keep_2t else 'no LLM prune')})",
                        flush=True,
                    )
                elif self.pruning_method in ("mico_s1", "mico_2", "mico_4", "mico_6"):
                    raise ValueError(
                        f"pruning_method={self.pruning_method!r} has no Stage-1 selection branch and would silently run UNPRUNED. Add a branch or use a supported method."
                    )
                self.visual_token_len = selected_masks.sum().item()
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features: {n_image_features}"
                    )
                mask = input_ids == self.config.image_token_id
                image_token_positions = mask[0].nonzero(as_tuple=False).squeeze(-1)
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.model.visual.dtype)
                video_embeds_out = self.model.visual(pixel_values_videos, grid_thw=video_grid_thw)
                if (
                    hasattr(video_embeds_out, "pooler_output")
                    and video_embeds_out.pooler_output is not None
                ):
                    video_embeds = video_embeds_out.pooler_output
                elif hasattr(video_embeds_out, "last_hidden_state"):
                    video_embeds = video_embeds_out.last_hidden_state
                else:
                    video_embeds = video_embeds_out
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features: {n_video_features}"
                    )
                mask = input_ids == self.config.video_token_id
                mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, video_embeds)
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                cache_position is not None
                and cache_position[0] == 0
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
        _past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        _is_prefill = _past_len == 0 and (cache_position is None or int(cache_position[0]) == 0)
        if _is_prefill:
            self._orig_prompt_len = inputs_embeds.shape[1]
            self.retain_token_indices = None
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
                    position_ids = position_ids[..., retain_token_indices]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, retain_token_indices]
                if cache_position is not None:
                    cache_position = cache_position[: retain_token_indices.shape[0]]
                new_visual_positions = torch.searchsorted(
                    retain_token_indices, selected_image_positions
                )
                if deepstack_image_embeds is not None:
                    deepstack_visual_embeds = [d[sel.to(d.device)] for d in deepstack_image_embeds]
                    visual_pos_masks = torch.zeros(
                        inputs_embeds.shape[:2], dtype=torch.bool, device=dev
                    )
                    visual_pos_masks[0, new_visual_positions] = True
            elif image_token_positions is not None:
                dev = inputs_embeds.device
                new_visual_positions = image_token_positions.to(dev)
                if deepstack_image_embeds is not None:
                    deepstack_visual_embeds = list(deepstack_image_embeds)
                    visual_pos_masks = torch.zeros(
                        inputs_embeds.shape[:2], dtype=torch.bool, device=dev
                    )
                    visual_pos_masks[0, new_visual_positions] = True
            if (
                self.pruning_method in ("mico_s1",)
                or self.pruning_method in ("mico_s1", "mico_2", "mico_4", "mico_6")
            ) and new_visual_positions is not None:
                self.model.set_visual_token_positions(new_visual_positions)
        else:
            ri = getattr(self, "retain_token_indices", None)
            if attention_mask is not None and ri is not None:
                tail = torch.arange(
                    self._orig_prompt_len, attention_mask.shape[1], device=ri.device, dtype=ri.dtype
                )
                attention_mask = attention_mask[:, torch.cat((ri, tail))]
        if self.pruning_method in ("mico_s1", "mico_2", "mico_4", "mico_6"):
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
                if position_ids is None:
                    position_ids = self.model.compute_3d_position_ids(
                        input_ids=input_ids,
                        inputs_embeds=inputs_embeds,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                    )
                    lm_kwargs["position_ids"] = position_ids
                lm_kwargs["visual_pos_masks"] = visual_pos_masks
                lm_kwargs["deepstack_visual_embeds"] = deepstack_visual_embeds
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
