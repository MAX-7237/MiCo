"""VisPruner (ECCV 2024, arXiv 2407.04620) selector.

Official reference: `Theia-4869/VisPruner @ aefa01adc7c7ce6334e880c88225e90cede760d1`,
`llava/model/llava_arch.py:140-186`, as vendored in
`/path/to/local-assets`.

VisPruner keeps `important_ratio` of the budget by CLS-attention rank, then
picks the remaining "diverse" tokens by an iterative cosine-redundancy cull
over the non-important set. Selection only -- it does not merge features.

InternVL3's `vispruner` branch applies this per patch on ViT-space features
(`vit_pre_proj`, before `mlp1`), matching the LLaVA path where the selector
runs on the pre-projector CLIP features. Qwen2.5-VL keeps its own inline
implementation in `modeling_qwen2_5_vl.py`.
"""

import math
from numbers import Integral, Real

import torch


@torch.no_grad()
def vispruner_select(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    keep: int,
    important_ratio: float = 0.5,
) -> torch.Tensor:
    """Return original-order Long[B, keep] patch indices on features.device.

    ``features`` are patch features [B, N, D] (ViT space). ``cls_attention`` is
    the CLS-to-patch attention, either [B, H, N] (averaged over heads here) or
    its precomputed mean [B, N].

    The authors' default is important_ratio=0.5. Arithmetic (no FP32 promotion,
    no epsilon on the normalization) matches upstream. Uneven partitions use
    their actual lengths, which occurs for InternVL3's 256 / 5 crop budget.
    """
    if features.ndim != 3 or any(size < 1 for size in features.shape):
        raise ValueError("features must have nonempty shape [B,N,D]")
    if not features.is_floating_point() or not cls_attention.is_floating_point():
        raise TypeError("features and cls_attention must be floating point")
    batch, count, _ = features.shape
    if cls_attention.device != features.device:
        raise ValueError("features and cls_attention must be on the same device")
    if cls_attention.ndim == 3:
        if cls_attention.shape[0] != batch or cls_attention.shape[2] != count or cls_attention.shape[1] < 1:
            raise ValueError("cls_attention must have shape [B,H,N] with H >= 1")
        attention = cls_attention.mean(dim=1)
    elif cls_attention.ndim == 2 and cls_attention.shape == (batch, count):
        attention = cls_attention
    else:
        raise ValueError("cls_attention must have shape [B,H,N] or [B,N]")
    if not isinstance(keep, Integral) or isinstance(keep, bool) or not 0 <= keep <= count:
        raise ValueError("keep must be an integer in [0,N]")
    if not isinstance(important_ratio, Real) or not math.isfinite(important_ratio) or not 0 <= important_ratio <= 1:
        raise ValueError("important_ratio must be finite and in [0,1]")

    important_count = int(keep * important_ratio)
    diverse_count = keep - important_count
    token_indices = attention.argsort(dim=-1, descending=True)
    important_indices = token_indices[:, :important_count]
    residual_indices = token_indices[:, important_count:]

    # Deliberately no FP32 conversion or epsilon: same arithmetic as upstream.
    normalized = features / features.norm(dim=-1, keepdim=True)
    rows = torch.arange(batch, device=features.device).unsqueeze(-1)
    while diverse_count > 0:
        residual_count = residual_indices.shape[1]
        remove_count = min(8, residual_count - diverse_count)
        if remove_count <= 0:
            break
        left_count = (residual_count + 1) // 2
        if remove_count > left_count:
            raise ValueError(
                "unsupported bipartite shape: removal count exceeds the left partition"
            )
        residual_tokens = normalized[rows.expand(-1, residual_count), residual_indices]
        a, b = residual_tokens[..., ::2, :], residual_tokens[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)
        scores = scores.max(dim=-1).values
        distinct_indices = scores.argsort(dim=-1, descending=True)[:, remove_count:]
        # Advanced indexing assumed equally sized partitions; gather also
        # handles the extra left token of an odd residual without padding.
        residual_indices = torch.cat([
            residual_indices[..., ::2].gather(1, distinct_indices),
            residual_indices[..., 1::2],
        ], dim=-1)

    selected = torch.cat([important_indices, residual_indices], dim=-1) if diverse_count > 0 else important_indices
    # Author scatters a bool mask then gathers patches in original sequence order.
    return selected.sort(dim=-1).values
