"""HoloV (NeurIPS 2025, arXiv 2510.02912) selector.

Official reference: `LLaVA-TOPS-ICLR/llava/model/utils.py:HoloV` (and the copy
vendored into `/path/to/local-assets`). An image's tokens are split into patches; a
keep-budget is allocated across patches by (CLS-attn + similarity-variance)
score; the top-CLS-attn tokens inside each patch are kept. Selection only --
HoloV does not merge.

InternVL3's `holov` branch implements this over its single fixed patch layout.
Qwen2.5-VL packs several variable-resolution images into one flat `[N, D]`
tensor, so the selector here works per image, using `image_grid_thw` for each
image's true lattice.
"""

import torch


HOLOV_ALPHA = 1.0
HOLOV_BETA = 0.09
HOLOV_POWER = 1.0
HOLOV_EPS = 1e-6


def holov_select(
    image_embeds: torch.Tensor,
    relevance: torch.Tensor,
    img_shapes,
    target_tokens: int,
    alpha: float = HOLOV_ALPHA,
    beta: float = HOLOV_BETA,
    power: float = HOLOV_POWER,
    eps: float = HOLOV_EPS,
) -> torch.Tensor:
    """Select tokens to keep, per image, under a shared token budget.

    Args:
        image_embeds: [N, D] flat concatenation of every image's tokens.
        relevance:    [N] per-token CLS-like attention score.
        img_shapes:   list of (frames, H, W) per image, in embedding-token units
                      (i.e. after the 2x2 patch merger).
        target_tokens: total tokens to keep across all images.

    Returns:
        Boolean mask [N]; the number of True entries is min(target_tokens, N)
        whenever that is achievable (i.e. at least one token per image).
    """
    n_total, dim = image_embeds.shape
    device = image_embeds.device
    img_lengths = [int(f * h * w) for (f, h, w) in img_shapes]
    if sum(img_lengths) != n_total:
        raise ValueError(
            f"HoloV: image shapes imply {sum(img_lengths)} tokens but got {n_total}")

    keep_total = min(int(target_tokens), n_total)
    num_images = len(img_lengths)

    # Distribute the budget across images proportional to size, >=1 each, then
    # correct the rounding drift so the total is exact.
    alloc = [max(1, keep_total * n_i // max(1, sum(img_lengths))) for n_i in img_lengths]
    while sum(alloc) > keep_total:
        j = int(max(range(num_images), key=lambda k: alloc[k]))
        if alloc[j] <= 1:
            break
        alloc[j] -= 1
    while sum(alloc) < keep_total:
        j = int(max(range(num_images), key=lambda k: img_lengths[k]))
        alloc[j] += 1

    norm_all = image_embeds / (image_embeds.norm(dim=-1, keepdim=True) + eps)
    mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    offset = 0
    for img_i, n_i in enumerate(img_lengths):
        if n_i <= 0:
            continue
        frames, Hs, Ws = img_shapes[img_i]
        keep_i = min(alloc[img_i], n_i)
        attn_i = relevance[offset:offset + n_i]
        norm_i = norm_all[offset:offset + n_i]

        # Region grid (near-square), ~keep_i regions so a small budget still
        # reaches every region.
        r_target = max(1, min(keep_i, 16))
        gh = max(1, int(r_target ** 0.5))
        gw = max(1, (r_target + gh - 1) // gh)
        row_id = (torch.arange(Hs, device=device) * gh) // Hs
        col_id = (torch.arange(Ws, device=device) * gw) // Ws
        region_of_token = (row_id.view(Hs, 1) * gw + col_id.view(1, Ws)).reshape(Hs * Ws)
        if frames > 1:
            region_of_token = region_of_token.repeat(frames)
        num_regions = gh * gw

        # score = alpha * scaled CLS attn + beta * scaled similarity variance
        S = torch.matmul(norm_i, norm_i.t())
        eye = 1.0 - torch.eye(n_i, device=device, dtype=S.dtype)
        S = S * eye
        denom = max(n_i - 1, 1)
        mean_sim = S.sum(dim=1) / denom
        var_sim = ((S - mean_sim.unsqueeze(1)) ** 2 * eye).sum(dim=1) / denom
        attn_scaled = attn_i * 1e3
        var_scale = attn_scaled.abs().mean() / (var_sim.abs().mean() + eps)
        token_scores = alpha * attn_scaled + beta * (var_sim * var_scale)

        region_score = torch.zeros(num_regions, device=device, dtype=token_scores.dtype)
        region_count = torch.zeros(num_regions, device=device, dtype=token_scores.dtype)
        region_score.scatter_add_(0, region_of_token, token_scores)
        region_count.scatter_add_(0, region_of_token, torch.ones(n_i, device=device, dtype=token_scores.dtype))
        region_score = region_score / region_count.clamp(min=1)

        w = region_score.clamp(min=0) ** power
        w = w / (w.sum() + eps)
        per_region = torch.floor(w * keep_i).long()
        remaining = keep_i - int(per_region.sum().item())
        if remaining > 0:
            extra = torch.topk(w, k=min(remaining, num_regions)).indices
            per_region[extra] += 1

        for r in range(num_regions):
            a = int(per_region[r].item())
            if a <= 0:
                continue
            region_idx = (region_of_token == r).nonzero(as_tuple=True)[0]
            if region_idx.numel() == 0:
                continue
            a = min(a, region_idx.numel())
            top = attn_i[region_idx].topk(a).indices
            mask[offset + region_idx[top]] = True

        # Fill any shortfall left by per-region size capping.
        cur = int(mask[offset:offset + n_i].sum().item())
        if cur < keep_i:
            fill = attn_i.clone()
            fill[mask[offset:offset + n_i]] = -float('inf')
            avail = int((~mask[offset:offset + n_i]).sum().item())
            need = min(keep_i - cur, avail)
            if need > 0:
                mask[offset + fill.topk(need).indices] = True
        offset += n_i

    return mask
