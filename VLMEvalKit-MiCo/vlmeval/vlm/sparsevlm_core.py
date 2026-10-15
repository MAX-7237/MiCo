"""SparseVLM (arXiv 2406.11717) core algorithm, backbone-agnostic.

The official implementation prunes *inside* the LLM decoder rather than at the
vision encoder. At each pruning layer it

  1. scores visual tokens by the cross-attention they receive from the text
     tokens that were selected as "raters" (computed once, before layer 0), and
  2. keeps the top-`keep` visual tokens, then recycles the rest: the most
     attended 30% of the dropped tokens are clustered with DPC-kmeans and
     merged into `10%` centroids, which are inserted back into the sequence.

Step 2 is why `keep` is not the final token count -- the merged centroids are
added on top. `merge_and_retain` therefore returns the actual new visual length.

The two backbones this repo evaluates (Qwen2.5-VL and InternVL3-8B) both use a
Qwen2 decoder, so the `forward` loops in the adapters differ only in how they
locate the visual span; everything numeric lives here.
"""

from typing import Optional

import torch
import torch.nn.functional as F


# Retention ratio applied at each pruning layer, per model. Values are the
# fraction of the *current* visual span kept outright (before recycling), tuned
# so the layer-average visual-token count lands on the requested budget. See
# `calibrate` in the adapters' debug output to re-derive.
#
# 28-layer backbones (Qwen2.5-VL-7B, InternVL3-8B). The official LLaVA schedule
# is [2, 6, 15] on 32 layers and [2, 8, 20] on 40 layers; scaled by relative
# depth to 28 layers that is [2, 5, 13].
SPARSEVLM_PRUNING_LAYERS_28 = (2, 5, 13)

# Relative sizes of the three post-prune visual spans, normalised from the
# official 7B/32-layer R_dict at T=128 (300/110/35 out of 576). Only the *shape*
# matters: the schedule solver below rescales it to hit the requested
# layer-average, so these are ratios, not token counts.
SPARSEVLM_SPAN_SHAPE_28 = (300.0, 110.0, 35.0)


def _layer_weight_distribution(num_layers: int, pruning_locs) -> list:
    """How many layers see each visual length, using the repo's convention.

    FastV/DUET/STAR count the visual length *after* each layer's own prune, so a
    pruning layer is counted at its post-prune length. Verified against FastV's
    documented identity `avg = (N*K + R*(L-K))/L` with K=2: weights `[2, 26]`.

        n0 = p1,  n1 = p2 - p1,  n2 = p3 - p2,  n3 = num_layers - p3
    """
    locs = sorted(pruning_locs)
    if len(locs) != 3:
        raise ValueError(f"expected 3 pruning layers, got {locs}")
    p1, p2, p3 = locs
    if not (0 <= p1 < p2 < p3 < num_layers):
        raise ValueError(f"pruning layers {locs} must be strictly increasing and < {num_layers}")
    counts = [p1, p2 - p1, p3 - p2, num_layers - p3]
    if sum(counts) != num_layers:
        raise ValueError(f"layer weights {counts} do not sum to {num_layers}")
    return counts


def _profile_spans(v0: int, scale: float, shape, pruning_locs, num_layers: int) -> list:
    """Map a scalar `scale` to a feasible strictly-decreasing span profile."""
    shares = [scale * s for s in shape]
    prev = v0
    spans = []
    for share in shares:
        s = int(round(share))
        s = max(1, min(s, prev - 1))
        spans.append(s)
        prev = s
    return spans


def build_schedule(v0: int, target_avg: int, num_layers: int = 28,
                   pruning_locs=SPARSEVLM_PRUNING_LAYERS_28,
                   shape=SPARSEVLM_SPAN_SHAPE_28) -> dict:
    """Solve per-pruning-layer target spans so the layer-average equals `target_avg`.

    Every method in this repo is compared iso-budget at the same layer-average,
    so the schedule must be derived from the requested budget rather than
    hardcoded -- otherwise T=256 and T=128 would produce identical runs.

    The official R_dict fixes only the *relative* depth profile (shallow layers
    prune least). We keep that profile and rescale it (monotonically in one
    scalar) until the weighted layer-average meets the target. Binary search is
    used instead of a closed form because the `prev - 1` ordering clamp makes
    the feasible set non-linear when the profile is squeezed.

    Returns {layer_idx: target_span_after_pruning}.
    """
    weights = _layer_weight_distribution(num_layers, pruning_locs)
    locs = sorted(pruning_locs)

    def avg_for(scale):
        spans = _profile_spans(v0, scale, shape, locs, num_layers)
        return sum(w * s for w, s in zip(weights, [v0] + spans)) / num_layers, spans

    # The average is monotone in `scale` (larger keeps -> larger average), so a
    # bisection converges quickly. Bracket generously.
    lo, hi = 0.0, float(v0) * 4.0
    best_spans = _profile_spans(v0, 0.0, shape, locs, num_layers)
    best_err = float('inf')
    for _ in range(64):
        mid = (lo + hi) / 2.0
        avg, spans = avg_for(mid)
        err = abs(avg - target_avg)
        if err < best_err:
            best_err, best_spans = err, spans
        if avg < target_avg:
            lo = mid
        else:
            hi = mid

    # Coarse fallback: when the profile cannot reach the target (e.g. a tiny
    # image at a large budget), fall back to the widest feasible profile.
    if best_err > 2.0:
        for scale in range(0, int(v0 * 4) + 1, max(1, v0 // 64)):
            avg, spans = avg_for(float(scale))
            if abs(avg - target_avg) < best_err:
                best_err, best_spans = abs(avg - target_avg), spans

    return {loc: best_spans[i] for i, loc in enumerate(locs)}


def solve_keep_for_span(current_span: int, target_span: int,
                        recycle_ratio: float = 0.3, cluster_ratio: float = 0.1) -> int:
    """Invert the recycling rule: how many tokens to keep so the span lands on `target_span`.

    The span after pruning is `keep + cluster_num`, and `cluster_num` depends on
    how many tokens were dropped. Fixed-point iterate -- `cluster_num` moves by
    at most a couple of tokens per step, so this converges in a handful of
    rounds.
    """
    target_span = int(max(1, min(target_span, current_span)))
    keep = max(1, target_span)
    for _ in range(16):
        keep = max(1, min(keep, current_span - 1))
        dropped = current_span - keep
        if dropped <= 0:
            return current_span
        merge_num = max(int(dropped * recycle_ratio), 1)
        cluster_num = max(int(merge_num * cluster_ratio), 1)
        corrected = max(1, min(target_span - cluster_num, current_span - 1))
        if corrected == keep:
            break
        keep = corrected
    return max(1, min(keep, current_span - 1))


def select_text_raters(
    hidden_states: torch.Tensor,
    visual_start: int,
    visual_end: int,
) -> torch.Tensor:
    """Pick the text tokens used to score visual relevance.

    Official SparseVLM embeds the question alone and correlates it with the
    visual tokens, keeping query tokens whose score exceeds the mean. Here the
    prompt text follows the visual span in `hidden_states`, so the same
    "above-mean" rule is applied to the trailing text positions. Returns
    indices RELATIVE to `visual_end`.
    """
    seq_len = hidden_states.shape[1]
    text_start = visual_end
    if seq_len <= text_start:
        # Degenerate: no trailing text. Fall back to the last position.
        return hidden_states.new_tensor([max(0, seq_len - 1 - visual_end)], dtype=torch.long)

    text_tokens = hidden_states[:, text_start:, :]          # [B, M, D]
    visual_tokens = hidden_states[:, visual_start:visual_end, :]  # [B, V, D]
    if visual_tokens.shape[1] == 0:
        return torch.arange(
            seq_len - text_start, device=hidden_states.device, dtype=torch.long
        )

    matrix = text_tokens @ visual_tokens.transpose(1, 2)     # [B, M, V]
    matrix = matrix.squeeze(0).softmax(0).mean(1)            # [M]
    keep = torch.where(matrix > matrix.mean())[0]
    if keep.numel() == 0:
        # Above-mean can be empty when every rater scores identically.
        keep = torch.arange(matrix.shape[0], device=matrix.device, dtype=torch.long)
    return keep


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather along the token dim: [B, N, C] x [B, S] -> [B, S, C]."""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def _pairwise_token_distances(
    x: torch.Tensor, max_working_bytes: int = 256 * 1024 * 1024,
) -> torch.Tensor:
    """Compute the original subtraction/norm in tiles, bounding its 4-D temporary."""
    B, N, C = x.shape
    bytes_per_pair = B * C * x.element_size()
    if max_working_bytes < bytes_per_pair:
        raise ValueError("distance workspace must hold at least one token pair")
    # The HallusionBench failure allocated 159 GiB for [B,N,N,C]. Keep the same
    # dtype and reduction, with a 256 MiB workspace instead of changing distances
    # through a matmul/cdist identity (which can change low-precision center ties).
    pair_budget = max_working_bytes // bytes_per_pair
    row_chunk = min(N, max(1, pair_budget // N))
    col_chunk = min(N, max(1, pair_budget // row_chunk))
    distance = x.new_empty((B, N, N))
    for row in range(0, N, row_chunk):
        for col in range(0, N, col_chunk):
            distance[:, row:row + row_chunk, col:col + col_chunk] = (
                x[:, row:row + row_chunk, None, :]
                - x[:, None, col:col + col_chunk, :]
            ).norm(dim=-1, p=2)
    return distance


def cluster_and_merge(x: torch.Tensor, cluster_num: int) -> torch.Tensor:
    """DPC-kmeans: pick `cluster_num` centers by density x distance, then average
    each cluster. Same algorithm as the official `cluster_and_merge`.

    Runs on GPU (the official code round-trips every token through CPU, which for
    a 1280-token span is pure overhead in an eval loop).
    """
    B, N, C = x.shape
    distance = _pairwise_token_distances(x)              # [B, N, N]
    dist_matrix = distance / (C ** 0.5)

    # local density from the nearest `k` neighbours
    k_nn = min(cluster_num, N)
    dist_nearest = torch.topk(dist_matrix, k=k_nn, dim=-1, largest=False).values
    density = (-(dist_nearest ** 2).mean(dim=-1)).exp()

    mask = (density[:, None, :] > density[:, :, None]).to(x.dtype)
    dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
    dist = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1).values

    score = dist * density
    index_down = torch.topk(score, k=cluster_num, dim=-1).indices   # [B, K]

    # index_points gathers along the *token* dim, so transpose the distance matrix
    # to [B, N, N] -> select rows: dist_matrix[b, index_down[b], :]
    dist_to_centers = index_points(dist_matrix.transpose(1, 2), index_down)   # [B, K, N]
    idx_cluster = dist_to_centers.argmin(dim=1)                     # [B, N]

    # a center must belong to its own cluster
    center_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
    center_slot = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
    idx_cluster[center_batch.reshape(-1), index_down.reshape(-1)] = center_slot.reshape(-1)

    # average the tokens assigned to each center: flat index (b, cluster) over B*cluster_num
    token_batch = torch.arange(B, device=x.device)[:, None].expand(B, N)
    idx = idx_cluster + token_batch * cluster_num                   # [B, N]
    x_flat = x.reshape(B * N, C)
    idx_flat = idx.reshape(B * N)
    counts = torch.zeros(B * cluster_num, device=x.device, dtype=x.dtype)
    counts.index_add_(0, idx_flat, torch.ones_like(idx_flat, dtype=x.dtype))
    merged = torch.zeros(B * cluster_num, C, device=x.device, dtype=x.dtype)
    merged.index_add_(0, idx_flat, x_flat)
    merged = merged / counts.clamp(min=1).unsqueeze(-1)
    return merged.reshape(B, cluster_num, C)


def merge_and_retain(
    visual_tokens: torch.Tensor,
    attn_weights: torch.Tensor,
    keep: int,
    recycle_ratio: float = 0.3,
    cluster_ratio: float = 0.1,
) -> tuple:
    """SparseVLM's retained-token + recycled-centroid construction.

    Args:
        visual_tokens: [B, V, C] the current visual span.
        attn_weights:  [B, V] text->visual attention used as the relevance score.
        keep: how many tokens to keep outright.
        recycle_ratio: fraction of the dropped tokens fed into merging.
        cluster_ratio: fraction of those that survive as centroids.

    Returns:
        (kept_tokens [B, keep', C], merged_tokens [B, K, C], keep_idx [B, keep'])
        where keep' == min(keep, V-1) to leave at least one token to recycle.
    """
    B, V, C = visual_tokens.shape
    # The official code caps at V-1 so recycling always has a source.
    keep = int(max(1, min(keep, V - 1)))
    keep_idx = torch.topk(attn_weights, keep, dim=1).indices          # [B, keep]

    keep_mask = torch.zeros_like(attn_weights, dtype=torch.bool)
    keep_mask.scatter_(1, keep_idx, True)

    dropped_tokens = visual_tokens[~keep_mask].reshape(B, -1, C)
    dropped_attn = attn_weights[~keep_mask].reshape(B, -1)

    if dropped_tokens.shape[1] == 0:
        empty = visual_tokens.new_zeros((B, 0, C))
        return visual_tokens.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, C)), empty, keep_idx

    merge_num = max(int(dropped_attn.shape[1] * recycle_ratio), 1)
    merge_token_index = dropped_attn.topk(merge_num, dim=1).indices   # [B, merge_num]
    merge_visual_tokens = dropped_tokens.gather(1, merge_token_index.unsqueeze(-1).expand(-1, -1, C))

    cluster_num = max(int(merge_visual_tokens.shape[1] * cluster_ratio), 1)
    merged_tokens = cluster_and_merge(merge_visual_tokens, cluster_num)

    keep_idx = torch.sort(keep_idx, dim=1).values
    kept_tokens = visual_tokens.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, C))
    return kept_tokens, merged_tokens, keep_idx
