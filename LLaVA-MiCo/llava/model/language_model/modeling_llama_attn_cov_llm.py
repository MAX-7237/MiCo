"""Attn-Cov-LLM: FastV-layer coverage pruning with post-visual text attention.

Prune at the same LLM layer as FastV (default K=2). Importance is the greedy
weighted coverage objective with inner product weights:

    Delta(i) = sum_v text_attn[v] * max(0, kappa(v, i) - m_v)

where text_attn averages attention from all tokens AFTER the visual span
(sys prompt cannot attend to visuals under the causal mask), and kappa is
nonnegative cosine similarity of layer-K visual hidden states.

Also used by merge_attn_cov_llm: after mm_projector, keep 2T CLS-pivots
and mean-merge leftovers onto the nearest pivot, then FastV-layer coverage
prune so the layer-average budget is T.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    Cache,
    DynamicCache,
    LlamaModel,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)


def merge_to_cls_pivots(
    features: torch.Tensor,
    cls_attn: torch.Tensor,
    n_pivots: int,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Keep top-n_pivots CLS tokens; merge the rest onto the nearest pivot.

    Each leftover token is assigned to the pivot with highest cosine similarity.
    Pivot features become the mean of the pivot plus its assigned members.

    Returns:
        out_features: (N, D) with pivot slots overwritten by cluster means.
        keep_mask: (N,) True on the  n_pivots survivors.
        stats: merge debug counts.
    """
    n_tokens, dim = features.shape
    device = features.device
    dtype = features.dtype
    n_pivots = int(max(1, min(int(n_pivots), n_tokens)))
    pivot_idx = cls_attn.float().topk(n_pivots, dim=0).indices
    keep = torch.zeros(n_tokens, dtype=torch.bool, device=device)
    keep[pivot_idx] = True
    out = features.clone()
    rest_idx = (~keep).nonzero(as_tuple=False).squeeze(-1)
    feats_f = features.float()
    mean_merge = 1.0
    if rest_idx.numel() > 0:
        pivots = feats_f[pivot_idx]
        rest = feats_f[rest_idx]
        assign = (F.normalize(rest, dim=-1) @ F.normalize(pivots, dim=-1).t()).argmax(dim=1)
        merged = pivots.clone()
        count = torch.ones(n_pivots, device=device, dtype=merged.dtype)
        merged.scatter_add_(0, assign.unsqueeze(-1).expand(-1, dim), rest)
        count.scatter_add_(0, assign, torch.ones(assign.shape[0], device=device, dtype=count.dtype))
        merged = merged / count.unsqueeze(-1)
        out[pivot_idx] = merged.to(dtype)
        mean_merge = float(count.mean().item())
    stats = {
        "n_before": n_tokens,
        "n_pivots": n_pivots,
        "n_rest": int(rest_idx.numel()),
        "n_after": n_pivots,
        "mean_merge_count": mean_merge,
    }
    return out, keep, stats


def compute_keep_from_avg(
    t_avg: int,
    n_enter: int,
    k_layer: int,
    n_layers: int,
) -> int:
    """Solve FastV-style average budget for post-layer-K keep count R.

        T_avg * L = K * N + (L - K) * R
        R = (T_avg * L - K * N) / (L - K)
    """
    if n_layers <= k_layer:
        return max(1, min(int(t_avg), n_enter))
    r = (float(t_avg) * n_layers - float(k_layer) * n_enter) / float(n_layers - k_layer)
    r = int(round(r))
    return max(1, min(r, n_enter))


def inner_weighted_raw_cls_select(
    features: torch.Tensor,
    raw_cls_attn: torch.Tensor,
    keep: int,
    coverage_only: bool = False,
    attn_h: bool = False,
) -> torch.Tensor:
    """Greedy keep-`keep` by sum_v w_v * max(0, kappa(v,i) - m_v).

    Default w = raw CLS (no min-max). coverage_only: w=1. attn_h: multiply
    by ||h||_2 of `features` (the hidden state at this prune site).
    kappa is nonnegative cosine of `features`.
    Returns a boolean mask of length N.
    """
    n_tokens = features.shape[0]
    keep = int(max(1, min(int(keep), n_tokens)))
    feats = F.normalize(features.float(), dim=-1)
    sim = (feats @ feats.t()).clamp(min=0.0)
    if coverage_only:
        w = torch.ones(n_tokens, device=features.device, dtype=torch.float32)
    else:
        w = raw_cls_attn.float().clamp(min=0.0)
        if attn_h:
            # S1 prune site is CLIP visual tokens; |h| = their L2.
            w = w * features.float().norm(dim=-1).clamp(min=0.0)
    device = features.device
    available = torch.ones(n_tokens, dtype=torch.bool, device=device)
    max_cov = torch.zeros(n_tokens, device=device, dtype=sim.dtype)
    selected: List[int] = []
    seed = int(torch.argmax((w.unsqueeze(0) * sim).sum(dim=1)).item())
    selected.append(seed)
    available[seed] = False
    max_cov = torch.maximum(max_cov, sim[seed])
    for _ in range(keep - 1):
        gain = (w.unsqueeze(0) * (sim - max_cov.unsqueeze(0)).clamp(min=0.0)).sum(dim=1)
        gain = gain.masked_fill(~available, float("-inf"))
        idx = int(torch.argmax(gain).item())
        selected.append(idx)
        available[idx] = False
        max_cov = torch.maximum(max_cov, sim[idx])
    mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)
    mask[torch.tensor(selected, device=device, dtype=torch.long)] = True
    return mask


# 4-region S1 (attn_cov_proj_llm_4regions_sg). Total keep stays 2T so the
# comparison with attn_cov_proj_llm_2 is only "where the 2T sit", not FLOPs.
_REGION_GRID = 2
_REGION_G = _REGION_GRID * _REGION_GRID
_REGION_LAMBDA = 0.5  # half uniform; λ=1 lets one high-s_g quadrant dominate
_REGION_BETA = 1.0  # s_g is already a token-mass scale; no extra temperature
_REGION_MAX_FRAC = 0.5  # 4-region cap: no quadrant gets more than half of 2T
_REGION_MIN_DENOM = 3  # k_min = max(2, floor(2T / (3G)))
# Entropy-gated S1 size: N1=(1+α)T with α in [0.5, 1] so K=2 still prunes
# and the pool never exceeds the 2T arm already on the table.
_ENTROPY_ALPHA_MIN = 0.5
_ENTROPY_ALPHA_MAX = 1.0
# prefixvlm_2 S2 keep table (exact post-layer count, not FastV dynamic_R).
# 7B uses 12/24; 13B uses 15/30. T=192 on 7B is a 3-locus schedule and is
# not in this table — that budget is not comparable to the 12/24 arm.
_PREFIX2_KEEP = {
    "7b": {
        12: {256: 128, 128: 64, 64: 32, 32: 16},
        24: {256: 64, 128: 32, 64: 16, 32: 8},
    },
    "13b": {
        15: {192: 128, 128: 64, 64: 32, 32: 16},
        30: {192: 64, 128: 32, 64: 16, 32: 8},
    },
}
_PREFIX2_LOCS = {"7b": [12, 24], "13b": [15, 30]}


def cls_attn_entropy_norm(raw_cls_attn: torch.Tensor) -> float:
    """Normalized Shannon entropy of raw CLS attention, in [0, 1].

    High H is treated as a more complex image (user-specified). Zero-sum
    attention is mapped to 0.5 so we neither starve nor max the pool.
    """
    w = raw_cls_attn.float().clamp(min=0.0)
    total = float(w.sum().item())
    n = int(w.numel())
    if n <= 1 or total <= 0.0:
        return 0.5
    p = w / total
    p = p[p > 0]
    h = float(-(p * p.log()).sum().item())
    return max(0.0, min(1.0, h / math.log(n)))


def n1_from_cls_entropy(t: int, h_norm: float, n_tokens: int) -> int:
    """Map H_norm to S1 keep count in [1.5T, 2T], clipped to N."""
    alpha = _ENTROPY_ALPHA_MIN + (_ENTROPY_ALPHA_MAX - _ENTROPY_ALPHA_MIN) * float(h_norm)
    n1 = int(round((1.0 + alpha) * float(t)))
    lo = int(round(1.5 * float(t)))
    hi = int(round(2.0 * float(t)))
    return max(1, min(int(n_tokens), max(lo, min(hi, n1))))


def prefix2_keep_count(scale: str, t_budget: int, layer_1b: int, anyres: bool) -> int:
    """prefixvlm_2 keep at a 1-based prune layer. anyres multiplies by 5.

    Table keys are nominal T (128/64/32). NeXT allbench already passes the
    5× budget (640/320/160); divide by 5 before lookup so anyres does not
    double-count. LLaVA-1.5 pad (anyres=False) keeps looking up T as-is.
    """
    t_key = int(t_budget)
    try:
        table = _PREFIX2_KEEP[scale][int(layer_1b)]
    except KeyError as exc:
        raise KeyError(
            f"prefixvlm_2 keep missing for scale={scale} T={t_budget} L{layer_1b}"
        ) from exc
    if t_key not in table and anyres and t_key % 5 == 0 and (t_key // 5) in table:
        t_key = t_key // 5
    try:
        keep = int(table[t_key])
    except KeyError as exc:
        raise KeyError(
            f"prefixvlm_2 keep missing for scale={scale} T={t_budget} L{layer_1b}"
        ) from exc
    if anyres:
        keep *= 5
    return max(1, keep)


def region_ids_2x2(n_tokens: int, device: torch.device) -> Optional[torch.Tensor]:
    """Row-major 2×2 quadrants on a square grid. None if N is not a square."""
    side = int(round(math.sqrt(n_tokens)))
    if side * side != n_tokens:
        return None
    idx = torch.arange(n_tokens, device=device)
    row = torch.div(idx, side, rounding_mode="floor")
    col = idx % side
    half = side // 2
    return (row >= half).to(torch.long) * 2 + (col >= half).to(torch.long)


def _region_sg_after_one_pick(
    sim: torch.Tensor,
    w: torch.Tensor,
    region_ids: torch.Tensor,
) -> torch.Tensor:
    """Leftover CLS-weighted mass in each quadrant after one in-region pick.

    s_g = sum_{v in g} w_v * (1 - κ(v, i*_g))_+. One prototype that already
    covers the cell leaves little residual; a diverse cell does not.
    Shannon entropy of CLS attention is not used: flat background would
    then get the larger quota.
    """
    s = torch.zeros(_REGION_G, device=sim.device, dtype=sim.dtype)
    for g in range(_REGION_G):
        idx = (region_ids == g).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        sim_gg = sim.index_select(0, idx).index_select(1, idx)
        w_g = w.index_select(0, idx)
        seed = int(torch.argmax((w_g.unsqueeze(0) * sim_gg).sum(dim=1)).item())
        leftover = (w_g * (1.0 - sim_gg[seed]).clamp(min=0.0)).sum()
        s[g] = leftover
    return s


def _quotas_from_sg(
    s_g: torch.Tensor,
    keep: int,
    region_sizes: List[int],
) -> List[int]:
    """k_g = 2T * [(1-λ)/G + λ softmax(β s_g)], then clamp and fix the sum."""
    p = torch.softmax(_REGION_BETA * s_g, dim=0)
    mix = (1.0 - _REGION_LAMBDA) / float(_REGION_G) + _REGION_LAMBDA * p
    k_float = keep * mix
    k_min = max(2, keep // (_REGION_MIN_DENOM * _REGION_G))
    k_max = max(1, int(_REGION_MAX_FRAC * keep))
    k = []
    lo = []
    hi = []
    for g in range(_REGION_G):
        lo_g = min(k_min, region_sizes[g])
        hi_g = min(k_max, region_sizes[g])
        if hi_g < lo_g:
            lo_g = hi_g
        lo.append(lo_g)
        hi.append(hi_g)
        kg = int(round(float(k_float[g].item())))
        k.append(min(hi_g, max(lo_g, kg)))
    diff = keep - sum(k)
    order = torch.argsort(s_g, descending=True).tolist()
    # Remainder goes to the highest-s_g region that still has room, as agreed.
    guard = 0
    while diff != 0 and guard < keep + _REGION_G:
        guard += 1
        moved = False
        walk = order if diff > 0 else list(reversed(order))
        for g in walk:
            if diff > 0 and k[g] < hi[g]:
                k[g] += 1
                diff -= 1
                moved = True
                break
            if diff < 0 and k[g] > lo[g]:
                k[g] -= 1
                diff += 1
                moved = True
                break
        if not moved:
            break
    return k


def inner_weighted_raw_cls_select_4regions_sg(
    features: torch.Tensor,
    raw_cls_attn: torch.Tensor,
    keep: int,
) -> Tuple[torch.Tensor, dict]:
    """Global CLS×coverage greedy with per-quadrant remaining quotas.

    Independent per-region greeds would pick near-duplicates on the 2×2
    borders; one shared max_cov avoids that. Non-square N falls back to
    unconstrained CLS×coverage so we do not invent a layout.
    """
    n_tokens = features.shape[0]
    keep = int(max(1, min(int(keep), n_tokens)))
    device = features.device
    region_ids = region_ids_2x2(n_tokens, device)
    if region_ids is None:
        mask = inner_weighted_raw_cls_select(features, raw_cls_attn, keep)
        return mask, {"fallback": "nonsquare", "k": [keep], "s_g": []}

    feats = F.normalize(features.float(), dim=-1)
    sim = (feats @ feats.t()).clamp(min=0.0)
    w = raw_cls_attn.float().clamp(min=0.0)
    sizes = [(region_ids == g).sum().item() for g in range(_REGION_G)]
    s_g = _region_sg_after_one_pick(sim, w, region_ids)
    quotas = _quotas_from_sg(s_g, keep, sizes)
    remaining = torch.tensor(quotas, device=device, dtype=torch.long)

    available = torch.ones(n_tokens, dtype=torch.bool, device=device)
    max_cov = torch.zeros(n_tokens, device=device, dtype=sim.dtype)
    selected: List[int] = []
    blocked = remaining[region_ids] <= 0
    gain0 = (w.unsqueeze(0) * sim).sum(dim=1).masked_fill(blocked, float("-inf"))
    seed = int(torch.argmax(gain0).item())
    selected.append(seed)
    available[seed] = False
    remaining[int(region_ids[seed].item())] -= 1
    max_cov = torch.maximum(max_cov, sim[seed])
    for _ in range(keep - 1):
        gain = (w.unsqueeze(0) * (sim - max_cov.unsqueeze(0)).clamp(min=0.0)).sum(dim=1)
        blocked = (~available) | (remaining[region_ids] <= 0)
        gain = gain.masked_fill(blocked, float("-inf"))
        if not torch.isfinite(gain).any():
            break
        idx = int(torch.argmax(gain).item())
        selected.append(idx)
        available[idx] = False
        remaining[int(region_ids[idx].item())] -= 1
        max_cov = torch.maximum(max_cov, sim[idx])
    mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)
    mask[torch.tensor(selected, device=device, dtype=torch.long)] = True
    stats = {
        "fallback": "",
        "k": quotas,
        "s_g": [float(x) for x in s_g.detach().cpu().tolist()],
        "n_keep": int(mask.sum().item()),
    }
    return mask, stats


def r6_from_r2(t_avg: int, n_enter: int, r2: int, k1: int, k2: int, n_layers: int) -> int:
    """Keep count after layer k2 so layer-average is T_avg.

        T*L = k1*N + (k2-k1)*R2 + (L-k2)*R6
    """
    denom = n_layers - k2
    if denom <= 0:
        return max(1, min(int(r2), n_enter))
    r6 = (float(t_avg) * n_layers - float(k1) * n_enter - float(k2 - k1) * r2) / float(denom)
    r6 = int(round(r6))
    return max(1, min(r6, int(r2)))


def r2_from_text_attn_entropy(
    text_attn: torch.Tensor,
    t_avg: int,
    n_enter: int,
    k1: int,
    k2: int,
    n_layers: int,
    front_min: float = 1.2,
) -> Tuple[int, float, int, int]:
    """Map normalized text-attn entropy in [0,1] to R2 in a feasible band.

    High entropy (text looks at many visual tokens) -> larger R2.
    Low entropy (attention peaked) -> smaller R2, still >= 1.2 T when feasible.
    """
    n_enter = int(n_enter)
    t_avg = int(t_avg)
    budget = float(t_avg) * n_layers - float(k1) * n_enter
    rmin = int(math.ceil(budget / float(n_layers - k1))) if n_layers > k1 else 1
    rmax = int(math.floor((budget - float(n_layers - k2)) / float(k2 - k1))) if k2 > k1 else n_enter
    rmin = max(1, rmin)
    rmax = min(n_enter, rmax)
    front = int(round(front_min * t_avg))
    if rmin <= rmax:
        rmin = min(max(rmin, front), rmax)
    else:
        rmin = rmax = max(1, min(front, n_enter))

    w = text_attn.float().clamp(min=0.0)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    n = max(int(w.numel()), 1)
    if n <= 1:
        h = 0.0
    else:
        ent = float((-(w * (w + 1e-12).log())).sum().item())
        h = min(1.0, max(0.0, ent / math.log(n)))
    r2 = int(round(rmin + h * (rmax - rmin)))
    r2 = max(rmin, min(r2, rmax))
    return r2, h, rmin, rmax


def post_visual_text_attn(
    last_attention: torch.Tensor,
    seq_len: int,
    system_prompt_length: int,
    visual_token_length: int,
) -> torch.Tensor:
    """Mean post-visual text attention onto the visual span, L1-normalized."""
    vis_start = system_prompt_length
    vis_end = system_prompt_length + visual_token_length
    attn = last_attention.mean(dim=1)[0]
    if vis_end < seq_len:
        text_attn = attn[vis_end:, vis_start:vis_end].mean(dim=0)
    else:
        text_attn = attn[-1, vis_start:vis_end]
    text_attn = text_attn.float().clamp(min=0.0)
    weight_sum = text_attn.sum()
    if weight_sum > 0:
        text_attn = text_attn / weight_sum
    else:
        text_attn = torch.full_like(text_attn, 1.0 / max(visual_token_length, 1))
    return text_attn


def attn_cov_llm_select(
    last_attention: torch.Tensor,
    hidden_states: torch.Tensor,
    system_prompt_length: int,
    visual_token_length: int,
    keep: int,
    coverage_only: bool = False,
    attn_h: bool = False,
    sim_cov: bool = False,
    anti_sim_cov: bool = False,
    last_token: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedy select visual tokens by weighted coverage.

    attn_cov_llm: w_v = mean post-visual text attention (L1-normalized).
    last_token: w_v = last-query-token attention onto visuals (L1-normalized).
        Used by last_attn_cov_proj_llm_2; FastV-style single query, not mean text.
    cov_llm (coverage_only): w_v = 1/V.
    attn_h_cov_llm: w_v = text_attn[v] * ||h_v||_2, then L1-normalize.
    sim_cov_llm: w_v = sum over ALL non-visual text tokens of
        [cos(h_t, h_v)]_+, then L1-normalize. Not attention: hidden-state
        cosine, including the system prompt (user: all_text_to_visual).
    anti_sim_cov_llm: same tokens and same nonnegative cosine, but
        w_v = sum_t (1 - [cos(h_t, h_v)]_+). Ablation of sim vs 1-sim.
    Kappa is nonnegative cosine of layer-K visual hidden states.

    Returns:
        visual_token_index: sorted (R,) indices into the visual span.
        text_attn: (V,) weights actually used.
    """
    vis_start = system_prompt_length
    vis_end = system_prompt_length + visual_token_length
    seq_len = hidden_states.shape[1]
    keep = int(min(max(keep, 1), visual_token_length))
    feats_raw = hidden_states[0, vis_start:vis_end].float()

    if coverage_only:
        # Uniform facility-location coverage; no text/CLS attention.
        text_attn = torch.full(
            (visual_token_length,),
            1.0 / max(visual_token_length, 1),
            device=hidden_states.device,
            dtype=torch.float32,
        )
    elif sim_cov or anti_sim_cov:
        # w_v = sum_t f(cos) over every non-visual token t.
        # sim_cov: f = [cos]_+ ; anti_sim_cov: f = 1 - [cos]_+.
        # Same cosine as sim_cov so this is a paired ablation, not a new sim.
        pre = hidden_states[0, :vis_start].float()
        post = hidden_states[0, vis_end:].float() if vis_end < seq_len else feats_raw.new_zeros((0, feats_raw.shape[-1]))
        text_h = torch.cat([pre, post], dim=0) if post.numel() else pre
        if text_h.numel() == 0:
            text_attn = torch.full(
                (visual_token_length,),
                1.0 / max(visual_token_length, 1),
                device=hidden_states.device,
                dtype=torch.float32,
            )
        else:
            t_n = F.normalize(text_h, dim=-1)
            v_n = F.normalize(feats_raw, dim=-1)
            sim_tv = (t_n @ v_n.t()).clamp(min=0.0)
            text_attn = ((1.0 - sim_tv) if anti_sim_cov else sim_tv).sum(dim=0)
            weight_sum = text_attn.sum()
            if weight_sum > 0:
                text_attn = text_attn / weight_sum
            else:
                text_attn = torch.full_like(text_attn, 1.0 / visual_token_length)
    elif last_token:
        # Last prefill query only (the token that will emit the answer).
        attn = last_attention.mean(dim=1)[0]  # (Q, K)
        text_attn = attn[-1, vis_start:vis_end]
        text_attn = text_attn.float().clamp(min=0.0)
        weight_sum = text_attn.sum()
        if weight_sum > 0:
            text_attn = text_attn / weight_sum
        else:
            text_attn = torch.full_like(text_attn, 1.0 / visual_token_length)
    else:
        # Mean over heads, then mean over all query tokens AFTER visuals.
        # Sys-prompt queries cannot see visuals under the causal mask.
        attn = last_attention.mean(dim=1)[0]  # (Q, K)
        if vis_end < seq_len:
            text_attn = attn[vis_end:, vis_start:vis_end].mean(dim=0)  # (V,)
        else:
            # Degenerate: no post-visual text (should not happen in VQA prefill).
            text_attn = attn[-1, vis_start:vis_end]

        text_attn = text_attn.float().clamp(min=0.0)
        if attn_h:
            # |h_v| is the raw hidden L2 of visual token v (before cosine norm).
            text_attn = text_attn * feats_raw.norm(dim=-1).clamp(min=0.0)
        weight_sum = text_attn.sum()
        if weight_sum > 0:
            text_attn = text_attn / weight_sum
        else:
            text_attn = torch.full_like(text_attn, 1.0 / visual_token_length)

    feats = F.normalize(feats_raw, dim=-1)
    sim = (feats @ feats.t()).clamp(min=0.0)  # (V, V)

    device = hidden_states.device
    selected: List[int] = []
    max_sim_to_S = torch.zeros(visual_token_length, device=device, dtype=sim.dtype)
    available = torch.ones(visual_token_length, dtype=torch.bool, device=device)

    for _ in range(keep):
        # Inner-weighted marginal coverage: sum_v w_v * [kappa(v,i) - m_v]_+
        gain = (text_attn.unsqueeze(0) * (sim - max_sim_to_S.unsqueeze(0)).clamp(min=0.0)).sum(dim=1)
        gain = gain.masked_fill(~available, float("-inf"))
        idx = int(torch.argmax(gain).item())
        selected.append(idx)
        available[idx] = False
        max_sim_to_S = torch.maximum(max_sim_to_S, sim[idx])

    visual_token_index = torch.tensor(selected, device=device, dtype=torch.long)
    visual_token_index = torch.sort(visual_token_index).values
    return visual_token_index, text_attn.detach()


class AttnCovLLMLlamaModel(LlamaModel):
    """LlamaModel with Attn-Cov-LLM pruning at layer K (FastV location).

    Config keys:
      K: prune after this layer index (1-based FastV convention; default 2)
      R / T: fixed keep count when dynamic_R is False (attn_cov_llm)
      T_avg / T: target layer-average visual count when dynamic_R is True
      dynamic_R: if True, recompute R from T_avg, N, K, L each forward
      merge_threshold: stored for encode_images (merge_attn_cov_llm only)
    """

    def __init__(self, config: LlamaConfig, attn_cov_llm_config: Dict, **kwargs):
        super().__init__(config)
        self.system_prompt_length = 35
        self.visual_token_num = 0

        if config.num_hidden_layers == 32:
            self.scale = "7b"
        elif config.num_hidden_layers == 40:
            self.scale = "13b"

        if config.image_aspect_ratio == "pad":
            self.visual_token_length = 576
            self.anyres = False
        elif config.image_aspect_ratio == "anyres":
            self.visual_token_length = 2880
            self.anyres = True
        else:
            self.visual_token_length = 576
            self.anyres = False

        self.K = int(attn_cov_llm_config.get("K", 2))
        self.K2 = int(attn_cov_llm_config.get("K2", 0))
        prune_layers = attn_cov_llm_config.get("prune_layers")
        if prune_layers is not None:
            self.prune_layers = [int(x) for x in prune_layers]
        elif self.K2 > 0:
            self.prune_layers = [self.K, self.K2]
        else:
            self.prune_layers = [self.K]
        self.adaptive_r2 = bool(attn_cov_llm_config.get("adaptive_r2", False))
        self.dynamic_R = bool(attn_cov_llm_config.get("dynamic_R", False))
        # Exact prefixvlm_2 (L, T) keep, not layer-average R. Isolates "same
        # attn×coverage select, different prune locus / keep table".
        self.prefix2_keep = bool(attn_cov_llm_config.get("prefix2_keep", False))
        # Single-prune sibling of 12_24: same first locus (7B L12 / 13B L15),
        # but keep is FastV-style average T, not the prefix2 keep table.
        self.prefix2_first = bool(attn_cov_llm_config.get("prefix2_first", False))
        if self.prefix2_keep and self.prefix2_first:
            raise ValueError("prefix2_keep and prefix2_first are mutually exclusive")
        if self.prefix2_keep:
            if not hasattr(self, "scale"):
                raise ValueError("prefix2_keep needs 7b/13b layer count")
            self.prune_layers = list(_PREFIX2_LOCS[self.scale])
            self.K = self.prune_layers[0]
            self.K2 = self.prune_layers[1]
            self.dynamic_R = False
        elif self.prefix2_first:
            if not hasattr(self, "scale"):
                raise ValueError("prefix2_first needs 7b/13b layer count")
            self.K = _PREFIX2_LOCS[self.scale][0]
            self.K2 = 0
            self.prune_layers = [self.K]
        self.T_avg = int(
            attn_cov_llm_config.get(
                "T_avg",
                attn_cov_llm_config.get("T", attn_cov_llm_config.get("R", 128)),
            )
        )
        self.merge_threshold = float(attn_cov_llm_config.get("merge_threshold", 0.9))
        self.coverage_only = bool(attn_cov_llm_config.get("coverage_only", False))
        self.attn_h = bool(attn_cov_llm_config.get("attn_h", False))
        self.sim_cov = bool(attn_cov_llm_config.get("sim_cov", False))
        self.anti_sim_cov = bool(attn_cov_llm_config.get("anti_sim_cov", False))
        self.last_token = bool(attn_cov_llm_config.get("last_token", False))
        if "R" in attn_cov_llm_config:
            self.R = int(attn_cov_llm_config["R"])
        else:
            self.R = int(attn_cov_llm_config.get("T", self.T_avg))

    def update_runtime_visual_length(self, n_visual: int) -> int:
        """Set N entering the LLM (after merge) and refresh R if dynamic."""
        n_visual = int(max(1, n_visual))
        self.visual_token_length = n_visual
        if self.dynamic_R:
            self.R = compute_keep_from_avg(
                t_avg=self.T_avg,
                n_enter=n_visual,
                k_layer=self.K,
                n_layers=len(self.layers),
            )
        return self.R

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        if seq_length > 1:
            visual_token_length = self.visual_token_length
            visual_token_num = 0
            visual_token_list = []

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                if seq_length > 1:
                    visual_token_num += visual_token_length
                    visual_token_list.append(visual_token_length)

                    layer_1b = decoder_layer.self_attn.layer_idx + 1
                    if layer_1b in self.prune_layers:
                        attn_mask = torch.ones((batch_size, hidden_states.shape[1]), device=hidden_states.device)
                        attn_mask = _prepare_4d_causal_attention_mask(
                            attn_mask, (batch_size, hidden_states.shape[1]), hidden_states, 0
                        )
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=attn_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=True,
                            use_cache=use_cache,
                        )
                        hidden_states = layer_outputs[0]
                        last_attention = layer_outputs[1]

                        device = hidden_states.device
                        cur_seq = hidden_states.shape[1]
                        orig_visual_len = int(visual_token_length)
                        n_layers = len(self.layers)
                        n_enter = int(self.visual_token_length)

                        if self.adaptive_r2 and layer_1b == self.prune_layers[0]:
                            text_attn_w = post_visual_text_attn(
                                last_attention, cur_seq, self.system_prompt_length, orig_visual_len
                            )
                            keep_r, h_norm, rmin, rmax = r2_from_text_attn_entropy(
                                text_attn_w,
                                t_avg=self.T_avg,
                                n_enter=n_enter,
                                k1=self.prune_layers[0],
                                k2=self.prune_layers[1] if len(self.prune_layers) > 1 else self.K,
                                n_layers=n_layers,
                            )
                            self.R2 = keep_r
                            self._r2_entropy = h_norm
                            self._r2_band = (rmin, rmax)
                        elif self.adaptive_r2 and len(self.prune_layers) > 1 and layer_1b == self.prune_layers[1]:
                            keep_r = r6_from_r2(
                                t_avg=self.T_avg,
                                n_enter=n_enter,
                                r2=int(getattr(self, "R2", orig_visual_len)),
                                k1=self.prune_layers[0],
                                k2=self.prune_layers[1],
                                n_layers=n_layers,
                            )
                            self.R6 = keep_r
                        elif self.prefix2_keep:
                            keep_r = prefix2_keep_count(
                                self.scale, self.T_avg, layer_1b, self.anyres
                            )
                            keep_r = max(1, min(keep_r, orig_visual_len))
                            self.R = keep_r
                        elif self.dynamic_R:
                            keep_r = compute_keep_from_avg(
                                t_avg=self.T_avg,
                                n_enter=orig_visual_len,
                                k_layer=layer_1b,
                                n_layers=n_layers,
                            )
                            self.R = keep_r
                            if getattr(self, "prefix2_first", False) or self.K == 12:
                                log_i = getattr(self, "_attn_cov_12_log_i", 0)
                                enable_debug = os.environ.get("ENABLE_DEBUG", "0") == "1"
                                if log_i == 0 or (enable_debug and log_i < 8):
                                    print(
                                        f"[attn_cov_proj_llm_12] sample={log_i} "
                                        f"L{layer_1b} N={orig_visual_len} T={self.T_avg} "
                                        f"keep={keep_r} (S2=text_attn*coverage, "
                                        f"R=(T*L-K*2T)/(L-K))",
                                        flush=True,
                                    )
                                self._attn_cov_12_log_i = log_i + 1
                        else:
                            keep_r = int(self.R)

                        visual_token_index, text_attn = attn_cov_llm_select(
                            last_attention=last_attention,
                            hidden_states=hidden_states,
                            system_prompt_length=self.system_prompt_length,
                            visual_token_length=orig_visual_len,
                            keep=keep_r,
                            coverage_only=self.coverage_only,
                            attn_h=self.attn_h,
                            sim_cov=self.sim_cov,
                            anti_sim_cov=self.anti_sim_cov,
                            last_token=self.last_token,
                        )
                        visual_token_length = int(visual_token_index.numel())

                        self.metric_cache = {
                            "text_attn": text_attn.detach().cpu(),
                            "selected_indices": visual_token_index.detach().cpu(),
                            "layer": layer_1b,
                            "N_enter": n_enter,
                            "keep": visual_token_length,
                            "R2": getattr(self, "R2", None),
                            "R6": getattr(self, "R6", None),
                            "entropy": getattr(self, "_r2_entropy", None),
                            "T_avg_target": self.T_avg,
                        }

                        full_token_index = torch.cat(
                            (
                                torch.arange(self.system_prompt_length, device=device),
                                visual_token_index + self.system_prompt_length,
                                torch.arange(
                                    self.system_prompt_length + orig_visual_len, cur_seq, device=device
                                ),
                            )
                        )

                        hidden_states = hidden_states[:, full_token_index]
                        position_ids = full_token_index.unsqueeze(0)
                        layer_outputs = (hidden_states, layer_outputs[2])
                        if self.adaptive_r2:
                            log_i = getattr(self, "_attn_cov_2_log_i", 0)
                            enable_debug = os.environ.get("ENABLE_DEBUG", "0") == "1"
                            if layer_1b == self.prune_layers[0]:
                                if log_i == 0 or (enable_debug and log_i < 8):
                                    print(
                                        f"[attn_cov_llm_2] sample={log_i} prune=attn_cov "
                                        f"L{layer_1b} N={n_enter} T={self.T_avg} L_llm={n_layers} "
                                        f"entropy={float(getattr(self, '_r2_entropy', 0)):.4f} "
                                        f"R2_band=[{self._r2_band[0]},{self._r2_band[1]}] "
                                        f"R2={keep_r} drop={n_enter - keep_r} "
                                        f"(same attn*coverage select)",
                                        flush=True,
                                    )
                            elif len(self.prune_layers) > 1 and layer_1b == self.prune_layers[1]:
                                r2 = int(getattr(self, "R2", orig_visual_len))
                                r6 = int(keep_r)
                                k1, k2 = self.prune_layers[0], self.prune_layers[1]
                                realized = (
                                    k1 * n_enter + (k2 - k1) * r2 + (n_layers - k2) * r6
                                ) / float(n_layers)
                                if log_i == 0 or (enable_debug and log_i < 8):
                                    print(
                                        f"[attn_cov_llm_2] sample={log_i} prune=attn_cov "
                                        f"L{layer_1b} R2={r2} R6={r6} drop={r2 - r6} "
                                        f"realized_avg={realized:.2f} target_T={self.T_avg} "
                                        f"(same attn*coverage select)",
                                        flush=True,
                                    )
                                self._attn_cov_2_log_i = log_i + 1
                        elif self.prefix2_keep:
                            log_i = getattr(self, "_attn_cov_1224_log_i", 0)
                            enable_debug = os.environ.get("ENABLE_DEBUG", "0") == "1"
                            if log_i == 0 or (enable_debug and log_i < 8):
                                print(
                                    f"[attn_cov_proj_llm_12_24] sample={log_i} "
                                    f"L{layer_1b} N={orig_visual_len} T={self.T_avg} "
                                    f"keep={keep_r} loc={self.prune_layers} "
                                    f"(S2=text_attn*coverage, prefixvlm_2 keep table)",
                                    flush=True,
                                )
                            if layer_1b == self.prune_layers[-1]:
                                self._attn_cov_1224_log_i = log_i + 1

                    else:
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                        )

                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if seq_length > 1:
            self.visual_token_num = visual_token_num / len(self.layers)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
