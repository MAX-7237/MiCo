# Shared S1 selector for attn_cov_vit_proj / attn_cov_vit_proj_llm_12.
# Copied from LLaVA modeling_llama_attn_cov_llm.inner_weighted_raw_cls_select
# so Qwen2.5-VL and InternVL3 use the same greedy, not a second implementation.

import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# Stage-1 budget multiplier for the attn_H_cov_vit_proj_llm_* arms.
# 2 = the published configuration (S1 keeps 2T, S2 cuts at layer K so the
# layer-average is T). The S1 ablation sweeps this to 3/4/5 while S2 keeps
# hitting the same average-T identity: T*L = K*(m*T) + (L-K)*R.
# Kept as an env knob on purpose: the method name's trailing integer is the
# S2 layer K (parsed with rsplit('_', 1)), so encoding m in the name would
# collide with that parse.
S1_MULT_ENV = "H_VIT_S1_MULT"
S1_MULT_ENV_DEFAULT = 2


def attn_h_s1_mult() -> int:
    """Stage-1 keep multiplier m (S1 keeps m*T). Defaults to 2."""
    raw = os.environ.get(S1_MULT_ENV)
    if raw is None or str(raw).strip() == "":
        return S1_MULT_ENV_DEFAULT
    try:
        mult = int(str(raw).strip())
    except ValueError:
        raise ValueError(f"{S1_MULT_ENV} must be an integer, got {raw!r}")
    if mult < 1:
        raise ValueError(f"{S1_MULT_ENV} must be >= 1, got {mult}")
    return mult


def attn_h_s2_keep(t_avg: int, n_layers: int, k_layer: int, s1_mult: int) -> int:
    """Post-prune keep R for an S1 budget of m*T at layer K.

    Same identity as fastv_keep_after_k, generalised over the S1 multiplier:
    T*L = K*(m*T) + (L-K)*R  =>  R = T*(L - K*m) / (L - K).

    R > 0 (Stage 2 can still cut) iff K*m < L. Both Qwen2.5-VL-7B and
    InternVL3-8B are L=28: K=2 admits m<=13, K=4 admits m<=6, so the
    m in {3,4,5} ablation is feasible on both.
    """
    n_enter = int(s1_mult) * int(t_avg)
    if n_layers <= k_layer:
        return max(1, min(int(t_avg), n_enter))
    r = (float(t_avg) * n_layers - float(k_layer) * n_enter) / float(n_layers - k_layer)
    return max(1, min(int(round(r)), n_enter))


def _comp_on(name: str) -> bool:
    """Component switch for the dual-stage ablation (default ON = unchanged).

    H_VIT_NO_H / H_VIT_NO_COV / H_VIT_NO_ATTN each set to 1 turns OFF that
    component in BOTH stages. Turning a component off means the selector
    degrades to pure top-k on whatever signal is left (see callers):
      no-attention -> relevance is constant, so selection is coverage-only
      no-coverage  -> no similarity structure, so selection is top-k on relevance
      no-H         -> drop the |h| multiplicand
    """
    return os.environ.get(name, "0").strip() not in ("1", "true", "True", "yes")


def s1_components() -> Tuple[bool, bool, bool]:
    """(use_attn, use_h, use_cov) for Stage 1."""
    return (_comp_on("H_VIT_NO_ATTN"), _comp_on("H_VIT_NO_H"), _comp_on("H_VIT_NO_COV"))


def inner_weighted_raw_cls_select(
    features: torch.Tensor,
    raw_cls_attn: torch.Tensor,
    keep: int,
    attn_h: bool = False,
) -> torch.Tensor:
    """Greedy keep-`keep` by sum_v w_v * max(0, kappa(v,i) - m_v).

    w = raw CLS (clamp ≥0, no min-max). kappa is nonnegative cosine of
    `features`. Returns a boolean mask of length N.

    With H_VIT_NO_COV=1 the coverage greedy is skipped and this degrades to a
    plain top-k over w; with H_VIT_NO_ATTN=1 w is constant, so the ranking is
    driven purely by coverage gain.
    """
    use_attn, use_h, use_cov = s1_components()
    n_tokens = features.shape[0]
    keep = int(max(1, min(int(keep), n_tokens)))
    feats = F.normalize(features.float(), dim=-1)
    sim = (feats @ feats.t()).clamp(min=0.0)
    w = raw_cls_attn.float().clamp(min=0.0)
    if not use_attn:
        w = torch.ones_like(w)
    if attn_h and use_h:
        # H is measured before projection, on exactly the features used for
        # coverage; using projected features here would change the S1 ablation.
        h = features.detach().float().norm(dim=-1)
        if h.shape != w.shape:
            raise ValueError(f"S1 H shape {tuple(h.shape)} != attention shape {tuple(w.shape)}")
        w = w * h
    device = features.device
    if not use_cov:
        # Coverage ablated: rank by the remaining relevance signal alone.
        idx = torch.topk(w, keep).indices
        mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)
        mask[idx] = True
        return mask
    available = torch.ones(n_tokens, dtype=torch.bool, device=device)
    max_cov = torch.zeros(n_tokens, device=device, dtype=sim.dtype)
    selected = []
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


def fastv_keep_after_k(t_avg: int, n_layers: int, k_layer: int, n_enter: int) -> int:
    """Post-prune keep R so the layer-average is T.

    Same identity as LLaVA dynamic_R: T*L = K*N + (L-K)*R.
    Qwen2.5-VL-7B / InternVL3-8B are 28 layers; TOPS first locus is K=12,
    so with N=2T this is R=0.25T. Do not use the TOPS table keep (71 at
    T=128): that budget is for the two-stage 12+16 schedule.
    """
    if n_layers <= k_layer:
        return max(1, min(int(t_avg), n_enter))
    r = (float(t_avg) * n_layers - float(k_layer) * n_enter) / float(n_layers - k_layer)
    return max(1, min(int(round(r)), n_enter))


def tops_first_prune_layer(
    schedule_by_t: Dict[int, List[Tuple[int, int]]],
    t: int,
    default: int = 12,
) -> int:
    """1-indexed first TOPS two-stage LLM prune layer.

    Both Qwen2.5-VL-7B and InternVL3-8B TOPS_SCHEDULE start at 12 for every T.
    Fall back to any scheduled T, then 12, so T=64/32 (not in the table) still
    prune at the same physical layer as TOPS stage-2 first cut.
    """
    pairs = schedule_by_t.get(int(t))
    if pairs:
        return int(pairs[0][0])
    for extra in schedule_by_t.values():
        if extra:
            return int(extra[0][0])
    return int(default)


def llm12_pruning_layers(
    schedule_by_t: Dict[int, List[Tuple[int, int]]],
    t: int,
    n_layers: int,
) -> Dict[int, int]:
    """Single S2 cut: TOPS first layer, FastV-style keep for incoming 2T."""
    k = tops_first_prune_layer(schedule_by_t, t)
    r = fastv_keep_after_k(int(t), int(n_layers), k, 2 * int(t))
    return {k: r}
