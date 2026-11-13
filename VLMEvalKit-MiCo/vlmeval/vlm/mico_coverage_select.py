import os
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

S1_MULT_ENV = "MICO_STAGE1_MULT"
S1_MULT_ENV_DEFAULT = 2


def mico_s1_mult() -> int:
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


def mico_s2_keep(t_avg: int, n_layers: int, k_layer: int, s1_mult: int) -> int:
    """Post-prune keep R for an S1 budget of m*T at layer K.

    Same identity as keep_after_layer, generalised over the S1 multiplier:
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

    MICO_NO_NORM / MICO_NO_COVERAGE / MICO_NO_ATTENTION each set to 1 turns OFF that
    component in BOTH stages. Turning a component off means the selector
    degrades to pure top-k on whatever signal is left (see callers):
      no-attention -> relevance is constant, so selection is coverage-only
      no-coverage  -> no similarity structure, so selection is top-k on relevance
      no-H         -> drop the |h| multiplicand
    """
    return os.environ.get(name, "0").strip() not in ("1", "true", "True", "yes")


def s1_components() -> Tuple[bool, bool, bool]:
    """(use_attn, use_h, use_cov) for Stage 1."""
    return (_comp_on("MICO_NO_ATTENTION"), _comp_on("MICO_NO_NORM"), _comp_on("MICO_NO_COVERAGE"))


def inner_weighted_raw_cls_select(
    features: torch.Tensor, raw_cls_attn: torch.Tensor, keep: int, mico: bool = False
) -> torch.Tensor:
    """Greedy keep-`keep` by sum_v w_v * max(0, kappa(v,i) - m_v).

    w = raw CLS (clamp ≥0, no min-max). kappa is nonnegative cosine of
    `features`. Returns a boolean mask of length N.

    With MICO_NO_COVERAGE=1 the coverage greedy is skipped and this degrades to a
    plain top-k over w; with MICO_NO_ATTENTION=1 w is constant, so the ranking is
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
    if mico and use_h:
        h = features.detach().float().norm(dim=-1)
        if h.shape != w.shape:
            raise ValueError(f"S1 H shape {tuple(h.shape)} != attention shape {tuple(w.shape)}")
        w = w * h
    device = features.device
    if not use_cov:
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
