# =============================================================================
# DIV-COV: Symmetric Cross-Set Coverage & Diversity Visual Token Pruning
# =============================================================================
#
# Mathematical Foundation:
# -------------------------
# Let V be the set of all N visual tokens.
# At any stage, let S \subset V be the set of already selected tokens,
# and U_S = V \setminus S be the remaining unselected tokens.
#
# 1. Cross-Set Coverage (F_C^cross):
#    F_C^cross(S) = \sum_{u \in U_S} w_u \max_{s \in S} \kappa(u, s)
#    where \kappa(u, s) \in [0, 1] is similarity with \kappa(v, v) = 1.
#    Let \bar F_C(S) = \sum_{v \in V} w_v \max_{s \in S} \kappa(v, s) be the standard
#    facility location function. Then:
#      \bar F_C(S) = F_C^cross(S) + \sum_{s \in S} w_s
#      F_C^cross(S) = \bar F_C(S) - \sum_{s \in S} w_s
#    For candidate i \in U_S, the marginal gain is:
#      \Delta_i F_C^cross(S) = C_i^{\neg i}(S) - w_i m_i(S)
#    where:
#      C_i^{\neg i}(S) = \sum_{u \in U_S \setminus \{i\}} w_u [\kappa(u, i) - m_u(S)]_+
#      m_u(S) = \max_{s \in S} \kappa(u, s)   (with m_u(\emptyset) = 0)
#
# 2. Cross-Set Diversity (F_D^cross):
#    F_D^cross(S) = \sum_{s \in S} w_s \max_{u \in V \setminus S} d(s, u)
#    where d(s, u) = 1 - \kappa(s, u) \in [0, 1] is the distance function.
#    Let R_S = U_S \setminus \{i\}. The marginal gain of adding i to S is:
#      \Delta_i F_D^cross(S) = M_i(S) - P_i(S)
#    where:
#      M_i(S) = \max_{u \in R_S} d(i, u)
#      P_i(S) = \sum_{s \in S} w_s [d(i, s) - \max_{u \in R_S} d(s, u)]_+
#
# 3. Unified Symmetric Objective:
#    F_CD(S) = \alpha F_C^cross(S) + \beta F_D^cross(S)
#    The exact greedy marginal selection criterion is:
#      \Delta_i F_CD(S) = \alpha \Delta_i F_C^cross(S) + \beta \Delta_i F_D^cross(S)
#
# Both F_C^cross and F_D^cross are submodular functions, making F_CD submodular.
# =============================================================================

import torch
from typing import Optional


def div_cov_select(
    features: torch.Tensor,
    target_token_num: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    token_weights: Optional[torch.Tensor] = None,
    metric: str = "cosine",
) -> torch.Tensor:
    """Select exactly target_token_num visual tokens using the DIV-COV criterion.

    Args:
        features: [N, D] visual token representations (e.g. post-projector, D=4096).
        target_token_num: Target number of tokens K to select (1 <= K <= N).
        alpha: Weight for cross-set coverage marginal gain (default: 1.0).
        beta: Weight for cross-set diversity marginal gain (default: 1.0).
        token_weights: Optional [N] positive weights w_v for each token.
                       If None, defaults to uniform weights (w_v = 1.0).
        metric: Distance metric; currently "cosine" (similarity in [0, 1]).

    Returns:
        torch.Tensor of shape [K] containing sorted indices of selected tokens.
    """
    if features.dim() != 2:
        raise ValueError(f"features must be 2D [N, D], got shape {tuple(features.shape)}")

    N, D = features.shape
    device = features.device
    target_token_num = max(0, min(int(target_token_num), N))

    if target_token_num == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if target_token_num >= N:
        return torch.arange(N, device=device, dtype=torch.long)

    # Work in float32 for stable distance & similarity calculations
    feat = features.float()
    norm = feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normed = feat / norm

    if metric == "cosine":
        # Cosine similarity clamped to [0, 1]
        K_mat = torch.clamp(normed @ normed.t(), 0.0, 1.0)
        D_mat = torch.clamp(1.0 - K_mat, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if token_weights is None:
        w = torch.ones(N, device=device, dtype=torch.float32)
    else:
        if token_weights.shape[0] != N:
            raise ValueError(f"token_weights must have length {N}, got {token_weights.shape[0]}")
        w = token_weights.to(device=device, dtype=torch.float32)

    S = []
    avail = torch.ones(N, dtype=torch.bool, device=device)
    m = torch.zeros(N, device=device, dtype=torch.float32)

    for step in range(target_token_num):
        # ---------------------------------------------------------------------
        # 1. Cross-Set Coverage Marginal Gain:
        #    \Delta_i F_C^cross(S) = \sum_{v \in V} w_v [\kappa(v, i) - m_v(S)]_+ - w_i
        #    Algebraically identical to:
        #      C_i^{\neg i}(S) - w_i m_i(S)
        # ---------------------------------------------------------------------
        if alpha != 0.0:
            G = torch.clamp(K_mat - m.unsqueeze(1), min=0.0)
            delta_C = (w.unsqueeze(1) * G).sum(dim=0) - w
        else:
            delta_C = torch.zeros(N, device=device, dtype=torch.float32)

        # ---------------------------------------------------------------------
        # 2. Cross-Set Diversity Marginal Gain:
        #    \Delta_i F_D^cross(S) = M_i(S) - P_i(S)
        #    where M_i(S) = \max_{u \in U_S \setminus \{i\}} d(i, u)
        #          P_i(S) = \sum_{s \in S} w_s [d(i, s) - \max_{u \in U_S \setminus \{i\}} d(s, u)]_+
        # ---------------------------------------------------------------------
        if beta != 0.0:
            n_unsel = int(avail.sum().item())
            if n_unsel <= 1:
                delta_D = torch.zeros(N, device=device, dtype=torch.float32)
            else:
                # For i \in U_S with |U_S| >= 2, since d(i, i) = 0 is minimal,
                # \max_{u \in U_S \setminus \{i\}} d(i, u) == \max_{u \in U_S} d(i, u)
                D_unsel_cols = D_mat[:, avail]
                M = D_unsel_cols.max(dim=1).values

                P = torch.zeros(N, device=device, dtype=torch.float32)
                if len(S) > 0:
                    S_tensor = torch.tensor(S, device=device, dtype=torch.long)
                    D_S_unsel = D_mat[S_tensor][:, avail]
                    if n_unsel == 1:
                        v1 = D_S_unsel.squeeze(1)
                        u_star = torch.where(avail)[0][0]
                        penalties = w[S_tensor] * v1
                        P[u_star] = penalties.sum()
                    else:
                        top2 = torch.topk(D_S_unsel, k=2, dim=1)
                        v1 = top2.values[:, 0]
                        v2 = top2.values[:, 1]
                        unsel_indices = torch.where(avail)[0]
                        u_star = unsel_indices[top2.indices[:, 0]]
                        penalties = w[S_tensor] * torch.clamp(v1 - v2, min=0.0)
                        P.scatter_add_(0, u_star, penalties)

                delta_D = M - P
        else:
            delta_D = torch.zeros(N, device=device, dtype=torch.float32)

        # ---------------------------------------------------------------------
        # 3. Total Combined Marginal Gain:
        #    \Delta_i F_CD(S) = \alpha \Delta_i F_C^cross(S) + \beta \Delta_i F_D^cross(S)
        # ---------------------------------------------------------------------
        score = alpha * delta_C + beta * delta_D
        score[~avail] = -float("inf")

        chosen = int(torch.argmax(score).item())
        S.append(chosen)
        avail[chosen] = False
        m = torch.maximum(m, K_mat[:, chosen])

    selected_tensor = torch.tensor(S, device=device, dtype=torch.long)
    return torch.sort(selected_tensor).values
