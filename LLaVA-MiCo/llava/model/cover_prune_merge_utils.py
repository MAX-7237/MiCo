"""Utilities for the COVER-Prune merge-first visual-token selector.

The selector intentionally operates in the pre-projector ViT feature space.
It first contracts the candidate set from ``N`` to ``ceil(sqrt(N * T))``
groups using highest-cosine edges, then runs raw-attention inner-weighted
facility-location coverage to retain the final ``T`` representatives.
"""

import math

import torch


def _nearest_edge_groups(cosine_sim, num_groups):
    """Return exactly ``num_groups`` components from maximum-similarity edges.

    Edges are considered in descending cosine-similarity order.  An edge is
    accepted only when it connects two different components (maximum-spanning
    forest / Kruskal), so every accepted edge reduces the group count by one.
    A progressively enlarged top-k edge pool avoids sorting all O(N^2) edges
    in the common case while preserving the same accepted-edge order.
    """
    num_tokens = int(cosine_sim.shape[0])
    num_groups = max(1, min(int(num_groups), num_tokens))
    edges_needed = num_tokens - num_groups
    if edges_needed == 0:
        return [[i] for i in range(num_tokens)]

    edge_ends = torch.triu_indices(
        num_tokens, num_tokens, offset=1, device=cosine_sim.device
    )
    edge_scores = cosine_sim[edge_ends[0], edge_ends[1]]
    total_edges = int(edge_scores.numel())
    candidate_count = min(
        total_edges, max(256, edges_needed + 64, edges_needed * 4)
    )

    while True:
        edge_order = torch.topk(
            edge_scores,
            k=candidate_count,
            largest=True,
            sorted=True,
        ).indices
        candidate_edges = edge_ends[:, edge_order].t().detach().cpu().tolist()

        parent = list(range(num_tokens))
        component_size = [1] * num_tokens

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        accepted = 0
        for left, right in candidate_edges:
            root_left = find(left)
            root_right = find(right)
            if root_left == root_right:
                continue
            if component_size[root_left] < component_size[root_right]:
                root_left, root_right = root_right, root_left
            parent[root_right] = root_left
            component_size[root_left] += component_size[root_right]
            accepted += 1
            if accepted == edges_needed:
                break

        if accepted == edges_needed:
            groups = {}
            for token_idx in range(num_tokens):
                root = find(token_idx)
                groups.setdefault(root, []).append(token_idx)
            result = list(groups.values())
            if len(result) != num_groups:
                raise RuntimeError(
                    f"merge grouping produced {len(result)} groups, expected {num_groups}"
                )
            return result

        if candidate_count == total_edges:
            raise RuntimeError(
                f"could only accept {accepted}/{edges_needed} merge edges"
            )
        candidate_count = min(total_edges, candidate_count * 2)


def _merge_candidates(features, raw_cls_attn, merged_token_num):
    """Merge tokens into similarity-edge groups and return group candidates.

    The member with maximum raw CLS attention is the group's spatial anchor.
    Its feature weight is exactly one; every other member is weighted by its
    non-negative cosine similarity to the anchor.  Group attention is the sum
    of member raw attentions, preserving saliency mass without normalization.
    """
    features_f = features.float()
    raw_cls_attn_f = raw_cls_attn.float()
    normalized = features_f / features_f.norm(
        dim=-1, keepdim=True
    ).clamp(min=1e-8)
    cosine_sim = torch.matmul(normalized, normalized.t())
    groups = _nearest_edge_groups(cosine_sim, merged_token_num)

    num_tokens = int(features.shape[0])
    num_groups = len(groups)
    group_ids_cpu = [0] * num_tokens
    raw_attn_cpu = raw_cls_attn_f.detach().cpu().tolist()
    anchor_indices_cpu = []
    for group_id, members_list in enumerate(groups):
        for token_idx in members_list:
            group_ids_cpu[token_idx] = group_id
        anchor_indices_cpu.append(
            max(members_list, key=lambda token_idx: raw_attn_cpu[token_idx])
        )

    group_ids = torch.tensor(
        group_ids_cpu, dtype=torch.long, device=features.device
    )
    anchor_indices = torch.tensor(
        anchor_indices_cpu, dtype=torch.long, device=features.device
    )
    token_indices = torch.arange(num_tokens, device=features.device)
    token_anchors = anchor_indices[group_ids]
    weights = cosine_sim[token_anchors, token_indices].clamp(min=0.0)
    weights[token_indices == token_anchors] = 1.0

    merged_features = torch.zeros(
        num_groups,
        features_f.shape[-1],
        dtype=features_f.dtype,
        device=features.device,
    )
    merged_features.index_add_(
        0, group_ids, features_f * weights.unsqueeze(-1)
    )
    weight_sums = torch.zeros(
        num_groups, dtype=features_f.dtype, device=features.device
    )
    weight_sums.index_add_(0, group_ids, weights)
    merged_features = merged_features / weight_sums.unsqueeze(-1).clamp(min=1e-8)

    merged_attentions = torch.zeros(
        num_groups, dtype=raw_cls_attn_f.dtype, device=features.device
    )
    merged_attentions.index_add_(0, group_ids, raw_cls_attn_f)
    return merged_features, merged_attentions, anchor_indices


def cover_prune_merge_select(features, raw_cls_attn, target_token_num):
    """Merge first, then apply raw-attention inner-weighted COVER pruning.

    Args:
        features: ``[N, C]`` pre-projector ViT patch features.
        raw_cls_attn: ``[N]`` raw CLS-to-patch attention (no min-max).
        target_token_num: final retained token count ``T``.

    Returns:
        selected_anchor_indices: ``[T]`` original spatial indices, sorted.
        selected_merged_features: ``[T, C]`` merged features corresponding to
            those anchors and in the same sorted order.
        merged_token_num: intermediate candidate count
            ``ceil(sqrt(N * T))``.
    """
    if features.dim() != 2:
        raise ValueError(f"features must be [N,C], got {tuple(features.shape)}")
    if raw_cls_attn.dim() != 1 or raw_cls_attn.shape[0] != features.shape[0]:
        raise ValueError(
            f"raw_cls_attn must be [N] matching features, got "
            f"{tuple(raw_cls_attn.shape)} vs {tuple(features.shape)}"
        )

    num_tokens = int(features.shape[0])
    target_token_num = max(1, min(int(target_token_num), num_tokens))
    merged_token_num = min(
        num_tokens,
        max(target_token_num, int(math.ceil(math.sqrt(num_tokens * target_token_num)))),
    )

    merged_features, merged_attn, anchor_indices = _merge_candidates(
        features, raw_cls_attn, merged_token_num
    )
    normalized = merged_features / merged_features.norm(
        dim=-1, keepdim=True
    ).clamp(min=1e-8)
    similarity = torch.clamp(torch.matmul(normalized, normalized.t()), min=0.0)

    available = torch.ones(
        merged_token_num, dtype=torch.bool, device=features.device
    )
    selected = torch.empty(
        target_token_num, dtype=torch.long, device=features.device
    )
    coverage = torch.zeros(
        merged_token_num, dtype=similarity.dtype, device=features.device
    )
    attn = merged_attn.to(similarity.dtype)

    seed = int(torch.argmax((attn.unsqueeze(0) * similarity).sum(dim=1)))
    selected[0] = seed
    available[seed] = False
    coverage = torch.maximum(coverage, similarity[seed])

    for step in range(1, target_token_num):
        gain = (
            attn.unsqueeze(0)
            * torch.clamp(similarity - coverage.unsqueeze(0), min=0.0)
        ).sum(dim=1)
        gain[~available] = -float("inf")
        chosen = int(torch.argmax(gain))
        selected[step] = chosen
        available[chosen] = False
        coverage = torch.maximum(coverage, similarity[chosen])

    selected_anchor_indices = anchor_indices[selected]
    spatial_order = torch.argsort(selected_anchor_indices)
    selected_anchor_indices = selected_anchor_indices[spatial_order]
    selected_merged_features = merged_features[selected][spatial_order]
    return selected_anchor_indices, selected_merged_features, merged_token_num
