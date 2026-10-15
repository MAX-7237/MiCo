"""Image mico_coverage_proj_llm_12 objective adapted to Video's pooled SigLIP tokens.

S1 raw visual weights are mean-query SigLIP attention (SigLIP has no CLS).
Both stages preserve image code's inner-weighted nonnegative cosine objective.
"""
import heapq
import math
import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv


def hidden_l2_weights(features, attention, normalize=False):
    """H is the raw feature L2 at the prune site, matching the image method."""
    weights = attention.float().clamp_min(0) * features.float().norm(dim=-1)
    if normalize:
        total = weights.sum()
        weights = weights / total if total > 0 else torch.full_like(weights, 1 / len(weights))
    return weights


def solve_budget(target_average, entering, layer, layers):
    target_average, entering, layer, layers = map(int, (target_average, entering, layer, layers))
    if target_average < 1 or entering < 1 or not 0 < layer < layers:
        raise ValueError('invalid positive budget or 1-based pruning layer')
    exact = (target_average * layers - layer * entering) / (layers - layer)
    if exact < 1 or exact > entering:
        raise ValueError(f'infeasible average budget: target={target_average}, N={entering}, K={layer}, L={layers}, R={exact}')
    keep = int(round(exact))
    realized = (layer * entering + (layers - layer) * keep) / layers
    return {'keep': keep, 'exact_keep': exact, 'realized_average': realized,
            'rounding_delta': realized - target_average, 'target_average': target_average}


def weighted_coverage_select(features, weights, keep):
    """Sorted exact greedy facility-location indices, with lazy bounds for long videos.

    Every accepted lazy candidate has its current marginal gain evaluated; bounds
    only avoid provably unnecessary recomputations. No candidate sampling/topk.
    """
    if features.ndim != 2 or weights.ndim != 1 or weights.numel() != len(features):
        raise ValueError('coverage expects features [N,D] and weights [N]')
    n = len(features)
    if not 1 <= int(keep) <= n:
        raise ValueError('keep must be between 1 and token count')
    f = F.normalize(features.float(), dim=-1)
    w = weights.float().clamp_min(0)
    if not torch.isfinite(f).all() or not torch.isfinite(w).all():
        raise ValueError('non-finite coverage input')
    sim = (f @ f.T).clamp_min(0)
    coverage = torch.zeros(n, dtype=sim.dtype, device=sim.device)
    chosen = []
    if n <= 512:
        available = torch.ones(n, dtype=torch.bool, device=sim.device)
        for _ in range(int(keep)):
            gains = (w[None] * (sim - coverage[None]).clamp_min(0)).sum(1)
            gains = gains.masked_fill(~available, -float('inf'))
            idx = int(gains.argmax())
            chosen.append(idx); available[idx] = False
            coverage = torch.maximum(coverage, sim[idx])
    else:
        initial = (w[None] * sim).sum(1).cpu().tolist()
        heap = [(-float(g), i, 0) for i, g in enumerate(initial)]
        heapq.heapify(heap)
        for step in range(int(keep)):
            while True:
                negative_bound, idx, stamp = heapq.heappop(heap)
                if stamp == step:
                    break
                gain = float((w * (sim[idx] - coverage).clamp_min(0)).sum())
                heapq.heappush(heap, (-gain, idx, step))
            chosen.append(idx)
            coverage = torch.maximum(coverage, sim[idx])
    return torch.tensor(sorted(chosen), dtype=torch.long, device=features.device)


def pack_video(features, masks, layout='flat'):
    """Keep shared Video pruning protocol: frame-major flat with no newlines."""
    if layout != 'flat':
        raise ValueError('mico_coverage Video supports flat packing only')
    if features.ndim != 3 or masks.shape != features.shape[:2] or masks.dtype != torch.bool:
        raise ValueError('expected frame features [F,N,D] and boolean masks [F,N]')
    if not masks.any(1).all():
        raise ValueError('every frame must retain at least one S1 patch')
    frame_ids, patch_ids = masks.nonzero(as_tuple=True)
    flat = features[frame_ids, patch_ids]
    return flat, {'patch_positions': torch.arange(len(flat), device=features.device),
                  'frame_ids': frame_ids, 'original_patch_ids': patch_ids,
                  'newline_count': 0, 'frames': features.shape[0],
                  'stage1_per_frame': masks.sum(1).tolist(),
                  'original_patches_per_frame': features.shape[1]}


def post_text_attention(decoder_layer, hidden_states, position_ids, visual_positions, visual_end):
    """Exact eval attention mean over heads and all text queries after video.

    Calculate only needed rows, retaining RoPE, full causal softmax denominator,
    GQA head repetition and the attention module's probability cast.
    """
    if hidden_states.shape[0] != 1:
        raise ValueError('mico_coverage Video currently requires batch_size=1')
    attn = decoder_layer.self_attn
    if decoder_layer.training:
        raise ValueError('mico_coverage inference requires eval mode')
    x = decoder_layer.input_layernorm(hidden_states)
    b, seq, _ = x.shape
    q = attn.q_proj(x).view(b, seq, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(b, seq, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    # v projection is not needed; the RoPE method only uses tensor dtype/device.
    cos, sin = attn.rotary_emb(k, seq_len=int(position_ids.max()) + 1)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    k = repeat_kv(k, attn.num_key_value_groups)
    start = int(visual_end) if int(visual_end) < seq else seq - 1
    key_order = torch.arange(seq, device=x.device)
    accumulated = torch.zeros(visual_positions.numel(), device=x.device, dtype=torch.float32)
    for lo in range(start, seq, 64):
        hi = min(seq, lo + 64)
        logits = (q[:, :, lo:hi] @ k.transpose(-1, -2)) / math.sqrt(attn.head_dim)
        future = key_order[None, :] > torch.arange(lo, hi, device=x.device)[:, None]
        logits = logits.masked_fill(future[None, None], torch.finfo(logits.dtype).min)
        probs = F.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
        accumulated += probs.mean(1)[0, :, visual_positions].float().sum(0)
    weights = (accumulated / (seq - start)).clamp_min(0)
    total = weights.sum()
    return (weights / total if total > 0 else torch.full_like(weights, 1 / len(weights))).detach()
