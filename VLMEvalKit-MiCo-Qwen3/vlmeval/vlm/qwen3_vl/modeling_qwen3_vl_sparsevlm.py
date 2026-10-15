# coding=utf-8
"""SparseVLM (text-guided, decoder-internal, multi-stage) visual-token pruning for Qwen3-VL.

Ported from the LLaVA reference `llava/model/language_model/modeling_llama_sparsevlm.py`
(LLaVA-TOPS-ICLR / LLaVA-STAR-Pro2-sparsevlm-audit), rebuilt on the Qwen3-VL FastV/DART
skeleton so it inherits the same Qwen3-specific correctness fixes:

  * P2 -- recompute RoPE from the pruned position_ids; never gather Qwen3-VL cos/sin on dim 2
    (they are [B, S, head_dim], so dim 2 is head_dim, not the sequence).
  * P3 -- rebuild the post-prune causal mask with attention_mask / past_key_values /
    position_ids dropped TOGETHER (the packed-sequence poisoning guard).
  * P5 -- accept, inject and re-prune the DeepStack side-features in sync with the prune mask.
  * P6 -- explicit set_visual_token_positions() index vector instead of the LLaVA
    `system_prompt_length=35` + `visual_token_length=576` offset window. Those two constants are
    the single biggest portability break: Qwen3-VL's first image token sits at index 4, the
    window shifts per resolution, and multi-image prompts are not a contiguous run at all.

Mechanism:
  1. TEXT RATER -- computed once, on the PRE-DECODER hidden states. `text @ visual^T`, softmaxed
     over text rows and averaged over visual columns; the text rows whose score exceeds the
     mean become the "rater" text tokens used by every pruning layer.
  2. PRUNE LAYERS -- at each of 3 decoder layers, score visual tokens by the cross-attention
     weights on the rater text rows (QK probe, cross-head mean), keep the top `keep` tokens,
     then CLUSTER-AND-MERGE the highest-scoring 30% of the discarded remainder into 10% as many
     merged tokens. The sequence is rebuilt as
         [prefix | kept visual | merged visual | text]  (original intra-block order preserved).

T denotes the layer-average budget, as in VLMEvalKit-TOPS/sparsevlm_core.py.
The reference's T=128 profile (300, 110, 35) is scaled per sample, accounting for
recycled centroids and the tokens actually entering every decoder layer. Prune
AFTER 1-based layers [2, 7, 17] on Qwen3's 36-layer decoder. Some large images
cannot reach T because these initial unpruned layers already exceed the budget;
their actual minimum and budget_reachable=False are reported, never relabelled T.
For interleaved images, selection/recycling stays within each visual span so
centroids cannot move evidence across the intervening text or image boundaries.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    from transformers.models.qwen3_vl.modeling_qwen3_vl import create_causal_mask as _create_causal_mask_impl
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb as _apply_rotary_pos_emb
    _BASE_MODEL_CLS = Qwen3VLModel
    _USING_QWEN3 = True
except ImportError:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel as _BASE_MODEL_CLS
    _create_causal_mask_impl = None
    _apply_rotary_pos_emb = None
    _USING_QWEN3 = False

logger = logging.get_logger(__name__)

SPARSEVLM_REV = "qwen3vl-sparsevlm-20260917/text-order-isoavg-tiled"

_FALSEY = ('', '0', 'false', 'False', 'FALSE', 'no', 'off')


def _env_on(name: str, default: str = '0') -> bool:
    return str(os.environ.get(name, default)).strip() not in _FALSEY


# ---------------------------------------------------------------------------------------------
# Budget schedule
# ---------------------------------------------------------------------------------------------
# Same profile used by the existing Qwen2.5/InternVL SparseVLM budget solver.
_REF_SPAN_SHAPE = (300.0, 110.0, 35.0)

# Reference pruning depths [2, 6, 15] on 32 layers -> depth-scaled to Qwen3-VL's 36.
_REF_PRUNE_LOC = (2, 6, 15)
_REF_LAYERS = 32


def _prune_layers_for(n_layers: int) -> Tuple[int, ...]:
    """Depth-scale the reference's 1-based prune layers to an `n_layers` decoder."""
    scaled = [max(1, min(n_layers, int(round(loc * n_layers / _REF_LAYERS)))) for loc in _REF_PRUNE_LOC]
    # Strictly increasing, so the budget is monotonically non-increasing even on odd depths.
    out: List[int] = []
    for s in scaled:
        out.append(s if not out else max(s, out[-1] + 1))
    return tuple(min(l, n_layers) for l in out)


def _recycled_length(count: int, keep: int) -> int:
    if keep >= count:
        return count
    merge_count = max(int((count - keep) * 0.3), 1)
    return keep + max(int(merge_count * 0.1), 1)


def _keep_for_span(count: int, target: int) -> int:
    # Recycling imposes a floor near 3% of the input; do not pretend an
    # impossible target was reached by logging the requested span.
    lo, hi = 1, count
    while lo < hi:
        mid = (lo + hi) // 2
        if _recycled_length(count, mid) < target:
            lo = mid + 1
        else:
            hi = mid
    candidates = {lo, max(1, lo - 1)}
    return min(candidates, key=lambda k: (abs(_recycled_length(count, k) - target), k))


def build_keep_schedule(target_tokens: int, n_layers: int, visual_lengths) -> tuple:
    """Solve per-image keeps against actual layer input lengths, including recycling."""
    locs = _prune_layers_for(n_layers)
    weights = (locs[0], locs[1] - locs[0], locs[2] - locs[1], n_layers - locs[2])
    if not all(w > 0 for w in weights):
        raise ValueError(f"SparseVLM needs three distinct internal prune layers, got {locs}")
    initial = tuple(int(n) for n in visual_lengths)
    if not initial or min(initial) < 1:
        raise ValueError("visual spans must be nonempty")

    def profile(scale):
        lengths = initial
        total = weights[0] * sum(lengths)
        schedule = {}
        for loc, shape, weight in zip(locs, _REF_SPAN_SHAPE, weights[1:]):
            wanted = min(sum(lengths), max(len(lengths), int(round(scale * shape))))
            # Proportional allocation preserves every image. The solver evaluates
            # the resulting integer/recycled counts, not the fractional wishes.
            desired = [max(1, round(wanted * n / sum(lengths))) for n in lengths]
            keeps = tuple(_keep_for_span(n, t) for n, t in zip(lengths, desired))
            lengths = tuple(_recycled_length(n, k) for n, k in zip(lengths, keeps))
            schedule[loc] = keeps
            total += weight * sum(lengths)
        return total / n_layers, schedule

    lower, upper = profile(0.0), profile(sum(initial) * 4.0)
    best = min((lower, upper), key=lambda p: abs(p[0] - target_tokens))
    lo, hi = 0.0, sum(initial) * 4.0
    for _ in range(64):
        mid = (lo + hi) / 2
        candidate = profile(mid)
        if abs(candidate[0] - target_tokens) < abs(best[0] - target_tokens):
            best = candidate
        if candidate[0] < target_tokens:
            lo = mid
        else:
            hi = mid
    return best[1], best[0], lower[0] <= target_tokens <= upper[0]


def _visual_spans(positions: torch.Tensor) -> List[torch.Tensor]:
    if positions.numel() == 0:
        return []
    boundaries = (positions[1:] != positions[:-1] + 1).nonzero().flatten().add(1).tolist()
    return list(torch.tensor_split(positions, boundaries))


def _compact_visual_spans(hidden_states, visual_positions, kept_blocks, merged_blocks):
    """Keep text in order, insert each centroid block inside its source image span.

    Return source indices for hidden/RoPE/DeepStack and an exact old-to-new map;
    augmented source indices are not sorted, so searchsorted would be incorrect.
    """
    seq_len = hidden_states.shape[1]
    device = hidden_states.device
    spans = _visual_spans(visual_positions)
    assert len(spans) == len(kept_blocks) == len(merged_blocks)
    source, pos_source, ds_source, visual_mask = [], [], [], []
    previous, merge_offset, vis_offset = 0, seq_len, 0
    for span, kept, merged in zip(spans, kept_blocks, merged_blocks):
        start, stop = int(span[0]), int(span[-1]) + 1
        assert kept.numel() and bool(torch.all((kept >= start) & (kept < stop)))
        text = torch.arange(previous, start, device=device)
        merge_ids = torch.arange(merge_offset, merge_offset + len(merged), device=device)
        source.extend((text, kept, merge_ids))
        pos_source.extend((text, kept, kept[-1:].expand(len(merged))))
        visual_mask.extend((torch.zeros_like(text, dtype=torch.bool),
                            torch.ones_like(kept, dtype=torch.bool),
                            torch.ones_like(merge_ids, dtype=torch.bool)))
        ds_source.extend((kept - start + vis_offset,
                          torch.full_like(merge_ids, -1)))
        previous, merge_offset = stop, merge_offset + len(merged)
        vis_offset += len(span)
    tail = torch.arange(previous, seq_len, device=device)
    source.append(tail)
    pos_source.append(tail)
    visual_mask.append(torch.zeros_like(tail, dtype=torch.bool))
    source = torch.cat(source)
    pos_source = torch.cat(pos_source)
    new_visual = torch.cat(visual_mask).nonzero().flatten()
    augmented = torch.cat([hidden_states] + [m.unsqueeze(0) for m in merged_blocks], dim=1)
    old_to_new = torch.full((merge_offset,), -1, device=device, dtype=torch.long)
    old_to_new[source] = torch.arange(len(source), device=device)
    return (augmented.index_select(1, source), source, pos_source,
            new_visual, torch.cat(ds_source), old_to_new)


def _pairwise_token_distances(x: torch.Tensor, max_working_bytes: int = 256 * 1024 * 1024):
    """Tile the original dtype/subtraction/norm, avoiding the HallB N*N*D OOM."""
    batch, count, dim = x.shape
    bytes_per_pair = batch * dim * x.element_size()
    if max_working_bytes < bytes_per_pair:
        raise ValueError("distance workspace must hold at least one token pair")
    pair_budget = max_working_bytes // bytes_per_pair
    rows = min(count, max(1, pair_budget // count))
    cols = min(count, max(1, pair_budget // rows))
    distance = x.new_empty((batch, count, count))
    for row in range(0, count, rows):
        for col in range(0, count, cols):
            distance[:, row:row + rows, col:col + cols] = (
                x[:, row:row + rows, None, :] - x[:, None, col:col + cols, :]
            ).norm(dim=-1, p=2)
    return distance


# ---------------------------------------------------------------------------------------------
# cluster_and_merge -- verbatim port of the reference (pure torch; upstream used einops for two
# trivial broadcasts, inlined here so this module keeps the repo's zero-extra-deps footprint).
# ---------------------------------------------------------------------------------------------
def cluster_and_merge(x: torch.Tensor, cluster_num: int) -> torch.Tensor:
    """Density-peak clustering + cluster-mean merge. x:[B, N, C] -> [B, cluster_num, C]."""
    B, N, C = x.shape
    cluster_num = max(1, min(int(cluster_num), N))

    distance = _pairwise_token_distances(x)
    dist_matrix = distance / (C ** 0.5)

    # local density from the `cluster_num` nearest neighbours
    dist_nearest = torch.topk(dist_matrix, k=cluster_num, dim=-1, largest=False).values
    density = (-(dist_nearest ** 2).mean(dim=-1)).exp()

    # distance indicator: distance to the nearest denser point
    mask = (density[:, None, :] > density[:, :, None]).type(x.dtype)
    dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
    dist = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1).values

    score = dist * density
    index_down = torch.topk(score, k=cluster_num, dim=-1).indices

    # assign every point to its nearest centre
    dist_matrix = _index_points(dist_matrix, index_down)
    idx_cluster = dist_matrix.argmin(dim=1)

    # a centre always merges to itself
    idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
    idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
    idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    idx_batch = torch.arange(B, device=x.device)[:, None]
    idx = idx_cluster + idx_batch * cluster_num

    token_weight = torch.ones(B, N, 1, device=x.device, dtype=x.dtype)
    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6

    norm_weight = token_weight / all_weight[idx]
    x_source = x * norm_weight

    x_merged = x.new_zeros(B * cluster_num, C)
    x_merged.index_add_(dim=0, index=idx.reshape(B * N), source=x_source.reshape(B * N, C))
    return x_merged.reshape(B, cluster_num, C)


def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather along dim 1. points:[B, N, C], idx:[B, S] -> [B, S, C]."""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def _past_length(cache) -> int:
    if cache is None:
        return 0
    try:
        return int(cache.get_seq_length())
    except Exception:
        return 0


class SparseVLMQwen3VLModel(_BASE_MODEL_CLS):
    """SparseVLM for Qwen3-VL: text-rater scored, clustered-merge pruning at 3 decoder depths."""

    def __init__(self, config, visual_token_num: int):
        super().__init__(config)
        lm = self.language_model
        self.embed_tokens = lm.embed_tokens
        self.layers = lm.layers
        self.norm = lm.norm
        self.rotary_emb = lm.rotary_emb
        self._lm_update_causal_mask = getattr(lm, '_update_causal_mask', None)

        # P6: explicit visual-position window (no system_prompt_length offset).
        self.visual_token_positions: Optional[torch.LongTensor] = None
        self._visual_positions_fresh = False

        self.visual_token_length = 1296   # nominal; logging / back-compat only
        self.visual_token_num = 0.0
        self.visual_token_len = 0.0

        try:
            target = int(visual_token_num)
        except (TypeError, ValueError):
            raise ValueError(f"visual_token_num must be an integer, got {visual_token_num!r}")
        if target <= 0:
            raise ValueError(
                f"SparseVLM needs a positive visual_token_num (got {visual_token_num!r}); "
                "-1 means 'no budget' and would divide the schedule by a non-positive T."
            )
        self.target = target

        self.n_layers = len(self.layers)
        self.prune_layers = _prune_layers_for(self.n_layers)
        self.keep_schedule = {}
        self.prune_set = set(self.prune_layers)

        # Reference constants for the merge half of each stage.
        self.merge_top_frac = 0.3    # highest-scoring 30% of the discarded tokens...
        self.merge_cluster_frac = 0.1  # ...are clustered into 10% as many merged tokens
        self.rater_frac = float(os.environ.get('SPARSEVLM_RATER_FRAC', '0.0'))  # 0 = reference (mean)
        self.debug = _env_on('ENABLE_DEBUG', '0')

        # Cross-attention probe cost guard: the reference clusters on the FULL visual hidden
        # state, which is N x D. Keep it, but expose an escape hatch for very large N.
        self.gpu_merge = _env_on('SPARSEVLM_GPU_MERGE', '1')

        print(f"[SparseVLM Init] rev={SPARSEVLM_REV} Qwen3-VL L={self.n_layers} "
              f"target T={self.target} prune_layers={list(self.prune_layers)} "
              f"keep_schedule={self.keep_schedule}")

        self.reset_state()

    def reset_state(self):
        self.prefill_done = False
        self.current_visual_length: Optional[int] = None
        self.original_visual_length: Optional[int] = None
        self.retained_visual_length: Optional[int] = None
        self.retain_token_indices: Optional[torch.LongTensor] = None
        self.visual_token_num = 0.0
        self.visual_token_len = 0.0
        self.layer_visual_token_counts = []
        self.budget_reachable = True

    def set_visual_token_positions(self, positions: Optional[torch.LongTensor]) -> None:
        """Absolute positions (post-stage-1 sequence) of the surviving image tokens."""
        if positions is None:
            self.visual_token_positions = None
            self._visual_positions_fresh = False
            return
        self.visual_token_positions = positions.detach().reshape(-1).to(torch.long)
        self._visual_positions_fresh = True

    def _get_causal_mask(self, input_tensor, attention_mask, cache_position, past_key_values, position_ids=None):
        # P3b: drop position_ids on exactly the packed-sequence trigger, so gapped post-prune
        # positions are not read as document boundaries.
        if position_ids is not None and attention_mask is None and past_key_values is None:
            position_ids = None
        if _create_causal_mask_impl is not None:
            return _create_causal_mask_impl(
                self.config, input_tensor, attention_mask, cache_position,
                past_key_values=past_key_values, position_ids=position_ids
            )
        elif self._lm_update_causal_mask is not None:
            return self._lm_update_causal_mask(
                attention_mask, input_tensor, cache_position, past_key_values, False
            )
        return None

    # ── 1. text rater ─────────────────────────────────────────────────────────────────────
    def _compute_text_rater(self, hidden_states: torch.Tensor,
                            visual_positions: torch.LongTensor) -> torch.LongTensor:
        """Rater text rows for THIS sample, as ABSOLUTE positions in the current sequence.

        Reference (modeling_llama_sparsevlm.py:249-256):
            matrix = text_tokens @ visual_tokens^T ; matrix.softmax(0).mean(1) ; rows > mean
        The reference indexes text at a FIXED [system_prompt_length+visual_len:] window; here the
        text rows are simply the complement of the pushed visual positions, which is correct for
        Qwen3-VL's prompt layout (images first, then text) AND for interleaved multi-image prompts.
        """
        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        vis = visual_positions.to(device)

        is_visual = torch.zeros(seq_len, dtype=torch.bool, device=device)
        is_visual[vis] = True
        text_positions = (~is_visual).nonzero(as_tuple=False).squeeze(-1)
        # Prefix/system tokens cannot attend to future images in a causal decoder.
        if vis.numel():
            text_positions = text_positions[text_positions > vis[0]]
        if text_positions.numel() == 0 or vis.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=device)

        h = hidden_states[0]                                  # [S, D]
        text_tokens = h.index_select(0, text_positions).float()      # [M, D]
        visual_tokens = h.index_select(0, vis).float()               # [Nv, D]

        matrix = text_tokens @ visual_tokens.transpose(0, 1)         # [M, Nv]
        matrix = matrix.softmax(dim=0).mean(dim=1)                   # [M]
        thr = matrix.mean()
        if self.rater_frac > 0:
            k = max(1, int(round(self.rater_frac * matrix.numel())))
            keep = torch.topk(matrix, k=min(k, matrix.numel())).indices
            rater_local = text_positions.index_select(0, keep.sort().values)
        else:
            rater_local = text_positions[matrix > thr]
        if rater_local.numel() == 0:
            rater_local = text_positions[matrix.argmax()].reshape(1)
        return rater_local

    # ── 2. per-layer text-guided scoring ──────────────────────────────────────────────────
    def _score_visual_tokens(self, hidden_states: torch.Tensor, decoder_layer: nn.Module,
                             visual_positions: torch.LongTensor, rater_positions: torch.LongTensor,
                             position_embeddings) -> torch.Tensor:
        """Cross-attention (rater text rows -> visual columns), cross-head mean. [Nv] float32.

        QK probe rather than layer_outputs[1]: the repo pins sdpa on the pruning path, and sdpa
        does not materialise attention weights. This reproduces the reference's
        `self_attn_weights[:, rater_text_rows, visual_cols].mean(1)` exactly (cos/sin come from
        the same position_embeddings the layer is about to consume, so the probe cannot drift).
        """
        self_attn = decoder_layer.self_attn
        ln = getattr(decoder_layer, 'input_layernorm', None)
        attn_input = ln(hidden_states) if ln is not None else hidden_states
        # Probe in the attention's OWN dtype. The wrapper loads the model in bf16 while a caller may
        # hand us fp32 embeds, and a mixed-dtype Linear raises.
        param_dtype = self_attn.q_proj.weight.dtype
        attn_input = attn_input.to(param_dtype)

        head_dim = getattr(self_attn, 'head_dim', None)
        if head_dim is None:
            head_dim = self_attn.q_proj.out_features // self_attn.num_heads
        n_heads = getattr(self_attn, 'num_heads', None) or (self_attn.q_proj.out_features // head_dim)
        n_kv = getattr(self_attn, 'num_key_value_heads', None) or (self_attn.k_proj.out_features // head_dim)

        B, S, _ = attn_input.shape
        if B != 1:
            raise RuntimeError(f"[SparseVLM] expected batch 1 for the text-rater probe, got {B}")

        q = self_attn.q_proj(attn_input).view(B, S, n_heads, head_dim).transpose(1, 2)
        k = self_attn.k_proj(attn_input).view(B, S, n_kv, head_dim).transpose(1, 2)
        if getattr(self_attn, 'q_norm', None) is not None:
            q = self_attn.q_norm(q)
        if getattr(self_attn, 'k_norm', None) is not None:
            k = self_attn.k_norm(k)

        if _apply_rotary_pos_emb is not None and position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        if n_kv != n_heads:   # GQA: repeat KV to head count
            k = k.repeat_interleave(n_heads // n_kv, dim=1)

        vis = visual_positions.to(q.device)
        rat = rater_positions.to(q.device)
        q_r = q[0].index_select(1, rat).float()     # [H, M, hd]
        k_v = k[0].float()                          # [H, S, hd]
        scores = torch.matmul(q_r, k_v.transpose(-1, -2)) / (head_dim ** 0.5)  # [H, M, S]
        scores.masked_fill_(torch.arange(S, device=q.device)[None, None, :] > rat[None, :, None],
                            float('-inf'))
        scores = scores.softmax(dim=-1)
        scores = scores.mean(dim=0).mean(dim=0)     # cross-head, then cross-rater mean -> [S]
        return scores.index_select(0, vis).float()  # [Nv]

    # ── 3. one prune stage ────────────────────────────────────────────────────────────────
    def _sparsevlm_select(self, hidden_states: torch.Tensor,
                          visual_positions: torch.LongTensor, scores: torch.Tensor,
                          keep: int) -> Tuple[torch.LongTensor, torch.Tensor]:
        """Return (kept ABS positions sorted, merged token embeddings [n_merged, D])."""
        device = hidden_states.device
        n_v = int(visual_positions.numel())
        keep = max(1, min(int(keep), n_v))

        top_idx = torch.topk(scores, k=keep).indices
        keep_mask = torch.zeros(n_v, dtype=torch.bool, device=device)
        keep_mask[top_idx] = True

        rest_local = (~keep_mask).nonzero(as_tuple=False).squeeze(-1)
        merged = hidden_states.new_zeros((0, hidden_states.shape[-1]))
        if rest_local.numel() > 0:
            merge_num = max(int(rest_local.numel() * self.merge_top_frac), 1)
            merge_num = min(merge_num, rest_local.numel())
            merge_local = rest_local[torch.topk(scores.index_select(0, rest_local), k=merge_num).indices]

            src = hidden_states[0].index_select(0, visual_positions.index_select(0, merge_local))
            cluster_num = max(int(merge_num * self.merge_cluster_frac), 1)
            merge_input = src.float() if not self.gpu_merge else src
            merged = cluster_and_merge(merge_input.unsqueeze(0), cluster_num)[0].to(hidden_states.dtype)

        kept_abs = visual_positions.index_select(0, top_idx.sort().values)
        return kept_abs, merged

    # ── forward ───────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        past_seen_tokens = _past_length(past_key_values)
        if past_seen_tokens == 0:
            self.reset_state()

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else getattr(self.config, 'output_hidden_states', False)
        )
        use_cache = use_cache if use_cache is not None else getattr(self.config, 'use_cache', True)
        return_dict = return_dict if return_dict is not None else getattr(self.config, 'use_return_dict', True)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if self.is_gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        seq_length = inputs_embeds.shape[1]
        is_prefill = (past_seen_tokens == 0) and (seq_length > 1)

        # Qwen3-VL 4D position_ids: [4, B, S] where [0]=text, [1:]=mrope T/H/W
        text_position_ids = None
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        elif position_ids.dim() == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        # else: already [3, B, S]

        causal_mask = self._get_causal_mask(
            inputs_embeds, attention_mask, cache_position, past_key_values, position_ids=text_position_ids
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None

        # ── Resolve this sample's visual window ──
        visual_positions: Optional[torch.LongTensor] = None
        n_visual = 0
        do_prune = False

        if is_prefill:
            self.prefill_done = True
            pushed = self.visual_token_positions if self._visual_positions_fresh else None
            self._visual_positions_fresh = False
            if pushed is None:
                logger.warning_once(
                    "[SparseVLM] no visual token positions pushed for this sample; this prefill runs "
                    "UNPRUNED (expected only for text-only prompts)."
                )
            else:
                pushed = pushed.reshape(-1).to(device=hidden_states.device, dtype=torch.long)
                if pushed.numel() and int(pushed.max()) >= seq_length:
                    raise RuntimeError(
                        f"[SparseVLM] stale visual_token_positions: max index {int(pushed.max())} >= "
                        f"seq_len {seq_length}."
                    )
                visual_positions = pushed
                n_visual = int(pushed.numel())

            if visual_positions is not None and visual_pos_masks is not None and self.debug:
                _pm = visual_pos_masks[0].nonzero(as_tuple=False).squeeze(-1)
                assert _pm.numel() == n_visual and bool(
                    torch.equal(_pm.to(visual_positions.device), visual_positions)
                ), "visual_pos_masks and visual_token_positions disagree (DeepStack would misalign)"

            if n_visual > 0:
                self.original_visual_length = n_visual
                self.current_visual_length = n_visual
                do_prune = (self.target > 0) and (n_visual > self.target)
                self.keep_schedule, self.planned_average, self.budget_reachable = build_keep_schedule(
                    self.target, self.n_layers, [len(s) for s in _visual_spans(visual_positions)]
                )
                if self.debug:
                    print(f"\n[SparseVLM Qwen3-VL] rev={SPARSEVLM_REV} prefill seq={seq_length} "
                          f"visual N={n_visual} target={self.target} "
                          f"prune={'yes' if do_prune else 'no'} "
                          f"schedule={list(zip(self.prune_layers, [self.keep_schedule[l] for l in self.prune_layers]))} "
                          f"deepstack={0 if deepstack_visual_embeds is None else len(deepstack_visual_embeds)}",
                          flush=True)

        # ── 1. text rater (computed ONCE, on the PRE-DECODER hidden states) ──
        rater_positions: Optional[torch.LongTensor] = None
        if is_prefill and do_prune and visual_positions is not None:
            rater_positions = self._compute_text_rater(hidden_states, visual_positions)
            if self.debug:
                print(f"[SparseVLM] text raters: {int(rater_positions.numel())} rows selected "
                      f"(visual N={n_visual})", flush=True)

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_idx = decoder_layer.self_attn.layer_idx + 1   # 1-BASED
            if is_prefill and n_visual:
                self.layer_visual_token_counts.append(int(visual_positions.numel()))
            prune_here = (
                is_prefill and do_prune and rater_positions is not None
                and layer_idx in self.prune_set and visual_positions is not None
            )
            if prune_here:
                # Probe this layer's input QK, then merge its OUTPUT features, as
                # in the official implementation that returns attention weights.
                scores = self._score_visual_tokens(
                    hidden_states, decoder_layer, visual_positions, rater_positions,
                    position_embeddings
                )

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            # ── P5a: DeepStack injection after text layers 0/1/2 (upstream UP:929) ──
            _ds = layer_idx - 1
            if (
                deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and 0 <= _ds < len(deepstack_visual_embeds)
            ):
                assert deepstack_visual_embeds[_ds].shape[0] == int(visual_pos_masks.sum()), \
                    "deepstack rows out of sync with visual_pos_masks"
                hidden_states = self.language_model._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[_ds]
                )

            if prune_here:
                orig_seq_len = hidden_states.shape[1]
                pre_visual = visual_positions
                spans = _visual_spans(pre_visual)
                kept_blocks, merged_blocks = [], []
                offset = 0
                for span, keep in zip(spans, self.keep_schedule[layer_idx]):
                    kept, merged = self._sparsevlm_select(
                        hidden_states, span, scores[offset:offset + len(span)], keep
                    )
                    kept_blocks.append(kept)
                    merged_blocks.append(merged)
                    offset += len(span)
                (hidden_states, source_indices, position_sources, visual_positions,
                 ds_sources, old_to_new) = _compact_visual_spans(
                    hidden_states, pre_visual, kept_blocks, merged_blocks
                )
                new_len = hidden_states.shape[1]
                position_ids = position_ids.index_select(-1, position_sources)
                if text_position_ids is not None:
                    text_position_ids = text_position_ids.index_select(-1, position_sources)
                cache_position = cache_position[:new_len]
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
                # Retained MRoPE positions have gaps; don't interpret them as
                # packed-document boundaries when rebuilding the causal mask.
                causal_mask = self._get_causal_mask(
                    hidden_states, None, cache_position, None, position_ids=None
                )
                if deepstack_visual_embeds is not None:
                    rebuilt = []
                    for d in deepstack_visual_embeds:
                        assert d.shape[0] == len(pre_visual), "DeepStack input rows misaligned"
                        rows = d.index_select(0, ds_sources.clamp_min(0)).clone()
                        # Recycled tokens have no original DeepStack row.
                        rows[ds_sources < 0] = 0
                        rebuilt.append(rows)
                    deepstack_visual_embeds = rebuilt
                if visual_pos_masks is not None:
                    visual_pos_masks = torch.zeros(
                        (hidden_states.shape[0], new_len), dtype=torch.bool, device=hidden_states.device
                    )
                    visual_pos_masks[:, visual_positions] = True
                rater_positions = old_to_new[rater_positions]
                assert bool(torch.all(rater_positions >= 0)), "SparseVLM dropped a text rater"
                self.visual_token_positions = visual_positions
                self.retain_token_indices = source_indices
                self.current_visual_length = len(visual_positions)
                self.retained_visual_length = self.current_visual_length
                self.visual_token_len = float(self.current_visual_length)
                if self.debug:
                    print(f"[SparseVLM PRUNE] after layer {layer_idx}: visual {len(pre_visual)} -> "
                          f"{self.current_visual_length} keeps={self.keep_schedule[layer_idx]} "
                          f"seq {orig_seq_len} -> {new_len}", flush=True)

        hidden_states = self.norm(hidden_states)

        if is_prefill and n_visual > 0:
            self.visual_token_num = sum(self.layer_visual_token_counts) / len(self.layer_visual_token_counts)
            self.visual_token_len = float(self.current_visual_length or n_visual)
            if self.debug or not self.budget_reachable:
                print(f"[SparseVLM BUDGET] N={n_visual} requestedT={self.target} "
                      f"actual_average={self.visual_token_num:.6f} reachable={self.budget_reachable} "
                      f"layer_tokens={self.layer_visual_token_counts}", flush=True)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=None,
        )
