"""MiCo source-weighted coverage and a CLS-free Qwen3.5 vision adapter.

The adapter follows the existing Qwen3-VL MiCo implementation: final-block
attention is averaged over native 2x2 query/key groups, and the coverage features
are the mean of each four-patch pre-merger group. This is an explicit Qwen3.5
adaptation, not an upstream MiCo implementation. VisionTap
observes native execution and does not replace its attention or merger output.
"""
from __future__ import annotations

from importlib import import_module
from numbers import Integral

import torch
import torch.nn.functional as F


@torch.no_grad()
def coverage_select(features: torch.Tensor, quality: torch.Tensor, keep: int,
                    *, normalize_weights: bool = False) -> torch.Tensor:
    """Return sorted original indices for greedy norm*quality coverage.

Both S1 and S2 use this primitive: S1 quality is pooled vision attention; S2
    quality is mean prompt attention with no carried-over S1 factor. Pass
    normalize_weights=True for Qwen3's S2 normalization and zero-sum fallback.
    Feature norms precede normalization. Selection indices remain on device.
"""
    if not isinstance(features, torch.Tensor) or not isinstance(quality, torch.Tensor):
        raise TypeError('features and quality must be tensors')
    if features.ndim != 2 or quality.ndim != 1 or features.shape[0] != quality.numel():
        raise ValueError('expected features[N,D] and quality[N]')
    n, d = features.shape
    if n == 0 or d == 0 or features.device != quality.device:
        raise ValueError('nonempty features and quality must share a device')
    if isinstance(keep, bool) or not isinstance(keep, Integral) or not 1 <= keep <= n:
        raise ValueError('keep must be an integer in [1,N]')
    x, q = features.float(), quality.float()
    if not bool(torch.isfinite(x).all() & torch.isfinite(q).all() & (q >= 0).all()):
        raise ValueError('features must be finite and quality finite and nonnegative')
    weights = x.norm(dim=-1) * q
    if normalize_weights:
        total = weights.sum()
        weights = weights / total if total > 0 else torch.full_like(weights, 1.0 / n)
    normalized = F.normalize(x, dim=-1)
    similarity = (normalized @ normalized.T).clamp_min(0)
    coverage = torch.zeros(n, dtype=similarity.dtype, device=x.device)
    available = torch.ones(n, dtype=torch.bool, device=x.device)
    selected = torch.empty(int(keep), dtype=torch.long, device=x.device)
    for step in range(int(keep)):
        gains = ((similarity - coverage.unsqueeze(0)).clamp_min(0) * weights.unsqueeze(0)).sum(dim=1)
        idx = gains.masked_fill(~available, -float('inf')).argmax().reshape(1)
        selected[step:step + 1].copy_(idx)
        available.scatter_(0, idx, False)
        coverage = torch.maximum(coverage, similarity.index_select(0, idx).squeeze(0))
    return selected.sort().values


class VisionTap:
    """Capture one image's pre-merger features and pooled incoming attention.

Construct once on the actual ``Qwen3_5VisionModel`` instance. Call ``pop`` after
each vision forward to consume and clear the captured data. Every new forward
clears stale data, including unsupported image/video inputs. The caller owns
this hook object's lifetime; ``close`` removes its registrations.
"""
    def __init__(self, visual):
        self.visual = visual
        self.merge_size = int(visual.config.spatial_merge_size)
        if self.merge_size != 2:
            raise ValueError('this Qwen3.5 port requires native 2x2 merger groups')
        self._rotary = import_module(type(visual).__module__).apply_rotary_pos_emb_vision
        self._features = None
        self._quality = None
        self._raw_count = None
        self._handles = [
            visual.register_forward_pre_hook(self._begin, with_kwargs=True),
            visual.blocks[-1].attn.register_forward_pre_hook(self._attention, with_kwargs=True),
            visual.register_forward_hook(self._finish, with_kwargs=True),
        ]

    def _clear(self):
        self._features = self._quality = self._raw_count = None

    def _begin(self, module, args, kwargs):
        self._clear()
        grid = kwargs.get('grid_thw', args[1] if len(args) > 1 else None)
        if not isinstance(grid, torch.Tensor) or grid.ndim != 2 or grid.shape != (1, 3):
            raise ValueError('VisionTap supports exactly one image per forward')
        t, h, w = map(int, grid[0].tolist())
        if t != 1 or h <= 0 or w <= 0 or h % 2 or w % 2:
            raise ValueError('VisionTap requires one image with valid 2x2 patch groups')
        self._raw_count = h * w

    @torch.no_grad()
    def _attention(self, module, args, kwargs):
        # This hook receives norm1(hidden), exactly the native attention input.
        hidden = args[0] if args else kwargs['hidden_states']
        cu_seqlens = kwargs.get('cu_seqlens', args[1] if len(args) > 1 else None)
        if self._raw_count is None or hidden.shape[0] != self._raw_count:
            raise ValueError('vision attention sequence does not match the image grid')
        if cu_seqlens is None or cu_seqlens.numel() != 2:
            raise ValueError('VisionTap does not mix multiple attention chunks')
        positions = kwargs.get('position_embeddings')
        if positions is None:
            raise ValueError('native vision rotary position embeddings are required')
        n = hidden.shape[0]
        q, k, _ = module.qkv(hidden).reshape(n, 3, module.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q, k = self._rotary(q, k, positions[0], positions[1])
        q, k = q.transpose(0, 1), k.transpose(0, 1)
        logits = (q @ k.transpose(-1, -2)) * module.scaling
        probs = F.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
        # Match Qwen3's grouping/reduction order, including probability dtype.
        grouped = probs.reshape(module.num_heads, n // 4, 4, n // 4, 4).mean(dim=(2, 4))
        self._quality = grouped.mean(dim=1).mean(dim=0).detach()

    def _finish(self, module, args, kwargs, output):
        raw = output.last_hidden_state
        if self._quality is None or raw.ndim != 2 or raw.shape[0] != self._raw_count:
            self._clear()
            raise RuntimeError('complete matching vision feature/attention capture is required')
        # Exact Qwen3 vit_proj feature space, not concatenation/projector output.
        self._features = raw.detach().float().reshape(raw.shape[0] // 4, 4, raw.shape[1]).mean(dim=1)
        if self._features.shape[0] != self._quality.numel():
            self._clear()
            raise RuntimeError('native merger feature/quality grouping mismatch')

    def pop(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._features is None or self._quality is None:
            self._clear()
            raise RuntimeError('no complete unconsumed vision capture')
        result = self._features, self._quality
        self._clear()
        return result

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._clear()
