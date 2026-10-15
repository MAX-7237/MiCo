"""ApET (Adaptive Patch-pruning with Efficient Token merging) selector + merger.

Official reference: `Maqkccx/ApET @ 70cc01b215f4d74c5661c699919fe3aeda48bed0`,
as vendored in `/path/to/local-assets`
(LLaVA path). The math below is the same arithmetic, lifted out of the LLaVA
hook machinery into pure tensor functions the two VLMEvalKit model paths can
call directly.

ApET is a *two-stage* method:
  S1 (`apet_stage1`): farthest-point-sampled `basis` seeds, linear reconstruction
      ranking, keep the top-`keep`, then absorb the rest into their nearest kept
      seed by cosine similarity. Runs on ViT-space features *before* the projector.
  S2 (`apet_stage2`): the same merge applied once, mid-decoder, to the visual span
      of the hidden states. Everything before/after the visual span is untouched.

Schedule (matches the LLaVA run and the paper table): with nominal budget T,
S1 keeps floor(3T/2) and S2 cuts to floor(T/2) after layer L//2, so the
average visual-token count across layers equals T -- the same average the
single-stage baselines are measured at.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def _nearest_cosine(rest, targets, workspace_bytes=64 * 1024 * 1024):
    """Original cosine/argmax arithmetic, with bounded broadcast row tiles.

    ViT merger groups have thousands of channels: broadcasting all rest/kept
    pairs materializes multi-GiB intermediates. Keep the reference dtype and
    cosine reduction (including first-index tie breaking), unlike a matmul
    rewrite that can change FP16 nearest-neighbour assignments.
    """
    if rest.ndim != 3 or targets.ndim != 3:
        raise ValueError('cosine inputs must be [B,N,D]')
    if rest.shape[0] != targets.shape[0] or rest.shape[2] != targets.shape[2]:
        raise ValueError('cosine batch/channel dimensions must match')
    if targets.shape[1] == 0:
        raise ValueError('cosine requires at least one target')
    batch, count, dim = rest.shape
    # Allow three broadcast-sized temporaries within the requested workspace.
    row_bytes = 3 * batch * targets.shape[1] * dim * rest.element_size()
    chunk = max(1, workspace_bytes // max(1, row_bytes))
    nearest = torch.empty((batch, count), device=rest.device, dtype=torch.long)
    for start in range(0, count, chunk):
        similarity = F.cosine_similarity(
            rest[:, start:start + chunk].unsqueeze(2), targets.unsqueeze(1), dim=3)
        nearest[:, start:start + chunk] = similarity.argmax(dim=2)
    return nearest


def _check_features(x, keep, basis, batched_crops=False):
    if x.ndim != 3 or (x.shape[0] != 1 and not batched_crops) or not x.is_floating_point():
        raise ValueError('ApET requires floating features [1,N,D]')
    if not 0 < basis < keep <= x.shape[1]:
        raise ValueError('require 0 < basis < keep <= visual token count')


def _fps(x, count, seed):
    b, n, _ = x.shape
    generator = None
    if seed is not None:
        generator = torch.Generator(device=x.device).manual_seed(seed)
    indices = torch.zeros(b, count, dtype=torch.long, device=x.device)
    distance = torch.full((b, n), 1e10, device=x.device)
    farthest = torch.randint(0, n, (b,), device=x.device, generator=generator)
    for i in range(count):
        indices[:, i] = farthest
        centroid = x[torch.arange(b, device=x.device), farthest].unsqueeze(1)
        distance = torch.min(distance, ((x - centroid) ** 2).sum(-1))
        farthest = distance.max(1)[1]
    return indices


@torch.no_grad()
def apet_stage1(features, keep=96, basis=10, seed=None, fp32_merge=False):
    """Patch-only [B,N,D] -> full-N merged features (FP16 math) + sorted keep indices.

    Caller owns projection then gather. Input features are never modified.
    Matches the official S1 Boolean-order basis assignment and FP16 merging
    by default. Qwen's BF16 pre-merger ViT activations need the explicit
    ``fp32_merge`` adaptation: cosine/accumulation stay FP32 and the result
    returns to the input dtype. This is ApET†, not bitwise-reference FP16.
    """
    _check_features(features, keep, basis, batched_crops=True)
    batch, count, dim = features.shape
    if keep >= count:  # dense S1 special case (author keeps the whole crop)
        return features.clone(), torch.arange(count, device=features.device)[None].expand(batch, -1)
    fps_indices = _fps(features, basis, seed)
    seeds = features.gather(1, fps_indices.unsqueeze(-1).expand(-1, -1, dim)).float()
    work = features.float().clone()
    gram = torch.matmul(seeds, seeds.transpose(-1, -2))
    gram.diagonal(dim1=-2, dim2=-1).add_(1e-5)
    rhs = torch.matmul(seeds, work.transpose(-1, -2))
    coefficients = torch.linalg.solve(gram, rhs).transpose(-1, -2)
    reconstruction = torch.matmul(coefficients, seeds)
    rank = torch.topk(torch.norm(work - reconstruction, dim=-1), k=keep, dim=1)[1]
    if not fp32_merge:
        work, seeds = work.half(), seeds.half()
    slots = torch.zeros(batch, count, dtype=torch.bool, device=work.device)
    slots.scatter_(1, rank[:, -basis:], True)
    for b in range(batch):
        work[b][slots[b]] = seeds[b]
    selected = torch.zeros_like(slots).scatter_(1, rank, True)
    targets = torch.stack([work[b][rank[b][:-basis]] for b in range(batch)])
    rest = torch.stack([work[b][~selected[b]] for b in range(batch)])
    if rest.shape[1] != 0:
        nearest = _nearest_cosine(rest, targets)
        counts = torch.zeros(batch, keep - basis, device=work.device, dtype=torch.int)
        targets.scatter_add_(1, nearest.unsqueeze(-1).expand(-1, -1, dim), rest)
        counts.scatter_add_(1, nearest, torch.ones_like(nearest, dtype=counts.dtype))
        targets /= 1 + counts.unsqueeze(2)
        for b in range(batch):
            work[b, rank[b][:-basis]] = targets[b]
    return (work.to(features.dtype) if fp32_merge else work), torch.sort(rank, dim=1)[0]


@torch.no_grad()
def apet_stage2(hidden_states, image_start, image_tokens, keep=32, basis=10, seed=None):
    """Compress the visual span of [B,seq,D], returning merged sequence + visual indices.

    FP32 merge, selected visual values cast to half, then restored to the
    sequence dtype. All before/after-image text stays unchanged.
    """
    if image_start < 0 or image_start + image_tokens > hidden_states.shape[1]:
        raise ValueError('visual span outside sequence')
    visual = hidden_states[:, image_start:image_start + image_tokens]
    _check_features(visual, keep, basis)
    _, count, dim = visual.shape
    if keep >= count:
        idx = torch.arange(count, device=visual.device)[None]
        return hidden_states.clone(), idx
    fps_indices = _fps(visual, basis, seed)
    seeds = visual.gather(1, fps_indices.unsqueeze(-1).expand(-1, -1, dim))[0].float()
    work = visual[0].float().clone()
    gram = torch.matmul(seeds, seeds.t())
    gram.diagonal().add_(1e-5)
    coefficients = torch.linalg.solve(gram, torch.matmul(seeds, work.t())).t()
    rank = torch.topk(torch.norm(work - torch.matmul(coefficients, seeds), dim=-1), k=keep)[1]
    work[rank[-basis:]] = seeds
    selected = torch.zeros(count, dtype=torch.bool, device=work.device).scatter_(0, rank, True)
    targets = work[rank[:-basis]]
    rest = work[~selected]
    if rest.shape[0]:
        nearest = _nearest_cosine(rest[None], targets[None])[0]
        counts = torch.zeros(keep - basis, device=work.device, dtype=torch.int)
        merged = torch.zeros_like(targets)
        merged += targets
        merged.scatter_add_(0, nearest.unsqueeze(-1).expand(-1, dim), rest)
        counts.scatter_add_(0, nearest, torch.ones_like(nearest, dtype=counts.dtype))
        merged /= 1 + counts.unsqueeze(1)
        work[rank[:-basis]] = merged
    indices = torch.sort(rank)[0]
    compact = work.half()[indices].to(hidden_states.dtype)[None]
    result = torch.cat([hidden_states[:, :image_start], compact,
                        hidden_states[:, image_start + image_tokens:]], dim=1)
    return result, indices[None]


def apet_schedule(budget: int, layers: int) -> dict:
    """Return the ApET S1/S2 schedule for a nominal per-layer budget T."""
    stage1 = 3 * budget // 2
    stage2 = budget // 2
    prune_after = layers // 2
    return dict(budget=budget, stage1=stage1, stage2=stage2, prune_after=prune_after,
                layers=layers, average=(stage1 * prune_after + stage2 * (layers - prune_after)) / layers)
