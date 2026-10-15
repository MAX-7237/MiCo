"""One versioned DeltaNet write-beta score; it never changes native tensors.

Q/K are actual post-convolution kernel inputs with native head repetition.
The score, unlike the native kernel, explicitly performs L2 and reductions in
FP32. This defines a research proxy, not exact recurrent output attribution.
"""
import math
import torch

REDUCTION = 'postconv_qk_l2_fp32_eps1e-6_abs_scaled_dot_write_beta_mean_question_heads'
UNIT_REDUCTION = 'postconv_qk_l2_fp32_eps1e-6_abs_scaled_dot_unit_mean_query_heads'


def linear_beta_quality(query, key, beta, question_indices, visual_indices, *, score_beta_mode='native_beta'):
    if score_beta_mode not in ('native_beta', 'unit'):
        raise ValueError('score_beta_mode must be native_beta or unit')
    if (not all(isinstance(x, torch.Tensor) for x in (query, key, beta))
            or query.ndim != 4 or query.shape[0] != 1
            or any(d < 1 for d in query.shape) or key.shape != query.shape
            or beta.shape != query.shape[:-1]
            or query.device != key.device or query.device != beta.device
            or not all(x.is_floating_point() for x in (query, key, beta))):
        raise ValueError('linear beta expects Q/K [1,S,H,D] and beta [1,S,H]')
    if (not all(bool(torch.isfinite(x).all()) for x in (query, key, beta))
            or not bool(((beta >= 0) & (beta <= 1)).all())):
        raise ValueError('linear beta native inputs are nonfinite or beta outside [0,1]')
    length = query.shape[1]
    for indices in (question_indices, visual_indices):
        if (not isinstance(indices, torch.Tensor) or indices.dtype != torch.long
                or indices.ndim != 1 or not indices.numel()
                or indices.device != query.device
                or int(indices[0]) < 0 or int(indices[-1]) >= length
                or not bool((indices[1:] > indices[:-1]).all())):
            raise ValueError('linear beta indices must be nonempty sorted unique positions')
    if int(question_indices[0]) <= int(visual_indices[-1]):
        raise ValueError('linear beta question queries must follow every visual token')
    # Work on derived FP32 tensors only, never mutate an observed native input.
    q = query.float().index_select(1, question_indices)
    k = key.float().index_select(1, visual_indices)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    dot = torch.einsum('bthd,bihd->bhti', q, k) / math.sqrt(query.shape[-1])
    writes = beta.float().index_select(1, visual_indices).transpose(1, 2)
    if score_beta_mode == 'unit':
        # This changes only the auxiliary score, never the native DeltaNet gate.
        writes = torch.ones_like(writes)
    quality = (dot.abs() * writes.unsqueeze(2)).mean(dim=(0, 1, 2))
    if not bool(torch.isfinite(quality).all() & (quality >= 0).all()):
        raise ValueError('linear beta score is nonfinite')
    return quality
