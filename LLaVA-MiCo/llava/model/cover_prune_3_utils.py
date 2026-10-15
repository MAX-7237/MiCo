"""Single-stage COVER-Prune-3 selection in CLIP's joint embedding space."""

import torch


def cover_prune_3_select(
    clip_image_features,
    clip_text_features,
    cls_attn,
    target_token_num,
    anti_text: bool = False,
):
    """Select exactly T patches with CLIP coverage and question relevance.

    For candidate ``i``, the marginal score is

    ``sum_v cls_attn[v] * text_weight[v] * coverage_gain[i, v]``.

    ``text_weight`` is the mean positive CLIP cosine between the question and
    visual token ``v`` (``text_sim``). When ``anti_text=True``, use
    ``1 - text_sim`` instead — the same complementary-text direction as local
    CDPruner / ``cdp3`` (which does ``(-tv_sim).mean`` then min-max).

    Both visual coverage and question-to-visual similarity are computed in
    CLIP's joint image/text embedding space.
    """
    if clip_image_features.dim() != 2:
        raise ValueError(
            "clip_image_features must be [N,C_clip], got "
            f"{tuple(clip_image_features.shape)}"
        )
    if (
        clip_text_features.dim() != 2
        or clip_text_features.shape[0] == 0
        or clip_text_features.shape[1] != clip_image_features.shape[1]
    ):
        raise ValueError(
            "clip_text_features must be non-empty [M,C_clip] and match "
            "clip_image_features; got "
            f"{tuple(clip_text_features.shape)} vs "
            f"{tuple(clip_image_features.shape)}"
        )
    if (
        cls_attn.dim() != 1
        or cls_attn.shape[0] != clip_image_features.shape[0]
    ):
        raise ValueError(
            "cls_attn must be [N] matching clip_image_features, got "
            f"{tuple(cls_attn.shape)} vs "
            f"{tuple(clip_image_features.shape)}"
        )

    num_tokens = int(clip_image_features.shape[0])
    target_token_num = max(1, min(int(target_token_num), num_tokens))
    device = clip_image_features.device

    image = clip_image_features.float()
    image = image / image.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    text = clip_text_features.float().to(device)
    text = text / text.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    visual_similarity = torch.clamp(image @ image.t(), min=0.0)
    # text_sim(v) in [0, 1] after ReLU-cosine + mean over text segments
    text_similarity = torch.clamp(text @ image.t(), min=0.0).mean(dim=0)
    text_weight = (1.0 - text_similarity) if anti_text else text_similarity

    cls = cls_attn.float().to(device)
    cls = (cls - cls.min()) / (cls.max() - cls.min() + 1e-8)
    token_weight = cls.to(text_weight.dtype) * text_weight

    available = torch.ones(num_tokens, dtype=torch.bool, device=device)
    selected = torch.empty(target_token_num, dtype=torch.long, device=device)
    covered = torch.zeros(
        num_tokens, dtype=visual_similarity.dtype, device=device
    )

    for step in range(target_token_num):
        coverage_gain = torch.clamp(
            visual_similarity - covered.unsqueeze(0), min=0.0
        )
        score = (token_weight.unsqueeze(0) * coverage_gain).sum(dim=1)
        score[~available] = -float("inf")
        chosen = int(torch.argmax(score))
        selected[step] = chosen
        available[chosen] = False
        covered = torch.maximum(covered, visual_similarity[chosen])

    return torch.sort(selected).values
