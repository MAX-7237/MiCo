"""MMTok video port: the published coverage selector, independently per frame.

Adapted from MMTok (Copyright 2025 Zoom Communications, Inc.; Apache-2.0),
semantic_selector.py and text_processor.py. The image implementation is also
present in LLaVA-TOPS-ICLR/llava/model/llava_arch.py (method ``mmtok``).

Video adaptation: Q uses the current vision encoder's pre-projector features,
pooled with the video's existing spatial pooling. P uses post-projector pooled
features and keyword-filtered question embeddings. No LLM layer is pruned.
This is an explicit video port, not an official video benchmark reproduction.
"""
import hashlib
import json
import os
from pathlib import Path
import re

import torch
import torch.nn.functional as F


NON_VISUAL_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "it", "they", "he", "she",
    "we", "you", "i", "what", "where", "when", "why", "how", "who", "which",
    "and", "or", "but", "so", "if", "then", "question", "answer", "to", "for",
    "with", "by", "from", "at", "into", "onto", "upon",
}
PROMPTS = (
    "Answer the question using a single word or phrase.",
    "Answer with the option's letter from the given choices directly.",
)


def extract_keywords_simple(text):
    """Match upstream VQATextProcessor, including its stopword-only fallback."""
    if not text:
        return ""
    for prompt in PROMPTS:
        text = text.replace(prompt, "")
    words = re.findall(r"\b\w+\b", text)
    filtered = [word for word in words if word.lower() not in NON_VISUAL_WORDS]
    return " ".join(filtered) if filtered else text


def encode_question(question, tokenizer, embed_tokens):
    """Use the request text, before image/system/template/time instructions."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("MMTok requires the current request's nonempty question")
    keywords = extract_keywords_simple(question)
    if not keywords.strip() or tokenizer is None:
        raise ValueError("MMTok requires nonempty keywords and the LLM tokenizer")
    encoded = tokenizer(keywords.split(), is_split_into_words=True,
                        return_tensors="pt", padding=True, truncation=True)
    ids = encoded["input_ids"].to(embed_tokens.weight.device)
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("MMTok question tokenizer must return one input sequence")
    embeddings = embed_tokens(ids)[0]
    if ids.shape[1] > 1 and ids[0, 0].item() == getattr(tokenizer, "bos_token_id", None):
        embeddings = embeddings[1:]
    if embeddings.shape[0] == 0:
        raise ValueError("MMTok question has no embedded tokens")
    return embeddings


def select_indices(projected, preprojector, text, keep, *, alpha=0.5,
                   tv_temp=0.02, vv_temp=0.2):
    """Return T×K unique indices in original order, for T separate frames."""
    if projected.ndim != 3 or preprojector.ndim != 3 or text.ndim != 2:
        raise ValueError("Expected frame/token/features and text/features tensors")
    if projected.shape[:2] != preprojector.shape[:2]:
        raise ValueError("P and Q features must refer to exactly the same frame tokens")
    frames, count, dim = projected.shape
    if frames < 1 or count < 1 or text.shape[0] < 1 or text.shape[1] != dim:
        raise ValueError("Invalid visual/text feature shape")
    if isinstance(keep, bool) or not isinstance(keep, int) or not 1 <= keep <= count:
        raise ValueError("MMTok keep must be an integer within the actual per-frame token count")
    if not (0 <= alpha < float("inf") and 0 < tv_temp < float("inf")
            and 0 < vv_temp < float("inf")):
        raise ValueError("Invalid MMTok coverage parameters")
    if not all(torch.isfinite(tensor).all().item() for tensor in (projected, preprojector, text)):
        raise ValueError("MMTok received nonfinite features")
    device = projected.device
    x = F.normalize(projected.float(), p=2, dim=-1, eps=1e-8)
    xc = F.normalize(preprojector.to(device).float(), p=2, dim=-1, eps=1e-8)
    z = F.normalize(text.to(device).float(), p=2, dim=-1, eps=1e-8)
    indices = []
    for frame in range(frames):
        p = F.softmax((z @ x[frame].T) / tv_temp, dim=1) / z.shape[0]
        q = F.softmax((xc[frame] @ xc[frame].T) / vv_temp, dim=1) / count
        combined = torch.cat([p, alpha * q], dim=0)
        best = torch.zeros(combined.shape[0], device=device, dtype=combined.dtype)
        score_mask = torch.zeros(count, device=device, dtype=combined.dtype)
        chosen = torch.empty(keep, device=device, dtype=torch.long)
        for step in range(keep):
            gain = (combined - best[:, None]).clamp_min(0).sum(0) + score_mask
            idx = torch.argmax(gain)
            chosen[step] = idx
            best = torch.maximum(best, combined[:, idx])
            score_mask[idx] = -float("inf")
        indices.append(chosen.sort().values)
    return torch.stack(indices)


def emit_trace(record):
    """Append one trace to the process's file; do not include question text."""
    target = os.environ.get("VIDEO_TRACE_DIR")
    if not target:
        return
    rank = int(os.environ.get("RANK", "0"))
    if rank < 0:
        raise ValueError("Invalid trace rank")
    path = Path(target)
    path.mkdir(parents=True, exist_ok=True)
    with (path / f"rank-{rank}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")


@torch.no_grad()
def prune_video(model, projected):
    """Integrate with the existing single-video, batch-size-one LLaVA adapter."""
    preprojector = getattr(model, "_mmtok_preprojector", None)
    model._mmtok_preprojector = None  # Each request must supply fresh vision output.
    if preprojector is None:
        raise ValueError("MMTok requires fresh pre-projector features from this request")
    question = getattr(model, "_mmtok_question", None)
    text = encode_question(question, getattr(model, "mmtok_tokenizer", None),
                           model.get_model().embed_tokens)
    pooled = model.get_2dPool(preprojector)
    selected = select_indices(projected, pooled, text, model.visual_token_num)
    rows = torch.arange(projected.shape[0], device=projected.device)[:, None]
    output = projected[rows, selected]
    frames, original, _ = projected.shape
    keep = output.shape[1]
    request_id = dict(getattr(model, "_tops_request_id", {}))
    if os.environ.get("VIDEO_TRACE_DIR") and not all(key in request_id for key in ("task", "split", "doc_id")):
        raise ValueError("MMTok trace requires the current task/split/doc_id")
    record = {
        "method": "mmtok_video_port", "request_id": request_id,
        "frames": frames, "tokens_before_per_frame": original,
        "target_tokens_per_frame": model.visual_token_num,
        "selected_counts": [keep] * frames, "total_visual_tokens": frames * keep,
        "selected_indices": selected.cpu().tolist(),
        "question_tokens": text.shape[0],
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "alpha": 0.5, "tv_temp": 0.02, "vv_temp": 0.2,
        "preprojector_dim": pooled.shape[-1], "projector_dim": projected.shape[-1],
        "vision_feature_space": "current_vision_encoder_preprojector_spatial_pooled",
    }
    model._mmtok_last_trace = record
    emit_trace(record)
    print(f"[MMTOK_VIDEO_TRACE] frames={frames} before={original} keep={keep} "
          f"total={frames * keep} question_tokens={text.shape[0]}")
    return output
