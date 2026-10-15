# coding=utf-8
"""Shared attention×coverage selector and single-cut S2 for Qwen3-VL.

S1 is routed by the outer wrapper: vit_proj uses pre-merger ViT features, proj
uses merger/projector output. LLM variants default to 2T; single-stage variants keep T.
The MiCo K=6 budget ablation can opt into nT with MICO_STAGE1_MULT=2,3,4,5.

S2: legacy llm_12 uses TOPS two-stage's FIRST LLM layer (Qwen3-VL-8B = 1-based L15);
the projector-space named variants use their named 1-based post-layer cut.
Keep is FastV dynamic_R, not TOPS's scheduled keep, so the
layer-average stays T: R=(T*L-K*N_enter)/(L-K). Score is all-text-attn × coverage.
"""
from __future__ import annotations

import os
from typing import List

import torch
import torch.nn.functional as F

from .modeling_qwen3_vl_coverprune2 import CoverPrune2Qwen3VLModel
def compute_keep_from_avg(t_avg: int, n_enter: int, k_layer: int, n_layers: int) -> int:
    """FastV average budget: T*L = K*N + (L-K)*R → R=(T*L-K*N)/(L-K)."""
    if n_layers <= k_layer:
        return max(1, min(int(t_avg), n_enter))
    r = (float(t_avg) * n_layers - float(k_layer) * n_enter) / float(n_layers - k_layer)
    return max(1, min(int(round(r)), n_enter))


def tops_first_prune_layer(scale: str) -> int:
    """Physical cut used by the historical ``*_llm_12`` baseline.

    It intentionally does not follow TOPS_SCHEDULE: the legacy method name denotes
    the already-reported baseline (8B L15), whereas new named variants are literal
    physical post-layer cuts.
    """
    return {"8b": 15, "7b": 12, "4b": 8, "2b": 12}.get(scale, 15)


def inner_weighted_raw_cls_select(
    features: torch.Tensor,
    raw_cls_attn: torch.Tensor,
    keep: int,
    attn_h: bool = False,
) -> torch.Tensor:
    """Greedy keep-`keep` by Σ_v raw_cls[v] * max(0, κ(v,i) − m_v). CLS is not min-maxed."""
    n_tokens = features.shape[0]
    keep = int(max(1, min(int(keep), n_tokens)))
    feats = F.normalize(features.float(), dim=-1)
    sim = (feats @ feats.t()).clamp(min=0.0)
    w = raw_cls_attn.float().clamp(min=0.0)
    if attn_h:
        # Keep H in the pre-merger ViT space used by this S1 selector; the
        # projector output must not leak into a ViT-space ablation.
        h = features.detach().float().norm(dim=-1)
        if h.shape != w.shape:
            raise ValueError(f"S1 H shape {tuple(h.shape)} != attention shape {tuple(w.shape)}")
        w = w * h
    device = features.device
    available = torch.ones(n_tokens, dtype=torch.bool, device=device)
    max_cov = torch.zeros(n_tokens, device=device, dtype=sim.dtype)
    seed = int(torch.argmax((w.unsqueeze(0) * sim).sum(dim=1)).item())
    selected: List[int] = [seed]
    available[seed] = False
    max_cov = torch.maximum(max_cov, sim[seed])
    for _ in range(keep - 1):
        if not available.any():
            break
        gain = (w.unsqueeze(0) * (sim - max_cov.unsqueeze(0)).clamp(min=0.0)).sum(dim=1)
        gain = gain.masked_fill(~available, float("-inf"))
        idx = int(torch.argmax(gain).item())
        selected.append(idx)
        available[idx] = False
        max_cov = torch.maximum(max_cov, sim[idx])
    mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)
    mask[torch.tensor(selected, device=device, dtype=torch.long)] = True
    return mask


class AttnCovVitProjLlm12Qwen3VLModel(CoverPrune2Qwen3VLModel):
    """Single post-layer prune; legacy default is TOPS-first; S2 = text_attn×coverage."""

    def __init__(
        self,
        config,
        visual_token_num: int,
        k_layer: int | None = None,
        method_name: str | None = None,
        attn_h: bool = False,
        single_stage: bool = False,
    ):
        super().__init__(config, visual_token_num)
        n_layers = getattr(config, "num_hidden_layers", None) or config.text_config.num_hidden_layers
        try:
            t_avg = int(visual_token_num)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"visual_token_num must be an integer, got {visual_token_num!r}") from exc
        self.single_stage = bool(single_stage)
        self.attn_h = bool(attn_h)
        self.stage1_multiplier = 1 if self.single_stage else 2
        if not self.single_stage and method_name == "attn_H_cov_vit_proj_llm_6":
            # Only this ablation arm consumes the opt-in; existing methods retain their budgets.
            raw_mult = os.environ.get("MICO_STAGE1_MULT", "2")
            try:
                self.stage1_multiplier = int(raw_mult)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"MICO_STAGE1_MULT must be one of 2,3,4,5, got {raw_mult!r}") from exc
            if self.stage1_multiplier not in (2, 3, 4, 5):
                raise ValueError(f"MICO_STAGE1_MULT must be one of 2,3,4,5, got {raw_mult!r}")
            if self.stage1_multiplier != 2:
                self.stage1_keep_tokens = max(1, t_avg * self.stage1_multiplier)
                self.stage1_expected_tokens = min(self.visual_token_length, self.stage1_keep_tokens)
        if self.single_stage:
            # attn_cov_vit_proj / attn_H_cov_vit_proj: S1 alone prunes straight to T, so there is no
            # post-layer locus to size. K is deliberately NOT defaulted here -- falling through to
            # tops_first_prune_layer would re-introduce a second cut the method name does not name.
            self.K = None
            self._attn_cov_log_name = method_name or "attn_cov_vit_proj"
            self.pruning_layers: Dict[int, int] = {}
            print(
                f"[{self._attn_cov_log_name}] scale={self.scale} L={n_layers} single-stage: "
                f"S1 -> T={t_avg}, S2 disabled (no in-LLM prune)"
            )
            return
        # Do not reinterpret the existing _12 alias: explicit K is opt-in for the new sweep arms.
        self.K = tops_first_prune_layer(self.scale) if k_layer is None else int(k_layer)
        if not 1 <= self.K < n_layers:
            raise ValueError(f"S2 post-layer K must be in [1, {n_layers - 1}], got {self.K}")
        # Explicit-K variants must retain their ViT-space method identity in
        # logs; otherwise a valid ViT S1 run looks indistinguishable from a
        # projector-space arm during provenance checks.
        self._attn_cov_log_name = method_name or (
            "attn_cov_vit_proj_llm_12" if k_layer is None else f"attn_cov_proj_llm_{self.K}"
        )
        n_enter = max(1, t_avg * self.stage1_multiplier)
        keep_r = compute_keep_from_avg(t_avg, n_enter, self.K, n_layers)
        # One locus only. TOPS's second prune (8B L21) is intentionally unused.
        self.pruning_layers = {self.K: keep_r}
        self.T_avg = t_avg
        print(
            f"[{self._attn_cov_log_name}] scale={self.scale} L={n_layers} "
            f"S2=post-layer L{self.K} N_enter={n_enter} T={t_avg} keep={keep_r} "
            f"(R=(T*L-K*N_enter)/(L-K), not TOPS scheduled keep)"
        )

    def set_visual_token_positions(self, positions: torch.LongTensor) -> None:
        super().set_visual_token_positions(positions)
        if self.single_stage:
            # No in-LLM locus exists, so there is no R to refresh. Silently sizing one here would
            # arm an S2 cut the method does not have.
            self.pruning_layers = {}
            return
        # Recompute R from the actual post-S1 count (may be below nT for small images).
        n_enter = int(positions.numel())
        n_layers = len(self.layers)
        keep_r = compute_keep_from_avg(self.T_avg, n_enter, self.K, n_layers)
        self.pruning_layers = {self.K: keep_r}

    def _rdc_greedy_select(
        self,
        attn: torch.Tensor,
        post_hidden: torch.Tensor,
        retained_num: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Inner-weighted text_attn×coverage. Raw attn (no min-max); no TOPS diversity term."""
        if self.single_stage:
            raise RuntimeError(
                f"{self._attn_cov_log_name} is single-stage: S2 selection must never be reached. "
                "pruning_layers is empty, so an S2 call means the prune gate was misconfigured."
            )
        vis = post_hidden[0]
        w = attn.detach().float().reshape(-1).clamp(min=0.0)
        if w.numel() != vis.shape[0]:
            raise ValueError(f"S2 attn {tuple(w.shape)} != visual {vis.shape[0]}")
        if self.attn_h:
            # S2's H is post-layer because the K cut is performed after that
            # decoder layer; this intentionally differs from pre-merger S1 H.
            w = w * vis.detach().float().norm(dim=-1)
        wsum = w.sum()
        if wsum > 0:
            w = w / wsum
        else:
            w = torch.full_like(w, 1.0 / max(vis.shape[0], 1))
        feats = F.normalize(vis.float(), dim=-1)
        sim = (feats @ feats.t()).clamp(min=0.0)
        keep = int(max(1, min(int(retained_num), vis.shape[0])))
        available = torch.ones(vis.shape[0], dtype=torch.bool, device=device)
        max_cov = torch.zeros(vis.shape[0], device=device, dtype=sim.dtype)
        seed = int(torch.argmax((w.unsqueeze(0) * sim).sum(dim=1)).item())
        selected: List[int] = [seed]
        available[seed] = False
        max_cov = torch.maximum(max_cov, sim[seed])
        for _ in range(keep - 1):
            if not available.any():
                break
            gain = (w.unsqueeze(0) * (sim - max_cov.unsqueeze(0)).clamp(min=0.0)).sum(dim=1)
            gain = gain.masked_fill(~available, float("-inf"))
            idx = int(torch.argmax(gain).item())
            selected.append(idx)
            available[idx] = False
            max_cov = torch.maximum(max_cov, sim[idx])
        log_i = getattr(self, "_attn_cov_12_s2_log_i", 0)
        self._attn_cov_12_s2_log_i = log_i + 1
        if log_i == 0 or os.environ.get("ENABLE_DEBUG", "0") == "1":
            print(
                f"[{self._attn_cov_log_name}] S2 L{self.K} N={vis.shape[0]} "
                f"keep={keep} (text_attn{'*|h|' if self.attn_h else ''}*coverage)",
                flush=True,
            )
        return torch.tensor(selected, device=device, dtype=torch.long)
