"""MiCo visual-token selection with an explicit decoder cut and layer-average budget."""

from __future__ import annotations
import os
from typing import List
import torch
import torch.nn.functional as F
from .modeling_qwen3_vl_mico import MiCoQwen3VLModel


def compute_keep_from_avg(t_avg: int, n_enter: int, k_layer: int, n_layers: int) -> int:
    """Layer-average token budget: T*L = K*N + (L-K)*R → R=(T*L-K*N)/(L-K)."""
    if n_layers <= k_layer:
        return max(1, min(int(t_avg), n_enter))
    r = (float(t_avg) * n_layers - float(k_layer) * n_enter) / float(n_layers - k_layer)
    return max(1, min(int(round(r)), n_enter))


def inner_weighted_raw_cls_select(
    features: torch.Tensor, raw_cls_attn: torch.Tensor, keep: int, mico: bool = False
) -> torch.Tensor:
    """Greedy keep-`keep` by Σ_v raw_cls[v] * max(0, κ(v,i) − m_v). CLS is not min-maxed."""
    n_tokens = features.shape[0]
    keep = int(max(1, min(int(keep), n_tokens)))
    feats = F.normalize(features.float(), dim=-1)
    sim = (feats @ feats.t()).clamp(min=0.0)
    w = raw_cls_attn.float().clamp(min=0.0)
    if mico:
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


class MiCoQwen3VLDecoder(MiCoQwen3VLModel):
    """MiCo pruning at the specified decoder layer."""

    def __init__(
        self,
        config,
        visual_token_num: int,
        k_layer: int | None = None,
        method_name: str | None = None,
        mico: bool = False,
        single_stage: bool = False,
    ):
        super().__init__(config, visual_token_num)
        n_layers = (
            getattr(config, "num_hidden_layers", None) or config.text_config.num_hidden_layers
        )
        try:
            t_avg = int(visual_token_num)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"visual_token_num must be an integer, got {visual_token_num!r}"
            ) from exc
        self.single_stage = bool(single_stage)
        self.mico = bool(mico)
        self.stage1_multiplier = 1 if self.single_stage else 2
        if not self.single_stage and method_name == "mico_6":
            raw_mult = os.environ.get("MICO_STAGE1_MULT", "2")
            try:
                self.stage1_multiplier = int(raw_mult)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"MICO_STAGE1_MULT must be one of 2,3,4,5, got {raw_mult!r}"
                ) from exc
            if self.stage1_multiplier not in (2, 3, 4, 5):
                raise ValueError(f"MICO_STAGE1_MULT must be one of 2,3,4,5, got {raw_mult!r}")
            if self.stage1_multiplier != 2:
                self.stage1_keep_tokens = max(1, t_avg * self.stage1_multiplier)
                self.stage1_expected_tokens = min(self.visual_token_length, self.stage1_keep_tokens)
        if self.single_stage:
            self.K = None
            self._mico_coverage_log_name = method_name or "mico_coverage_vit_proj"
            self.pruning_layers: Dict[int, int] = {}
            print(
                f"[{self._mico_coverage_log_name}] scale={self.scale} L={n_layers} single-stage: S1 -> T={t_avg}, S2 disabled (no in-LLM prune)"
            )
            return
        self.K = int(k_layer)
        if not 1 <= self.K < n_layers:
            raise ValueError(f"S2 post-layer K must be in [1, {n_layers - 1}], got {self.K}")
        self._mico_coverage_log_name = method_name or (
            "mico_coverage_vit_proj_llm_12"
            if k_layer is None
            else f"mico_coverage_proj_llm_{self.K}"
        )
        n_enter = max(1, t_avg * self.stage1_multiplier)
        keep_r = compute_keep_from_avg(t_avg, n_enter, self.K, n_layers)
        self.pruning_layers = {self.K: keep_r}
        self.T_avg = t_avg
        print(
            f"[{self._mico_coverage_log_name}] scale={self.scale} L={n_layers} S2=post-layer L{self.K} N_enter={n_enter} T={t_avg} keep={keep_r} (R=(T*L-K*N_enter)/(L-K))"
        )

    def set_visual_token_positions(self, positions: torch.LongTensor) -> None:
        super().set_visual_token_positions(positions)
        if self.single_stage:
            self.pruning_layers = {}
            return
        n_enter = int(positions.numel())
        n_layers = len(self.layers)
        keep_r = compute_keep_from_avg(self.T_avg, n_enter, self.K, n_layers)
        self.pruning_layers = {self.K: keep_r}

    def _select_visual_tokens(
        self, attn: torch.Tensor, post_hidden: torch.Tensor, retained_num: int, device: torch.device
    ) -> torch.Tensor:
        """Greedy coverage weighted by text attention and current visual feature norms."""
        if self.single_stage:
            raise RuntimeError(
                f"{self._mico_coverage_log_name} is single-stage: S2 selection must never be reached. pruning_layers is empty, so an S2 call means the prune gate was misconfigured."
            )
        vis = post_hidden[0]
        w = attn.detach().float().reshape(-1).clamp(min=0.0)
        if w.numel() != vis.shape[0]:
            raise ValueError(f"S2 attn {tuple(w.shape)} != visual {vis.shape[0]}")
        if self.mico:
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
        log_i = getattr(self, "_mico_coverage_12_s2_log_i", 0)
        self._mico_coverage_12_s2_log_i = log_i + 1
        if log_i == 0 or os.environ.get("ENABLE_DEBUG", "0") == "1":
            print(
                f"[{self._mico_coverage_log_name}] S2 L{self.K} N={vis.shape[0]} keep={keep} (text_attn{('*|h|' if self.mico else '')}*coverage)",
                flush=True,
            )
        return torch.tensor(selected, device=device, dtype=torch.long)
