"""MiCo stage-one selection, attention-weighted decoder coverage, and layer-average token budgets."""

from __future__ import annotations
import math
import os
from types import MethodType
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    Cache,
    DynamicCache,
    LlamaModel,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)


def compute_keep_from_avg(t_avg: int, n_enter: int, k_layer: int, n_layers: int) -> int:
    """Solve layer-average average budget for post-layer-K keep count R.

    T_avg * L = K * N + (L - K) * R
    R = (T_avg * L - K * N) / (L - K)
    """
    if n_layers <= k_layer:
        return max(1, min(int(t_avg), n_enter))
    r = (float(t_avg) * n_layers - float(k_layer) * n_enter) / float(n_layers - k_layer)
    r = int(round(r))
    return max(1, min(r, n_enter))


_REGION_GRID = 2


def _rotary_cache_for_original_positions(self, x, seq_len=None):
    """Keep original token positions valid when decoder layers have shorter caches."""
    if seq_len > self.max_seq_len_cached:
        self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
    return (
        self.cos_cached[: self.max_seq_len_cached].to(dtype=x.dtype),
        self.sin_cached[: self.max_seq_len_cached].to(dtype=x.dtype),
    )


def mico_coverage_llm_select(
    last_attention: torch.Tensor,
    hidden_states: torch.Tensor,
    system_prompt_length: int,
    visual_token_length: int,
    keep: int,
    coverage_only: bool = False,
    mico: bool = False,
    sim_cov: bool = False,
    anti_sim_cov: bool = False,
    last_token: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MiCo greedy attention-weighted coverage selection; optional feature-norm weighting uses the current feature space."""
    vis_start = system_prompt_length
    vis_end = system_prompt_length + visual_token_length
    seq_len = hidden_states.shape[1]
    keep = int(min(max(keep, 1), visual_token_length))
    feats_raw = hidden_states[0, vis_start:vis_end].float()
    attn = last_attention.mean(dim=1)[0]
    if vis_end < seq_len:
        text_attn = attn[vis_end:, vis_start:vis_end].mean(dim=0)
    else:
        text_attn = attn[-1, vis_start:vis_end]
    text_attn = text_attn.float().clamp(min=0.0)
    if mico:
        text_attn = text_attn * feats_raw.norm(dim=-1).clamp(min=0.0)
    weight_sum = text_attn.sum()
    if weight_sum > 0:
        text_attn = text_attn / weight_sum
    else:
        text_attn = torch.full_like(text_attn, 1.0 / visual_token_length)
    feats = F.normalize(feats_raw, dim=-1)
    sim = (feats @ feats.t()).clamp(min=0.0)
    device = hidden_states.device
    selected: List[int] = []
    max_sim_to_S = torch.zeros(visual_token_length, device=device, dtype=sim.dtype)
    available = torch.ones(visual_token_length, dtype=torch.bool, device=device)
    for _ in range(keep):
        gain = (text_attn.unsqueeze(0) * (sim - max_sim_to_S.unsqueeze(0)).clamp(min=0.0)).sum(
            dim=1
        )
        gain = gain.masked_fill(~available, float("-inf"))
        idx = int(torch.argmax(gain).item())
        selected.append(idx)
        available[idx] = False
        max_sim_to_S = torch.maximum(max_sim_to_S, sim[idx])
    visual_token_index = torch.tensor(selected, device=device, dtype=torch.long)
    visual_token_index = torch.sort(visual_token_index).values
    return (visual_token_index, text_attn.detach())


def inner_weighted_raw_cls_select(
    features: torch.Tensor,
    raw_cls_attn: torch.Tensor,
    keep: int,
    coverage_only: bool = False,
    mico: bool = False,
) -> torch.Tensor:
    """MiCo greedy attention-weighted coverage selection; optional feature-norm weighting uses the current feature space."""
    n_tokens = features.shape[0]
    keep = int(max(1, min(int(keep), n_tokens)))
    feats = F.normalize(features.float(), dim=-1)
    sim = (feats @ feats.t()).clamp(min=0.0)
    if coverage_only:
        w = torch.ones(n_tokens, device=features.device, dtype=torch.float32)
    else:
        w = raw_cls_attn.float().clamp(min=0.0)
        if mico:
            w = w * features.float().norm(dim=-1).clamp(min=0.0)
    device = features.device
    available = torch.ones(n_tokens, dtype=torch.bool, device=device)
    max_cov = torch.zeros(n_tokens, device=device, dtype=sim.dtype)
    selected: List[int] = []
    seed = int(torch.argmax((w.unsqueeze(0) * sim).sum(dim=1)).item())
    selected.append(seed)
    available[seed] = False
    max_cov = torch.maximum(max_cov, sim[seed])
    for _ in range(keep - 1):
        gain = (w.unsqueeze(0) * (sim - max_cov.unsqueeze(0)).clamp(min=0.0)).sum(dim=1)
        gain = gain.masked_fill(~available, float("-inf"))
        idx = int(torch.argmax(gain).item())
        selected.append(idx)
        available[idx] = False
        max_cov = torch.maximum(max_cov, sim[idx])
    mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)
    mask[torch.tensor(selected, device=device, dtype=torch.long)] = True
    return mask


class MiCoCoverageLLMLlamaModel(LlamaModel):
    """LlamaModel with MiCo pruning at layer K (configured layer).

    Config keys:
      K: prune after this layer index (1-based one-based convention; default 2)
      R / T: fixed keep count when dynamic_R is False (mico_coverage_llm)
      T_avg / T: target layer-average visual count when dynamic_R is True
      dynamic_R: if True, recompute R from T_avg, N, K, L each forward
      merge_threshold: stored for encode_images (merge_mico_coverage_llm only)
    """

    def __init__(self, config: LlamaConfig, mico_coverage_llm_config: Dict, **kwargs):
        super().__init__(config)
        for layer in self.layers:
            rotary = layer.self_attn.rotary_emb
            rotary.forward = MethodType(_rotary_cache_for_original_positions, rotary)
        self.system_prompt_length = 35
        self.visual_token_num = 0
        if config.num_hidden_layers == 32:
            self.scale = "7b"
        elif config.num_hidden_layers == 40:
            self.scale = "13b"
        if config.image_aspect_ratio == "pad":
            self.visual_token_length = 576
            self.anyres = False
        elif config.image_aspect_ratio == "anyres":
            self.visual_token_length = 2880
            self.anyres = True
        else:
            self.visual_token_length = 576
            self.anyres = False
        self.K = int(mico_coverage_llm_config.get("K", 2))
        self.K2 = int(mico_coverage_llm_config.get("K2", 0))
        prune_layers = mico_coverage_llm_config.get("prune_layers")
        if prune_layers is not None:
            self.prune_layers = [int(x) for x in prune_layers]
        elif self.K2 > 0:
            self.prune_layers = [self.K, self.K2]
        else:
            self.prune_layers = [self.K]
        self.adaptive_r2 = False
        self.dynamic_R = bool(mico_coverage_llm_config.get("dynamic_R", False))
        self.prefix2_keep = False
        self.prefix2_first = False
        self.T_avg = int(
            mico_coverage_llm_config.get(
                "T_avg", mico_coverage_llm_config.get("T", mico_coverage_llm_config.get("R", 128))
            )
        )
        self.merge_threshold = float(mico_coverage_llm_config.get("merge_threshold", 0.9))
        self.coverage_only = False
        self.mico = bool(mico_coverage_llm_config.get("mico", False))
        self.sim_cov = False
        self.anti_sim_cov = False
        self.last_token = False
        if "R" in mico_coverage_llm_config:
            self.R = int(mico_coverage_llm_config["R"])
        else:
            self.R = int(mico_coverage_llm_config.get("T", self.T_avg))

    def update_runtime_visual_length(self, n_visual: int) -> int:
        """Set N entering the LLM (after merge) and refresh R if dynamic."""
        n_visual = int(max(1, n_visual))
        self.visual_token_length = n_visual
        if self.dynamic_R:
            self.R = compute_keep_from_avg(
                t_avg=self.T_avg, n_enter=n_visual, k_layer=self.K, n_layers=len(self.layers)
            )
        return self.R

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if self._use_flash_attention_2:
            attention_mask = (
                attention_mask if attention_mask is not None and 0 in attention_mask else None
            )
        elif self._use_sdpa and (not output_attentions):
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        if seq_length > 1:
            visual_token_length = self.visual_token_length
            visual_token_num = 0
            visual_token_list = []
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            elif seq_length > 1:
                visual_token_num += visual_token_length
                visual_token_list.append(visual_token_length)
                layer_1b = decoder_layer.self_attn.layer_idx + 1
                if layer_1b in self.prune_layers:
                    attn_mask = torch.ones(
                        (batch_size, hidden_states.shape[1]), device=hidden_states.device
                    )
                    attn_mask = _prepare_4d_causal_attention_mask(
                        attn_mask, (batch_size, hidden_states.shape[1]), hidden_states, 0
                    )
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attn_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=True,
                        use_cache=use_cache,
                    )
                    hidden_states = layer_outputs[0]
                    last_attention = layer_outputs[1]
                    device = hidden_states.device
                    cur_seq = hidden_states.shape[1]
                    orig_visual_len = int(visual_token_length)
                    n_layers = len(self.layers)
                    n_enter = int(self.visual_token_length)
                    if self.dynamic_R:
                        keep_r = compute_keep_from_avg(
                            t_avg=self.T_avg,
                            n_enter=orig_visual_len,
                            k_layer=layer_1b,
                            n_layers=n_layers,
                        )
                        self.R = keep_r
                        if getattr(self, "prefix2_first", False) or self.K == 12:
                            log_i = getattr(self, "_mico_coverage_12_log_i", 0)
                            enable_debug = os.environ.get("ENABLE_DEBUG", "0") == "1"
                            if log_i == 0 or (enable_debug and log_i < 8):
                                print(
                                    f"[mico_coverage_proj_llm_12] sample={log_i} L{layer_1b} N={orig_visual_len} T={self.T_avg} keep={keep_r} (S2=text_attn*coverage, R=(T*L-K*2T)/(L-K))",
                                    flush=True,
                                )
                            self._mico_coverage_12_log_i = log_i + 1
                    else:
                        keep_r = int(self.R)
                    visual_token_index, text_attn = mico_coverage_llm_select(
                        last_attention=last_attention,
                        hidden_states=hidden_states,
                        system_prompt_length=self.system_prompt_length,
                        visual_token_length=orig_visual_len,
                        keep=keep_r,
                        coverage_only=self.coverage_only,
                        mico=self.mico,
                        sim_cov=self.sim_cov,
                        anti_sim_cov=self.anti_sim_cov,
                        last_token=self.last_token,
                    )
                    visual_token_length = int(visual_token_index.numel())
                    self.metric_cache = {
                        "text_attn": text_attn.detach().cpu(),
                        "selected_indices": visual_token_index.detach().cpu(),
                        "layer": layer_1b,
                        "N_enter": n_enter,
                        "keep": visual_token_length,
                        "R2": getattr(self, "R2", None),
                        "R6": getattr(self, "R6", None),
                        "entropy": getattr(self, "_r2_entropy", None),
                        "T_avg_target": self.T_avg,
                    }
                    full_token_index = torch.cat(
                        (
                            torch.arange(self.system_prompt_length, device=device),
                            visual_token_index + self.system_prompt_length,
                            torch.arange(
                                self.system_prompt_length + orig_visual_len, cur_seq, device=device
                            ),
                        )
                    )
                    hidden_states = hidden_states[:, full_token_index]
                    position_ids = full_token_index.unsqueeze(0)
                    layer_outputs = (hidden_states, layer_outputs[2])
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        if seq_length > 1:
            self.visual_token_num = visual_token_num / len(self.layers)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                (
                    v
                    for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                    if v is not None
                )
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
