# coding=utf-8
"""
TOPS Stage 2: Progressive visual token pruning with RDC (Relevance-Diversity-Coverage)
greedy selection for InternVL3, adapted from TOPS (modeling_llama_tops.py).

Key design:
  - Relevance (R): attention importance computed via QKV probe on PRE-layer hidden_states
  - Diversity (D): cosine dissimilarity computed on POST-layer hidden_states
  - Coverage (C): representativeness computed on POST-layer hidden_states
  - Greedy selection jointly optimizing R + α·D + λ·C
"""

import os
import math
from functools import partial
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import logging

logger = logging.get_logger(__name__)

# ── TOPS pruning schedule: InternVL3-8B (28 layers) ──
# Format: {scale: {target_per_patch: [(layer_idx_1indexed, keep_per_patch), ...]}}
# Actual retention = keep_per_patch × dynamic_res (number of image patches)
TOPS_SCHEDULE: Dict[str, Dict[int, List[Tuple[int, int]]]] = {
    "internvl3_8b": {
        512: [(12, 281), (16, 77)],
        256:  [(12, 139), (16, 39)], #剪枝率=（1280-256）/1280=80%
        128:  [(12, 71), (16, 19)], #剪枝率=（1280-128）/1280=90%
    },
}


class TOPSQwen2Model(Qwen2Model):
    """
    TOPS Stage 2: Progressive RDC pruning inside Qwen2 decoder layers.
    Mirrors the logic of TOPS_LlamaModel but adapted for Qwen2 (Flash Attention, GQA, RoPE).
    """

    def __init__(self, config: Qwen2Config, tops_config: Dict):
        super().__init__(config)

        self.system_prompt_length = 41  # InternVL3 system prompt tokens
        self.num_image_token = 256     # tokens per patch (448/14=32, 32²=1024, ×0.5²=256)

        # Scale detection
        if config.num_hidden_layers == 28:
            self.scale = "internvl3_8b"
        else:
            self.scale = "internvl3_8b"
            logger.warning(f"Unknown scale with {config.num_hidden_layers} layers, defaulting to internvl3_8b")

        self.retained_num = tops_config["T"]
        self.mico = bool(tops_config.get("mico", False))
        self.dynamic_r = bool(tops_config.get("dynamic_r", False))
        self.visual_token_length = 0  # dynamically set per image by modeling_internvl_chat.py
        # COVER_Prune_2 / mico_coverage_vit_proj_llm_12: all-text rater + inner-weighted
        # coverage. Env COVER_S2_INSIDE still works for the existing cp2 path.
        self.cover_s2_inside = bool(tops_config.get("cover_s2_inside", False)) or (
            os.environ.get("COVER_S2_INSIDE", "0") == "1"
        )

        # Load pruning schedule (1-indexed layer_idx)
        schedule = TOPS_SCHEDULE.get(self.scale, {}).get(self.retained_num)
        if tops_config.get("disable_stage2", False):
            # tops_2 (single-stage): keep Stage-1 pruning, disable in-LLM Stage-2 entirely.
            self.pruning_layers: Dict[int, int] = {}
            logger.warning("tops_2 single-stage: in-LLM Stage-2 pruning DISABLED (pruning_layers = {})")
        elif tops_config.get("pruning_layers") is not None:
            # mico_coverage_vit_proj_llm_12: single TOPS-first-layer cut; T=64/32 are
            # not in TOPS_SCHEDULE and must not disable S2.
            self.pruning_layers = {int(k): int(v) for k, v in tops_config["pruning_layers"].items()}
        elif schedule is None:
            self.pruning_layers: Dict[int, int] = {}
            logger.warning(f"No TOPS schedule for target={self.retained_num}, progressive pruning disabled")
        else:
            self.pruning_layers = {layer_idx: keep for layer_idx, keep in schedule}
        # H-ViT all-layer sweeps must use the actual post-S1 token count: tiled
        # InternVL inputs can round 2T down on a per-patch allocation.
        self._dynamic_k_layers = tuple(sorted(self.pruning_layers))

        print(f"[TOPS Init]")
        print(f"  Scale: {self.scale} ({config.num_hidden_layers} layers)")
        print(f"  Target (per patch): {self.retained_num}")
        print(f"  Stage1 visual_token_length: {self.visual_token_length}")
        print(f"  Pruning schedule (1-indexed): {self.pruning_layers}")

        self.reset_state()

    def _refresh_dynamic_r(self, n_enter: int) -> None:
        if not self.dynamic_r:
            return
        n_enter = int(n_enter)
        n_layers = int(self.config.num_hidden_layers)
        refreshed: Dict[int, int] = {}
        for k_layer in self._dynamic_k_layers:
            if n_enter <= self.retained_num:
                keep = n_enter
            else:
                raw_keep = (self.retained_num * n_layers - k_layer * n_enter) / (n_layers - k_layer)
                keep = max(1, min(int(round(raw_keep)), n_enter))
            refreshed[int(k_layer)] = int(keep)
        self.pruning_layers = refreshed
        print(
            f"[TOPS H-ViT] refresh dynamic R: N_enter={n_enter} T={self.retained_num} "
            f"layers={self.pruning_layers}", flush=True,
        )

    def reset_state(self):
        self.prefill_done = False
        self.current_visual_length: Optional[int] = None
        self.stage1_actual_tokens: Optional[int] = None
        self.stage2_events: List[Tuple[int, int, int]] = []
        self.stage2_skip_events: List[Tuple[int, int, int]] = []
        self.stage2_reported_layers: Set[int] = set()
        self._stage_summary_emitted = False
        self.visual_token_num = 0.0

    # ─────────── helpers ───────────

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _select_text_raters(
        self,
        hidden_states: torch.Tensor,
        visual_start: int,
        visual_end: int,
    ) -> torch.Tensor:
        """Select informative text tokens as raters via normalized text-visual similarity."""
        text_hidden = hidden_states[:, visual_end:, :]
        if text_hidden.size(1) == 0:
            return hidden_states.new_tensor([hidden_states.shape[1] - 1], dtype=torch.long)

        # COVER_Prune_2 / mico_coverage_vit_proj_llm_12: all text tokens as raters
        # (no >mean subset), matching Qwen CoverPrune2 all_text_attn.
        if getattr(self, "cover_s2_inside", False) or os.environ.get("COVER_S2_INSIDE", "0") == "1":
            return torch.arange(text_hidden.size(1), device=hidden_states.device, dtype=torch.long)

        visual_hidden = hidden_states[:, visual_start:visual_end, :]
        text_norm = F.normalize(text_hidden, dim=-1)
        visual_norm = F.normalize(visual_hidden, dim=-1)

        # [batch, text_len, visual_len] → average over visual → text importance
        text_visual_sim = torch.matmul(text_norm, visual_norm.transpose(1, 2))
        text_importance = text_visual_sim.mean(dim=-1)
        if text_importance.dim() == 2:
            text_importance = text_importance.mean(dim=0)

        mean_importance = text_importance.mean()
        selected = torch.nonzero(text_importance > mean_importance, as_tuple=False).squeeze(-1)
        if selected.numel() == 0:
            keep = max(1, text_importance.numel() // 2)
            selected = text_importance.topk(keep).indices
        return selected

    def _compute_visual_token_importance(
        self,
        hidden_states: torch.Tensor,
        decoder_layer: nn.Module,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        visual_start: int,
        visual_end: int,
    ) -> torch.Tensor:
        """
        Compute attention-based relevance via QKV probe on PRE-layer hidden_states.
        Uses text rater tokens as queries, visual tokens as keys.
        Returns: (V,) importance scores for each visual token.
        """
        self_attn = decoder_layer.self_attn

        # Pre-attention layernorm
        attn_input = hidden_states
        input_layernorm = getattr(decoder_layer, "input_layernorm", None)
        if input_layernorm is not None:
            attn_input = input_layernorm(attn_input)

        bsz, seq_len, _ = attn_input.shape
        num_heads = self_attn.num_heads if hasattr(self_attn, 'num_heads') else self_attn.config.num_attention_heads
        num_kv_heads = self_attn.num_key_value_heads if hasattr(self_attn, 'num_key_value_heads') else self_attn.config.num_key_value_heads
        head_dim = self_attn.head_dim if hasattr(self_attn, 'head_dim') else (self_attn.config.hidden_size // num_heads)

        # Project Q and K
        query_states = self_attn.q_proj(attn_input)
        key_states = self_attn.k_proj(attn_input)

        query_states = query_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        # Handle GQA: expand KV heads to match Q heads
        if num_kv_heads != num_heads:
            repeat_factor = num_heads // num_kv_heads
            key_states = key_states.repeat_interleave(repeat_factor, dim=1)

        # Apply RoPE
        cos, sin = position_embeddings
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif cos.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        query_states = (query_states * cos) + (self._rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)

        # Dynamic text rater selection
        text_rater_index = self._select_text_raters(hidden_states, visual_start, visual_end)

        # Map text rater indices to absolute positions
        text_positions = (text_rater_index + visual_end).clamp(max=seq_len - 1)

        # Q from text raters, K from visual tokens
        query_raters = torch.index_select(query_states, 2, text_positions)
        key_visual = key_states[:, :, visual_start:visual_end, :]

        # Attention scores
        attn_scores = torch.matmul(query_raters, key_visual.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Average over raters, heads, batch → (V,)
        cross_attn = attn_weights.mean(dim=2).mean(dim=1).mean(dim=0)

        print(f"[TOPS S2] text_rater_indices numel: {text_rater_index.numel()}, "
              f"importance: min={cross_attn.min():.4f}, max={cross_attn.max():.4f}")
        return cross_attn

    def _rdc_greedy_select(
        self,
        attn: torch.Tensor,
        post_hidden: torch.Tensor,
        retained_num: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        RDC greedy selection on POST-layer hidden states.
        attn: (V,) normalized relevance scores
        post_hidden: (B, V, D) POST-layer visual token features
        Returns: (retained_num,) sorted selected indices (batch=1 path)
        """
        a2 = float(os.environ.get("A2", "1.0"))
        alpha_2 = float(os.environ.get("ALPHA_2", "1.0"))
        lambda_2 = float(os.environ.get("LAMBDA_2", "1.0"))
        # COVER_Prune_2 / mico_coverage_vit_proj_llm_12: inner-weighted coverage
        cover_inside = getattr(self, "cover_s2_inside", False) or (
            os.environ.get("COVER_S2_INSIDE", "0") == "1"
        )

        B, V, D = post_hidden.shape

        coverage_method = os.environ.get("COVERAGE", "MEAN")

        attn = attn.detach()
        if attn.dim() == 1:
            attn = attn.unsqueeze(0)  # (1, V)
        if self.mico:
            if attn.shape[0] == 1 and B > 1:
                attn = attn.expand(B, -1)
            if attn.shape != (B, V):
                raise ValueError(f"S2 attention shape {tuple(attn.shape)} != post-hidden {(B, V)}")
            # S2 H is post-layer by construction; applying it before the layer
            # would no longer match the physical pruning locus K.
            attn = attn.float().clamp(min=0.0) * post_hidden.detach().float().norm(dim=-1)
            attn = attn / attn.sum(dim=1, keepdim=True).clamp(min=1e-8)
        else:
            attn = (attn - attn.min(dim=1, keepdim=True).values) / (
                attn.max(dim=1, keepdim=True).values -
                attn.min(dim=1, keepdim=True).values + 1e-8
            )

        # Normalized features for cosine similarity (POST-layer)
        vis_norm = F.normalize(post_hidden, p=2, dim=-1)  # (B, V, D)

        all_selected = []
        for b in range(B):
            a = attn[b]        # (V,)
            feat = vis_norm[b]  # (V, D)

            available = torch.ones(V, dtype=torch.bool, device=device)
            selected = []

            # Full similarity matrix for coverage
            full_sim = feat @ feat.t()  # (V, V)

            # Seed
            if cover_inside:
                # MiCo uses positive cosine for the seed and all later coverage updates;
                # leave signed similarity unchanged for the separate TOPS branch.
                full_sim = torch.clamp(full_sim, min=0.0)
                i0 = torch.argmax((a.unsqueeze(0) * full_sim).sum(dim=1)).item()
            else:
                # 默认: 最高 attention token
                i0 = torch.argmax(a).item()
            selected.append(i0)
            available[i0] = False

            max_sim_to_S = full_sim[i0].clone()  # (V,)
            max_sim_to_S[i0] = 1.0

            for _ in range(1, retained_num):
                if not available.any():
                    break

                if cover_inside:
                    # COVER_Prune_2 内加权: gain_i = Σ_v a_v·max(0, sim(v,i)-g_v), 直接作为打分
                    scores = (a.unsqueeze(0) * torch.clamp(full_sim - max_sim_to_S.unsqueeze(0), min=0.0)).sum(dim=1)
                    scores[~available] = -float("inf")
                    idx = torch.argmax(scores).item()
                else:
                    diversity = 1.0 - max_sim_to_S

                    # Coverage
                    if coverage_method == "SCOPE":
                        # SCOPE: marginal coverage gain
                        coverage = torch.clamp(full_sim - max_sim_to_S.unsqueeze(0), min=0.0).sum(dim=1)
                        coverage[~available] = 0.0
                    else:
                        # MEAN: avg similarity to remaining available tokens
                        sim_to_avail = full_sim[:, available].sum(dim=1)
                        n_avail = available.sum().item()
                        coverage = (sim_to_avail - 1.0) / (n_avail - 1) if n_avail > 1 else torch.zeros(V, device=device)

                    a_mean = a.mean().clamp(min=1e-8)
                    div_mean = diversity.mean().clamp(min=1e-8)
                    cov_mean = coverage.mean().clamp(min=1e-8)

                    scores = a2 * (a / a_mean) + alpha_2 * (diversity / div_mean) + lambda_2 * (coverage / cov_mean)
                    scores[~available] = -float("inf")
                    idx = torch.argmax(scores).item()

                selected.append(idx)
                available[idx] = False

                sim_vec = full_sim[idx]
                max_sim_to_S = torch.maximum(max_sim_to_S, sim_vec)
                max_sim_to_S[idx] = 1.0

            selected_t = torch.tensor(selected, device=device, dtype=torch.long)
            selected_t = torch.sort(selected_t).values  # preserve spatial order
            all_selected.append(selected_t)

        # Debug metrics
        a_debug = attn[0]
        feat_debug = vis_norm[0]
        sel = all_selected[0]
        avg_attn = a_debug[sel].mean().item()
        if sel.numel() > 1:
            sel_feat = feat_debug[sel]
            sel_sim = sel_feat @ sel_feat.t()
            diag_mask = ~torch.eye(sel_sim.size(0), dtype=torch.bool, device=sel_sim.device)
            avg_sim = sel_sim[diag_mask].mean().item()
        else:
            avg_sim = 0.0
        print(f"[TOPS S2] RDC: retained={sel.numel()}, mode={'COVER-INSIDE(product)' if cover_inside else 'RDC-add'}, a2={a2}, alpha_2={alpha_2}, lambda_2={lambda_2}, "
              f"avg_attn={avg_attn:.4f}, avg_sim={avg_sim:.4f}")

        return all_selected[0]  # batch=1 common path

    # ─────────── forward ───────────

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        """Forward with TOPS RDC progressive pruning."""

        # Reset state at the start of each new prefill
        if past_key_values is None or (
            isinstance(past_key_values, DynamicCache) and past_key_values.get_seq_length() == 0
        ):
            self.reset_state()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_length, device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # ── Prefill phase: init visual token tracking ──
        if seq_length > 1 and not self.prefill_done:
            self.current_visual_length = self.visual_token_length
            self.stage1_actual_tokens = int(self.current_visual_length)
            self._refresh_dynamic_r(self.current_visual_length)
            self.prefill_done = True
            self.visual_token_num = 0.0
            print(f"[TOPS] Prefill: seq_length={seq_length}, visual_token_length={self.current_visual_length}")

        tops_enabled = (
            self.prefill_done
            and self.current_visual_length is not None
            and self.current_visual_length > 0
            and len(self.pruning_layers) > 0
        )
        visual_token_accumulator = 0.0

        # Flag to disable flash attention after pruning (KV cache shape mismatch)
        pruning_performed = False

        for layer_idx, decoder_layer in enumerate(self.layers[:self.config.num_hidden_layers], start=1):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if seq_length > 1 and self.current_visual_length is not None:
                visual_token_accumulator += float(self.current_visual_length)

            should_prune = (
                tops_enabled
                and seq_length > 1
                and layer_idx in self.pruning_layers
                and self.current_visual_length is not None
                and self.current_visual_length > self.pruning_layers[layer_idx]
            )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__),
                    hidden_states, causal_mask, position_ids, past_key_values,
                    output_attentions, use_cache, cache_position, position_embeddings,
                )

            elif should_prune:
                target = self.pruning_layers[layer_idx]
                visual_start = self.system_prompt_length
                visual_end = visual_start + self.current_visual_length
                device = hidden_states.device

                # Validate range
                if visual_end > hidden_states.shape[1]:
                    visual_end = hidden_states.shape[1]
                    self.current_visual_length = visual_end - visual_start

                print(f"\n[TOPS S2] Layer {layer_idx}: {self.current_visual_length} → {target} tokens")

                # ── Step 1: Attention importance on PRE-layer hidden_states ──
                attn = self._compute_visual_token_importance(
                    hidden_states, decoder_layer, position_embeddings,
                    visual_start, visual_end
                )

                # ── Step 2: Run layer normally → POST-layer hidden_states ──
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                hidden_states = layer_outputs[0]

                # ── Step 3: RDC greedy selection on POST-layer visual tokens ──
                post_visual = hidden_states[:, visual_start:visual_end]  # (B, V, D)
                retained_num = min(int(target), post_visual.shape[1])
                selected_t = self._rdc_greedy_select(attn, post_visual, retained_num, device)

                # ── Step 4: Rebuild hidden_states ──
                orig_seq_len = hidden_states.shape[1]

                select_visual_tokens = post_visual.gather(
                    1, selected_t.unsqueeze(0).unsqueeze(-1).expand(-1, -1, post_visual.shape[-1])
                )  # (B, R, D)

                hidden_states = torch.cat((
                    hidden_states[:, :visual_start],
                    select_visual_tokens,
                    hidden_states[:, visual_end:]
                ), dim=1)

                # Build retain_token_indices for correct position tracking
                retain_token_indices = torch.cat((
                    torch.arange(visual_start, device=device),
                    selected_t + visual_start,
                    torch.arange(visual_end, orig_seq_len, device=device),
                ))

                # Update position_ids using retained indices (preserves original positions)
                position_ids = position_ids[:, retain_token_indices]
                if cache_position is not None:
                    cache_position = cache_position[retain_token_indices]

                # Index into existing position_embeddings (preserves correct RoPE positions)
                cos, sin = position_embeddings
                if cos.dim() == 2:  # [S, head_dim]
                    position_embeddings = (cos[retain_token_indices], sin[retain_token_indices])
                else:               # [B, S, head_dim]
                    position_embeddings = (cos[:, retain_token_indices], sin[:, retain_token_indices])

                causal_mask = self._update_causal_mask(
                    None, hidden_states, cache_position, past_key_values, output_attentions
                )

                prev_length = self.current_visual_length
                self.current_visual_length = selected_t.numel()
                tops_enabled = self.current_visual_length > 0
                pruning_performed = True
                self.stage2_events.append((layer_idx, prev_length, self.current_visual_length))
                self.stage2_reported_layers.add(layer_idx)

                print(f"[TOPS S2] Layer {layer_idx} done: "
                      f"{prev_length} → {self.current_visual_length} tokens, "
                      f"total_seq={hidden_states.shape[1]}")

                layer_outputs = (hidden_states, layer_outputs[-1])

            else:
                # Skip-prune logging
                if (
                    tops_enabled
                    and seq_length > 1
                    and layer_idx in self.pruning_layers
                    and layer_idx not in self.stage2_reported_layers
                ):
                    target = self.pruning_layers[layer_idx]
                    print(f"[TOPS S2] Layer {layer_idx}: skip (current {self.current_visual_length} ≤ target {target})")
                    self.stage2_skip_events.append((layer_idx, self.current_visual_length, target))
                    self.stage2_reported_layers.add(layer_idx)

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs if not pruning_performed else {},
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if seq_length > 1 and self.current_visual_length is not None:
            average_tokens = visual_token_accumulator / len(self.layers) if self.layers else 0.0
            self.visual_token_num = float(average_tokens)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # ── Stage summary ──
        if seq_length > 1 and self.prefill_done and not self._stage_summary_emitted:
            print("\n[TOPS] Stage summary:")
            if self.stage2_events:
                for layer_idx, before, after in self.stage2_events:
                    print(f"  Stage 2 Layer {layer_idx}: {before} → {after}")
                print(f"  Final visual tokens: {self.current_visual_length}")
            else:
                print("  Stage 2: no pruning applied")
            self._stage_summary_emitted = True

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class TopsQwen2ForCausalLM(Qwen2ForCausalLM):
    """Wrapper: Qwen2ForCausalLM with TOPS RDC decoder."""
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, tops_config: Dict):
        super().__init__(config)
        self.model = TOPSQwen2Model(config, tops_config)

    def set_system_prompt_length(self, length: int):
        self.model.system_prompt_length = length
