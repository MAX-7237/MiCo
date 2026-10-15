# coding=utf-8
"""
TOPS Stage 2: Progressive visual token pruning with RDC (Relevance-Diversity-Coverage)
greedy selection for LLaVA-Video with Qwen2.

Key design:
  - Relevance (R): attention importance computed via QKV probe on PRE-layer hidden_states
  - Diversity (D): cosine dissimilarity computed on POST-layer hidden_states
  - Coverage (C): representativeness computed on POST-layer hidden_states (SCOPE or MEAN)
  - Greedy selection jointly optimizing a2·R + alpha_2·D + lambda_2·C

Adapted from /path/to/local-assets
for LLaVA-Video (multi-frame video, older transformers API without cache_position).
"""

import os
import math
from typing import Dict, List, Optional, Tuple, Union, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers import Qwen2Model
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Config,
    Cache,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
# from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding


# ── TOPS pruning schedule: 28-layer Qwen2 (LLaVA-Video 7B) ──
# Values are per-frame token counts.
# Total target at each pruning layer = per_frame × actual_video_frames.
# Average formula: (stage1_per_frame×12 + layer12_target×9 + layer21_target×7) / 28
TOPS_SCHEDULE: Dict[str, Dict[int, List[Tuple[int, int]]]] = {
    "7b": {
        64: [(12, 23), (21, 7)],   # per-frame: 128→23→7,  layer-avg≈64
        32: [(12, 11), (21, 4)],   # per-frame: 64→11→4,   layer-avg≈32
        16: [(12, 5),  (21, 3)],   # per-frame: 32→5→3,    layer-avg≈16
    }
}


class TOPSQwen2Model(Qwen2Model):
    """
    TOPS Stage 2 for LLaVA-Video with Qwen2.

    Replaces topk selection with full RDC (Relevance-Diversity-Coverage) greedy selection.
    Processes multi-frame video: targets are per-frame × actual_video_frames.
    """

    def __init__(self, config: Qwen2Config, tops_config: Dict):
        super().__init__(config)
        
        # LLaVA-Video defaults
        self.system_prompt_length = 35
        self.visual_token_length = 169   # 13×13 per frame; updated externally
        self.video_frame_num = 0
        self.visual_token_num = 0.0

        # TOPS config
        try:
            self.target_visual_tokens = int(tops_config["T"])
        except (TypeError, ValueError, KeyError):
            raise ValueError(f"tops_config must contain 'T' (int), got {tops_config}")

        self.stage1_keep_tokens = max(1, self.target_visual_tokens) * 2
        self.stage0_visual_tokens = self.visual_token_length

        # Scale detection
        self.scale = "7b" if config.num_hidden_layers == 28 else "unknown"

        schedule = TOPS_SCHEDULE.get(self.scale, {}).get(self.target_visual_tokens)
        if schedule is None:
            self.pruning_layers: Dict[int, int] = {}
            print(f"[TOPS Warning] No schedule for T={self.target_visual_tokens}, "
                  f"scale={self.scale}. Stage 2 pruning disabled.")
        else:
            self.pruning_layers = {layer_idx: target for layer_idx, target in schedule}

        self.reset_state()

        print(f"[TOPS Stage2 Init] T={self.target_visual_tokens}, "
              f"stage1_per_frame={self.stage1_keep_tokens}, "
              f"pruning layers: {list(self.pruning_layers.keys())}")

    def reset_state(self):
        """Reset per-generation state."""
        self.prefill_done = False
        self.current_visual_length: Optional[int] = None
        self.original_visual_length: Optional[int] = None
        self.actual_video_frames: Optional[int] = None
        self.visual_token_sum = 0.0
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
        """Select informative text tokens as raters via text-visual similarity."""
        seq_len = hidden_states.shape[1]
        text_hidden = hidden_states[:, visual_end:, :]
        text_len = text_hidden.size(1)
        print(f"[DEBUG _select_text_raters] hs.shape={hidden_states.shape}, "
              f"visual_start={visual_start}, visual_end={visual_end}, text_len={text_len}")
        if text_hidden.size(1) == 0:
            # BUG WARNING: this returns an ABSOLUTE index, but caller adds visual_end to it!
            fallback = hidden_states.new_tensor([seq_len - 1 - visual_end], dtype=torch.long)
            print(f"[DEBUG _select_text_raters] text empty, returning relative fallback={fallback.tolist()}")
            return fallback

        visual_hidden = hidden_states[:, visual_start:visual_end, :]
        print(f"[DEBUG _select_text_raters] visual_hidden.shape={visual_hidden.shape}, text_hidden.shape={text_hidden.shape}")
        text_norm = F.normalize(text_hidden, dim=-1)
        visual_norm = F.normalize(visual_hidden, dim=-1)
        print(f"text_norm.shape{text_norm.shape}")
        print(f"visual_norm.shape{visual_norm.shape}")

        # [batch, text_len, visual_len] → avg over visual → text importance
        text_visual_sim = torch.matmul(text_norm, visual_norm.transpose(1, 2))
        text_importance = text_visual_sim.mean(dim=-1)
        if text_importance.dim() == 2:
            text_importance = text_importance.mean(dim=0)

        mean_importance = text_importance.mean()
        print(f"text_importance.shape{text_importance.shape}")
        print(f"mean_importance.shape{mean_importance.shape}")
        mask = text_importance > mean_importance
        print("mask.shape", mask.shape)
        print("mask.dtype", mask.dtype)
        print("mask.sum()", mask.sum().item())

        selected = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        print("selected.shape", selected.shape)
        if selected.numel() == 0:
            keep = max(1, text_importance.numel() // 2)
            selected = text_importance.topk(keep).indices
        # selected are RELATIVE indices into text_hidden (range [0, text_len-1])
        sel_min = selected.min().item() if selected.numel() > 0 else -1
        sel_max = selected.max().item() if selected.numel() > 0 else -1
        print(f"[DEBUG _select_text_raters] selected.numel={selected.numel()}, "
              f"range=[{sel_min}, {sel_max}], expected<{text_len}")
        if selected.numel() > 0 and sel_max >= text_len:
            print(f"[DEBUG _select_text_raters] !!! RELATIVE INDEX OUT OF RANGE: "
                  f"max={sel_max} >= text_len={text_len}")
        return selected

    def _compute_visual_token_importance(
        self,
        hidden_states: torch.Tensor,
        decoder_layer: nn.Module,
        visual_start: int,
        visual_end: int,
    ) -> torch.Tensor:
        """
        Compute attention-based relevance (R) via QKV probe on PRE-layer hidden_states.
        Text rater tokens as queries, visual tokens as keys.
        Returns: (V,) importance scores.
        """
        self_attn = decoder_layer.self_attn

        attn_input = hidden_states
        input_layernorm = getattr(decoder_layer, "input_layernorm", None)
        if input_layernorm is not None:
            attn_input = input_layernorm(attn_input)

        bsz, seq_len, _ = attn_input.shape
        num_heads = self_attn.num_heads
        num_kv_heads = self_attn.num_key_value_heads
        head_dim = self_attn.head_dim

        query_states = self_attn.q_proj(attn_input)
        key_states = self_attn.k_proj(attn_input)
        print("query_states.shape:",query_states.shape)
        print("key_states.shape:",key_states.shape)
        query_states = query_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        # GQA expansion
        if num_kv_heads != num_heads:
            key_states = key_states.repeat_interleave(num_heads // num_kv_heads, dim=1)

        # Note: RoPE skipped for simplicity; attention scores remain meaningful
        text_rater_indices = self._select_text_raters(hidden_states, visual_start, visual_end)
        text_positions = (text_rater_indices + visual_end).clamp(max=seq_len - 1)

        # DEBUG: validate before index_select
        tp_min = text_positions.min().item() if text_positions.numel() > 0 else -1
        tp_max = text_positions.max().item() if text_positions.numel() > 0 else -1
        print(f"[DEBUG index_select] seq_len={seq_len}, visual_end={visual_end}, "
              f"text_positions range=[{tp_min}, {tp_max}], query_states.shape={query_states.shape}")
        if text_positions.numel() > 0 and tp_max >= seq_len:
            print(f"[DEBUG index_select] !!! OUT OF BOUNDS: tp_max={tp_max} >= seq_len={seq_len}")
        torch.cuda.synchronize()

        query_raters = torch.index_select(query_states, 2, text_positions)
        torch.cuda.synchronize()  # force CUDA error to surface here if OOB
        key_visual = key_states[:, :, visual_start:visual_end, :]

        attn_scores = torch.matmul(query_raters, key_visual.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Average over raters, heads, batch → (V,)
        visual_importance = attn_weights.mean(dim=2).mean(dim=1).mean(dim=0)

        print(f"[TOPS S2] text_raters={text_rater_indices.numel()}, "
              f"importance: min={visual_importance.min():.4f}, max={visual_importance.max():.4f}")
        return visual_importance

    def _rdc_greedy_select(
        self,
        attn: torch.Tensor,
        post_hidden: torch.Tensor,
        retained_num: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        RDC greedy selection on POST-layer hidden states.

        attn:       (V,)    normalized relevance scores from PRE-layer QKV probe
        post_hidden:(B,V,D) POST-layer visual token features
        Returns:    (retained_num,) sorted selected indices (assumes batch=1)
        """
        a2       = float(os.environ.get("A2",       "1.0"))
        alpha_2  = float(os.environ.get("ALPHA_2",  "1.0"))
        lambda_2 = float(os.environ.get("LAMBDA_2", "1.0"))
        coverage_method = os.environ.get("COVERAGE", "MEAN")

        B, V, D = post_hidden.shape

        # Normalize relevance to [0, 1] per batch
        attn = attn.detach()
        if attn.dim() == 1:
            attn = attn.unsqueeze(0)  # (1, V)
        attn = (attn - attn.min(dim=1, keepdim=True).values) / (
            attn.max(dim=1, keepdim=True).values - attn.min(dim=1, keepdim=True).values + 1e-8
        )

        # Normalized features for cosine similarity (POST-layer LLM space)
        vis_norm = F.normalize(post_hidden, p=2, dim=-1)  # (B, V, D)

        all_selected = []
        for b in range(B):
            a    = attn[b]       # (V,)
            feat = vis_norm[b]   # (V, D)

            available = torch.ones(V, dtype=torch.bool, device=device)
            selected = []

            # Seed: highest attention token
            i0 = torch.argmax(a).item()
            selected.append(i0)
            available[i0] = False
            max_sim_to_S = feat @ feat[i0]  # (V,)
            max_sim_to_S[i0] = 1.0

            full_sim = feat @ feat.t()  # (V, V)

            for _ in range(1, retained_num):
                if not available.any():
                    break

                diversity = 1.0 - max_sim_to_S

                if coverage_method == "SCOPE":
                    # SCOPE: marginal coverage gain
                    coverage = torch.clamp(full_sim - max_sim_to_S.unsqueeze(0), min=0.0).sum(dim=1)
                    coverage[~available] = 0.0
                else:
                    # MEAN: avg similarity to remaining unselected tokens
                    sim_to_avail = full_sim[:, available].sum(dim=1)
                    n_avail = available.sum().item()
                    coverage = (
                        (sim_to_avail - 1.0) / (n_avail - 1)
                        if n_avail > 1
                        else torch.zeros(V, device=device)
                    )

                a_mean   = a.mean().clamp(min=1e-8)
                div_mean = diversity.mean().clamp(min=1e-8)
                cov_mean = coverage.mean().clamp(min=1e-8)

                scores = (a2 * (a / a_mean)
                          + alpha_2 * (diversity / div_mean)
                          + lambda_2 * (coverage / cov_mean))
                scores[~available] = -float("inf")
                idx = torch.argmax(scores).item()

                selected.append(idx)
                available[idx] = False
                max_sim_to_S = torch.maximum(max_sim_to_S, full_sim[idx])
                max_sim_to_S[idx] = 1.0

            selected_t = torch.sort(
                torch.tensor(selected, device=device, dtype=torch.long)
            ).values
            all_selected.append(selected_t)

        # Debug metrics (batch=0)
        sel = all_selected[0]
        avg_attn = attn[0][sel].mean().item()
        if sel.numel() > 1:
            sel_feat = vis_norm[0][sel]
            sel_sim  = sel_feat @ sel_feat.t()
            diag_mask = ~torch.eye(sel_sim.size(0), dtype=torch.bool, device=sel_sim.device)
            avg_sim = sel_sim[diag_mask].mean().item()
        else:
            avg_sim = 0.0
        print(f"[TOPS S2] RDC: retained={sel.numel()}, "
              f"a2={a2}, alpha_2={alpha_2}, lambda_2={lambda_2}, coverage={coverage_method}, "
              f"avg_attn={avg_attn:.4f}, avg_sim={avg_sim:.4f}")

        return all_selected[0]  # batch=1 common path

    # ─────────── forward ───────────

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # Reset state at the start of each new generation
        if past_key_values is None or (
            isinstance(past_key_values, DynamicCache) and past_key_values.get_seq_length() == 0
        ):
            self.reset_state()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache  = use_cache  if use_cache  is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("Must specify input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            print("[TOPS] Warning: use_cache=True incompatible with gradient checkpointing. Setting False.")
            use_cache = False

        # Initialize KV cache
        past_seen_tokens = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_usable_length(seq_length) if past_key_values else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        # if cache_position is None:
        #     past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        #     cache_position = torch.arange(
        #         past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        #     )

        # Prepare position_ids
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1],
                dtype=torch.long, device=device,
            ).unsqueeze(0).view(-1, inputs_embeds.shape[1])
        else:
            position_ids = position_ids.view(-1, inputs_embeds.shape[1]).long()

        # Prepare attention_mask
        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                inputs_embeds,
                past_seen_tokens,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                inputs_embeds,
                past_seen_tokens,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds
        # position_embeddings = self.rotary_emb(hidden_states, position_ids)
        seq_length = inputs_embeds.shape[1]

        # ── Prefill initialization ──
        if seq_length > 1 and not self.prefill_done:
            detected_visual_len = self.stage1_keep_tokens * self.video_frame_num
            
            self.actual_video_frames = self.video_frame_num
            self.current_visual_length = detected_visual_len
            self.original_visual_length = detected_visual_len
            self.stage1_actual_tokens = detected_visual_len
            self.prefill_done = True
            self.visual_token_sum = 0.0

            print(f"\n[TOPS Stage 2 Prefill] Detected {detected_visual_len} visual tokens, "
                  f"S2 Frames={self.actual_video_frames}, "
                  f"stage1_per_frame={self.stage1_keep_tokens}")

        # Layer-wise processing
        all_hidden_states = () if output_hidden_states else None
        all_self_attns    = () if output_attentions    else None
        next_decoder_cache = None

        tops_enabled = (
            seq_length > 1
            and self.prefill_done
            and self.current_visual_length is not None
            and self.current_visual_length > 0
            and len(self.pruning_layers) > 0
        )
        visual_token_accumulator = 0.0

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_idx = decoder_layer.self_attn.layer_idx + 1

            # Determine if pruning should happen at this layer
            if (tops_enabled
                    and layer_idx in self.pruning_layers
                    and self.actual_video_frames is not None):
                target_per_frame = self.pruning_layers[layer_idx]
                target = target_per_frame * self.actual_video_frames
                print(f"target{target}")
                should_prune = self.current_visual_length > target
            else:
                should_prune = False
                target = 0

            # ── PRE-layer: compute R (relevance) if pruning at this layer ──
            if should_prune:
                visual_start = self.system_prompt_length
                visual_end   = visual_start + self.current_visual_length
                device       = hidden_states.device

                print(f"\n{'='*80}")
                print(f"[TOPS S2] Layer {layer_idx}: {self.current_visual_length} → {target} tokens "
                      f"({self.actual_video_frames} frames × {target_per_frame}/frame)")

                attn_importance = self._compute_visual_token_importance(
                    hidden_states, decoder_layer, visual_start, visual_end
                )

            # Accumulate visual tokens (records what THIS layer actually sees)
            if tops_enabled and self.current_visual_length is not None:
                visual_token_accumulator += float(self.current_visual_length)

            # ── Execute decoder layer ──
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            # ── POST-layer: RDC greedy selection using PRE-layer R + POST-layer D/C ──
            if should_prune:
                device = hidden_states.device
                prev_length  = self.current_visual_length
                visual_start = self.system_prompt_length
                visual_end   = visual_start + prev_length

                post_visual  = hidden_states[:, visual_start:visual_end]  # (B, V, D)
                retained_num = min(target, post_visual.shape[1])

                visual_token_indices = self._rdc_greedy_select(
                    attn_importance, post_visual, retained_num, device
                )

                # Build retain indices to update hidden_states and position_ids
                retain_token_indices = torch.cat([
                    torch.arange(self.system_prompt_length, device=device),
                    visual_token_indices + self.system_prompt_length,
                    torch.arange(visual_end, hidden_states.shape[1], device=device),
                ])

                # DEBUG: validate before fancy indexing
                rti_max = retain_token_indices.max().item()
                hs_len = hidden_states.shape[1]
                vi_min = visual_token_indices.min().item() if visual_token_indices.numel() > 0 else -1
                vi_max = visual_token_indices.max().item() if visual_token_indices.numel() > 0 else -1
                print(f"[DEBUG retain_indices] hs_len={hs_len}, retain_max={rti_max}, "
                      f"visual_token_indices range=[{vi_min}, {vi_max}], "
                      f"visual_end={visual_end}, prev_length={prev_length}")
                if rti_max >= hs_len:
                    print(f"[DEBUG retain_indices] !!! OUT OF BOUNDS: retain_max={rti_max} >= hs_len={hs_len}")
                torch.cuda.synchronize()

                hidden_states = hidden_states[:, retain_token_indices].contiguous()
                torch.cuda.synchronize()  # force CUDA error to surface here if OOB
                position_ids  = position_ids[:, retain_token_indices]
                # Update position_ids (3D: [3, B, S]), cache_position, position_embeddings, causal_mask
                # new_len = hidden_states.shape[1]
                # position_ids = position_ids[:, :, :new_len]
                # cache_position = cache_position[:new_len]
                # position_embeddings = tuple(
                #     pos_emb[:, :, retain_token_indices] for pos_emb in position_embeddings
                # )
                # attention_mask = self._update_causal_mask(
                #     None, hidden_states, cache_position, past_key_values, output_attentions
                # )

                # Rebuild attention_mask for the shortened sequence
                if self._attn_implementation == "flash_attention_2":
                    attention_mask = None
                elif self._attn_implementation == "sdpa":
                    attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                        None,
                        (hidden_states.shape[0], hidden_states.shape[1]),
                        hidden_states, 0,
                    )
                else:
                    attention_mask = _prepare_4d_causal_attention_mask(
                        None,
                        (hidden_states.shape[0], hidden_states.shape[1]),
                        hidden_states, 0,
                        sliding_window=self.config.sliding_window,
                    )

                self.current_visual_length = visual_token_indices.numel()
                tops_enabled = self.current_visual_length > 0
                self.stage2_events.append((layer_idx, prev_length, self.current_visual_length))
                self.stage2_reported_layers.add(layer_idx)

                print(f"  Pruned: {prev_length} → {self.current_visual_length} tokens")
                print(f"{'='*80}\n")

            elif (tops_enabled
                    and layer_idx in self.pruning_layers
                    and layer_idx not in self.stage2_reported_layers):
                # Skip-prune logging
                print(f"[TOPS S2] Layer {layer_idx}: skip "
                      f"(current {self.current_visual_length} ≤ target {target})")
                self.stage2_skip_events.append((layer_idx, self.current_visual_length, target))
                self.stage2_reported_layers.add(layer_idx)

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # Compute average tokens per layer for progress display
        if seq_length > 1 and self.current_visual_length is not None:
            average_tokens = visual_token_accumulator / len(self.layers) if self.layers else 0.0
            self.visual_token_num = float(average_tokens)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # ── Stage summary ──
        if seq_length > 1 and self.prefill_done and not self._stage_summary_emitted:
            print(f"\n{'='*80}")
            print(f"[TOPS Summary]")
            print(f"  Stage 1 detected tokens : {self.stage1_actual_tokens}")
            print(f"  Frame count        : {self.actual_video_frames}")
            if self.stage2_events:
                print(f"  Stage 2 progressive pruning:")
                for l, before, after in self.stage2_events:
                    print(f"    Layer {l}: {before} → {after} tokens")
            else:
                print("  Stage 2: no pruning applied")
            print(f"  Final visual tokens      : {self.current_visual_length}")
            print(f"  Layer-avg visual tokens  : {self.visual_token_num:.2f}")
            print(f"  Target per frame         : {self.target_visual_tokens}")
            print(f"{'='*80}\n")
            self._stage_summary_emitted = True

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
