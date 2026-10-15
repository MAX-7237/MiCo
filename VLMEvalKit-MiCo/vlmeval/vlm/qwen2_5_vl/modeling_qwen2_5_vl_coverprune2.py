# coding=utf-8
"""
TOPS Stage 2: Progressive visual token pruning with RDC (Relevance-Diversity-Coverage)
greedy selection for Qwen2.5-VL, ported from modeling_qwen2_tops.py (InternVL3).

Key design:
  - Relevance (R): attention importance computed via QKV probe on PRE-layer hidden_states
  - Diversity (D): cosine dissimilarity computed on POST-layer hidden_states
  - Coverage (C): representativeness computed on POST-layer hidden_states
  - Greedy selection jointly optimizing R + alpha*D + lambda*C
"""

import os
import math
from typing import Dict, List, Optional, Tuple, Union, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel


logger = logging.get_logger(__name__)


# TOPS pruning schedule: Qwen2.5-VL-7B (28 layers)
# Format: {scale: {target_tokens: [(layer_idx_1indexed, keep_tokens), ...]}}
TOPS_SCHEDULE: Dict[str, Dict[int, List[Tuple[int, int]]]] = {
    "7b": {
        512: [(12, 281), (16, 77)],
        256: [(12, 139), (16, 39)],
        128: [(12, 71),  (16, 19)],
    },
}


class CoverPrune2Qwen2_5_VLModel(Qwen2_5_VLModel):
    """
    TOPS Stage-2 for Qwen2.5-VL: progressive RDC greedy pruning inside LLM layers.
    Ported from modeling_qwen2_tops.py (InternVL3 TOPS implementation).

    Assumes Stage-1 (THCP) has already reduced visual tokens before the LLM.
    """

    def __init__(
        self,
        config: Qwen2_5_VLConfig,
        visual_token_num: int,
        disable_stage2: bool = False,
        pruning_layers: Optional[Dict[int, int]] = None,
        attn_h: bool = False,
        dynamic_r: bool = False,
        s1_mult: int = 2,
    ):
        super().__init__(config)
        # transformers 4.52+: transformer layers live inside self.language_model
        lm = self.language_model
        self.embed_tokens = lm.embed_tokens
        self.layers = lm.layers
        self.norm = lm.norm
        self.rotary_emb = lm.rotary_emb
        self._update_causal_mask = lm._update_causal_mask

        self.system_prompt_length = 15
        self.visual_token_length = 1296  # default Stage-0 tokens
        self.visual_token_num = 0.0
        self.visual_token_len = 0.0

        try:
            visual_token_num = int(visual_token_num)
        except (TypeError, ValueError):
            raise ValueError(f"visual_token_num must be an integer, got {visual_token_num!r}")

        n_layers = getattr(config, 'num_hidden_layers', None) or config.text_config.num_hidden_layers
        self.scale = "7b" if n_layers == 28 else "unknown"

        self.target_visual_tokens = visual_token_num
        self.attn_h = bool(attn_h)
        self.dynamic_r = bool(dynamic_r)
        # S1 budget multiple of T (outer forward reads this; default 2 = legacy).
        self.s1_mult = max(1, int(s1_mult))
        # tops_2 (single-stage): Stage-1 prunes directly to target (NOT target×2).
        self.disable_stage2 = disable_stage2
        if disable_stage2:
            self.stage1_keep_tokens = max(1, visual_token_num) if visual_token_num > 0 else self.visual_token_length
        else:
            self.stage1_keep_tokens = max(1, visual_token_num) * self.s1_mult if visual_token_num > 0 else self.visual_token_length
        self.stage0_visual_tokens = self.visual_token_length
        self.stage1_expected_tokens = min(self.visual_token_length, self.stage1_keep_tokens)

        schedule = TOPS_SCHEDULE.get(self.scale, {}).get(self.target_visual_tokens)
        if disable_stage2:
            # Single-stage: disable in-LLM Stage-2 pruning entirely.
            self.pruning_layers: Dict[int, int] = {}
            logger.warning_once("tops_2 single-stage: in-LLM Stage-2 pruning DISABLED (pruning_layers = {})")
        elif pruning_layers is not None:
            # attn_cov_vit_proj_llm_12: one cut at TOPS first layer, FastV-style R.
            # TOPS_SCHEDULE only lists 512/256/128; T=64/32 must not fall through
            # to "missing schedule → S2 off".
            self.pruning_layers = {int(k): int(v) for k, v in pruning_layers.items()}
        elif schedule is None:
            self.pruning_layers: Dict[int, int] = {}
            if visual_token_num > 0:
                logger.warning_once(
                    "TOPS schedule missing for target %s on scale %s. Progressive pruning disabled.",
                    visual_token_num, self.scale,
                )
        else:
            self.pruning_layers = {layer_idx: target for layer_idx, target in schedule}
        # Explicit H-ViT sweep arms recompute the sole S2 keep count after S1,
        # because small images can enter the LLM with fewer than nominal 2T tokens.
        self._dynamic_k_layers = tuple(sorted(self.pruning_layers))

        print(f"[TOPS Init] Qwen2.5-VL")
        print(f"  Scale: {self.scale} ({n_layers} layers)")
        print(f"  Target visual tokens: {self.target_visual_tokens}")
        print(f"  Pruning schedule (1-indexed): {self.pruning_layers}")

        self.reset_state()

    def _refresh_dynamic_r(self, n_enter: int) -> None:
        if not self.dynamic_r:
            return
        n_enter = int(n_enter)
        n_layers = len(self.layers)
        refreshed: Dict[int, int] = {}
        for k_layer in self._dynamic_k_layers:
            if n_enter <= self.target_visual_tokens:
                keep = n_enter
            else:
                raw_keep = (
                    self.target_visual_tokens * n_layers - k_layer * n_enter
                ) / (n_layers - k_layer)
                keep = int(round(raw_keep))
                keep = max(1, min(keep, n_enter))
            refreshed[int(k_layer)] = int(keep)
        self.pruning_layers = refreshed
        print(
            f"[CoverPrune2 H-ViT] refresh dynamic R: N_enter={n_enter} "
            f"T={self.target_visual_tokens} layers={self.pruning_layers}",
            flush=True,
        )

    def reset_state(self):
        self.prefill_done = False
        self.current_visual_length: Optional[int] = None
        self.original_visual_length: Optional[int] = None
        self.visual_token_sum = 0.0
        self.stage1_actual_tokens: Optional[int] = None
        self.stage2_events: List[Tuple[int, int, int]] = []
        self.stage2_skip_events: List[Tuple[int, int, int]] = []
        self.stage2_reported_layers: Set[int] = set()
        self._stage_summary_emitted = False
        self.visual_token_num = 0.0
        self.visual_token_len = 0.0

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
        Compute the (V,) Stage-2 task_relevance for each visual token.

        The relevance definition is an ABLATION KNOB (env S2_TASK_REL); the
        attn_H_cov_vit_proj_llm framework around it is unchanged (this vector is
        multiplied by |h| and fed to the RDC coverage greedy downstream).

          all_text      (default): softmax attn from EVERY text token to each
                                    visual token, averaged (heads,text,batch).
          last_token              : softmax attn from ONLY the last text token.
          sim_mean                : mean cosine sim (all text tokens ↔ visual)
                                    in pre-layer hidden space, no softmax.
          sim_mean_inv            : 1 - sim_mean  (favour dissimilar tokens).
          uniform                 : constant 1 for every visual token (relevance
                                    ablated out; selection driven by |h|+coverage).
        """
        mode = os.environ.get("S2_TASK_REL", "all_text").strip().lower()
        # Dual-stage component ablation: H_VIT_NO_ATTN=1 forces the constant
        # relevance, same as S2_TASK_REL=uniform.
        if os.environ.get("H_VIT_NO_ATTN", "0").strip() in ("1", "true", "True", "yes"):
            mode = "uniform"
        V = int(visual_end - visual_start)
        device = hidden_states.device

        if mode == "uniform":
            imp = torch.ones(V, dtype=torch.float32, device=device)
            print(f"[CoverPrune2 S2] task_rel=uniform: V={V} (relevance=1)")
            return imp

        if mode in ("sim_mean", "sim_mean_inv"):
            # Cosine similarity in the PRE-layer hidden space (no attention probe).
            vis = hidden_states[:, visual_start:visual_end, :]          # (B,V,D)
            txt = hidden_states[:, visual_end:, :]                       # (B,Tt,D)
            if txt.size(1) == 0:
                txt = hidden_states[:, -1:, :]
            vis_n = F.normalize(vis.float(), dim=-1)
            txt_n = F.normalize(txt.float(), dim=-1)
            sim = torch.matmul(txt_n, vis_n.transpose(1, 2))            # (B,Tt,V)
            imp = sim.mean(dim=1).mean(dim=0)                            # (V,)
            if mode == "sim_mean_inv":
                imp = 1.0 - imp
            print(f"[CoverPrune2 S2] task_rel={mode} (n_text={txt.size(1)}): "
                  f"V={V} min={imp.min():.4f} max={imp.max():.4f}")
            return imp

        # ── attention-based variants (all_text / last_token) ──
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

        query_states = query_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        if num_kv_heads != num_heads:
            repeat_factor = num_heads // num_kv_heads
            key_states = key_states.repeat_interleave(repeat_factor, dim=1)

        # Apply MRoPE using the same function as the model itself
        cos, sin = position_embeddings
        mrope_section = self_attn.rope_scaling["mrope_section"]
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, mrope_section
        )

        # COVER_Prune_1: 用【所有文本token】做query (非TOPS的dynamic rater子集)
        text_positions = torch.arange(visual_end, seq_len, device=hidden_states.device)
        if text_positions.numel() == 0:
            text_positions = torch.tensor([seq_len - 1], device=hidden_states.device)
        # last_token variant: keep only the final text position as the single rater.
        if mode == "last_token":
            text_positions = text_positions[-1:]

        query_raters = torch.index_select(query_states, 2, text_positions)
        key_visual = key_states[:, :, visual_start:visual_end, :]

        attn_scores = torch.matmul(query_raters, key_visual.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Average over the selected text tokens, heads, batch → (V,)
        visual_importance = attn_weights.mean(dim=2).mean(dim=1).mean(dim=0)

        print(f"[CoverPrune2 S2] task_rel={mode} (n_text={text_positions.numel()}), "
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
        attn: (V,) normalized relevance scores
        post_hidden: (B, V, D) POST-layer visual token features
        Returns: (retained_num,) sorted selected indices (batch=1 path)
        """
        a2 = float(os.environ.get("A2", "1.0"))
        alpha_2 = float(os.environ.get("ALPHA_2", "1.0"))
        lambda_2 = float(os.environ.get("LAMBDA_2", "1.0"))

        # S2 component switches for the dual-stage ablation (default ON = unchanged).
        # See attn_cov_select._comp_on for the shared env convention.
        def _on(n):
            return os.environ.get(n, "0").strip() not in ("1", "true", "True", "yes")
        use_h = _on("H_VIT_NO_H")
        use_cov = _on("H_VIT_NO_COV")

        B, V, D = post_hidden.shape

        # H is deliberately read after the physical S2 layer, unlike S1 where
        # it is measured on pre-projector ViT features.
        attn = attn.detach()
        if attn.dim() == 1:
            attn = attn.unsqueeze(0)  # (1, V)
        if self.attn_h:
            if attn.shape[0] == 1 and B > 1:
                attn = attn.expand(B, -1)
            if attn.shape != (B, V):
                raise ValueError(f"S2 attention shape {tuple(attn.shape)} != post-hidden {(B, V)}")
            attn = attn.float().clamp(min=0.0)
            if use_h:
                attn = attn * post_hidden.detach().float().norm(dim=-1)
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

            if not use_cov:
                # Coverage ablated: no similarity structure left, so take the
                # top-k by the remaining relevance signal.
                k = min(int(retained_num), V)
                sel = torch.topk(a, k).indices
                all_selected.append(torch.sort(sel).values)
                continue

            available = torch.ones(V, dtype=torch.bool, device=device)
            selected = []

            # Seed: 加权覆盖最大的 token (g=0时 Σ_v a_v·max(0,sim(v,i)))
            cov0 = (a.unsqueeze(0) * torch.clamp(feat @ feat.t(), min=0.0)).sum(dim=1)  # (V,)
            i0 = torch.argmax(cov0).item()
            selected.append(i0)
            available[i0] = False

            # Clamp coverage itself: negative-to-less-negative similarity is not recovered information.
            max_sim_to_S = (feat @ feat[i0]).clamp(min=0.0)  # (V,)
            max_sim_to_S[i0] = 1.0

            full_sim = (feat @ feat.t()).clamp(min=0.0)  # (V, V)

            for _ in range(1, retained_num):
                if not available.any():
                    break

                # COVER_Prune_2: attn 乘进求和的每一项 (w_v=被覆盖token v的attn), 外面不再乘候选自身attn
                # gain[i] = Σ_v a_v · max(0, sim(v,i) - g_v)
                gain = (a.unsqueeze(0) * torch.clamp(full_sim - max_sim_to_S.unsqueeze(0), min=0.0)).sum(dim=1)
                scores = gain
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
        print(f"[TOPS S2] RDC: retained={sel.numel()}, a2={a2} , alpha_2={alpha_2}, lambda_2={lambda_2}, "
              f"avg_attn={avg_attn:.4f}, avg_sim={avg_sim:.4f}")

        return all_selected[0]  # batch=1 common path

    def _compute_schedule_average(self, stage1_tokens, total_layers):
        if stage1_tokens is None or total_layers <= 0:
            return None
        sorted_layers = sorted(self.pruning_layers.items())
        total_tokens = 0.0
        current_tokens = float(stage1_tokens)
        current_layer = 1
        for layer_idx, target in sorted_layers:
            if layer_idx < current_layer or layer_idx > total_layers:
                continue
            if layer_idx > current_layer:
                total_tokens += (layer_idx - current_layer) * current_tokens
            # Pruning layer processes current_tokens then prunes to new_tokens
            total_tokens += current_tokens
            new_tokens = float(min(current_tokens, target))
            current_tokens = new_tokens
            current_layer = layer_idx + 1
        if current_layer <= total_layers:
            total_tokens += (total_layers - current_layer + 1) * current_tokens
        return total_tokens / total_layers

    def _compute_observed_average(self, total_layers):
        if not self.prefill_done or self.stage1_actual_tokens is None or total_layers <= 0:
            return None
        total_tokens = 0.0
        current_tokens = float(self.stage1_actual_tokens)
        current_layer = 1
        for layer_idx, before_tokens, after_tokens in sorted(self.stage2_events):
            if layer_idx < current_layer or layer_idx > total_layers:
                continue
            if layer_idx > current_layer:
                total_tokens += (layer_idx - current_layer) * current_tokens
            # Pruning layer processes before_tokens then prunes to after_tokens
            total_tokens += float(before_tokens) if before_tokens is not None else float(after_tokens)
            current_tokens = float(after_tokens)
            current_layer = layer_idx + 1
        if current_layer <= total_layers:
            total_tokens += (total_layers - current_layer + 1) * current_tokens
        return total_tokens / total_layers

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
        cache_position: Optional[torch.LongTensor] = None,
        stage1_visual_span: Optional[Tuple[int, int]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        def _is_empty_cache(cache):
            if cache is None:
                return True
            if isinstance(cache, DynamicCache):
                return cache.get_seq_length() == 0
            return False

        if _is_empty_cache(past_key_values):
            self.reset_state()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.is_gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # Qwen2.5-VL uses 3D position_ids: [3, B, S]
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        seq_length = inputs_embeds.shape[1]

        # ── Prefill phase init ──
        if seq_length > 1 and not self.prefill_done:
            if stage1_visual_span is not None:
                visual_start, detected_visual_len = stage1_visual_span
                if (inputs_embeds.shape[0] != 1 or visual_start < 0 or detected_visual_len <= 0
                        or detected_visual_len > self.stage1_keep_tokens
                        or visual_start + detected_visual_len >= seq_length):
                    raise ValueError(f'Invalid post-S1 visual span {stage1_visual_span} for sequence length {seq_length}')
                self.system_prompt_length = visual_start
            else:
                # Preserve the existing interface for pruning methods that do not supply exact S1 metadata.
                detected_visual_len = max(seq_length - self.system_prompt_length - 10, 0)
                if self.target_visual_tokens > 0:
                    detected_visual_len = min(detected_visual_len, self.stage1_expected_tokens)
                if detected_visual_len <= 0:
                    detected_visual_len = max(min(self.visual_token_length, seq_length - self.system_prompt_length - 1), 0)

            self.current_visual_length = detected_visual_len
            self.original_visual_length = detected_visual_len
            self.stage1_actual_tokens = detected_visual_len
            self._refresh_dynamic_r(detected_visual_len)
            self.prefill_done = True
            self.visual_token_sum = 0.0
            print(f"[TOPS] Prefill: seq_length={seq_length}, detected_visual_len={detected_visual_len}, "
                  f"stage1_keep_tokens={self.stage1_keep_tokens}")

        tops_enabled = (
            self.prefill_done
            and self.current_visual_length is not None
            and self.current_visual_length > 0
            and len(self.pruning_layers) > 0
        )
        visual_token_accumulator = 0.0

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_idx = decoder_layer.self_attn.layer_idx + 1

            if seq_length > 1 and self.current_visual_length is not None:
                visual_token_accumulator += float(self.current_visual_length)

            should_prune = (
                tops_enabled
                and layer_idx in self.pruning_layers
                and self.current_visual_length is not None
                and self.current_visual_length > self.pruning_layers[layer_idx]
            )

            if should_prune:
                target = self.pruning_layers[layer_idx]
                visual_start = self.system_prompt_length
                visual_end = visual_start + self.current_visual_length
                device = hidden_states.device

                if visual_end > hidden_states.shape[1]:
                    visual_end = hidden_states.shape[1]
                    self.current_visual_length = visual_end - visual_start

                print(f"\n[TOPS S2] Layer {layer_idx}: {self.current_visual_length} → {target} tokens")

                # ── Step 1: Compute attention relevance on PRE-layer hidden_states ──
                attn = self._compute_visual_token_importance(
                    hidden_states, decoder_layer, position_embeddings,
                    visual_start, visual_end,
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
                orig_seq_len = hidden_states.shape[1]  # save before rebuild

                select_visual_tokens = post_visual.gather(
                    1, selected_t.unsqueeze(0).unsqueeze(-1).expand(-1, -1, post_visual.shape[-1])
                )  # (B, R, D)

                hidden_states = torch.cat((
                    hidden_states[:, :visual_start],
                    select_visual_tokens,
                    hidden_states[:, visual_end:],
                ), dim=1)

                # retain_token_indices: sys + selected visual + text (using original seq positions)
                retain_token_indices = torch.cat((
                    torch.arange(visual_start, device=device),
                    selected_t + visual_start,
                    torch.arange(visual_end, orig_seq_len, device=device),
                ))

                # Update position_ids (3D: [3, B, S]), cache_position, position_embeddings, causal_mask
                new_len = hidden_states.shape[1]
                position_ids = position_ids[:, :, :new_len]
                cache_position = cache_position[:new_len]
                position_embeddings = tuple(
                    pos_emb[:, :, retain_token_indices] for pos_emb in position_embeddings
                )
                causal_mask = self._update_causal_mask(
                    None, hidden_states, cache_position, past_key_values, output_attentions
                )

                prev_length = self.current_visual_length
                self.current_visual_length = selected_t.numel()
                tops_enabled = self.current_visual_length > 0
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
                    and self.current_visual_length is not None
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
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if seq_length > 1 and self.current_visual_length is not None:
            average_tokens = visual_token_accumulator / len(self.layers) if self.layers else 0.0
            self.visual_token_num = float(average_tokens)
            self.visual_token_len = float(average_tokens)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        # ── Stage summary ──
        if seq_length > 1 and self.prefill_done and not self._stage_summary_emitted:
            total_layers = len(self.layers)
            observed_avg = self._compute_observed_average(total_layers)
            schedule_avg = self._compute_schedule_average(self.stage1_expected_tokens, total_layers)
            print("\n[TOPS] Stage summary:")
            if self.stage1_actual_tokens is not None:
                print(f"  Stage 1: {self.stage0_visual_tokens} → {self.stage1_actual_tokens} "
                      f"(expected cap {self.stage1_expected_tokens})")
            if self.stage2_events:
                for layer_idx, before, after in self.stage2_events:
                    print(f"  Stage 2 Layer {layer_idx}: {before} → {after}")
                print(f"  Final visual tokens: {self.current_visual_length}")
            else:
                print("  Stage 2: no pruning applied")
            if observed_avg is not None:
                print(f"  Layer-avg tokens (observed): {observed_avg:.2f} (target {self.target_visual_tokens})")
            if schedule_avg is not None:
                print(f"  Layer-avg tokens (schedule): {schedule_avg:.2f}")
            self._stage_summary_emitted = True

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
