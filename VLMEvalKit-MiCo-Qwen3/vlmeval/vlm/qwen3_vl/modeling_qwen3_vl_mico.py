"""MiCo weighted coverage selection and decoder pruning for qwen3_vl."""

import os
import math
from typing import Dict, List, Optional, Tuple, Union, Set
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        create_causal_mask as _create_causal_mask_impl,
    )

    _BASE_MODEL_CLS = Qwen3VLModel
    _USING_QWEN3 = True
except ImportError:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb as apply_rotary_pos_emb,
    )
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLModel as _BASE_MODEL_CLS,
    )

    _USING_QWEN3 = False
    _create_causal_mask_impl = None
logger = logging.get_logger(__name__)


class MiCoQwen3VLModel(_BASE_MODEL_CLS):
    """
    MiCo Stage-2 for Qwen3-VL: progressive weighted coverage greedy pruning inside LLM layers.
    Ported from modeling_qwen2_5_vl_mico.py.

    Assumes Stage-1 has already reduced visual tokens before the LLM.
    """

    def __init__(self, config, visual_token_num: int):
        super().__init__(config)
        lm = self.language_model
        self.embed_tokens = lm.embed_tokens
        self.layers = lm.layers
        self.norm = lm.norm
        self.rotary_emb = lm.rotary_emb
        self._lm_update_causal_mask = getattr(lm, "_update_causal_mask", None)
        self.visual_token_positions: Optional[torch.LongTensor] = None
        self.visual_token_length = 1296
        self.visual_token_num = 0.0
        self.visual_token_len = 0.0
        try:
            visual_token_num = int(visual_token_num)
        except (TypeError, ValueError):
            raise ValueError(f"visual_token_num must be an integer, got {visual_token_num!r}")
        n_layers = (
            getattr(config, "num_hidden_layers", None) or config.text_config.num_hidden_layers
        )
        if n_layers <= 16:
            self.scale = "2b"
        elif n_layers <= 22:
            self.scale = "4b"
        elif n_layers <= 30:
            self.scale = "7b"
        elif n_layers <= 40:
            self.scale = "8b"
        else:
            self.scale = "unknown"
        self.target_visual_tokens = visual_token_num
        self.stage1_keep_tokens = (
            max(1, visual_token_num) * 2 if visual_token_num > 0 else self.visual_token_length
        )
        self.stage0_visual_tokens = self.visual_token_length
        self.stage1_expected_tokens = min(self.visual_token_length, self.stage1_keep_tokens)
        self.pruning_layers = {}
        print(f"  Scale: {self.scale} ({n_layers} layers)")
        print(f"  Target visual tokens: {self.target_visual_tokens}")
        print(f"  Pruning schedule (1-indexed): {self.pruning_layers}")
        self.reset_state()

    def _get_causal_mask(
        self, input_tensor, attention_mask, cache_position, past_key_values, position_ids=None
    ):
        position_ids = None
        if _create_causal_mask_impl is not None:
            return _create_causal_mask_impl(
                self.config,
                input_tensor,
                attention_mask,
                cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        elif self._lm_update_causal_mask is not None:
            return self._lm_update_causal_mask(
                attention_mask, input_tensor, cache_position, past_key_values, False
            )
        return None

    def set_visual_token_positions(self, positions: torch.LongTensor) -> None:
        """Absolute positions (in the post-stage-1 sequence) of the surviving image tokens."""
        self.visual_token_positions = positions.detach()

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

    def _select_text_raters(
        self, hidden_states: torch.Tensor, visual_positions: torch.LongTensor
    ) -> torch.Tensor:
        """Select informative text tokens as raters via normalized text-visual similarity.

        Returns ABSOLUTE positions (not offsets), per qwen2_5_vl/modeling_qwen2_5_vl_mico_variant3.py:191-221.
        """
        seq_len = hidden_states.shape[1]
        non_visual = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)
        non_visual[visual_positions] = False
        text_positions = non_visual.nonzero(as_tuple=False).squeeze(-1)
        text_hidden = hidden_states.index_select(1, text_positions)
        if text_hidden.size(1) == 0:
            return hidden_states.new_empty((0,), dtype=torch.long)
        visual_hidden = hidden_states.index_select(1, visual_positions)
        text_norm = F.normalize(text_hidden, dim=-1)
        visual_norm = F.normalize(visual_hidden, dim=-1)
        text_visual_sim = torch.matmul(text_norm, visual_norm.transpose(1, 2))
        text_importance = text_visual_sim.mean(dim=-1)
        if text_importance.dim() == 2:
            text_importance = text_importance.mean(dim=0)
        mean_importance = text_importance.mean()
        selected = torch.nonzero(text_importance > mean_importance, as_tuple=False).squeeze(-1)
        if selected.numel() == 0:
            keep = max(1, text_importance.numel() // 2)
            selected = text_importance.topk(keep).indices
        return text_positions.index_select(0, selected)

    def _compute_visual_token_importance(
        self,
        hidden_states: torch.Tensor,
        decoder_layer: nn.Module,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        visual_positions: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Compute attention-based relevance via QKV probe on PRE-layer hidden_states.
        Uses dynamic text rater tokens as queries, visual tokens as keys.
        Returns: (V,) importance scores for each visual token.
        """
        self_attn = decoder_layer.self_attn
        attn_input = hidden_states
        input_layernorm = getattr(decoder_layer, "input_layernorm", None)
        if input_layernorm is not None:
            attn_input = input_layernorm(attn_input)
        bsz, seq_len, _ = attn_input.shape
        head_dim = self_attn.head_dim
        num_heads = self_attn.q_proj.out_features // head_dim
        num_kv_heads = self_attn.k_proj.out_features // head_dim
        query_states = self_attn.q_proj(attn_input)
        key_states = self_attn.k_proj(attn_input)
        query_states = query_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        if hasattr(self_attn, "q_norm"):
            query_states = self_attn.q_norm(query_states)
        if hasattr(self_attn, "k_norm"):
            key_states = self_attn.k_norm(key_states)
        if num_kv_heads != num_heads:
            repeat_factor = num_heads // num_kv_heads
            key_states = key_states.repeat_interleave(repeat_factor, dim=1)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        non_visual = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)
        non_visual[visual_positions] = False
        text_positions = non_visual.nonzero(as_tuple=False).squeeze(-1)
        if text_positions.numel() == 0:
            text_positions = torch.tensor(
                [seq_len - 1], device=hidden_states.device, dtype=torch.long
            )
        query_raters = torch.index_select(query_states, 2, text_positions)
        key_visual = torch.index_select(key_states, 2, visual_positions)
        attn_scores = torch.matmul(query_raters, key_visual.transpose(-2, -1))
        attn_scores = attn_scores * self_attn.scaling
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        visual_importance = attn_weights.mean(dim=2).mean(dim=1).mean(dim=0)
        print(
            f"[MiCo S2] text_rater positions numel: {text_positions.numel()}, importance: min={visual_importance.min():.4f}, max={visual_importance.max():.4f}"
        )
        return visual_importance

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

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
            total_tokens += current_tokens
            current_tokens = float(min(current_tokens, target))
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
            total_tokens += (
                float(before_tokens) if before_tokens is not None else float(after_tokens)
            )
            current_tokens = float(after_tokens)
            current_layer = layer_idx + 1
        if current_layer <= total_layers:
            total_tokens += (total_layers - current_layer + 1) * current_tokens
        return total_tokens / total_layers

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
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        def _is_empty_cache(cache):
            if cache is None:
                return True
            if isinstance(cache, DynamicCache):
                return cache.get_seq_length() == 0
            return False

        if _is_empty_cache(past_key_values):
            self.reset_state()
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
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if self.is_gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        if use_cache and past_key_values is None and (not torch.jit.is_tracing()):
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        text_position_ids = None
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        elif position_ids.dim() == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        causal_mask = self._get_causal_mask(
            inputs_embeds,
            attention_mask,
            cache_position,
            past_key_values,
            position_ids=text_position_ids,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        seq_length = inputs_embeds.shape[1]
        if seq_length > 1 and (not self.prefill_done):
            if self.visual_token_positions is None:
                raise RuntimeError(
                    "MiCo stage-2: visual token positions were not pushed by the wrapper. The 'seq_length - 15 - 10' heuristic is off by 11 tokens on Qwen3-VL and must not be used as a fallback."
                )
            detected_visual_len = int(self.visual_token_positions.numel())
            self.current_visual_length = detected_visual_len
            self.original_visual_length = detected_visual_len
            self.stage1_actual_tokens = detected_visual_len
            self.prefill_done = True
            self.visual_token_sum = 0.0
            print(
                f"[MiCo] Prefill: seq_length={seq_length}, detected_visual_len={detected_visual_len}, stage1_keep_tokens={self.stage1_keep_tokens}"
            )
        mico_enabled = (
            self.prefill_done
            and self.current_visual_length is not None
            and (self.current_visual_length > 0)
            and (self.visual_token_positions is not None)
            and (len(self.pruning_layers) > 0)
        )
        visual_token_accumulator = 0.0
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_idx = decoder_layer.self_attn.layer_idx + 1
            if seq_length > 1 and self.current_visual_length is not None:
                visual_token_accumulator += float(self.current_visual_length)
            should_prune = (
                mico_enabled
                and seq_length > 1
                and (layer_idx in self.pruning_layers)
                and (self.current_visual_length is not None)
                and (self.current_visual_length > self.pruning_layers[layer_idx])
            )
            if should_prune:
                target = self.pruning_layers[layer_idx]
                device = hidden_states.device
                visual_positions = self.visual_token_positions.to(device)
                print(
                    f"\n[MiCo S2] Layer {layer_idx}: {self.current_visual_length} → {target} tokens"
                )
                attn = self._compute_visual_token_importance(
                    hidden_states, decoder_layer, position_embeddings, visual_positions
                )
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                post_visual = hidden_states.index_select(1, visual_positions)
                retained_num = min(int(target), post_visual.shape[1])
                selected_t = self._select_visual_tokens(attn, post_visual, retained_num, device)
                orig_seq_len = hidden_states.shape[1]
                kept_visual_abs = visual_positions.index_select(0, selected_t)
                retain_mask = torch.ones(orig_seq_len, dtype=torch.bool, device=device)
                retain_mask[visual_positions] = False
                retain_mask[kept_visual_abs] = True
                retain_token_indices = retain_mask.nonzero(as_tuple=False).squeeze(-1)
                hidden_states = hidden_states.index_select(1, retain_token_indices).contiguous()
                new_len = hidden_states.shape[1]
                position_ids = position_ids[:, :, retain_token_indices]
                if text_position_ids is not None:
                    text_position_ids = text_position_ids[:, retain_token_indices]
                cache_position = cache_position[:new_len]
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
                causal_mask = self._get_causal_mask(
                    hidden_states, None, cache_position, None, position_ids=None
                )
                if os.environ.get("ENABLE_DEBUG", "0") not in ("", "0", "false", "False"):
                    assert causal_mask is None or bool(
                        causal_mask[0, 0, -1].sum() == hidden_states.shape[1]
                    ), "post-prune causal mask is not plain-causal (packed-sequence poisoning)"
                if visual_pos_masks is not None:
                    _vis_pos = visual_pos_masks[0].nonzero(as_tuple=True)[0]
                    _surv = torch.isin(_vis_pos, retain_token_indices.to(_vis_pos.device))
                    if deepstack_visual_embeds is not None:
                        deepstack_visual_embeds = [d[_surv] for d in deepstack_visual_embeds]
                    visual_pos_masks = visual_pos_masks[
                        :, retain_token_indices.to(visual_pos_masks.device)
                    ]
                self.visual_token_positions = torch.searchsorted(
                    retain_token_indices, kept_visual_abs
                )
                prev_length = self.current_visual_length
                self.current_visual_length = selected_t.numel()
                mico_enabled = self.current_visual_length > 0
                self.stage2_events.append((layer_idx, prev_length, self.current_visual_length))
                self.stage2_reported_layers.add(layer_idx)
                print(
                    f"[MiCo S2] Layer {layer_idx} done: {prev_length} → {self.current_visual_length} tokens, total_seq={hidden_states.shape[1]}"
                )
            else:
                if (
                    mico_enabled
                    and self.current_visual_length is not None
                    and (layer_idx in self.pruning_layers)
                    and (layer_idx not in self.stage2_reported_layers)
                ):
                    target = self.pruning_layers[layer_idx]
                    print(
                        f"[MiCo S2] Layer {layer_idx}: skip (current {self.current_visual_length} ≤ target {target})"
                    )
                    self.stage2_skip_events.append((layer_idx, self.current_visual_length, target))
                    self.stage2_reported_layers.add(layer_idx)
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            _ds = layer_idx - 1
            if (
                deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and (0 <= _ds < len(deepstack_visual_embeds))
            ):
                assert deepstack_visual_embeds[_ds].shape[0] == int(
                    visual_pos_masks.sum()
                ), "deepstack rows out of sync with visual_pos_masks"
                hidden_states = self.language_model._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[_ds]
                )
        hidden_states = self.norm(hidden_states)
        if seq_length > 1 and self.current_visual_length is not None:
            average_tokens = visual_token_accumulator / len(self.layers) if self.layers else 0.0
            self.visual_token_num = float(average_tokens)
            self.visual_token_len = float(average_tokens)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = past_key_values if use_cache else None
        if seq_length > 1 and self.prefill_done and (not self._stage_summary_emitted):
            total_layers = len(self.layers)
            observed_avg = self._compute_observed_average(total_layers)
            schedule_avg = self._compute_schedule_average(self.stage1_expected_tokens, total_layers)
            print("\n[MiCo] Stage summary:")
            if self.stage1_actual_tokens is not None:
                print(
                    f"  Stage 1: {self.stage0_visual_tokens} → {self.stage1_actual_tokens} (expected cap {self.stage1_expected_tokens})"
                )
            if self.stage2_events:
                for layer_idx, before, after in self.stage2_events:
                    print(f"  Stage 2 Layer {layer_idx}: {before} → {after}")
                print(f"  Final visual tokens: {self.current_visual_length}")
            else:
                print("  Stage 2: no pruning applied")
            if observed_avg is not None:
                print(
                    f"  Layer-avg tokens (observed): {observed_avg:.2f} (target {self.target_visual_tokens})"
                )
            if schedule_avg is not None:
                print(f"  Layer-avg tokens (schedule): {schedule_avg:.2f}")
            self._stage_summary_emitted = True
        if not return_dict:
            return tuple(
                (v for v in [hidden_states, next_cache, all_hidden_states] if v is not None)
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=None,
        )
