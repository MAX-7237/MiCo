"""
Custom Qwen2 model with FastV visual token pruning for InternVL3.
Based on transformers Qwen2 implementation with inline pruning support.
"""

from functools import partial
from typing import Optional, Tuple, Union

import math
import torch
import torch.nn.functional as F

from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import logging


logger = logging.get_logger(__name__)


# Configuration dictionary for FastV pruning
# Format: R_dict[model_scale][pruning_layer][original_tokens][target_avg_tokens] = actual_tokens_to_keep
#
# The "target_avg_tokens" is the AVERAGE number of visual tokens across ALL 28 layers:
#   avg = (original_tokens * num_layers_before_pruning + R * num_layers_after_pruning) / total_layers
#   avg = (original_tokens * 2 + R * 26) / 28  (for pruning at layer 2)
#
# Solving for R:
#   R = (avg * 28 - original_tokens * 2) / 26
#
# Example for 1280 original tokens:
#   - Target avg=512: R = (512*28 - 1280*2) / 26 = (14336 - 2560) / 26 = 452.9 ≈ 453
#   - Target avg=384: R = (384*28 - 1280*2) / 26 = (10752 - 2560) / 26 = 315.1 ≈ 315
#   - Target avg=256: R = (256*28 - 1280*2) / 26 = (7168 - 2560) / 26 = 177.2 ≈ 177
#   - Target avg=128: R = (128*28 - 1280*2) / 26 = (3584 - 2560) / 26 = 39.4 ≈ 39
R_dict = {
    "8b": {  # InternVL3-8B (28 layers total)
        2: {  # Prune at layer 2 (0-indexed, so layers 0-1 keep original, layers 2-27 use R)
            256: {  # For 256 original visual tokens (e.g., 1 patch * 256 tokens/patch)
                # avg = (256 * 2 + R * 26) / 28
                256: 256,  # Target avg=256 → R = (256*28 - 256*2) / 26 = 250.5 ≈ 251
                              # But since we can't exceed 256, just keep all 256
                128: 122,  # Target avg=128 → R = (128*28 - 256*2) / 26 = 118.8 ≈ 119
                64: 60,    # Target avg=64 → R = (64*28 - 256*2) / 26 = 49.2 ≈ 49
            },
            1280: {  # For 1280 original visual tokens (e.g., 5 patches * 256 tokens/patch)
                # avg = (1280 * 2 + R * 26) / 28
                512: 453,  # Target avg=512 → R = (512*28 - 1280*2) / 26 = 452.9 ≈ 453
                384: 315,  # Target avg=384 → R = (384*28 - 1280*2) / 26 = 315.1 ≈ 315
                256: 177,  # Target avg=256 → R = (256*28 - 1280*2) / 26 = 177.2 ≈ 177
                128: 39,   # Target avg=128 → R = (128*28 - 1280*2) / 26 = 39.4 ≈ 39
            },
        },
    },
}


class VTPQwen2Model(Qwen2Model):
    """
    Qwen2Model with Visual Token Pruning support.
    Performs FastV pruning at a specific layer during forward pass.
    """

    def __init__(self, config: Qwen2Config, visual_token_num: int, num_image_token: int = 256):
        super().__init__(config)

        # InternVL3 parameters (fixed values, like Qwen2.5-VL)
        self.system_prompt_length = 41
        self.visual_token_length = 1280  # Fixed: 5 patches × 256 tokens/patch
        self.visual_token_num = visual_token_num  # Target average for display
        self.visual_token_len = 0
        self.fastv_debug = False
        self.pruning_method = 'fastv'

        # Determine model scale
        if config.num_hidden_layers == 28:
            self.scale = "8b"
        else:
            self.scale = "8b"
            logger.warning(f"Unknown model scale with {config.num_hidden_layers} layers, defaulting to 8b")

        # FastV configuration - directly lookup R from R_dict
        self.K = 2
        self.R = R_dict[self.scale][self.K][1280][visual_token_num]

        print(f"[FastV VTPQwen2Model Init]")
        print(f"  Model scale: {self.scale} ({config.num_hidden_layers} layers)")
        print(f"  Visual tokens: {self.visual_token_length}")
        print(f"  Target average: {visual_token_num}")
        print(f"  Pruning layer (K): {self.K}")
        print(f"  Retention (R): {self.R}")

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate last dimension by splitting heads into two halves for RoPE."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _compute_visual_token_importance(
        self,
        hidden_states: torch.Tensor,
        decoder_layer,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        current_visual_length: int,
    ) -> torch.Tensor:
        """
        Compute visual token importance using lightweight attention probe.

        This avoids requesting full attention weights from the model,
        keeping memory usage linear in the number of visual tokens.

        Args:
            hidden_states: Current hidden states [batch, seq_len, hidden_dim]
            decoder_layer: The decoder layer to probe
            position_embeddings: (cos, sin) tuple for RoPE
            current_visual_length: Current number of visual tokens

        Returns:
            visual_importance: Tensor of shape [current_visual_length] with importance scores
        """
        self_attn = decoder_layer.self_attn

        # Get pre-attention normalized input if available
        attn_input = hidden_states
        input_layernorm = getattr(decoder_layer, "input_layernorm", None)
        if input_layernorm is not None:
            attn_input = input_layernorm(attn_input)

        bsz, seq_len, _ = attn_input.shape
        num_heads = self_attn.num_heads if hasattr(self_attn, 'num_heads') else self_attn.config.num_attention_heads
        num_kv_heads = self_attn.num_key_value_heads if hasattr(self_attn, 'num_key_value_heads') else self_attn.config.num_key_value_heads
        head_dim = self_attn.head_dim if hasattr(self_attn, 'head_dim') else self_attn.config.hidden_size // num_heads

        # Project to Q and K (without V, as we only need attention scores)
        query_states = self_attn.q_proj(attn_input)
        key_states = self_attn.k_proj(attn_input)

        # Reshape for multi-head attention
        query_states = query_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        # Handle GQA (Grouped Query Attention) - repeat key/value heads if needed
        if num_kv_heads != num_heads:
            num_groups = num_heads // num_kv_heads
            key_states = key_states.repeat_interleave(num_groups, dim=1)

        # Apply RoPE
        cos, sin = position_embeddings
        query_states = (query_states * cos) + (self._rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)

        # Extract only the visual tokens region
        visual_start = self.system_prompt_length
        visual_end = visual_start + current_visual_length

        # Compute attention only for last query token → visual key tokens
        query_last = query_states[:, :, -1:, :]  # [batch, num_heads, 1, head_dim]
        key_visual = key_states[:, :, visual_start:visual_end, :]  # [batch, num_heads, visual_len, head_dim]

        # Attention scores: Q @ K^T / sqrt(d)
        attn_scores = torch.matmul(query_last, key_visual.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(head_dim)

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Aggregate: average across heads and batch
        visual_importance = attn_weights.mean(dim=1).squeeze(-2)  # [batch, visual_len]
        visual_importance = visual_importance.mean(dim=0)  # [visual_len]

        return visual_importance

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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # Create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # Flag to track if visual token pruning has been performed
        pruning_performed = False

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                ####################################################################################################
                # FastV Visual Token Pruning
                fastv_enabled = inputs_embeds.shape[1] > 1  # Prefill phase detection

                if fastv_enabled and layer_idx == self.K:
                    # Perform FastV pruning at layer K
                    device = hidden_states.device

                    if self.fastv_debug:
                        print(f"\n[FastV Pruning] Layer {layer_idx}")
                        print(f"  Visual tokens: {self.visual_token_length}")
                        print(f"  Retention (R): {self.R}")

                    # Compute visual token importance using lightweight probe
                    visual_importance = self._compute_visual_token_importance(
                        hidden_states, decoder_layer, position_embeddings, self.visual_token_length
                    )

                    # Select top-R tokens
                    keep = min(self.R, visual_importance.shape[0])
                    if keep > 0:
                        visual_token_indices = visual_importance.topk(keep).indices
                        visual_token_indices = torch.sort(visual_token_indices).values
                    else:
                        visual_token_indices = visual_importance.new_empty((0,), dtype=torch.long)

                    # Forward through the layer
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

                    # Construct retention indices
                    retain_token_indices = torch.cat((
                        torch.arange(self.system_prompt_length, device=device),
                        visual_token_indices + self.system_prompt_length,
                        torch.arange(
                            self.system_prompt_length + self.visual_token_length,
                            hidden_states.shape[1],
                            device=device,
                        ),
                    ))

                    # Apply pruning
                    hidden_states = hidden_states[:, retain_token_indices].contiguous()
                    position_ids = position_ids[:, retain_token_indices]
                    cache_position = cache_position[: retain_token_indices.shape[0]]
                    position_embeddings = [pos_emb[:, retain_token_indices] for pos_emb in position_embeddings]

                    # Recalculate causal mask
                    causal_mask = self._update_causal_mask(
                        None, hidden_states, cache_position, past_key_values, output_attentions
                    )

                    print(f"  ✓ Visual tokens: {self.visual_token_length} → {keep}")
                    print(f"{'='*80}\n")

                    # Mark that pruning has been performed
                    pruning_performed = True
                    layer_outputs = (hidden_states, layer_outputs[-1])

                else:
                    # Normal forward pass (or after pruning)
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
                ####################################################################################################

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class Qwen2ForCausalLMPruning(Qwen2ForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, pruning_method=None, visual_token_num=None, num_image_token=256):
        super().__init__(config)
        self.model = VTPQwen2Model(config, visual_token_num, num_image_token)

    def set_system_prompt_length(self, length: int):
        """Set the system prompt length for pruning calculations."""
        self.model.system_prompt_length = length

    def enable_fastv_debug(self, enable: bool = True):
        """Enable/disable FastV debug logging."""
        self.model.fastv_debug = enable
