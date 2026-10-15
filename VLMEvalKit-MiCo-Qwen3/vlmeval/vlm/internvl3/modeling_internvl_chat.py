# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
import types
from typing import Any, Dict, List, Optional, Tuple, Union

import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, LlamaForCausalLM, Qwen2ForCausalLM
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from .modeling_qwen2_fastv import Qwen2ForCausalLMPruning

logger = logging.get_logger(__name__)

# Variable names used to hold the cache at generation time
ALL_CACHE_NAMES = [
    "past_key_values",  # default
    "cache_params",  # mamba-based models
    "state",  # rwkv
    "mems",  # xlnet
    "past_buckets_states",  # reformer
]


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLChatModelPruning(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'Qwen2DecoderLayer']

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True,
                 pruning_method='vanilla', visual_token_num=-1):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                # Use custom pruning model for fastv method
                if pruning_method == 'fastv':
                    if visual_token_num <= 0:
                        raise ValueError(f"visual_token_num must be positive for fastv pruning, got {visual_token_num}")

                    print("\n" + "="*80)
                    print("[InternVL3] 使用 FastV 剪枝方法初始化 Language Model")
                    print("="*80)
                    print(f"  Pruning method: fastv")
                    print(f"  Target AVERAGE visual tokens: {visual_token_num} (across all 28 layers)")
                    print(f"  Per-patch token count: {self.num_image_token}")
                    print(f"  Note: FastV will prune at layer 2:")
                    print(f"        - Layers 0-1: Keep all visual tokens (original count)")
                    print(f"        - Layers 2-27: Keep R tokens (calculated from target average)")
                    print(f"        - Formula: avg = (original*2 + R*26) / 28 = {visual_token_num}")
                    print("="*80 + "\n")

                    self.language_model = Qwen2ForCausalLMPruning(
                        config.llm_config,
                        pruning_method='fastv',
                        visual_token_num=visual_token_num,
                        num_image_token=self.num_image_token
                    )
                elif pruning_method == 'tops_variant3':
                    if visual_token_num <= 0:
                        raise ValueError(f"visual_token_num must be positive for tops_variant3, got {visual_token_num}")

                    print("\n" + "="*80)
                    print("[InternVL3] 使用 TOPS-Variant3 两阶段剪枝方法初始化 Language Model")
                    print("="*80)
                    print(f"  Pruning method: tops_variant3")
                    print(f"  Stage 1 (THCP): Vision encoder → target×2 tokens")
                    print(f"  Stage 2 (Progressive): Decoder progressive pruning")
                    print(f"  Target FINAL visual tokens: {visual_token_num} (average across layers)")

                    from .modeling_qwen2_tops_variant3 import TOPSVariant3Qwen2ForCausalLM, TOPS_VARIANT3_SCHEDULE
                    schedule = TOPS_VARIANT3_SCHEDULE.get("internvl3_8b", {}).get(visual_token_num, [])
                    print(f"  Stage 2 schedule: {schedule}")
                    print("="*80 + "\n")

                    # Initialize Stage 1 parameters
                    self.stage1_keep_tokens = max(1, visual_token_num * 2)

                    self.language_model = TOPSVariant3Qwen2ForCausalLM(
                        config.llm_config,
                        visual_token_num=visual_token_num,
                        num_image_token=self.num_image_token
                    )
                elif pruning_method in ['divprune', 'cdpruner', 'dppruner', 'visionzip', 'visionzip+']:
                    # These methods prune in extract_feature(), use vanilla LLM
                    if visual_token_num <= 0:
                        raise ValueError(f"visual_token_num must be positive for {pruning_method}, got {visual_token_num}")

                    print("\n" + "="*80)
                    print(f"[InternVL3] Using {pruning_method.upper()} Pruning Method")
                    print("="*80)
                    print(f"  Pruning location: Vision encoder output (extract_feature)")
                    print(f"  Target visual tokens: {visual_token_num}")
                    print(f"  Method description:")
                    if pruning_method == 'divprune':
                        print(f"    - Diversity-based pruning using cosine distance")
                        print(f"    - Selects most diverse visual tokens greedily")
                    elif pruning_method == 'cdpruner':
                        print(f"    - Content-Diversity Pruner (DPP-based)")
                        print(f"    - Combines text relevance + visual diversity")
                        print(f"    - Kernel: relevance * similarity * relevance")
                    elif pruning_method == 'dppruner':
                        print(f"    - Pure DPP-based pruning")
                        print(f"    - Maximizes determinantal point process over visual tokens")
                    elif pruning_method == 'visionzip':
                        print(f"    - VisionZip: Dominant + Contextual token compression")
                        print(f"    - Total target tokens: {visual_token_num}")
                        print(f"    - Ratio per patch: Dominant 27/32, Contextual 5/32")
                    elif pruning_method == 'visionzip+':
                        print(f"    - VisionZip+: Enhanced with fixed 5% contextual tokens")
                        print(f"    - Total target tokens: {visual_token_num}")
                        print(f"    - Contextual: 5% of original per patch")
                    print("="*80 + "\n")

                    self.language_model = Qwen2ForCausalLM(config.llm_config)
                else:
                    self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

        self.pruning_method = pruning_method
        self.visual_token_num = visual_token_num

        # Process position_ids with pruning
        def _update_model_kwargs_for_generation_pruning(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            num_new_tokens: int = 1,
        ) -> Dict[str, Any]:
            # update past_key_values keeping its naming used in model code
            for possible_cache_name in ALL_CACHE_NAMES:
                if possible_cache_name in outputs:
                    # TODO (joao): remove output/input mismatch when these old models (xlnet, reformer) are deprecated
                    if possible_cache_name in ("past_buckets_states", "mems"):
                        cache_name = "past_key_values"
                    else:
                        cache_name = possible_cache_name
                    model_kwargs[cache_name] = getattr(outputs, possible_cache_name)
                    break

            # update token_type_ids with last value
            if "token_type_ids" in model_kwargs:
                token_type_ids = model_kwargs["token_type_ids"]
                model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

            if not is_encoder_decoder:
                # update attention mask
                if "attention_mask" in model_kwargs:
                    attention_mask = model_kwargs["attention_mask"]
                    model_kwargs["attention_mask"] = torch.cat(
                        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                    )
            if "position_ids" in model_kwargs:
                position_ids = model_kwargs["position_ids"]
                if position_ids is not None:
                    model_kwargs["position_ids"] = torch.cat(
                        [position_ids, position_ids[:, -1].unsqueeze(-1) + 1], dim=-1
                    )
                else:
                    model_kwargs.pop("position_ids")
            else:
                # update decoder attention mask
                if "decoder_attention_mask" in model_kwargs:
                    decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                    model_kwargs["decoder_attention_mask"] = torch.cat(
                        [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                        dim=-1,
                    )

            if model_kwargs.get("use_cache", True):
                model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
            else:
                past_positions = model_kwargs.pop("cache_position")
                new_positions = torch.arange(
                    past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
                ).to(past_positions.device)
                model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
            return model_kwargs

        self.language_model._update_model_kwargs_for_generation = types.MethodType(_update_model_kwargs_for_generation_pruning, self.language_model)

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        # Extract visual features with pruning
        vit_embeds, selected_masks = self.extract_feature(pixel_values)

        # DEBUG: Log before/after pruning
        original_tokens = vit_embeds.shape[0] * vit_embeds.shape[1] if len(vit_embeds.shape) > 2 else vit_embeds.shape[0]

        # Apply pruning mask if any tokens were pruned
        if selected_masks.sum() < selected_masks.numel():
            # Filter vit_embeds using selected_masks: keep only selected tokens
            # selected_masks shape: [B, N] where B=num_patches, N=tokens_per_patch
            # We need to flatten and filter
            vit_embeds = vit_embeds[selected_masks]  # [total_selected_tokens, D]

            print(f"[DEBUG] Stage 1 pruning applied: {original_tokens} → {vit_embeds.shape[0]} tokens")
        else:
            print(f"[DEBUG] No Stage 1 pruning: {original_tokens} tokens kept")

        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = min(selected.sum(), vit_embeds.size(0))
            input_embeds[selected][:n_token] = input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        # TOPS-Variant3: Set system prompt length for pruning
        if self.pruning_method == 'tops_variant3' and N > 1:  # Prefill mode
            system_prompt_length = int(selected.max(dim=-1).indices) if selected.sum() > 0 else 41
            if hasattr(self.language_model, 'set_system_prompt_length'):
                self.language_model.set_system_prompt_length(system_prompt_length)

        # FastV: Set system prompt length for pruning
        if self.pruning_method == 'fastv' and N > 1:  # Prefill mode
            # Calculate where visual tokens start (right after the last image context token in the first occurrence)
            # selected is a boolean mask where image context tokens are located
            if selected.sum() > 0:
                # Find the first image context token position
                first_img_token = int(selected.nonzero()[0].item())
                # System prompt is everything before the first image token
                system_prompt_length = first_img_token

                # Typical InternVL3 system prompt is ~41 tokens
                # Warn if significantly different (might indicate tokenization issues)
                if system_prompt_length < 30 or system_prompt_length > 100:
                    logger.warning(f"[FastV] Unexpected system_prompt_length: {system_prompt_length} "
                                 f"(expected ~41). This might affect pruning performance.")
            else:
                system_prompt_length = 41  # Default for InternVL3

            # Set system_prompt_length on the language model
            if hasattr(self.language_model, 'set_system_prompt_length'):
                self.language_model.set_system_prompt_length(system_prompt_length)
                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    print(f"[FastV] System prompt length: {system_prompt_length}")
                    print(f"[FastV] Visual tokens: {self.num_image_token}")
                    print(f"[FastV] Target tokens: {self.visual_token_num}")

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values, image_num_patches=None, text_embeds=None):
        vit_attns = None
        if self.pruning_method in ['fastervlm', 'prumerge', 'prumerge+', 'visionzip', 'visionzip+', 'vispruner']:
            if self.select_layer == -1:
                vit_outputs = self.vision_model(
                    pixel_values=pixel_values,
                    output_attentions=True,
                    output_hidden_states=False,
                    return_dict=True,
                    select_layer=self.select_layer)
                vit_embeds = vit_outputs.last_hidden_state
                vit_attns = vit_outputs.attentions[self.select_layer]
            else:
                vit_outputs = self.vision_model(
                    pixel_values=pixel_values,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                    select_layer=self.select_layer)
                vit_embeds = vit_outputs.hidden_states[self.select_layer]
                vit_attns = vit_outputs.attentions[self.select_layer]
        else:
            if self.select_layer == -1:
                vit_embeds = self.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=False,
                    return_dict=True).last_hidden_state
            else:
                vit_embeds = self.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]
        if vit_attns is not None:
            vit_attns = vit_attns[:, :, 0, 1:]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)

        if vit_attns is not None:
            vit_attns = vit_attns.reshape(-1, h, w).unsqueeze(-1)
            vit_attns = self.pixel_shuffle(vit_attns, scale_factor=self.downsample_ratio).mean(dim=-1)
            vit_attns = vit_attns.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[1])

        # Generate selection mask
        B, N, D = vit_embeds.shape
        device = vit_embeds.device
        selected_masks = torch.ones(B, N, dtype=torch.bool, device=device)

        if self.pruning_method == 'thumbnail':
            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
        elif self.pruning_method == 'random':
            random_scores = torch.rand(B, N).to(device)
            selected_indices = random_scores.topk(self.visual_token_num, dim=-1).indices

            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            selected_masks.scatter_(1, selected_indices, True)
            selected_masks[-1, :] = True
        elif self.pruning_method == "fastervlm":
            image_attention = vit_attns.mean(dim=1)
            patch_attention, thumb_attention = image_attention[:-1], image_attention[-1:]
            patch_token_num = self.visual_token_num // 2 // patch_attention.shape[0]
            patch_indices = patch_attention.topk(patch_token_num, dim=-1).indices
            thumb_token_num = self.visual_token_num // 2
            thumb_indices = thumb_attention.topk(thumb_token_num, dim=-1).indices
            # selected_indices = image_attention.topk(self.visual_token_num, dim=-1).indices

            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            selected_masks[:-1].scatter_(1, patch_indices, True)
            selected_masks[-1:].scatter_(1, thumb_indices, True)
            # selected_masks.scatter_(1, selected_indices, True)
        elif self.pruning_method == "trim":
            _text_embeds_list = [text_embed[None].expand(num_patch, -1, -1) for text_embed, num_patch in zip(text_embeds, image_num_patches)]
            text_embeds = torch.cat(_text_embeds_list, dim=0)

            image_normalized = (vit_embeds / vit_embeds.norm(dim=-1, keepdim=True))
            text_normalized = (text_embeds / text_embeds.norm(dim=-1, keepdim=True))
            query_relevance = torch.matmul(image_normalized, text_normalized.transpose(1, 2)).max(dim=-1).values
            selected_indices = query_relevance.topk(self.visual_token_num, dim=-1).indices

            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            selected_masks.scatter_(1, selected_indices, True)
        elif self.pruning_method == 'visionzip' or self.pruning_method == 'visionzip+':
            # VisionZip: Select dominant tokens + merge contextual tokens
            # B = num_patches (e.g., 5 patches), N = tokens_per_patch (e.g., 256)
            # visual_token_num = target TOTAL tokens across ALL patches (e.g., 256)
            if vit_attns is None:
                raise ValueError("VisionZip requires attention weights, but they are not available")

            # Calculate tokens per patch: total_target / num_patches
            tokens_per_patch = self.visual_token_num // B
            if tokens_per_patch == 0:
                tokens_per_patch = 1  # At least 1 token per patch

            if self.pruning_method == 'visionzip':
                # For each patch: dominant = tokens_per_patch * 27/32
                dominant_per_patch = int(tokens_per_patch * 27 / 32)
                contextual_per_patch = tokens_per_patch - dominant_per_patch
            elif self.pruning_method == 'visionzip+':
                # For each patch: contextual = 5% of original N
                contextual_per_patch = int(N * 5 / 100)
                dominant_per_patch = tokens_per_patch - contextual_per_patch

            # Select dominant tokens based on attention
            image_attention = vit_attns.mean(dim=1)  # [B, N]
            dominant_indices = torch.empty((B, dominant_per_patch), dtype=torch.long, device=device)
            for b in range(B):
                dominant_indices[b] = image_attention[b].topk(dominant_per_patch, dim=-1).indices

            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            selected_masks.scatter_(1, dominant_indices, True)

            # For each image in batch, process contextual tokens
            for b in range(B):
                # Get normalized embeddings for non-dominant tokens
                filtered_embeds = vit_embeds[b, ~selected_masks[b]]  # [remaining, D]
                if filtered_embeds.shape[0] == 0:
                    continue

                normalized_embeds = filtered_embeds / filtered_embeds.norm(dim=-1, keepdim=True)

                # Select contextual tokens evenly spaced
                target_step = max(1, normalized_embeds.shape[0] // contextual_per_patch)
                target_indices = torch.arange(0, normalized_embeds.shape[0], target_step, device=device)[:contextual_per_patch]
                target_embeds = normalized_embeds[target_indices, :]

                # Compute similarity for source tokens to targets
                source_mask = ~torch.isin(torch.arange(normalized_embeds.shape[0], device=device), target_indices)
                source_embeds = normalized_embeds[source_mask]

                if source_embeds.shape[0] > 0 and target_embeds.shape[0] > 0:
                    similarity = torch.matmul(source_embeds, target_embeds.t())
                    # Use vit_embeds dtype to avoid dtype mismatch (BFloat16)
                    assign_one_hot = vit_embeds.new_zeros(source_embeds.shape[0], contextual_per_patch)
                    assign_one_hot.scatter_(1, similarity.argmax(dim=-1).unsqueeze(-1), 1)
                    counts = assign_one_hot.sum(dim=0).clamp(min=1).unsqueeze(-1)

                    # Get actual indices in the filtered space
                    contextual_positions = (~selected_masks[b]).nonzero(as_tuple=True)[0]
                    target_positions = contextual_positions[target_indices]
                    source_positions = contextual_positions[source_mask]

                    # Mark target tokens as selected
                    selected_masks[b, target_positions] = True

                    # Aggregate source embeddings into targets
                    aggregated_embeds = torch.matmul(assign_one_hot.t(), source_embeds) / counts
                    vit_embeds[b, target_positions] += aggregated_embeds
        elif self.pruning_method == 'divprune':
            # Calculate tokens per patch: total_target / num_patches (same as VisionZip)
            tokens_per_patch = self.visual_token_num // B
            if tokens_per_patch == 0:
                tokens_per_patch = 1  # At least 1 token per patch

            image_normalized = (vit_embeds / vit_embeds.norm(dim=-1, keepdim=True))
            cosine_matrix = 1.0 - torch.matmul(image_normalized, image_normalized.transpose(1, 2))

            select_idx = torch.empty((B, tokens_per_patch), dtype=torch.long, device=device)

            # Process each patch independently to avoid indexing issues
            for b in range(B):
                for i in range(tokens_per_patch):
                    if i == 0:
                        m2 = cosine_matrix[b]  # [N, N]
                        scores = torch.topk(m2, 2, dim=0, largest=False).values[1]  # [N]
                    else:
                        # Select rows corresponding to already selected tokens
                        m2 = cosine_matrix[b, select_idx[b, :i], :]  # [i, N]
                        scores = torch.min(m2, dim=0).values  # [N]

                    select_idx[b, i] = torch.argmax(scores)

            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            selected_masks.scatter_(1, select_idx, True)
        elif self.pruning_method == 'dppruner':
            # Calculate tokens per patch: total_target / num_patches (same as VisionZip)
            tokens_per_patch = self.visual_token_num // B
            if tokens_per_patch == 0:
                tokens_per_patch = 1  # At least 1 token per patch

            image_normalized = vit_embeds / vit_embeds.norm(dim=-1, keepdim=True)
            kernel_matrix = torch.matmul(image_normalized, image_normalized.transpose(1, 2))

            cis = torch.zeros((tokens_per_patch, B, N), device=device)
            di2s = torch.diagonal(kernel_matrix, dim1=1, dim2=2).clone()
            select_idx = torch.empty((tokens_per_patch, B), dtype=torch.long, device=device)
            for i in range(tokens_per_patch):
                j = torch.argmax(di2s, dim=-1)
                select_idx[i] = j

                eis = (kernel_matrix[torch.arange(B), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B), j], cis[:i])) \
                    / torch.sqrt(di2s[torch.arange(B), j]).unsqueeze(-1)
                cis[i, :, :] = eis
                di2s -= torch.square(eis)
                di2s[torch.arange(B), j] = -float('inf')

            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            selected_masks.scatter_(1, select_idx.t(), True)
        elif self.pruning_method == 'cdpruner':
            # Calculate tokens per patch: total_target / num_patches (same as VisionZip)
            tokens_per_patch = self.visual_token_num // B
            if tokens_per_patch == 0:
                tokens_per_patch = 1  # At least 1 token per patch

            # CDPruner: Content-Diversity Pruner
            # Combines text relevance and visual diversity using DPP kernel
            image_normalized = vit_embeds / vit_embeds.norm(dim=-1, keepdim=True)

            # Compute text relevance
            if text_embeds is not None and len(text_embeds) > 0:
                _text_embeds_list = [text_embed[None].expand(num_patch, -1, -1)
                                    for text_embed, num_patch in zip(text_embeds, image_num_patches)]
                text_embeds_expanded = torch.cat(_text_embeds_list, dim=0)  # [B, text_len, D]
                text_normalized = text_embeds_expanded / text_embeds_expanded.norm(dim=-1, keepdim=True)

                # Relevance: max similarity to any text token
                relevance = torch.matmul(image_normalized, text_normalized.transpose(1, 2)).max(dim=-1).values  # [B, N]
                # Normalize relevance to [0, 1] per batch
                relevance_min = relevance.min(dim=1, keepdim=True).values
                relevance_max = relevance.max(dim=1, keepdim=True).values
                relevance = (relevance - relevance_min) / (relevance_max - relevance_min + 1e-8)
            else:
                # Fallback: uniform relevance
                relevance = torch.ones(B, N, device=device)

            # Compute visual similarity (for DPP)
            similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2))  # [B, N, N]

            # DPP kernel: relevance * similarity * relevance
            kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)  # [B, N, N]

            # DPP sampling per batch
            select_idx = torch.empty((B, tokens_per_patch), dtype=torch.long, device=device)
            for b in range(B):
                cis = torch.zeros((tokens_per_patch, N), device=device)
                di2s = torch.diag(kernel[b])  # [N]
                for i in range(tokens_per_patch):
                    j = torch.argmax(di2s)
                    select_idx[b, i] = j

                    # Update for next iteration
                    if i < tokens_per_patch - 1:  # No need to update on last iteration
                        eis = (kernel[b, j, :] - (cis[:i, j] @ cis[:i, :])) / torch.sqrt(di2s[j] + 1e-8)
                        cis[i, :] = eis
                        di2s -= torch.square(eis)
                        di2s[j] = -float('inf')

            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            selected_masks.scatter_(1, select_idx, True)
        elif self.pruning_method == 'tops_variant3':
            # TOPS-Variant3 Stage 1: THCP (Text-Concept Hierarchical Pruning)
            total_tokens = B * N  # Total visual tokens across all patches
            print(f"\n{'='*80}")
            print(f"[TOPS-Variant3 Stage 1] THCP Text-Concept Coverage")
            print(f"  Visual tokens: {N} per patch ({B} patches) = {total_tokens} total")
            print(f"  Target (final): {self.visual_token_num}")
            print(f"  Stage 1 target: {self.stage1_keep_tokens} (adaptive: target × 2)")

            # Stage 1: Keep target × 2 tokens GLOBALLY (not per-patch)
            stage1_keep_num = min(self.stage1_keep_tokens, total_tokens)

            # Check if text embeddings are available
            if text_embeds is not None and len(text_embeds) > 0:
                # Expand text embeddings for each image patch
                _text_embeds_list = [text_embed[None].expand(num_patch, -1, -1)
                                    for text_embed, num_patch in zip(text_embeds, image_num_patches)]
                text_embeds_expanded = torch.cat(_text_embeds_list, dim=0)  # [B, M, D]
                M = text_embeds_expanded.shape[1]
                print(f"  Text tokens: {M}")
                print(f"  Mode: {'Coverage' if M > 1 else 'Relevance'}")

                # Flatten to treat all patches globally: [B*N, D]
                image_flat = vit_embeds.reshape(-1, D)  # [B*N, D]
                image_normalized = image_flat / (image_flat.norm(dim=-1, keepdim=True) + 1e-8)

                # Use first text embedding for all patches (same question for all)
                text_normalized = text_embeds_expanded[0] / (text_embeds_expanded[0].norm(dim=-1, keepdim=True) + 1e-8)  # [M, D]

                # Compute response matrix globally [B*N, M]
                response_matrix = torch.matmul(image_normalized, text_normalized.t())  # [B*N, M]

                if M > 1:
                    # Coverage mode: average relevance across text tokens
                    text_relevance = response_matrix.mean(dim=-1)  # [B*N]
                else:
                    # Relevance mode: single text token
                    text_relevance = response_matrix.squeeze(-1)  # [B*N]

                # Compute visual similarity globally [B*N, B*N]
                visual_similarity = torch.matmul(image_normalized, image_normalized.t())

                # Global greedy selection
                selected_indices = []
                available_mask = torch.ones(total_tokens, dtype=torch.bool, device=device)
                relevance_weight = 0.5
                diversity_weight = 0.5

                for step in range(stage1_keep_num):
                    if not available_mask.any():
                        break

                    scores = text_relevance.clone()
                    if step > 0 and len(selected_indices) > 0:
                        selected_tensor = torch.tensor(selected_indices, device=device, dtype=torch.long)
                        max_similarity = visual_similarity[:, selected_tensor].max(dim=1).values
                        diversity_scores = 1 - max_similarity
                        scores = relevance_weight * scores + diversity_weight * diversity_scores

                    scores[~available_mask] = -float('inf')
                    selected_idx = scores.argmax().item()
                    selected_indices.append(selected_idx)
                    available_mask[selected_idx] = False

                # Convert global indices to per-patch mask
                selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
                if len(selected_indices) > 0:
                    selected_tensor = torch.tensor(selected_indices, device=device, dtype=torch.long)
                    # Convert flat indices to (batch, position) indices
                    batch_indices = selected_tensor // N
                    pos_indices = selected_tensor % N
                    selected_masks[batch_indices, pos_indices] = True

                print(f"  Selected {len(selected_indices)} tokens total ({len(selected_indices)/B:.1f} per patch average)")
            else:
                # No text embeddings: use attention-based selection as fallback
                print(f"  No text embeddings available, using attention-based fallback")
                if vit_attns is not None:
                    # Use attention from vision encoder - global selection
                    image_attention = vit_attns.mean(dim=1).reshape(-1)  # [B*N]
                    selected_indices_flat = image_attention.topk(stage1_keep_num, dim=-1).indices

                    # Convert to per-patch mask
                    selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
                    batch_indices = selected_indices_flat // N
                    pos_indices = selected_indices_flat % N
                    selected_masks[batch_indices, pos_indices] = True
                else:
                    # Random fallback - global selection
                    random_scores = torch.rand(total_tokens).to(device)
                    selected_indices_flat = random_scores.topk(stage1_keep_num, dim=-1).indices

                    # Convert to per-patch mask
                    selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
                    batch_indices = selected_indices_flat // N
                    pos_indices = selected_indices_flat % N
                    selected_masks[batch_indices, pos_indices] = True

                print(f"  Selected {stage1_keep_num} tokens total ({stage1_keep_num/B:.1f} per patch average)")

            print(f"{'='*80}\n")

        return vit_embeds, selected_masks

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            num_patches_list: Optional[List[int]] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

            reverse_input_ids = input_ids.flip(-1)
            reverse_mask = (reverse_input_ids == self.img_context_token_id)
            text_instruction_length = int(reverse_mask.max(dim=-1).indices[0])

            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds, selected_masks = self.extract_feature(
                    pixel_values,
                    image_num_patches=num_patches_list,
                    text_embeds=input_embeds[:, -text_instruction_length:, :],
                )
            # For FastV or TOPS-Variant3 pruning, use average visual tokens from language model
            if self.pruning_method in ['fastv', 'tops_variant3'] and hasattr(self.language_model, 'model') and hasattr(self.language_model.model, 'visual_token_num'):
                lm_model = self.language_model.model
                self.visual_token_len = lm_model.visual_token_num
            else:
                # For vision encoder pruning methods, use total selected tokens
                # selected_masks shape: [num_patches, tokens_per_patch]
                # We want to display TOTAL tokens across all patches, not per-patch average
                self.visual_token_len = selected_masks.sum().item()

            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)

            # Prepare default position ids
            position_ids = torch.arange(input_embeds.shape[1], device=input_embeds.device).unsqueeze(0)

            # Select visual tokens using `selected_masks`
            if selected_masks.sum() < selected_masks.numel():
                system_prompt_length = int(selected.max(dim=-1).indices)
                selected_masks = selected_masks.reshape(-1)
                visual_token_length = selected_masks.shape[0]
                visual_token_indices = selected_masks.nonzero().squeeze(-1)
                retain_token_indices = torch.cat((
                    torch.arange(system_prompt_length, device=input_embeds.device),
                    visual_token_indices + system_prompt_length,
                    torch.arange(system_prompt_length+visual_token_length, input_embeds.shape[1], device=input_embeds.device),
                ))

                input_embeds = input_embeds[:, retain_token_indices, :]
                attention_mask = attention_mask[:, retain_token_indices]
                position_ids = position_ids[:, retain_token_indices]

                # Update system_prompt_length for TOPS-Variant3 Stage 2
                if self.pruning_method == 'tops_variant3':
                    self.system_prompt_length = system_prompt_length
            else:
                position_ids = None
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            position_ids = None

        # FastV: Set system prompt length for pruning in generate mode
        if self.pruning_method == 'fastv' and pixel_values is not None and input_embeds.shape[1] > 1:
            # Calculate where visual tokens start
            if 'selected' in locals() and selected.sum() > 0:
                # Find the first image context token position
                first_img_token = int(selected.nonzero()[0].item())
                system_prompt_length = first_img_token

                # Typical InternVL3 system prompt is ~41 tokens
                if system_prompt_length < 30 or system_prompt_length > 100:
                    logger.warning(f"[FastV Generate] Unexpected system_prompt_length: {system_prompt_length} "
                                 f"(expected ~41).")
            else:
                system_prompt_length = 41  # Default for InternVL3

            # Set system_prompt_length on the language model
            if hasattr(self.language_model, 'set_system_prompt_length'):
                self.language_model.set_system_prompt_length(system_prompt_length)

        # TOPS-Variant3: Set system prompt length
        if self.pruning_method == 'tops_variant3' and input_embeds.shape[1] > 1:
            if 'selected' in locals() and selected.sum() > 0:
                first_img_token = int(selected.nonzero()[0].item())
                system_prompt_length = first_img_token
            else:
                system_prompt_length = 41
            if hasattr(self.language_model, 'set_system_prompt_length'):
                self.language_model.set_system_prompt_length(system_prompt_length)

        generate_inputs = dict(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )
        if position_ids is not None:
            generate_inputs["position_ids"] = position_ids

        outputs = self.language_model.generate(**generate_inputs)

        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
