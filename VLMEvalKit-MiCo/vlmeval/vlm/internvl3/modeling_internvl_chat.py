# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
import types
import warnings
import types
import warnings
import types
import warnings
import types
import warnings
import types
import warnings
import types
import warnings
import types
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

logger = logging.get_logger(__name__)


def _is_mico(method: str) -> bool:
    prefix = "mico_"
    if not isinstance(method, str) or not method.startswith(prefix):
        return False
    body = method[len(prefix) :]
    if "_s1m" in body:
        body = body.rsplit("_s1m", 1)[0]
    return body.isdigit()


def _mico_s1_mult(method: str) -> int:
    """S1 budget multiple of T, resolved method-name-first.

    `<...>_llm_<K>_s1m<M>` pins it per arm; otherwise fall back to the
    MICO_STAGE1_MULT env knob. Default 2 preserves the published
    `mico_K` behaviour exactly.
    """
    from vlmeval.vlm.mico_coverage_select import mico_s1_mult as _env_mult

    if isinstance(method, str):
        head, _, tail = method.rpartition("_s1m")
        if head and tail.isdigit() and (int(tail) >= 1):
            return int(tail)
    return _env_mult()


def _is_mico_s1(method: str) -> bool:
    return method == "mico_s1" or _is_mico(method)


ALL_CACHE_NAMES = ["past_key_values", "cache_params", "state", "mems", "past_buckets_states"]


def version_cmp(v1, v2, op="eq"):
    import operator
    from packaging import version

    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLChatModelPruning(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = "pixel_values"
    base_model_prefix = "language_model"
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["InternVisionModel", "LlamaDecoderLayer", "Qwen2DecoderLayer"]

    def __init__(
        self,
        config: InternVLChatConfig,
        vision_model=None,
        language_model=None,
        use_flash_attn=True,
        pruning_method="vanilla",
        visual_token_num=-1,
    ):
        super().__init__(config)
        assert version_cmp(transformers.__version__, "4.37.0", "ge")
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * config.downsample_ratio**2)
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = "flash_attention_2" if use_flash_attn else "eager"
        logger.info(f"num_image_token: {self.num_image_token}")
        logger.info(f"ps_version: {self.ps_version}")
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        elif config.llm_config.architectures[0] == "LlamaForCausalLM":
            self.language_model = LlamaForCausalLM(config.llm_config)
        elif config.llm_config.architectures[0] == "Qwen2ForCausalLM":
            if pruning_method in ("mico_2", "mico_4", "mico_6"):
                if visual_token_num <= 0:
                    raise ValueError(
                        f"visual_token_num must be positive for {pruning_method}, got {visual_token_num}"
                    )
                from .modeling_qwen2_mico import MiCoQwen2ForCausalLM
                from vlmeval.vlm.mico_coverage_select import mico_s2_keep

                n_layers = config.llm_config.num_hidden_layers
                t = int(visual_token_num)
                mico = pruning_method in ("mico_2", "mico_4", "mico_6")
                k_layer = int(pruning_method.rsplit("_s1m", 1)[0].rsplit("_", 1)[1])
                if not 1 <= k_layer < n_layers:
                    raise ValueError(f"H-ViT S2 K must be in [1, {n_layers - 1}], got {k_layer}")
                s1_m = _mico_s1_mult(pruning_method)
                s2 = {k_layer: mico_s2_keep(t, n_layers, k_layer, s1_m)}
                space = "projector"
                print(
                    f"\n[InternVL3] {pruning_method}: S1 {space} raw_cls×coverage {s1_m}T, S2={s2} (MiCo first layer, R=(T*L-K*{s1_m}T)/(L-K), L={n_layers})"
                )
                self.stage1_keep_tokens = max(1, t * s1_m)
                self._s1_mult = s1_m
                self.language_model = MiCoQwen2ForCausalLM(
                    config.llm_config,
                    mico_config={
                        "T": t,
                        "pruning_layers": s2,
                        "cover_s2_inside": True,
                        "mico": mico,
                        "dynamic_r": mico,
                    },
                )
            elif pruning_method in ("mico_s1",):
                if visual_token_num <= 0:
                    raise ValueError(
                        f"visual_token_num must be positive for {pruning_method}, got {visual_token_num}"
                    )
                print("\n" + "=" * 80)
                print(f"[InternVL3] Using {pruning_method.upper()} Pruning Method")
                print("=" * 80)
                print(f"  Pruning location: Vision encoder output (extract_feature)")
                print(f"  Target visual tokens: {visual_token_num}")
                print(f"  Method description:")
                print("=" * 80 + "\n")
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                self.language_model = Qwen2ForCausalLM(config.llm_config)
        else:
            raise NotImplementedError(f"{config.llm_config.architectures[0]} is not implemented.")
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message
        self.pruning_method = pruning_method
        self.visual_token_num = visual_token_num

        def _update_model_kwargs_for_generation_pruning(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            num_new_tokens: int = 1,
        ) -> Dict[str, Any]:
            for possible_cache_name in ALL_CACHE_NAMES:
                if possible_cache_name in outputs:
                    if possible_cache_name in ("past_buckets_states", "mems"):
                        cache_name = "past_key_values"
                    else:
                        cache_name = possible_cache_name
                    model_kwargs[cache_name] = getattr(outputs, possible_cache_name)
                    break
            if "token_type_ids" in model_kwargs:
                token_type_ids = model_kwargs["token_type_ids"]
                model_kwargs["token_type_ids"] = torch.cat(
                    [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
                )
            if not is_encoder_decoder:
                if "attention_mask" in model_kwargs:
                    attention_mask = model_kwargs["attention_mask"]
                    model_kwargs["attention_mask"] = torch.cat(
                        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                        dim=-1,
                    )
            if "position_ids" in model_kwargs:
                position_ids = model_kwargs["position_ids"]
                if position_ids is not None:
                    model_kwargs["position_ids"] = torch.cat(
                        [position_ids, position_ids[:, -1].unsqueeze(-1) + 1], dim=-1
                    )
                else:
                    model_kwargs.pop("position_ids")
            elif "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [
                        decoder_attention_mask,
                        decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1)),
                    ],
                    dim=-1,
                )
            if model_kwargs.get("use_cache", True):
                model_kwargs["cache_position"] = (
                    model_kwargs["cache_position"][-1:] + num_new_tokens
                )
            else:
                past_positions = model_kwargs.pop("cache_position")
                new_positions = torch.arange(
                    past_positions[-1] + 1,
                    past_positions[-1] + num_new_tokens + 1,
                    dtype=past_positions.dtype,
                ).to(past_positions.device)
                model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
            return model_kwargs

        self.language_model._update_model_kwargs_for_generation = types.MethodType(
            _update_model_kwargs_for_generation_pruning, self.language_model
        )

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
        vit_embeds, selected_masks = self.extract_feature(pixel_values)
        original_tokens = (
            vit_embeds.shape[0] * vit_embeds.shape[1]
            if len(vit_embeds.shape) > 2
            else vit_embeds.shape[0]
        )
        if selected_masks.sum() < selected_masks.numel():
            vit_embeds = vit_embeds[selected_masks]
            print(
                f"[DEBUG] Stage 1 pruning applied: {original_tokens} → {vit_embeds.shape[0]} tokens"
            )
        else:
            print(f"[DEBUG] No Stage 1 pruning: {original_tokens} tokens kept")
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(
                f"dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}"
            )
        input_ids = input_ids.reshape(B * N)
        selected = input_ids == self.img_context_token_id
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(
                f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, vit_embeds.shape={vit_embeds.shape}"
            )
            n_token = min(selected.sum(), vit_embeds.size(0))
            input_embeds[selected][:n_token] = (
                input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]
            )
        input_embeds = input_embeds.reshape(B, N, C)
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
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
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
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor))
        )
        if self.ps_version == "v1":
            warnings.warn(
                "In ps_version 'v1', the height and width have not been swapped back, which results in a transposed image."
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values, image_num_patches=None, text_embeds=None):
        vit_attns = None
        if self.pruning_method in ("mico_s1", "mico_2", "mico_4", "mico_6"):
            if self.select_layer == -1:
                vit_outputs = self.vision_model(
                    pixel_values=pixel_values,
                    output_attentions=True,
                    output_hidden_states=False,
                    return_dict=True,
                    select_layer=self.select_layer,
                )
                vit_embeds = vit_outputs.last_hidden_state
                vit_attns = vit_outputs.attentions[self.select_layer]
            else:
                vit_outputs = self.vision_model(
                    pixel_values=pixel_values,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                    select_layer=self.select_layer,
                )
                vit_embeds = vit_outputs.hidden_states[self.select_layer]
                vit_attns = vit_outputs.attentions[self.select_layer]
        elif self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]
        if vit_attns is not None:
            vit_attns = vit_attns[:, :, 0, 1:]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_pre_proj = vit_embeds
        vit_embeds = self.mlp1(vit_embeds)
        if vit_attns is not None:
            vit_attns = vit_attns.reshape(-1, h, w).unsqueeze(-1)
            vit_attns = self.pixel_shuffle(vit_attns, scale_factor=self.downsample_ratio).mean(
                dim=-1
            )
            vit_attns = vit_attns.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[1])
        B, N, D = vit_embeds.shape
        device = vit_embeds.device
        selected_masks = torch.ones(B, N, dtype=torch.bool, device=device)
        if self.pruning_method in ("mico_s1", "mico_2", "mico_4", "mico_6"):
            from vlmeval.vlm.mico_coverage_select import inner_weighted_raw_cls_select

            keep_2t = self.pruning_method in ("mico_2", "mico_4", "mico_6")
            use_vit = False
            _s1m = int(getattr(self, "_s1_mult", 2) or 2) if keep_2t else 1
            keep_total = min(self.visual_token_num * _s1m, B * N)
            base_per_patch, _rem = divmod(int(keep_total), B)
            per_patch_keep = [min(base_per_patch + (1 if b < _rem else 0), N) for b in range(B)]
            kt = min(max(per_patch_keep) if per_patch_keep else 1, N)
            cov = vit_pre_proj if use_vit else vit_embeds
            if vit_attns is not None:
                cls_attn_all = vit_attns.mean(dim=1) if vit_attns.dim() == 3 else vit_attns
            else:
                print(
                    f"[{self.pruning_method}] vit_attns is None, fallback to {('pre-proj' if use_vit else 'proj')} L2"
                )
                cls_attn_all = cov.norm(dim=-1)
            masks = []
            for b in range(B):
                masks.append(
                    inner_weighted_raw_cls_select(
                        cov[b],
                        cls_attn_all[b],
                        per_patch_keep[b],
                        mico=self.pruning_method in ("mico_s1", "mico_2", "mico_4", "mico_6"),
                    )
                )
            selected_masks = torch.stack(masks, dim=0)
            s2 = "S2=MiCo first LLM layer avg T" if keep_2t else "no LLM prune"
            _per = f"{min(per_patch_keep)}-{max(per_patch_keep)}" if per_patch_keep else "0"
            print(
                f"[{self.pruning_method}] S1 {('ViT' if use_vit else 'proj')} raw_cls{('*|h|' if self.pruning_method in ('mico_s1', 'mico_2', 'mico_4', 'mico_6') else '')}*coverage: B={B} N={N} keep/patch={_per} kept={int(selected_masks.sum().item())} ({(f'{_s1m}T' if keep_2t else 'T')}={keep_total}, {s2})"
            )
        return (vit_embeds, selected_masks)

    def batch_chat(
        self,
        tokenizer,
        pixel_values,
        questions,
        generation_config,
        num_patches_list=None,
        history=None,
        return_history=False,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
        image_counts=None,
    ):
        if history is not None or return_history:
            print("Now multi-turn chat is not supported in batch_chat.")
            raise NotImplementedError
        if image_counts is not None:
            num_patches_list = image_counts
            print("Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.")
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")
        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and "<image>" not in question:
                question = "<image>\n" + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)
            queries.append(query)
        tokenizer.padding_side = "left"
        model_inputs = tokenizer(queries, return_tensors="pt", padding=True)
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        history=None,
        return_history=False,
        num_patches_list=None,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
    ):
        if history is None and pixel_values is not None and ("<image>" not in question):
            question = "<image>\n" + question
        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        history = [] if history is None else history
        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")
        for num_patches in num_patches_list:
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)
        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return (response, history)
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, "")
            query_to_print = query_to_print.replace(f"{IMG_START_TOKEN}{IMG_END_TOKEN}", "<image>")
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
            reverse_mask = reverse_input_ids == self.img_context_token_id
            text_instruction_length = int(reverse_mask.max(dim=-1).indices[0])
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds, selected_masks = self.extract_feature(
                    pixel_values,
                    image_num_patches=num_patches_list,
                    text_embeds=input_embeds[:, -text_instruction_length:, :],
                )
            self.visual_token_len = selected_masks.sum().item()
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            input_ids = input_ids.reshape(B * N)
            selected = input_ids == self.img_context_token_id
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            input_embeds = input_embeds.reshape(B, N, C)
            position_ids = torch.arange(
                input_embeds.shape[1], device=input_embeds.device
            ).unsqueeze(0)
            if selected_masks.sum() < selected_masks.numel():
                system_prompt_length = int(selected.max(dim=-1).indices)
                selected_masks = selected_masks.reshape(-1)
                visual_token_length = selected_masks.shape[0]
                visual_token_indices = selected_masks.nonzero().squeeze(-1)
                retain_token_indices = torch.cat(
                    (
                        torch.arange(system_prompt_length, device=input_embeds.device),
                        visual_token_indices + system_prompt_length,
                        torch.arange(
                            system_prompt_length + visual_token_length,
                            input_embeds.shape[1],
                            device=input_embeds.device,
                        ),
                    )
                )
                input_embeds = input_embeds[:, retain_token_indices, :]
                attention_mask = attention_mask[:, retain_token_indices]
                position_ids = position_ids[:, retain_token_indices]
            else:
                position_ids = None
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            position_ids = None
        if (
            self.pruning_method in ("mico_2", "mico_4", "mico_6")
            and pixel_values is not None
            and (input_embeds.shape[1] > 1)
        ):
            if "selected" in locals() and selected.sum() > 0:
                first_img_token = int(selected.nonzero()[0].item())
                system_prompt_length = first_img_token
            else:
                system_prompt_length = 41
            if hasattr(self.language_model, "set_system_prompt_length"):
                self.language_model.set_system_prompt_length(system_prompt_length)
            if hasattr(self.language_model, "model"):
                self.language_model.model.visual_token_length = int(selected_masks.sum().item())
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
