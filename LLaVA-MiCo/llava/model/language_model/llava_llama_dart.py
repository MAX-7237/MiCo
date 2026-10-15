"""
DART LLaVA-Llama Integration Layer  (官方 DART LLM 层剪枝忠实移植)

Source: 官方 DART  https://github.com/ZichenWen1/DART  (EMNLP'25)
移植自 /path/to/local-assets

机制: 剪枝完全发生在 LLM forward 内部(第K层), 视觉侧 encode_images 不剪枝(传全 576).
      DART 类在第K层用 key_states 的 L1 norm 选 image/text pivot, 再对每个 pivot 选
      最不相似(负余弦)的 image token 去重保留. 与 VScan 同为"LLM 层剪枝", 故照 VScan 集成.

参数(LLaVA-1.5): K=2, image_token_start_index=35, image_token_length=576,
      pivot_image_token=4, pivot_text_token=4, reduction_ratio=0.778,
      retain_token_num_for_llava_next=预算(text_length被设时实际控制保留数),
      max_num_trunction=预算(备用). 每次 forward 前设 config.text_length=input_ids.shape[-1]-1.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .llava_llama import LlavaLlamaConfig
from .modeling_llama_dart import DARTLlamaModel


class DARTLlavaConfig(LlavaLlamaConfig):
    """Configuration class for DART LLaVA model."""
    model_type = "dart_llava_llama"


class DARTLlavaLlamaModel(LlavaMetaModel, DARTLlamaModel):
    """LLaVA meta model + 官方 DART LLM 模型(第K层剪枝)."""
    config_class = DARTLlavaConfig

    def __init__(self, config):
        super(DARTLlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM_DART(LlamaForCausalLM, LlavaMetaForCausalLM):
    """
    DART LLaVA for Causal LM. 视觉侧不剪枝, 剪枝在 self.model (DART) 的第K层 forward.
    forward/generate 走标准 LLaVA 流程; generate 里每次设 config.text_length.
    """
    config_class = DARTLlavaConfig

    def __init__(self, config, dart_config: dict = None, **kwargs):
        super(LlamaForCausalLM, self).__init__(config)

        # 把 DART 运行时参数写进 config, 供 DART.forward / get_retained_image_token 读取.
        if dart_config is None:
            dart_config = {}
        budget = dart_config.get("T", 64)          # 视觉 token 预算 (64/32)
        config.DART_config = {
            "K": dart_config.get("K", 2),
            "image_token_start_index": dart_config.get("image_token_start_index", 35),
            "image_token_length": dart_config.get("image_token_length", 576),
            "max_num_trunction": budget,
            "reduction_ratio": dart_config.get("reduction_ratio", 0.778),
            "pivot_image_token": dart_config.get("pivot_image_token", 4),
            "pivot_text_token": dart_config.get("pivot_text_token", 4),
            # text_length 被设时, 实际保留数由此参数控制 => 设成预算
            "retain_token_num_for_llava_next": budget,
            # use_coverage=True: dart_coverage 变体, diversity 换成 SCOPE coverage 边际增益
            "use_coverage": bool(dart_config.get("use_coverage", False)),
        }
        config.text_length = None  # 每次 forward/generate 前更新

        self.model = DARTLlavaLlamaModel(config)

        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.pruning_method = "dart_llm"
        self.visual_token_num = budget
        self.dart_config = config.DART_config

        # FLOPS/兼容占位 (不含 cuda.Event, 避免 8-bit 加载时 deepcopy 失败)
        self.track_flops = False
        self.layer_visual_tokens = []
        self.total_flops = 0.0
        self.flops_count = 0

        self.post_init()

    def get_model(self):
        return self.model

    def get_pruning_method(self):
        return self.pruning_method

    def get_visual_token_num(self):
        return self.visual_token_num

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
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

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        texts: Optional[str] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        # 官方 DART: 每次生成前设 text_length = 原始 input_ids 长度 - 1 (image 占1个placeholder)
        if inputs is not None:
            self.config.text_length = inputs.shape[-1] - 1

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                visual_token_num,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                texts=texts,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            visual_token_num = 0

        output = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return output, visual_token_num

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        kwargs.pop("texts", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("dart_llava_llama", DARTLlavaConfig)
AutoModelForCausalLM.register(DARTLlavaConfig, LlavaLlamaForCausalLM_DART)

DARTLlavaLlamaForCausalLM = LlavaLlamaForCausalLM_DART
