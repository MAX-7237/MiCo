#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.model.language_model.modeling_qwen2_mico_coverage import MiCoCoverageQwen2Model
from llava.utils import rank0_print


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class MiCoCoverageLlavaQwenModel(LlavaMetaModel, MiCoCoverageQwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config, mico_coverage_config):
        super().__init__(config, mico_coverage_config=mico_coverage_config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config, pruning_method=None, visual_token_num=None, **kwargs):
        rank0_print("=" * 80)
        rank0_print(f"[llava_qwen.__init__] 初始化开始")
        rank0_print(f"[llava_qwen.__init__] pruning_method = {pruning_method}")
        rank0_print(f"[llava_qwen.__init__] visual_token_num = {visual_token_num}")
        rank0_print(f"[llava_qwen.__init__] kwargs.keys() = {list(kwargs.keys())}")
        rank0_print("=" * 80)
        super(Qwen2ForCausalLM, self).__init__(config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None
        if pruning_method in ("mico",):
            import os

            if any((False, False, False, False, False)):
                raise ValueError("mico_coverage cannot be combined with another LLM pruner")
            if visual_token_num is None:
                raise ValueError("mico_coverage requires an explicit visual_token_num target")
            mico_coverage_config = {
                "T": int(visual_token_num),
                "K": int(os.environ.get("MICO_COVERAGE_K", "6")),
                "mico": pruning_method in ("mico",),
            }
            self.model = MiCoCoverageLlavaQwenModel(config, mico_coverage_config)
        else:
            self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.pruning_method = pruning_method
        self.visual_token_num = visual_token_num
        rank0_print("=" * 80)
        rank0_print(f"[llava_qwen.__init__] 剪枝参数已存储")
        rank0_print(f"[llava_qwen.__init__] self.pruning_method = {self.pruning_method}")
        rank0_print(f"[llava_qwen.__init__] self.visual_token_num = {self.visual_token_num}")
        rank0_print("=" * 80)
        self.post_init()

    def get_pruning_method(self):
        return self.pruning_method

    def get_visual_token_num(self):
        return self.visual_token_num

    def get_model(self):
        return self.model

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
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, _ = (
                self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    modalities,
                    image_sizes,
                )
            )
        if dpo_forward:
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
            logits = self.lm_head(hidden_states)
            return (logits, labels)
        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            inputs, position_ids, attention_mask, _, inputs_embeds, _, visual_token_num = (
                self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    modalities,
                    image_sizes=image_sizes,
                )
            )
            rank0_print(
                f"[qwen-debug] generate() received visual_token_num={visual_token_num} from prepare_inputs_labels_for_multimodal"
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            visual_token_num = 0
            rank0_print(f"[qwen-debug] generate() no images, visual_token_num={visual_token_num}")
        rank0_print(f"[qwen-debug] generate() returning visual_token_num={visual_token_num}")
        return (
            super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            ),
            visual_token_num,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
