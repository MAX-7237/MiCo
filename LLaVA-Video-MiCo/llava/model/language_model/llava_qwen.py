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


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.model.language_model.modeling_qwen2_fastv import FastVQwen2Model
from llava.model.language_model.modeling_qwen2_pdrop import PDropQwen2Model
from llava.model.language_model.modeling_qwen2_sparsevlm import SparseQwen2Model
from llava.model.language_model.modeling_qwen2_starv3 import STARV3Qwen2Model
from llava.model.language_model.modeling_qwen2_tops import TOPSQwen2Model
from llava.model.language_model.modeling_qwen2_attn_cov import AttnCovQwen2Model
from llava.utils import rank0_print

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class FastVLlavaQwenModel(LlavaMetaModel, FastVQwen2Model):
    # Alter LlavaQwenModel to FastVLlavaQwenModel
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config, fastv_config: dict):
        super(FastVLlavaQwenModel, self).__init__(config, fastv_config=fastv_config)


class PDropLlavaQwenModel(LlavaMetaModel, PDropQwen2Model):
    # Alter LlavaQwenModel to PDropLlavaQwenModel
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config, pdrop_config: dict):
        super(PDropLlavaQwenModel, self).__init__(config, pdrop_config=pdrop_config)


class SparseLlavaQwenModel(LlavaMetaModel, SparseQwen2Model):
    # Alter LlavaQwenModel to SparseVLlavaQwenModel
    config_class = LlavaQwenConfig
    def __init__(self, config: Qwen2Config, sparsevlm_config: dict):
        super(SparseLlavaQwenModel, self).__init__(config, sparsevlm_config=sparsevlm_config)


class STARV3LlavaQwenModel(LlavaMetaModel, STARV3Qwen2Model):
    # Alter LlavaQwenModel to STARV3LlavaQwenModel
    config_class = LlavaQwenConfig
    def __init__(self, config: Qwen2Config, starv3_config: dict):
        super(STARV3LlavaQwenModel, self).__init__(config, starv3_config=starv3_config)
        
class TOPSLlavaQwenModel(LlavaMetaModel, TOPSQwen2Model):
    # Alter LlavaQwenModel to STARV3LlavaQwenModel
    config_class = LlavaQwenConfig
    def __init__(self, config: Qwen2Config, tops_config: dict):
        super(TOPSLlavaQwenModel, self).__init__(config, tops_config=tops_config)


class AttnCovLlavaQwenModel(LlavaMetaModel, AttnCovQwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config, attn_cov_config):
        super().__init__(config, attn_cov_config=attn_cov_config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config, pruning_method=None, visual_token_num=None,
                 use_fastv=False, fastv_config=None,
                 use_sparsevlm=False, sparsevlm_config=None,
                 use_pdrop=False, pdrop_config=None,
                 use_starv3=False, starv3_config=None, 
                 use_tops=False, tops_config=None, **kwargs):
        # Debug: Show what parameters __init__ received
        rank0_print("="*80)
        rank0_print(f"[llava_qwen.__init__] 初始化开始")
        rank0_print(f"[llava_qwen.__init__] pruning_method = {pruning_method}")
        rank0_print(f"[llava_qwen.__init__] visual_token_num = {visual_token_num}")
        rank0_print(f"[llava_qwen.__init__] kwargs.keys() = {list(kwargs.keys())}")
        rank0_print("="*80)

        super(Qwen2ForCausalLM, self).__init__(config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        if pruning_method in ('attn_cov_proj_llm_12', 'attn_H_cov_proj_llm_12'):
            import os
            if any((use_fastv, use_pdrop, use_sparsevlm, use_starv3, use_tops)):
                raise ValueError('attn_cov cannot be combined with another LLM pruner')
            if visual_token_num is None:
                raise ValueError('attn_cov requires an explicit visual_token_num target')
            attn_cov_config = {'T': int(visual_token_num), 'K': int(os.environ.get('ATTN_COV_K', '12')),
                               'attn_h': pruning_method == 'attn_H_cov_proj_llm_12'}
            self.model = AttnCovLlavaQwenModel(config, attn_cov_config)
        elif use_fastv:
            print(f"Use FastV: {fastv_config}")
            self.model = FastVLlavaQwenModel(config, fastv_config)
        elif use_pdrop:
            print(f"Use PDrop: {pdrop_config}")
            self.model = PDropLlavaQwenModel(config, pdrop_config)
        elif use_sparsevlm:
            print(f"Use SparseVLM: {sparsevlm_config}")
            self.model = SparseLlavaQwenModel(config, sparsevlm_config)
        elif use_starv3:
            print(f"Use STAR-V3 Stage2: {starv3_config}")
            self.model = STARV3LlavaQwenModel(config, starv3_config)
        elif use_tops:
            print(f"Use TOPS Stage2: {tops_config}")
            self.model = TOPSLlavaQwenModel(config, tops_config)
        else:
            self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Visual Token Pruning config
        self.pruning_method = pruning_method
        self.visual_token_num = visual_token_num

        # Debug: Confirm parameters are stored
        rank0_print("="*80)
        rank0_print(f"[llava_qwen.__init__] 剪枝参数已存储")
        rank0_print(f"[llava_qwen.__init__] self.pruning_method = {self.pruning_method}")
        rank0_print(f"[llava_qwen.__init__] self.visual_token_num = {self.visual_token_num}")
        rank0_print("="*80)

        # Initialize weights and apply final processing
        self.post_init()
    
    # Visual Token Pruning
    def get_pruning_method(self):
        return self.pruning_method

    # Visual Token Pruning
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
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, _) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

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
            return logits, labels

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
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                visual_token_num
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                modalities,
                image_sizes=image_sizes
            )
            rank0_print(f"[qwen-debug] generate() received visual_token_num={visual_token_num} from prepare_inputs_labels_for_multimodal")
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            visual_token_num = 0
            rank0_print(f"[qwen-debug] generate() no images, visual_token_num={visual_token_num}")

        rank0_print(f"[qwen-debug] generate() returning visual_token_num={visual_token_num}")
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        ), visual_token_num

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
