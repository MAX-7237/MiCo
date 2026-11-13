#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from .modeling_llama_mico_coverage_llm import MiCoCoverageLLMLlamaModel
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaLlamaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class MiCoCoverageLLMLlavaLlamaModel(LlavaMetaModel, MiCoCoverageLLMLlamaModel):
    config_class = LlavaLlamaConfig

    def __init__(self, config: LlamaConfig, mico_coverage_llm_config: dict):
        super(MiCoCoverageLLMLlavaLlamaModel, self).__init__(
            config, mico_coverage_llm_config=mico_coverage_llm_config
        )


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaLlamaConfig

    def __init__(
        self,
        config,
        pruning_method=None,
        visual_token_num=None,
        use_mico_coverage_llm=False,
        mico_coverage_llm_config=None,
        **kwargs,
    ):
        pruning_method = pruning_method or "vanilla"
        if pruning_method not in ("vanilla", "mico_s1", "mico_7", "mico_8", "mico_14"):
            raise ValueError("Unsupported MiCo configuration")
        super(LlamaForCausalLM, self).__init__(config)
        if use_mico_coverage_llm:
            print(f"Use Attn-Cov-LLM: {mico_coverage_llm_config}")
            self.model = MiCoCoverageLLMLlavaLlamaModel(config, mico_coverage_llm_config)
        else:
            self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.pruning_method = pruning_method
        self.visual_token_num = visual_token_num
        self.start_latency = False
        self.phase = "prefill"
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.prefill_latency = 0.0
        self.decode_latency = 0.0
        self.track_flops = False
        self.layer_visual_tokens = []
        self.total_flops = 0.0
        self.flops_count = 0
        self.post_init()

    def get_pruning_method(self):
        return self.pruning_method

    def get_visual_token_num(self):
        return self.visual_token_num

    def get_model(self):
        return self.model

    def calculate_flops(self):
        """
        Calculate FLOPs based on recorded visual token numbers per layer.

        Formula per layer (from cal_flops.py):
        FLOPs = 8 * n * d^2 + 4 * n^2 * d + 6 * n * d * m
        where:
        - n: number of visual tokens in this layer
        - d: hidden dimension (config.hidden_size)
        - m: FFN intermediate dimension (config.intermediate_size)

        Returns:
            float: Total FLOPs in TFLOPs (10^12 FLOPs)
        """
        if not self.layer_visual_tokens:
            return 0.0
        d = self.config.hidden_size
        m = self.config.intermediate_size
        flops = 0.0
        for n in self.layer_visual_tokens:
            flops += 8 * n * d * d + 4 * n * n * d + 6 * n * d * m
        flops_tflops = flops / 1000000000000.0
        return flops_tflops

    def record_layer_tokens(self, layer_tokens):
        """
        Record visual token numbers for each layer.

        Args:
            layer_tokens: List of token numbers per layer
        """
        self.layer_visual_tokens = layer_tokens
        if self.track_flops:
            flops = self.calculate_flops()
            self.total_flops += flops
            self.flops_count += 1

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
            input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = (
                self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    image_sizes,
                )
            )
        if hasattr(self.model, "img_seq") and images is not None and (inputs_embeds is not None):
            self.model.img_seq = 576
            self.model.key_set = None
            self.model.pre_prompt_length_list = [35]
            self.model.token_length_list = [inputs_embeds.shape[1]]
        if self.start_latency:
            if self.phase == "prefill" and input_ids is None:
                self.start_event.record()
            elif self.phase == "decode" and input_ids is not None and (input_ids.shape[1] == 1):
                self.start_event.record()
        output = super().forward(
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
        if self.start_latency:
            if self.phase == "prefill" and input_ids is None:
                self.end_event.record()
                torch.cuda.synchronize()
                self.prefill_latency += self.start_event.elapsed_time(self.end_event)
                self.phase = "decode"
            elif self.phase == "decode" and input_ids is not None and (input_ids.shape[1] == 1):
                self.end_event.record()
                torch.cuda.synchronize()
                self.decode_latency += self.start_event.elapsed_time(self.end_event)
                self.phase = "prefill"
        return output

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
        if images is not None:
            inputs, position_ids, attention_mask, _, inputs_embeds, _, visual_token_num = (
                self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    image_sizes=image_sizes,
                    texts=texts,
                )
            )
            if hasattr(self.model, "update_runtime_visual_length") and visual_token_num:
                r_keep = self.model.update_runtime_visual_length(int(visual_token_num))
                if getattr(self.model, "dynamic_R", False) and (
                    not getattr(self, "_mico_coverage_dyn_r_logged", False)
                ):
                    print(
                        f"[mico_coverage_llm dynamic_R] N_enter={visual_token_num} T_avg={getattr(self.model, 'T_avg', None)} K={getattr(self.model, 'K', None)} L={len(self.model.layers)} -> R={r_keep}",
                        flush=True,
                    )
                    self._mico_coverage_dyn_r_logged = True
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            visual_token_num = 0
        output = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if hasattr(self.model, "layer_visual_tokens") and self.model.layer_visual_tokens:
            self.record_layer_tokens(self.model.layer_visual_tokens)
        return (output, visual_token_num)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        texts = kwargs.pop("texts", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if texts is not None:
            inputs["texts"] = texts
        return inputs


AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModelForCausalLM.register(LlavaLlamaConfig, LlavaLlamaForCausalLM)
