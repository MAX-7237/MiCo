#    Copyright 2023 Haotian Liu
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


import re
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaLlamaForCausalLM
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    use_flash_attn=False,
    **kwargs,
):
    """Load the supported LLaVA backbones with the MiCo or unpruned decoder."""
    if model_base is not None:
        raise ValueError("Use a complete LLaVA checkpoint through model_path")
    method = kwargs.pop("pruning_method", "vanilla")
    tokens = kwargs.pop("visual_token_num", None)
    use_mico = kwargs.pop("use_mico_coverage_llm", False)
    mico_config = kwargs.pop("mico_coverage_llm_config", None)
    match = re.fullmatch(r"mico_([1-9][0-9]*)", method)
    if method not in ("vanilla", "mico_s1") and match is None:
        raise ValueError("Unsupported MiCo configuration: " + str(method))
    if match and not use_mico:
        use_mico = True
        mico_config = {
            "K": int(match.group(1)),
            "T": tokens,
            "T_avg": tokens,
            "dynamic_R": True,
            "mico": True,
        }
    options = {"device_map": device_map}
    if device != "cuda":
        options["device_map"] = {"": device}
    if load_8bit:
        options["load_in_8bit"] = True
    elif load_4bit:
        options["load_in_4bit"] = True
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        options["torch_dtype"] = torch.float16
    if use_flash_attn:
        options["attn_implementation"] = "flash_attention_2"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        pruning_method=method,
        visual_token_num=tokens,
        use_mico_coverage_llm=use_mico,
        mico_coverage_llm_config=mico_config,
        **options,
    )
    if getattr(model.config, "mm_use_im_patch_token", True):
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if getattr(model.config, "mm_use_im_start_end", False):
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != "auto":
        vision_tower.to(device=device_map, dtype=torch.float16)
    context_len = getattr(
        model.config, "max_sequence_length", getattr(model.config, "max_position_embeddings", 2048)
    )
    return tokenizer, model, vision_tower.image_processor, context_len
