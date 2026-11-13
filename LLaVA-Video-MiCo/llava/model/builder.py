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


import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaQwenForCausalLM, LlavaQwenConfig
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    torch_dtype="float16",
    attn_implementation="flash_attention_2",
    customized_config=None,
    overwrite_config=None,
    **kwargs,
):
    """Load the Qwen2-based LLaVA-Video model used by MiCo."""
    if model_base is not None:
        raise ValueError("Use a complete LLaVA-Video checkpoint")
    method = kwargs.pop("pruning_method", "vanilla")
    tokens = kwargs.pop("visual_token_num", None)
    if method not in ("vanilla", "mico"):
        raise ValueError("Unsupported MiCo video configuration")
    options = {"device_map": device_map}
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
        if torch_dtype not in ("float16", "bfloat16"):
            raise ValueError("torch_dtype must be float16 or bfloat16")
        options["torch_dtype"] = getattr(torch, torch_dtype)
    if overwrite_config is not None or customized_config is not None:
        config = customized_config or LlavaQwenConfig.from_pretrained(model_path)
        for key, value in (overwrite_config or {}).items():
            setattr(config, key, value)
        options["config"] = config
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlavaQwenForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        pruning_method=method,
        visual_token_num=tokens,
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
        vision_tower.to(device="cuda", dtype=torch.float16)
    context = getattr(
        model.config,
        "max_sequence_length",
        getattr(
            model.config,
            "max_position_embeddings",
            getattr(model.config, "tokenizer_model_max_length", 2048),
        ),
    )
    return tokenizer, model, vision_tower.image_processor, context
