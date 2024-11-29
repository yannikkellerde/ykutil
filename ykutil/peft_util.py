import json
import os
from typing import Optional

import torch
from peft import AutoPeftModelForCausalLM
from transformer_heads import load_lora_with_heads
from transformer_heads.util.helpers import get_model_params
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def load_maybe_peft_model_tokenizer(
    model_path,
    device_map="auto",
    quantization_config: Optional[BitsAndBytesConfig] = None,
    flash_attn="flash_attention_2",
    only_inference=True,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    extra_args = {}
    if quantization_config is not None:
        extra_args["quantization_config"] = quantization_config
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        with open(os.path.join(model_path, "adapter_config.json")) as f:
            base_model_name = json.load(f)["base_model_name_or_path"]
        if os.path.isfile(os.path.join(model_path, "head_configs.json")):
            model_params = get_model_params(base_model_name)
            model = load_lora_with_heads(
                base_model_class=model_params["model_class"],
                path=model_path,
                quantization_config=quantization_config,
                only_inference=only_inference,
                attn_implementation=flash_attn,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
            )
        else:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                attn_implementation=flash_attn,
                torch_dtype=torch.bfloat16,
                **extra_args,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            attn_implementation=flash_attn,
            torch_dtype=torch.bfloat16,
            **extra_args,
        )
        base_model_name = model_path

    # For some reason, the instruct model got the wrong eos_token here
    # https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/commit/a8977699a3d0820e80129fb3c93c20fbd9972c41
    # So we load the base model instead
    base_model_name = base_model_name.replace("-Instruct", "")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
