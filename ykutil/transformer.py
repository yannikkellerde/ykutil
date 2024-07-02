import json
import os
from functools import lru_cache
from operator import itemgetter
from typing import Dict, List, Optional, Sequence

import torch
from peft import AutoPeftModelForCausalLM
from tokenizers import Encoding
from torch.nn.functional import log_softmax
from tqdm import trange
from transformer_heads import load_lora_with_heads
from transformer_heads.output import HeadedModelOutput
from transformer_heads.util.helpers import get_model_params
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
)

from .python import list_squeeze


@lru_cache()
def find_tokens_with_str(tokenizer: PreTrainedTokenizer, string: str):
    return tuple(
        id
        for id in trange(len(tokenizer), desc="Find tks")
        if string in tokenizer.decode(id)
    )


def load_tk_with_pad_tk(model_path):
    tk = AutoTokenizer.from_pretrained(model_path)
    if tk.pad_token_id is None:
        tk.pad_token = tk.eos_token
    return tk


def batch_tokenization(
    tk: PreTrainedTokenizer, texts: List[str], batch_size: int, **tk_args
):
    """Use when batch of texts too large to be tokenized at once"""
    out = {"input_ids": [], "attention_mask": []}
    for s in trange(0, len(texts), batch_size, desc="Batch tokenization"):
        stuff = tk(texts[s : s + batch_size], **tk_args)
        out["input_ids"].extend(stuff["input_ids"])
        out["attention_mask"].extend(stuff["attention_mask"])
    return out


def tokenize_instances(
    tokenizer: PreTrainedTokenizer, instances: Sequence[Dict[str, str]]
) -> Dict[str, torch.Tensor]:
    texts = [
        f"{tokenizer.bos_token}{example['text']}{tokenizer.eos_token}"
        for example in instances
    ]

    print("tokenizing all text")

    tokenized_texts = batch_tokenization(
        tk=tokenizer,
        texts=texts,
        batch_size=500,
        add_special_tokens=False,
    )
    return tokenized_texts


class TokenStoppingCriteria(StoppingCriteria):
    def __init__(self, delimiter_token):
        self.delimiter_token = (
            [delimiter_token] if type(delimiter_token) == int else delimiter_token
        )
        super().__init__()

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids[0][-1] in self.delimiter_token:
            return True  # Stop generation
        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self


def flat_encode(tokenizer, inputs, add_special_tokens=False):
    if isinstance(inputs, list):
        inputs = list_squeeze(inputs)
    if isinstance(inputs, str):
        inputs = [inputs]
    if len(inputs) == 0:
        return []
    return sum(
        [tokenizer.encode(x, add_special_tokens=add_special_tokens) for x in inputs], []
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


@torch.inference_mode()
def compute_seq_log_probability(
    model: PreTrainedModel,
    pre_seq_tokens: List[int],
    post_seq_tokens: List[int],
) -> float:
    inputs = torch.tensor(
        pre_seq_tokens + post_seq_tokens, device=model.device
    ).unsqueeze(0)

    output = model(inputs)
    if isinstance(output, HeadedModelOutput):
        logits = output.preds_by_head["lm_head"][
            0, -len(post_seq_tokens) - 1 : -1
        ].cpu()
    else:
        logits = output.logits[0, -len(post_seq_tokens) - 1 : -1].cpu()

    logprobs = log_softmax(logits, dim=-1)

    return float(
        torch.exp(
            torch.sum(
                logprobs.gather(1, torch.tensor(post_seq_tokens).unsqueeze(1)).squeeze()
            )
        )
    )


def get_token_begins(encoding: Encoding | BatchEncoding):
    get_start = lambda x: list(map(itemgetter(0), x))
    if isinstance(encoding, Encoding):
        return get_start(encoding.offsets)
    else:
        return [get_start(x.offsets) for x in encoding.encodings]
