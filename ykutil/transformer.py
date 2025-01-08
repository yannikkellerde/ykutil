import functools
import itertools
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from operator import itemgetter
from typing import Any, Dict, List, Optional, Sequence

import torch
from tokenizers import Encoding
from torch.nn.functional import log_softmax
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
)
from transformers.generation.utils import GenerateOutput

from ykutil.python import list_squeeze

try:
    from transformer_heads.output import HeadedModelOutput

    th_available = True
except ImportError:
    th_available = False


def generate_different_sequences(
    model: PreTrainedModel,
    context: torch.Tensor,
    sequence_bias_add: int,
    sequence_bias_decay: float,
    generation_args: GenerationConfig,
    num_generations: int,
):
    gen_sequences = []
    sequence_bias = defaultdict(float)
    for _ in range(num_generations):
        gen = model.generate(
            context.unsqueeze(0),
            generation_config=generation_args,
            sequence_bias=sequence_bias or None,
        )
        if isinstance(gen, GenerateOutput):
            gen = gen.sequences
        gen = gen[0][context.shape[0] :]
        gen_sequences.append(gen)
        for key in sequence_bias:
            sequence_bias[key] *= sequence_bias_decay
        if sequence_bias_add != 0:
            for tok in gen:
                sequence_bias[tok.item()] += sequence_bias_add
    return gen_sequences


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


have_warned = False


def obtain_offsets(
    be: BatchEncoding, str_lengths: Optional[List[int]] = None
) -> list[list[tuple[int, int]]]:
    """
    >>> from transformers import AutoTokenizer
    >>> obtain_offsets(AutoTokenizer.from_pretrained("gpt2")("Hello, my dog is cute."))
    [[(0, 5), (5, 6), (6, 9), (9, 13), (13, 16), (16, 21), (21, 22)]]
    >>> tk = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    >>> tk.pad_token = tk.eos_token
    >>> obtain_offsets(tk("Hello, my dog is cute."), [22])
    Broken offsets detected. This is a known bug for llama3. Will try to fix it.
    [[(0, 0), (0, 5), (5, 6), (6, 9), (9, 13), (13, 16), (16, 21), (21, 22)]]
    >>> obtain_offsets(tk(["Hello, my dog is cute.", "Big Tree"], return_tensors="pt", padding=True), [22, 8])[1]
    [(0, 0), (0, 3), (3, 8), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    """
    global have_warned
    offsets = [x.offsets for x in be.encodings]
    if not all(
        functools.reduce(
            lambda x, y: ((x[1] == y[0] or y[0] == y[1] == 0) and x[0], y[1]),
            li,
            (True, 0),
        )[0]
        for li in offsets
    ):
        if not have_warned:
            print(
                "Broken offsets detected. This is a known bug for llama3. Will try to fix it."
            )
            have_warned = True
        assert str_lengths is not None, "Cannot fix offsets without str_lengths"
        for o, sl in zip(offsets, str_lengths):
            beyond_start = False
            for i in range(len(o)):
                if o[i] != (0, 0):
                    beyond_start = True
                if i == len(o) - 1 or (beyond_start and o[i + 1] == (0, 0)):
                    if o[i] != (0, 0):
                        o[i] = (o[i][0], sl)
                else:
                    o[i] = (o[i][0], o[i + 1][0])
    return offsets


def batch_tokenization(
    tk: PreTrainedTokenizer,
    texts: List[str],
    batch_size: int,
    include_offsets: bool = False,
    **tk_args,
):
    """Use when batch of texts too large to be tokenized at once"""
    out = {"input_ids": [], "attention_mask": []}
    if include_offsets:
        out["offsets"] = []
    for s in trange(0, len(texts), batch_size, desc="Batch tokenization"):
        stuff = tk(texts[s : s + batch_size], **tk_args)
        out["input_ids"].extend(stuff["input_ids"])
        out["attention_mask"].extend(stuff["attention_mask"])
        if include_offsets:
            out["offsets"].extend(
                obtain_offsets(stuff, [len(x) for x in texts[s : s + batch_size]])
            )
    return out


def transform_with_offsets(
    offsets: list[tuple[int, int]],
    spans: list[tuple[int, int]],
    include_left=True,
    include_right=True,
):
    """
    >>> transform_with_offsets([(0, 3), (3, 6), (6, 9)], [(1, 2), (4, 8)])
    [(0, 0), (1, 2)]
    >>> transform_with_offsets([(0, 3), (3, 6), (6, 9)], [(1, 4), (5, 8)], include_left=False)
    [(1, 1), (2, 2)]
    >>> transform_with_offsets([(0, 3), (3, 6), (6, 9)], [(1, 7)], include_left=False, include_right=False)
    [(1, 1)]
    >>> transform_with_offsets([(0, 3), (3, 6), (6, 9)], [(0, 8)], include_left=False, include_right=False)
    [(0, 2)]
    """
    out_offsets = []
    offset_index = 0
    for i, num in enumerate(itertools.chain.from_iterable(spans)):
        for j in range(offset_index, len(offsets)):
            if offsets[j][0] <= num < offsets[j][1]:
                if not i % 2 and not include_left and num != offsets[j][0]:
                    out_offsets.append(j + 1)
                elif i % 2 and not include_right and num != offsets[j][1] - 1:
                    out_offsets.append(j - 1)
                else:
                    out_offsets.append(j)
                offset_index = j
                break
        else:
            assert i % 2, f"Span out of bounds, {(i,offsets[j],num)}"
            out_offsets.append(j + 1)

    return list(zip(out_offsets[::2], out_offsets[1::2]))


def regex_tokens_using_offsets(
    offsets: list[tuple[int, int]],
    text: str,
    regex: str | re.Pattern,
    include_left=True,
    include_right=True,
):
    """
    >>> regex_tokens_using_offsets([(0, 3), (3, 6), (6, 9)], "abc def ghi", "d")
    [(1, 1)]
    >>> regex_tokens_using_offsets([(0, 5), (5, 11), (11, 17)], "I am I and great", r"I \\w+",)
    [(0, 0), (1, 1)]
    """
    if isinstance(regex, str):
        regex = re.compile(regex)

    spans = [x.span() for x in regex.finditer(text)]
    return transform_with_offsets(
        offsets, spans, include_left=include_left, include_right=include_right
    )


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


def get_token_begins(encoding: Encoding | BatchEncoding):
    get_start = lambda x: list(map(itemgetter(0), x))
    if isinstance(encoding, Encoding):
        return get_start(encoding.offsets)
    else:
        return [get_start(x.offsets) for x in encoding.encodings]


@dataclass
class DataCollatorWithPadding:
    """
    A data collator that pads sequences to the same length.

    Attributes:
        feature_name_to_padding_value (dict[str, int]): A dictionary mapping feature names to their padding values.

    Methods:
        __call__(features: List[Dict[str, Any]]) -> Dict[str, Any]: Pad the sequences in the features to the same length.
    """

    feature_name_to_padding_value: dict[str, int | float]
    feature_name_to_new_data_type: dict[str, torch.dtype] = field(default_factory=dict)
    padding_side: str = "right"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Pad the sequences in the features to the same length.

        Args:
            features (List[Dict[str, Any]]): A list of features, where each feature is a dictionary mapping feature names to sequences.

        Returns:
            Dict[str, Any]: A dictionary mapping feature names to padded sequences.
        """
        assert len(features) > 0, "features must not be empty."
        batch = dict()
        for key, value in self.feature_name_to_padding_value.items():
            if key in features[0]:
                if self.padding_side == "right":
                    if key in self.feature_name_to_new_data_type:
                        to_pad = [
                            feature[key]
                            .clone()
                            .detach()
                            .type(self.feature_name_to_new_data_type[key])
                            for feature in features
                        ]
                    else:
                        to_pad = [feature[key].clone().detach() for feature in features]
                    batch[key] = pad_sequence(
                        to_pad,
                        batch_first=True,
                        padding_value=value,
                    )
                elif self.padding_side == "left":
                    if key in self.feature_name_to_new_data_type:
                        to_pad = [
                            feature[key]
                            .clone()
                            .detach()
                            .flip(dims=[0])
                            .type(self.feature_name_to_new_data_type[key])
                            for feature in features
                        ]
                    else:
                        to_pad = [
                            feature[key].clone().detach().flip(dims=[0])
                            for feature in features
                        ]
                    batch[key] = pad_sequence(
                        to_pad,
                        batch_first=True,
                        padding_value=value,
                    ).flip(dims=[1])
                else:
                    raise ValueError(
                        f"padding_side must be either 'right' or 'left', but got {self.padding_side}."
                    )

        for key in features[0].keys():
            if key not in self.feature_name_to_padding_value:
                if isinstance(features[0][key], torch.Tensor):
                    try:
                        batch[key] = torch.stack(
                            [feature[key].clone().detach() for feature in features]
                        )
                    except RuntimeError as e:
                        print("Failure for key", key)
                        print(e)
                elif isinstance(features[0][key], list) and type(
                    features[0][key][0]
                ) in (
                    int,
                    float,
                ):
                    try:
                        batch[key] = torch.stack([feature[key] for feature in features])
                    except RuntimeError as e:
                        print("Failure for key", key)
                        print(e)
                else:
                    batch[key] = [feature[key] for feature in features]
                if key in self.feature_name_to_new_data_type:
                    batch[key] = batch[key].to(self.feature_name_to_new_data_type[key])

        return batch


def dict_from_chat_template(chat_template_str: str, tk_type="llama3"):
    """
    >>> from transformers import AutoTokenizer
    >>> tk = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    >>> msg_dict = [{"role":"system", "content":"You are assistant"}, {"role":"user", "content":"I am user"}, {"role":"assistant", "content":"I am assistant"}]
    >>> assert dict_from_chat_template(tk.apply_chat_template(msg_dict, tokenize=False, add_generation_prompt=False)) == msg_dict
    """
    if tk_type == "llama3":
        rex = re.compile(
            r"(<\|eot_id\|>|<\|begin_of_text\|>)?<\|start_header_id\|>(\w+)<\|end_header_id\|>\n\n(.*?)(<\|eot_id\|>|$)",
            re.DOTALL,
        )
        return [
            {"role": m.group(2), "content": m.group(3)}
            for m in rex.finditer(chat_template_str)
        ]
    elif tk_type == "gpt2":
        raise NotImplementedError("gpt2 not implemented")
    else:
        raise ValueError(f"Unknown tk_type: {tk_type}")


def tokenize(tk_name: str, text: str):
    tk = AutoTokenizer.from_pretrained(tk_name)
    return tk(text)


def untokenize(tk_name: str, tokens: List[int]):
    tk = AutoTokenizer.from_pretrained(tk_name)
    return tk.decode(tokens)


@torch.inference_mode()
def compute_seq_log_probability(
    model: PreTrainedModel,
    pre_seq_tokens: list[int] | torch.Tensor,
    post_seq_tokens: list[int] | torch.Tensor,
    reduction: str = "sum",
) -> float:
    if isinstance(pre_seq_tokens, list):
        inputs = torch.tensor(
            pre_seq_tokens + post_seq_tokens, device=model.device
        ).unsqueeze(0)
    else:
        inputs = (
            torch.cat([pre_seq_tokens.view(-1), post_seq_tokens.view(-1)], dim=0)
            .unsqueeze(0)
            .to(model.device)
        )

    output = model(inputs)
    if th_available and isinstance(output, HeadedModelOutput):
        logits = output.preds_by_head["lm_head"][
            0, -len(post_seq_tokens) - 1 : -1
        ].cpu()
    else:
        logits = output.logits[0, -len(post_seq_tokens) - 1 : -1].cpu()

    logprobs = log_softmax(logits, dim=-1)

    red = torch.mean if reduction == "mean" else torch.sum

    res = float(
        red(logprobs.gather(1, torch.tensor(post_seq_tokens).unsqueeze(1)).squeeze())
    )

    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod()
