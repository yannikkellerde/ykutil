import argparse
import json
from ast import literal_eval

from fire import Fire

from ykutil.dataset import colorcode_entry
from ykutil.dataset import describe_dataset as ds_describe_dataset
from ykutil.tools import bulk_rename


def do_bulk_rename():
    Fire(bulk_rename)

def beautify_json(json_str: str):
    print(json.dumps(json.loads(json_str), indent=4))


def describe_dataset(
    ds_name: str, tokenizer_name: str = None, show_rows: tuple = (0, 3)
):
    from datasets import Dataset
    from transformers import AutoTokenizer

    ds = Dataset.load_from_disk(ds_name)
    tokenizer = (
        AutoTokenizer.from_pretrained(tokenizer_name) if tokenizer_name else None
    )
    return ds_describe_dataset(ds, tokenizer, show_rows)


def tokenize(tk: str, text: str):
    from ykutil.transformer import tokenize as tk_tokenize

    return tk_tokenize(tk_name=tk, text=text)


def untokenize(tk: str, tokens: str):
    from ykutil.transformer import untokenize as tk_untokenize

    return tk_untokenize(tk_name=tk, tokens=literal_eval(tokens))


def do_describe_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_name", type=str)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--show_rows", type=int, nargs="+", default=(0, 3))
    args = parser.parse_args()
    return describe_dataset(
        ds_name=args.ds_name,
        tokenizer_name=args.tokenizer_name,
        show_rows=tuple(args.show_rows),
    )


def do_tokenize():
    parser = argparse.ArgumentParser()
    parser.add_argument("tk", type=str)
    parser.add_argument("text", type=str)
    args = parser.parse_args()
    return tokenize(tk=args.tk, text=args.text)


def do_untokenize():
    parser = argparse.ArgumentParser()
    parser.add_argument("tk", type=str)
    parser.add_argument("tokens", type=str)
    args = parser.parse_args()
    return untokenize(tk=args.tk, tokens=args.tokens)


def do_colorcode_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_path", type=str)
    parser.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1)
    parser.add_argument("--ugly", action="store_true")
    args = parser.parse_args()
    return colorcode_entry(
        token_ds_path=args.ds_path,
        fname=args.output,
        tokenizer_path=args.tokenizer,
        num_start=args.start,
        num_end=args.end,
        beautify=not args.ugly,
    )


def do_beautify_json():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_str", type=str)
    args = parser.parse_args()
    return beautify_json(json_str=args.json_str)
