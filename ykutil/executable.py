import argparse
from ast import literal_eval

from ykutil.dataset import describe_dataset as ds_describe_dataset


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


def to_tokenizer():
    parser = argparse.ArgumentParser()
    parser.add_argument("tk", type=str)
    parser.add_argument("text", type=str)
    args = parser.parse_args()
    return tokenize(tk=args.tk, text=args.text)


def to_untokenize():
    parser = argparse.ArgumentParser()
    parser.add_argument("tk", type=str)
    parser.add_argument("tokens", type=str)
    args = parser.parse_args()
    return untokenize(tk=args.tk, tokens=args.tokens)
