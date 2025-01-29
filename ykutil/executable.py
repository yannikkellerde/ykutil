import argparse
import os
import json
from ast import literal_eval

from fire import Fire

from ykutil.dataset import colorcode_entry, count_token_sequence
from ykutil.dataset import describe_dataset as ds_describe_dataset
from ykutil.tools import bulk_rename
from ykutil.sql import merge_databases


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


def tokenize(tk: str, text: str, add_special_tokens: bool = False):
    from ykutil.transformer import tokenize as tk_tokenize

    return tk_tokenize(tk_name=tk, text=text, add_special_tokens=add_special_tokens)


def untokenize(tk: str, tokens: str):
    from ykutil.transformer import untokenize as tk_untokenize

    return tk_untokenize(tk_name=tk, tokens=literal_eval(tokens))


def do_merge_databases():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_db", type=str)
    parser.add_argument("source_dbs", type=str, nargs="+")
    args = parser.parse_args()
    return merge_databases(args.target_db, args.source_dbs)


def do_merge_database_folder():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_db", type=str)
    parser.add_argument("source_folder", type=str)
    args = parser.parse_args()

    source_dbs = [
        os.path.join(args.source_folder, f) for f in os.listdir(args.source_folder)
    ]

    return merge_databases(args.target_db, source_dbs)


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
    parser.add_argument("--add_special_tokens", action="store_true")
    args = parser.parse_args()
    return tokenize(
        tk=args.tk, text=args.text, add_special_tokens=args.add_special_tokens
    )


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


def do_count_token_sequence():
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_path", type=str)
    parser.add_argument("token_sequence", type=str)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1)
    args = parser.parse_args()
    token_sequence = literal_eval(args.token_sequence)

    count = count_token_sequence(
        token_ds_path=args.ds_path,
        token_sequence=token_sequence,
        num_start=args.start,
        num_end=args.end,
    )
    print(count)


def do_beautify_json():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_str", type=str)
    args = parser.parse_args()
    return beautify_json(json_str=args.json_str)
