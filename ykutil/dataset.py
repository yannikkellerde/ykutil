from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer

from ykutil.constants import IGNORE_INDEX
from ykutil.python import count_sublist_occurrences
from ykutil.transformer import load_tk_with_pad_tk
from ykutil.types_util import describe_type


def describe_dataset(ds: Dataset, tokenizer=None, show_rows=(0,), zip_labels=False):
    pr = lambda p: print("###############\n" + p)
    pr("Metadata:")
    print(ds.info)
    pr("Columns:")
    print([{col: describe_type(ds[0][col])} for col in ds.column_names])
    pr("Number of rows:")
    print(len(ds))
    pr("Example rows:")
    for i in range(*show_rows):
        example = ds[i]
        if "input_ids" in example:
            if tokenizer is None:
                raise ValueError(
                    "tokenizer is required to show rows for token datasets"
                )
            else:
                if zip_labels:
                    assert "labels" in example
                    tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
                    for token, label in zip(tokens, example["labels"]):
                        if label != IGNORE_INDEX:
                            print(f"({token}, {float(label)})", end=" ")
                        else:
                            print(token, end=" ")
                else:
                    example["input_ids"] = tokenizer.decode(example["input_ids"])
                    print(example)
        else:
            for key, value in example.items():
                print(f"{key}: {value}")


def colorcode_dataset(
    dd: DatasetDict,
    tk: PreTrainedTokenizer,
    num_start=5,
    num_end=6,
    data_key="train",
    fname=None,
    beautify=True,
):
    from termcolor import colored

    for i in range(num_start, num_end):
        data = dd[data_key][i]
        data["input_ids"] = data["input_ids"].squeeze()
        data["labels"] = data["labels"].squeeze()
        data["labels"] = data["labels"][data["input_ids"] != tk.eos_token_id]
        data["input_ids"] = data["input_ids"][data["input_ids"] != tk.eos_token_id]
        ignores = data["labels"] == IGNORE_INDEX
        if "weights" in data:
            print("Max weight:", data["weights"].max())
            data["weights"] = data["weights"] / data["weights"].max()
            colors = [
                "red" if x < 0.33 else "red" if x < 0.67 else "red"
                for x in data["weights"]
            ]
            attr = [
                ["dark"] if x < 0.33 else None if x < 0.67 else ["underline"]
                for x in data["weights"]
            ]

        tok_strs = tk.convert_ids_to_tokens(data["input_ids"])
        tok_strs = [
            (
                x
                if ignores[i]
                else colored(
                    x,
                    colors[i] if "weights" in data else "red",
                    attrs=attr[i] if "weights" in data else None,
                )
            )
            for i, x in enumerate(tok_strs)
        ]
        msg = "".join(tok_strs)
        if beautify:
            msg = (
                msg.replace("▁", " ")
                .replace("<0x0A>", "\n")
                .replace("Ġ", " ")
                .replace("Ċ", "\n")
                .replace("<|reserved_special_token_200|>", "\n")
            )
        if fname is None:
            print(msg)
        else:
            with open(fname, "w") as f:
                f.write(msg)


def colorcode_entry(
    token_ds_path,
    fname=None,
    tokenizer_path="mistralai/Mistral-7B-v0.1",
    num_start=0,
    num_end=1,
    beautify=True,
):
    token_dd = DatasetDict.load_from_disk(token_ds_path).with_format("torch")
    tokenizer = load_tk_with_pad_tk(tokenizer_path)
    colorcode_dataset(
        token_dd,
        tokenizer,
        fname=fname,
        num_start=num_start,
        num_end=num_end,
        beautify=beautify,
    )


def count_token_sequence(
    token_ds_path: str, token_sequence: list[int], num_start=0, num_end=1
):
    token_ds = Dataset.load_from_disk(token_ds_path)
    count = 0
    for i in range(num_start, num_end):
        data = token_ds[i]
        in_ids = data["input_ids"]
        count += count_sublist_occurrences(in_ids, token_sequence)

    return count
