from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer

from ykutil.constants import IGNORE_INDEX
from ykutil.transformer import load_tk_with_pad_tk
from ykutil.types_util import describe_type


def describe_dataset(ds: Dataset, tokenizer=None, show_rows=(0, 3)):
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
        if "input_ids" in example and tokenizer is not None:
            example["input_ids"] = tokenizer.decode(example["input_ids"])
        print(example)


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
