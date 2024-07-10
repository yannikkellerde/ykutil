from datasets import Dataset
from ykutil.types import describe_type


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
