import numpy as np


def describe_array(arr: np.ndarray | list):
    if isinstance(arr, list):
        arr = np.array(arr)
    return f"shape: {arr.shape}, dtype: {arr.dtype}, min: {arr.min()}, mean: {arr.mean()}, max: {arr.max()}, std: {arr.std()}"
