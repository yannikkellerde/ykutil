from collections import defaultdict
from typing import Any, Type

import numpy as np
import torch


def describe_recursive(
    l: Any,
    types: defaultdict[int, set[str]],
    lengths: defaultdict[int, set[int]],
    arrays: defaultdict[int, set[tuple[Type, Type, tuple]]],
    dict_keys: defaultdict[int, set[Any]],
    depth=0,
):
    types[depth].add(type(l).__name__)
    if isinstance(l, list) or isinstance(l, tuple):
        lengths[depth].add(len(l))
        for el in l:
            describe_recursive(el, types, lengths, arrays, dict_keys, depth=depth + 1)
    elif isinstance(l, np.ndarray):
        arrays[depth].add((np.ndarray, l.dtype, l.shape))
    elif isinstance(l, torch.Tensor):
        arrays[depth].add((torch.Tensor, l.dtype, l.shape))
    elif isinstance(l, dict):
        dict_keys[depth].update(l.keys())


def describe_list(l: list | tuple, no_empty=True):
    assert isinstance(l, list) or isinstance(l, tuple), f"l is {type(l)}"
    types_by_depth = defaultdict(set)
    lenghts = defaultdict(set)
    arrays = defaultdict(set)
    dict_keys = defaultdict(set)
    describe_recursive(l, types_by_depth, lenghts, arrays, dict_keys, depth=0)

    obj = {
        "types": dict(types_by_depth),
        "lengths": dict(lenghts),
        "arrays": dict(arrays),
        "dict_keys": dict(dict_keys),
    }

    if no_empty:
        for k, v in tuple(obj.items()):
            if not v:
                del obj[k]

    return obj


def describe_array(arr: np.ndarray | list | torch.Tensor):
    if isinstance(arr, list):
        arr = np.array(arr)
    elif isinstance(arr, torch.Tensor):
        return f"shape: {arr.shape}, dtype: {arr.dtype}, min: {arr.min()}, mean: {arr.mean(dtype=float)}, max: {arr.max()}"
    assert isinstance(arr, np.ndarray)
    return f"shape: {arr.shape}, dtype: {arr.dtype}, min: {arr.min()}, mean: {arr.mean(dtype=float)}, max: {arr.max()}, std: {arr.std()}"
