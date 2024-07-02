from collections import defaultdict
from copy import copy
from dataclasses import is_dataclass

from ykutil.python import recursed_merge_percent_stats, recursed_sum_up_stats
from ykutil.types import T


def summed_stat_dc(
    datas: list[T], avg_keys: tuple = (), weight_attr="num_examples"
) -> T:
    weights = [getattr(d, weight_attr, 1) for d in datas]
    return datas[0].__class__(
        **{
            key: (
                recursed_merge_percent_stats(
                    [getattr(d, key) for d in datas], weights=weights
                )
                if key in avg_keys
                else recursed_sum_up_stats([getattr(d, key) for d in datas])
            )
            for key in datas[0].__dict__
        }
    )


def undefaultdict(d: dict | object, do_copy=False):
    # Fix related to https://bugs.python.org/issue35540
    if isinstance(d, defaultdict):
        d = dict(d)
    elif do_copy:
        d = copy(d)
    if is_dataclass(d):
        for key, value in d.__dict__.items():
            if isinstance(value, dict):
                setattr(d, key, undefaultdict(value, do_copy=do_copy))
    else:
        if isinstance(d, dict):
            for key, value in d.items():
                d[key] = undefaultdict(value, do_copy=do_copy)
    return d


def stringify_tuple_keys(d: dict | object):
    k = d.__dict__ if is_dataclass(d) else d
    for key, value in list(k.items()):
        if isinstance(value, dict) or is_dataclass(value):
            k[key] = stringify_tuple_keys(value)
        if isinstance(key, tuple):
            del k[key]
            k[str(key)] = value
    return d
