from collections import defaultdict
from copy import copy
from dataclasses import asdict, is_dataclass
from typing import Callable

from ykutil.python import recursed_merge_percent_stats, recursed_sum_up_stats
from ykutil.types_util import T


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


class Serializable(dict):
    """Inherit from this when creating a dataclass to make it
    json serializable and hashable.
    """

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    def items(self):
        return asdict(self).items()

    def __len__(self):
        return len(asdict(self))

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result


def sortedtuple(sort_fun: Callable, fixed_len=None):
    def __new__(cls, *args):
        assert fixed_len is None or len(args) == fixed_len
        return tuple.__new__(cls, sorted(args, key=sort_fun))

    return type("SortedTuple", (tuple,), {"__new__": __new__})
