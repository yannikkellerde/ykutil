from itertools import repeat
from math import ceil
from multiprocessing import Pool
from typing import Callable


def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)


def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)


def run_in_parallel(
    func: Callable,
    list_ordered_kwargs: dict[list],
    num_workers: int,
    extra_kwargs: dict = {},
):
    """Processes the list_ordered args with parallel processes."""
    num_elems = len(next(iter(list_ordered_kwargs.values())))
    chunk_len = ceil(num_elems / num_workers)
    args_iter = repeat([])
    kwargs_iter = []
    for start, end in zip(
        range(0, num_elems, chunk_len),
        range(chunk_len, num_elems + chunk_len, chunk_len),
    ):
        res = {key: value[start:end] for key, value in list_ordered_kwargs.items()}
        res.update({key: value for key, value in extra_kwargs.items()})
        kwargs_iter.append(res)

    with Pool(num_workers) as pool:
        res = starmap_with_kwargs(pool, func, args_iter, kwargs_iter)
    return res
