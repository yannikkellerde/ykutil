import torch

from ykutil.multiprocess import (
    apply_args_and_kwargs,
    run_in_parallel,
    starmap_with_kwargs,
)


def tensor_fill(tensor, fill_value, extra_add=0):
    for t, f in zip(tensor, fill_value):
        t[:] = f + extra_add
    return tensor


def test_fill_value():
    tensors = [torch.zeros(n) for n in range(4, 8)]
    extra_kwargs = {"extra_add": 1}
    list_kws = {"fill_value": [1, 2, 3, 4], "tensor": tensors}
    res = run_in_parallel(tensor_fill, list_kws, 4, extra_kwargs)
    # for i in range(4):
    #     tensor_fill(tensors[i], list_kws["fill_value"][i], 1)
    print(tensors)


if __name__ == "__main__":
    test_fill_value()
