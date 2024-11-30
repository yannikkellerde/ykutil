import torch

from ykutil.multiprocess import (
    run_in_parallel,
)


def tensor_fill(tensor, fill_value, floatadd, extra_add=0):
    for t, f, fl in zip(tensor, fill_value, floatadd):
        t[:] = f + fl + extra_add
    return tensor


def test_fill_value():
    tensors = [torch.zeros(n) for n in range(4, 8)]
    extra_kwargs = {"extra_add": 1}
    list_kws = {
        "fill_value": [1, 2, 3, 4],
        "tensor": tensors,
        "floatadd": [0.1, 0.2, 0.3, 0.4],
    }
    res = run_in_parallel(tensor_fill, list_kws, 3, extra_kwargs)
    # for i in range(4):
    #     tensor_fill(tensors[i], list_kws["fill_value"][i], 1)
    print(tensors)


if __name__ == "__main__":
    test_fill_value()
