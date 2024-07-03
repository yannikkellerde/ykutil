import gc

import torch


def rolling_window(a, size):
    """
    >>> from torch import Tensor
    >>> rolling_window(Tensor([1,2,3]), 2)
    tensor([[1., 2.],
            [2., 3.]])
    """
    size = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.stride() + (a.stride()[-1],)
    return torch.as_strided(a, size=size, stride=strides)


def find_all_subarray_poses(arr, subarr, end=False):
    """
    >>> from torch import Tensor
    >>> find_all_subarray_poses(Tensor([1,2,3,4,5,1,2,3]), Tensor([1,2]))
    tensor([0, 5])
    """
    if len(subarr) > len(arr):
        return torch.tensor([], dtype=torch.int64)
    poses = torch.nonzero(
        torch.all(rolling_window(arr, len(subarr)) == subarr, dim=1)
    ).squeeze(1)
    if end and len(poses) > 0:
        poses += len(subarr)
    return poses


def disable_gradients(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    return model


def tensor_in(needles: torch.Tensor, haystack: torch.Tensor):
    """
    >>> from torch import Tensor
    >>> tensor_in(Tensor([7,3,4,1,9,12]), Tensor([9,4,13,3,0]))
    tensor([False,  True,  True, False,  True, False])
    """
    assert needles.ndim == haystack.ndim == 1
    if len(needles) == 0:
        return torch.tensor([], dtype=torch.bool)
    a1 = needles.expand((haystack.size(0), needles.size(0)))
    b1 = haystack.expand((needles.size(0), haystack.size(0))).t()
    return (a1 == b1).max(dim=0).values


def print_memory_info():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    fragment = r - a
    free = t - r
    print(
        f"\nMemory:\nTotal: {t}, Reserved: {r}, Allocated: {a}, Fragment: {fragment}, Free: {free}\n"
    )


def free_cuda_memory():
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
