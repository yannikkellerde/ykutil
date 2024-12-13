import gc
import io
from typing import Optional

import torch
import torch.nn.functional as F


def rolling_window(a: torch.Tensor, size: int) -> torch.Tensor:
    """
    >>> from torch import Tensor
    >>> rolling_window(Tensor([1,2,3]), 2)
    tensor([[1., 2.],
            [2., 3.]])
    """
    size = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.stride() + (a.stride()[-1],)
    return torch.as_strided(a, size=size, stride=strides)


def find_all_subarray_poses(
    arr: torch.Tensor,
    subarr: torch.Tensor,
    end=False,
    roll_window: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    >>> from torch import Tensor
    >>> find_all_subarray_poses(Tensor([1,2,3,4,5,1,2,3]), Tensor([1,2]))
    tensor([0, 5])
    """
    if len(subarr) > len(arr):
        return torch.tensor([], dtype=torch.int64)
    if roll_window is None:
        roll_window = rolling_window(arr, len(subarr))
    poses = torch.nonzero(torch.all(roll_window == subarr, dim=1)).squeeze(1)
    if end and len(poses) > 0:
        poses += len(subarr)
    return poses


def disable_gradients(model: torch.nn.Module) -> torch.nn.Module:
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


def pad_along_dimension(tensors, dim, pad_value=0):
    """

    Pads a list of tensors along a specified dimension so they can be stacked.

    Args:
        tensors (list of torch.Tensor): List of tensors to pad.
        dim (int): The dimension along which to pad.

    Returns:
        torch.Tensor: A single stacked tensor with all inputs padded along the specified dimension.

    >>> pad_along_dimension([torch.randn(4,12), torch.randn(4,21), torch.randn(4,5)], dim=1).shape
    torch.Size([3, 4, 21])
    >>> pad_along_dimension([torch.randn(4,12,3), torch.randn(4,21,3), torch.randn(4,5,3)], dim=1).shape
    torch.Size([3, 4, 21, 3])
    >>> pad_along_dimension([torch.randn(12), torch.randn(21), torch.randn(5)], dim=0).shape
    torch.Size([3, 21])
    """
    # Determine the target size for the specified dimension
    max_size = max(t.size(dim) for t in tensors)

    # Create a function to apply padding to the tensor
    padded_tensors = []
    for t in tensors:
        pad_sizes = [0] * t.dim() * 2  # Create a list of zeros for all dimensions
        pad_sizes[len(pad_sizes) - 1 - dim * 2] = max_size - t.size(
            dim
        )  # Set the padding for the specified dimension
        padded_tensors.append(F.pad(t, pad_sizes, value=pad_value))

    # Stack the padded tensors
    return torch.stack(padded_tensors)


def serialize_tensor(t: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()


def deserialize_tensor(b: bytes, map_location=None) -> torch.Tensor:
    buffer = io.BytesIO(b)
    return torch.load(buffer, map_location=map_location)


if __name__ == "__main__":
    # pad_along_dimension([torch.randn(4, 12), torch.randn(4, 21), torch.randn(4, 5)], 1)
    # pad_along_dimension([torch.randn(12), torch.randn(21), torch.randn(5)], 0)
    import doctest

    doctest.testmod()
