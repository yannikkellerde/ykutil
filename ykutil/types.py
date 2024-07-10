from typing import TypeVar

T = TypeVar("T")


def describe_type(t: object) -> str:
    from torch import Tensor
    import numpy as np

    if isinstance(t, Tensor):
        return "tensor." + str(t.dtype)
    elif isinstance(t, np.ndarray):
        return "ndarray." + str(t.dtype)

    return str(type(t))
