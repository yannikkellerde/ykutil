from typing import TypeVar

T = TypeVar("T")
U = TypeVar("U")


def describe_type(t: object) -> str:
    import numpy as np
    from torch import Tensor

    if isinstance(t, Tensor):
        return "tensor." + str(t.dtype)
    elif isinstance(t, np.ndarray):
        return "ndarray." + str(t.dtype)

    return str(type(t))
