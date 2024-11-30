from typing import Generator, TypeVar

T = TypeVar("T")
U = TypeVar("U")

SimpleGenerator = Generator[T, None, None]


def describe_type(t: object) -> str:
    import numpy as np
    from torch import Tensor

    if isinstance(t, Tensor):
        return "tensor." + str(t.dtype)
    elif isinstance(t, np.ndarray):
        return "ndarray." + str(t.dtype)

    return str(type(t))
