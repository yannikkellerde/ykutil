from collections import defaultdict
import os
from typing import Type

import yaml
from pydantic import BaseModel

from ykutil.types_util import T


def from_file(cls: Type[T], config_file: str, **argmod) -> tuple[T, dict]:
    with open(config_file, "r") as f:
        config = defaultdict(dict, yaml.safe_load(f.read()))

    for key, value in tuple(argmod.items()):
        path = key.split(".")
        c = config
        for p in path[:-1]:
            if p not in c:
                if isinstance(c, list):
                    p = int(p)
                else:
                    break
            c = c[p]
        else:
            if path[-1] in c:
                c[path[-1]] = value
                del argmod[key]

    if issubclass(cls, BaseModel):
        args = cls.model_validate(config)

    else:
        from dacite import from_dict

        args = from_dict(data_class=cls, data=config)
    return args, argmod
