from collections import defaultdict
from typing import Type

import yaml
from dacite import from_dict

from ykutil.types_util import T


def from_file(cls: Type[T], config_file: str, **argmod) -> T:
    with open(config_file, "r") as f:
        config = defaultdict(dict, yaml.safe_load(f.read()))

    for key, value in argmod.items():
        path = key.split(".")
        c = config
        for p in path[:-1]:
            c = c[p]
        c[path[-1]] = value

    args = from_dict(data_class=cls, data=config)
    return args
