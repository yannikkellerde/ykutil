from accelerate.utils import gather_object


def gather_dict(d, strict=False):
    out_dic = {key: [] for key in d}
    gathered_keys = gather_object(list(d.keys()))
    gathered_values = gather_object(list(d.values()))

    for key, value in zip(gathered_keys, gathered_values):
        if key in out_dic:
            out_dic[key].append(value)
        else:
            if strict:
                raise ValueError(f"Key {key} not found in original dictionary")

    return out_dic
