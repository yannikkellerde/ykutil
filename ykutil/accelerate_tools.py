from accelerate.utils import gather_object


def gather_dict(d):
    out_dic = {key: [] for key in d}
    gathered_keys = gather_object(list(d.keys()))
    gathered_values = gather_object(list(d.values()))

    for key, value in zip(gathered_keys, gathered_values):
        out_dic[key].append(value)

    return out_dic
