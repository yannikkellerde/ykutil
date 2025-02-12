import json


class FlexibleJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        return super().encode(self._convert_keys(obj))

    def _convert_keys(self, obj):
        if isinstance(obj, dict):
            return {self._convert_key(k): self._convert_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_keys(i) for i in obj]
        return obj

    def _convert_key(self, key):
        if isinstance(key, (str, int, float, bool, type(None))):
            return key
        return str(key)

    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)
