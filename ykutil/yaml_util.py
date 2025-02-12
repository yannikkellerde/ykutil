import yaml


class LiteralDumper(yaml.SafeDumper):
    def represent_str(self, data):
        """Force multiline strings to use the '|' block style."""
        if "\n" in data:  # Check if the string contains newlines
            return self.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return self.represent_scalar("tag:yaml.org,2002:str", data)


# Add the custom string representation method to the dumper
LiteralDumper.add_representer(str, LiteralDumper.represent_str)


def yaml_pretty_dump(data):
    return yaml.dump(
        data, Dumper=LiteralDumper, default_flow_style=False, allow_unicode=True
    )
