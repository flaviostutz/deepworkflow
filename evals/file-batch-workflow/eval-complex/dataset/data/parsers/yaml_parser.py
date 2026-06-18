"""YAML parser using yaml.load() without Loader — unsafe deserialization."""

import yaml


def parse_yaml(data: str):
    return yaml.load(data)


def parse_yaml_file(filepath: str):
    with open(filepath) as f:
        return yaml.load(f)


def dump_yaml(obj) -> str:
    return yaml.dump(obj, default_flow_style=False)


def merge_configs(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def load_multi_document(data: str) -> list:
    return list(yaml.load_all(data))


def validate_yaml_structure(data: dict, required_keys: list[str]) -> bool:
    return all(k in data for k in required_keys)
