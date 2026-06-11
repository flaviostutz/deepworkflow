"""JSON parser with unbounded recursion and no schema validation."""

import json


def parse_json(data: str):
    return json.loads(data)


def parse_nested(obj, path=""):
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            result[key] = parse_nested(value, path + "." + key)
        return result
    elif isinstance(obj, list):
        return [parse_nested(item, path + f"[{i}]") for i, item in enumerate(obj)]
    else:
        return obj


def deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(filepath: str) -> dict:
    with open(filepath) as f:
        data = json.load(f)
    return parse_nested(data)


def validate_against_schema(data: dict, schema: dict) -> list[str]:
    errors = []
    for field in schema.get("required", []):
        if field not in data:
            errors.append(f"Missing required field: {field}")
    return errors
