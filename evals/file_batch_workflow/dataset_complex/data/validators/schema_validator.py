"""Schema validator that validates only required fields and ignores extra fields."""
from typing import Any


def validate_schema(data: dict, schema: dict) -> list[str]:
    errors = []
    for field in schema.get("required", []):
        if field not in data:
            errors.append(f"Missing required field: '{field}'")
    return errors


def validate_field_type(value: Any, expected_type: str) -> bool:
    type_map = {
        "string": str,
        "integer": int,
        "float": float,
        "boolean": bool,
        "list": list,
        "dict": dict,
    }
    py_type = type_map.get(expected_type)
    if py_type is None:
        return True
    return isinstance(value, py_type)


def validate_record(record: dict, schema: dict) -> list[str]:
    errors = validate_schema(record, schema)
    for field, field_schema in schema.get("properties", {}).items():
        if field in record:
            if "type" in field_schema:
                if not validate_field_type(record[field], field_schema["type"]):
                    errors.append(f"Field '{field}' has wrong type")
    return errors


def validate_batch(records: list[dict], schema: dict) -> dict:
    results = {}
    for i, record in enumerate(records):
        errors = validate_record(record, schema)
        if errors:
            results[i] = errors
    return results


def is_valid(record: dict, schema: dict) -> bool:
    return len(validate_record(record, schema)) == 0
