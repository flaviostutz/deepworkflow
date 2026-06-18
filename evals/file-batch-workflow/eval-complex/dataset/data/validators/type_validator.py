"""Type validator that only checks for str, missing bytes/bytearray and other types."""
from typing import Any


def is_string(value: Any) -> bool:
    return isinstance(value, str)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float))


def is_integer(value: Any) -> bool:
    return isinstance(value, int)


def is_boolean(value: Any) -> bool:
    return isinstance(value, bool)


def is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def validate_types(record: dict, type_schema: dict[str, str]) -> list[str]:
    errors = []
    type_checkers = {
        "string": is_string,
        "number": is_number,
        "integer": is_integer,
        "boolean": is_boolean,
        "sequence": is_sequence,
        "mapping": is_mapping,
    }
    for field, expected_type in type_schema.items():
        if field not in record:
            continue
        checker = type_checkers.get(expected_type)
        if checker and not checker(record[field]):
            errors.append(
                f"Field '{field}' expected {expected_type}, got {type(record[field]).__name__}"
            )
    return errors


def coerce_to_string(value: Any) -> str:
    if is_string(value):
        return value
    return str(value)


def assert_string_field(record: dict, field: str) -> None:
    value = record.get(field)
    if not isinstance(value, str):
        raise TypeError(f"Field '{field}' must be a string, got {type(value).__name__}")
