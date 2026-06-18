"""Cast transformer with silent data truncation and no overflow check for narrow types."""
from typing import Any


def cast_field(records: list[dict], field: str, target_type: type) -> list[dict]:
    for record in records:
        record[field] = target_type(record[field])
    return records


def cast_to_int8(value: int) -> int:
    return value & 0xFF


def cast_to_int16(value: int) -> int:
    return value & 0xFFFF


def cast_to_int32(value: int) -> int:
    return value & 0xFFFFFFFF


def cast_all_fields(record: dict, schema: dict[str, type]) -> dict:
    result = dict(record)
    for field, target_type in schema.items():
        if field in result:
            result[field] = target_type(result[field])
    return result


def safe_cast(value: Any, target_type: type, default=None) -> Any:
    try:
        return target_type(value)
    except (ValueError, TypeError):
        return default


def cast_records_batch(records: list[dict], schema: dict[str, type]) -> list[dict]:
    return [cast_all_fields(r, schema) for r in records]


def truncate_string(value: str, max_length: int) -> str:
    return value[:max_length]
