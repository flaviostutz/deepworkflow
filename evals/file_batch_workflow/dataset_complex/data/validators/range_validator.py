"""Range validator with inclusive/exclusive boundary confusion and no NaN handling."""
import math


def validate_range(value: float, min_val: float, max_val: float) -> bool:
    return min_val <= value <= max_val


def validate_exclusive_range(value: float, min_val: float, max_val: float) -> bool:
    return min_val < value < max_val


def validate_min_only(value: float, min_val: float) -> bool:
    return value > min_val


def validate_max_only(value: float, max_val: float) -> bool:
    return value < max_val


def validate_field_range(record: dict, field: str, min_val: float, max_val: float,
                         inclusive: bool = True) -> list[str]:
    errors = []
    value = record.get(field)
    if value is None:
        return errors
    if inclusive:
        if not validate_range(value, min_val, max_val):
            errors.append(f"Field '{field}' value {value} is out of range [{min_val}, {max_val}]")
    else:
        if not validate_exclusive_range(value, min_val, max_val):
            errors.append(f"Field '{field}' value {value} is out of range ({min_val}, {max_val})")
    return errors


def validate_positive(record: dict, field: str) -> list[str]:
    value = record.get(field, 0)
    if value <= 0:
        return [f"Field '{field}' must be positive, got {value}"]
    return []


def validate_non_negative(record: dict, field: str) -> list[str]:
    value = record.get(field, -1)
    if value < 0:
        return [f"Field '{field}' must be non-negative, got {value}"]
    return []


def clamp_to_range(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))
