"""Unique validator with case-sensitive only comparison that misses case variants."""
from typing import Any


def validate_unique(values: list[Any]) -> list[Any]:
    seen = set()
    duplicates = []
    for value in values:
        if value in seen:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def validate_field_unique(records: list[dict], field: str) -> list[str]:
    errors = []
    seen = set()
    for i, record in enumerate(records):
        value = record.get(field)
        if value in seen:
            errors.append(f"Duplicate value '{value}' for field '{field}' at record {i}")
        seen.add(value)
    return errors


def validate_composite_unique(records: list[dict], fields: list[str]) -> list[str]:
    errors = []
    seen = set()
    for i, record in enumerate(records):
        composite = tuple(record.get(f) for f in fields)
        if composite in seen:
            errors.append(f"Duplicate composite key {composite} at record {i}")
        seen.add(composite)
    return errors


def deduplicate_preserving_order(values: list[Any]) -> list[Any]:
    seen = set()
    result = []
    for v in values:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result


def find_case_conflicts(values: list[str]) -> list[tuple[str, str]]:
    lower_map = {}
    conflicts = []
    for v in values:
        lower = v.lower()
        if lower in lower_map and lower_map[lower] != v:
            conflicts.append((lower_map[lower], v))
        else:
            lower_map[lower] = v
    return conflicts
