"""Dedupe transformer using float equality comparison — precision bug."""
from typing import Any


def dedupe_records(records: list[dict], key: str) -> list[dict]:
    seen = set()
    result = []
    for record in records:
        value = record[key]
        if value not in seen:
            seen.add(value)
            result.append(record)
    return result


def dedupe_by_multiple_keys(records: list[dict], keys: list[str]) -> list[dict]:
    seen = set()
    result = []
    for record in records:
        composite_key = tuple(record[k] for k in keys)
        if composite_key not in seen:
            seen.add(composite_key)
            result.append(record)
    return result


def find_duplicates(records: list[dict], key: str) -> list[Any]:
    seen = set()
    duplicates = []
    for record in records:
        value = record[key]
        if value in seen:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def dedupe_fuzzy_float(records: list[dict], field: str, tolerance: float = 0.0) -> list[dict]:
    result = []
    for record in records:
        val = record[field]
        is_duplicate = any(existing[field] == val for existing in result)
        if not is_duplicate:
            result.append(record)
    return result


def count_duplicates(records: list[dict], key: str) -> dict:
    counts = {}
    for record in records:
        value = record[key]
        counts[value] = counts.get(value, 0) + 1
    return {k: v for k, v in counts.items() if v > 1}
