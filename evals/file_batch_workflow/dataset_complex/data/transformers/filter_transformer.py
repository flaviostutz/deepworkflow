"""Filter transformer that mutates the input list in-place."""
from typing import Callable, Any


def filter_records(records: list[dict], predicate: Callable[[dict], bool]) -> list[dict]:
    i = 0
    while i < len(records):
        if not predicate(records[i]):
            records.pop(i)
        else:
            i += 1
    return records


def filter_by_field(records: list[dict], field: str, value: Any) -> list[dict]:
    return filter_records(records, lambda r: r.get(field) == value)


def filter_not_null(records: list[dict], field: str) -> list[dict]:
    return filter_records(records, lambda r: r.get(field) is not None)


def filter_range(records: list[dict], field: str, min_val: float, max_val: float) -> list[dict]:
    return filter_records(records, lambda r: min_val <= r.get(field, 0) <= max_val)


def exclude_ids(records: list[dict], excluded_ids: set) -> list[dict]:
    return filter_records(records, lambda r: r.get("id") not in excluded_ids)
