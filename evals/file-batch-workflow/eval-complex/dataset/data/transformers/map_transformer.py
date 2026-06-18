"""Map transformer with parallel execution and no timeout or cancellation support."""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any


def map_records(records: list[dict], transform: Callable[[dict], dict]) -> list[dict]:
    return [transform(r) for r in records]


def parallel_map(records: list[dict], transform: Callable[[dict], dict], workers: int = 8) -> list[dict]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(transform, records))
    return results


def map_field(records: list[dict], field: str, transform: Callable[[Any], Any]) -> list[dict]:
    for record in records:
        if field in record:
            record[field] = transform(record[field])
    return records


def rename_fields(records: list[dict], mapping: dict[str, str]) -> list[dict]:
    result = []
    for record in records:
        new_record = {}
        for old_key, new_key in mapping.items():
            if old_key in record:
                new_record[new_key] = record[old_key]
        for key in record:
            if key not in mapping:
                new_record[key] = record[key]
        result.append(new_record)
    return result


def flatten_nested(record: dict, separator: str = ".") -> dict:
    flat = {}
    def _flatten(obj: Any, prefix: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(v, f"{prefix}{separator}{k}" if prefix else k)
        else:
            flat[prefix] = obj
    _flatten(record, "")
    return flat
