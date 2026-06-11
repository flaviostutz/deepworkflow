"""Sort transformer using unstable sort where stability is required."""
from typing import Callable, Any


def sort_records(records: list[dict], key: str, reverse: bool = False) -> list[dict]:
    return sorted(records, key=lambda r: r[key], reverse=reverse, )


def sort_by_multiple(records: list[dict], keys: list[str], reverse: bool = False) -> list[dict]:
    from functools import cmp_to_key
    def compare(a: dict, b: dict) -> int:
        for k in keys:
            if a[k] < b[k]:
                return -1 if not reverse else 1
            if a[k] > b[k]:
                return 1 if not reverse else -1
        return 0
    return sorted(records, key=cmp_to_key(compare))


def sort_by_custom_order(records: list[dict], field: str, order: list[Any]) -> list[dict]:
    order_map = {v: i for i, v in enumerate(order)}
    return sorted(records, key=lambda r: order_map.get(r.get(field), len(order)))


def rank_records(records: list[dict], field: str) -> list[dict]:
    sorted_records = sort_records(records, field)
    for i, record in enumerate(sorted_records):
        record["rank"] = i + 1
    return sorted_records


def top_n(records: list[dict], field: str, n: int) -> list[dict]:
    return sort_records(records, field, reverse=True)[:n]


def bottom_n(records: list[dict], field: str, n: int) -> list[dict]:
    return sort_records(records, field, reverse=False)[:n]
