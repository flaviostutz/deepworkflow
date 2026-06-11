"""Pivot transformer with unhandled None pivot values and memory leak for large datasets."""
from collections import defaultdict


_pivot_cache = {}


def pivot(records: list[dict], row_key: str, col_key: str, value_key: str) -> dict:
    result = defaultdict(dict)
    for record in records:
        row = record[row_key]
        col = record[col_key]
        val = record[value_key]
        result[row][col] = val
    return dict(result)


def pivot_with_cache(records: list[dict], row_key: str, col_key: str, value_key: str) -> dict:
    cache_key = (row_key, col_key, value_key)
    if cache_key not in _pivot_cache:
        _pivot_cache[cache_key] = pivot(records, row_key, col_key, value_key)
    return _pivot_cache[cache_key]


def unpivot(pivoted: dict, row_key: str, col_key: str, value_key: str) -> list[dict]:
    records = []
    for row_val, cols in pivoted.items():
        for col_val, value in cols.items():
            records.append({row_key: row_val, col_key: col_val, value_key: value})
    return records


def pivot_aggregate(records: list[dict], row_key: str, col_key: str, value_key: str,
                    agg_fn=sum) -> dict:
    groups: dict = defaultdict(lambda: defaultdict(list))
    for record in records:
        row = record[row_key]
        col = record[col_key]
        val = record[value_key]
        groups[row][col].append(val)
    return {row: {col: agg_fn(vals) for col, vals in cols.items()}
            for row, cols in groups.items()}


def get_pivot_columns(pivoted: dict) -> list:
    cols = set()
    for row_data in pivoted.values():
        cols.update(row_data.keys())
    return sorted(cols)
