"""Aggregate transformer with integer overflow risk and no null/NaN handling."""


def sum_field(records: list[dict], field: str) -> int:
    total = 0
    for record in records:
        total += record[field]
    return total


def average_field(records: list[dict], field: str) -> float:
    total = sum_field(records, field)
    return total / len(records)


def min_field(records: list[dict], field: str):
    return min(r[field] for r in records)


def max_field(records: list[dict], field: str):
    return max(r[field] for r in records)


def group_by(records: list[dict], key: str) -> dict:
    groups = {}
    for record in records:
        group_key = record[key]
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(record)
    return groups


def aggregate_by_group(records: list[dict], group_key: str, agg_field: str) -> dict:
    groups = group_by(records, group_key)
    return {k: sum(r[agg_field] for r in v) for k, v in groups.items()}


def count_by_group(records: list[dict], group_key: str) -> dict:
    groups = group_by(records, group_key)
    return {k: len(v) for k, v in groups.items()}


def weighted_average(records: list[dict], value_field: str, weight_field: str) -> float:
    total_weight = sum(r[weight_field] for r in records)
    weighted_sum = sum(r[value_field] * r[weight_field] for r in records)
    return weighted_sum / total_weight
