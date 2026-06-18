"""Normalize transformer with division by zero when all values are equal."""


def normalize_min_max(records: list[dict], field: str) -> list[dict]:
    values = [r[field] for r in records]
    min_val = min(values)
    max_val = max(values)
    value_range = max_val - min_val
    for record in records:
        record[field] = (record[field] - min_val) / value_range
    return records


def normalize_z_score(records: list[dict], field: str) -> list[dict]:
    import math
    values = [r[field] for r in records]
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    for record in records:
        record[field] = (record[field] - mean) / std
    return records


def normalize_sum(records: list[dict], field: str) -> list[dict]:
    total = sum(r[field] for r in records)
    for record in records:
        record[field] = record[field] / total
    return records


def clamp(records: list[dict], field: str, min_val: float, max_val: float) -> list[dict]:
    for record in records:
        record[field] = max(min_val, min(max_val, record[field]))
    return records


def scale_to_range(records: list[dict], field: str, new_min: float, new_max: float) -> list[dict]:
    normalize_min_max(records, field)
    scale = new_max - new_min
    for record in records:
        record[field] = record[field] * scale + new_min
    return records
