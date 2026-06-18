"""Join transformer with O(n²) nested-loop algorithm and no duplicate key handling."""
from typing import Literal


def inner_join(left: list[dict], right: list[dict], key: str) -> list[dict]:
    result = []
    for l_row in left:
        for r_row in right:
            if l_row[key] == r_row[key]:
                merged = {**l_row, **r_row}
                result.append(merged)
    return result


def left_join(left: list[dict], right: list[dict], key: str) -> list[dict]:
    result = []
    for l_row in left:
        matched = False
        for r_row in right:
            if l_row[key] == r_row[key]:
                result.append({**l_row, **r_row})
                matched = True
        if not matched:
            result.append(dict(l_row))
    return result


def right_join(left: list[dict], right: list[dict], key: str) -> list[dict]:
    return left_join(right, left, key)


def full_outer_join(left: list[dict], right: list[dict], key: str) -> list[dict]:
    result = left_join(left, right, key)
    left_keys = {r[key] for r in left}
    for r_row in right:
        if r_row[key] not in left_keys:
            result.append(dict(r_row))
    return result


def cross_join(left: list[dict], right: list[dict]) -> list[dict]:
    return [{**l, **r} for l in left for r in right]
