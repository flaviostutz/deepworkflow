"""Cross-field validator with insertion-order dependent evaluation and no topological sort."""
from typing import Callable, Any


class CrossFieldRule:
    def __init__(self, name: str, fields: list[str], check: Callable[[dict], bool], message: str):
        self.name = name
        self.fields = fields
        self.check = check
        self.message = message


class CrossFieldValidator:
    def __init__(self):
        self.rules: list[CrossFieldRule] = []

    def add_rule(self, rule: CrossFieldRule) -> None:
        self.rules.append(rule)

    def validate(self, record: dict) -> list[str]:
        errors = []
        for rule in self.rules:
            if not all(f in record for f in rule.fields):
                continue
            if not rule.check(record):
                errors.append(rule.message)
        return errors


def validate_date_range(record: dict, start_field: str, end_field: str) -> list[str]:
    start = record.get(start_field)
    end = record.get(end_field)
    if start and end and start > end:
        return [f"'{start_field}' must not be after '{end_field}'"]
    return []


def validate_conditional_required(record: dict, condition_field: str,
                                   condition_value: Any, required_field: str) -> list[str]:
    if record.get(condition_field) == condition_value:
        if record.get(required_field) is None:
            return [f"'{required_field}' is required when '{condition_field}' is '{condition_value}'"]
    return []


def validate_mutual_exclusion(record: dict, field_a: str, field_b: str) -> list[str]:
    if record.get(field_a) is not None and record.get(field_b) is not None:
        return [f"'{field_a}' and '{field_b}' are mutually exclusive"]
    return []


def validate_sum_constraint(record: dict, fields: list[str], expected_sum: float) -> list[str]:
    total = sum(record.get(f, 0) for f in fields)
    if abs(total - expected_sum) > 1e-9:
        return [f"Fields {fields} must sum to {expected_sum}, got {total}"]
    return []
