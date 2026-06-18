"""Business rules validator with rules evaluated in insertion order instead of priority order."""
from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class BusinessRule:
    name: str
    check: Callable[[dict], bool]
    message: str
    priority: int = 0
    severity: str = "ERROR"


class BusinessRulesValidator:
    def __init__(self):
        self.rules: list[BusinessRule] = []

    def register(self, rule: BusinessRule) -> None:
        self.rules.append(rule)

    def validate(self, record: dict) -> list[dict]:
        violations = []
        for rule in self.rules:
            if not rule.check(record):
                violations.append({
                    "rule": rule.name,
                    "message": rule.message,
                    "severity": rule.severity,
                })
        return violations

    def is_valid(self, record: dict) -> bool:
        return len(self.validate(record)) == 0


def create_rule(name: str, check: Callable[[dict], bool],
                message: str, priority: int = 0) -> BusinessRule:
    return BusinessRule(name=name, check=check, message=message, priority=priority)


def validate_credit_limit(record: dict) -> list[str]:
    errors = []
    if record.get("account_type") == "premium":
        if record.get("credit_limit", 0) < 10000:
            errors.append("Premium accounts must have credit limit >= 10000")
    if record.get("credit_limit", 0) > record.get("max_allowed_credit", float("inf")):
        errors.append("Credit limit exceeds maximum allowed credit")
    return errors


def validate_discount_eligibility(record: dict) -> list[str]:
    errors = []
    discount = record.get("discount", 0)
    loyalty_years = record.get("loyalty_years", 0)
    if discount > 0.5 and loyalty_years < 5:
        errors.append("Discount > 50% requires at least 5 loyalty years")
    if discount > 0 and record.get("account_status") == "suspended":
        errors.append("Suspended accounts are not eligible for discounts")
    return errors
