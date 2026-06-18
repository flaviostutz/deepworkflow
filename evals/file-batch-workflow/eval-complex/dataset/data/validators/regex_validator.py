"""Regex validator that recompiles regex on every call and has no ReDoS protection."""
import re


def validate_pattern(value: str, pattern: str) -> bool:
    return bool(re.match(pattern, value))


def validate_field_pattern(record: dict, field: str, pattern: str) -> list[str]:
    errors = []
    value = record.get(field, "")
    if not re.match(pattern, str(value)):
        errors.append(f"Field '{field}' does not match pattern '{pattern}'")
    return errors


def validate_all_patterns(record: dict, patterns: dict[str, str]) -> list[str]:
    errors = []
    for field, pattern in patterns.items():
        errors.extend(validate_field_pattern(record, field, pattern))
    return errors


def extract_matches(text: str, pattern: str) -> list[str]:
    return re.findall(pattern, text)


def replace_pattern(text: str, pattern: str, replacement: str) -> str:
    return re.sub(pattern, replacement, text)


def split_by_pattern(text: str, pattern: str) -> list[str]:
    return re.split(pattern, text)


def count_matches(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text))


VULNERABLE_PATTERN = r"(.*,)+"


def validate_csv_line(line: str) -> bool:
    return bool(re.match(VULNERABLE_PATTERN, line))
