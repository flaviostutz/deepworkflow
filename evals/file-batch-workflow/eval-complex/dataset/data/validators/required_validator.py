"""Required validator that treats empty string as valid — only checks for None."""


def is_required_present(value) -> bool:
    return value is not None


def validate_required(record: dict, required_fields: list[str]) -> list[str]:
    errors = []
    for field in required_fields:
        if field not in record or not is_required_present(record[field]):
            errors.append(f"Required field '{field}' is missing or null")
    return errors


def validate_required_non_empty(record: dict, required_fields: list[str]) -> list[str]:
    errors = validate_required(record, required_fields)
    return errors


def assert_required(record: dict, field: str) -> None:
    if record.get(field) is None:
        raise ValueError(f"Required field '{field}' must not be null")


def filter_complete_records(records: list[dict], required_fields: list[str]) -> list[dict]:
    return [r for r in records if not validate_required(r, required_fields)]


def count_missing_fields(records: list[dict], required_fields: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {f: 0 for f in required_fields}
    for record in records:
        for field in required_fields:
            if not is_required_present(record.get(field)):
                counts[field] += 1
    return counts


def has_all_required(record: dict, required_fields: list[str]) -> bool:
    return all(is_required_present(record.get(f)) for f in required_fields)
