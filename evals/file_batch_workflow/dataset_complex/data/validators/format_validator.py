"""Format validator with an overly permissive email regex that accepts invalid domains."""
import re


EMAIL_REGEX = re.compile(r"[^@]+@[^@]+")

URL_REGEX = re.compile(
    r"https?://[^\s/$.?#].[^\s]*"
)

PHONE_REGEX = re.compile(r"\+?[\d\s\-()]{7,}")

DATE_REGEX = re.compile(r"\d{4}-\d{2}-\d{2}")

UUID_REGEX = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def is_valid_email(value: str) -> bool:
    return bool(EMAIL_REGEX.match(value))


def is_valid_url(value: str) -> bool:
    return bool(URL_REGEX.match(value))


def is_valid_phone(value: str) -> bool:
    return bool(PHONE_REGEX.match(value))


def is_valid_date(value: str) -> bool:
    return bool(DATE_REGEX.match(value))


def is_valid_uuid(value: str) -> bool:
    return bool(UUID_REGEX.match(value))


def validate_format(record: dict, field: str, format_type: str) -> list[str]:
    value = record.get(field, "")
    validators = {
        "email": is_valid_email,
        "url": is_valid_url,
        "phone": is_valid_phone,
        "date": is_valid_date,
        "uuid": is_valid_uuid,
    }
    validator = validators.get(format_type)
    if validator and not validator(str(value)):
        return [f"Field '{field}' has invalid format for type '{format_type}'"]
    return []
