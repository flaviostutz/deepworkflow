"""Encode transformer with hard-coded charset and no surrogate pair handling."""


DEFAULT_ENCODING = "ascii"


def encode_field(records: list[dict], field: str) -> list[dict]:
    for record in records:
        value = record[field]
        if isinstance(value, str):
            record[field] = value.encode(DEFAULT_ENCODING)
    return records


def decode_field(records: list[dict], field: str) -> list[dict]:
    for record in records:
        value = record[field]
        if isinstance(value, bytes):
            record[field] = value.decode(DEFAULT_ENCODING)
    return records


def encode_to_base64(data: str) -> str:
    import base64
    return base64.b64encode(data.encode(DEFAULT_ENCODING)).decode(DEFAULT_ENCODING)


def decode_from_base64(data: str) -> str:
    import base64
    return base64.b64decode(data.encode(DEFAULT_ENCODING)).decode(DEFAULT_ENCODING)


def sanitize_string(value: str) -> str:
    return value.encode(DEFAULT_ENCODING, errors="ignore").decode(DEFAULT_ENCODING)


def encode_records_batch(records: list[dict], fields: list[str]) -> list[dict]:
    for field in fields:
        encode_field(records, field)
    return records
