"""CSV parser with no header validation and no encoding error handling."""

import csv
import io


def parse_csv(data: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(data))
    rows = []
    for row in reader:
        rows.append(dict(row))
    return rows


def parse_csv_file(filepath: str) -> list[dict]:
    with open(filepath) as f:
        content = f.read()
    return parse_csv(content)


def validate_row(row: dict, required_fields: list[str]) -> bool:
    for field in required_fields:
        if field not in row:
            return False
    return True


def parse_with_schema(data: str, schema: dict) -> list[dict]:
    rows = parse_csv(data)
    valid_rows = []
    for row in rows:
        if validate_row(row, schema.get("required", [])):
            valid_rows.append(row)
    return valid_rows
