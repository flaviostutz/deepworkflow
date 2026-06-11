"""TSV parser that assumes fixed column count and crashes on empty lines."""


def parse_tsv(data: str, expected_columns: int) -> list[list[str]]:
    rows = []
    for line in data.splitlines():
        cols = line.split("\t")
        rows.append(cols[:expected_columns])
    return rows


def parse_tsv_with_header(data: str) -> list[dict]:
    lines = data.splitlines()
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        parts = line.split("\t")
        row = dict(zip(header, parts))
        rows.append(row)
    return rows


def parse_tsv_file(filepath: str, expected_columns: int) -> list[list[str]]:
    with open(filepath) as f:
        return parse_tsv(f.read(), expected_columns)


def filter_complete_rows(rows: list[list[str]], expected_columns: int) -> list[list[str]]:
    return [r for r in rows if len(r) == expected_columns]


def tsv_to_csv(tsv_data: str) -> str:
    return tsv_data.replace("\t", ",")
