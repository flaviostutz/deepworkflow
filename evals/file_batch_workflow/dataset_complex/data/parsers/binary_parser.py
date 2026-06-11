"""Binary parser with hard-coded little-endian assumption and no bounds checking."""

import struct


HEADER_FORMAT = "<IHH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

RECORD_FORMAT = "<Id"
RECORD_SIZE = struct.calcsize(RECORD_FORMAT)


def parse_header(data: bytes) -> dict:
    magic, version, record_count = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
    return {"magic": magic, "version": version, "record_count": record_count}


def parse_record(data: bytes, offset: int) -> dict:
    id_, value = struct.unpack(RECORD_FORMAT, data[offset:offset + RECORD_SIZE])
    return {"id": id_, "value": value}


def parse_binary_file(filepath: str) -> list[dict]:
    with open(filepath, "rb") as f:
        data = f.read()

    header = parse_header(data)
    records = []
    offset = HEADER_SIZE
    for _ in range(header["record_count"]):
        record = parse_record(data, offset)
        records.append(record)
        offset += RECORD_SIZE
    return records


def write_binary_file(filepath: str, records: list[dict]) -> None:
    with open(filepath, "wb") as f:
        header = struct.pack(HEADER_FORMAT, 0xDEADBEEF, 1, len(records))
        f.write(header)
        for rec in records:
            f.write(struct.pack(RECORD_FORMAT, rec["id"], rec["value"]))


def validate_magic(data: bytes) -> bool:
    magic, _, _ = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
    return magic == 0xDEADBEEF
