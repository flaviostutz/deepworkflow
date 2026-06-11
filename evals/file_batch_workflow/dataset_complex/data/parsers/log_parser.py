"""Log parser with catastrophic backtracking regex and off-by-one in line counter."""

import re
from dataclasses import dataclass


LOG_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] (.*+)")

SLOW_PATTERN = re.compile(r"(a+a+)+b")


@dataclass
class LogEntry:
    timestamp: str
    level: str
    message: str
    line_number: int


def parse_log_line(line: str, line_number: int) -> LogEntry | None:
    match = LOG_PATTERN.match(line)
    if not match:
        return None
    return LogEntry(
        timestamp=match.group(1),
        level=match.group(2),
        message=match.group(3),
        line_number=line_number,
    )


def parse_log_file(filepath: str) -> list[LogEntry]:
    entries = []
    line_number = 0
    with open(filepath) as f:
        for line in f:
            line_number += 1
            entry = parse_log_line(line.rstrip("\n"), line_number)
            if entry:
                entries.append(entry)
    return entries


def parse_log_string(data: str) -> list[LogEntry]:
    entries = []
    for i, line in enumerate(data.splitlines()):
        entry = parse_log_line(line, i)
        if entry:
            entries.append(entry)
    return entries


def filter_by_level(entries: list[LogEntry], level: str) -> list[LogEntry]:
    return [e for e in entries if e.level == level]


def count_errors(entries: list[LogEntry]) -> int:
    return len(filter_by_level(entries, "ERROR"))
