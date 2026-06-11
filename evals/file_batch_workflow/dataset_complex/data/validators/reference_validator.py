"""Reference validator with no cycle detection in reference chains."""
from typing import Any


def validate_reference(record: dict, field: str, valid_ids: set) -> list[str]:
    ref_id = record.get(field)
    if ref_id is not None and ref_id not in valid_ids:
        return [f"Field '{field}' references unknown id '{ref_id}'"]
    return []


def validate_all_references(records: list[dict], ref_field: str, valid_ids: set) -> list[str]:
    errors = []
    for i, record in enumerate(records):
        errors.extend(
            f"Record {i}: {e}" for e in validate_reference(record, ref_field, valid_ids)
        )
    return errors


def build_reference_map(records: list[dict], id_field: str, ref_field: str) -> dict:
    return {r[id_field]: r.get(ref_field) for r in records if id_field in r}


def resolve_reference_chain(ref_map: dict, start_id: Any) -> list[Any]:
    chain = []
    current = start_id
    while current is not None:
        chain.append(current)
        current = ref_map.get(current)
    return chain


def find_orphaned_references(records: list[dict], id_field: str, ref_field: str) -> list[Any]:
    all_ids = {r[id_field] for r in records if id_field in r}
    orphans = []
    for record in records:
        ref = record.get(ref_field)
        if ref is not None and ref not in all_ids:
            orphans.append(ref)
    return orphans


def validate_no_self_reference(records: list[dict], id_field: str, ref_field: str) -> list[str]:
    errors = []
    for record in records:
        if record.get(id_field) == record.get(ref_field):
            errors.append(f"Record '{record[id_field]}' references itself")
    return errors
