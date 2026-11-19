"""Deterministic quality rules and stratified allocation for Phase 9."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from reproduction.data.pretraining_pool import stable_priority


DEFAULT_SEED = 42
POOL_SIZE = 4_000
SFT_SIZE = 3_200
RL_SIZE = 800
MAX_ROWS = 50
MAX_COLS = 25
MAX_LOGICAL_SLOTS = 800
MAX_PHYSICAL_CELLS = 800
MAX_TARGET_TOKENS = 8_192
COMPLEXITY_ORDER = ("simple", "merged", "complex")


def complexity_bucket(rows: int, cols: int, physical_cells: int, merge_count: int) -> str:
    """Use the same mutually exclusive complexity semantics as Phase 5."""
    if rows >= 20 or cols >= 10 or physical_cells >= 100 or merge_count >= 5:
        return "complex"
    if merge_count > 0:
        return "merged"
    return "simple"


def quality_reasons(sample, parsed: dict) -> list[str]:
    """Return stable rejection reasons that do not mutate or repair a sample."""
    reasons: list[str] = []
    if sample.source != "pubtables":
        reasons.append("wrong_source")
    if not parsed["is_valid"]:
        reasons.append("invalid_html_structure")
        return reasons

    rows = parsed["row_count"]
    cols = parsed["col_count"]
    physical_cells = len(parsed["cells"])
    if rows < 2:
        reasons.append("too_few_rows")
    if cols < 2:
        reasons.append("too_few_columns")
    if rows > MAX_ROWS:
        reasons.append("too_many_rows")
    if cols > MAX_COLS:
        reasons.append("too_many_columns")
    if rows * cols > MAX_LOGICAL_SLOTS:
        reasons.append("too_many_logical_slots")
    if physical_cells > MAX_PHYSICAL_CELLS:
        reasons.append("too_many_physical_cells")
    if not any(not cell["is_empty"] for cell in parsed["cells"]):
        reasons.append("all_cells_empty")

    metadata_rows = sample.metadata.get("rows")
    metadata_cols = sample.metadata.get("cols")
    try:
        if int(metadata_rows) != rows:
            reasons.append("metadata_row_mismatch")
    except (TypeError, ValueError):
        reasons.append("metadata_rows_invalid")
    try:
        if int(metadata_cols) != cols:
            reasons.append("metadata_col_mismatch")
    except (TypeError, ValueError):
        reasons.append("metadata_cols_invalid")

    annotations = sample.metadata.get("cell_annotations")
    if not isinstance(annotations, list) or len(annotations) != physical_cells:
        reasons.append("cell_annotation_count_mismatch")
    return reasons


def proportional_quotas(counts: dict[str, int], target: int) -> dict[str, int]:
    """Allocate an exact target proportionally using deterministic largest remainder."""
    available = sum(counts.get(bucket, 0) for bucket in COMPLEXITY_ORDER)
    if target < 0 or target > available:
        raise ValueError(f"cannot allocate target={target} from available={available}")
    if target == 0:
        return {bucket: 0 for bucket in COMPLEXITY_ORDER}

    raw = {
        bucket: target * counts.get(bucket, 0) / available
        for bucket in COMPLEXITY_ORDER
    }
    quotas = {bucket: math.floor(raw[bucket]) for bucket in COMPLEXITY_ORDER}
    remaining = target - sum(quotas.values())
    ranked = sorted(
        COMPLEXITY_ORDER,
        key=lambda bucket: (-(raw[bucket] - quotas[bucket]), COMPLEXITY_ORDER.index(bucket)),
    )
    for bucket in ranked[:remaining]:
        quotas[bucket] += 1
    if any(quotas[bucket] > counts.get(bucket, 0) for bucket in COMPLEXITY_ORDER):
        raise AssertionError("proportional allocation exceeded a stratum")
    return quotas


def select_stratified(records: Iterable[dict], target: int, seed: int, namespace: str) -> list[dict]:
    records = list(records)
    counts = Counter(record["complexity"] for record in records)
    unknown = set(counts) - set(COMPLEXITY_ORDER)
    if unknown:
        raise ValueError(f"unknown complexity buckets: {sorted(unknown)}")
    quotas = proportional_quotas(dict(counts), target)
    selected: list[dict] = []
    for bucket in COMPLEXITY_ORDER:
        ordered = sorted(
            (record for record in records if record["complexity"] == bucket),
            key=lambda record: (
                stable_priority(seed, f"{namespace}:{bucket}", record["sample"].sample_id),
                record["sample"].sample_id,
            ),
        )
        selected.extend(ordered[: quotas[bucket]])
    return sorted(selected, key=lambda record: record["sample"].sample_id)


def assign_usage(records: list[dict], rl_size: int, seed: int) -> None:
    """Assign exact, complexity-stratified SFT/RL usage in place."""
    counts = Counter(record["complexity"] for record in records)
    rl_quotas = proportional_quotas(dict(counts), rl_size)
    for bucket in COMPLEXITY_ORDER:
        bucket_records = [record for record in records if record["complexity"] == bucket]
        ordered = sorted(
            bucket_records,
            key=lambda record: (
                stable_priority(seed, f"pubtables-usage:{bucket}", record["sample"].sample_id),
                record["sample"].sample_id,
            ),
        )
        rl_ids = {record["sample"].sample_id for record in ordered[: rl_quotas[bucket]]}
        for record in bucket_records:
            record["usage"] = "rl" if record["sample"].sample_id in rl_ids else "sft"
