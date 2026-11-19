"""Selection and persistence helpers for the Phase 10 Anchor audit cache."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

from reproduction.data.pretraining_pool import stable_priority


COMPLEXITY_ORDER = ("simple", "merged", "complex")
DEFAULT_PER_COMPLEXITY = 2
DEFAULT_MAX_TARGET_TOKENS = 1024


def select_anchor_audit_records(
    pool_records: Iterable[dict],
    rl_ids: set[str],
    *,
    seed: int = 42,
    per_complexity: int = DEFAULT_PER_COMPLEXITY,
    max_target_tokens: int = DEFAULT_MAX_TARGET_TOKENS,
) -> list[dict]:
    if per_complexity < 1 or max_target_tokens < 1:
        raise ValueError("selection limits must be positive")
    candidates = []
    for record in pool_records:
        if record["sample_id"] not in rl_ids:
            continue
        phase9 = record["metadata"]["phase9"]
        if phase9["stage2_target_token_count"] > max_target_tokens:
            continue
        candidates.append(record)
    counts = Counter(record["metadata"]["phase9"]["complexity"] for record in candidates)
    selected = []
    for complexity in COMPLEXITY_ORDER:
        ordered = sorted(
            (record for record in candidates if record["metadata"]["phase9"]["complexity"] == complexity),
            key=lambda record: (
                stable_priority(seed, f"anchor-audit:{complexity}", record["sample_id"]),
                record["sample_id"],
            ),
        )
        if len(ordered) < per_complexity:
            raise ValueError(
                f"anchor audit has only {len(ordered)}/{per_complexity} eligible {complexity} records"
            )
        selected.extend(ordered[:per_complexity])
    selected.sort(key=lambda record: (COMPLEXITY_ORDER.index(record["metadata"]["phase9"]["complexity"]), record["sample_id"]))
    return selected


def selection_manifest(records: Iterable[dict], seed: int, max_target_tokens: int) -> list[dict]:
    rows = []
    for record in records:
        phase9 = record["metadata"]["phase9"]
        rows.append(
            {
                "sample_id": record["sample_id"],
                "complexity": phase9["complexity"],
                "image_path": record["image_path"],
                "row_count": phase9["row_count"],
                "col_count": phase9["col_count"],
                "merge_count": phase9["merge_count"],
                "stage2_target_token_count": phase9["stage2_target_token_count"],
                "reference_html_sha256": hashlib.sha256(record["html"].encode("utf-8")).hexdigest(),
                "seed": seed,
                "selection_target_token_limit": max_target_tokens,
            }
        )
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()
