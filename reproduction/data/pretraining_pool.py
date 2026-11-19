"""Deterministic source selection and statistics for the pretraining pool."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


DEFAULT_SEED = 42
SYNTH_TARGET = 10_000
SYNTH_RESERVE = 500
VALIDATION_RATIO = 0.05
MAX_TARGET_TOKENS = 8_192


def stable_priority(seed: int, namespace: str, key: str) -> int:
    payload = f"{seed}:{namespace}:{key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def select_synth_candidates(source: Path, count: int, seed: int = DEFAULT_SEED) -> tuple[list[dict], Counter]:
    """Select the smallest seeded hashes from the source train split in one scan."""
    heap: list[tuple[int, int, str]] = []
    split_counts: Counter[str] = Counter()
    with source.open(encoding="utf-8") as input_file:
        for row_index, line in enumerate(input_file):
            record = json.loads(line)
            split = str(record["split"]).lower()
            split_counts[split] += 1
            if split != "train":
                continue
            source_key = f"train:{record['imgid']}"
            priority = stable_priority(seed, "synthtabnet-selection", source_key)
            entry = (-priority, row_index, source_key)
            if len(heap) < count:
                heapq.heappush(heap, entry)
            elif priority < -heap[0][0]:
                heapq.heapreplace(heap, entry)
    if len(heap) != count:
        raise ValueError(f"only {len(heap)}/{count} SynthTabNet train candidates available")
    selected = [
        {"source": "synthtabnet", "source_key": key, "source_row_index": row, "priority": -neg}
        for neg, row, key in heap
    ]
    selected.sort(key=lambda item: (item["priority"], item["source_row_index"]))
    for rank, item in enumerate(selected):
        item["selection_rank"] = rank
    return selected, split_counts


def select_all_fintabnet(source_dir: Path, seed: int = DEFAULT_SEED) -> list[dict]:
    selected = []
    for shard in sorted(source_dir.glob("*.parquet")):
        rows = pq.ParquetFile(shard).metadata.num_rows
        for row_index in range(rows):
            source_key = f"{shard.name}:{row_index}"
            selected.append(
                {
                    "source": "fintabnet",
                    "source_key": source_key,
                    "source_parquet": shard.name,
                    "source_row_index": row_index,
                    "priority": stable_priority(seed, "fintabnet-selection", source_key),
                }
            )
    selected.sort(key=lambda item: (item["source_parquet"], item["source_row_index"]))
    for rank, item in enumerate(selected):
        item["selection_rank"] = rank
    return selected


def percentile(values: list[int], quantile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(quantile * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def distribution(values: Iterable[int]) -> dict[str, Any]:
    values = list(values)
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def complexity_bucket(row_count: int, col_count: int, physical_cells: int, merge_count: int) -> str:
    if row_count >= 20 or col_count >= 10 or physical_cells >= 100 or merge_count >= 5:
        return "complex"
    if merge_count > 0:
        return "merged"
    return "simple"


def assign_pool_splits(records: list[dict], seed: int, validation_ratio: float) -> None:
    """Assign exact source-stratified validation counts in place."""
    by_source: dict[str, list[dict]] = {}
    for record in records:
        by_source.setdefault(record["sample"].source, []).append(record)
    for source, source_records in by_source.items():
        validation_count = round(len(source_records) * validation_ratio)
        ordered = sorted(
            source_records,
            key=lambda item: stable_priority(
                seed, f"{source}-pool-split", item["sample"].sample_id
            ),
        )
        validation_ids = {item["sample"].sample_id for item in ordered[:validation_count]}
        for item in source_records:
            item["pool_split"] = (
                "validation" if item["sample"].sample_id in validation_ids else "train"
            )
