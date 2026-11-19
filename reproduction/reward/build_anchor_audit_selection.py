"""Select six deterministic RL references for Phase 10 Anchor validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reproduction.common.paths import project_path
from reproduction.reward.anchor_cache import (
    DEFAULT_MAX_TARGET_TOKENS,
    DEFAULT_PER_COMPLEXITY,
    select_anchor_audit_records,
    selection_manifest,
    write_jsonl_atomic,
)


DEFAULT_POOL = project_path("reproduction/data/high_quality/pubtables_high_quality_pool.jsonl")
DEFAULT_RL_SPLIT = project_path("reproduction/data/splits/rl.jsonl")
DEFAULT_OUTPUT = project_path("reproduction/data/reward/anchor_audit_selection.jsonl")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--rl-split", type=Path, default=DEFAULT_RL_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-complexity", type=int, default=DEFAULT_PER_COMPLEXITY)
    parser.add_argument("--max-target-tokens", type=int, default=DEFAULT_MAX_TARGET_TOKENS)
    args = parser.parse_args()
    pool = _read_jsonl(args.pool.resolve())
    rl_ids = {row["sample_id"] for row in _read_jsonl(args.rl_split.resolve())}
    selected = select_anchor_audit_records(
        pool, rl_ids, seed=args.seed, per_complexity=args.per_complexity,
        max_target_tokens=args.max_target_tokens,
    )
    rows = selection_manifest(selected, args.seed, args.max_target_tokens)
    digest = write_jsonl_atomic(args.output.resolve(), rows)
    print(json.dumps({"records": len(rows), "selection": rows, "sha256": digest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
