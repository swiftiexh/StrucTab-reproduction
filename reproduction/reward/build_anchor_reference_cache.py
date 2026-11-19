"""Build a resumable HunyuanOCR destylized-reference cache for the audit set."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from reproduction.common.paths import project_path
from reproduction.reward.anchor import LocalAnchorScorer
from reproduction.reward.anchor_cache import write_jsonl_atomic
from reproduction.reward.rendering import LocalHtmlRenderer


DEFAULT_POOL = project_path("reproduction/data/high_quality/pubtables_high_quality_pool.jsonl")
DEFAULT_SELECTION = project_path("reproduction/data/reward/anchor_audit_selection.jsonl")
DEFAULT_CACHE = project_path("reproduction/data/reward/anchor_reference_cache.jsonl")
DEFAULT_SUMMARY = project_path("reproduction/data/reward/anchor_reference_cache_summary.json")
DEFAULT_RENDER_DIR = project_path("reproduction/results/reward/rendered/anchor_references")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--render-dir", type=Path, default=DEFAULT_RENDER_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    pool = {row["sample_id"]: row for row in _read_jsonl(args.pool.resolve())}
    selection = _read_jsonl(args.selection.resolve())
    if len({row["sample_id"] for row in selection}) != len(selection):
        raise ValueError("selection contains duplicate sample IDs")
    missing = [row["sample_id"] for row in selection if row["sample_id"] not in pool]
    if missing:
        raise ValueError(f"selection IDs absent from pool: {missing}")

    cache_path = args.cache.resolve()
    indexed = {} if args.restart or not cache_path.exists() else {
        row["sample_id"]: row for row in _read_jsonl(cache_path)
    }
    pending = [row for row in selection if indexed.get(row["sample_id"], {}).get("status") != "ok"]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        pending = pending[: args.limit]

    started = time.perf_counter()
    if pending:
        with LocalHtmlRenderer(args.render_dir.resolve()) as renderer:
            anchor = LocalAnchorScorer(renderer, max_new_tokens=args.max_new_tokens)
            for position, selected in enumerate(pending, 1):
                sample_id = selected["sample_id"]
                reference_html = pool[sample_id]["html"]
                item_started = time.perf_counter()
                try:
                    parsed = anchor.parse_reference(reference_html)
                    destylized = parsed["inference"]["pred_html"]
                    if not destylized:
                        raise ValueError("Anchor OCR did not produce a complete table")
                    record = {
                        "status": "ok",
                        "sample_id": sample_id,
                        "complexity": selected["complexity"],
                        "should_de_stylize": True,
                        "de_stylize_ref": destylized,
                        "reference_html_sha256": selected["reference_html_sha256"],
                        "render": parsed["render"],
                        "inference": parsed["inference"],
                        "seconds": round(time.perf_counter() - item_started, 4),
                    }
                except Exception as exc:
                    record = {
                        "status": "failed",
                        "sample_id": sample_id,
                        "complexity": selected["complexity"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "seconds": round(time.perf_counter() - item_started, 4),
                    }
                indexed[sample_id] = record
                write_jsonl_atomic(
                    cache_path,
                    [indexed[row["sample_id"]] for row in selection if row["sample_id"] in indexed],
                )
                print(
                    f"[{position}/{len(pending)}] {sample_id} status={record['status']} "
                    f"seconds={record['seconds']}", flush=True,
                )
                if record["status"] == "failed" and "cuda" in record.get("error", "").lower():
                    break

    ordered = [indexed[row["sample_id"]] for row in selection if row["sample_id"] in indexed]
    completed = [row for row in ordered if row["status"] == "ok"]
    failed = [row for row in ordered if row["status"] == "failed"]
    summary = {
        "status": "ok" if len(completed) == len(selection) else "incomplete",
        "selection_records": len(selection),
        "completed": len(completed),
        "failed": len(failed),
        "pending": len(selection) - len(ordered),
        "should_de_stylize_policy": "true for every explicitly selected Phase 10 audit reference",
        "max_new_tokens": args.max_new_tokens,
        "cache": str(cache_path),
        "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest() if cache_path.exists() else None,
        "elapsed_seconds_this_run": round(time.perf_counter() - started, 4),
    }
    _write_json(args.summary.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
