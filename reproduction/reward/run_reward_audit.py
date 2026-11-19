"""Run four interpretable Uni-TabRL reward cases with the local Anchor path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from reproduction.common.paths import project_path
from reproduction.eval.teds_client import TedsClient
from reproduction.reward.anchor import LocalAnchorScorer
from reproduction.reward.audit_cases import build_audit_cases
from reproduction.reward.pipeline import RewardPipeline
from reproduction.reward.rendering import LocalHtmlRenderer


DEFAULT_POOL = project_path("reproduction/data/high_quality/pubtables_high_quality_pool.jsonl")
DEFAULT_CACHE = project_path("reproduction/data/reward/anchor_reference_cache.jsonl")
DEFAULT_OUTPUT = project_path("reproduction/results/reward/reward_audit.json")
DEFAULT_RENDER_DIR = project_path("reproduction/results/reward/rendered/audit_predictions")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2, allow_nan=False)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--render-dir", type=Path, default=DEFAULT_RENDER_DIR)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18910")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    pool = {row["sample_id"]: row for row in _read_jsonl(args.pool.resolve())}
    cache_rows = _read_jsonl(args.cache.resolve())
    cache = {row["sample_id"]: row for row in cache_rows if row.get("status") == "ok"}
    merged = [row for row in cache_rows if row.get("status") == "ok" and row.get("complexity") == "merged"]
    if not merged:
        raise ValueError("Anchor cache contains no successful merged reference")
    cache_record = merged[0]
    sample_id = cache_record["sample_id"]
    if sample_id not in pool:
        raise ValueError(f"cached sample is absent from pool: {sample_id}")
    reference_html = pool[sample_id]["html"]
    reference_hash = hashlib.sha256(reference_html.encode("utf-8")).hexdigest()
    if reference_hash != cache_record.get("reference_html_sha256"):
        raise ValueError("Anchor cache reference hash does not match the pool")

    teds = TedsClient(args.endpoint, timeout_seconds=60.0, max_retries=1)
    health = teds.health_check()
    if not health["healthy"]:
        raise RuntimeError(f"TEDS service health check failed: {health}")

    cases = build_audit_cases(reference_html)
    started = time.perf_counter()
    results: list[dict] = []
    with LocalHtmlRenderer(args.render_dir.resolve()) as renderer:
        anchor = LocalAnchorScorer(
            renderer,
            teds_client=teds,
            max_new_tokens=args.max_new_tokens,
        )
        pipeline = RewardPipeline(teds_scorer=teds.score, anchor_scorer=anchor)
        for index, case in enumerate(cases, 1):
            case_started = time.perf_counter()
            score = pipeline.compute(
                case["response"],
                case["reference"],
                split="train",
                ref_destylized=cache_record,
            )
            result = {
                "case": case["case"],
                "description": case["description"],
                "status": score["status"],
                "reward": score["reward"],
                "validity_reward": score["validity_reward"],
                "structure_reward": score["structure_reward"],
                "content_reward": score["content_reward"],
                "base_metrics": score["base_metrics"],
                "one_d_probe": score["one_d_probe"],
                "fallbacks": score["fallbacks"],
                "reason": score["reason"],
                "strict_parsing": score["parsing"],
                "anchor": score["content_detail"],
                "seconds": round(time.perf_counter() - case_started, 4),
            }
            results.append(result)
            partial = {
                "status": "running",
                "sample_id": sample_id,
                "reference_html_sha256": reference_hash,
                "weights": [0.4, 0.3, 0.3],
                "variants": {"structure": "1d_probe", "content": "anchor"},
                "teds_health": health,
                "cases": results,
            }
            _write_json_atomic(args.output.resolve(), partial)
            print(
                f"[{index}/{len(cases)}] {result['case']} status={result['status']} "
                f"reward={result['reward']:.6f} validity={result['validity_reward']:.6f} "
                f"structure={result['structure_reward']:.6f} content={result['content_reward']:.6f} "
                f"seconds={result['seconds']}",
                flush=True,
            )

    completed = all(row["status"] == "ok" and not row["fallbacks"] for row in results)
    summary = {
        "status": "ok" if completed else "failed",
        "sample_id": sample_id,
        "reference_html_sha256": reference_hash,
        "anchor_cache_sha256": hashlib.sha256(args.cache.resolve().read_bytes()).hexdigest(),
        "weights": [0.4, 0.3, 0.3],
        "variants": {"structure": "1d_probe", "content": "anchor"},
        "teds_health": health,
        "cases": results,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    _write_json_atomic(args.output.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not completed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

