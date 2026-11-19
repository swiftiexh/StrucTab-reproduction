"""Score the completed baseline predictions with TEDS and TEDS-S."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from reproduction.common.paths import project_path
from reproduction.eval.teds_client import DEFAULT_ENDPOINT, TedsClient


DEFAULT_SUBSET = project_path("reproduction/data/splits/baseline_test.jsonl")
DEFAULT_PREDICTIONS = project_path("reproduction/results/baseline/predictions.jsonl")
DEFAULT_SCORES = project_path("reproduction/results/baseline/teds_scores.jsonl")
DEFAULT_METRICS = project_path("reproduction/results/baseline/metrics.json")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def index_unique(records: list[dict], label: str) -> dict[str, dict]:
    indexed = {}
    for record in records:
        sample_id = record["sample_id"]
        if sample_id in indexed:
            raise ValueError(f"duplicate {label} sample_id: {sample_id}")
        indexed[sample_id] = record
    return indexed


def write_jsonl_atomic(path: Path, ordered_ids: list[str], indexed: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        for sample_id in ordered_ids:
            if sample_id in indexed:
                output_file.write(json.dumps(indexed[sample_id], ensure_ascii=False) + "\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    temporary.replace(path)


def validate_completed_predictions(subset: list[dict], predictions: list[dict]) -> dict[str, dict]:
    subset_ids = {row["sample_id"] for row in subset}
    indexed = index_unique(predictions, "prediction")
    for sample_id, prediction in indexed.items():
        if sample_id not in subset_ids:
            raise ValueError(f"prediction is not in fixed subset: {sample_id}")
    successful = {key for key, value in indexed.items() if value.get("status") == "ok"}
    missing = subset_ids - successful
    if missing:
        raise RuntimeError(
            f"baseline inference is incomplete: {len(successful)}/{len(subset)} successful, "
            f"{len(missing)} still missing or failed"
        )
    attention_backends = {indexed[sample_id].get("attention_implementation") for sample_id in successful}
    if attention_backends != {"sdpa"}:
        raise ValueError(
            f"baseline predictions must all use the unified sdpa backend: {attention_backends}"
        )
    return indexed


def summarize_scores(subset: list[dict], indexed: dict[str, dict]) -> dict:
    ordered = [indexed.get(row["sample_id"]) for row in subset]
    completed = [row for row in ordered if row is not None and row.get("error") is None]
    service_errors = [row for row in ordered if row is not None and row.get("error") is not None]
    count = len(subset)
    by_primary_stratum = {}
    primary_names = sorted({row.get("primary_stratum", "unspecified") for row in subset})
    for primary in primary_names:
        group = [
            indexed.get(row["sample_id"])
            for row in subset
            if row.get("primary_stratum", "unspecified") == primary
        ]
        successful_group = [row for row in group if row is not None and row.get("error") is None]
        group_count = len(group)
        by_primary_stratum[primary] = {
            "sample_count": group_count,
            "scored_count": len(successful_group),
            "teds": (
                sum(row["teds"] for row in successful_group) / group_count
                if group_count
                else 0.0
            ),
            "teds_s": (
                sum(row["teds_s"] for row in successful_group) / group_count
                if group_count
                else 0.0
            ),
        }

    return {
        "metric_scope": "fixed baseline_test.jsonl",
        "sample_count": count,
        "scored_count": len(completed),
        "pending_count": count - len(completed),
        "service_error_count": len(service_errors),
        "empty_prediction_count": sum(
            row.get("diagnostic") == "empty_prediction" for row in completed
        ),
        "teds": sum(row["teds"] for row in completed) / count if count else 0.0,
        "teds_s": sum(row["teds_s"] for row in completed) / count if count else 0.0,
        "by_primary_stratum": by_primary_stratum,
    }


def build_tracked_failure_cases(
    subset: list[dict], predictions: dict[str, dict], scores: dict[str, dict]
) -> list[dict]:
    """Keep baseline repetition cases as stable cross-stage comparison targets."""
    tracked = []
    for sample in subset:
        sample_id = sample["sample_id"]
        prediction = predictions[sample_id]
        diagnostics = prediction.get("diagnostics", {})
        if not diagnostics.get("suspected_repetition"):
            continue
        score = scores[sample_id]
        tracked.append(
            {
                "sample_id": sample_id,
                "source_index": sample["source_index"],
                "image_path": sample["image_path"],
                "primary_stratum": sample["primary_stratum"],
                "reason": "multiple_complete_table_blocks",
                "table_open_count": diagnostics.get("table_open_count"),
                "table_close_count": diagnostics.get("table_close_count"),
                "output_tokens": prediction.get("output_tokens"),
                "teds": score["teds"],
                "teds_s": score["teds_s"],
            }
        )
    return tracked


def update_metrics_atomic(path: Path, evaluation: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"baseline runtime metrics missing: {path}")
    with path.open(encoding="utf-8") as metrics_file:
        payload = json.load(metrics_file)
    payload["evaluation"] = evaluation
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset = read_jsonl(args.subset.resolve())
    predictions = validate_completed_predictions(subset, read_jsonl(args.predictions.resolve()))
    ordered_ids = [row["sample_id"] for row in subset]
    scores_path = args.scores.resolve()
    scores = (
        {}
        if args.restart or not scores_path.exists()
        else index_unique(read_jsonl(scores_path), "TEDS score")
    )
    client = TedsClient(
        args.endpoint,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    )
    health = client.health_check()
    if not health["healthy"]:
        raise RuntimeError(f"TEDS service health check failed: {health}")

    for position, sample in enumerate(subset, start=1):
        sample_id = sample["sample_id"]
        if sample_id in scores and scores[sample_id].get("error") is None:
            continue
        prediction = predictions[sample_id]
        result = client.score(prediction.get("pred_html", ""), sample["reference_html"])
        scores[sample_id] = {"sample_id": sample_id, **result}
        write_jsonl_atomic(scores_path, ordered_ids, scores)
        print(
            f"[{position}/{len(subset)}] {sample_id} "
            f"TEDS={result['teds']:.6f} TEDS-S={result['teds_s']:.6f}",
            flush=True,
        )

    summary = summarize_scores(subset, scores)
    if summary["pending_count"] or summary["service_error_count"]:
        raise RuntimeError(f"TEDS scoring incomplete: {summary}")
    summary["tracked_failure_cases"] = build_tracked_failure_cases(
        subset, predictions, scores
    )
    update_metrics_atomic(args.metrics.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
