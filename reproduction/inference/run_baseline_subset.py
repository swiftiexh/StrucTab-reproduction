"""Run resumable HunyuanOCR inference over the fixed baseline subset."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from reproduction.common.paths import project_path
from reproduction.common.reproducibility import DEFAULT_SEED
from reproduction.inference.hunyuan_baseline import (
    DEFAULT_ATTENTION_IMPLEMENTATION,
    DEFAULT_MODEL_PATH,
    HunyuanBaselineEngine,
)


DEFAULT_SUBSET = project_path("reproduction/data/splits/baseline_test.jsonl")
DEFAULT_PREDICTIONS = project_path("reproduction/results/baseline/predictions.jsonl")
DEFAULT_METRICS = project_path("reproduction/results/baseline/metrics.json")
RESUME_CONFIG_KEYS = (
    "model_path",
    "seed",
    "dtype",
    "do_sample",
    "repetition_penalty",
    "max_new_tokens",
    "attention_implementation",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def index_predictions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    indexed = {}
    for record in read_jsonl(path):
        sample_id = record["sample_id"]
        if sample_id in indexed:
            raise ValueError(f"duplicate sample_id in predictions: {sample_id}")
        indexed[sample_id] = record
    return indexed


def resume_config_mismatches(previous: dict, current: dict) -> list[str]:
    return [key for key in RESUME_CONFIG_KEYS if previous.get(key) != current.get(key)]


def validate_resume_config(metrics_path: Path, indexed: dict[str, dict], run_config: dict) -> None:
    if not indexed:
        return
    if not metrics_path.is_file():
        raise ValueError("predictions exist but metrics.json is missing; use --restart")
    with metrics_path.open(encoding="utf-8") as metrics_file:
        previous_config = json.load(metrics_file).get("run_config", {})
    mismatches = resume_config_mismatches(previous_config, run_config)
    if mismatches:
        joined = ", ".join(mismatches)
        raise ValueError(
            f"existing predictions use a different run config ({joined}); use --restart"
        )


def is_cuda_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "accelerator" in message)


def write_predictions_atomic(path: Path, subset: list[dict], indexed: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for sample in subset:
            prediction = indexed.get(sample["sample_id"])
            if prediction is not None:
                output_file.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    temporary_path.replace(path)


def summarize_predictions(subset: list[dict], indexed: dict[str, dict]) -> dict:
    completed = [record for record in indexed.values() if record.get("status") == "ok"]
    failed = [record for record in indexed.values() if record.get("status") == "error"]
    subset_ids = {record["sample_id"] for record in subset}
    completed = [record for record in completed if record["sample_id"] in subset_ids]
    failed = [record for record in failed if record["sample_id"] in subset_ids]
    generation_times = [record["generation_seconds"] for record in completed]
    peak_memories = [
        record["peak_generation_memory_gib"]
        for record in completed
        if record.get("peak_generation_memory_gib") is not None
    ]
    completed_count = len(completed)
    return {
        "metric_scope": "runtime_and_exact_match_only; TEDS/TEDS-S are added in Phase 1.3",
        "subset_size": len(subset),
        "completed": completed_count,
        "failed": len(failed),
        "pending": len(subset) - completed_count,
        "complete_html_count": sum(record["has_complete_table"] for record in completed),
        "exact_match_count": sum(record["exact_match"] for record in completed),
        "exact_match_rate": (
            sum(record["exact_match"] for record in completed) / completed_count
            if completed_count
            else None
        ),
        "suspected_truncation_count": sum(
            record["diagnostics"]["suspected_truncation"] for record in completed
        ),
        "suspected_repetition_count": sum(
            record["diagnostics"]["suspected_repetition"] for record in completed
        ),
        "total_generation_seconds": round(sum(generation_times), 4),
        "mean_generation_seconds": (
            round(sum(generation_times) / completed_count, 4) if completed_count else None
        ),
        "max_peak_generation_memory_gib": (
            round(max(peak_memories), 4) if peak_memories else None
        ),
    }


def write_metrics(path: Path, metrics: dict, run_config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"run_config": run_config, **metrics}
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager"),
        default=DEFAULT_ATTENTION_IMPLEMENTATION,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=None, help="Maximum pending samples this run")
    parser.add_argument("--restart", action="store_true", help="Discard existing progress")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset_path = args.subset.resolve()
    predictions_path = args.predictions.resolve()
    metrics_path = args.metrics.resolve()
    subset = read_jsonl(subset_path)
    if len({record["sample_id"] for record in subset}) != len(subset):
        raise ValueError("subset contains duplicate sample_id values")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")

    run_config = {
        "subset_path": str(subset_path),
        "predictions_path": str(predictions_path),
        "model_path": str(args.model.resolve()),
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "do_sample": False,
        "repetition_penalty": 1.03,
        "max_new_tokens": args.max_new_tokens,
        "attention_implementation": args.attn_implementation,
    }
    indexed = {} if args.restart else index_predictions(predictions_path)
    if args.restart:
        write_predictions_atomic(predictions_path, subset, indexed)
    else:
        validate_resume_config(metrics_path, indexed, run_config)
    completed_ids = {
        sample_id for sample_id, record in indexed.items() if record.get("status") == "ok"
    }
    pending = [record for record in subset if record["sample_id"] not in completed_ids]
    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        metrics = summarize_predictions(subset, indexed)
        write_metrics(metrics_path, metrics, run_config)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    engine = HunyuanBaselineEngine(
        args.model,
        args.device,
        args.dtype,
        args.seed,
        args.attn_implementation,
    )
    started = time.perf_counter()
    for position, sample in enumerate(pending, start=1):
        sample_id = sample["sample_id"]
        fatal_cuda_oom = False
        try:
            inference = engine.infer(
                project_path(sample["image_path"]), sample["prompt"], args.max_new_tokens
            )
            prediction = {
                "status": "ok",
                "sample_id": sample_id,
                "source_index": sample["source_index"],
                "image_path": sample["image_path"],
                "primary_stratum": sample["primary_stratum"],
                "strata": sample["table_stats"]["strata"],
                "reference_html": sample["reference_html"],
                "attention_implementation": args.attn_implementation,
                "exact_match": inference["pred_html"] == sample["reference_html"],
                **inference,
            }
        except Exception as exc:
            prediction = {
                "status": "error",
                "sample_id": sample_id,
                "source_index": sample["source_index"],
                "image_path": sample["image_path"],
                "primary_stratum": sample["primary_stratum"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "attention_implementation": args.attn_implementation,
            }
            # CUDA calls (including empty_cache) may fail after an accelerator
            # OOM. Persist the failed sample, then end this process cleanly.
            fatal_cuda_oom = args.device.startswith("cuda") and is_cuda_oom(exc)

        indexed[sample_id] = prediction
        write_predictions_atomic(predictions_path, subset, indexed)
        metrics = summarize_predictions(subset, indexed)
        write_metrics(metrics_path, metrics, run_config)
        elapsed = time.perf_counter() - started
        print(
            f"[{position}/{len(pending)}] {sample_id} status={prediction['status']} "
            f"completed={metrics['completed']}/{len(subset)} elapsed={elapsed:.1f}s",
            flush=True,
        )
        if fatal_cuda_oom:
            print(
                "CUDA OOM progress was saved. Restart the same command to retry this sample.",
                flush=True,
            )
            raise SystemExit(2)

    print(json.dumps(summarize_predictions(subset, indexed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
