"""Build balanced Stage 1 tasks and Stage 2 sequential targets atomically."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from reproduction.common.data_policy import assert_data_usage
from reproduction.common.paths import project_path
from reproduction.data.conversion import project_relative, validate_image
from reproduction.data.schema import NormalizedTableSample
from reproduction.data.training_examples import build_stage1_examples, build_stage2_example


DEFAULT_INPUTS = [
    project_path("reproduction/data/normalized/synthtabnet_smoke.jsonl"),
    project_path("reproduction/data/normalized/fintabnet_smoke.jsonl"),
]
DEFAULT_OUTPUT_DIR = project_path("reproduction/data/training")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(inputs: list[Path], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = output_dir / "stage1_smoke.jsonl"
    stage2_path = output_dir / "stage2_smoke.jsonl"
    failures_path = output_dir / "stage_data_failures.jsonl"
    summary_path = output_dir / "stage_data_summary.json"
    stage1_tmp = stage1_path.with_suffix(".jsonl.tmp")
    stage2_tmp = stage2_path.with_suffix(".jsonl.tmp")
    failures_tmp = failures_path.with_suffix(".jsonl.tmp")

    base_ids: set[str] = set()
    derived_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    scanned = 0

    with stage1_tmp.open("w", encoding="utf-8") as stage1_file, stage2_tmp.open(
        "w", encoding="utf-8"
    ) as stage2_file, failures_tmp.open("w", encoding="utf-8") as failure_file:
        for input_path in inputs:
            with input_path.open(encoding="utf-8") as input_file:
                for line_number, line in enumerate(input_file, 1):
                    scanned += 1
                    base_id = f"{input_path.name}:{line_number}"
                    try:
                        base = NormalizedTableSample.from_dict(json.loads(line))
                        base_id = base.sample_id
                        assert_data_usage(base.source, "pretrain")
                        if base.sample_id in base_ids:
                            raise ValueError(f"duplicate base_sample_id: {base.sample_id}")
                        validate_image(project_path(base.image_path))
                        stage1 = build_stage1_examples(base)
                        stage2 = build_stage2_example(base)
                        examples = [*stage1, stage2]
                        if any(example.sample_id in derived_ids for example in examples):
                            raise ValueError("duplicate derived sample ID")
                        for example in stage1:
                            stage1_file.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
                            task_counts[example.task_type] += 1
                        stage2_file.write(json.dumps(stage2.to_dict(), ensure_ascii=False) + "\n")
                        task_counts[stage2.task_type] += 1
                        base_ids.add(base.sample_id)
                        derived_ids.update(example.sample_id for example in examples)
                        source_counts[base.source] += 1
                    except Exception as exc:
                        reason = type(exc).__name__
                        failures[reason] += 1
                        failure_file.write(
                            json.dumps(
                                {
                                    "input": project_relative(input_path),
                                    "line_number": line_number,
                                    "base_sample_id": base_id,
                                    "reason": reason,
                                    "message": str(exc),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
        for output_file in (stage1_file, stage2_file, failure_file):
            output_file.flush()
            os.fsync(output_file.fileno())

    stage1_tmp.replace(stage1_path)
    stage2_tmp.replace(stage2_path)
    if failures:
        failures_tmp.replace(failures_path)
    else:
        failures_tmp.unlink(missing_ok=True)
        failures_path.unlink(missing_ok=True)

    summary = {
        "inputs": [project_relative(path) for path in inputs],
        "base_records_scanned": scanned,
        "base_records_succeeded": len(base_ids),
        "base_records_failed": sum(failures.values()),
        "failure_reasons": dict(sorted(failures.items())),
        "source_base_counts": dict(sorted(source_counts.items())),
        "stage1_records": sum(task_counts[task] for task in ("row_col", "merge", "html")),
        "stage1_task_counts": {
            task: task_counts[task] for task in ("row_col", "merge", "html")
        },
        "stage2_records": task_counts["sequential"],
        "unique_derived_ids": len(derived_ids),
        "outputs": {
            "stage1": project_relative(stage1_path),
            "stage2": project_relative(stage2_path),
        },
        "sha256": {"stage1": _sha256(stage1_path), "stage2": _sha256(stage2_path)},
    }
    summary_tmp = summary_path.with_suffix(".json.tmp")
    summary_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_tmp.replace(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = run([path.resolve() for path in args.inputs], args.output_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
