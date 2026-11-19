"""Build the Phase 9 PubTables high-quality pool and SFT/RL data manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from reproduction.common.data_policy import assert_data_usage
from reproduction.common.paths import project_path
from reproduction.data.conversion import project_relative, validate_image
from reproduction.data.pretraining_pool import distribution, stable_priority
from reproduction.data.pubtables_quality import (
    DEFAULT_SEED,
    MAX_TARGET_TOKENS,
    POOL_SIZE,
    RL_SIZE,
    SFT_SIZE,
    assign_usage,
    complexity_bucket,
    quality_reasons,
    select_stratified,
)
from reproduction.data.schema import NormalizedTableSample
from reproduction.data.training_examples import TrainingExample, build_stage2_example, validate_stage2_semantics
from reproduction.structure.parser import parse_html_structure


DEFAULT_INPUT = project_path("reproduction/data/normalized/pubtables_all.jsonl")
DEFAULT_CONVERSION_FAILURES = project_path(
    "reproduction/data/normalized/pubtables_all_failures.jsonl"
)
DEFAULT_MODEL = project_path("assets/models/HunyuanOCR/v1.0")
DEFAULT_OUTPUT_DIR = project_path("reproduction/data/high_quality")
DEFAULT_SFT_SPLIT = project_path("reproduction/data/splits/sft.jsonl")
DEFAULT_RL_SPLIT = project_path("reproduction/data/splits/rl.jsonl")
DEFAULT_SFT_TRAINING = project_path("reproduction/data/training/sft.jsonl")
DEFAULT_EXPECTED_INPUT_RECORDS = 5_822


def _portable_path(path: Path) -> str:
    """Prefer project-relative paths while keeping custom external outputs usable."""
    try:
        return project_relative(path)
    except ValueError:
        return str(path.resolve())


def _write_jsonl(path: Path, rows) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _expanded_sft_example(record: dict) -> TrainingExample:
    base = record["sample"]
    example = build_stage2_example(base)
    metadata = dict(example.metadata)
    metadata["phase9"] = {
        "usage": "sft",
        "complexity": record["complexity"],
        "html_token_count": record["html_token_count"],
        "stage2_target_token_count": record["stage2_target_token_count"],
        "selection_priority": record["selection_priority"],
    }
    return TrainingExample(
        sample_id=example.sample_id,
        base_sample_id=example.base_sample_id,
        source=example.source,
        split="sft",
        image_path=example.image_path,
        stage=example.stage,
        task_type=example.task_type,
        prompt=example.prompt,
        target=example.target,
        metadata=metadata,
    )


def run(
    input_path: Path,
    conversion_failures_path: Path,
    model_path: Path,
    output_dir: Path,
    sft_split_path: Path,
    rl_split_path: Path,
    sft_training_path: Path,
    pool_size: int = POOL_SIZE,
    sft_size: int = SFT_SIZE,
    rl_size: int = RL_SIZE,
    seed: int = DEFAULT_SEED,
    max_target_tokens: int = MAX_TARGET_TOKENS,
    expected_input_records: int = DEFAULT_EXPECTED_INPUT_RECORDS,
) -> dict:
    if sft_size + rl_size != pool_size:
        raise ValueError("sft_size + rl_size must equal pool_size")
    assert_data_usage("pubtables", "sft")
    assert_data_usage("pubtables", "rl")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    eligible: list[dict] = []
    rejected: list[dict] = []
    conversion_failure_count = 0
    failed_source_keys: set[str] = set()
    if conversion_failures_path.exists():
        with conversion_failures_path.open(encoding="utf-8") as failure_file:
            for line_number, line in enumerate(failure_file, 1):
                if not line.strip():
                    continue
                failure = json.loads(line)
                source_key = str(failure.get("source_key", ""))
                if not source_key or source_key in failed_source_keys:
                    raise ValueError(
                        f"invalid or duplicate conversion failure source_key at line {line_number}"
                    )
                failed_source_keys.add(source_key)
                conversion_failure_count += 1
                rejected.append(
                    {
                        "sample_id": None,
                        "source_key": source_key,
                        "reasons": ["conversion_failure"],
                        "conversion_error_type": failure.get("reason"),
                        "conversion_error": failure.get("message"),
                    }
                )
    seen_ids: set[str] = set()
    seen_source_keys: set[str] = set()
    with input_path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            sample = NormalizedTableSample.from_dict(json.loads(line))
            if sample.sample_id in seen_ids:
                raise ValueError(f"duplicate sample ID at line {line_number}: {sample.sample_id}")
            source_key = str(sample.metadata.get("source_key", ""))
            if not source_key or source_key in seen_source_keys or source_key in failed_source_keys:
                raise ValueError(f"invalid, duplicate, or failed source key: {source_key!r}")
            seen_ids.add(sample.sample_id)
            seen_source_keys.add(source_key)
            parsed = parse_html_structure(sample.html)
            reasons = quality_reasons(sample, parsed)
            try:
                validate_image(project_path(sample.image_path))
            except Exception as exc:
                reasons.append("invalid_or_missing_image")
                image_error = str(exc)
            else:
                image_error = None
            if reasons:
                rejected.append(
                    {
                        "sample_id": sample.sample_id,
                        "source_key": sample.metadata.get("source_key"),
                        "reasons": reasons,
                        **({"image_error": image_error} if image_error else {}),
                    }
                )
                continue
            stage2 = build_stage2_example(sample)
            semantics = validate_stage2_semantics(stage2.target)
            if not semantics["is_valid"]:
                rejected.append(
                    {
                        "sample_id": sample.sample_id,
                        "source_key": sample.metadata.get("source_key"),
                        "reasons": ["invalid_stage2_semantics"],
                        "errors": semantics["errors"],
                    }
                )
                continue
            html_tokens = len(tokenizer(sample.html, add_special_tokens=False)["input_ids"])
            target_tokens = len(tokenizer(stage2.target, add_special_tokens=False)["input_ids"])
            if target_tokens > max_target_tokens:
                rejected.append(
                    {
                        "sample_id": sample.sample_id,
                        "source_key": sample.metadata.get("source_key"),
                        "reasons": ["stage2_target_too_long"],
                        "stage2_target_token_count": target_tokens,
                        "target_token_limit": max_target_tokens,
                    }
                )
                continue
            merge_count = len(parsed["merged_cells"])
            eligible.append(
                {
                    "sample": sample,
                    "parsed": parsed,
                    "complexity": complexity_bucket(
                        parsed["row_count"], parsed["col_count"], len(parsed["cells"]), merge_count
                    ),
                    "html_token_count": html_tokens,
                    "stage2_target_token_count": target_tokens,
                    "selection_priority": stable_priority(
                        seed, "pubtables-high-quality", sample.sample_id
                    ),
                }
            )

    accounted_records = len(seen_ids) + conversion_failure_count
    if expected_input_records > 0 and accounted_records != expected_input_records:
        raise ValueError(
            f"input accounts for {accounted_records}/{expected_input_records} PubTables records "
            f"({len(seen_ids)} converted + {conversion_failure_count} conversion failures); "
            "resume full conversion before building the pool"
        )

    selected = select_stratified(eligible, pool_size, seed, "pubtables-high-quality-selection")
    assign_usage(selected, rl_size, seed)
    usage_counts = Counter(record["usage"] for record in selected)
    if usage_counts != {"sft": sft_size, "rl": rl_size}:
        raise AssertionError(f"unexpected usage counts: {dict(usage_counts)}")

    pool_rows = []
    split_rows = {"sft": [], "rl": []}
    for record in selected:
        sample = record["sample"]
        parsed = record["parsed"]
        phase9 = {
            "seed": seed,
            "usage": record["usage"],
            "complexity": record["complexity"],
            "row_count": parsed["row_count"],
            "col_count": parsed["col_count"],
            "physical_cell_count": len(parsed["cells"]),
            "merge_count": len(parsed["merged_cells"]),
            "html_token_count": record["html_token_count"],
            "stage2_target_token_count": record["stage2_target_token_count"],
            "selection_priority": record["selection_priority"],
        }
        metadata = dict(sample.metadata)
        metadata["phase9"] = phase9
        pooled = NormalizedTableSample(
            sample_id=sample.sample_id,
            source=sample.source,
            split=record["usage"],
            image_path=sample.image_path,
            html=sample.html,
            metadata=metadata,
        )
        pool_rows.append(pooled.to_dict())
        split_rows[record["usage"]].append(
            {
                "sample_id": sample.sample_id,
                "source": sample.source,
                "source_key": sample.metadata["source_key"],
                "usage": record["usage"],
                "complexity": record["complexity"],
                "image_path": sample.image_path,
                "selection_priority": record["selection_priority"],
            }
        )

    pool_rows.sort(key=lambda row: row["sample_id"])
    for rows in split_rows.values():
        rows.sort(key=lambda row: row["sample_id"])
    sft_records = [record for record in selected if record["usage"] == "sft"]
    sft_examples = sorted(
        (_expanded_sft_example(record).to_dict() for record in sft_records),
        key=lambda row: row["sample_id"],
    )
    sft_ids = {row["sample_id"] for row in split_rows["sft"]}
    rl_ids = {row["sample_id"] for row in split_rows["rl"]}
    if not sft_ids.isdisjoint(rl_ids):
        raise AssertionError("SFT and RL base sample IDs overlap")
    if {row["base_sample_id"] for row in sft_examples} != sft_ids:
        raise AssertionError("SFT examples do not exactly cover the SFT split")

    pool_path = output_dir / "pubtables_high_quality_pool.jsonl"
    rejected_path = output_dir / "pubtables_high_quality_rejected.jsonl"
    summary_path = output_dir / "pubtables_high_quality_summary.json"
    hashes = {
        "pool": _write_jsonl(pool_path, pool_rows),
        "sft_split": _write_jsonl(sft_split_path, split_rows["sft"]),
        "rl_split": _write_jsonl(rl_split_path, split_rows["rl"]),
        "sft_training": _write_jsonl(sft_training_path, sft_examples),
        "rejected": _write_jsonl(rejected_path, rejected),
    }
    complexity_counts = Counter(record["complexity"] for record in selected)
    usage_complexity = Counter(
        f"{record['usage']}/{record['complexity']}" for record in selected
    )
    rejection_counts = Counter(
        reason for row in rejected for reason in row["reasons"]
    )
    summary = {
        "status": "ok",
        "seed": seed,
        "input": _portable_path(input_path),
        "conversion_failures_input": _portable_path(conversion_failures_path),
        "source_records_accounted": accounted_records,
        "converted_input_records": len(seen_ids),
        "conversion_failure_records": conversion_failure_count,
        "eligible_before_selection": len(eligible),
        "rejected_before_selection": len(rejected),
        "pool_records": len(selected),
        "usage_counts": dict(sorted(usage_counts.items())),
        "sft_rl_disjoint": True,
        "complexity_definition": {
            "complex": "rows>=20 or cols>=10 or physical_cells>=100 or merges>=5",
            "merged": "not complex and merges>0",
            "simple": "not complex and merges=0",
        },
        "complexity_counts": dict(sorted(complexity_counts.items())),
        "usage_complexity_counts": dict(sorted(usage_complexity.items())),
        "quality_limits": {
            "rows": "2..50",
            "cols": "2..25",
            "logical_slots_max": 800,
            "physical_cells_max": 800,
            "stage2_target_tokens_max": max_target_tokens,
            "target_policy": "exclude; never truncate",
        },
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "distributions": {
            "row_count": distribution(record["parsed"]["row_count"] for record in selected),
            "col_count": distribution(record["parsed"]["col_count"] for record in selected),
            "physical_cell_count": distribution(len(record["parsed"]["cells"]) for record in selected),
            "merge_count": distribution(len(record["parsed"]["merged_cells"]) for record in selected),
            "html_token_count": distribution(record["html_token_count"] for record in selected),
            "stage2_target_token_count": distribution(record["stage2_target_token_count"] for record in selected),
        },
        "outputs": {
            "pool": _portable_path(pool_path),
            "rejected": _portable_path(rejected_path),
            "sft_split": _portable_path(sft_split_path),
            "rl_split": _portable_path(rl_split_path),
            "sft_training": _portable_path(sft_training_path),
        },
        "hashes": hashes,
        "training_status": "not_run_by_project_decision",
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--conversion-failures", type=Path, default=DEFAULT_CONVERSION_FAILURES
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sft-split", type=Path, default=DEFAULT_SFT_SPLIT)
    parser.add_argument("--rl-split", type=Path, default=DEFAULT_RL_SPLIT)
    parser.add_argument("--sft-training", type=Path, default=DEFAULT_SFT_TRAINING)
    parser.add_argument("--pool-size", type=int, default=POOL_SIZE)
    parser.add_argument("--sft-size", type=int, default=SFT_SIZE)
    parser.add_argument("--rl-size", type=int, default=RL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-target-tokens", type=int, default=MAX_TARGET_TOKENS)
    parser.add_argument(
        "--expected-input-records", type=int, default=DEFAULT_EXPECTED_INPUT_RECORDS,
        help="refuse partial conversion; use 0 only for an intentional custom dataset",
    )
    args = parser.parse_args()
    summary = run(
        args.input.resolve(), args.conversion_failures.resolve(), args.model.resolve(),
        args.output_dir.resolve(),
        args.sft_split.resolve(), args.rl_split.resolve(), args.sft_training.resolve(),
        args.pool_size, args.sft_size, args.rl_size, args.seed, args.max_target_tokens,
        args.expected_input_records,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
