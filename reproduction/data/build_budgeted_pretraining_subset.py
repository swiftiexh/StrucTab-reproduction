"""Build the fixed Stage 1/2 subset that fits the local time and memory budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from PIL import Image
from transformers import AutoProcessor

from reproduction.common.paths import project_path
from reproduction.data.pretraining_pool import distribution, stable_priority
from reproduction.data.schema import NormalizedTableSample
from reproduction.data.training_examples import (
    TrainingExample,
    build_stage1_examples,
    build_stage2_example,
)
from reproduction.inference.hunyuan_baseline import build_messages


DEFAULT_POOL = project_path("reproduction/data/pools/pretraining_pool.jsonl")
DEFAULT_MODEL = project_path("assets/models/HunyuanOCR/v1.0")
DEFAULT_OUTPUT_DIR = project_path("reproduction/data/training/budgeted")
DEFAULT_SEED = 42
DEFAULT_PROMPT_LIMIT = 512
DEFAULT_SEQUENCE_LIMIT = 1536
DEFAULT_TEXT_PREFILTER = 1200

# Preserve the Phase 5 source/complexity mix as closely as the small SynthTabNet
# simple stratum permits. Values are base-table counts, not derived examples.
DEFAULT_QUOTAS = {
    ("train", "synthtabnet", "complex"): 120,
    ("train", "synthtabnet", "merged"): 68,
    ("train", "synthtabnet", "simple"): 2,
    ("train", "fintabnet", "complex"): 30,
    ("train", "fintabnet", "merged"): 49,
    ("train", "fintabnet", "simple"): 51,
    ("validation", "synthtabnet", "complex"): 16,
    ("validation", "synthtabnet", "merged"): 8,
    ("validation", "synthtabnet", "simple"): 0,
    ("validation", "fintabnet", "complex"): 3,
    ("validation", "fintabnet", "merged"): 7,
    ("validation", "fintabnet", "simple"): 6,
}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quota_totals(quotas: dict[tuple[str, str, str], int]) -> dict[str, int]:
    totals = Counter()
    for (split, _source, _complexity), count in quotas.items():
        if count < 0:
            raise ValueError("quota counts must be non-negative")
        totals[split] += count
    return dict(totals)


def select_with_quotas(
    strata: dict[tuple[str, str, str], list[NormalizedTableSample]],
    quotas: dict[tuple[str, str, str], int],
    seed: int,
    evaluate: Callable[[NormalizedTableSample], tuple[dict | None, dict | None]],
) -> tuple[list[dict], list[dict]]:
    """Select the first eligible records in deterministic per-stratum hash order."""
    selected: list[dict] = []
    rejected: list[dict] = []
    for key in sorted(quotas):
        required = quotas[key]
        if required == 0:
            continue
        candidates = sorted(
            strata.get(key, []),
            key=lambda sample: stable_priority(
                seed, "budgeted-subset:" + ":".join(key), sample.sample_id
            ),
        )
        accepted = 0
        for sample in candidates:
            record, failure = evaluate(sample)
            if failure is not None:
                rejected.append(failure)
                continue
            assert record is not None
            selected.append(record)
            accepted += 1
            if accepted == required:
                break
        if accepted != required:
            raise ValueError(
                f"budget stratum {key} has only {accepted}/{required} eligible samples; "
                f"evaluated {len(candidates)} candidates"
            )
    return selected, rejected


def _texts(processor, example: TrainingExample, image_path: Path) -> tuple[str, str]:
    prompt_messages = build_messages(image_path, example.prompt)
    full_messages = [*prompt_messages, {"role": "assistant", "content": example.target}]
    return (
        processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=False
        ),
        processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        ),
    )


def measure_sample(
    processor,
    sample: NormalizedTableSample,
    prompt_limit: int,
    sequence_limit: int,
) -> tuple[dict | None, dict | None]:
    stage1 = build_stage1_examples(sample)
    stage2 = build_stage2_example(sample)
    examples = [*stage1, stage2]
    image_path = project_path(sample.image_path)
    measurements = {}
    violations = []
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    for example in examples:
        prompt_text, full_text = _texts(processor, example, image_path)
        prompt_inputs = processor(
            text=[prompt_text], images=image, padding=False, return_tensors="pt"
        )
        full_inputs = processor(
            text=[full_text], images=image, padding=False, return_tensors="pt"
        )
        prompt_tokens = int(prompt_inputs["input_ids"].shape[1])
        sequence_tokens = int(full_inputs["input_ids"].shape[1])
        if not full_inputs["input_ids"][:, :prompt_tokens].equal(prompt_inputs["input_ids"]):
            raise ValueError(f"prompt prefix mismatch for {example.sample_id}")
        supervised_tokens = sequence_tokens - prompt_tokens
        measurements[example.task_type] = {
            "prompt_tokens": prompt_tokens,
            "sequence_tokens": sequence_tokens,
            "supervised_tokens": supervised_tokens,
        }
        if prompt_tokens > prompt_limit:
            violations.append(
                f"{example.task_type}:prompt_tokens={prompt_tokens}>{prompt_limit}"
            )
        if sequence_tokens > sequence_limit:
            violations.append(
                f"{example.task_type}:sequence_tokens={sequence_tokens}>{sequence_limit}"
            )
        if supervised_tokens < 1:
            violations.append(f"{example.task_type}:no_supervised_tokens")
        del prompt_inputs, full_inputs
    if violations:
        return None, {
            "sample_id": sample.sample_id,
            "source": sample.source,
            "pool_split": sample.split,
            "reason": "multimodal_training_budget_exceeded",
            "violations": violations,
            "measurements": measurements,
        }
    return {
        "sample": sample,
        "stage1": stage1,
        "stage2": stage2,
        "image_size": list(image.size),
        "measurements": measurements,
    }, None


def _load_strata(pool: Path, text_prefilter: int) -> tuple[dict, Counter]:
    strata: dict[tuple[str, str, str], list[NormalizedTableSample]] = {}
    counts = Counter()
    with pool.open(encoding="utf-8") as input_file:
        for line in input_file:
            sample = NormalizedTableSample.from_dict(json.loads(line))
            metadata = sample.metadata["pretraining_pool"]
            key = (sample.split, sample.source, metadata["complexity"])
            counts[(key, "pool")] += 1
            if metadata["stage2_target_token_count"] > text_prefilter:
                counts[(key, "text_prefiltered")] += 1
                continue
            strata.setdefault(key, []).append(sample)
            counts[(key, "candidates")] += 1
    return strata, counts


def run(
    pool: Path,
    model: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    prompt_limit: int = DEFAULT_PROMPT_LIMIT,
    sequence_limit: int = DEFAULT_SEQUENCE_LIMIT,
    text_prefilter: int = DEFAULT_TEXT_PREFILTER,
    quotas: dict[tuple[str, str, str], int] = DEFAULT_QUOTAS,
) -> dict:
    started = time.time()
    totals = quota_totals(quotas)
    if totals != {"train": 320, "validation": 40}:
        raise ValueError(f"unexpected budget totals: {totals}")
    strata, candidate_counts = _load_strata(pool, text_prefilter)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / (
        f"screening_cache_p{prompt_limit}_s{sequence_limit}_t{text_prefilter}.jsonl"
    )
    cache: dict[str, dict] = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as cache_input:
            for line in cache_input:
                if line.strip():
                    cached = json.loads(line)
                    cache[cached["sample_id"]] = cached
    processor = AutoProcessor.from_pretrained(
        model, local_files_only=True, trust_remote_code=True, backend="pil"
    )
    evaluated = 0
    cache_hits = 0
    cache_output = cache_path.open("a", encoding="utf-8")

    def eligible_record(sample, image_size, measurements):
        return {
            "sample": sample,
            "stage1": build_stage1_examples(sample),
            "stage2": build_stage2_example(sample),
            "image_size": image_size,
            "measurements": measurements,
        }

    def evaluate(sample):
        nonlocal evaluated, cache_hits
        cached = cache.get(sample.sample_id)
        if cached is not None:
            cache_hits += 1
            if cached["eligible"]:
                return eligible_record(
                    sample, cached["image_size"], cached["measurements"]
                ), None
            return None, cached["failure"]
        evaluated += 1
        if evaluated == 1 or evaluated % 25 == 0:
            print(f"evaluated={evaluated} sample_id={sample.sample_id}", flush=True)
        record, failure = measure_sample(processor, sample, prompt_limit, sequence_limit)
        cached = {
            "sample_id": sample.sample_id,
            "eligible": record is not None,
            "image_size": record["image_size"] if record is not None else None,
            "measurements": (
                record["measurements"] if record is not None else failure["measurements"]
            ),
            "failure": failure,
        }
        cache[sample.sample_id] = cached
        cache_output.write(json.dumps(cached, ensure_ascii=False) + "\n")
        cache_output.flush()
        os.fsync(cache_output.fileno())
        return record, failure

    try:
        selected, rejected = select_with_quotas(strata, quotas, seed, evaluate)
    finally:
        cache_output.close()
    selected.sort(
        key=lambda item: (
            item["sample"].split,
            stable_priority(seed, f"budgeted-{item['sample'].split}-order", item["sample"].sample_id),
        )
    )
    hashes = {}
    manifest_rows = []
    for split in ("train", "validation"):
        records = [item for item in selected if item["sample"].split == split]
        hashes[f"{split}_bases"] = write_jsonl(
            output_dir / f"{split}_bases.jsonl",
            [item["sample"].to_dict() for item in records],
        )
        hashes[f"stage1_{split}"] = write_jsonl(
            output_dir / f"stage1_{split}.jsonl",
            [example.to_dict() for item in records for example in item["stage1"]],
        )
        hashes[f"stage2_{split}"] = write_jsonl(
            output_dir / f"stage2_{split}.jsonl",
            [item["stage2"].to_dict() for item in records],
        )
        for item in records:
            sample = item["sample"]
            manifest_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "source": sample.source,
                    "split": split,
                    "complexity": sample.metadata["pretraining_pool"]["complexity"],
                    "image_path": sample.image_path,
                    "image_size": item["image_size"],
                    "measurements": item["measurements"],
                }
            )
    hashes["manifest"] = write_jsonl(output_dir / "manifest.jsonl", manifest_rows)
    hashes["rejected"] = write_jsonl(output_dir / "screened_rejected.jsonl", rejected)

    selected_counts = Counter(
        (item["sample"].split, item["sample"].source,
         item["sample"].metadata["pretraining_pool"]["complexity"])
        for item in selected
    )
    all_measurements = [
        measurement
        for item in selected
        for measurement in item["measurements"].values()
    ]
    summary = {
        "status": "ok",
        "seed": seed,
        "paper_reference": {
            "stage1_epochs": 1,
            "stage1_learning_rate": 4e-5,
            "stage2_epochs": 1,
            "stage2_learning_rate": "not separately disclosed; local assumption is 4e-5",
        },
        "local_budget": {
            "train_bases": totals["train"],
            "validation_bases": totals["validation"],
            "stage1_train_examples": totals["train"] * 3,
            "stage2_train_examples": totals["train"],
            "prompt_token_limit": prompt_limit,
            "sequence_token_limit": sequence_limit,
            "text_target_prefilter": text_prefilter,
            "overlong_policy": "exclude before training; never truncate",
        },
        "selected_strata": {"/".join(key): value for key, value in sorted(selected_counts.items())},
        "screened_candidates_this_run": evaluated,
        "screening_cache_hits": cache_hits,
        "screening_cache_records": len(cache),
        "screened_rejected": len(rejected),
        "sequence_tokens": distribution(m["sequence_tokens"] for m in all_measurements),
        "prompt_tokens": distribution(m["prompt_tokens"] for m in all_measurements),
        "hashes": hashes,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prompt-token-limit", type=int, default=DEFAULT_PROMPT_LIMIT)
    parser.add_argument("--sequence-token-limit", type=int, default=DEFAULT_SEQUENCE_LIMIT)
    parser.add_argument("--text-target-prefilter", type=int, default=DEFAULT_TEXT_PREFILTER)
    args = parser.parse_args()
    summary = run(
        args.pool.resolve(),
        args.model.resolve(),
        args.output_dir.resolve(),
        args.seed,
        args.prompt_token_limit,
        args.sequence_token_limit,
        args.text_target_prefilter,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
