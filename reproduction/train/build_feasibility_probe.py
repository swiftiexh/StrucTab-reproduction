"""Build a fixed 12-example Stage 1 dataset for Phase 6 GPU probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from reproduction.common.paths import project_path
from reproduction.data.schema import NormalizedTableSample
from reproduction.data.training_examples import build_stage1_examples


DEFAULT_POOL = project_path("reproduction/data/pools/pretraining_pool.jsonl")
DEFAULT_OUTPUT = project_path("reproduction/data/training/feasibility_probe.jsonl")
DEFAULT_SUMMARY = project_path("reproduction/data/training/feasibility_probe_summary.json")
TARGET_LENGTHS = (600, 1600)


def _select_bases(pool: Path) -> list[tuple[NormalizedTableSample, str]]:
    candidates = {source: {target: None for target in TARGET_LENGTHS} for source in ("synthtabnet", "fintabnet")}
    distances = {source: {target: None for target in TARGET_LENGTHS} for source in candidates}
    with pool.open(encoding="utf-8") as input_file:
        for line in input_file:
            sample = NormalizedTableSample.from_dict(json.loads(line))
            if sample.split != "train" or sample.source not in candidates:
                continue
            length = sample.metadata["pretraining_pool"]["stage2_target_token_count"]
            for target in TARGET_LENGTHS:
                distance = (abs(length - target), sample.sample_id)
                if distances[sample.source][target] is None or distance < distances[sample.source][target]:
                    if any(
                        selected is not None and selected.sample_id == sample.sample_id
                        for selected in candidates[sample.source].values()
                    ):
                        continue
                    candidates[sample.source][target] = sample
                    distances[sample.source][target] = distance
    selected = []
    for source in ("synthtabnet", "fintabnet"):
        for target in TARGET_LENGTHS:
            sample = candidates[source][target]
            if sample is None:
                raise ValueError(f"no probe candidate for {source} near {target} tokens")
            selected.append((sample, f"{source}-near-{target}"))
    return selected


def run(pool: Path, output: Path, summary_path: Path) -> dict:
    bases = _select_bases(pool)
    by_base = [(base, profile, build_stage1_examples(base)) for base, profile in bases]
    examples = []
    # Base-major order makes the default first three steps exercise all tasks
    # on exactly the same image, which keeps vision memory comparable.
    for base, profile, stage1 in by_base:
        for example in stage1:
            record = example.to_dict()
            record["metadata"]["probe_profile"] = profile
            record["metadata"]["base_stage2_target_token_count"] = base.metadata[
                "pretraining_pool"
            ]["stage2_target_token_count"]
            examples.append(record)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        for record in examples:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output)
    summary = {
        "base_samples": len(bases),
        "training_examples": len(examples),
        "task_counts": {task: 4 for task in ("row_col", "merge", "html")},
        "profiles": [profile for _, profile in bases],
        "base_sample_ids": [base.sample_id for base, _ in bases],
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    summary_tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    summary_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_tmp.replace(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    print(json.dumps(run(args.pool.resolve(), args.output.resolve(), args.summary.resolve()), indent=2))


if __name__ == "__main__":
    main()
