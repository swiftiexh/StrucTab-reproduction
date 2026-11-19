"""Select eight fixed budgeted bases for the 32-step Stage 1/2 soak test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from reproduction.common.paths import project_path
from reproduction.data.training_examples import TrainingExample


DEFAULT_DIR = project_path("reproduction/data/training/budgeted")
DEFAULT_OUTPUT_DIR = project_path("reproduction/data/training/soak")


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_base_ids(manifest_rows: list[dict], count: int = 8) -> list[str]:
    train = [row for row in manifest_rows if row["split"] == "train"]
    selected = []
    # First cover every available source/complexity stratum with its longest
    # Stage 2 example, then fill remaining positions with global longest examples.
    for source in ("synthtabnet", "fintabnet"):
        for complexity in ("complex", "merged", "simple"):
            candidates = [
                row for row in train
                if row["source"] == source and row["complexity"] == complexity
            ]
            if candidates:
                selected.append(
                    max(
                        candidates,
                        key=lambda row: (
                            row["measurements"]["sequential"]["sequence_tokens"],
                            row["sample_id"],
                        ),
                    )["sample_id"]
                )
    for row in sorted(
        train,
        key=lambda row: (
            -row["measurements"]["sequential"]["sequence_tokens"],
            row["sample_id"],
        ),
    ):
        if row["sample_id"] not in selected:
            selected.append(row["sample_id"])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"only selected {len(selected)}/{count} soak bases")
    return selected


def load_examples(path: Path) -> list[TrainingExample]:
    with path.open(encoding="utf-8") as input_file:
        return [TrainingExample.from_dict(json.loads(line)) for line in input_file if line.strip()]


def run(input_dir: Path, output_dir: Path) -> dict:
    with (input_dir / "manifest.jsonl").open(encoding="utf-8") as input_file:
        manifest = [json.loads(line) for line in input_file if line.strip()]
    base_ids = select_base_ids(manifest)
    stage1_by_base = {}
    for example in load_examples(input_dir / "stage1_train.jsonl"):
        stage1_by_base.setdefault(example.base_sample_id, {})[example.task_type] = example
    stage2_by_base = {
        example.base_sample_id: example
        for example in load_examples(input_dir / "stage2_train.jsonl")
    }
    stage1 = [
        stage1_by_base[base_id][task].to_dict()
        for base_id in base_ids
        for task in ("row_col", "merge", "html")
    ]
    stage2 = [stage2_by_base[base_id].to_dict() for base_id in base_ids]
    hashes = {
        "stage1": write_jsonl(output_dir / "stage1_soak.jsonl", stage1),
        "stage2": write_jsonl(output_dir / "stage2_soak.jsonl", stage2),
    }
    manifest_by_id = {row["sample_id"]: row for row in manifest}
    summary = {
        "base_samples": 8,
        "stage1_steps": 24,
        "stage2_steps": 8,
        "base_sample_ids": base_ids,
        "profiles": [
            {
                "sample_id": base_id,
                "source": manifest_by_id[base_id]["source"],
                "complexity": manifest_by_id[base_id]["complexity"],
                "stage2_sequence_tokens": manifest_by_id[base_id]["measurements"]["sequential"]["sequence_tokens"],
            }
            for base_id in base_ids
        ],
        "hashes": hashes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = (output_dir / "summary.json.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.input_dir.resolve(), args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
