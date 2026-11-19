"""CPU-only validation of Phase 6 multimodal batches and label masking."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoProcessor

from reproduction.common.paths import project_path
from reproduction.data.pretraining_pool import distribution
from reproduction.train.hunyuan_feasibility import load_examples, prepare_example


DEFAULT_MODEL = project_path("assets/models/HunyuanOCR/v1.0")
DEFAULT_DATA = project_path("reproduction/data/training/feasibility_probe.jsonl")
DEFAULT_OUTPUT = project_path("reproduction/results/feasibility/data_validation.json")


def run(model: Path, data: Path, output: Path, max_sequence_tokens: int = 8192) -> dict:
    examples = load_examples(data, 10_000)
    processor = AutoProcessor.from_pretrained(
        model, local_files_only=True, trust_remote_code=True, backend="pil"
    )
    rows = []
    for example in examples:
        _, info = prepare_example(processor, example, "cpu", max_sequence_tokens)
        rows.append(info)
    summary = {
        "status": "ok",
        "examples": len(rows),
        "task_counts": dict(sorted(Counter(row["task_type"] for row in rows).items())),
        "sequence_tokens": distribution(row["sequence_tokens"] for row in rows),
        "prompt_tokens": distribution(row["prompt_tokens"] for row in rows),
        "supervised_tokens": distribution(row["supervised_tokens"] for row in rows),
        "max_sequence_tokens": max_sequence_tokens,
        "all_targets_have_supervision": all(row["supervised_tokens"] > 0 for row in rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-sequence-tokens", type=int, default=8192)
    args = parser.parse_args()
    result = run(
        args.model.resolve(), args.data.resolve(), args.output.resolve(), args.max_sequence_tokens
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
