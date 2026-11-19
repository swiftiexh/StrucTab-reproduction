"""Create the deterministic source-row selection used by Phase 5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from reproduction.common.paths import project_path
from reproduction.data.conversion import project_relative
from reproduction.data.pretraining_pool import (
    DEFAULT_SEED,
    SYNTH_RESERVE,
    SYNTH_TARGET,
    select_all_fintabnet,
    select_synth_candidates,
)


DEFAULT_SYNTH = project_path("assets/datasets/pretrain/SynthTabNet/sparse/synthetic_data.jsonl")
DEFAULT_FIN_DIR = project_path("assets/datasets/pretrain/FinTabNet-HTML/en")
DEFAULT_OUTPUT = project_path("reproduction/data/pools/pretraining_source_selection.json")


def run(synth: Path, fin_dir: Path, output: Path, seed: int = DEFAULT_SEED) -> dict:
    synth_selected, split_counts = select_synth_candidates(
        synth, SYNTH_TARGET + SYNTH_RESERVE, seed
    )
    fin_selected = select_all_fintabnet(fin_dir, seed)
    result = {
        "seed": seed,
        "policy": "smallest seeded SHA-256 priorities",
        "synth_source": project_relative(synth),
        "synth_source_split_counts": dict(sorted(split_counts.items())),
        "synth_target": SYNTH_TARGET,
        "synth_reserve": SYNTH_RESERVE,
        "synth_candidates": synth_selected,
        "fintabnet_source_dir": project_relative(fin_dir),
        "fintabnet_candidates": fin_selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    result["selection_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    result["output"] = project_relative(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synth", type=Path, default=DEFAULT_SYNTH)
    parser.add_argument("--fin-dir", type=Path, default=DEFAULT_FIN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    result = run(args.synth.resolve(), args.fin_dir.resolve(), args.output.resolve(), args.seed)
    print(
        json.dumps(
            {
                "seed": result["seed"],
                "synth_source_split_counts": result["synth_source_split_counts"],
                "synth_candidates": len(result["synth_candidates"]),
                "fintabnet_candidates": len(result["fintabnet_candidates"]),
                "selection_sha256": result["selection_sha256"],
                "output": result["output"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
