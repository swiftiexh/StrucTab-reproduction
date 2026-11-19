"""Build the fixed, stratified Phase 5 pretraining pool and its statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from reproduction.common.data_policy import assert_data_usage
from reproduction.common.paths import project_path
from reproduction.common.protocol import format_sequential_target
from reproduction.data.conversion import project_relative
from reproduction.data.pretraining_pool import (
    DEFAULT_SEED,
    MAX_TARGET_TOKENS,
    SYNTH_TARGET,
    VALIDATION_RATIO,
    assign_pool_splits,
    complexity_bucket,
    distribution,
)
from reproduction.data.schema import NormalizedTableSample
from reproduction.structure.parser import parse_html_structure


DEFAULT_INPUTS = [
    project_path("reproduction/data/normalized/synthtabnet_pretrain_candidates.jsonl"),
    project_path("reproduction/data/normalized/fintabnet_pretrain_candidates.jsonl"),
]
DEFAULT_SELECTION = project_path("reproduction/data/pools/pretraining_source_selection.json")
DEFAULT_MODEL = project_path("assets/models/HunyuanOCR/v1.0")
DEFAULT_OUTPUT = project_path("reproduction/data/pools/pretraining_pool.jsonl")
DEFAULT_MANIFEST = project_path("reproduction/data/pools/pretraining_pool_manifest.jsonl")
DEFAULT_REJECTED = project_path("reproduction/data/pools/pretraining_pool_rejected.jsonl")
DEFAULT_SUMMARY = project_path("reproduction/data/pools/pretraining_pool_summary.json")


def _load_candidates(paths: list[Path], selection: dict) -> tuple[list[dict], list[dict]]:
    ranks = {
        source: {item["source_key"]: item["selection_rank"] for item in selection[key]}
        for source, key in (
            ("synthtabnet", "synth_candidates"),
            ("fintabnet", "fintabnet_candidates"),
        )
    }
    candidates = []
    rejected = []
    ids: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as input_file:
            for line in input_file:
                sample = NormalizedTableSample.from_dict(json.loads(line))
                assert_data_usage(sample.source, "pretrain")
                if sample.sample_id in ids:
                    raise ValueError(f"duplicate candidate sample ID: {sample.sample_id}")
                source_key = str(sample.metadata["source_key"])
                if source_key not in ranks[sample.source]:
                    raise ValueError(f"candidate absent from selection: {sample.sample_id}")
                parsed = parse_html_structure(sample.html)
                if not parsed["is_valid"]:
                    rejected.append(
                        {
                            "sample_id": sample.sample_id,
                            "source": sample.source,
                            "source_key": source_key,
                            "reason": "invalid_html_structure",
                            "errors": parsed["errors"],
                        }
                    )
                    ids.add(sample.sample_id)
                    continue
                stage2_target = format_sequential_target(
                    parsed["row_count"], parsed["col_count"], parsed["merge_info"], sample.html
                )
                candidates.append(
                    {
                        "sample": sample,
                        "selection_rank": ranks[sample.source][source_key],
                        "row_count": parsed["row_count"],
                        "col_count": parsed["col_count"],
                        "physical_cell_count": len(parsed["cells"]),
                        "merge_count": len(parsed["merged_cells"]),
                        "stage2_target": stage2_target,
                    }
                )
                ids.add(sample.sample_id)
    return candidates, rejected


def _add_token_lengths(candidates: list[dict], model_path: Path, batch_size: int = 128) -> str:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        html_ids = tokenizer(
            [item["sample"].html for item in batch],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        stage2_ids = tokenizer(
            [item["stage2_target"] for item in batch],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        for item, html_tokens, stage2_tokens in zip(batch, html_ids, stage2_ids):
            item["html_token_count"] = len(html_tokens)
            item["stage2_target_token_count"] = len(stage2_tokens)
            del item["stage2_target"]
    return tokenizer.__class__.__name__


def _write_jsonl(path: Path, rows) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    inputs: list[Path],
    selection_path: Path,
    model_path: Path,
    output: Path,
    manifest: Path,
    rejected: Path,
    summary_path: Path,
    seed: int = DEFAULT_SEED,
    max_target_tokens: int = MAX_TARGET_TOKENS,
) -> dict:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["seed"] != seed:
        raise ValueError(f"selection seed={selection['seed']} disagrees with pool seed={seed}")
    candidates, structural_rejected = _load_candidates(inputs, selection)
    tokenizer_class = _add_token_lengths(candidates, model_path)

    overlong = [
        item for item in candidates if item["stage2_target_token_count"] > max_target_tokens
    ]
    eligible = [
        item for item in candidates if item["stage2_target_token_count"] <= max_target_tokens
    ]
    synth = sorted(
        (item for item in eligible if item["sample"].source == "synthtabnet"),
        key=lambda item: item["selection_rank"],
    )
    fin = sorted(
        (item for item in eligible if item["sample"].source == "fintabnet"),
        key=lambda item: item["selection_rank"],
    )
    if len(synth) < SYNTH_TARGET:
        raise ValueError(f"only {len(synth)}/{SYNTH_TARGET} eligible SynthTabNet samples")
    if len(fin) < 5_000:
        raise ValueError(f"only {len(fin)}/5000 eligible FinTabNet samples")
    selected = [*synth[:SYNTH_TARGET], *fin]
    assign_pool_splits(selected, seed, VALIDATION_RATIO)

    pool_rows = []
    manifest_rows = []
    for item in selected:
        sample = item["sample"]
        bucket = complexity_bucket(
            item["row_count"], item["col_count"], item["physical_cell_count"], item["merge_count"]
        )
        pool_metadata = {
            "seed": seed,
            "source_split": sample.split,
            "selection_rank": item["selection_rank"],
            "complexity": bucket,
            "row_count": item["row_count"],
            "col_count": item["col_count"],
            "physical_cell_count": item["physical_cell_count"],
            "merge_count": item["merge_count"],
            "html_token_count": item["html_token_count"],
            "stage2_target_token_count": item["stage2_target_token_count"],
            "target_token_limit": max_target_tokens,
        }
        metadata = dict(sample.metadata)
        metadata["pretraining_pool"] = pool_metadata
        pooled = NormalizedTableSample(
            sample_id=sample.sample_id,
            source=sample.source,
            split=item["pool_split"],
            image_path=sample.image_path,
            html=sample.html,
            metadata=metadata,
        )
        pool_rows.append(pooled.to_dict())
        manifest_rows.append(
            {
                "sample_id": sample.sample_id,
                "source": sample.source,
                "source_key": sample.metadata["source_key"],
                "pool_split": item["pool_split"],
                "image_path": sample.image_path,
                **pool_metadata,
            }
        )

    overlong_rows = [
        {
            "sample_id": item["sample"].sample_id,
            "source": item["sample"].source,
            "source_key": item["sample"].metadata["source_key"],
            "reason": "stage2_target_too_long",
            "stage2_target_token_count": item["stage2_target_token_count"],
            "target_token_limit": max_target_tokens,
        }
        for item in overlong
    ]
    rejected_rows = [*structural_rejected, *overlong_rows]
    pool_sha = _write_jsonl(output, pool_rows)
    manifest_sha = _write_jsonl(manifest, manifest_rows)
    if rejected_rows:
        _write_jsonl(rejected, rejected_rows)
    else:
        rejected.unlink(missing_ok=True)

    source_counts = Counter(item["sample"].source for item in selected)
    split_counts = Counter(item["pool_split"] for item in selected)
    source_split_counts = Counter(
        f"{item['sample'].source}:{item['pool_split']}" for item in selected
    )
    complexity_counts = Counter(row["pretraining_pool"]["complexity"] for row in (r["metadata"] for r in pool_rows))
    summary = {
        "seed": seed,
        "selection": project_relative(selection_path),
        "tokenizer": tokenizer_class,
        "tokenizer_path": project_relative(model_path),
        "token_count_scope": "text target only; multimodal prompt/image tokens are measured in Phase 6",
        "max_stage2_target_tokens": max_target_tokens,
        "overlong_policy": "exclude before training; never truncate HTML silently",
        "materialized_candidate_counts": {
            "synthtabnet": len(selection["synth_candidates"]),
            "fintabnet": len(selection["fintabnet_candidates"]),
        },
        "structurally_valid_candidate_counts": dict(
            Counter(item["sample"].source for item in candidates)
        ),
        "structural_rejected_counts": dict(Counter(row["source"] for row in structural_rejected)),
        "overlong_rejected_counts": dict(Counter(item["sample"].source for item in overlong)),
        "pool_records": len(selected),
        "source_counts": dict(sorted(source_counts.items())),
        "source_ratios": {
            source: round(count / len(selected), 8) for source, count in sorted(source_counts.items())
        },
        "split_counts": dict(sorted(split_counts.items())),
        "source_split_counts": dict(sorted(source_split_counts.items())),
        "complexity_definition": {
            "complex": "rows>=20 or cols>=10 or physical_cells>=100 or merges>=5",
            "merged": "not complex and merges>0",
            "simple": "not complex and merges=0",
        },
        "complexity_counts": dict(sorted(complexity_counts.items())),
        "distributions": {
            "row_count": distribution(item["row_count"] for item in selected),
            "col_count": distribution(item["col_count"] for item in selected),
            "physical_cell_count": distribution(item["physical_cell_count"] for item in selected),
            "merge_count": distribution(item["merge_count"] for item in selected),
            "html_token_count": distribution(item["html_token_count"] for item in selected),
            "stage2_target_token_count": distribution(
                item["stage2_target_token_count"] for item in selected
            ),
        },
        "outputs": {
            "pool": project_relative(output),
            "manifest": project_relative(manifest),
            "rejected": project_relative(rejected) if rejected_rows else None,
        },
        "sha256": {"pool": pool_sha, "manifest": manifest_sha},
    }
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-target-tokens", type=int, default=MAX_TARGET_TOKENS)
    args = parser.parse_args()
    summary = run(
        [path.resolve() for path in args.inputs], args.selection.resolve(), args.model.resolve(),
        args.output.resolve(), args.manifest.resolve(), args.rejected.resolve(),
        args.summary.resolve(), args.seed, args.max_target_tokens,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
