"""Validate Phase 5 pool integrity, isolation, splits, images, and manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from reproduction.common.data_policy import assert_data_usage
from reproduction.common.paths import project_path
from reproduction.data.conversion import validate_image
from reproduction.data.schema import NormalizedTableSample


DEFAULT_POOL = project_path("reproduction/data/pools/pretraining_pool.jsonl")
DEFAULT_MANIFEST = project_path("reproduction/data/pools/pretraining_pool_manifest.jsonl")
DEFAULT_SUMMARY = project_path("reproduction/data/pools/pretraining_pool_summary.json")


def run(pool: Path, manifest: Path, summary_path: Path, decode_images: bool = True) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifests = {}
    with manifest.open(encoding="utf-8") as input_file:
        for line in input_file:
            record = json.loads(line)
            sample_id = record["sample_id"]
            if sample_id in manifests:
                raise ValueError(f"duplicate manifest sample ID: {sample_id}")
            manifests[sample_id] = record

    ids = set()
    image_paths = set()
    sources: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    complexities: Counter[str] = Counter()
    with pool.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            sample = NormalizedTableSample.from_dict(json.loads(line))
            assert_data_usage(sample.source, "pretrain")
            if sample.sample_id in ids:
                raise ValueError(f"duplicate pool sample ID at line {line_number}")
            if sample.image_path in image_paths:
                raise ValueError(f"duplicate pool image path: {sample.image_path}")
            if sample.sample_id not in manifests:
                raise ValueError(f"pool sample missing from manifest: {sample.sample_id}")
            pool_metadata = sample.metadata["pretraining_pool"]
            manifest_record = manifests[sample.sample_id]
            if sample.split != manifest_record["pool_split"]:
                raise ValueError(f"pool/manifest split mismatch: {sample.sample_id}")
            if pool_metadata["stage2_target_token_count"] > summary["max_stage2_target_tokens"]:
                raise ValueError(f"overlong sample leaked into pool: {sample.sample_id}")
            image = project_path(sample.image_path)
            if decode_images:
                validate_image(image)
            elif not image.is_file():
                raise FileNotFoundError(image)
            ids.add(sample.sample_id)
            image_paths.add(sample.image_path)
            sources[sample.source] += 1
            splits[sample.split] += 1
            complexities[pool_metadata["complexity"]] += 1

    if ids != set(manifests):
        raise ValueError("pool and manifest sample-ID sets differ")
    if len(ids) != summary["pool_records"]:
        raise ValueError("pool count disagrees with summary")
    if dict(sorted(sources.items())) != summary["source_counts"]:
        raise ValueError("source counts disagree with summary")
    if dict(sorted(splits.items())) != summary["split_counts"]:
        raise ValueError("split counts disagree with summary")
    if dict(sorted(complexities.items())) != summary["complexity_counts"]:
        raise ValueError("complexity counts disagree with summary")
    pool_sha = hashlib.sha256(pool.read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if pool_sha != summary["sha256"]["pool"] or manifest_sha != summary["sha256"]["manifest"]:
        raise ValueError("pool or manifest SHA-256 disagrees with summary")
    return {
        "status": "ok",
        "records": len(ids),
        "unique_sample_ids": len(ids),
        "unique_image_paths": len(image_paths),
        "source_counts": dict(sorted(sources.items())),
        "split_counts": dict(sorted(splits.items())),
        "complexity_counts": dict(sorted(complexities.items())),
        "images_decoded": len(ids) if decode_images else 0,
        "overlong_samples_in_pool": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--skip-image-decode", action="store_true")
    args = parser.parse_args()
    result = run(
        args.pool.resolve(), args.manifest.resolve(), args.summary.resolve(),
        decode_images=not args.skip_image_decode,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
