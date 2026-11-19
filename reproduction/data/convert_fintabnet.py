"""Convert FinTabNet Parquet shards into the canonical schema."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

from reproduction.common.data_policy import assert_data_usage
from reproduction.common.ids import make_sample_id
from reproduction.common.paths import project_path
from reproduction.data.conversion import ConversionSession, project_relative, write_image_bytes
from reproduction.data.schema import NormalizedTableSample


DEFAULT_SOURCE_DIR = project_path("assets/datasets/pretrain/FinTabNet-HTML/en")
DEFAULT_OUTPUT = project_path("reproduction/data/normalized/fintabnet_smoke.jsonl")
DEFAULT_IMAGE_DIR = project_path("reproduction/data/normalized/images/fintabnet")


def _split_from_shard(path: Path) -> str:
    match = re.match(r"([A-Za-z]+)-\d+-of-\d+\.parquet$", path.name)
    return match.group(1).lower() if match else "train"


def convert_record(record: dict, shard: Path, row_index: int, image_dir: Path) -> tuple[NormalizedTableSample, bytes]:
    image_record = record["image"]
    image_bytes = image_record["bytes"]
    original_path = image_record.get("path") or f"row-{row_index}.png"
    html = record["html_table"].strip()
    split = _split_from_shard(shard)
    source_key = f"{shard.name}:{row_index}"
    sample_id = make_sample_id("fintabnet", source_key, original_path, html)
    suffix = Path(original_path).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"
    output_image = image_dir / f"{sample_id}{suffix}"
    sample = NormalizedTableSample(
        sample_id=sample_id,
        source="fintabnet",
        split=split,
        image_path=project_relative(output_image),
        html=html,
        metadata={
            "source_key": source_key,
            "source_parquet": project_relative(shard),
            "source_row_index": row_index,
            "original_image_path": original_path,
        },
    )
    return sample, image_bytes


def run(source_dir: Path, output: Path, image_dir: Path, limit: int, restart: bool = False) -> dict:
    assert_data_usage("fintabnet", "pretrain")
    session = ConversionSession(output, "fintabnet", restart=restart)
    scanned = skipped = 0
    shards = sorted(source_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no Parquet shards found under {source_dir}")
    for shard in shards:
        row_index = 0
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=16, columns=["image", "html_table"]):
            for record in batch.to_pylist():
                scanned += 1
                source_key = f"{shard.name}:{row_index}"
                try:
                    if session.has_source_key(source_key):
                        skipped += 1
                    else:
                        sample, image_bytes = convert_record(record, shard, row_index, image_dir)
                        write_image_bytes(project_path(sample.image_path), image_bytes)
                        session.append(sample)
                except Exception as exc:
                    session.record_failure(source_key, exc)
                row_index += 1
                if limit > 0 and len(session.sample_ids) >= limit:
                    return session.write_summary(scanned=scanned, skipped=skipped)
    return session.write_summary(scanned=scanned, skipped=skipped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--limit", type=int, default=30, help="target total records; 0 means all")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    summary = run(
        args.source_dir.resolve(), args.output.resolve(), args.image_dir.resolve(), args.limit, args.restart
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
