"""Convert PubTables Parquet shards with OTSL/HTML alignment validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from reproduction.common.data_policy import assert_data_usage
from reproduction.common.ids import make_sample_id
from reproduction.common.paths import project_path
from reproduction.data.conversion import ConversionSession, project_relative, write_image_bytes
from reproduction.data.html_reconstruction import reconstruct_html, validate_pubtables_alignment
from reproduction.data.schema import NormalizedTableSample


DEFAULT_SOURCE_DIR = project_path("assets/datasets/high_quality/PubTables-1M")
DEFAULT_OUTPUT = project_path("reproduction/data/normalized/pubtables_smoke.jsonl")
DEFAULT_IMAGE_DIR = project_path("reproduction/data/normalized/images/pubtables")
PUBTABLES_COLUMNS = [
    "filename", "split", "imgid", "dataset", "table_bbox", "cells", "otsl",
    "html", "html_restored", "cols", "rows", "html_len", "otsl_len", "image",
]


def _flatten_cells(value: list[list[dict]]) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("PubTables cells must be a nested list")
    return [cell for group in value for cell in group]


def convert_record(record: dict, shard: Path, row_index: int, image_dir: Path) -> tuple[NormalizedTableSample, bytes]:
    cells = _flatten_cells(record["cells"])
    html = reconstruct_html(record["html_restored"], cells)
    alignment = validate_pubtables_alignment(
        record["otsl"], int(record["rows"]), int(record["cols"]), html, len(cells)
    )
    if int(record["otsl_len"]) != len(record["otsl"]):
        raise ValueError("stored otsl_len disagrees with the OTSL token list")
    if int(record["html_len"]) != len(record["html"]):
        raise ValueError("stored html_len disagrees with the original HTML token list")

    filename = str(record["filename"])
    source_key = f"{shard.name}:{row_index}"
    sample_id = make_sample_id("pubtables", source_key, filename, html)
    suffix = Path(filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".jpg"
    output_image = image_dir / f"{sample_id}{suffix}"
    image_record = record["image"]
    annotations = [{"tokens": cell["tokens"], "bbox": cell.get("bbox")} for cell in cells]
    sample = NormalizedTableSample(
        sample_id=sample_id,
        source="pubtables",
        split=str(record["split"]).lower(),
        image_path=project_relative(output_image),
        html=html,
        metadata={
            "source_key": source_key,
            "source_parquet": project_relative(shard),
            "source_row_index": row_index,
            "original_filename": filename,
            "original_image_path": image_record.get("path"),
            "imgid": record["imgid"],
            "dataset": record["dataset"],
            "table_bbox": record["table_bbox"],
            "rows": record["rows"],
            "cols": record["cols"],
            "otsl": record["otsl"],
            "alignment": alignment,
            "cell_annotations": annotations,
            "html_structure_field": "html_restored",
        },
    )
    return sample, image_record["bytes"]


def run(source_dir: Path, output: Path, image_dir: Path, limit: int, restart: bool = False) -> dict:
    assert_data_usage("pubtables", "sft")
    session = ConversionSession(output, "pubtables", restart=restart)
    scanned = skipped = 0
    shards = sorted(source_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no Parquet shards found under {source_dir}")
    for shard in shards:
        row_index = 0
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=16, columns=PUBTABLES_COLUMNS):
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
