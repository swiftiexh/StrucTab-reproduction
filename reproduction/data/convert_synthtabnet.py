"""Convert SynthTabNet JSONL annotations into the canonical schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reproduction.common.data_policy import assert_data_usage
from reproduction.common.ids import make_sample_id
from reproduction.common.paths import project_path
from reproduction.data.conversion import ConversionSession, project_relative, validate_image
from reproduction.data.html_reconstruction import reconstruct_html
from reproduction.data.schema import NormalizedTableSample


DEFAULT_SOURCE = project_path("assets/datasets/pretrain/SynthTabNet/sparse/synthetic_data.jsonl")
DEFAULT_OUTPUT = project_path("reproduction/data/normalized/synthtabnet_smoke.jsonl")


def convert_record(record: dict, source_path: Path, row_index: int) -> NormalizedTableSample:
    split = str(record["split"]).lower()
    source_key = f"{split}:{record['imgid']}"
    filename = str(record["filename"])
    image = source_path.parent / "images" / split / filename
    structure = record["html"]["structure"]["tokens"]
    cells = record["html"]["cells"]
    html = reconstruct_html(structure, cells)
    sample_id = make_sample_id("synthtabnet", source_key, f"images/{split}/{filename}", html)
    annotations = [
        {
            key: cell[key]
            for key in ("cell_id", "is_header", "span", "bbox")
            if key in cell
        }
        for cell in cells
    ]
    return NormalizedTableSample(
        sample_id=sample_id,
        source="synthtabnet",
        split=split,
        image_path=project_relative(image),
        html=html,
        metadata={
            "source_key": source_key,
            "source_jsonl": project_relative(source_path),
            "source_row_index": row_index,
            "original_filename": filename,
            "imgid": record["imgid"],
            "cell_annotations": annotations,
        },
    )


def run(source: Path, output: Path, limit: int, restart: bool = False) -> dict:
    assert_data_usage("synthtabnet", "pretrain")
    session = ConversionSession(output, "synthtabnet", restart=restart)
    scanned = skipped = 0
    if limit > 0 and len(session.sample_ids) >= limit:
        return session.write_summary(scanned=scanned, skipped=skipped)

    with source.open(encoding="utf-8") as input_file:
        for row_index, line in enumerate(input_file):
            scanned += 1
            source_key = f"row:{row_index}"
            try:
                record = json.loads(line)
                source_key = f"{str(record['split']).lower()}:{record['imgid']}"
                if session.has_source_key(source_key):
                    skipped += 1
                    continue
                sample = convert_record(record, source, row_index)
                validate_image(project_path(sample.image_path))
                session.append(sample)
            except Exception as exc:
                session.record_failure(source_key, exc)
            if limit > 0 and len(session.sample_ids) >= limit:
                break
    return session.write_summary(scanned=scanned, skipped=skipped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=30, help="target total records; 0 means all")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.source.resolve(), args.output.resolve(), args.limit, args.restart), indent=2))


if __name__ == "__main__":
    main()
