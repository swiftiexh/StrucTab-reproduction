"""Validate normalized JSONL records, images, source consistency, and unique IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reproduction.common.paths import project_path
from reproduction.data.conversion import validate_image
from reproduction.data.schema import NormalizedTableSample


DEFAULT_INPUTS = [
    project_path("reproduction/data/normalized/synthtabnet_smoke.jsonl"),
    project_path("reproduction/data/normalized/fintabnet_smoke.jsonl"),
    project_path("reproduction/data/normalized/pubtables_smoke.jsonl"),
]


def validate_files(paths: list[Path], expected_per_file: int | None = None) -> dict:
    all_ids: set[str] = set()
    counts: dict[str, int] = {}
    for path in paths:
        count = 0
        source: str | None = None
        source_keys: set[str] = set()
        with path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, 1):
                sample = NormalizedTableSample.from_dict(json.loads(line))
                source = source or sample.source
                if sample.source != source:
                    raise ValueError(f"mixed sources in {path}:{line_number}")
                source_key = str(sample.metadata.get("source_key", ""))
                if not source_key or source_key in source_keys:
                    raise ValueError(f"missing/duplicate source_key in {path}:{line_number}")
                if sample.sample_id in all_ids:
                    raise ValueError(f"duplicate global sample_id: {sample.sample_id}")
                validate_image(project_path(sample.image_path))
                source_keys.add(source_key)
                all_ids.add(sample.sample_id)
                count += 1
        if expected_per_file is not None and count != expected_per_file:
            raise ValueError(f"{path} contains {count}, expected {expected_per_file}")
        counts[source or path.stem] = count
    return {"status": "ok", "total_records": sum(counts.values()), "counts": counts, "unique_ids": len(all_ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--expected-per-file", type=int, default=30)
    args = parser.parse_args()
    result = validate_files([path.resolve() for path in args.inputs], args.expected_per_file)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
