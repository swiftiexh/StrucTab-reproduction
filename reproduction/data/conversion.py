"""Small resumable JSONL conversion session used by all dataset converters."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from reproduction.common.paths import PROJECT_ROOT
from reproduction.data.schema import NormalizedTableSample


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def validate_image(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"image does not exist: {path}")
    with Image.open(path) as image:
        image.verify()


def write_image_bytes(path: Path, data: bytes) -> None:
    if not isinstance(data, bytes) or not data:
        raise ValueError("embedded image bytes are empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        validate_image(path)
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    validate_image(path)


class ConversionSession:
    """Append-only output with source-key resume and traceable failures."""

    def __init__(self, output: Path, source: str, restart: bool = False):
        self.output = output
        self.source = source
        self.failures = output.with_name(output.stem + "_failures.jsonl")
        self.summary = output.with_name(output.stem + "_summary.json")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if restart:
            for path in (self.output, self.failures, self.summary):
                path.unlink(missing_ok=True)

        self.source_keys: set[str] = set()
        self.sample_ids: set[str] = set()
        self.failure_reasons: Counter[str] = Counter()
        self.existing = 0
        if self.output.exists():
            with self.output.open(encoding="utf-8") as input_file:
                for line_number, line in enumerate(input_file, 1):
                    if not line.strip():
                        continue
                    try:
                        sample = NormalizedTableSample.from_dict(json.loads(line))
                        source_key = str(sample.metadata["source_key"])
                    except Exception as exc:
                        raise ValueError(
                            f"invalid existing output at {self.output}:{line_number}: {exc}"
                        ) from exc
                    if sample.source != source:
                        raise ValueError(f"existing output contains source {sample.source!r}")
                    if source_key in self.source_keys or sample.sample_id in self.sample_ids:
                        raise ValueError("existing output contains duplicate source keys or sample IDs")
                    self.source_keys.add(source_key)
                    self.sample_ids.add(sample.sample_id)
                    self.existing += 1

    def has_source_key(self, source_key: str) -> bool:
        return str(source_key) in self.source_keys

    def append(self, sample: NormalizedTableSample) -> None:
        source_key = str(sample.metadata.get("source_key", ""))
        if not source_key:
            raise ValueError("metadata.source_key is required for resume")
        if source_key in self.source_keys or sample.sample_id in self.sample_ids:
            raise ValueError("duplicate source key or sample ID")
        with self.output.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        self.source_keys.add(source_key)
        self.sample_ids.add(sample.sample_id)

    def record_failure(self, source_key: str, exc: Exception) -> None:
        reason = type(exc).__name__
        self.failure_reasons[reason] += 1
        record = {"source": self.source, "source_key": source_key, "reason": reason, "message": str(exc)}
        with self.failures.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_summary(self, *, scanned: int, skipped: int) -> dict[str, Any]:
        sha256 = hashlib.sha256(self.output.read_bytes()).hexdigest() if self.output.exists() else None
        summary = {
            "source": self.source,
            "output": project_relative(self.output),
            "records": len(self.sample_ids),
            "existing_records_at_start": self.existing,
            "new_records": len(self.sample_ids) - self.existing,
            "source_rows_scanned": scanned,
            "already_completed_rows_skipped": skipped,
            "failures_this_run": sum(self.failure_reasons.values()),
            "failure_reasons_this_run": dict(sorted(self.failure_reasons.items())),
            "sha256": sha256,
        }
        temporary = self.summary.with_suffix(self.summary.suffix + ".tmp")
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.summary)
        return summary
