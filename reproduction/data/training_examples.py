"""Schemas and builders for Stage 1 independent tasks and Stage 2 sequences."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping

from reproduction.common.ids import make_sample_id
from reproduction.common.protocol import format_sequential_target, parse_sequential_output
from reproduction.data.schema import NormalizedTableSample
from reproduction.structure.parser import parse_html_structure


TRAINING_EXAMPLE_SCHEMA_VERSION = 1
STAGE1_TASKS = ("row_col", "merge", "html")
STAGE1_PROMPTS = {
    "row_col": "请判断图中表格的行数和列数。",
    "merge": "请识别图中表格的所有合并单元格。",
    "html": "把图中的表格解析为HTML。",
}
STAGE2_PROMPT = "把图中的表格解析为HTML，并依次给出行列数分析、合并单元格分析和最终表格解析结果。"


@dataclass(frozen=True)
class TrainingExample:
    sample_id: str
    base_sample_id: str
    source: str
    split: str
    image_path: str
    stage: str
    task_type: str
    prompt: str
    target: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid = (self.stage == "stage1" and self.task_type in STAGE1_TASKS) or (
            self.stage == "stage2" and self.task_type == "sequential"
        )
        if not valid:
            raise ValueError(f"invalid stage/task combination: {self.stage}/{self.task_type}")
        for name in ("sample_id", "base_sample_id", "source", "split", "prompt", "target"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        path = PurePosixPath(self.image_path.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("image_path must be project-relative")
        metadata = dict(self.metadata)
        metadata.setdefault("schema_version", TRAINING_EXAMPLE_SCHEMA_VERSION)
        if metadata["schema_version"] != TRAINING_EXAMPLE_SCHEMA_VERSION:
            raise ValueError("unsupported training-example schema version")
        json.dumps(metadata, ensure_ascii=False, allow_nan=False)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "image_path", path.as_posix())

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "base_sample_id": self.base_sample_id,
            "source": self.source,
            "split": self.split,
            "image_path": self.image_path,
            "stage": self.stage,
            "task_type": self.task_type,
            "prompt": self.prompt,
            "target": self.target,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingExample":
        fields = {
            "sample_id", "base_sample_id", "source", "split", "image_path",
            "stage", "task_type", "prompt", "target", "metadata",
        }
        if set(value) != fields:
            raise ValueError(
                f"training-example keys mismatch: missing={sorted(fields - set(value))}, "
                f"extra={sorted(set(value) - fields)}"
            )
        return cls(**{name: value[name] for name in fields})


def _derived_id(base: NormalizedTableSample, stage: str, task_type: str, target: str) -> str:
    return make_sample_id(
        stage,
        f"{base.sample_id}:{task_type}",
        base.image_path,
        target,
    )


def _metadata(base: NormalizedTableSample, parsed: dict) -> dict[str, Any]:
    return {
        "base_source_key": base.metadata["source_key"],
        "row_count": parsed["row_count"],
        "col_count": parsed["col_count"],
        "merge_count": len(parsed["merged_cells"]),
    }


def build_stage1_examples(base: NormalizedTableSample) -> list[TrainingExample]:
    parsed = parse_html_structure(base.html)
    if not parsed["is_valid"]:
        raise ValueError("invalid base HTML: " + "; ".join(parsed["errors"]))
    targets = {
        "row_col": f"这是一个{parsed['row_count']}行{parsed['col_count']}列的表格。",
        "merge": parsed["merge_info"],
        "html": base.html,
    }
    return [
        TrainingExample(
            sample_id=_derived_id(base, "stage1", task_type, targets[task_type]),
            base_sample_id=base.sample_id,
            source=base.source,
            split=base.split,
            image_path=base.image_path,
            stage="stage1",
            task_type=task_type,
            prompt=STAGE1_PROMPTS[task_type],
            target=targets[task_type],
            metadata=_metadata(base, parsed),
        )
        for task_type in STAGE1_TASKS
    ]


def build_stage2_example(base: NormalizedTableSample) -> TrainingExample:
    parsed = parse_html_structure(base.html)
    if not parsed["is_valid"]:
        raise ValueError("invalid base HTML: " + "; ".join(parsed["errors"]))
    target = format_sequential_target(
        parsed["row_count"], parsed["col_count"], parsed["merge_info"], base.html
    )
    return TrainingExample(
        sample_id=_derived_id(base, "stage2", "sequential", target),
        base_sample_id=base.sample_id,
        source=base.source,
        split=base.split,
        image_path=base.image_path,
        stage="stage2",
        task_type="sequential",
        prompt=STAGE2_PROMPT,
        target=target,
        metadata=_metadata(base, parsed),
    )


def validate_stage2_semantics(target: str) -> dict[str, Any]:
    """Require syntactic validity and exact agreement with the target HTML."""
    protocol = parse_sequential_output(target)
    errors = list(protocol["errors"])
    if protocol["html"] is not None:
        structure = parse_html_structure(protocol["html"])
        if not structure["is_valid"]:
            errors.extend(f"html_structure: {error}" for error in structure["errors"])
        else:
            if (protocol["row_count"], protocol["col_count"]) != (
                structure["row_count"],
                structure["col_count"],
            ):
                errors.append("semantic_row_col_mismatch")
            if protocol["merge"] != structure["merge_info"]:
                errors.append("semantic_merge_mismatch")
    return {"is_valid": not errors, "errors": errors, "protocol": protocol}
