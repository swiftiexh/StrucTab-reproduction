"""Canonical training-record schema shared by all StrucTab data stages."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping

from bs4 import BeautifulSoup


SCHEMA_VERSION = 1
TRAINING_SOURCES = frozenset({"synthtabnet", "fintabnet", "pubtables"})
_SAMPLE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*-[0-9a-f]{8,64}$")


def _validate_image_path(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("image_path must be a non-empty string")
    value = value.strip().replace("\\", "/")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("image_path must be project-relative and may not contain '..'")
    return path.as_posix()


def validate_single_table_html(html: str) -> str:
    """Validate that *html* is one complete table, with only whitespace outside it."""
    if not isinstance(html, str) or not html.strip():
        raise ValueError("html must be a non-empty string")
    html = html.strip()
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    if len(tables) != 1:
        raise ValueError(f"html must contain exactly one table, found {len(tables)}")
    table = tables[0]
    if table.find_parent("table") is not None:
        raise ValueError("nested tables are not supported")

    # BeautifulSoup may repair fragments. Require explicit outer table tags and
    # reject non-whitespace siblings so malformed fragments cannot pass silently.
    if not re.match(r"(?is)^<table(?:\s[^>]*)?>", html) or not re.search(
        r"(?is)</table>\s*$", html
    ):
        raise ValueError("html must be a complete outer <table>...</table> document")
    if len(re.findall(r"(?is)<table(?:\s[^>]*)?>", html)) != 1 or len(
        re.findall(r"(?is)</table>", html)
    ) != 1:
        raise ValueError("html must contain exactly one explicit table pair")
    return html


@dataclass(frozen=True)
class NormalizedTableSample:
    """Versioned, JSON-serializable record used by later training phases."""

    sample_id: str
    source: str
    split: str
    image_path: str
    html: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source = self.source.strip().lower() if isinstance(self.source, str) else ""
        if source not in TRAINING_SOURCES:
            raise ValueError(f"unsupported training source: {self.source!r}")
        if not isinstance(self.sample_id, str) or not _SAMPLE_ID_RE.fullmatch(self.sample_id):
            raise ValueError(f"invalid sample_id: {self.sample_id!r}")
        if not isinstance(self.split, str) or not self.split.strip():
            raise ValueError("split must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")

        image_path = _validate_image_path(self.image_path)
        html = validate_single_table_html(self.html)
        metadata = dict(self.metadata)
        metadata.setdefault("schema_version", SCHEMA_VERSION)
        if metadata["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version: {metadata['schema_version']!r}; expected {SCHEMA_VERSION}"
            )
        try:
            json.dumps(metadata, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON-serializable without NaN") from exc

        object.__setattr__(self, "source", source)
        object.__setattr__(self, "split", self.split.strip().lower())
        object.__setattr__(self, "image_path", image_path)
        object.__setattr__(self, "html", html)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source": self.source,
            "split": self.split,
            "image_path": self.image_path,
            "html": self.html,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NormalizedTableSample":
        expected = {"sample_id", "source", "split", "image_path", "html", "metadata"}
        missing = expected - set(value)
        extra = set(value) - expected
        if missing or extra:
            raise ValueError(f"schema keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        return cls(**{key: value[key] for key in expected})
