"""Stable sample identifiers for records whose image paths are not unique."""

import hashlib
import json


def _normalize_path(path: str) -> str:
    return path.strip().replace("\\", "/").removeprefix("./")


def make_sample_id(
    source: str,
    source_key: str | int,
    image_path: str,
    annotation: str,
    digest_length: int = 16,
) -> str:
    """Return a deterministic ID derived from source identity and annotation.

    ``source_key`` should be the source dataset's native ID, or its stable JSONL
    row index when no native ID exists. ``image_path`` must be source-relative,
    not an absolute machine-specific path.
    """
    source = source.strip().lower()
    source_key = str(source_key).strip()
    image_path = _normalize_path(image_path)
    if not source or not source_key or not image_path:
        raise ValueError("source, source_key and image_path must not be empty")
    if annotation is None:
        raise ValueError("annotation must not be None")
    if not 8 <= digest_length <= 64:
        raise ValueError("digest_length must be between 8 and 64")

    payload = {
        "annotation_sha256": hashlib.sha256(annotation.encode("utf-8")).hexdigest(),
        "image_path": image_path,
        "source": source,
        "source_key": source_key,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:digest_length]
    return f"{source}-{digest}"

