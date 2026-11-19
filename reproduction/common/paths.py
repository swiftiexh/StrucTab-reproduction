"""Project-relative path helpers.

All persisted configuration paths are relative to ``PROJECT_ROOT`` so that the
project can be moved without rewriting user-specific absolute paths.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str | Path) -> Path:
    """Resolve path components below the project root."""
    return PROJECT_ROOT.joinpath(*parts).resolve()

