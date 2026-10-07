"""Utilities for datasets whose terms prohibit automatic downloading."""

from __future__ import annotations

from pathlib import Path


class ManualDownloadError(FileNotFoundError):
    """Raised when a registration-gated dataset has not been supplied."""


def require_manual_path(
    data_dir: str | Path,
    expected_path: str | Path,
    *,
    homepage: str,
    instructions: str = "",
) -> Path:
    """Return a user-supplied file/directory or raise an actionable error."""
    root = Path(data_dir).expanduser().resolve()
    expected = Path(expected_path)
    path = expected if expected.is_absolute() else root / expected
    if path.exists():
        return path
    detail = f" {instructions.strip()}" if instructions.strip() else ""
    raise ManualDownloadError(
        f"This dataset must be downloaded manually from {homepage}. "
        f"Place it at {path}.{detail}"
    )
