"""Canonical data models shared by the OCR engine and runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PageInput:
    """A single, consistently rendered document page."""

    document_id: str
    source_path: Path
    page_number: int
    image_path: Path
    image_sha256: str
    dpi: int


@dataclass
class OCRResult:
    """Normalized result from one engine for one page."""

    document_id: str
    page_number: int
    engine: str
    engine_version: str | None
    backend: str | None
    status: str
    text: str
    elapsed_ms: int
    # Normalized 0-100 across engines; confidence_scores keeps the raw
    # provider payload (Mistral's native values are 0-1).
    confidence: float | None = None
    confidence_scores: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OCRResult:
        """Restore a serialized result, ignoring unknown (newer/older) fields."""

        known_fields = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known_fields})
