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
class PageAlterations:
    """Handwritten-alteration detection for one page (vision model verdict).

    The verdict is deliberately flag-only. The vision model dependably
    identifies WHICH page/clause carries a hand alteration, but its value
    transcriptions proved unreliable in benchmarks, so entries carry only
    ``clause`` and ``kind`` — never the struck or replacement values.
    Consumers route flagged pages to human review of the actual scan.

    ``status`` mirrors OCRResult statuses ("success", "unavailable",
    "parse_error", "rate_limited", "auth_error", "timeout", "crash").
    Detection soft-fails: a non-success status annotates the page, it never
    fails the page or document.
    """

    status: str
    model: str
    # Flag-only entries: {"clause": ..., "kind": "dollar_amount|date|other"},
    # validated and bounded. Value transcriptions are stripped before this
    # is built.
    alterations: list[dict[str, Any]] = field(default_factory=list)
    # Derived from ``alterations`` on success (never taken from the model,
    # so it cannot contradict ``flagged``); None on failed detections.
    none_found: bool | None = None
    # Second-pass adversarial verification of a flagged first pass:
    # True = confirmed, False = rejected (entries cleared: blank page or
    # verifier refutation), None = not applicable (nothing flagged,
    # verification disabled, or the verification call itself failed and we
    # failed open to preserve recall).
    verified: bool | None = None
    elapsed_ms: int = 0
    transport_retries: int = 0
    error_type: str | None = None
    error_message: str | None = None

    @property
    def flagged(self) -> bool:
        """Whether the vision model reported at least one alteration."""

        return self.status == "success" and bool(self.alterations)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PageAlterations:
        """Restore a serialized result, ignoring unknown (newer/older) fields."""

        known_fields = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known_fields})


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
