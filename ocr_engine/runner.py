"""Per-page OCR execution and the document result types.

``run_page`` is the low-level building block; most consumers should use
:func:`ocr_engine.document.ocr_document`, which owns rendering, concurrency,
and strict aggregation on top of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ocr_engine.adapters.base import OCREngine
from ocr_engine.models import OCRResult, PageInput
from ocr_engine.policy import OcrPolicy


@dataclass
class PageOutcome:
    """Everything observed for one page."""

    page_number: int
    selected: OCRResult

    @property
    def confidence(self) -> float | None:
        """Confidence of the selected result (0-100 scale)."""

        return self.selected.confidence

    def trail(self) -> list[dict[str, Any]]:
        """Compact attempt record for the page (one entry per engine tried)."""

        return [
            {
                "engine": self.selected.engine,
                "engine_version": self.selected.engine_version,
                "status": self.selected.status,
                "confidence": self.selected.confidence,
                "elapsed_ms": self.selected.elapsed_ms,
                "error_type": self.selected.error_type,
                "error_message": self.selected.error_message,
            }
        ]


@dataclass
class OcrDocumentResult:
    """Ordered page outcomes for one document."""

    document_id: str
    profile: str
    policy_version: str
    pages: list[PageOutcome]
    # Hash of the fully resolved policy (profile + overrides + version), so
    # results from materially different runs (e.g. 300 vs 400 DPI) carry
    # distinguishable provenance.
    policy_fingerprint: str = ""

    @property
    def text(self) -> str:
        """Concatenated document text, pages separated by form-feed."""

        return "\f".join(outcome.selected.text for outcome in self.pages)

    def summary(self) -> dict[str, Any]:
        """Document-level rollup for logging/metrics.

        Note: elapsed_ms sums per-page engine time across concurrent pages,
        so it can exceed wall-clock — it approximates paid work, not latency.
        """

        confidences = [
            outcome.confidence for outcome in self.pages if outcome.confidence is not None
        ]
        engines_used: dict[str, int] = {}
        for outcome in self.pages:
            engines_used[outcome.selected.engine] = (
                engines_used.get(outcome.selected.engine, 0) + 1
            )
        return {
            "page_count": len(self.pages),
            "pages_failed": sum(
                1 for outcome in self.pages if outcome.selected.status != "success"
            ),
            "min_confidence": min(confidences) if confidences else None,
            "mean_confidence": (
                sum(confidences) / len(confidences) if confidences else None
            ),
            "engines_used": engines_used,
            "elapsed_ms": sum(outcome.selected.elapsed_ms for outcome in self.pages),
        }


def run_page(
    page: PageInput,
    engines: Mapping[str, OCREngine],
    policy: OcrPolicy,
) -> PageOutcome:
    """OCR one rendered page through the configured engine."""

    result = engines[policy.engine].extract(page)
    return PageOutcome(page_number=page.page_number, selected=result)
