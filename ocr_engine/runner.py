"""Per-page OCR execution and the document result types.

``run_page`` is the low-level building block; most consumers should use
:func:`ocr_engine.document.ocr_document`, which owns rendering, concurrency,
and strict aggregation on top of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from ocr_engine.adapters.base import OCREngine
from ocr_engine.classification import DIGITAL_TEXT_ENGINE
from ocr_engine.models import OCRResult, PageAlterations, PageInput
from ocr_engine.policy import OcrPolicy
from ocr_engine.review import PageReviewFlags, detect_review_flags
from ocr_engine.vision import detect_alterations


@dataclass
class PageOutcome:
    """Everything observed for one page."""

    page_number: int
    selected: OCRResult
    # Non-blocking human-review flags (signature page / suspected
    # handwriting), computed from the selected result's signals.
    review: PageReviewFlags = field(default_factory=PageReviewFlags)
    # Vision-model alteration detection; None when vision is disabled by
    # policy. Runs concurrently with OCR (it needs only the page image),
    # so it is present even when the page's OCR failed. Soft-fails via its
    # status field.
    alterations: PageAlterations | None = None

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
        mean_confidence blends model confidences with the exact-100 scores
        of born-digital pages.
        """

        confidences = [
            outcome.confidence for outcome in self.pages if outcome.confidence is not None
        ]
        engines_used: dict[str, int] = {}
        for outcome in self.pages:
            engines_used[outcome.selected.engine] = (
                engines_used.get(outcome.selected.engine, 0) + 1
            )
        digital_page_numbers = [
            outcome.page_number
            for outcome in self.pages
            if outcome.selected.engine == DIGITAL_TEXT_ENGINE
        ]
        return {
            "page_count": len(self.pages),
            "pages_failed": sum(
                1 for outcome in self.pages if outcome.selected.status != "success"
            ),
            "signature_pages": sum(
                1 for outcome in self.pages if outcome.review.signature_page
            ),
            "handwriting_pages": sum(
                1 for outcome in self.pages if outcome.review.handwriting_suspected
            ),
            "digital_pages": engines_used.get(DIGITAL_TEXT_ENGINE, 0),
            # Routing detail for consumers' metrics/dashboards (e.g. Sentry):
            # which pages skipped OCR and which took the render+OCR path.
            "digital_page_numbers": digital_page_numbers,
            "ocr_page_numbers": [
                outcome.page_number
                for outcome in self.pages
                if outcome.selected.engine != DIGITAL_TEXT_ENGINE
            ],
            "min_confidence": min(confidences) if confidences else None,
            "mean_confidence": (
                sum(confidences) / len(confidences) if confidences else None
            ),
            "alteration_pages": sum(
                1
                for outcome in self.pages
                if outcome.alterations is not None and outcome.alterations.flagged
            ),
            "vision_failed_pages": sum(
                1
                for outcome in self.pages
                if outcome.alterations is not None
                and outcome.alterations.status != "success"
            ),
            "vision_elapsed_ms": sum(
                outcome.alterations.elapsed_ms
                for outcome in self.pages
                if outcome.alterations is not None
            ),
            "engines_used": engines_used,
            # OCR engine time only; vision time is reported separately above.
            "elapsed_ms": sum(outcome.selected.elapsed_ms for outcome in self.pages),
        }


def run_page(
    page: PageInput,
    engines: Mapping[str, OCREngine],
    policy: OcrPolicy,
) -> PageOutcome:
    """OCR one rendered page through the configured engine."""

    # Resolve the engine before submitting vision: a bad engine name must
    # raise immediately, not after blocking on an in-flight vision call.
    engine = engines[policy.engine]
    if policy.vision_enabled:
        # Vision needs only the rendered image, never the OCR text, so both
        # network calls run concurrently: page latency is max(ocr, vision),
        # not their sum. detect_alterations never raises, so .result() is
        # safe to collect unconditionally. The nested single-thread pool is
        # deliberate: one short-lived extra thread per in-flight page
        # (bounded by the document's max_workers) is cheap for I/O-bound
        # calls, and a shared pool would add lifetime/shutdown coupling.
        with ThreadPoolExecutor(max_workers=1) as pool:
            vision_future = pool.submit(detect_alterations, page, policy)
            result = engine.extract(page)
            alterations = vision_future.result()
    else:
        result = engine.extract(page)
        alterations = None
    review = (
        detect_review_flags(result)
        if result.status == "success"
        else PageReviewFlags()
    )
    return PageOutcome(
        page_number=page.page_number,
        selected=result,
        review=review,
        alterations=alterations,
    )
