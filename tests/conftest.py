"""Shared fixtures: PIL-built PDFs/pages and a deterministic fake engine."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from ocr_engine.models import OCRResult, PageInput


@pytest.fixture(autouse=True)
def _no_real_mistral_key(monkeypatch):
    """Tests must never reach the real API via a developer's exported key.

    Vision detection (policy default: enabled) soft-fails to "unavailable"
    without touching the SDK when the key is absent; tests that need a key
    set their own after this runs.
    """

    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)


def draw_text_like_page(path: Path) -> Path:
    """Create a small PNG that reads as a text page."""

    image = Image.new("L", (400, 500), color=255)
    drawing = ImageDraw.Draw(image)
    for row in range(40, 460, 24):
        drawing.rectangle([40, row, 360, row + 10], fill=0)
    image.save(path, format="PNG")
    return path


def build_pdf(path: Path, pages: int = 1) -> Path:
    """Build a real multi-page PDF (parses with pdfinfo/pdftoppm) via Pillow."""

    images = []
    for _ in range(pages):
        image = Image.new("L", (400, 500), color=255)
        drawing = ImageDraw.Draw(image)
        for row in range(40, 460, 24):
            drawing.rectangle([40, row, 360, row + 10], fill=0)
        images.append(image.convert("RGB"))
    images[0].save(path, format="PDF", save_all=True, append_images=images[1:])
    return path


def page_input_for(image_path: Path, page_number: int = 1) -> PageInput:
    return PageInput(
        document_id="doc-test",
        source_path=image_path,
        page_number=page_number,
        image_path=image_path,
        image_sha256="0" * 64,
        dpi=300,
    )


def word_confidence_values(word_confidences: list[float]) -> dict:
    """Build the real Mistral values shape from a list of word confidences."""

    return {
        "average_page_confidence_score": sum(word_confidences) / len(word_confidences),
        "minimum_page_confidence_score": min(word_confidences),
        "word_confidence_scores": [
            {"text": f"w{index}", "confidence": value, "start_index": index}
            for index, value in enumerate(word_confidences)
        ],
    }


class FakeEngine:
    """Deterministic engine for runner tests; records the pages it saw.

    ``word_confidences`` / ``page_signals`` script the review-flag inputs;
    ``per_page`` overrides any of text/word_confidences/page_signals for
    specific page numbers.
    """

    def __init__(
        self,
        name: str,
        *,
        confidence: float | None = 90.0,
        text: str = "recognized text " * 5,
        status: str = "success",
        word_confidences: list[float] | None = None,
        page_signals: dict | None = None,
        per_page: dict[int, dict] | None = None,
    ) -> None:
        self.name = name
        self.confidence = confidence
        self.text = text
        self.status = status
        self.word_confidences = word_confidences
        self.page_signals = page_signals
        self.per_page = per_page or {}
        self.calls: list[int] = []

    @classmethod
    def availability(cls):
        return True, "fake"

    def extract(self, page: PageInput) -> OCRResult:
        self.calls.append(page.page_number)
        started = time.perf_counter()
        if self.status != "success":
            return OCRResult(
                document_id=page.document_id,
                page_number=page.page_number,
                engine=self.name,
                engine_version="fake",
                backend="fake",
                status=self.status,
                text="",
                elapsed_ms=1,
                error_type=self.status,
                error_message=f"{self.name} simulated {self.status}",
            )
        overrides = self.per_page.get(page.page_number, {})
        word_confidences = overrides.get("word_confidences", self.word_confidences)
        page_signals = overrides.get("page_signals", self.page_signals)
        metadata = {"page_signals": page_signals} if page_signals is not None else {}
        return OCRResult(
            document_id=page.document_id,
            page_number=page.page_number,
            engine=self.name,
            engine_version="fake",
            backend="fake",
            status="success",
            text=overrides.get("text", self.text),
            elapsed_ms=max(1, round((time.perf_counter() - started) * 1000)),
            confidence=self.confidence,
            # Mirrors the Mistral adapter: OCRResult.confidence is 0-100,
            # while the raw provider payload in confidence_scores is 0-1.
            confidence_scores={
                "granularity": "word",
                "scale": "0-1",
                "values": (
                    word_confidence_values(word_confidences)
                    if word_confidences
                    else []
                ),
            },
            metadata=metadata,
        )
