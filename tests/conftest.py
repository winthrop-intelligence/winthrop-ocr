"""Shared fixtures: PIL-built PDFs/pages and a deterministic fake engine."""

from __future__ import annotations

import time
from collections.abc import Collection, Sequence
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from ocr_engine.models import OCRResult, PageInput

# Contract-like page text comfortably above the 120 non-whitespace-char
# digital threshold (for build_digital_pdf pages).
DIGITAL_PAGE_TEXT = (
    "This Employment Agreement is entered into by and between the\n"
    "University and the Head Coach, effective July 1, 2026, and sets\n"
    "forth the terms of employment, including base salary of $340,000\n"
    "per year, supplemental compensation, and termination provisions\n"
    "as described in the sections below. Both parties agree to the\n"
    "obligations stated herein for the full term of this Agreement."
)


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


def build_digital_pdf(
    path: Path,
    page_texts: Sequence[str],
    image_on_pages: Collection[int] = (),
    curve_on_pages: Collection[int] = (),
    ink_annot_on_pages: Collection[int] = (),
) -> Path:
    """Hand-assemble a born-digital PDF: real text objects, no raster pages.

    Pillow cannot make one (its PDF pages are embedded images), so this
    writes the classic minimal PDF by hand: Catalog, Pages, a built-in
    Helvetica font, and one Page + content stream per entry in
    ``page_texts``. All page collections are 1-based:
    - ``image_on_pages``: draw a tiny 1x1 raster XObject so
      ``pdfimages -list`` reports it.
    - ``curve_on_pages``: stroke a bezier squiggle (a stylus-signature
      stand-in) — invisible to pdfimages, visible to pdfplumber.
    - ``ink_annot_on_pages``: attach an /Ink markup annotation.
    Parses with real Poppler.
    """

    image_pages = {int(number) for number in image_on_pages}
    curve_pages = {int(number) for number in curve_on_pages}
    annot_pages = {int(number) for number in ink_annot_on_pages}
    include_image = bool(image_pages)

    # Objects 1-3 (+4 for the shared image) are fixed; each page then
    # consumes a Page object, a content stream, and optionally an
    # annotation object — so page object numbers must be precomputed.
    next_object = 5 if include_image else 4
    page_objects = []
    for index in range(len(page_texts)):
        page_objects.append(next_object)
        next_object += 2 + (1 if index + 1 in annot_pages else 0)

    def escape(text: str) -> str:
        return (
            text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        )

    objects: list[bytes] = []
    kids = " ".join(f"{number} 0 R" for number in page_objects)
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_texts)} >>".encode(
            "latin-1"
        )
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    if include_image:
        objects.append(
            b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 "
            b"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Length 3 >>\n"
            b"stream\n\x80\x80\x80\nendstream"
        )

    for index, text in enumerate(page_texts):
        page_number = index + 1
        page_object = page_objects[index]
        with_image = page_number in image_pages
        resources = "<< /Font << /F1 3 0 R >>"
        if with_image:
            resources += " /XObject << /Im1 4 0 R >>"
        resources += " >>"
        page_dict = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources {resources} /Contents {page_object + 1} 0 R"
        )
        if page_number in annot_pages:
            page_dict += f" /Annots [{page_object + 2} 0 R]"
        page_dict += " >>"
        objects.append(page_dict.encode("latin-1"))
        ops = ["BT", "/F1 12 Tf", "14 TL", "72 720 Td"]
        for line_number, line in enumerate(text.split("\n")):
            if line_number:
                ops.append("T*")
            ops.append(f"({escape(line)}) Tj")
        ops.append("ET")
        if with_image:
            ops.append("q 40 0 0 40 500 706 cm /Im1 Do Q")
        if page_number in curve_pages:
            ops.append("1 w 100 200 m 120 240 140 160 160 200 c S")
        stream = "\n".join(ops).encode("latin-1")
        objects.append(
            b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)
        )
        if page_number in annot_pages:
            objects.append(
                b"<< /Type /Annot /Subtype /Ink /Rect [100 100 200 200] "
                b"/InkList [[100 100 130 150 160 110]] /F 4 >>"
            )

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"
    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("latin-1")
    path.write_bytes(bytes(out))
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
