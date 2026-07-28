"""Pre-flight born-digital page classification (SCR-2285).

A page skips OCR only when its PDF already carries the complete text:
at least ``MIN_DIGITAL_TEXT_CHARS`` non-whitespace characters of embedded
digital text (pdftotext) AND zero embedded raster images (pdfimages).

There is deliberately NO image-size threshold: on real contracts a
DocuSign signature image covers ~1% of the page while a decorative
letterhead logo covers ~9%, so size cannot separate content-bearing
images from decoration. Any image at all routes the page to OCR — the
failure direction is an unnecessary OCR call, never lost content.

Classification is fail-safe: every error (missing tool, unreadable or
password-protected PDF, timeout, unrecognized pdfimages output) downgrades
to "send to OCR" and never raises past :func:`classify_document`. The
whole document runs on two Poppler calls — one ``pdfimages -list`` and one
``pdftotext`` covering all pages — so pre-flight latency is bounded by two
subprocess timeouts regardless of page count.

Known limitation (accepted in SCR-2285): purely VECTOR-drawn marks —
ink/markup annotations or signatures drawn as paths rather than pixels —
are invisible to ``pdfimages``, so a text-rich page carrying only vector
marks classifies as digital and skips OCR and vision. Zero such pages
existed across the 281 validated contract pages (e-sign tools embed
signatures as raster images), and anything printed-and-scanned becomes
raster anyway. If vector marks show up in practice, harden by counting
``page.curves``/annotations via pdfplumber before trusting the skip.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from ocr_engine.rendering import run_poppler

logger = logging.getLogger(__name__)

# A page skips OCR only when pdftotext yields at least this many
# non-whitespace characters AND pdfimages reports zero images on the page.
# DocuSign envelope stamps run ~40-80 chars; real contract pages 300+.
MIN_DIGITAL_TEXT_CHARS = 120

# Each tool scans the whole document exactly once per classification.
PDFIMAGES_LIST_TIMEOUT_SECONDS = 60
PDFTOTEXT_DOCUMENT_TIMEOUT_SECONDS = 120

# pdfimages -list "type" values that do NOT count as an image: soft masks
# and stencil masks always accompany the parent image row they shape.
# Every other type (image, stencil, or anything a future Poppler adds)
# counts as an image — unknown content must route to OCR, never skip it.
NON_IMAGE_ROW_TYPES = frozenset({"smask", "mask"})

# Engine name recorded on skipped pages (surfaces in summary()["engines_used"]).
DIGITAL_TEXT_ENGINE = "digital-text"


@dataclass(frozen=True)
class PageClassification:
    """Pre-flight verdict for one PDF page."""

    page_number: int
    is_digital: bool
    text: str = ""  # pdftotext output for the page; "" unless digital
    reason: str = ""  # "digital-text" | "has-images" | "sparse-text" | "classification-error"
    char_count: int = 0  # non-whitespace chars seen by pdftotext
    image_count: int = 0  # pdfimages rows on this page
    elapsed_ms: int = 0  # this page's share of the document's pdftotext time


def classify_document(
    pdf_path: Path, page_count: int
) -> dict[int, PageClassification]:
    """Classify every page of a PDF; never raises.

    Any failure — either Poppler call, or output that does not parse
    cleanly — yields needs-OCR verdicts for every page.
    """

    try:
        image_counts = _pdfimages_page_counts(pdf_path)
        image_free = [
            number
            for number in range(1, page_count + 1)
            if image_counts.get(number, 0) == 0
        ]
        if image_free:
            started = time.monotonic()
            page_texts = _document_page_texts(pdf_path, page_count)
            elapsed_ms = int((time.monotonic() - started) * 1000)
        else:
            page_texts, elapsed_ms = [], 0
    except Exception as exc:
        logger.warning(
            "digital-page classification failed for %s; all pages go to OCR: %s",
            pdf_path.name,
            exc,
        )
        return {
            number: PageClassification(
                number, is_digital=False, reason="classification-error"
            )
            for number in range(1, page_count + 1)
        }

    # summary()["elapsed_ms"] sums per-page times, so the single pdftotext
    # call's cost is spread across the pages it served, not repeated.
    elapsed_share = elapsed_ms // len(image_free) if image_free else 0

    verdicts: dict[int, PageClassification] = {}
    for number in range(1, page_count + 1):
        count = image_counts.get(number, 0)
        if count > 0:
            verdicts[number] = PageClassification(
                number, is_digital=False, reason="has-images", image_count=count
            )
            continue
        text = page_texts[number - 1]
        chars = sum(1 for ch in text if not ch.isspace())
        if chars >= MIN_DIGITAL_TEXT_CHARS:
            verdicts[number] = PageClassification(
                number,
                is_digital=True,
                text=text,
                reason="digital-text",
                char_count=chars,
                elapsed_ms=elapsed_share,
            )
        else:
            verdicts[number] = PageClassification(
                number,
                is_digital=False,
                reason="sparse-text",
                char_count=chars,
                elapsed_ms=elapsed_share,
            )

    digital = sum(1 for verdict in verdicts.values() if verdict.is_digital)
    logger.info(
        "classification for %s: %d of %d page(s) digital",
        pdf_path.name,
        digital,
        page_count,
    )
    return verdicts


def _pdfimages_page_counts(pdf_path: Path) -> dict[int, int]:
    """Count embedded raster images per page via pdfimages (memory-capped)."""

    output = run_poppler(
        ["pdfimages", "-list", str(pdf_path)],
        timeout=PDFIMAGES_LIST_TIMEOUT_SECONDS,
        failure=f"pdfimages could not read {pdf_path.name}",
    ).decode("utf-8", errors="replace")
    return _parse_pdfimages_list(output)


def _parse_pdfimages_list(output: str) -> dict[int, int]:
    """Parse ``pdfimages -list`` output into per-page image counts.

    Format: a column-header line, a dashed rule, then one whitespace-
    separated row per image. The header and every data row are validated —
    a page with images MUST end up with a non-zero count, so anything
    unrecognized raises (and the caller's fail-safe routes all pages to
    OCR) rather than being silently dropped as "no images".
    """

    lines = output.splitlines()
    if len(lines) < 2:
        raise ValueError("unrecognized pdfimages -list output: missing header")
    header = lines[0].split()
    if header[:3] != ["page", "num", "type"]:
        raise ValueError(
            f"unrecognized pdfimages -list header: {lines[0].strip()[:80]!r}"
        )
    rule = lines[1].strip()
    if not rule or set(rule) != {"-"}:
        raise ValueError(
            f"unrecognized pdfimages -list rule line: {rule[:80]!r}"
        )
    counts: dict[int, int] = {}
    for line in lines[2:]:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 3 or not fields[0].isdigit():
            raise ValueError(
                f"unrecognized pdfimages -list row: {line.strip()[:80]!r}"
            )
        if fields[2] not in NON_IMAGE_ROW_TYPES:
            page = int(fields[0])
            counts[page] = counts.get(page, 0) + 1
    return counts


def _document_page_texts(pdf_path: Path, page_count: int) -> list[str]:
    """Extract every page's embedded text in ONE pdftotext call.

    pdftotext ends each page with a form-feed, so the document splits into
    exactly ``page_count`` texts plus one empty trailing segment. Any other
    shape is ambiguous — per-page attribution could be wrong — so it raises
    and the caller's fail-safe routes all pages to OCR.
    """

    raw = run_poppler(
        ["pdftotext", "-enc", "UTF-8", str(pdf_path), "-"],
        timeout=PDFTOTEXT_DOCUMENT_TIMEOUT_SECONDS,
        failure=f"pdftotext could not read {pdf_path.name}",
    )
    segments = raw.decode("utf-8", errors="replace").split("\f")
    if segments[-1] != "":
        raise ValueError(
            f"pdftotext output for {pdf_path.name} did not end with a form-feed"
        )
    pages = segments[:-1]
    if len(pages) != page_count:
        raise ValueError(
            f"pdftotext returned {len(pages)} page(s) for {pdf_path.name}, "
            f"expected {page_count}"
        )
    return pages
