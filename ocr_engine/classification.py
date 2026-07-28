"""Pre-flight born-digital page classification (SCR-2285).

A page skips OCR only when its PDF already carries the complete text:
at least ``MIN_DIGITAL_TEXT_CHARS`` non-whitespace characters of embedded
digital text (pdftotext), zero embedded raster images (pdfimages), AND
zero vector-drawn marks — bezier curves or markup annotations (pdfplumber
via :mod:`ocr_engine.vector_marks`).

There is deliberately NO image-size threshold: on real contracts a
DocuSign signature image covers ~1% of the page while a decorative
letterhead logo covers ~9%, so size cannot separate content-bearing
images from decoration. Any image at all routes the page to OCR — the
failure direction is an unnecessary OCR call, never lost content.

The vector-mark gate runs LAST and only for pages that would otherwise be
skipped, so scanned documents never pay its cost. It counts curves (drawn
handwriting is made of curves) and markup annotations (Ink, Stamp, ...),
while straight lines and rectangles — table borders and rules present on
virtually every contract — stay benign. Residual gap, accepted: a mark
composed purely of straight segments with no annotation entry would not
be flagged; zero pages in the 281 validated contract pages carried vector
marks of any kind.

Classification is fail-safe: every error (missing tool, unreadable or
password-protected PDF, timeout, unrecognized tool output) downgrades to
"send to OCR" and never raises past :func:`classify_document`. The whole
document runs on at most three sandboxed subprocess calls — one
``pdfimages -list``, one ``pdftotext`` covering all pages, and one vector
scan covering the would-be-skipped pages — so pre-flight latency is
bounded by three timeouts regardless of page count.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ocr_engine.rendering import run_sandboxed

logger = logging.getLogger(__name__)

# A page skips OCR only when pdftotext yields at least this many
# non-whitespace characters AND pdfimages reports zero images on the page.
# DocuSign envelope stamps run ~40-80 chars; real contract pages 300+.
MIN_DIGITAL_TEXT_CHARS = 120

# Each tool scans the whole document at most once per classification.
PDFIMAGES_LIST_TIMEOUT_SECONDS = 60
PDFTOTEXT_DOCUMENT_TIMEOUT_SECONDS = 120
VECTOR_SCAN_TIMEOUT_SECONDS = 120

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
    # "digital-text" | "has-images" | "has-vector-marks" | "sparse-text"
    # | "classification-error"
    reason: str = ""
    char_count: int = 0  # non-whitespace chars seen by pdftotext
    image_count: int = 0  # pdfimages rows on this page
    vector_mark_count: int = 0  # curves + markup annotations on this page
    elapsed_ms: int = 0  # this page's share of the document's pre-flight time


def classify_document(
    pdf_path: Path, page_count: int
) -> dict[int, PageClassification]:
    """Classify every page of a PDF; never raises.

    Any failure — any of the three tool calls, or output that does not
    parse cleanly — yields needs-OCR verdicts for every page.
    """

    try:
        image_counts = _pdfimages_page_counts(pdf_path)
        image_free = [
            number
            for number in range(1, page_count + 1)
            if image_counts.get(number, 0) == 0
        ]
        started = time.monotonic()
        page_texts = (
            _document_page_texts(pdf_path, page_count) if image_free else []
        )
        char_counts = {
            number: sum(
                1 for ch in page_texts[number - 1] if not ch.isspace()
            )
            for number in image_free
        }
        # The expensive pdfplumber gate runs only for pages that would
        # otherwise skip OCR; scanned documents never pay its cost.
        candidates = [
            number
            for number in image_free
            if char_counts[number] >= MIN_DIGITAL_TEXT_CHARS
        ]
        vector_counts = (
            _vector_mark_counts(pdf_path, candidates) if candidates else {}
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
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

    # summary()["elapsed_ms"] sums per-page times, so the document-wide
    # pdftotext + vector-scan cost is spread across the pages it served.
    elapsed_share = elapsed_ms // len(image_free) if image_free else 0

    verdicts: dict[int, PageClassification] = {}
    for number in range(1, page_count + 1):
        count = image_counts.get(number, 0)
        if count > 0:
            verdicts[number] = PageClassification(
                number, is_digital=False, reason="has-images", image_count=count
            )
            continue
        chars = char_counts[number]
        if chars < MIN_DIGITAL_TEXT_CHARS:
            verdicts[number] = PageClassification(
                number,
                is_digital=False,
                reason="sparse-text",
                char_count=chars,
                elapsed_ms=elapsed_share,
            )
            continue
        vector_marks = vector_counts.get(number, 0)
        if vector_marks > 0:
            verdicts[number] = PageClassification(
                number,
                is_digital=False,
                reason="has-vector-marks",
                char_count=chars,
                vector_mark_count=vector_marks,
                elapsed_ms=elapsed_share,
            )
            continue
        verdicts[number] = PageClassification(
            number,
            is_digital=True,
            text=page_texts[number - 1],
            reason="digital-text",
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

    output = run_sandboxed(
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

    raw = run_sandboxed(
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


def _vector_mark_counts(pdf_path: Path, page_numbers: list[int]) -> dict[int, int]:
    """Count vector-drawn marks (curves + markup annotations) per page.

    Runs :mod:`ocr_engine.vector_marks` (pdfplumber) in the same sandboxed
    subprocess harness as the Poppler tools, so the heavyweight PDF parse
    of an untrusted file happens in a memory-capped child process. Every
    requested page must appear in the response — a page the scan skipped
    is ambiguous and raises into the caller's fail-safe.
    """

    raw = run_sandboxed(
        [
            sys.executable,
            "-m",
            "ocr_engine.vector_marks",
            str(pdf_path),
            "--pages",
            ",".join(str(number) for number in page_numbers),
        ],
        timeout=VECTOR_SCAN_TIMEOUT_SECONDS,
        failure=f"vector scan could not read {pdf_path.name}",
        install_hint="reinstall winthrop-ocr",
    )
    counts = json.loads(raw.decode("utf-8"))
    missing = [
        number for number in page_numbers if str(number) not in counts
    ]
    if missing:
        raise ValueError(
            f"vector scan of {pdf_path.name} returned no verdict for "
            f"page(s) {missing}"
        )
    return {
        number: counts[str(number)]["curves"]
        + counts[str(number)]["markup_annots"]
        for number in page_numbers
    }
