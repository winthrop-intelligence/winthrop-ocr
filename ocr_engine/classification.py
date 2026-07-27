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
password-protected PDF, timeout) downgrades to "send to OCR" and never
raises past :func:`classify_document`.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ocr_engine.rendering import POPPLER_MEMORY_LIMIT_BYTES, _stderr_excerpt

logger = logging.getLogger(__name__)

# A page skips OCR only when pdftotext yields at least this many
# non-whitespace characters AND pdfimages reports zero images on the page.
# DocuSign envelope stamps run ~40-80 chars; real contract pages 300+.
MIN_DIGITAL_TEXT_CHARS = 120

# pdfimages -list scans the whole document once; pdftotext extracts one page.
PDFIMAGES_LIST_TIMEOUT_SECONDS = 60
PDFTOTEXT_PAGE_TIMEOUT_SECONDS = 20

# pdfimages -list "type" values that count as an embedded image. 'stencil'
# covers 1-bit stamp/signature masks; smask/mask rows always accompany a
# parent image row, so ignoring them loses nothing.
IMAGE_ROW_TYPES = frozenset({"image", "stencil"})

# Engine name recorded on skipped pages (surfaces in summary()["engines_used"]).
DIGITAL_TEXT_ENGINE = "digital-text"


@dataclass(frozen=True)
class PageClassification:
    """Pre-flight verdict for one PDF page."""

    page_number: int
    is_digital: bool
    text: str = ""  # pdftotext output (trailing form-feed stripped); "" unless digital
    reason: str = ""  # "digital-text" | "has-images" | "sparse-text" | "classification-error"
    char_count: int = 0  # non-whitespace chars seen by pdftotext
    image_count: int = 0  # pdfimages rows on this page
    elapsed_ms: int = 0  # this page's pdftotext time (0 when pdftotext never ran)


def classify_document(
    pdf_path: Path, page_count: int
) -> dict[int, PageClassification]:
    """Classify every page of a PDF; never raises.

    Any document-level failure yields needs-OCR verdicts for all pages;
    a page-level failure downgrades only that page.
    """

    try:
        image_counts = _pdfimages_page_counts(pdf_path)
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

    verdicts: dict[int, PageClassification] = {}
    for number in range(1, page_count + 1):
        count = image_counts.get(number, 0)
        if count > 0:
            verdicts[number] = PageClassification(
                number, is_digital=False, reason="has-images", image_count=count
            )
            continue
        try:
            started = time.monotonic()
            text = _page_text(pdf_path, number)
            elapsed_ms = int((time.monotonic() - started) * 1000)
        except Exception as exc:
            logger.warning(
                "digital-page classification failed for %s page %d; page goes "
                "to OCR: %s",
                pdf_path.name,
                number,
                exc,
            )
            verdicts[number] = PageClassification(
                number, is_digital=False, reason="classification-error"
            )
            continue
        chars = sum(1 for ch in text if not ch.isspace())
        if chars >= MIN_DIGITAL_TEXT_CHARS:
            verdicts[number] = PageClassification(
                number,
                is_digital=True,
                text=text,
                reason="digital-text",
                char_count=chars,
                elapsed_ms=elapsed_ms,
            )
        else:
            verdicts[number] = PageClassification(
                number,
                is_digital=False,
                reason="sparse-text",
                char_count=chars,
                elapsed_ms=elapsed_ms,
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

    # pylint: disable-next=import-outside-toplevel
    from ocr_engine.adapters import subprocess_runner

    command = [
        sys.executable,
        "-m",
        "ocr_engine.adapters.subprocess_runner",
        "--memory-limit-bytes",
        str(POPPLER_MEMORY_LIMIT_BYTES),
        "--",
        "pdfimages",
        "-list",
        str(pdf_path),
    ]
    try:
        output = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=PDFIMAGES_LIST_TIMEOUT_SECONDS,
        ).stdout
    except subprocess.CalledProcessError as exc:
        if exc.returncode == subprocess_runner.COMMAND_NOT_FOUND_EXIT_CODE:
            raise RuntimeError("pdfimages is required; install Poppler") from exc
        raise ValueError(
            f"pdfimages could not read {pdf_path.name} ({_stderr_excerpt(exc)})"
        ) from exc
    return _parse_pdfimages_list(output)


def _parse_pdfimages_list(output: str) -> dict[int, int]:
    """Parse ``pdfimages -list`` output into per-page image counts.

    Format: a column-header line, a dashed rule, then one whitespace-
    separated row per image. Unrecognized lines are skipped defensively.
    """

    counts: dict[int, int] = {}
    for line in output.splitlines()[2:]:
        fields = line.split()
        if len(fields) < 3 or not fields[0].isdigit():
            continue
        if fields[2] in IMAGE_ROW_TYPES:
            page = int(fields[0])
            counts[page] = counts.get(page, 0) + 1
    return counts


def _page_text(pdf_path: Path, page_number: int) -> str:
    """Extract one page's embedded digital text via pdftotext (memory-capped)."""

    # pylint: disable-next=import-outside-toplevel
    from ocr_engine.adapters import subprocess_runner

    command = [
        sys.executable,
        "-m",
        "ocr_engine.adapters.subprocess_runner",
        "--memory-limit-bytes",
        str(POPPLER_MEMORY_LIMIT_BYTES),
        "--",
        "pdftotext",
        "-f",
        str(page_number),
        "-l",
        str(page_number),
        "-enc",
        "UTF-8",
        str(pdf_path),
        "-",
    ]
    try:
        raw = subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=PDFTOTEXT_PAGE_TIMEOUT_SECONDS,
        ).stdout
    except subprocess.CalledProcessError as exc:
        if exc.returncode == subprocess_runner.COMMAND_NOT_FOUND_EXIT_CODE:
            raise RuntimeError("pdftotext is required; install Poppler") from exc
        raise ValueError(
            f"pdftotext could not read {pdf_path.name} page {page_number} "
            f"({_stderr_excerpt(exc)})"
        ) from exc
    text = raw.decode("utf-8", errors="replace")
    # pdftotext ends every page with a form-feed; OcrDocumentResult.text
    # joins pages with form-feed too, so keeping it would double-separate.
    return text.removesuffix("\f")
