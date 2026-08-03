"""Render PDF pages once so every OCR engine sees identical pixels."""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from ocr_engine.models import PageInput

logger = logging.getLogger(__name__)

# Poppler tools decode untrusted PDFs; cap their memory (Linux; no-op
# elsewhere) so a crafted document cannot OOM the host.
POPPLER_MEMORY_LIMIT_BYTES = 1536 * 1024 * 1024

STDERR_EXCERPT_CHARS = 300

# Cap the longest rendered side so one abnormally large page (a phone-scan
# "poster" declaring 1 px = 1 pt) cannot explode into an 80+ megapixel
# raster that blows the pdftoppm timeout. 4200 px is calibrated so letter
# AND legal pages (longest side <= 14 in = 1008 pts) keep a full 300 DPI —
# only larger-than-legal pages clamp at all. The cap is absolute: a 4200 px
# output carries ample pixels for OCR whatever physical size the page
# claims, so no DPI floor is allowed to override it.
MAX_RENDER_DIM_PX = 4200

# "Page size:" for whole-document runs, "Page    N size:" under -f/-l.
_PAGE_SIZE_PATTERN = re.compile(
    r"^Page(?:\s+\d+)?\s+size:\s+([0-9.]+)\s+x\s+([0-9.]+)\s+pts",
    flags=re.MULTILINE,
)


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stderr_excerpt(exc: subprocess.CalledProcessError) -> str:
    """The tail of a failed subprocess's stderr — the actual diagnosis.

    Without it, a password-protected PDF surfaces as an opaque
    "returned non-zero exit status 1".
    """

    stderr = exc.stderr
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    stderr = (stderr or "").strip()
    if not stderr:
        return f"exit status {exc.returncode}"
    return f"exit status {exc.returncode}: {stderr[-STDERR_EXCERPT_CHARS:]}"


def document_id(path: Path) -> str:
    """Build a readable, collision-resistant document identifier."""

    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-") or "document"
    return f"{safe_stem}-{sha256_file(path)[:10]}"


def run_sandboxed(
    arguments: Sequence[str],
    *,
    timeout: int,
    failure: str,
    install_hint: str = "install Poppler",
) -> bytes:
    """Run one external tool under the memory cap; return its stdout bytes.

    The canonical process adapter for every tool that decodes untrusted
    PDFs — the Poppler CLI (pdfinfo, pdftoppm, pdfimages, pdftotext) and
    the pdfplumber vector scan: one command assembly, one error
    normalization. Environmental failures raise ``RuntimeError``: exit
    code 127 (tool missing, named with ``install_hint``), the sandbox's
    own exit codes (memory-limit setup or exec failure), and death by
    signal. Only a tool that ran to completion and failed raises
    ``ValueError`` built from ``failure`` plus the stderr tail — callers
    may read ``ValueError`` as the tool's verdict on the input itself.
    """

    # pylint: disable-next=import-outside-toplevel
    from ocr_engine.adapters import subprocess_runner

    tool = arguments[0]
    command = [
        sys.executable,
        "-m",
        "ocr_engine.adapters.subprocess_runner",
        "--memory-limit-bytes",
        str(POPPLER_MEMORY_LIMIT_BYTES),
        "--",
        *arguments,
    ]
    try:
        return subprocess.run(
            command, check=True, capture_output=True, timeout=timeout
        ).stdout
    except subprocess.CalledProcessError as exc:
        if exc.returncode == subprocess_runner.COMMAND_NOT_FOUND_EXIT_CODE:
            raise RuntimeError(f"{tool} is required; {install_hint}") from exc
        sandbox_failures = (
            subprocess_runner.MEMORY_LIMIT_SETUP_FAILURE_EXIT_CODE,
            subprocess_runner.COMMAND_EXEC_FAILURE_EXIT_CODE,
        )
        if exc.returncode in sandbox_failures or exc.returncode < 0:
            raise RuntimeError(
                f"sandbox could not run {tool} ({_stderr_excerpt(exc)})"
            ) from exc
        raise ValueError(f"{failure} ({_stderr_excerpt(exc)})") from exc


def pdf_page_count(pdf_path: Path) -> int:
    """Read the PDF page count using Poppler's pdfinfo (memory-capped)."""

    output = run_sandboxed(
        ["pdfinfo", str(pdf_path)],
        timeout=30,
        failure=f"pdfinfo could not read {pdf_path.name}",
    ).decode("utf-8", errors="replace")
    match = re.search(r"^Pages:\s+(\d+)\s*$", output, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"could not read page count from {pdf_path}")
    return int(match.group(1))


def pdf_page_size(pdf_path: Path, page_number: int) -> tuple[float, float]:
    """Read one page's declared size in points via pdfinfo (memory-capped)."""

    output = run_sandboxed(
        ["pdfinfo", "-f", str(page_number), "-l", str(page_number), str(pdf_path)],
        timeout=30,
        failure=f"pdfinfo could not read page {page_number} of {pdf_path.name}",
    ).decode("utf-8", errors="replace")
    match = _PAGE_SIZE_PATTERN.search(output)
    if not match:
        raise ValueError(
            f"could not read page {page_number} size from {pdf_path.name}"
        )
    return float(match.group(1)), float(match.group(2))


def bounded_dpi(width_pts: float, height_pts: float, requested_dpi: int) -> int:
    """The largest DPI (never above requested) that keeps the longest
    rendered side within ``MAX_RENDER_DIM_PX``.

    Floored only at pdftoppm's minimum of 1 DPI, which holds the cap for
    any page up to 4200 inches — 25x the PDF spec's own 200-inch page
    limit. Anything beyond that renders at 1 DPI and is bounded by the
    sandbox's memory cap and timeout like every other pathological input.
    """

    longest_pts = max(width_pts, height_pts)
    if longest_pts <= 0:
        return requested_dpi
    cap = int(MAX_RENDER_DIM_PX * 72 / longest_pts)
    return min(requested_dpi, max(1, cap))


def _effective_render_dpi(
    pdf_path: Path, page_number: int, requested_dpi: int
) -> int:
    """Clamp the render DPI to the page's physical size.

    The size probe is best-effort: if pdfinfo cannot report this page's
    size, keep the requested DPI and let the render itself succeed or
    fail — the probe must never take down a page the renderer could
    have handled.
    """

    try:
        width_pts, height_pts = pdf_page_size(pdf_path, page_number)
    except Exception:  # pylint: disable=broad-exception-caught
        logger.debug(
            "Page-size probe failed for page %d of %s; keeping %d DPI",
            page_number,
            pdf_path.name,
            requested_dpi,
        )
        return requested_dpi
    dpi = bounded_dpi(width_pts, height_pts, requested_dpi)
    if dpi < requested_dpi:
        logger.warning(
            "Clamping render DPI for page %d of %s: %.0f x %.0f pts "
            "(%.1f x %.1f in) would exceed %d px at %d DPI; rendering at %d DPI",
            page_number,
            pdf_path.name,
            width_pts,
            height_pts,
            width_pts / 72,
            height_pts / 72,
            MAX_RENDER_DIM_PX,
            requested_dpi,
            dpi,
        )
    return dpi


def render_pdf_page(
    pdf_path: Path, page_number: int, output_path: Path, dpi: int
) -> None:
    """Render one PDF page to a stable RGB PNG via memory-bounded pdftoppm."""

    prefix = output_path.with_suffix("")
    run_sandboxed(
        [
            "pdftoppm",
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-r",
            str(dpi),
            "-png",
            "-singlefile",
            str(pdf_path),
            str(prefix),
        ],
        timeout=120,
        failure=f"pdftoppm failed on page {page_number}",
    )


def render_page_input(
    pdf_path: Path,
    page_number: int,
    output_dir: Path,
    *,
    dpi: int = 300,
    identifier: str | None = None,
) -> PageInput:
    """Render a single PDF page and return its canonical PageInput record.

    ``dpi`` is the requested ceiling; abnormally large pages render at a
    lower effective DPI (see ``bounded_dpi``) and the returned record's
    ``dpi`` field reports the value actually used.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"page-{page_number:04d}.png"
    dpi = _effective_render_dpi(pdf_path, page_number, dpi)
    render_pdf_page(pdf_path, page_number, image_path, dpi)
    return PageInput(
        document_id=identifier or document_id(pdf_path),
        source_path=pdf_path.resolve(),
        page_number=page_number,
        image_path=image_path,
        image_sha256=sha256_file(image_path),
        dpi=dpi,
    )


def image_page_input(
    image_source: Path,
    output_dir: Path,
    *,
    dpi: int = 300,
    identifier: str | None = None,
) -> PageInput:
    """Convert a standalone image into a single-page PageInput record.

    Detection is content-based (Pillow sniffs the format), not suffix-based:
    service spool files carry no extension.
    """

    # pylint: disable-next=import-outside-toplevel
    from PIL import Image, UnidentifiedImageError

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "page-0001.png"
    try:
        with Image.open(image_source) as image:
            # A silently-dropped frame would violate the library's strict
            # "returned result is complete" promise.
            if getattr(image, "n_frames", 1) > 1:
                raise ValueError(
                    f"multi-frame image {image_source.name} is not supported; "
                    "convert it to a PDF"
                )
            image.convert("RGB").save(image_path, format="PNG")
    except UnidentifiedImageError as exc:
        raise ValueError(f"unsupported input type: {image_source.name}") from exc
    return PageInput(
        document_id=identifier or document_id(image_source),
        source_path=image_source.resolve(),
        page_number=1,
        image_path=image_path,
        image_sha256=sha256_file(image_path),
        dpi=dpi,
    )
