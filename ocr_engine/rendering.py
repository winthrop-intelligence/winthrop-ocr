"""Render PDF pages once so every OCR engine sees identical pixels."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from ocr_engine.models import PageInput

# Poppler tools decode untrusted PDFs; cap their memory (Linux; no-op
# elsewhere) so a crafted document cannot OOM the host.
POPPLER_MEMORY_LIMIT_BYTES = 1536 * 1024 * 1024

STDERR_EXCERPT_CHARS = 300


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
    normalization. Exit code 127 (tool missing) raises ``RuntimeError``
    naming the tool plus ``install_hint``; any other failure raises
    ``ValueError`` built from ``failure`` plus the stderr tail.
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
    """Render a single PDF page and return its canonical PageInput record."""

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"page-{page_number:04d}.png"
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
