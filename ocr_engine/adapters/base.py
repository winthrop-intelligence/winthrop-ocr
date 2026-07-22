"""Shared OCR engine interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from ocr_engine.models import OCRResult, PageInput


class OCREngine(ABC):
    """Interface implemented by every OCR adapter."""

    name: str
    remote: bool = False

    @abstractmethod
    def extract(self, page: PageInput) -> OCRResult:
        """Extract text from one rendered page."""

    @classmethod
    @abstractmethod
    def availability(cls) -> tuple[bool, str]:
        """Return whether the adapter can run and a diagnostic message."""


def module_available(module: str) -> bool:
    """Return whether a Python module can be imported."""

    return find_spec(module) is not None


def failed_result(
    page: PageInput,
    engine: str,
    status: str,
    error: Exception | str,
    *,
    elapsed_ms: int = 0,
    backend: str | None = None,
    engine_version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> OCRResult:
    """Create a normalized failed result without raising."""

    message = str(error)
    error_type = error.__class__.__name__ if isinstance(error, Exception) else status
    return OCRResult(
        document_id=page.document_id,
        page_number=page.page_number,
        engine=engine,
        engine_version=engine_version,
        backend=backend,
        status=status,
        text="",
        elapsed_ms=elapsed_ms,
        error_type=error_type,
        error_message=message[:2000],
        metadata=metadata or {},
    )


# Inline data URLs travel in the request body; beyond this the provider
# rejects the request (or worse, times it out). A 300-DPI letter page PNG is
# ~2-8 MB, so this only trips on extreme DPI/page-size combinations.
IMAGE_MAX_BYTES = 25 * 1024 * 1024

_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def image_data_url(path: Path) -> str:
    """Encode a local image as a base64 data URL.

    Raises ValueError when the image exceeds ``IMAGE_MAX_BYTES`` — a clear
    "lower the dpi" signal instead of an opaque provider-side failure.
    """

    import base64  # pylint: disable=import-outside-toplevel

    size = path.stat().st_size
    if size > IMAGE_MAX_BYTES:
        raise ValueError(
            f"page image is {size} bytes (limit {IMAGE_MAX_BYTES}); "
            "lower the profile dpi"
        )
    media_type = _MEDIA_TYPES.get(path.suffix.lower(), "image/png")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"
