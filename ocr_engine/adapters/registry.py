"""Adapter registry. Mistral OCR is the only engine."""

from __future__ import annotations

from ocr_engine.adapters.base import OCREngine
from ocr_engine.adapters.mistral import MistralEngine


def engine_registry() -> dict[str, OCREngine]:
    """Build adapter instances; the heavy SDK stays lazily imported inside."""

    return {"mistral": MistralEngine()}
