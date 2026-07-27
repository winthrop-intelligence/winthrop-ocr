"""Pure OCR library: the Mistral adapter, page rendering, and per-profile policy.

This package has no service dependencies (FastAPI/Celery/Redis) so any repo
can install it from a git tag and run OCR in-process.

The supported import surface is this package root::

    from ocr_engine import OcrDocumentError, ocr_document

Submodule paths are internal layout and may move between minor versions.
"""

import logging

from ocr_engine.document import OcrDocumentError, ocr_document
from ocr_engine.models import OCRResult, PageAlterations
from ocr_engine.policy import POLICY_VERSION, OcrPolicy, resolve_policy
from ocr_engine.review import PageReviewFlags, detect_review_flags
from ocr_engine.runner import OcrDocumentResult, PageOutcome

__all__ = [
    "OCRResult",
    "OcrDocumentError",
    "OcrDocumentResult",
    "OcrPolicy",
    "POLICY_VERSION",
    "PageAlterations",
    "PageOutcome",
    "PageReviewFlags",
    "detect_review_flags",
    "ocr_document",
    "resolve_policy",
]

# Library convention: emit nothing unless the consumer configures logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())
