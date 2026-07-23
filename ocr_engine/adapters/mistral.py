"""Mistral OCR adapter using an inline page image, not the Files API."""

from __future__ import annotations

import logging
import os
import random
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from ocr_engine.adapters.base import (
    OCREngine,
    failed_result,
    image_data_url,
    module_available,
)
from ocr_engine.models import OCRResult, PageInput

logger = logging.getLogger(__name__)

MISTRAL_MAX_ATTEMPTS = 3
MISTRAL_RETRY_DELAYS_SECONDS = (1, 3)
# Rate limits outlive a 1s pause; back off harder before burning an attempt.
MISTRAL_RATE_LIMIT_DELAYS_SECONDS = (5, 15)

# Own the per-attempt deadline. Without it the SDK falls back to an
# undocumented internal default (300s), which also bounds how long a caller
# abort can be forced to wait for in-flight pages to drain.
MISTRAL_ATTEMPT_TIMEOUT_MS = 120_000

# Bounds for the preserved per-page layout signals (review-flag inputs).
MAX_SERIALIZED_BLOCKS = 300
MAX_SERIALIZED_IMAGES = 50
SIGNATURE_CONTENT_MAX_CHARS = 200


class MistralEngine(OCREngine):
    """Call Mistral OCR with the same rendered image used by local engines."""

    name = "mistral"
    remote = True

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        """Report whether the SDK and API key are present."""

        if not module_available("mistralai"):
            return False, "the mistralai package is not installed"
        if not os.getenv("MISTRAL_API_KEY"):
            return False, "MISTRAL_API_KEY is not set"
        return True, "Mistral SDK and API key found"

    @staticmethod
    def version() -> str | None:
        """Return the installed SDK version."""

        try:
            return version("mistralai")
        except PackageNotFoundError:
            return None

    def extract(self, page: PageInput) -> OCRResult:
        """Process one page through mistral-ocr-latest."""

        started = time.perf_counter()
        available, message = self.availability()
        if not available:
            return failed_result(page, self.name, "unavailable", message)

        try:
            response, transport_retries = _process_ocr_with_retries(
                api_key=os.environ["MISTRAL_API_KEY"],
                image_url=image_data_url(page.image_path),
            )
            response_pages = list(response.pages)
            if not response_pages:
                raise ValueError("Mistral returned no pages")
            response_page = response_pages[0]
            confidence = _page_confidence(response_page)
            text = response_page.markdown or ""
        except Exception as exc:  # SDK exposes several transport exception classes.
            elapsed = round((time.perf_counter() - started) * 1000)
            status = _mistral_error_status(exc)
            return failed_result(
                page,
                self.name,
                status,
                exc,
                elapsed_ms=elapsed,
                backend="mistral-ocr-latest",
                engine_version=self.version(),
            )

        elapsed = round((time.perf_counter() - started) * 1000)
        return OCRResult(
            document_id=page.document_id,
            page_number=page.page_number,
            engine=self.name,
            engine_version=self.version(),
            backend="mistral-ocr-latest",
            status="success",
            text=text,
            elapsed_ms=elapsed,
            confidence=confidence,
            confidence_scores={
                "granularity": "word",
                "scale": "0-1",
                "values": _serialize_confidence_scores(response_page),
            },
            metadata={
                "transport_retries": transport_retries,
                "page_signals": _serialize_page_signals(response_page),
            },
        )


def _process_ocr_with_retries(*, api_key: str, image_url: str) -> tuple[Any, int]:
    """Retry transient OCR transport failures with a fresh SDK client."""

    # pylint: disable-next=import-outside-toplevel,import-error
    from mistralai.client import Mistral

    for attempt in range(MISTRAL_MAX_ATTEMPTS):
        try:
            client = Mistral(api_key=api_key)
            response = client.ocr.process(
                model="mistral-ocr-latest",
                document={"type": "image_url", "image_url": image_url},
                confidence_scores_granularity="word",
                include_blocks=True,
                include_image_base64=False,
                timeout_ms=MISTRAL_ATTEMPT_TIMEOUT_MS,
            )
            return response, attempt
        except Exception as exc:
            final_attempt = attempt == MISTRAL_MAX_ATTEMPTS - 1
            if final_attempt or not _is_retryable_mistral_error(exc):
                raise
            delay = _retry_delay(exc, attempt)
            logger.debug(
                "Mistral attempt %d/%d failed (%s); retrying in %.1fs",
                attempt + 1,
                MISTRAL_MAX_ATTEMPTS,
                type(exc).__name__,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError("Mistral retry loop ended without a response")


def _retry_delay(error: Exception, attempt: int) -> float:
    """Backoff before the next attempt; jittered so workers don't collide."""

    if _error_status_code(error) == 429:
        delays = MISTRAL_RATE_LIMIT_DELAYS_SECONDS
    else:
        delays = MISTRAL_RETRY_DELAYS_SECONDS
    # Clamp so growing MISTRAL_MAX_ATTEMPTS can never IndexError mid-retry.
    base = delays[min(attempt, len(delays) - 1)]
    return base + random.uniform(0, 1)


def _error_status_code(error: Exception) -> int | None:
    """The HTTP status the SDK attached to the error, when there is one."""

    code = getattr(error, "status_code", None)
    if isinstance(code, bool) or not isinstance(code, int):
        return None
    return code


def _is_retryable_mistral_error(error: Exception) -> bool:
    """Identify connection and temporary provider failures safe to retry.

    Prefer the parsed HTTP status; only fall back to transport-level
    heuristics for errors that never reached the server (a bare "500" in a
    message body must not trigger retries of a permanent failure).
    """

    code = _error_status_code(error)
    if code is not None:
        return code == 429 or 500 <= code <= 599
    error_type = type(error).__name__.lower()
    message = str(error).lower()
    retryable_types = ("connect", "connection", "network", "readerror", "timeout")
    retryable_messages = (
        "broken pipe",
        "connection reset",
        "connection refused",
        "temporarily unavailable",
        "timed out",
        "timeout",
    )
    return any(token in error_type for token in retryable_types) or any(
        token in message for token in retryable_messages
    )


def _page_confidence(page: Any) -> float | None:
    """Extract Mistral's page-average confidence when returned."""

    scores = getattr(page, "confidence_scores", None)
    if scores is None:
        return None
    if isinstance(scores, dict):
        value = scores.get("average_page_confidence_score")
    else:
        value = getattr(scores, "average_page_confidence_score", None)
    return float(value) * 100 if value is not None else None


def _serialize_confidence_scores(page: Any) -> Any:
    """Preserve the complete provider confidence payload for one page."""

    scores = getattr(page, "confidence_scores", None)
    if scores is None:
        return []
    if hasattr(scores, "model_dump"):
        return scores.model_dump(mode="json")
    return scores


def _serialize_page_signals(page: Any) -> dict[str, Any]:
    """Preserve the page's layout signals for review-flag detection.

    Bounded and content-light: block text is reduced to a length except for
    signature blocks, whose transcribed name is the signal. Never raises —
    the OCRResult carrying this is built outside extract()'s try block, so a
    serializer error here must degrade, not doom the page.
    """

    try:
        blocks = getattr(page, "blocks", None)
        blocks = blocks if isinstance(blocks, list) else []
        images = getattr(page, "images", None)
        images = images if isinstance(images, list) else []
        dimensions = getattr(page, "dimensions", None)

        signals: dict[str, Any] = {
            "dimensions": (
                {
                    "dpi": getattr(dimensions, "dpi", None),
                    "height": getattr(dimensions, "height", None),
                    "width": getattr(dimensions, "width", None),
                }
                if dimensions is not None
                else None
            ),
            "blocks": [_serialize_block_safe(block) for block in blocks[:MAX_SERIALIZED_BLOCKS]],
            "images": [
                {"id": getattr(image, "id", None), "bbox": _bbox(image)}
                for image in images[:MAX_SERIALIZED_IMAGES]
            ],
        }
        if len(blocks) > MAX_SERIALIZED_BLOCKS:
            signals["blocks_truncated"] = True
        if len(images) > MAX_SERIALIZED_IMAGES:
            signals["images_truncated"] = True
        return signals
    except Exception as exc:
        return {"serialization_error": f"{type(exc).__name__}: {exc}"}


def _serialize_block_safe(block: Any) -> dict[str, Any]:
    """One block, or a marker entry — one bad block must not erase the rest
    (a poisoned signals dict would silently hide real signature blocks)."""

    try:
        return _serialize_block(block)
    except Exception as exc:
        return {"type": "SERIALIZATION_ERROR", "error": f"{type(exc).__name__}: {exc}"}


def _serialize_block(block: Any) -> dict[str, Any]:
    """One block's type, position, and (for signatures) transcribed content."""

    block_type = getattr(block, "type", None) or "UNKNOWN"
    content = getattr(block, "content", "") or ""
    serialized: dict[str, Any] = {
        "type": block_type,
        "bbox": _bbox(block),
        "content_chars": len(content),
    }
    if block_type == "signature":
        # The transcribed signer name (or "" when illegible) is the signal.
        serialized["content"] = content[:SIGNATURE_CONTENT_MAX_CHARS]
    return serialized


def _bbox(item: Any) -> list | None:
    """The item's bounding box, or None when any coordinate is absent."""

    coordinates = [
        getattr(item, "top_left_x", None),
        getattr(item, "top_left_y", None),
        getattr(item, "bottom_right_x", None),
        getattr(item, "bottom_right_y", None),
    ]
    if any(value is None for value in coordinates):
        return None
    return coordinates


def _mistral_error_status(error: Exception) -> str:
    """Map common API failures to stable statuses."""

    code = _error_status_code(error)
    if code in (401, 403):
        return "auth_error"
    if code == 429:
        return "rate_limited"
    message = str(error).lower()
    if code is None and ("401" in message or "authentication" in message):
        return "auth_error"
    if code is None and ("429" in message or "rate limit" in message):
        return "rate_limited"
    if "timeout" in message or "timed out" in message:
        return "timeout"
    return "crash"
