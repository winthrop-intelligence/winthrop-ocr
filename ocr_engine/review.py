"""Per-page human-review flags: signature pages and suspected handwriting.

Both flags are derived from data the Mistral response already carries — no
extra API calls:

- ``signature_page`` reads Mistral's own layout classifier: any block of
  type ``signature`` (its ``content`` is the transcribed signer name, or
  ``""`` when illegible — an illegible signature is still a signature).
- ``handwriting_suspected`` is a deliberately BROAD heuristic over per-word
  confidences (each word's confidence is ``exp(mean(token_logprobs))``, 0-1;
  printed words typically score above 0.9, handwriting makes the model
  uncertain). Broad recall first, tuned later — the thresholds below are the
  tuning knobs, and every flag carries its evidence in ``signals``.

Flags are annotations: they never change the OCR text (POLICY_VERSION is
unaffected) and detection never fails a page — any internal error yields
clean flags with a ``detector_error`` note.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ocr_engine.models import OCRResult

# Printed words rarely score below 0.9, so 0.70 is a deliberately broad
# "the model was unsure about this word" cutoff.
HANDWRITING_WORD_CONFIDENCE_THRESHOLD = 0.70
# Flag the page when at least this fraction of its words are uncertain.
HANDWRITING_LOW_CONFIDENCE_RATIO = 0.10
# The ratio rule needs a denominator at least this big to be meaningful.
HANDWRITING_MIN_WORDS = 15
# Sparse-page fallback: a mostly-blank page with a few scrawled words never
# reaches the word minimum; flag it on the page-average confidence instead.
SPARSE_PAGE_AVERAGE_CONFIDENCE_FLOOR = 0.60


@dataclass(frozen=True)
class PageReviewFlags:
    """Non-blocking review flags for one page, with their evidence.

    Frozen for value semantics, but ``signals`` is a plain dict — treat the
    whole object as read-only; do not rely on hashing it.
    """

    signature_page: bool = False
    handwriting_suspected: bool = False
    signals: dict[str, Any] = field(default_factory=dict)


def detect_review_flags(result: OCRResult) -> PageReviewFlags:
    """Compute the review flags for one successful page result.

    Never raises: a page must not fail because flagging it did.
    """

    try:
        return _detect(result)
    except Exception as exc:
        return PageReviewFlags(
            signals={"detector_error": f"{type(exc).__name__}: {exc}"}
        )


def _detect(result: OCRResult) -> PageReviewFlags:
    signature_blocks = _signature_blocks(result)
    signature_page = bool(signature_blocks)

    handwriting_suspected = False
    signals: dict[str, Any] = {
        "signature_block_count": len(signature_blocks),
        "signature_names": [
            block.get("content") for block in signature_blocks if block.get("content")
        ],
        "thresholds": {
            "word_confidence": HANDWRITING_WORD_CONFIDENCE_THRESHOLD,
            "low_confidence_ratio": HANDWRITING_LOW_CONFIDENCE_RATIO,
            "min_words": HANDWRITING_MIN_WORDS,
            "sparse_average_floor": SPARSE_PAGE_AVERAGE_CONFIDENCE_FLOOR,
        },
    }

    values = (result.confidence_scores or {}).get("values")
    if isinstance(values, dict):
        word_scores = values.get("word_confidence_scores") or []
        confidences = [
            entry.get("confidence")
            for entry in word_scores
            if isinstance(entry, dict) and isinstance(entry.get("confidence"), (int, float))
        ]
        word_count = len(confidences)
        low_confidence = [
            value
            for value in confidences
            if value < HANDWRITING_WORD_CONFIDENCE_THRESHOLD
        ]
        average = values.get("average_page_confidence_score")

        signals.update(
            {
                "word_count": word_count,
                "low_confidence_word_count": len(low_confidence),
                "low_confidence_ratio": (
                    round(len(low_confidence) / word_count, 4) if word_count else None
                ),
                "average_page_confidence": average,
                "minimum_page_confidence": values.get("minimum_page_confidence_score"),
            }
        )

        if word_count >= HANDWRITING_MIN_WORDS:
            handwriting_suspected = (
                len(low_confidence) / word_count >= HANDWRITING_LOW_CONFIDENCE_RATIO
            )
        elif word_count >= 1 and isinstance(average, (int, float)):
            handwriting_suspected = average < SPARSE_PAGE_AVERAGE_CONFIDENCE_FLOOR

    return PageReviewFlags(
        signature_page=signature_page,
        handwriting_suspected=handwriting_suspected,
        signals=signals,
    )


def _signature_blocks(result: OCRResult) -> list[dict[str, Any]]:
    """Signature blocks from the preserved page signals, defensively."""

    page_signals = (result.metadata or {}).get("page_signals")
    if not isinstance(page_signals, dict):
        return []
    blocks = page_signals.get("blocks")
    if not isinstance(blocks, list):
        return []
    return [
        block
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "signature"
    ]
