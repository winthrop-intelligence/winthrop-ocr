"""Review-flag detection: the ticket's three page types plus the edges."""

from __future__ import annotations

from ocr_engine.models import OCRResult
from ocr_engine.review import (
    HANDWRITING_LOW_CONFIDENCE_RATIO,
    HANDWRITING_MIN_WORDS,
    PageReviewFlags,
    detect_review_flags,
)

from tests.conftest import word_confidence_values


def page_result(
    *,
    word_confidences: list[float] | None = None,
    blocks: list[dict] | None = None,
    confidence_scores=None,
    metadata=None,
) -> OCRResult:
    """A success OCRResult shaped like the Mistral adapter produces."""

    if confidence_scores is None:
        confidence_scores = {
            "granularity": "word",
            "scale": "0-1",
            "values": (
                word_confidence_values(word_confidences) if word_confidences else []
            ),
        }
    if metadata is None:
        metadata = {"page_signals": {"dimensions": None, "blocks": blocks or [], "images": []}}
    return OCRResult(
        document_id="doc",
        page_number=1,
        engine="mistral",
        engine_version="test",
        backend="mistral-ocr-latest",
        status="success",
        text="page text",
        elapsed_ms=5,
        confidence=90.0,
        confidence_scores=confidence_scores,
        metadata=metadata,
    )


def signature_block(name: str = "J. Smith") -> dict:
    return {"type": "signature", "bbox": [10, 700, 300, 760], "content": name}


def text_block(chars: int = 400) -> dict:
    return {"type": "text", "bbox": [10, 10, 500, 600], "content_chars": chars}


class TestTicketCases:
    """The three deterministic cases SCR-2227 mandates."""

    def test_signature_page_flags_signature_only(self):
        # (a) A signature page: signature blocks + healthy printed words.
        flags = detect_review_flags(
            page_result(
                word_confidences=[0.96] * 20,
                blocks=[text_block(), signature_block(), signature_block("")],
            )
        )
        assert flags.signature_page is True
        assert flags.handwriting_suspected is False
        assert flags.signals["signature_block_count"] == 2
        assert flags.signals["signature_names"] == ["J. Smith"]

    def test_handwritten_edit_page_flags_handwriting_only(self):
        # (b) A non-signature page with a handwritten edit: a cluster of
        # uncertain words (crossed-out salary + written replacement).
        confidences = [0.96] * 24 + [0.41, 0.38, 0.55, 0.49, 0.62, 0.58]
        flags = detect_review_flags(
            page_result(word_confidences=confidences, blocks=[text_block()])
        )
        assert flags.handwriting_suspected is True
        assert flags.signature_page is False
        assert flags.signals["low_confidence_word_count"] == 6
        assert flags.signals["word_count"] == 30

    def test_clean_printed_page_has_no_flags(self):
        # (c) A clean printed page: high confidence everywhere, text blocks.
        flags = detect_review_flags(
            page_result(word_confidences=[0.95] * 40, blocks=[text_block()])
        )
        assert flags.signature_page is False
        assert flags.handwriting_suspected is False

    def test_handwritten_edit_on_a_signature_page_flags_both(self):
        # The ticket's motivating scenario in one page: a crossed-out value
        # rewritten by hand ON the signature page. Flags are independent.
        confidences = [0.96] * 24 + [0.41, 0.38, 0.55, 0.49, 0.62, 0.58]
        flags = detect_review_flags(
            page_result(
                word_confidences=confidences,
                blocks=[text_block(), signature_block()],
            )
        )
        assert flags.signature_page is True
        assert flags.handwriting_suspected is True


class TestHandwritingBoundaries:
    def test_ratio_exactly_at_threshold_flags(self):
        # 2 of 20 = exactly 0.10 — >= semantics must flag.
        confidences = [0.95] * 18 + [0.5, 0.5]
        assert len(confidences) == 20
        flags = detect_review_flags(page_result(word_confidences=confidences))
        assert 2 / 20 == HANDWRITING_LOW_CONFIDENCE_RATIO
        assert flags.handwriting_suspected is True

    def test_just_below_ratio_does_not_flag(self):
        confidences = [0.95] * 29 + [0.5, 0.5]  # 2/31 ≈ 0.065
        flags = detect_review_flags(page_result(word_confidences=confidences))
        assert flags.handwriting_suspected is False

    def test_min_words_boundary_uses_ratio_rule(self):
        # Exactly MIN_WORDS words uses the ratio rule, not the sparse rule.
        confidences = [0.95] * (HANDWRITING_MIN_WORDS - 2) + [0.5, 0.5]
        assert len(confidences) == HANDWRITING_MIN_WORDS
        flags = detect_review_flags(page_result(word_confidences=confidences))
        assert flags.handwriting_suspected is True  # 2/15 ≈ 0.133 >= 0.10

    def test_sparse_page_low_average_flags(self):
        flags = detect_review_flags(page_result(word_confidences=[0.4] * 5))
        assert flags.handwriting_suspected is True

    def test_sparse_page_high_average_is_clean(self):
        flags = detect_review_flags(page_result(word_confidences=[0.9] * 5))
        assert flags.handwriting_suspected is False

    def test_word_confidence_exactly_at_cutoff_is_not_low(self):
        # Strict <: a word at exactly 0.70 is not "uncertain".
        confidences = [0.95] * 18 + [0.70, 0.70]
        flags = detect_review_flags(page_result(word_confidences=confidences))
        assert flags.handwriting_suspected is False
        assert flags.signals["low_confidence_word_count"] == 0

    def test_sparse_average_exactly_at_floor_is_clean(self):
        # Strict <: an average of exactly 0.60 does not flag.
        flags = detect_review_flags(page_result(word_confidences=[0.60] * 5))
        assert flags.handwriting_suspected is False


class TestRobustness:
    def test_empty_values_and_missing_signals_are_clean(self):
        # The adapter emits values=[] when Mistral returns no word scores.
        flags = detect_review_flags(
            page_result(
                confidence_scores={"granularity": "word", "scale": "0-1", "values": []},
                metadata={},
            )
        )
        assert flags.signature_page is False
        assert flags.handwriting_suspected is False

    def test_illegible_signature_still_flags(self):
        flags = detect_review_flags(
            page_result(word_confidences=[0.95] * 20, blocks=[signature_block("")])
        )
        assert flags.signature_page is True
        assert flags.signals["signature_names"] == []

    def test_garbage_inputs_never_raise(self):
        result = page_result()
        result.confidence_scores = {"values": {"word_confidence_scores": "garbage"}}
        result.metadata = {"page_signals": {"blocks": "not-a-list"}}
        flags = detect_review_flags(result)
        assert flags.signature_page is False
        assert flags.handwriting_suspected is False

    def test_detector_exception_degrades_to_clean_flags(self):
        class Hostile:
            def get(self, *_args, **_kwargs):
                raise RuntimeError("boom")

        result = page_result()
        result.metadata = Hostile()
        flags = detect_review_flags(result)
        assert flags.signature_page is False
        assert flags.handwriting_suspected is False
        assert "detector_error" in flags.signals
