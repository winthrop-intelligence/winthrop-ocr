"""Runner behavior with fake engines."""

from ocr_engine.policy import OcrPolicy
from ocr_engine.runner import OcrDocumentResult, run_page

from tests.conftest import FakeEngine, draw_text_like_page, page_input_for

POLICY = OcrPolicy()


def engines(confidence=90.0, status="success"):
    return {"mistral": FakeEngine("mistral", confidence=confidence, status=status)}


class TestRunPage:
    def test_page_goes_through_mistral(self, tmp_path):
        page = page_input_for(draw_text_like_page(tmp_path / "p.png"))
        registry = engines(confidence=95.0)
        outcome = run_page(page, registry, POLICY)
        assert outcome.selected.engine == "mistral"
        assert outcome.confidence == 95.0
        assert registry["mistral"].calls == [1]
        assert [entry["engine"] for entry in outcome.trail()] == ["mistral"]

    def test_engine_failure_yields_failed_result(self, tmp_path):
        page = page_input_for(draw_text_like_page(tmp_path / "p.png"))
        registry = engines(status="rate_limited")
        outcome = run_page(page, registry, POLICY)
        assert outcome.selected.status == "rate_limited"
        assert outcome.selected.text == ""
        assert outcome.trail()[0]["status"] == "rate_limited"
        # Detection is skipped for failed pages: default clean flags.
        assert outcome.review.signature_page is False
        assert outcome.review.handwriting_suspected is False
        assert outcome.review.signals == {}


class TestOcrDocumentResult:
    def test_text_joined_with_form_feed_and_summary_rollup(self, tmp_path):
        registry = engines()
        outcomes = [
            run_page(
                page_input_for(
                    draw_text_like_page(tmp_path / f"p{n}.png"), page_number=n
                ),
                registry,
                POLICY,
            )
            for n in (1, 2)
        ]
        result = OcrDocumentResult(
            document_id="doc-test",
            profile="default",
            policy_version="test",
            pages=outcomes,
        )
        assert result.text.count("\f") == 1
        assert registry["mistral"].calls == [1, 2]
        summary = result.summary()
        assert summary["page_count"] == 2
        assert summary["pages_failed"] == 0
        assert summary["engines_used"] == {"mistral": 2}
