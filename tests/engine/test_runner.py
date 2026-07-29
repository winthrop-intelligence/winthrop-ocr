"""Runner behavior with fake engines."""

from ocr_engine import runner as runner_module
from ocr_engine.models import PageAlterations
from ocr_engine.policy import OcrPolicy
from ocr_engine.runner import OcrDocumentResult, run_page

from tests.conftest import FakeEngine, draw_text_like_page, page_input_for

POLICY = OcrPolicy()


def fake_detector(monkeypatch, result: PageAlterations):
    """Patch detect_alterations at the runner's lookup site; record calls."""

    calls = []

    def _detect(page, policy):
        calls.append((page.page_number, policy.vision_model))
        return result

    monkeypatch.setattr(runner_module, "detect_alterations", _detect)
    return calls


def flagged_alterations() -> PageAlterations:
    return PageAlterations(
        status="success",
        model="mistral-medium-2505",
        alterations=[{"kind": "dollar_amount", "clause": "4"}],
        none_found=False,
        elapsed_ms=3600,
    )


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


class TestVisionWiring:
    def test_vision_runs_on_successful_pages(self, tmp_path, monkeypatch):
        calls = fake_detector(monkeypatch, flagged_alterations())
        page = page_input_for(draw_text_like_page(tmp_path / "p.png"))
        outcome = run_page(page, engines(), POLICY)
        assert calls == [(1, "mistral-medium-2505")]
        assert outcome.alterations is not None
        assert outcome.alterations.flagged is True

    def test_vision_skipped_when_disabled(self, tmp_path, monkeypatch):
        calls = fake_detector(monkeypatch, flagged_alterations())
        page = page_input_for(draw_text_like_page(tmp_path / "p.png"))
        outcome = run_page(page, engines(), OcrPolicy(vision_enabled=False))
        assert calls == []
        assert outcome.alterations is None

    def test_vision_runs_even_when_ocr_fails(self, tmp_path, monkeypatch):
        # Vision is concurrent with OCR (image-only input), so an OCR
        # failure cannot retroactively skip it; the verdict is recorded.
        calls = fake_detector(monkeypatch, flagged_alterations())
        page = page_input_for(draw_text_like_page(tmp_path / "p.png"))
        outcome = run_page(page, engines(status="rate_limited"), POLICY)
        assert calls == [(1, "mistral-medium-2505")]
        assert outcome.alterations is not None
        assert outcome.alterations.flagged is True

    def test_blank_page_flag_is_suppressed_without_verification(
        self, tmp_path, monkeypatch
    ):
        # An altered printed value cannot exist on a page with no printed
        # text; the guard clears the flag and never spends a verify call.
        fake_detector(monkeypatch, flagged_alterations())
        verify_calls = []
        monkeypatch.setattr(
            runner_module, "verify_alterations", lambda *a: verify_calls.append(a)
        )
        page = page_input_for(draw_text_like_page(tmp_path / "p.png"))
        registry = {"mistral": FakeEngine("mistral", text="  ")}
        outcome = run_page(page, registry, POLICY)
        assert outcome.alterations.flagged is False
        assert outcome.alterations.verified is False
        assert outcome.alterations.none_found is True
        assert verify_calls == []

    def test_flagged_pages_are_verified(self, tmp_path, monkeypatch):
        fake_detector(monkeypatch, flagged_alterations())
        verified = flagged_alterations()
        verified.verified = True
        verify_calls = []

        def _verify(page, policy, first_pass):
            verify_calls.append(page.page_number)
            return verified

        monkeypatch.setattr(runner_module, "verify_alterations", _verify)
        page = page_input_for(draw_text_like_page(tmp_path / "p.png"))
        outcome = run_page(page, engines(), POLICY)
        assert verify_calls == [1]
        assert outcome.alterations.verified is True

    def test_verification_can_be_disabled_by_policy(self, tmp_path, monkeypatch):
        fake_detector(monkeypatch, flagged_alterations())
        monkeypatch.setattr(
            runner_module,
            "verify_alterations",
            lambda *a: (_ for _ in ()).throw(AssertionError("must not verify")),
        )
        page = page_input_for(draw_text_like_page(tmp_path / "p.png"))
        outcome = run_page(page, engines(), OcrPolicy(vision_verify=False))
        assert outcome.alterations.flagged is True
        assert outcome.alterations.verified is None

    def test_without_key_vision_soft_fails_as_unavailable(self, tmp_path):
        # The real detector, no monkeypatch: the autouse fixture removed the
        # key, so wiring must degrade to a status, never an exception.
        page = page_input_for(draw_text_like_page(tmp_path / "p.png"))
        outcome = run_page(page, engines(), POLICY)
        assert outcome.alterations is not None
        assert outcome.alterations.status == "unavailable"
        assert outcome.alterations.flagged is False


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

    def test_summary_vision_rollups(self, tmp_path, monkeypatch):
        registry = engines()
        pages = [
            page_input_for(draw_text_like_page(tmp_path / f"p{n}.png"), page_number=n)
            for n in (1, 2, 3)
        ]
        verdicts = {
            1: flagged_alterations(),
            2: PageAlterations(
                status="success", model="m", none_found=True, elapsed_ms=2000
            ),
            3: PageAlterations(status="crash", model="m", elapsed_ms=400),
        }
        monkeypatch.setattr(
            runner_module,
            "detect_alterations",
            lambda page, policy: verdicts[page.page_number],
        )
        outcomes = [run_page(page, registry, POLICY) for page in pages]
        summary = OcrDocumentResult(
            document_id="doc-test",
            profile="default",
            policy_version="test",
            pages=outcomes,
        ).summary()
        assert summary["alteration_pages"] == 1
        assert summary["vision_failed_pages"] == 1
        assert summary["vision_elapsed_ms"] == 6000
        # OCR elapsed stays vision-free for continuity with prior versions.
        assert summary["elapsed_ms"] < 6000
