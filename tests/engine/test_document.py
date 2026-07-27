"""ocr_document facade: strict whole-document OCR with a fake engine."""

from __future__ import annotations

import pytest

from ocr_engine import document
from ocr_engine.document import OcrDocumentError, ocr_document

from tests.conftest import (
    DIGITAL_PAGE_TEXT,
    FakeEngine,
    build_digital_pdf,
    build_pdf,
    draw_text_like_page,
)


@pytest.fixture(name="fake_registry")
def fake_registry_fixture(monkeypatch):
    """Route the facade at a deterministic engine; return it for assertions."""

    engine = FakeEngine("mistral")
    monkeypatch.setattr(document, "engine_registry", lambda: {"mistral": engine})
    return engine


class TestSuccess:
    def test_three_page_pdf_returns_ordered_complete_text(
        self, tmp_path, fake_registry
    ):
        pdf = build_pdf(tmp_path / "doc.pdf", pages=3)
        result = ocr_document(pdf, profile="contracts")

        assert [outcome.page_number for outcome in result.pages] == [1, 2, 3]
        assert result.text == "\f".join([fake_registry.text] * 3)
        assert result.profile == "contracts"
        assert result.document_id.startswith("doc-")
        assert result.summary()["page_count"] == 3
        assert result.summary()["pages_failed"] == 0
        assert sorted(fake_registry.calls) == [1, 2, 3]

    def test_pdf_detection_is_content_based_not_suffix_based(
        self, tmp_path, fake_registry
    ):
        # A real PDF with no extension still routes through the PDF path.
        pdf = build_pdf(tmp_path / "spool-file", pages=2)
        result = ocr_document(pdf)
        assert len(result.pages) == 2

        # A PNG misnamed .pdf routes through the image path and still works.
        image = draw_text_like_page(tmp_path / "scan.png")
        misnamed = tmp_path / "actually-a-png.pdf"
        misnamed.write_bytes(image.read_bytes())
        result = ocr_document(misnamed)
        assert len(result.pages) == 1

    def test_profile_overrides_are_applied(self, tmp_path, fake_registry):
        pdf = build_pdf(tmp_path / "doc.pdf", pages=1)
        result = ocr_document(pdf, profile="contracts", overrides={"dpi": 200})
        assert len(result.pages) == 1
        # Provenance distinguishes overridden runs from default ones.
        assert result.policy_fingerprint
        baseline = ocr_document(pdf, profile="contracts")
        assert result.policy_fingerprint != baseline.policy_fingerprint

    def test_single_image_source_yields_one_page(self, tmp_path, fake_registry):
        image = draw_text_like_page(tmp_path / "scan.png")
        result = ocr_document(image)
        assert len(result.pages) == 1
        assert result.pages[0].page_number == 1

    def test_runtime_check_fires_while_pages_are_in_flight(
        self, tmp_path, fake_registry
    ):
        calls = []
        pdf = build_pdf(tmp_path / "doc.pdf", pages=3)
        # One worker guarantees pages remain in flight across iterations.
        ocr_document(pdf, max_workers=1, runtime_check=lambda: calls.append(1))
        assert calls  # at least once, from the collection loop

    def test_runtime_check_never_fires_after_the_last_page(self):
        """A deadline raised after completion must not discard a finished doc.

        Drives _collect_pages directly with instant pages so the only
        runtime_check opportunity would be AFTER the final page completed —
        which the contract forbids.
        """

        def instant_page(number):
            from ocr_engine.models import OCRResult
            from ocr_engine.runner import PageOutcome

            return PageOutcome(
                page_number=number,
                selected=OCRResult(
                    document_id="doc",
                    page_number=number,
                    engine="fake",
                    engine_version=None,
                    backend=None,
                    status="success",
                    text="t",
                    elapsed_ms=1,
                ),
            )

        def explode():
            raise AssertionError("runtime_check fired with no pages in flight")

        outcomes, failures = document._collect_pages(
            instant_page, page_count=1, max_workers=1, runtime_check=explode
        )
        assert not failures
        assert set(outcomes) == {1}

    def test_max_workers_is_clamped_to_page_count(self, tmp_path, fake_registry):
        pdf = build_pdf(tmp_path / "doc.pdf", pages=1)
        result = ocr_document(pdf, max_workers=16)
        assert len(result.pages) == 1

    def test_review_flags_flow_through_per_page(self, tmp_path, monkeypatch):
        """Clean, signature, and handwritten pages each carry their flags."""
        engine = FakeEngine(
            "mistral",
            word_confidences=[0.95] * 40,  # default: clean printed page
            per_page={
                2: {
                    "page_signals": {
                        "dimensions": None,
                        "blocks": [
                            {"type": "signature", "bbox": None, "content": "J. Smith"}
                        ],
                        "images": [],
                    }
                },
                3: {"word_confidences": [0.95] * 24 + [0.4] * 6},
            },
        )
        monkeypatch.setattr(document, "engine_registry", lambda: {"mistral": engine})
        pdf = build_pdf(tmp_path / "doc.pdf", pages=3)
        result = ocr_document(pdf, max_workers=3)

        by_page = {outcome.page_number: outcome.review for outcome in result.pages}
        assert not by_page[1].signature_page and not by_page[1].handwriting_suspected
        assert by_page[2].signature_page and not by_page[2].handwriting_suspected
        assert by_page[3].handwriting_suspected and not by_page[3].signature_page
        summary = result.summary()
        assert summary["signature_pages"] == 1
        assert summary["handwriting_pages"] == 1


class TestDigitalPages:
    def test_digital_pdf_skips_the_engine_entirely(self, tmp_path, fake_registry):
        pdf = build_digital_pdf(
            tmp_path / "digital.pdf", [DIGITAL_PAGE_TEXT, DIGITAL_PAGE_TEXT]
        )
        result = ocr_document(pdf, profile="contracts")

        assert fake_registry.calls == []
        for outcome in result.pages:
            selected = outcome.selected
            assert selected.engine == "digital-text"
            assert selected.backend == "pdftotext"
            assert selected.status == "success"
            assert selected.confidence == 100.0
            assert selected.metadata["classification"]["reason"] == "digital-text"
            assert not outcome.review.signature_page
            assert not outcome.review.handwriting_suspected
            assert outcome.alterations is None
        # Two pages, one form-feed separator — no double separators from
        # pdftotext's own trailing form-feed.
        assert result.text.count("\f") == 1
        assert result.text.count("Employment Agreement") == 2
        summary = result.summary()
        assert summary["digital_pages"] == 2
        assert summary["engines_used"] == {"digital-text": 2}
        assert summary["pages_failed"] == 0

    def test_mixed_document_routes_only_image_pages_to_engine(
        self, tmp_path, fake_registry
    ):
        pdf = build_digital_pdf(
            tmp_path / "mixed.pdf",
            [DIGITAL_PAGE_TEXT, DIGITAL_PAGE_TEXT],
            image_on_pages={2},
        )
        result = ocr_document(pdf, profile="contracts")

        assert fake_registry.calls == [2]
        by_page = {outcome.page_number: outcome.selected for outcome in result.pages}
        assert by_page[1].engine == "digital-text"
        assert by_page[2].engine == "mistral"
        assert result.summary()["digital_pages"] == 1
        assert result.summary()["engines_used"] == {"digital-text": 1, "mistral": 1}

    def test_vision_never_runs_for_digital_pages(
        self, tmp_path, fake_registry, monkeypatch
    ):
        calls = TestVision.install_detector(monkeypatch, lambda _n: None)
        pdf = build_digital_pdf(
            tmp_path / "mixed.pdf",
            [DIGITAL_PAGE_TEXT, DIGITAL_PAGE_TEXT],
            image_on_pages={2},
        )
        result = ocr_document(pdf, profile="contracts")

        assert calls == [2]
        by_page = {outcome.page_number: outcome for outcome in result.pages}
        assert by_page[1].alterations is None

    def test_skip_digital_pages_opt_out_forces_ocr(self, tmp_path, fake_registry):
        pdf = build_digital_pdf(
            tmp_path / "digital.pdf", [DIGITAL_PAGE_TEXT, DIGITAL_PAGE_TEXT]
        )
        result = ocr_document(pdf, overrides={"skip_digital_pages": False})

        assert sorted(fake_registry.calls) == [1, 2]
        assert all(
            outcome.selected.engine == "mistral" for outcome in result.pages
        )
        assert result.summary()["digital_pages"] == 0
        baseline = ocr_document(pdf)
        assert result.policy_fingerprint != baseline.policy_fingerprint

    def test_stamp_only_page_still_goes_to_ocr(self, tmp_path, fake_registry):
        # The SCR-2285 trap: a scanned-page stand-in whose only digital text
        # is a short DocuSign stamp must not be skipped.
        pdf = build_digital_pdf(
            tmp_path / "stamp.pdf",
            ["DocuSign Envelope ID: 4C1AB2F8-0D3E-4B5A-9C87-1F2E3D4C5B6A"],
        )
        ocr_document(pdf)
        assert fake_registry.calls == [1]

    def test_classification_failure_falls_back_to_ocr(
        self, tmp_path, fake_registry, monkeypatch
    ):
        from ocr_engine.classification import PageClassification

        def broken(_pdf_path, page_count):
            return {
                number: PageClassification(
                    number, is_digital=False, reason="classification-error"
                )
                for number in range(1, page_count + 1)
            }

        monkeypatch.setattr(document, "classify_document", broken)
        pdf = build_digital_pdf(
            tmp_path / "digital.pdf", [DIGITAL_PAGE_TEXT, DIGITAL_PAGE_TEXT]
        )
        result = ocr_document(pdf)  # must not raise

        assert sorted(fake_registry.calls) == [1, 2]
        assert result.summary()["pages_failed"] == 0

    def test_single_image_source_never_classifies(
        self, tmp_path, fake_registry, monkeypatch
    ):
        def fail(*_args, **_kwargs):
            pytest.fail("classify_document must not run for non-PDF sources")

        monkeypatch.setattr(document, "classify_document", fail)
        image = draw_text_like_page(tmp_path / "scan.png")
        ocr_document(image)
        assert fake_registry.calls == [1]


class TestVision:
    @staticmethod
    def install_detector(monkeypatch, verdict_for_page):
        """Patch detection at the runner's lookup site; return the call log."""

        from ocr_engine import runner as runner_module

        calls = []

        def _detect(page, policy):
            calls.append(page.page_number)
            return verdict_for_page(page.page_number)

        monkeypatch.setattr(runner_module, "detect_alterations", _detect)
        return calls

    def test_flagged_page_flows_through(self, tmp_path, fake_registry, monkeypatch):
        from ocr_engine.models import PageAlterations

        def verdict(page_number):
            if page_number == 2:
                return PageAlterations(
                    status="success",
                    model="mistral-medium-2505",
                    alterations=[{"kind": "date", "clause": "1"}],
                    none_found=False,
                )
            return PageAlterations(
                status="success", model="mistral-medium-2505", none_found=True
            )

        calls = self.install_detector(monkeypatch, verdict)
        pdf = build_pdf(tmp_path / "doc.pdf", pages=3)
        result = ocr_document(pdf, profile="contracts")
        assert sorted(calls) == [1, 2, 3]
        flagged = [outcome.page_number for outcome in result.pages if outcome.alterations.flagged]
        assert flagged == [2]
        assert result.summary()["alteration_pages"] == 1
        assert result.summary()["vision_failed_pages"] == 0

    def test_vision_failure_never_fails_the_document(
        self, tmp_path, fake_registry, monkeypatch
    ):
        from ocr_engine.models import PageAlterations

        self.install_detector(
            monkeypatch,
            lambda _page_number: PageAlterations(status="crash", model="m"),
        )
        pdf = build_pdf(tmp_path / "doc.pdf", pages=2)
        result = ocr_document(pdf, profile="contracts")  # must not raise
        assert result.summary()["pages_failed"] == 0
        assert result.summary()["vision_failed_pages"] == 2

    def test_job_postings_profile_skips_vision(
        self, tmp_path, fake_registry, monkeypatch
    ):
        calls = self.install_detector(monkeypatch, lambda _n: None)
        pdf = build_pdf(tmp_path / "doc.pdf", pages=2)
        result = ocr_document(pdf, profile="job_postings")
        assert calls == []
        assert all(outcome.alterations is None for outcome in result.pages)
        assert result.summary()["alteration_pages"] == 0
        assert result.summary()["vision_elapsed_ms"] == 0


class TestStrictFailures:
    def test_any_failed_page_raises_naming_the_pages(self, tmp_path, monkeypatch):
        engine = FakeEngine("mistral", status="crash")
        monkeypatch.setattr(document, "engine_registry", lambda: {"mistral": engine})
        pdf = build_pdf(tmp_path / "doc.pdf", pages=2)
        with pytest.raises(OcrDocumentError, match=r"page\(s\) 1, 2"):
            ocr_document(pdf, max_workers=2)

    def test_exception_carries_structured_triage_data(self, tmp_path, monkeypatch):
        engine = FakeEngine("mistral", status="rate_limited")
        monkeypatch.setattr(document, "engine_registry", lambda: {"mistral": engine})
        pdf = build_pdf(tmp_path / "doc.pdf", pages=2)
        with pytest.raises(OcrDocumentError) as excinfo:
            ocr_document(pdf, max_workers=2)
        error = excinfo.value
        assert error.failed_pages == [1, 2]
        assert error.page_count == 2
        assert error.status_counts == {"rate_limited": 2}

    def test_doomed_document_stops_submitting_pages(self, tmp_path, monkeypatch):
        engine = FakeEngine("mistral", status="crash")
        monkeypatch.setattr(document, "engine_registry", lambda: {"mistral": engine})
        pdf = build_pdf(tmp_path / "doc.pdf", pages=4)
        with pytest.raises(OcrDocumentError):
            ocr_document(pdf, max_workers=1)
        # Rolling-window scheduling makes fail-fast DETERMINISTIC: with one
        # worker, the window holds one page, so after page 1 fails nothing
        # further is ever submitted.
        assert engine.calls == [1]

    def test_runtime_check_exception_propagates_unwrapped(
        self, tmp_path, monkeypatch
    ):
        engine = FakeEngine("mistral")
        monkeypatch.setattr(document, "engine_registry", lambda: {"mistral": engine})
        pdf = build_pdf(tmp_path / "doc.pdf", pages=3)

        class CallerDeadline(Exception):
            pass

        def deadline():
            raise CallerDeadline("out of time")

        with pytest.raises(CallerDeadline):
            ocr_document(pdf, max_workers=1, runtime_check=deadline)
        # The caller's own signal, not an OcrDocumentError; queued pages
        # were cancelled, so not every page ran.
        assert len(engine.calls) <= 2

    def test_invalid_overrides_raise_ocr_document_error(
        self, tmp_path, fake_registry
    ):
        pdf = build_pdf(tmp_path / "doc.pdf", pages=1)
        bad_payloads = (
            {"dip": 400},        # typo must fail loudly, not keep the default
            {"dpi": "x"},        # wrong type
            {"dpi": 9999},       # out of range
            [("dpi", 400)],      # not a mapping
        )
        for bad in bad_payloads:
            with pytest.raises(OcrDocumentError, match="invalid OCR profile"):
                ocr_document(pdf, overrides=bad)

    def test_worker_raised_cancelled_error_is_a_failed_page(
        self, tmp_path, monkeypatch
    ):
        """A worker leaking CancelledError must never yield a partial result."""
        from concurrent.futures import CancelledError

        class LeakyEngine(FakeEngine):
            def extract(self, page):
                if page.page_number == 2:
                    self.calls.append(page.page_number)
                    raise CancelledError("leaked from a nested pool")
                return super().extract(page)

        engine = LeakyEngine("mistral")
        monkeypatch.setattr(document, "engine_registry", lambda: {"mistral": engine})
        pdf = build_pdf(tmp_path / "doc.pdf", pages=3)
        with pytest.raises(OcrDocumentError) as excinfo:
            ocr_document(pdf, max_workers=3)
        assert 2 in excinfo.value.failed_pages

    def test_multi_frame_tiff_is_rejected(self, tmp_path, fake_registry):
        from PIL import Image

        frames = [Image.new("L", (50, 50), color=value) for value in (255, 0)]
        tiff = tmp_path / "pages.tif"
        frames[0].save(tiff, format="TIFF", save_all=True, append_images=frames[1:])
        with pytest.raises(OcrDocumentError, match="multi-frame"):
            ocr_document(tiff)

    def test_unavailable_engine_raises_before_any_extract(self, tmp_path, monkeypatch):
        class UnavailableEngine(FakeEngine):
            @classmethod
            def availability(cls):
                return False, "MISTRAL_API_KEY is not set"

        engine = UnavailableEngine("mistral")
        monkeypatch.setattr(document, "engine_registry", lambda: {"mistral": engine})
        pdf = build_pdf(tmp_path / "doc.pdf", pages=1)
        with pytest.raises(OcrDocumentError, match="MISTRAL_API_KEY"):
            ocr_document(pdf)
        assert engine.calls == []

    def test_unknown_profile_raises(self, tmp_path, fake_registry):
        pdf = build_pdf(tmp_path / "doc.pdf", pages=1)
        with pytest.raises(OcrDocumentError, match="invalid OCR profile"):
            ocr_document(pdf, profile="nope")

    def test_missing_file_raises(self, tmp_path, fake_registry):
        with pytest.raises(OcrDocumentError, match="does not exist"):
            ocr_document(tmp_path / "ghost.pdf")

    def test_unreadable_pdf_bytes_raise(self, tmp_path, fake_registry):
        bogus = tmp_path / "broken.pdf"
        bogus.write_bytes(b"%PDF-not really a pdf")
        with pytest.raises(OcrDocumentError, match="could not read PDF"):
            ocr_document(bogus)

    def test_unsupported_image_source_raises(self, tmp_path, fake_registry):
        junk = tmp_path / "notes.txt"
        junk.write_text("plain text, not an image")
        with pytest.raises(OcrDocumentError):
            ocr_document(junk)
