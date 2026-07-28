"""Digital-page pre-flight classifier: verdicts, parsing, and fail-safety."""

from __future__ import annotations

import subprocess

import pytest

from ocr_engine import classification
from ocr_engine.classification import (
    MIN_DIGITAL_TEXT_CHARS,
    _document_page_texts,
    _parse_pdfimages_list,
    classify_document,
)

from tests.conftest import DIGITAL_PAGE_TEXT, build_digital_pdf, build_pdf


class TestClassifyDocument:
    def test_born_digital_pages_are_digital(self, tmp_path):
        pdf = build_digital_pdf(
            tmp_path / "digital.pdf", [DIGITAL_PAGE_TEXT, DIGITAL_PAGE_TEXT]
        )
        verdicts = classify_document(pdf, 2)

        assert sorted(verdicts) == [1, 2]
        for verdict in verdicts.values():
            assert verdict.is_digital
            assert verdict.reason == "digital-text"
            assert verdict.char_count >= MIN_DIGITAL_TEXT_CHARS
            assert "Employment Agreement" in verdict.text
            assert verdict.image_count == 0
            assert verdict.elapsed_ms >= 0

    def test_text_never_ends_with_form_feed(self, tmp_path):
        # OcrDocumentResult.text joins pages with form-feed; a trailing \f
        # from pdftotext would double every separator.
        pdf = build_digital_pdf(tmp_path / "digital.pdf", [DIGITAL_PAGE_TEXT])
        verdict = classify_document(pdf, 1)[1]
        assert verdict.text
        assert not verdict.text.endswith("\f")

    def test_image_page_needs_ocr(self, tmp_path):
        # Pillow PDFs embed every page as a raster image (a scan stand-in).
        pdf = build_pdf(tmp_path / "scan.pdf", pages=2)
        verdicts = classify_document(pdf, 2)

        for verdict in verdicts.values():
            assert not verdict.is_digital
            assert verdict.reason == "has-images"
            assert verdict.image_count >= 1
            assert verdict.text == ""

    def test_text_page_with_any_image_needs_ocr(self, tmp_path):
        # The no-size-threshold rule: abundant digital text cannot clear a
        # page that carries even a tiny raster image (e.g. a signature).
        pdf = build_digital_pdf(
            tmp_path / "mixed.pdf",
            [DIGITAL_PAGE_TEXT, DIGITAL_PAGE_TEXT],
            image_on_pages={2},
        )
        verdicts = classify_document(pdf, 2)

        assert verdicts[1].is_digital
        assert not verdicts[2].is_digital
        assert verdicts[2].reason == "has-images"

    def test_stamp_only_page_is_sparse(self, tmp_path):
        pdf = build_digital_pdf(
            tmp_path / "stamp.pdf",
            ["DocuSign Envelope ID: 4C1AB2F8-0D3E-4B5A-9C87-1F2E3D4C5B6A"],
        )
        verdict = classify_document(pdf, 1)[1]
        assert not verdict.is_digital
        assert verdict.reason == "sparse-text"
        assert 0 < verdict.char_count < MIN_DIGITAL_TEXT_CHARS

    def test_whitespace_only_text_is_sparse(self, tmp_path):
        pdf = build_digital_pdf(tmp_path / "blank.pdf", ["   \n \n  "])
        verdict = classify_document(pdf, 1)[1]
        assert not verdict.is_digital
        assert verdict.reason == "sparse-text"
        assert verdict.char_count == 0

    def test_min_chars_boundary(self, tmp_path):
        # Short lines keep every glyph inside the MediaBox — pdftotext drops
        # characters rendered past the page edge, which would skew the count.
        def wrapped(chars: int) -> str:
            return "\n".join(
                "x" * min(40, chars - start) for start in range(0, chars, 40)
            )

        at_threshold = wrapped(MIN_DIGITAL_TEXT_CHARS)
        below_threshold = wrapped(MIN_DIGITAL_TEXT_CHARS - 1)
        pdf = build_digital_pdf(
            tmp_path / "boundary.pdf", [at_threshold, below_threshold]
        )
        verdicts = classify_document(pdf, 2)

        assert verdicts[1].is_digital
        assert verdicts[1].char_count == MIN_DIGITAL_TEXT_CHARS
        assert not verdicts[2].is_digital
        assert verdicts[2].reason == "sparse-text"
        assert verdicts[2].char_count == MIN_DIGITAL_TEXT_CHARS - 1


PDFIMAGES_HEADER = (
    "page num type width height color comp bpc enc interp object ID"
    " x-ppi y-ppi size ratio"
)
PDFIMAGES_RULE = "-" * 85


class TestParsePdfimagesList:
    def test_counts_rows_per_page_ignoring_masks(self):
        output = "\n".join(
            [
                PDFIMAGES_HEADER,
                PDFIMAGES_RULE,
                "   1   0 image    2550  3300 gray    1   8 jpeg  no    26  0   300   300  180K  6.7%",
                "   1   1 smask    2550  3300 gray    1   8 image no    27  0   300   300  120K  4.4%",
                "   3   2 stencil     1     1 -       1   1 image no    28  0   469   469     3B 100%",
                "",
            ]
        )
        assert _parse_pdfimages_list(output) == {1: 1, 3: 1}

    def test_unknown_row_type_counts_as_an_image(self):
        # Unknown content must route to OCR, never be dropped as "no image".
        output = "\n".join(
            [
                PDFIMAGES_HEADER,
                PDFIMAGES_RULE,
                "   2   0 newtype     10    10 rgb    3   8 jpeg  no    26  0   300   300    1K  1.0%",
            ]
        )
        assert _parse_pdfimages_list(output) == {2: 1}

    def test_empty_document_has_no_counts(self):
        assert (
            _parse_pdfimages_list(f"{PDFIMAGES_HEADER}\n{PDFIMAGES_RULE}\n") == {}
        )

    def test_garbage_data_row_raises(self):
        # A silently dropped row could hide an image-bearing page, letting
        # it classify as digital — ambiguity must fail (to OCR), not pass.
        output = "\n".join(
            [PDFIMAGES_HEADER, PDFIMAGES_RULE, "garbage that is not a row"]
        )
        with pytest.raises(ValueError, match="row"):
            _parse_pdfimages_list(output)

    def test_unrecognized_header_raises(self):
        with pytest.raises(ValueError, match="header"):
            _parse_pdfimages_list("something entirely different\n-----\n")

    def test_missing_output_raises(self):
        with pytest.raises(ValueError, match="header"):
            _parse_pdfimages_list("")


class TestDocumentPageTexts:
    def test_page_count_mismatch_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            classification,
            "run_sandboxed",
            lambda *_args, **_kwargs: b"page one\fpage two\f",
        )
        with pytest.raises(ValueError, match="expected 3"):
            _document_page_texts(tmp_path / "doc.pdf", 3)

    def test_output_without_trailing_form_feed_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            classification,
            "run_sandboxed",
            lambda *_args, **_kwargs: b"page one without terminator",
        )
        with pytest.raises(ValueError, match="form-feed"):
            _document_page_texts(tmp_path / "doc.pdf", 1)


class TestVectorMarks:
    def test_curve_page_needs_ocr(self, tmp_path):
        # A bezier squiggle drawn as vector paths (a stylus-signature
        # stand-in): invisible to pdfimages, caught by the pdfplumber gate.
        pdf = build_digital_pdf(
            tmp_path / "curve.pdf",
            [DIGITAL_PAGE_TEXT, DIGITAL_PAGE_TEXT],
            curve_on_pages={2},
        )
        verdicts = classify_document(pdf, 2)

        assert verdicts[1].is_digital
        assert not verdicts[2].is_digital
        assert verdicts[2].reason == "has-vector-marks"
        assert verdicts[2].vector_mark_count >= 1

    def test_ink_annotation_page_needs_ocr(self, tmp_path):
        pdf = build_digital_pdf(
            tmp_path / "ink.pdf", [DIGITAL_PAGE_TEXT], ink_annot_on_pages={1}
        )
        verdict = classify_document(pdf, 1)[1]
        assert not verdict.is_digital
        assert verdict.reason == "has-vector-marks"

    def test_scan_runs_only_for_would_be_skipped_pages(
        self, tmp_path, monkeypatch
    ):
        # Scanned documents (all pages carry images) must never pay the
        # pdfplumber cost.
        def fail(*_args, **_kwargs):
            pytest.fail("vector scan must not run when no page is skippable")

        monkeypatch.setattr(classification, "_vector_mark_counts", fail)
        pdf = build_pdf(tmp_path / "scan.pdf", pages=2)
        verdicts = classify_document(pdf, 2)
        assert all(v.reason == "has-images" for v in verdicts.values())

    def test_scan_failure_is_fail_safe(self, tmp_path, monkeypatch):
        def boom(_pdf_path, _pages):
            raise ValueError("vector scan could not read doc.pdf (exit status 1)")

        monkeypatch.setattr(classification, "_vector_mark_counts", boom)
        pdf = build_digital_pdf(
            tmp_path / "digital.pdf", [DIGITAL_PAGE_TEXT]
        )
        verdicts = classify_document(pdf, 1)  # must not raise
        assert not verdicts[1].is_digital
        assert verdicts[1].reason == "classification-error"

    def test_missing_scan_verdict_raises(self, tmp_path, monkeypatch):
        # A page the scan skipped is ambiguous — it must not default to
        # "no marks".
        monkeypatch.setattr(
            classification,
            "run_sandboxed",
            lambda *_args, **_kwargs: b'{"1": {"curves": 0, "markup_annots": 0}}',
        )
        with pytest.raises(ValueError, match=r"page\(s\) \[2\]"):
            classification._vector_mark_counts(tmp_path / "doc.pdf", [1, 2])


class TestFailSafe:
    def test_pdfimages_failure_marks_all_pages_needs_ocr(
        self, tmp_path, monkeypatch
    ):
        # The shape a password-protected PDF produces (CalledProcessError
        # re-raised as ValueError with the stderr excerpt).
        def boom(_pdf_path):
            raise ValueError("pdfimages could not read doc.pdf (exit status 1)")

        monkeypatch.setattr(classification, "_pdfimages_page_counts", boom)
        verdicts = classify_document(tmp_path / "doc.pdf", 3)

        assert sorted(verdicts) == [1, 2, 3]
        for verdict in verdicts.values():
            assert not verdict.is_digital
            assert verdict.reason == "classification-error"

    def test_pdftotext_failure_marks_all_pages_needs_ocr(
        self, tmp_path, monkeypatch
    ):
        # The single document-wide pdftotext call is all-or-nothing: on any
        # failure, per-page attribution would be guesswork, so every page
        # falls back to OCR.
        monkeypatch.setattr(
            classification, "_pdfimages_page_counts", lambda _pdf_path: {}
        )

        def timeout(_pdf_path, _page_count):
            raise subprocess.TimeoutExpired(cmd="pdftotext", timeout=120)

        monkeypatch.setattr(classification, "_document_page_texts", timeout)
        verdicts = classify_document(tmp_path / "doc.pdf", 2)

        for verdict in verdicts.values():
            assert not verdict.is_digital
            assert verdict.reason == "classification-error"

    def test_missing_pdfimages_binary_is_fail_safe(self, tmp_path, monkeypatch):
        def missing(_pdf_path):
            raise RuntimeError("pdfimages is required; install Poppler")

        monkeypatch.setattr(classification, "_pdfimages_page_counts", missing)
        verdicts = classify_document(tmp_path / "doc.pdf", 1)  # must not raise
        assert not verdicts[1].is_digital
        assert verdicts[1].reason == "classification-error"
