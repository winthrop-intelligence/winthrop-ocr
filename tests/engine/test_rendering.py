"""Rendering: sandbox failure classification and page-size-aware DPI."""

from __future__ import annotations

import pytest
from PIL import Image

from ocr_engine import rendering
from ocr_engine.rendering import run_sandboxed

from tests.conftest import build_pdf


def build_giant_pdf(path, width_px=1892, height_px=2544):
    """A phone-scan 'poster' PDF: Pillow writes 1 px = 1 pt at 72 DPI."""

    image = Image.new("L", (width_px, height_px), color=255)
    image.convert("RGB").save(path, format="PDF")
    return path


class TestRunSandboxedClassification:
    """Exception type encodes who failed: the environment or the input."""

    def test_stdout_returned_on_success(self):
        output = run_sandboxed(["echo", "hello"], timeout=30, failure="boom")
        assert output.strip() == b"hello"

    def test_tool_failure_raises_value_error(self):
        with pytest.raises(ValueError, match="boom"):
            run_sandboxed(
                ["sh", "-c", "echo rejected >&2; exit 99"],
                timeout=30,
                failure="boom",
            )

    def test_memory_limit_setup_failure_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="sandbox could not run"):
            run_sandboxed(["sh", "-c", "exit 125"], timeout=30, failure="boom")

    def test_exec_failure_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="sandbox could not run"):
            run_sandboxed(["sh", "-c", "exit 126"], timeout=30, failure="boom")

    def test_death_by_signal_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="sandbox could not run"):
            run_sandboxed(["sh", "-c", "kill -9 $$"], timeout=30, failure="boom")

    def test_missing_tool_raises_runtime_error_with_hint(self):
        with pytest.raises(RuntimeError, match="install Poppler"):
            run_sandboxed(
                ["definitely-not-a-real-tool-xyz"], timeout=30, failure="boom"
            )


class TestBoundedDpi:
    """Clamp math: only larger-than-legal pages ever lose DPI."""

    def test_letter_page_keeps_requested_dpi(self):
        assert rendering.bounded_dpi(612, 792, 300) == 300

    def test_legal_page_keeps_requested_dpi(self):
        # 14 in = 1008 pts is exactly the calibration boundary.
        assert rendering.bounded_dpi(612, 1008, 300) == 300

    def test_phone_scan_poster_clamps(self):
        # The USF CLC ADDENDUM shape: 26.3 x 35.3 inches at 300 DPI would
        # be an ~84 MP render; the clamp keeps the longest side <= 4200 px.
        assert rendering.bounded_dpi(1892.32, 2544.4, 300) == 118

    def test_pixel_cap_is_absolute(self):
        # No declared size may push the longest rendered side past the cap
        # (until the 1-DPI floor, far beyond the PDF spec's page limit).
        for longest_pts in (2544.4, 8000, 20_000, 100_000, 302_400):
            dpi = rendering.bounded_dpi(longest_pts, longest_pts, 300)
            assert dpi * longest_pts / 72 <= rendering.MAX_RENDER_DIM_PX

    def test_absurd_page_size_still_honors_pixel_cap(self):
        # Greptile P1 regression: a 100000-pt page must NOT render at a
        # floor DPI that overflows the cap (50 DPI would be ~69444 px).
        assert rendering.bounded_dpi(100_000, 100_000, 300) == 3

    def test_impossible_page_size_floors_at_one_dpi(self):
        assert rendering.bounded_dpi(1_000_000, 1_000_000, 300) == 1

    def test_degenerate_size_keeps_requested_dpi(self):
        assert rendering.bounded_dpi(0, 0, 300) == 300


class TestPdfPageSize:
    """pdfinfo-backed per-page size probe."""

    def test_reads_page_size_in_points(self, tmp_path):
        pdf = build_pdf(tmp_path / "doc.pdf")  # Pillow: 400x500 px = pts
        width_pts, height_pts = rendering.pdf_page_size(pdf, 1)
        assert (round(width_pts), round(height_pts)) == (400, 500)

    def test_unreadable_pdf_raises_value_error(self, tmp_path):
        bogus = tmp_path / "broken.pdf"
        bogus.write_bytes(b"%PDF-not really a pdf")
        with pytest.raises(ValueError):
            rendering.pdf_page_size(bogus, 1)

    def test_missing_page_raises_value_error(self, tmp_path):
        pdf = build_pdf(tmp_path / "doc.pdf", pages=2)
        with pytest.raises(ValueError, match="page 99"):
            rendering.pdf_page_size(pdf, 99)

    def test_one_pdfinfo_probe_per_document(self, tmp_path, monkeypatch):
        pdf = build_pdf(tmp_path / "doc.pdf", pages=3)
        spawned = []
        real_run_sandboxed = rendering.run_sandboxed

        def counting_run_sandboxed(arguments, **kwargs):
            spawned.append(arguments[0])
            return real_run_sandboxed(arguments, **kwargs)

        monkeypatch.setattr(rendering, "run_sandboxed", counting_run_sandboxed)
        for page_number in (1, 2, 3):
            assert rendering.pdf_page_size(pdf, page_number)
        assert spawned.count("pdfinfo") == 1


class TestRenderDpiClamp:
    """render_page_input renders at the effective, size-aware DPI."""

    def test_normal_page_renders_at_requested_dpi(self, tmp_path):
        pdf = build_pdf(tmp_path / "doc.pdf")
        page = rendering.render_page_input(pdf, 1, tmp_path / "out")
        assert page.dpi == 300

    def test_giant_page_renders_at_clamped_dpi(self, tmp_path):
        pdf = build_giant_pdf(tmp_path / "giant.pdf")
        page = rendering.render_page_input(pdf, 1, tmp_path / "out")
        assert page.dpi == rendering.bounded_dpi(1892, 2544, 300)
        assert page.dpi < 300
        with Image.open(page.image_path) as rendered:
            # pdftoppm rounds the raster up; allow one pixel of slack.
            assert max(rendered.size) <= rendering.MAX_RENDER_DIM_PX + 1

    def test_probe_failure_keeps_requested_dpi(self, tmp_path, monkeypatch):
        pdf = build_pdf(tmp_path / "doc.pdf")

        def raise_probe_failure(*_args):
            raise ValueError("probe down")

        monkeypatch.setattr(rendering, "pdf_page_size", raise_probe_failure)
        page = rendering.render_page_input(pdf, 1, tmp_path / "out")
        assert page.dpi == 300
