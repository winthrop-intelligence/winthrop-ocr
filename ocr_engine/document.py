"""One-call, in-process document OCR.

``ocr_document`` is the entry point consumers use: hand it a local PDF (or
image) path and it renders every page, runs them through the configured
engine concurrently, and returns the ordered page texts — or raises
``OcrDocumentError`` if the document could not be fully processed. Callers
own everything outside the file: fetching the source, storing the text.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import tempfile
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from ocr_engine.adapters.registry import engine_registry
from ocr_engine.classification import (
    DIGITAL_TEXT_ENGINE,
    MIN_DIGITAL_TEXT_CHARS,
    PageClassification,
    classify_document,
)
from ocr_engine.models import OCRResult
from ocr_engine.policy import POLICY_VERSION, OcrPolicy, resolve_policy
from ocr_engine.rendering import (
    document_id,
    image_page_input,
    pdf_page_count,
    render_page_input,
)
from ocr_engine.review import PageReviewFlags
from ocr_engine.runner import OcrDocumentResult, PageOutcome, run_page

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = 4

# The collection loop wakes at least this often to fire runtime_check even
# while no page has completed (e.g. a slow network call in every worker).
RUNTIME_CHECK_INTERVAL_SECONDS = 10

# Keep error messages readable for many-page failures.
MAX_REPORTED_FAILED_PAGES = 10

# Status recorded for pages whose worker raised (render crash etc.), as
# opposed to engine-reported statuses like rate_limited/auth_error/timeout.
WORKER_EXCEPTION_STATUS = "exception"


class OcrDocumentError(Exception):
    """The document could not be fully OCR'd; no usable result exists.

    Structured attributes let callers triage without parsing the message:
    - ``failed_pages``: page numbers that failed (empty for pre-flight
      failures such as a missing file or unavailable engine; may omit
      pages that were cancelled once the document was already doomed).
    - ``page_count``: total pages in the document (0 when unknown).
    - ``status_counts``: failed-page count per status — engine statuses
      (``rate_limited``, ``auth_error``, ``timeout``, ``crash``, ...) plus
      ``exception`` for worker crashes. A document whose failures are all
      ``rate_limited``/``auth_error`` indicates a provider/config problem
      (retry later / fix credentials) rather than a bad document.
    """

    def __init__(
        self,
        message: str,
        *,
        failed_pages: list[int] | None = None,
        page_count: int = 0,
        status_counts: dict[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.failed_pages = list(failed_pages or [])
        self.page_count = page_count
        self.status_counts = dict(status_counts or {})


def ocr_document(
    source_path: str | Path,
    *,
    profile: str = "default",
    overrides: Mapping[str, Any] | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    runtime_check: Callable[[], None] | None = None,
) -> OcrDocumentResult:
    """Render and OCR every page of a local document, strictly.

    Args:
    - source_path: local path to a PDF or a single-page image (PNG/JPG/TIF).
      PDFs are detected by content (magic bytes), not filename.
    - profile: policy profile name (``default``, ``contracts``, ...).
    - overrides: policy field overrides for the selected profile, e.g.
      ``{"dpi": 400}``. Unknown field names raise.
    - max_workers: page-level concurrency. Engine calls are network I/O, so
      threads overlap them; rendering stays bounded because each worker
      renders its own page and deletes the image right after OCR.
    - runtime_check: optional zero-arg callable invoked from the calling
      thread at least every ``RUNTIME_CHECK_INTERVAL_SECONDS`` while pages
      are in flight (keeps callers' runtime alerting alive). If it raises,
      its exception propagates UNWRAPPED (it is the caller's own signal,
      not an OCR failure); no further pages are submitted, but pages
      already in flight finish first — each bounded by the larger of the
      engine's retry budget (attempts x per-attempt timeout, a few minutes
      worst case) and, when vision detection is enabled, the concurrent
      vision retry budget (2 attempts x 60s) — so the raise can take that
      long to surface.

    Returns an :class:`OcrDocumentResult` with pages in order; ``.text``
    joins pages with form-feed. Raises :class:`OcrDocumentError` when the
    source is unusable, the engine is unavailable, or ANY page fails —
    partial documents are never returned, so callers can treat a return
    value as complete. Once a document is doomed, still-queued pages are
    cancelled rather than processed.
    """

    # Absolute path: a relative name starting with "-" must never reach a
    # subprocess argv looking like an option.
    source = Path(source_path).resolve()
    if not source.is_file():
        raise OcrDocumentError(f"source file does not exist: {source}")

    try:
        policy = resolve_policy(profile, overrides=overrides)
    except Exception as exc:
        raise OcrDocumentError(f"invalid OCR profile {profile!r}: {exc}") from exc

    engines = engine_registry()
    engine = engines.get(policy.engine)
    if engine is None:
        raise OcrDocumentError(f"no engine registered as {policy.engine!r}")
    available, diagnostic = engine.availability()
    if not available:
        raise OcrDocumentError(
            f"OCR engine {policy.engine!r} is not available: {diagnostic}"
        )

    is_pdf = _is_pdf(source)
    if is_pdf:
        if shutil.which("pdftoppm") is None:
            raise OcrDocumentError(
                "poppler-utils is not installed (pdftoppm not found)"
            )
        try:
            page_count = pdf_page_count(source)
        except Exception as exc:
            raise OcrDocumentError(f"could not read PDF {source.name}: {exc}") from exc
        if page_count < 1:
            raise OcrDocumentError(f"PDF {source.name} reports no pages")
    else:
        page_count = 1

    try:
        identifier = document_id(source)
    except OSError as exc:
        raise OcrDocumentError(f"could not read {source.name}: {exc}") from exc

    # Computed once on the calling thread before workers start; workers
    # only read it. classify_document never raises (errors mean OCR).
    classifications = _preflight_classifications(source, is_pdf, policy, page_count)
    digital_count = sum(
        1 for verdict in classifications.values() if verdict.is_digital
    )

    with tempfile.TemporaryDirectory(prefix="ocr-doc-") as temp_dir:
        pages_dir = Path(temp_dir)

        def process_page(page_number: int) -> PageOutcome:
            verdict = classifications.get(page_number)
            if verdict is not None and verdict.is_digital:
                return _digital_page_outcome(identifier, verdict)
            if is_pdf:
                page = render_page_input(
                    source, page_number, pages_dir, dpi=policy.dpi, identifier=identifier
                )
            else:
                page = image_page_input(
                    source, pages_dir, dpi=policy.dpi, identifier=identifier
                )
            try:
                return run_page(page, engines, policy)
            finally:
                # Bound disk usage to ~max_workers rendered images.
                with contextlib.suppress(OSError):
                    page.image_path.unlink()

        logger.info(
            "OCR starting for %s: %d page(s), profile=%s, %d digital page(s) skip OCR",
            source.name,
            page_count,
            profile,
            digital_count,
        )
        outcomes, failures = _collect_pages(
            process_page, page_count, max_workers, runtime_check
        )

    failed_pages = sorted(
        set(failures)
        | {
            number
            for number, outcome in outcomes.items()
            if outcome.selected.status != "success"
        }
    )
    if failed_pages:
        raise OcrDocumentError(
            _failure_message(source, failed_pages, failures, outcomes),
            failed_pages=failed_pages,
            page_count=page_count,
            status_counts=_status_counts(failed_pages, failures, outcomes),
        )

    return OcrDocumentResult(
        document_id=identifier,
        profile=profile,
        policy_version=POLICY_VERSION,
        pages=[outcomes[number] for number in sorted(outcomes)],
        policy_fingerprint=policy.fingerprint(),
    )


def _preflight_classifications(
    source: Path, is_pdf: bool, policy: OcrPolicy, page_count: int
) -> dict[int, PageClassification]:
    """Classify pages for the digital skip; empty when it does not apply."""

    if not (is_pdf and policy.skip_digital_pages):
        return {}
    return classify_document(source, page_count)


def _digital_page_outcome(
    identifier: str, verdict: PageClassification
) -> PageOutcome:
    """Synthesize the success outcome for a born-digital page.

    The page never renders, so there is no image for the vision pass
    (``alterations=None``, the same shape as ``vision_enabled=False``);
    with zero embedded raster images the vision/alteration pass is
    skipped (see the vector-content caveat in ``ocr_engine.classification``).
    Review flags stay all-False: they are OCR-confidence heuristics and
    exact digital text needs no review.
    """

    result = OCRResult(
        document_id=identifier,
        page_number=verdict.page_number,
        engine=DIGITAL_TEXT_ENGINE,
        engine_version=None,
        backend="pdftotext",
        status="success",
        text=verdict.text,
        elapsed_ms=verdict.elapsed_ms,
        confidence=100.0,  # exact digital extraction, not a model estimate
        metadata={
            "classification": {
                "reason": verdict.reason,
                "char_count": verdict.char_count,
                "image_count": verdict.image_count,
                "min_chars": MIN_DIGITAL_TEXT_CHARS,
            }
        },
    )
    return PageOutcome(
        page_number=verdict.page_number,
        selected=result,
        review=PageReviewFlags(),
        alterations=None,
    )


def _is_pdf(source: Path) -> bool:
    """Detect PDFs by content so extension-less or misnamed files route right."""

    try:
        with source.open("rb") as stream:
            return stream.read(5) == b"%PDF-"
    except OSError as exc:
        raise OcrDocumentError(f"could not read {source.name}: {exc}") from exc


def _collect_pages(
    process_page: Callable[[int], PageOutcome],
    page_count: int,
    max_workers: int,
    runtime_check: Callable[[], None] | None,
) -> tuple[dict[int, PageOutcome], dict[int, str]]:
    """Run all pages through the pool; return outcomes and worker failures.

    Pages are submitted through a rolling window of at most ``workers``
    in-flight futures, replenished only while the document is healthy.
    Strictness dooms the whole document on the first failure, so at that
    point no further page is ever submitted — fail-fast is inherent to the
    scheduler, not a best-effort cancellation race — and the work queue
    never holds more than ``workers`` futures regardless of page count.
    In-flight pages still drain and their results are recorded.
    """

    outcomes: dict[int, PageOutcome] = {}
    failures: dict[int, str] = {}
    doomed = False
    workers = max(1, min(max_workers, page_count))
    next_page = 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict[Any, int] = {}

        def replenish_window() -> None:
            nonlocal next_page
            while (
                not doomed
                and next_page <= page_count
                and len(futures) < workers
            ):
                futures[pool.submit(process_page, next_page)] = next_page
                next_page += 1

        replenish_window()
        try:
            while futures:
                done, _pending = wait(
                    set(futures),
                    timeout=RUNTIME_CHECK_INTERVAL_SECONDS,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    number = futures.pop(future)
                    try:
                        outcome = future.result()
                    except Exception as exc:
                        failures[number] = f"{type(exc).__name__}: {exc}"
                    else:
                        outcomes[number] = outcome
                        if outcome.selected.status != "success":
                            doomed = True
                if failures:
                    doomed = True
                if doomed and next_page <= page_count:
                    logger.warning(
                        "document doomed after page failure; %d page(s) will not be submitted",
                        page_count - next_page + 1,
                    )
                    next_page = page_count + 1  # log once; nothing more submits
                replenish_window()
                # Only while pages remain in flight: firing after the last
                # page completed could convert a finished document into the
                # callback's exception.
                if runtime_check is not None and futures:
                    runtime_check()
        except BaseException:
            # In-flight pages (there are at most `workers`) cannot be
            # interrupted, but nothing further may start (e.g. runtime_check
            # raised).
            for future in futures:
                future.cancel()
            raise
    return outcomes, failures


def _status_counts(
    failed_pages: list[int],
    failures: dict[int, str],
    outcomes: dict[int, PageOutcome],
) -> dict[str, int]:
    """Failed pages per status, for caller triage (outage vs bad document)."""

    counts: dict[str, int] = {}
    for number in failed_pages:
        if number in failures:
            status = WORKER_EXCEPTION_STATUS
        else:
            status = outcomes[number].selected.status
        counts[status] = counts.get(status, 0) + 1
    return counts


def _failure_message(
    source: Path,
    failed_pages: list[int],
    failures: dict[int, str],
    outcomes: dict[int, PageOutcome],
) -> str:
    shown = failed_pages[:MAX_REPORTED_FAILED_PAGES]
    listed = ", ".join(str(number) for number in shown)
    if len(failed_pages) > len(shown):
        listed += f", ... ({len(failed_pages)} total)"
    first = failed_pages[0]
    if first in failures:
        detail = failures[first]
    else:
        selected = outcomes[first].selected
        detail = selected.error_message or selected.error_type or selected.status
    return (
        f"OCR failed for {source.name}: page(s) {listed} did not produce text "
        f"(first error: {detail})"
    )
