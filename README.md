# winthrop-ocr

OCR as a library: hand it a local PDF (or single-page image), get every
page's text back with per-page confidence — extracted by **Mistral OCR**.
It runs inside your process and returns when the work is done; there is no
service, no queue, and no polling.

## Install

```toml
# pyproject.toml
winthrop-ocr = {git = "https://github.com/winthrop-intelligence/winthrop-ocr.git", tag = "v0.2.0"}
```

System requirements:

- **poppler-utils** (`pdftoppm`/`pdfinfo`) for page rendering — `apt-get install poppler-utils`
- **`MISTRAL_API_KEY`** in the environment

## Use

```python
from ocr_engine import OcrDocumentError, ocr_document

try:
    result = ocr_document("/tmp/contract.pdf", profile="contracts")
except OcrDocumentError as exc:
    # exc.failed_pages / exc.page_count / exc.status_counts support triage:
    # all rate_limited/auth_error means a provider or credential problem
    # (retry later / fix config) rather than a bad document.
    run_local_fallback()
else:
    full_text = result.text                  # pages joined with "\f"
    for page in result.pages:                # ordered, complete
        page.page_number, page.selected.text, page.confidence
    result.summary()                         # page_count, mean_confidence, ...
```

Import from the package root (`from ocr_engine import ...`) — that is the
supported surface; submodule paths are internal layout.

`ocr_document` is **strict**: it either returns a complete document or
raises `OcrDocumentError` (missing/unreadable file, unknown profile, engine
unavailable, or any page failing). A returned result never has silent gaps,
and once a document is doomed, still-queued pages are cancelled rather than
processed. PDFs are detected by content (magic bytes), not filename.

Pages are OCR'd concurrently (`max_workers`, default 4 — the calls are
network I/O; each attempt is bounded by a 120s timeout, with up to 3
attempts for transient failures). Pass `runtime_check=` a zero-arg callable
and it fires at least every 10 seconds while pages are in flight; if it
raises, that exception propagates unwrapped (it is your signal, not an OCR
failure) after queued pages are cancelled and in-flight pages drain.

`confidence` values on results are on a **0–100** scale (the raw provider
payload inside `confidence_scores` keeps Mistral's native 0–1 values).

## Review flags (v0.2.0)

Every successful page carries non-blocking human-review flags:

```python
for page in result.pages:
    page.review.signature_page          # Mistral's native signature-block classifier
    page.review.handwriting_suspected   # broad heuristic over word confidences
    page.review.signals                 # the evidence (counts, ratios, thresholds)
result.summary()["signature_pages"], result.summary()["handwriting_pages"]
```

Flags never change the OCR text or fail a page — detection errors degrade
to clean flags with a `detector_error` note. Handwriting thresholds are
module constants in `ocr_engine/review.py`, deliberately broad first
(tune with real flagged pages later). Reporting policy (Sentry, manifests)
belongs to consumers.

The library never touches your storage: fetching the source file and
persisting the text are the caller's job.

## Profiles

`default`, `contracts`, `job_postings` — currently identical settings
(300 DPI); the split exists so consumers can be tuned independently via
`ocr_engine/policy.py` or the `overrides` argument, e.g.
`ocr_document(path, profile="contracts", overrides={"dpi": 400})`.
Unknown override fields raise (a typo never silently keeps the default),
and `result.policy_fingerprint` records the fully resolved configuration
for provenance.

## Layout

```
ocr_engine/
├── document.py    # ocr_document() — the entry point
├── review.py      # per-page review flags (signature / handwriting)
├── runner.py      # run_page / OcrDocumentResult
├── rendering.py   # pdftoppm rendering, page counting, hashing
├── policy.py      # profiles + validation
├── models.py      # PageInput / OCRResult
└── adapters/      # mistral (status-aware retries), registry
```

## Develop

```bash
poetry install
poetry run pytest
poetry run pylint ocr_engine
```

Releases are git tags (semver): consumers pin a tag and upgrade by bumping
it. This repo was seeded from `winthrop-ocr-api`'s `ocr_engine` package;
the service wrapper remains in that repo, dormant.
