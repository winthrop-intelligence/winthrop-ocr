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

- **poppler-utils** (`pdftoppm`/`pdfinfo` for rendering, `pdftotext`/`pdfimages`
  for the digital-page skip) — `apt-get install poppler-utils`
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

## Alteration detection (vision, v0.3.0)

Scanned contracts sometimes carry **hand alterations**: a printed dollar
amount or game date crossed out and replaced in pen, usually initialed.
Plain OCR either garbles these or — worse — silently returns the
superseded printed value as clean text. When `vision_enabled` is on
(default for `default`/`contracts`), every page is also sent — same
rendered image — to the **`mistral-medium-2505`** vision model. The
vision call runs **concurrently with the OCR call** (it needs only the
image), so page latency is max(ocr, vision), not their sum (~2,500 input
tokens / ~4s per page):

```python
for page in result.pages:
    alt = page.alterations            # PageAlterations | None (None = vision off)
    if alt is not None and alt.flagged:
        alt.alterations               # flag-only entries: clause, kind
result.summary()["alteration_pages"], result.summary()["vision_failed_pages"]
```

**The verdict is flag-only by design.** Benchmark (July 2026), with
denominators: detection recall was 5/5 on the confirmed hand-altered
pages available (from 3 user-verified contracts; a 4th genuinely altered
contract was discovered during evaluation and is also flagged). The flag
rate was 124 of 1,386 pages (~9%) across a held-out corpus of 133 real
game, employment, vendor, and financial documents with no known
alterations — an upper bound on the false-positive rate, since sampled
adjudication showed most (not all) of those flags were false. These are
corpus-specific measurements, not detection guarantees. The prompt is
evidence-grounded — the model must inventory the physical marks it sees
and may only report alterations backed by pen handwriting or
strike-through marks — which suppresses false flags from e-signature
fonts, typed form fill-ins, and scanner noise. The model
dependably identifies which page/clause was altered but routinely misread
the struck and replacement values — so values are neither requested nor
accepted (entries carry only `clause` and `kind`). Route flagged pages to
human review of the actual scan to read the real values.

Flagged pages pass two false-positive filters before surfacing (v0.4.1):
a deterministic blank-page guard (a page with no printed text cannot
carry an altered printed value), and a second adversarial vision call
that must re-confirm the pen ink (`alt.verified`: True confirmed /
False rejected-and-cleared / None not-applicable or failed-open).
Verification adds one call only on flagged pages; disable with
`overrides={"vision_verify": False}`.

Detection **soft-fails**: a vision error records a non-success
`alt.status` ("rate_limited", "timeout", "parse_error", ...) and never
fails the page or document — the OCR text stands on its own. Opt out
per-run with `overrides={"vision_enabled": False}`. The model must be a
pinned dated ID: `-latest` aliases are rejected by policy validation
(they silently ride model upgrades and price changes).

## Digital-page skip (v0.4.0)

Born-digital PDF pages don't need OCR: before rendering, every page is
classified with Poppler (`pdftotext` + `pdfimages -list`) plus a
pdfplumber vector scan, and a page **skips the Mistral call** only when it
has **≥ 120 non-whitespace characters of embedded digital text, zero
embedded raster images, AND zero vector-drawn marks** (bezier curves,
diagonal lines, or markup annotations — a stylus signature, a drawn
X-mark, or annotation ink that raster tools can't see; axis-aligned
lines and rectangles like table borders and underlines stay benign). There is deliberately **no image-size threshold** —
on real contracts a DocuSign signature image covers ~1% of the page while
a decorative letterhead logo covers ~9%, so size cannot separate content
from decoration; any image at all routes the page to OCR. The character
floor keeps stamp-only scans (e.g. a scanned page carrying just a digital
"DocuSign Envelope ID" line) on the OCR path too.

Skipped pages come back as normal success pages:

```python
from ocr_engine import DIGITAL_TEXT_ENGINE

for page in result.pages:
    if page.selected.engine == DIGITAL_TEXT_ENGINE:   # "digital-text"
        page.selected.text                # exact embedded text (pdftotext)
        page.selected.confidence          # 100.0 — exact, not a model estimate
        page.selected.metadata["classification"]  # reason / char_count / image_count
        page.alterations                  # None: no rendered image, so the
                                          # vision/alteration pass doesn't run
result.summary()["digital_pages"]         # count of pages that skipped OCR
result.summary()["digital_page_numbers"]  # e.g. [9, 10, 11, 12]
result.summary()["ocr_page_numbers"]      # pages that took the render+OCR path
```

OCR-routed pages of a classified PDF also carry the routing evidence in
`page.selected.metadata["classification"]` (`reason` of `has-images` /
`has-vector-marks` / `sparse-text` / `classification-error`, plus
char/image/vector counts), so consumers can ship `summary()` and per-page
reasons straight into their metrics (e.g. a Sentry dashboard) without
parsing logs.

Classification **fail-safes to OCR**: any error (unreadable PDF, missing
tool, timeout, unrecognized tool output) sends the page through the
normal render+OCR path and never fails the document. The worst failure
mode is an unnecessary OCR call — never lost content. Classification
costs at most three sandboxed subprocess calls per document — one
`pdfimages -list`, one `pdftotext`, and one pdfplumber vector scan that
runs **only for pages about to be skipped** (scanned documents never pay
it) — and a **fully digital document requires neither `MISTRAL_API_KEY`
nor `pdftoppm`**: OCR dependencies are checked only when at least one
page actually needs OCR. Opt out per-run with
`overrides={"skip_digital_pages": False}`. Single-image sources are never
classified (PDF-only). Residual limitation (accepted): a purely
horizontal or vertical drawn mark (e.g. a strikethrough drawn as a plain
line) is indistinguishable from an underline without rendering the page,
and is treated as layout — axis-aligned segments appear on 57% of
legitimately skippable real contract pages, so flagging them would gut
the feature. Drawn handwriting, X-marks, and check marks are curves or
diagonals, which are flagged.

## Profiles

`default`, `contracts` (300 DPI, vision on), and `job_postings` (vision
off — job posts have no hand-altered contract values). Consumers are tuned
independently via `ocr_engine/policy.py` or the `overrides` argument, e.g.
`ocr_document(path, profile="contracts", overrides={"dpi": 400})`.
Unknown override fields raise (a typo never silently keeps the default),
and `result.policy_fingerprint` records the fully resolved configuration
for provenance.

## Layout

```
ocr_engine/
├── document.py        # ocr_document() — the entry point
├── classification.py  # pre-flight born-digital page detection (skip OCR)
├── vector_marks.py    # pdfplumber vector/annotation scan (sandboxed child)
├── review.py          # per-page review flags (signature / handwriting)
├── vision.py          # per-page hand-alteration detection (Mistral vision)
├── runner.py          # run_page / OcrDocumentResult
├── rendering.py       # pdftoppm rendering, page counting, hashing
├── policy.py          # profiles + validation
├── models.py          # PageInput / OCRResult / PageAlterations
└── adapters/          # mistral (status-aware retries), registry
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
