"""Handwritten-alteration detection via the Mistral vision model.

One chat-completions call per page when vision is enabled, sending the
same rendered PNG the OCR engine uses (the call runs concurrently with
OCR, so a verdict is recorded even when the page's OCR fails). Detection
is an annotation like :mod:`ocr_engine.review`'s flags: it soft-fails —
``detect_alterations`` never raises, and a non-success status must never
fail a page or document.

Validated against user-confirmed hand-altered contracts: the model's flag
(which page/clause) is reliable, its value transcriptions are not — so the
verdict is flag-only (clause + kind, no values) and consumers route
flagged pages to human review of the actual scan.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from ocr_engine.adapters.base import image_data_url, module_available
from ocr_engine.adapters.mistral import (
    is_retryable_mistral_error,
    mistral_error_status,
    retry_delay,
)
from ocr_engine.models import PageAlterations, PageInput
from ocr_engine.policy import OcrPolicy

logger = logging.getLogger(__name__)

# Fewer attempts than OCR's 3: vision soft-fails, so exhausting retries
# degrades an annotation, not the document. One retry covers transient
# transport blips, the dominant failure class.
VISION_MAX_ATTEMPTS = 2
# Calls average ~3.6s; OCR's 120s budget is oversized here and would
# stretch the worst-case in-flight drain after a caller abort.
VISION_ATTEMPT_TIMEOUT_MS = 60_000
MAX_ALTERATION_ENTRIES = 50
# Entry values are model-generated strings; bound them like the OCR
# adapter bounds its serialized signals.
ENTRY_VALUE_MAX_CHARS = 300
RAW_EXCERPT_MAX_CHARS = 2000

# Benchmark-validated against a 1,386-page held-out corpus (133 real
# contracts: game, employment, vendor, financial) plus 6 user-confirmed
# hand-altered pages: 100% detection recall, ~9% page flag rate (half the
# naive prompt's rate). Do not edit without re-running that benchmark.
# Structure: evidence grounding — the model must first inventory the
# physical marks it can SEE and classify them against a named taxonomy;
# only pen_handwriting / pen_strikethrough marks may ground an alteration.
# This suppresses the dominant failure mode (confabulating an alteration
# on a clean typed page anchored to a salient salary/date clause) and the
# e-signature-font / typed-form-fill-in / scan-noise false-positive
# classes. Flag-only: value transcriptions are neither requested nor
# accepted (see ALLOWED_ENTRY_KEYS).
ALTERATIONS_PROMPT = (
    "You are examining one page of a contract for physical handwritten "
    "alterations made with a pen: printed/typed text that has been crossed out "
    "and/or replaced by handwriting. Work in two steps in one JSON response. "
    "STEP 1 — inventory every non-body-text mark you can SEE and classify it "
    "honestly as one of: pen_handwriting (irregular ink strokes written by "
    "hand), pen_strikethrough (an ink line crossing THROUGH the middle of "
    "printed characters), esignature_font (DocuSign/Adobe-style cursive "
    "script rendered by a computer, usually near a signature line or inside a "
    "signature box), typed_fill_in (typed or monospace text inserted into a "
    "form blank or styled differently from the body — including bold inserted "
    "phrases and values sitting ON TOP OF an underline; an underline UNDER "
    "text is not a strikethrough), page_number, stamp, smudge_or_scan_noise, "
    "signature. STEP 2 — report an alteration ONLY for marks classified "
    "pen_handwriting or pen_strikethrough that overlap or replace printed "
    "text in the document body. All other mark types are NEVER alterations. "
    "A page that is uniformly machine-printed with no pen ink anywhere has no "
    "alterations, no matter what values it contains — never infer alterations "
    "from the meaning of typed text or from document quality. Most pages have "
    "none. Respond ONLY with JSON: {\"visible_marks\": [{\"location\": "
    "\"...\", \"type\": \"...\"}], \"alterations\": [{\"clause\": \"...\", "
    "\"kind\": \"dollar_amount|date|other\", \"mark_index\": 0}], "
    "\"none_found\": false}. Do NOT transcribe values. If no alterations: "
    "\"alterations\": [] and \"none_found\": true."
)

# The only entry fields consumers may see. Anything else the model volunteers
# (e.g. value transcriptions) is dropped — those were wrong too often to
# ever be ingested as data.
ALLOWED_ENTRY_KEYS = ("clause", "kind")
# Every surfaced entry has exactly this kind vocabulary; anything else the
# model invents is normalized to "other".
ALLOWED_KINDS = frozenset({"dollar_amount", "date", "other"})


def detect_alterations(page: PageInput, policy: OcrPolicy) -> PageAlterations:
    """Ask the vision model whether the page carries hand alterations.

    Never raises: every failure is mapped to a ``status`` on the returned
    :class:`PageAlterations` so callers can log/route it without special
    casing.
    """

    started = time.perf_counter()
    # run_page() is a public building block, so guard availability here even
    # though ocr_document()'s pre-flight checks the same SDK and key.
    if not module_available("mistralai"):
        return _failed(policy, "unavailable", "the mistralai package is not installed")
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return _failed(policy, "unavailable", "MISTRAL_API_KEY is not set")

    try:
        image_url = image_data_url(page.image_path)
        response, transport_retries = _complete_vision_with_retries(
            api_key=api_key,
            image_url=image_url,
            model=policy.vision_model,
        )
        raw_text = _response_text(response)
        alterations = _parse_alterations(raw_text)
    except ValueError as exc:
        elapsed = round((time.perf_counter() - started) * 1000)
        return _failed(policy, "parse_error", exc, elapsed_ms=elapsed)
    except Exception as exc:  # SDK exposes several transport exception classes.
        elapsed = round((time.perf_counter() - started) * 1000)
        return _failed(policy, mistral_error_status(exc), exc, elapsed_ms=elapsed)

    elapsed = round((time.perf_counter() - started) * 1000)
    return PageAlterations(
        status="success",
        model=policy.vision_model,
        alterations=alterations,
        # Derived, never taken from the model: one source of truth means a
        # verdict can never claim "flagged" and "none found" at once.
        none_found=not alterations,
        elapsed_ms=elapsed,
        transport_retries=transport_retries,
    )


def _failed(
    policy: OcrPolicy,
    status: str,
    error: Exception | str,
    *,
    elapsed_ms: int = 0,
) -> PageAlterations:
    """Create a normalized failed detection without raising."""

    error_type = error.__class__.__name__ if isinstance(error, Exception) else status
    return PageAlterations(
        status=status,
        model=policy.vision_model,
        elapsed_ms=elapsed_ms,
        error_type=error_type,
        error_message=str(error)[:RAW_EXCERPT_MAX_CHARS],
    )


def _vision_request_kwargs(
    *, model: str, image_url: str, prompt: str = ALTERATIONS_PROMPT
) -> dict[str, Any]:
    """The exact chat-completions request the vision call sends.

    Kept as a separate builder so tests can bind it against the REAL SDK's
    signature — an SDK upgrade that renames or drops a parameter must fail
    a unit test, not silently soft-fail vision on every production page.
    """

    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": image_url},
                ],
            }
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "timeout_ms": VISION_ATTEMPT_TIMEOUT_MS,
    }


def _complete_vision_with_retries(
    *, api_key: str, image_url: str, model: str, prompt: str = ALTERATIONS_PROMPT
) -> tuple[Any, int]:
    """Retry transient vision transport failures with a fresh SDK client."""

    # pylint: disable-next=import-outside-toplevel,import-error
    from mistralai.client import Mistral

    request = _vision_request_kwargs(model=model, image_url=image_url, prompt=prompt)
    for attempt in range(VISION_MAX_ATTEMPTS):
        try:
            client = Mistral(api_key=api_key)
            response = client.chat.complete(**request)
            return response, attempt
        except Exception as exc:
            final_attempt = attempt == VISION_MAX_ATTEMPTS - 1
            if final_attempt or not is_retryable_mistral_error(exc):
                raise
            delay = retry_delay(exc, attempt)
            logger.debug(
                "Vision attempt %d/%d failed (%s); retrying in %.1fs",
                attempt + 1,
                VISION_MAX_ATTEMPTS,
                type(exc).__name__,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError("vision retry loop ended without a response")


def _response_text(response: Any) -> str:
    """The completion's text content, tolerating chunked content lists."""

    choices = list(getattr(response, "choices", None) or [])
    if not choices:
        raise ValueError("vision response has no choices")
    content = getattr(getattr(choices[0], "message", None), "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Multimodal SDKs may return content as chunks; keep the text parts.
        parts = []
        for chunk in content:
            text = getattr(chunk, "text", None)
            if text is None and isinstance(chunk, dict):
                text = chunk.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    raise ValueError(f"vision response content has unexpected type {type(content).__name__}")


def _parse_alterations(text: str) -> list[dict[str, Any]]:
    """Parse the model's JSON verdict; raises ValueError on any mismatch.

    With response_format=json_object malformed output should be rare, so a
    single local repair (outermost braces) is attempted — never a second
    API call. Entries are validated strictly: an entry survives only with a
    non-empty string clause, its kind is normalized to ALLOWED_KINDS, and
    the model's own none_found claim is ignored (the caller derives it).
    """

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[len("json") :]
        cleaned = cleaned.strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(
                f"vision response is not JSON: {cleaned[:RAW_EXCERPT_MAX_CHARS]}"
            ) from None
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"vision response is not JSON: {cleaned[:RAW_EXCERPT_MAX_CHARS]}"
            ) from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("alterations"), list):
        raise ValueError(
            f"vision response missing 'alterations' list: {cleaned[:RAW_EXCERPT_MAX_CHARS]}"
        )
    alterations = []
    for entry in payload["alterations"][:MAX_ALTERATION_ENTRIES]:
        validated = _validated_entry(entry)
        if validated is not None:
            alterations.append(validated)
    return alterations


def _validated_entry(entry: Any) -> dict[str, Any] | None:
    """One strict, bounded entry — or None when the model's entry is junk.

    A flag with no locatable clause is useless to the human reviewer, so
    entries without a non-empty string clause are dropped rather than
    surfaced as empty dicts.
    """

    if not isinstance(entry, dict):
        return None
    clause = entry.get("clause")
    if not isinstance(clause, str) or not clause.strip():
        return None
    kind = entry.get("kind")
    if kind not in ALLOWED_KINDS:
        kind = "other"
    return {"clause": clause.strip()[:ENTRY_VALUE_MAX_CHARS], "kind": kind}


# Second-pass verification: skeptical re-review of a flagged page. The
# first pass optimizes recall; confabulated flags are unstable under
# adversarial re-questioning while real pen ink is stable (measured on
# production false positives: blank pages, e-signature fonts, and typed
# form fill-ins get rejected; confirmed hand alterations survive).
VERIFICATION_PROMPT = (
    "A first-pass reviewer claimed this scanned contract page contains a "
    "handwritten alteration: printed/typed text crossed out and/or replaced "
    "with pen handwriting. You are the skeptical second reviewer. CONFIRM "
    "only if you can clearly and unambiguously SEE pen ink that crosses out "
    "printed text, or handwriting that overwrites or replaces a printed "
    "value in the document body. Do NOT confirm for: blank or nearly blank "
    "pages; signatures or e-signature script fonts; typed or "
    "computer-rendered text of any style, including values sitting on an "
    "underline in a fill-in blank (that is form-filling, not an alteration, "
    "even if it looks hand-entered); checked checkboxes; handwriting that "
    "only fills an empty blank; stamps, smudges, scanner noise, or page "
    "numbers. If in any doubt, reject. Respond ONLY JSON: "
    '{"confirmed": true, "reason": "..."} or '
    '{"confirmed": false, "reason": "..."} with reason under 15 words.'
)


def verify_alterations(
    page: PageInput, policy: OcrPolicy, first_pass: PageAlterations
) -> PageAlterations:
    """Adversarially re-check a flagged page; never raises.

    Returns the first-pass result updated in place semantics-wise:
    confirmed -> verified True; rejected -> entries cleared and verified
    False; the verification call itself failing -> flag kept with
    verified None (fail open: a transport blip must not silently drop a
    real alteration).
    """

    if not first_pass.flagged:
        return first_pass
    started = time.perf_counter()
    try:
        image_url = image_data_url(page.image_path)
        response, _retries = _complete_vision_with_retries(
            api_key=os.environ["MISTRAL_API_KEY"],
            image_url=image_url,
            model=policy.vision_model,
            prompt=VERIFICATION_PROMPT,
        )
        payload = json.loads(_response_text(response).strip())
        confirmed = bool(payload.get("confirmed"))
    except Exception:  # noqa: BLE001 - fail open, keep the flag
        first_pass.elapsed_ms += round((time.perf_counter() - started) * 1000)
        return first_pass

    first_pass.elapsed_ms += round((time.perf_counter() - started) * 1000)
    if confirmed:
        first_pass.verified = True
    else:
        first_pass.alterations = []
        first_pass.none_found = True
        first_pass.verified = False
    return first_pass
