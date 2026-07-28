"""OCR policy: per-consumer profiles for rendering and page handling.

POLICY_VERSION identifies the resolved behavior (consumers may cache or
dedupe on it) — bump it whenever behavior changes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from typing import Any

POLICY_VERSION = "2026-07-28"

DPI_MIN, DPI_MAX = 72, 600


@dataclass(frozen=True)
class OcrPolicy:
    """One consumer profile's OCR configuration."""

    name: str = "default"
    engine: str = "mistral"
    dpi: int = 300
    # Handwritten-alteration detection: one vision call per page, run
    # concurrently with OCR (~2,500 input tokens / ~3.6s each). The model
    # must be a pinned dated ID — Mistral's "-latest" aliases silently ride
    # upgrades and price changes (mistral-medium-latest resolves to 3.5).
    vision_enabled: bool = True
    vision_model: str = "mistral-medium-2505"
    # Born-digital pages (>= 120 non-whitespace pdftotext chars AND zero
    # embedded raster images) skip the render+OCR path and return their
    # exact digital text. Classification errors always fall back to OCR.
    skip_digital_pages: bool = True

    def fingerprint(self) -> str:
        """Stable hash of the fully resolved policy, including POLICY_VERSION.

        The dedupe cache key component: two requests share a cached result
        only when every behavior-affecting knob matches.
        """

        payload = asdict(self)
        payload["policy_version"] = POLICY_VERSION
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def validate(self) -> None:
        """Reject incoherent policies before any worker touches them."""

        if self.engine != "mistral":
            raise ValueError(f"unknown engine: {self.engine}")
        if not isinstance(self.dpi, int) or isinstance(self.dpi, bool):
            raise ValueError(f"dpi must be an integer, got {self.dpi!r}")
        if not DPI_MIN <= self.dpi <= DPI_MAX:
            raise ValueError(f"dpi must be within [{DPI_MIN}, {DPI_MAX}]")
        if not isinstance(self.vision_enabled, bool):
            raise ValueError(
                f"vision_enabled must be a boolean, got {self.vision_enabled!r}"
            )
        if not isinstance(self.skip_digital_pages, bool):
            raise ValueError(
                f"skip_digital_pages must be a boolean, got {self.skip_digital_pages!r}"
            )
        if not isinstance(self.vision_model, str) or not self.vision_model:
            raise ValueError(
                f"vision_model must be a non-empty string, got {self.vision_model!r}"
            )
        if self.vision_model.endswith("-latest"):
            raise ValueError(
                f"vision_model must be a pinned dated ID, not an alias: "
                f"{self.vision_model!r} (aliases silently ride model upgrades "
                "and price changes)"
            )
        if re.search(r"-\d{4}$", self.vision_model) is None:
            # Because vision soft-fails, a typo'd model would silently fail
            # detection on every page; catch it at configuration time.
            raise ValueError(
                f"vision_model must be a pinned dated ID ending in a date "
                f"suffix like -2505, got {self.vision_model!r}"
            )


def built_in_profiles() -> dict[str, OcrPolicy]:
    """Return the built-in consumer profiles."""

    return {
        "default": OcrPolicy(name="default"),
        "contracts": OcrPolicy(name="contracts"),
        # Job posts have no hand-altered contract values; the profile is
        # reserved for a future job_scraper migration and skips vision.
        "job_postings": OcrPolicy(name="job_postings", vision_enabled=False),
    }


def resolve_policy(
    profile: str, *, overrides: Mapping[str, Any] | None = None
) -> OcrPolicy:
    """Resolve a profile name plus optional field overrides.

    ``overrides`` maps policy field names to values for the selected
    profile, e.g. ``{"dpi": 400}``. Unknown fields raise — a typo like
    ``{"dip": 400}`` must fail loudly, not silently keep the default.
    (Consumers with JSON/env-driven config parse it themselves.)
    """

    profiles = built_in_profiles()
    if profile not in profiles:
        raise KeyError(f"unknown OCR profile: {profile}")
    policy = profiles[profile]

    if overrides:
        if not isinstance(overrides, Mapping):
            raise ValueError(
                f"overrides must be a mapping of field names, got {type(overrides).__name__}"
            )
        allowed = {item.name for item in fields(OcrPolicy)} - {"name"}
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(
                f"unknown policy override field(s) {sorted(unknown)}; "
                f"allowed: {sorted(allowed)}"
            )
        try:
            policy = replace(policy, **overrides)
        except TypeError as exc:
            raise ValueError(f"invalid override for profile {profile}: {exc}") from exc

    policy.validate()
    return policy
