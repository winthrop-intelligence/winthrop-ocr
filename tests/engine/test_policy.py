"""Profile resolution and overrides."""

import pytest

from ocr_engine.policy import resolve_policy


class TestResolvePolicy:
    def test_built_in_profiles_use_mistral(self):
        for profile in ("default", "contracts", "job_postings"):
            policy = resolve_policy(profile)
            assert policy.engine == "mistral"
            assert policy.dpi == 300

    def test_unknown_profile_raises(self):
        with pytest.raises(KeyError):
            resolve_policy("nope")

    def test_overrides_apply(self):
        assert resolve_policy("contracts", overrides={"dpi": 400}).dpi == 400

    def test_invalid_override_values_are_rejected(self):
        with pytest.raises(ValueError):
            resolve_policy("default", overrides={"dpi": 9999})
        with pytest.raises(ValueError):
            resolve_policy("default", overrides={"engine": "not-a-real-engine"})
        with pytest.raises(ValueError):
            resolve_policy("default", overrides={"dpi": "high"})

    def test_unknown_override_fields_are_rejected(self):
        # A typo must fail loudly, not silently keep the default.
        with pytest.raises(ValueError, match="unknown policy override"):
            resolve_policy("default", overrides={"dip": 400})
        with pytest.raises(ValueError, match="unknown policy override"):
            resolve_policy("default", overrides={"name": "hacked", "dpi": 400})

    def test_non_mapping_overrides_are_rejected(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            resolve_policy("default", overrides=[("dpi", 400)])

    def test_fingerprint_changes_with_overrides(self):
        base = resolve_policy("default")
        tuned = resolve_policy("default", overrides={"dpi": 400})
        assert base.fingerprint() != tuned.fingerprint()


class TestVisionPolicy:
    def test_vision_defaults_per_profile(self):
        # Alterations are a contracts concept; the (currently unused)
        # job_postings profile must not pay for vision.
        assert resolve_policy("default").vision_enabled is True
        assert resolve_policy("contracts").vision_enabled is True
        assert resolve_policy("job_postings").vision_enabled is False

    def test_vision_model_is_pinned_by_default(self):
        for profile in ("default", "contracts", "job_postings"):
            assert resolve_policy(profile).vision_model == "mistral-medium-2505"

    def test_vision_overrides_apply(self):
        off = resolve_policy("contracts", overrides={"vision_enabled": False})
        assert off.vision_enabled is False
        pinned = resolve_policy(
            "contracts", overrides={"vision_model": "mistral-medium-2508"}
        )
        assert pinned.vision_model == "mistral-medium-2508"

    def test_vision_overrides_change_fingerprint(self):
        base = resolve_policy("contracts")
        off = resolve_policy("contracts", overrides={"vision_enabled": False})
        assert base.fingerprint() != off.fingerprint()

    def test_alias_models_are_rejected(self):
        # "-latest" aliases silently ride upgrades and price changes.
        with pytest.raises(ValueError, match="pinned dated ID"):
            resolve_policy(
                "contracts", overrides={"vision_model": "mistral-medium-latest"}
            )

    def test_invalid_vision_values_are_rejected(self):
        with pytest.raises(ValueError, match="vision_model"):
            resolve_policy("contracts", overrides={"vision_model": ""})
        with pytest.raises(ValueError, match="vision_enabled"):
            resolve_policy("contracts", overrides={"vision_enabled": "yes"})

    def test_undated_model_names_are_rejected(self):
        # Vision soft-fails, so a typo'd model would silently fail detection
        # on every page; the dated-suffix contract catches it up front.
        for bad in ("mistral-medium", "mistral-medium-25o5", "pixtral-large"):
            with pytest.raises(ValueError, match="date suffix"):
                resolve_policy("contracts", overrides={"vision_model": bad})
