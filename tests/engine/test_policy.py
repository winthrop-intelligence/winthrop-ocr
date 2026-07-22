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
