"""Vision alteration detection: request shape, parsing, retries, soft-fail."""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from ocr_engine import vision as vision_module
from ocr_engine.models import PageAlterations, PageInput
from ocr_engine.policy import resolve_policy
from ocr_engine.vision import (
    ALTERATIONS_PROMPT,
    MAX_ALTERATION_ENTRIES,
    VISION_ATTEMPT_TIMEOUT_MS,
    VISION_MAX_ATTEMPTS,
    detect_alterations,
)


def sdk_error(message: str, status_code: int) -> Exception:
    """An exception shaped like the SDK's: message plus a status_code attr."""

    error = Exception(message)
    error.status_code = status_code
    return error


def make_page(tmp_path: Path) -> PageInput:
    image = tmp_path / "page-1.png"
    image.write_bytes(b"png-bytes")
    return PageInput(
        document_id="doc-1",
        source_path=tmp_path / "doc.pdf",
        page_number=1,
        image_path=image,
        image_sha256="abc",
        dpi=300,
    )


def chat_response(content) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


FOUND = json.dumps(
    {
        "alterations": [
            {
                "clause": "4",
                # A model volunteering value transcriptions anyway — these
                # must be stripped, never surfaced.
                "printed_value_struck": "$250,000",
                "handwritten_value": "$325,000",
                "kind": "dollar_amount",
            }
        ],
        "none_found": False,
    }
)
NONE_FOUND = json.dumps({"alterations": [], "none_found": True})


class TestDetectAlterations:
    @pytest.fixture(name="fake_sdk")
    def fake_sdk_fixture(self, monkeypatch):
        """Install a scriptable mistralai chat SDK; return (calls, script, reply)."""

        calls = []
        script = []  # exceptions to raise before finally succeeding
        reply = {"content": FOUND}

        class FakeChat:
            def complete(self, **kwargs):
                calls.append(kwargs)
                if script:
                    raise script.pop(0)
                return chat_response(reply["content"])

        class FakeMistral:
            def __init__(self, api_key):
                self.api_key = api_key
                self.chat = FakeChat()

        parent = types.ModuleType("mistralai")
        client_module = types.ModuleType("mistralai.client")
        client_module.Mistral = FakeMistral
        parent.client = client_module
        monkeypatch.setitem(sys.modules, "mistralai", parent)
        monkeypatch.setitem(sys.modules, "mistralai.client", client_module)
        monkeypatch.setattr(vision_module, "module_available", lambda _m: True)
        monkeypatch.setenv("MISTRAL_API_KEY", "k")
        monkeypatch.setattr(vision_module.time, "sleep", lambda _s: None)
        return calls, script, reply

    def test_request_shape(self, fake_sdk, tmp_path):
        calls, _script, _reply = fake_sdk
        policy = resolve_policy("contracts")
        result = detect_alterations(make_page(tmp_path), policy)
        assert result.status == "success"
        assert len(calls) == 1
        call = calls[0]
        assert call["model"] == "mistral-medium-2505"
        assert call["temperature"] == 0
        assert call["response_format"] == {"type": "json_object"}
        assert call["timeout_ms"] == VISION_ATTEMPT_TIMEOUT_MS
        content = call["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": ALTERATIONS_PROMPT}
        assert content[1]["image_url"].startswith("data:image/png;base64,")

    def test_alterations_found(self, fake_sdk, tmp_path):
        _calls, _script, _reply = fake_sdk
        result = detect_alterations(make_page(tmp_path), resolve_policy("contracts"))
        assert result.flagged is True
        assert result.none_found is False
        assert result.model == "mistral-medium-2505"

    def test_value_transcriptions_are_stripped(self, fake_sdk, tmp_path):
        # The flag is reliable, transcribed values are not: even when the
        # model volunteers them, only clause + kind may surface.
        _calls, _script, _reply = fake_sdk
        result = detect_alterations(make_page(tmp_path), resolve_policy("contracts"))
        assert result.alterations == [{"clause": "4", "kind": "dollar_amount"}]

    def test_none_found(self, fake_sdk, tmp_path):
        _calls, _script, reply = fake_sdk
        reply["content"] = NONE_FOUND
        result = detect_alterations(make_page(tmp_path), resolve_policy("contracts"))
        assert result.status == "success"
        assert result.flagged is False
        assert result.none_found is True
        assert result.alterations == []

    def test_fenced_json_is_tolerated(self, fake_sdk, tmp_path):
        _calls, _script, reply = fake_sdk
        reply["content"] = f"```json\n{NONE_FOUND}\n```"
        result = detect_alterations(make_page(tmp_path), resolve_policy("contracts"))
        assert result.status == "success"
        assert result.none_found is True

    def test_chunked_content_is_tolerated(self, fake_sdk, tmp_path):
        _calls, _script, reply = fake_sdk
        reply["content"] = [SimpleNamespace(text=NONE_FOUND)]
        result = detect_alterations(make_page(tmp_path), resolve_policy("contracts"))
        assert result.status == "success"

    def test_json_with_prose_wrapper_is_repaired(self, fake_sdk, tmp_path):
        _calls, _script, reply = fake_sdk
        reply["content"] = f"Here is the verdict: {NONE_FOUND} Done."
        result = detect_alterations(make_page(tmp_path), resolve_policy("contracts"))
        assert result.status == "success"

    def test_garbage_is_a_parse_error(self, fake_sdk, tmp_path):
        _calls, _script, reply = fake_sdk
        reply["content"] = "not json at all"
        result = detect_alterations(make_page(tmp_path), resolve_policy("contracts"))
        assert result.status == "parse_error"
        assert result.flagged is False
        assert "not json at all" in result.error_message

    def test_missing_alterations_key_is_a_parse_error(self, fake_sdk, tmp_path):
        _calls, _script, reply = fake_sdk
        reply["content"] = json.dumps({"verdict": "clean"})
        result = detect_alterations(make_page(tmp_path), resolve_policy("contracts"))
        assert result.status == "parse_error"

    def test_entries_are_bounded(self, fake_sdk, tmp_path):
        _calls, _script, reply = fake_sdk
        reply["content"] = json.dumps(
            {
                "alterations": [{"clause": "c", "kind": "other"}] * 80,
                "none_found": False,
            }
        )
        result = detect_alterations(make_page(tmp_path), resolve_policy("contracts"))
        assert len(result.alterations) == MAX_ALTERATION_ENTRIES

    def test_contradictory_verdict_cannot_surface(self, fake_sdk, tmp_path):
        # {"alterations": [{}], "none_found": true}: the empty entry is junk
        # (no locatable clause) and none_found is derived, never trusted —
        # so the verdict resolves to a consistent "nothing found".
        _calls, _script, reply = fake_sdk
        reply["content"] = json.dumps({"alterations": [{}], "none_found": True})
        result = detect_alterations(make_page(tmp_path), resolve_policy("contracts"))
        assert result.status == "success"
        assert result.alterations == []
        assert result.flagged is False
        assert result.none_found is True

    def test_none_found_is_derived_not_coerced(self, fake_sdk, tmp_path):
        # A model returning the STRING "false" must not become True via
        # bool(); none_found comes from the entries alone.
        _calls, _script, reply = fake_sdk
        reply["content"] = json.dumps(
            {
                "alterations": [{"clause": "4", "kind": "date"}],
                "none_found": "false",
            }
        )
        result = detect_alterations(make_page(tmp_path), resolve_policy("contracts"))
        assert result.flagged is True
        assert result.none_found is False

    def test_unknown_kind_is_normalized_and_junk_entries_dropped(
        self, fake_sdk, tmp_path
    ):
        _calls, _script, reply = fake_sdk
        reply["content"] = json.dumps(
            {
                "alterations": [
                    {"clause": "4", "kind": "banana"},  # invented kind
                    {"kind": "date"},  # no clause: useless to a reviewer
                    {"clause": 12, "kind": "date"},  # non-string clause
                    {"clause": "   ", "kind": "date"},  # blank clause
                ],
                "none_found": False,
            }
        )
        result = detect_alterations(make_page(tmp_path), resolve_policy("contracts"))
        assert result.alterations == [{"clause": "4", "kind": "other"}]

    def test_entry_values_are_bounded(self, fake_sdk, tmp_path):
        from ocr_engine.vision import ENTRY_VALUE_MAX_CHARS

        _calls, _script, reply = fake_sdk
        reply["content"] = json.dumps(
            {
                "alterations": [{"clause": "x" * 5000, "kind": "other"}],
                "none_found": False,
            }
        )
        result = detect_alterations(make_page(tmp_path), resolve_policy("contracts"))
        assert len(result.alterations[0]["clause"]) == ENTRY_VALUE_MAX_CHARS

    def test_transient_error_is_retried_then_succeeds(self, fake_sdk, tmp_path):
        calls, script, _reply = fake_sdk
        script.append(ConnectionError("connection reset"))
        result = detect_alterations(make_page(tmp_path), resolve_policy("contracts"))
        assert result.status == "success"
        assert result.transport_retries == 1
        assert len(calls) == 2

    def test_auth_error_fails_after_one_attempt(self, fake_sdk, tmp_path):
        calls, script, _reply = fake_sdk
        script.append(sdk_error("unauthorized", 401))
        result = detect_alterations(make_page(tmp_path), resolve_policy("contracts"))
        assert result.status == "auth_error"
        assert len(calls) == 1

    def test_persistent_transient_errors_exhaust_attempts(self, fake_sdk, tmp_path):
        calls, script, _reply = fake_sdk
        script.extend(
            ConnectionError("connection reset") for _ in range(VISION_MAX_ATTEMPTS)
        )
        result = detect_alterations(make_page(tmp_path), resolve_policy("contracts"))
        assert result.status == "crash"
        assert result.error_type == "ConnectionError"
        assert len(calls) == VISION_MAX_ATTEMPTS

    def test_rate_limit_uses_longer_backoff(self, monkeypatch, fake_sdk, tmp_path):
        from ocr_engine.adapters import mistral as mistral_module

        calls, script, _reply = fake_sdk
        script.append(sdk_error("slow down", 429))
        sleeps = []
        monkeypatch.setattr(vision_module.time, "sleep", sleeps.append)
        result = detect_alterations(make_page(tmp_path), resolve_policy("contracts"))
        assert result.status == "success"
        assert len(calls) == 2
        assert sleeps[0] >= mistral_module.MISTRAL_RATE_LIMIT_DELAYS_SECONDS[0]

    def test_hostile_response_never_raises(self, fake_sdk, tmp_path, monkeypatch):
        _calls, _script, _reply = fake_sdk

        class Hostile:
            @property
            def choices(self):
                raise RuntimeError("boom")

        monkeypatch.setattr(
            vision_module,
            "_complete_vision_with_retries",
            lambda **_kw: (Hostile(), 0),
        )
        result = detect_alterations(make_page(tmp_path), resolve_policy("contracts"))
        assert result.status == "crash"

    def test_missing_api_key_is_unavailable_without_sdk_call(
        self, fake_sdk, tmp_path, monkeypatch
    ):
        calls, _script, _reply = fake_sdk
        monkeypatch.delenv("MISTRAL_API_KEY")
        result = detect_alterations(make_page(tmp_path), resolve_policy("contracts"))
        assert result.status == "unavailable"
        assert calls == []


class TestRealSdkContract:
    def test_request_kwargs_bind_to_the_real_sdk_signature(self):
        """Guard against SDK upgrades renaming/dropping request parameters.

        The fake-SDK tests verify what we SEND; this binds the exact
        production request against the REAL installed mistralai client's
        signature (no network) so an incompatible upgrade fails here
        instead of soft-failing vision on every production page.
        """

        import inspect

        pytest.importorskip("mistralai")
        # The real import path used by the production code.
        from mistralai.client import Mistral  # pylint: disable=import-error

        from ocr_engine.vision import _vision_request_kwargs

        client = Mistral(api_key="test-key")  # construction is offline
        request = _vision_request_kwargs(
            model="mistral-medium-2505",
            image_url="data:image/png;base64,x",
        )
        # Raises TypeError if any parameter name is not accepted.
        inspect.signature(client.chat.complete).bind(**request)


class TestPageAlterationsModel:
    def test_roundtrip(self):
        original = PageAlterations(
            status="success",
            model="mistral-medium-2505",
            alterations=[{"kind": "date", "clause": "1"}],
            none_found=False,
            elapsed_ms=3600,
            transport_retries=1,
        )
        restored = PageAlterations.from_dict(original.to_dict())
        assert restored == original

    def test_from_dict_ignores_unknown_fields(self):
        restored = PageAlterations.from_dict(
            {"status": "success", "model": "m", "future_field": 1}
        )
        assert restored.status == "success"

    def test_flagged_requires_success(self):
        failed = PageAlterations(
            status="crash", model="m", alterations=[{"kind": "other"}]
        )
        assert failed.flagged is False
