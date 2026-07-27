"""Pure-logic tests for the Mistral adapter helpers and retry loop."""

import sys
import types
from types import SimpleNamespace

import pytest

from ocr_engine.adapters import mistral as mistral_module
from ocr_engine.adapters.mistral import (
    MISTRAL_ATTEMPT_TIMEOUT_MS,
    SIGNATURE_CONTENT_MAX_CHARS,
    MistralEngine,
    _is_retryable_mistral_error,
    _mistral_error_status,
    _page_confidence,
    _process_ocr_with_retries,
    _serialize_page_signals,
)
from ocr_engine.adapters.registry import engine_registry


def sdk_error(message: str, status_code: int) -> Exception:
    """An exception shaped like the SDK's: message plus a status_code attr."""

    error = Exception(message)
    error.status_code = status_code
    return error


class TestErrorMapping:
    def test_auth_errors_via_status_code(self):
        assert _mistral_error_status(sdk_error("nope", 401)) == "auth_error"
        assert _mistral_error_status(sdk_error("nope", 403)) == "auth_error"

    def test_auth_errors_via_message_fallback(self):
        assert _mistral_error_status(Exception("401 authentication failed")) == "auth_error"

    def test_rate_limits(self):
        assert _mistral_error_status(sdk_error("slow down", 429)) == "rate_limited"
        assert _mistral_error_status(Exception("429 rate limit hit")) == "rate_limited"

    def test_timeouts(self):
        assert _mistral_error_status(Exception("request timeout")) == "timeout"

    def test_everything_else_is_a_crash(self):
        assert _mistral_error_status(Exception("boom")) == "crash"


class TestRetryClassification:
    def test_server_side_statuses_are_retryable(self):
        assert _is_retryable_mistral_error(sdk_error("err", 429))
        assert _is_retryable_mistral_error(sdk_error("err", 503))
        assert _is_retryable_mistral_error(sdk_error("cdn", 520))  # full 5xx range
        assert not _is_retryable_mistral_error(sdk_error("bad request", 400))
        assert not _is_retryable_mistral_error(sdk_error("unauthorized", 401))

    def test_retry_delay_clamps_beyond_the_schedule(self):
        # Growing MISTRAL_MAX_ATTEMPTS must never IndexError mid-retry.
        delay = mistral_module._retry_delay(ConnectionError("reset"), attempt=7)
        assert delay >= mistral_module.MISTRAL_RETRY_DELAYS_SECONDS[-1]

    def test_transport_errors_are_retryable(self):
        assert _is_retryable_mistral_error(ConnectionError("connection reset"))
        assert _is_retryable_mistral_error(Exception("read timed out"))

    def test_status_digits_in_message_content_do_not_trigger_retries(self):
        # A permanent validation error mentioning "500" (e.g. a pixel size)
        # must not be retried.
        assert not _is_retryable_mistral_error(Exception("image width 500px invalid"))
        assert not _is_retryable_mistral_error(Exception("invalid request"))


class TestRetryLoop:
    @pytest.fixture(name="fake_sdk")
    def fake_sdk_fixture(self, monkeypatch):
        """Install a scriptable mistralai.client.Mistral; return the call log."""

        calls = []
        script = []  # exceptions to raise before finally succeeding

        class FakeOcr:
            def process(self, **kwargs):
                calls.append(kwargs)
                if script:
                    raise script.pop(0)
                return SimpleNamespace(
                    pages=[SimpleNamespace(markdown="hi", confidence_scores=None)]
                )

        class FakeMistral:
            def __init__(self, api_key):
                self.api_key = api_key
                self.ocr = FakeOcr()

        parent = types.ModuleType("mistralai")
        client_module = types.ModuleType("mistralai.client")
        client_module.Mistral = FakeMistral
        parent.client = client_module
        monkeypatch.setitem(sys.modules, "mistralai", parent)
        monkeypatch.setitem(sys.modules, "mistralai.client", client_module)
        monkeypatch.setattr(mistral_module.time, "sleep", lambda _s: None)
        return calls, script

    def test_first_attempt_success_sends_owned_timeout(self, fake_sdk):
        calls, _script = fake_sdk
        response, retries = _process_ocr_with_retries(api_key="k", image_url="data:x")
        assert retries == 0
        assert len(calls) == 1
        assert calls[0]["timeout_ms"] == MISTRAL_ATTEMPT_TIMEOUT_MS
        assert response.pages[0].markdown == "hi"

    def test_transient_error_is_retried_then_succeeds(self, fake_sdk):
        calls, script = fake_sdk
        script.append(ConnectionError("connection reset"))
        _response, retries = _process_ocr_with_retries(api_key="k", image_url="data:x")
        assert retries == 1
        assert len(calls) == 2

    def test_non_retryable_error_raises_after_one_attempt(self, fake_sdk):
        calls, script = fake_sdk
        script.append(sdk_error("unauthorized", 401))
        with pytest.raises(Exception, match="unauthorized"):
            _process_ocr_with_retries(api_key="k", image_url="data:x")
        assert len(calls) == 1

    def test_persistent_transient_errors_exhaust_attempts(self, fake_sdk):
        calls, script = fake_sdk
        script.extend(ConnectionError("connection reset") for _ in range(3))
        with pytest.raises(ConnectionError):
            _process_ocr_with_retries(api_key="k", image_url="data:x")
        assert len(calls) == 3  # MISTRAL_MAX_ATTEMPTS

    def test_rate_limit_uses_longer_backoff(self, monkeypatch, fake_sdk):
        calls, script = fake_sdk
        script.append(sdk_error("slow down", 429))
        sleeps = []
        monkeypatch.setattr(mistral_module.time, "sleep", sleeps.append)
        _process_ocr_with_retries(api_key="k", image_url="data:x")
        assert len(calls) == 2
        assert len(sleeps) == 1
        assert sleeps[0] >= mistral_module.MISTRAL_RATE_LIMIT_DELAYS_SECONDS[0]


class TestImageDataUrl:
    def test_oversized_image_raises_actionable_error(self, tmp_path, monkeypatch):
        from ocr_engine.adapters import base

        image = tmp_path / "huge.png"
        image.write_bytes(b"x" * 100)
        monkeypatch.setattr(base, "IMAGE_MAX_BYTES", 50)
        with pytest.raises(ValueError, match="lower the profile dpi"):
            base.image_data_url(image)

    def test_image_under_limit_encodes(self, tmp_path):
        from ocr_engine.adapters import base

        image = tmp_path / "page.png"
        image.write_bytes(b"data")
        assert base.image_data_url(image).startswith("data:image/png;base64,")

    def test_media_type_follows_suffix(self, tmp_path):
        from ocr_engine.adapters import base

        image = tmp_path / "page.tif"
        image.write_bytes(b"data")
        assert base.image_data_url(image).startswith("data:image/tiff;base64,")


class TestPageSignalsSerialization:
    def test_blocks_images_dimensions_preserved(self):
        page = SimpleNamespace(
            dimensions=SimpleNamespace(dpi=300, height=3300, width=2550),
            blocks=[
                SimpleNamespace(
                    type="text",
                    top_left_x=10,
                    top_left_y=10,
                    bottom_right_x=500,
                    bottom_right_y=600,
                    content="x" * 5000,
                ),
                SimpleNamespace(
                    type="signature",
                    top_left_x=10,
                    top_left_y=700,
                    bottom_right_x=300,
                    bottom_right_y=760,
                    content="J. Smith",
                ),
            ],
            images=[
                SimpleNamespace(
                    id="img-1",
                    top_left_x=1,
                    top_left_y=2,
                    bottom_right_x=3,
                    bottom_right_y=4,
                )
            ],
        )
        signals = _serialize_page_signals(page)
        assert signals["dimensions"] == {"dpi": 300, "height": 3300, "width": 2550}
        text_block, sig_block = signals["blocks"]
        # Long non-signature content is reduced to a length only.
        assert text_block == {
            "type": "text",
            "bbox": [10, 10, 500, 600],
            "content_chars": 5000,
        }
        assert "content" not in text_block
        # Signature content (the transcribed name) is kept.
        assert sig_block["content"] == "J. Smith"
        assert signals["images"] == [{"id": "img-1", "bbox": [1, 2, 3, 4]}]

    def test_signature_content_is_truncated(self):
        page = SimpleNamespace(
            dimensions=None,
            blocks=[SimpleNamespace(type="signature", content="n" * 1000)],
            images=[],
        )
        block = _serialize_page_signals(page)["blocks"][0]
        assert len(block["content"]) == SIGNATURE_CONTENT_MAX_CHARS
        assert block["bbox"] is None  # missing coordinates degrade to None

    def test_missing_attributes_yield_empty_shape(self):
        # The retry-loop fakes (SimpleNamespace without blocks) must serialize.
        page = SimpleNamespace(markdown="hi", confidence_scores=None)
        signals = _serialize_page_signals(page)
        assert signals == {"dimensions": None, "blocks": [], "images": []}

    def test_serializer_never_raises(self):
        class Hostile:
            @property
            def blocks(self):
                raise RuntimeError("boom")

        signals = _serialize_page_signals(Hostile())
        assert "serialization_error" in signals

    def test_one_bad_block_does_not_erase_the_rest(self):
        # A poisoned block must not hide the signature block next to it.
        page = SimpleNamespace(
            dimensions=None,
            blocks=[
                SimpleNamespace(type="text", content=12345),  # len(int) raises
                SimpleNamespace(type="signature", content="J. Smith"),
            ],
            images=[],
        )
        blocks = _serialize_page_signals(page)["blocks"]
        assert blocks[0]["type"] == "SERIALIZATION_ERROR"
        assert blocks[1]["type"] == "signature"
        assert blocks[1]["content"] == "J. Smith"

    def test_image_truncation_is_flagged(self):
        page = SimpleNamespace(
            dimensions=None,
            blocks=[],
            images=[SimpleNamespace(id=f"img-{n}") for n in range(60)],
        )
        signals = _serialize_page_signals(page)
        assert len(signals["images"]) == 50
        assert signals["images_truncated"] is True


class TestPageConfidence:
    def test_dict_payload_scales_to_percent(self):
        page = type(
            "Page", (), {"confidence_scores": {"average_page_confidence_score": 0.87}}
        )()
        assert _page_confidence(page) == 87.0

    def test_missing_scores_is_none(self):
        page = type("Page", (), {"confidence_scores": None})()
        assert _page_confidence(page) is None


class TestRegistry:
    def test_registry_is_mistral_only(self):
        registry = engine_registry()
        assert set(registry) == {"mistral"}
        assert isinstance(registry["mistral"], MistralEngine)

    def test_availability_reports_missing_key(self, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        available, message = MistralEngine.availability()
        # Either the SDK extra is absent or the key is unset — both disable it.
        assert available is False
        assert message
