"""run_sandboxed: sandbox-infrastructure failures vs the tool's own verdict."""

from __future__ import annotations

import pytest

from ocr_engine.rendering import run_sandboxed


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
