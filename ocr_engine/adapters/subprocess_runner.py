"""Run an OCR command after applying child-process resource limits.

Ported from winad_scrapers job_scraper/pdf_ocr_runner.py. Invoked as
``python -m ocr_engine.adapters.subprocess_runner --memory-limit-bytes N -- <cmd>``
so the limit applies to a fresh process (avoids preexec_fn, which is unsafe
under threaded/prefork parents).
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on some platforms.
    resource = None


MEMORY_LIMIT_SETUP_FAILURE_EXIT_CODE = 125
COMMAND_EXEC_FAILURE_EXIT_CODE = 126
COMMAND_NOT_FOUND_EXIT_CODE = 127


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse runner arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-limit-bytes",
        type=int,
        required=True,
        help="Linux RLIMIT_AS value to apply before running the command.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to exec after --.",
    )
    args = parser.parse_args(argv)

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]

    return args


def apply_memory_limit(memory_limit_bytes: int) -> bool:
    """Apply a Linux address-space limit to the current process."""

    if not sys.platform.startswith("linux"):
        return True

    if resource is None or not hasattr(resource, "RLIMIT_AS"):
        print(
            "resource.RLIMIT_AS is unavailable; refusing uncapped OCR.", file=sys.stderr
        )
        return False

    try:
        resource.setrlimit(
            resource.RLIMIT_AS,
            (memory_limit_bytes, memory_limit_bytes),
        )
    except (OSError, ValueError) as exc:
        print(f"Failed to apply OCR memory limit: {exc}", file=sys.stderr)
        return False

    return True


def main(argv: list[str] | None = None) -> int:
    """Apply process limits and replace this process with the OCR command."""

    args = parse_args(argv)
    if not args.command:
        print("No OCR command provided.", file=sys.stderr)
        return COMMAND_EXEC_FAILURE_EXIT_CODE

    if not apply_memory_limit(args.memory_limit_bytes):
        return MEMORY_LIMIT_SETUP_FAILURE_EXIT_CODE

    try:
        os.execvp(args.command[0], args.command)
    except FileNotFoundError as exc:
        print(f"OCR command not found: {exc}", file=sys.stderr)
        return COMMAND_NOT_FOUND_EXIT_CODE
    except OSError as exc:
        print(f"Failed to exec OCR command: {exc}", file=sys.stderr)
        return COMMAND_EXEC_FAILURE_EXIT_CODE

    return 0


if __name__ == "__main__":
    sys.exit(main())
