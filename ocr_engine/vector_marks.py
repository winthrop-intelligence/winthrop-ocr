"""Detect vector-drawn marks on PDF pages (the content pdfimages can't see).

Poppler's CLI tools report embedded text and raster images, but marks
drawn as vector paths — a stylus signature, annotation ink, a stamp drawn
with shape tools — are invisible to both. This module uses pdfplumber to
count, per page:

- ``curves``: bezier path segments. Straight lines and rectangles are
  deliberately NOT counted — they are table borders, signature-line rules,
  and underlines on virtually every contract — while drawn handwriting is
  made of curves.
- ``markup_annots``: annotations of the markup subtypes (Ink, FreeText,
  Stamp, ...). Benign subtypes (Link hyperlinks, Widget form fields) are
  ignored.

Run as a module (``python -m ocr_engine.vector_marks file.pdf --pages
1,2,3``) it prints a JSON object mapping each requested page number to its
counts. Classification invokes it exactly like the Poppler tools — inside
the memory-capped subprocess runner — so pdfplumber parses untrusted PDFs
only in a sandboxed child process, never in the caller's process.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

# Markup annotation subtypes carry visible reader-added content; Link,
# Widget, Popup, and friends do not.
MARKUP_ANNOTATION_SUBTYPES = frozenset(
    {
        "Ink",
        "FreeText",
        "Stamp",
        "Line",
        "Square",
        "Circle",
        "Polygon",
        "PolyLine",
        "Highlight",
        "StrikeOut",
        "Underline",
        "Squiggly",
        "Caret",
    }
)


def scan(pdf_path: Path, page_numbers: list[int]) -> dict[int, dict[str, int]]:
    """Count curves and markup annotations on the requested pages."""

    # pylint: disable-next=import-outside-toplevel
    import pdfplumber  # heavyweight; imported only in the sandboxed child

    counts: dict[int, dict[str, int]] = {}
    with pdfplumber.open(pdf_path, pages=page_numbers) as pdf:
        for page in pdf.pages:
            markup = 0
            for annot in page.annots or []:
                subtype = annot.get("data", {}).get("Subtype")
                name = getattr(subtype, "name", None) or str(subtype or "")
                if name.strip("/'\"") in MARKUP_ANNOTATION_SUBTYPES:
                    markup += 1
            counts[page.page_number] = {
                "curves": len(page.curves),
                "markup_annots": markup,
            }
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument(
        "--pages",
        required=True,
        help="comma-separated 1-based page numbers to scan",
    )
    arguments = parser.parse_args(argv)
    page_numbers = [int(number) for number in arguments.pages.split(",")]

    # pdfminer warns freely on real-world PDFs; keep stdout pure JSON and
    # stderr for actual failures.
    warnings.filterwarnings("ignore")
    counts = scan(arguments.pdf_path, page_numbers)
    json.dump({str(page): row for page, row in counts.items()}, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
