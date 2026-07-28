"""Detect vector-drawn marks on PDF pages (the content pdfimages can't see).

Poppler's CLI tools report embedded text and raster images, but marks
drawn as vector paths — a stylus signature, annotation ink, a stamp drawn
with shape tools — are invisible to both. This module uses pdfplumber to
count, per page:

- ``curves``: bezier path segments. Drawn handwriting is made of curves.
- ``diagonal_lines``: straight segments that are neither horizontal nor
  vertical. Document layout (table borders, signature-line rules,
  underlines — present on 57% of legitimately skippable real contract
  pages) is axis-aligned, while a drawn "X" or check mark built from
  straight strokes is not.
- ``markup_annots``: annotations of the markup subtypes (Ink, FreeText,
  Stamp, ...). Benign subtypes (Link hyperlinks, Widget form fields) are
  ignored.

Axis-aligned lines and rectangles are deliberately NOT counted — doing so
would flag most born-digital contracts over their layout.

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

# A segment is axis-aligned (benign layout) when its extent on one axis
# stays within this many points — generous enough for rendering jitter,
# far below any deliberate diagonal stroke.
AXIS_ALIGNED_TOLERANCE_PTS = 2.0


def _is_diagonal(line: dict) -> bool:
    width = abs(line["x1"] - line["x0"])
    height = abs(line["bottom"] - line["top"])
    return (
        width > AXIS_ALIGNED_TOLERANCE_PTS
        and height > AXIS_ALIGNED_TOLERANCE_PTS
    )


def scan(pdf_path: Path, page_numbers: list[int]) -> dict[int, dict[str, int]]:
    """Count curves, diagonal lines, and markup annotations per page."""

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
                "diagonal_lines": sum(
                    1 for line in page.lines if _is_diagonal(line)
                ),
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
