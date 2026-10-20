"""Open the report of a `tabpilot run` directory and check it (P13 validation; PRD §21.1).

Usage:
    python scripts/check_report.py <run_dir>

Parses `report/report.html`, prints the section headings, and fails when the HTML does not
parse, a referenced figure is missing, `report.json` does not validate as `report/1.0`, or the
section count is not 17.
"""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path

from tabpilot.schemas.report import FinalReport

SECTIONS = 17


class _Outline(HTMLParser):
    """Collects <h2> headings and <img src> references."""

    def __init__(self) -> None:
        super().__init__()
        self.headings: list[str] = []
        self.images: list[str] = []
        self._in_h2 = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "h2":
            self._in_h2 = True
            self.headings.append("")
        if tag == "img":
            self.images.extend(v for k, v in attrs if k == "src" and v)

    def handle_endtag(self, tag: str) -> None:
        if tag == "h2":
            self._in_h2 = False

    def handle_data(self, data: str) -> None:
        if self._in_h2:
            self.headings[-1] += data


def check(run_dir: Path) -> list[str]:
    """Return the problems found (empty when the report is complete)."""
    report_dir = run_dir / "report"
    outline = _Outline()
    outline.feed((report_dir / "report.html").read_text(encoding="utf-8"))
    for heading in outline.headings:
        sys.stdout.write(heading + "\n")
    problems = [f"missing figure {src}" for src in outline.images
                if not (report_dir / src).is_file()]  # fmt: skip
    if len(outline.headings) != SECTIONS:
        problems.append(f"expected {SECTIONS} sections, found {len(outline.headings)}")
    report = FinalReport.model_validate_json((report_dir / "report.json").read_text("utf-8"))
    sys.stdout.write(f"narrative: {report.sections.narrative.source} "
                     f"(fact-check {report.sections.narrative.fact_check}); "
                     f"explanation: {report.sections.explanations.method}; "
                     f"figures: {len(outline.images)}\n")  # fmt: skip
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write(__doc__ or "")
        return 2
    problems = check(Path(argv[0]))
    for problem in problems:
        sys.stderr.write(problem + "\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
