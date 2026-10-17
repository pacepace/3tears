"""The report: one page every door into a campaign's report renders, and the bundle it was computed from.

- :mod:`model` — the :class:`Report` and its blocks (``text``, ``table``, ``chart``, ``trial``, ``disclosure``), and
  :meth:`Report.write`, which puts the page, its Markdown and ``bundle.json`` in one directory.
- :mod:`build` — :func:`build_report`, laying a stored analysis out over the bundle frozen with it, and
  :func:`build_code_only_report`, laying a campaign's evidence out when no analysis exists. One page either way.
- :mod:`verdict` — the first screen's basis and verdict lines, and the words for every way a trial is lost or fails.
- :mod:`svg` — the two charts, an outcome bar and interval rows, as dependency-free inline SVG.
- :mod:`serialize_md` and :mod:`serialize_html` — Markdown, and HTML that reads without a script; :mod:`serialize`
  is the switch, whose ``json`` is the bundle.
- :mod:`words` and :mod:`figures` — the words a report is read in, and how it states a figure.

Hosts import these from :mod:`threetears.evals.analysis`, the public root.
"""

from __future__ import annotations

from threetears.evals.analysis.report.build import (
    NO_ANALYSIS,
    analysis_bundle,
    analysis_verdict,
    build_code_only_report,
    build_report,
)
from threetears.evals.analysis.report.model import (
    SECTION_TITLES,
    BarOutcome,
    CellValue,
    ChartBlock,
    ChartKind,
    ContrastOutcome,
    DisclosureBlock,
    DisclosureSource,
    Fact,
    GuardrailOutcome,
    IntervalRow,
    OutcomeBar,
    OutcomeSegment,
    Report,
    ReportBasis,
    ReportBlock,
    ReportFile,
    ReportSection,
    ReportSource,
    TableBlock,
    TableCell,
    TableColumn,
    TextBlock,
    TextRole,
    TrialBlock,
    Verdict,
    VerdictKind,
    VerdictReason,
)
from threetears.evals.analysis.report.serialize_html import report_html
from threetears.evals.analysis.report.serialize_md import report_markdown
from threetears.evals.analysis.report.verdict import MIN_CASES

__all__ = [
    "MIN_CASES",
    "NO_ANALYSIS",
    "SECTION_TITLES",
    "BarOutcome",
    "CellValue",
    "ChartBlock",
    "ChartKind",
    "ContrastOutcome",
    "DisclosureBlock",
    "DisclosureSource",
    "Fact",
    "GuardrailOutcome",
    "IntervalRow",
    "OutcomeBar",
    "OutcomeSegment",
    "Report",
    "ReportBasis",
    "ReportBlock",
    "ReportFile",
    "ReportSection",
    "ReportSource",
    "TableBlock",
    "TableCell",
    "TableColumn",
    "TextBlock",
    "TextRole",
    "TrialBlock",
    "Verdict",
    "VerdictKind",
    "VerdictReason",
    "analysis_bundle",
    "analysis_verdict",
    "build_code_only_report",
    "build_report",
    "report_html",
    "report_markdown",
]
