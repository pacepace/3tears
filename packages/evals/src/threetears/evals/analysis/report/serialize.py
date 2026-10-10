"""A report in one of its three forms: the one switch every surface that hands a report out reads.

Markdown is the memo and what an agent reads, JSON the report's canonical dump (what the published
schema validates), and HTML a page that reads without any script.
"""

from __future__ import annotations

from typing import Literal

from threetears.evals.analysis.report.model import Report
from threetears.evals.analysis.report.serialize_html import report_html
from threetears.evals.analysis.report.serialize_md import report_markdown

#: The forms a report is read in: Markdown (the memo, and what an agent reads), its canonical JSON (what
#: the published schema validates) and HTML that reads without any script.
ReportFormat = Literal["markdown", "json", "html"]


def serialize_report(report: Report, format: ReportFormat) -> str:
    """A report in one of its three forms.

    Args:
        report: The report.
        format: ``markdown``, ``json`` (canonical, what the published schema validates) or ``html``.

    Returns:
        The serialized report.
    """
    match format:
        case "markdown":
            return report_markdown(report)
        case "html":
            return report_html(report)
        case "json":
            return report.to_canonical_json()


__all__ = ["ReportFormat", "serialize_report"]
