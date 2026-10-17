"""A report in one of its three forms: the one switch every door that hands a report out reads.

Markdown is what an agent reads and a pull request quotes, HTML the one page a person reads with no script, and JSON
the bundle the report was computed from — the machine form, versioned by its ``schema_version`` — so a program never
reads the page's layout as data.
"""

from __future__ import annotations

from typing import Literal

from threetears.evals.analysis.report.model import Report
from threetears.evals.analysis.report.serialize_html import report_html
from threetears.evals.analysis.report.serialize_md import report_markdown

#: The forms a report is read in: Markdown, HTML that reads without any script, and the bundle behind it as JSON.
ReportFormat = Literal["markdown", "json", "html"]


def serialize_report(report: Report, format: ReportFormat) -> str:
    """A report in one of its three forms.

    :param report: The report.
    :ptype report: Report
    :param format: ``markdown``, ``html`` or ``json`` (the bundle the report was computed from, as ``bundle.json``
        holds it).
    :ptype format: ReportFormat
    :return: The serialized report.
    :rtype: str
    """
    match format:
        case "markdown":
            return report_markdown(report)
        case "html":
            return report_html(report)
        case "json":
            return report.bundle.to_json(indent=2)


__all__ = ["ReportFormat", "serialize_report"]
