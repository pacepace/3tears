"""A report in one of its three forms: the one switch every door that hands a report out reads.

Markdown is what an agent reads and a pull request quotes, HTML the one page a person reads, its charts drawn by the kit's inlined script, and JSON
the bundle the report was computed from — the machine form, versioned by its ``schema_version`` — so a program never
reads the page's layout as data.
"""

from __future__ import annotations

import json
from typing import Literal

from threetears.evals.analysis.report.model import CampaignReport, Report
from threetears.evals.analysis.report.serialize_html import campaign_report_html, report_html
from threetears.evals.analysis.report.serialize_md import campaign_report_markdown, report_markdown

#: The forms a report is read in: Markdown, HTML that reads without any script, and the bundle behind it as JSON.
ReportFormat = Literal["markdown", "json", "html"]


def serialize_report(report: Report | CampaignReport, format: ReportFormat) -> str:
    """A report in one of its three forms: an aspect's page, or a campaign's.

    :param report: The report: one aspect's page, or a campaign's page over its aspects.
    :ptype report: Report | CampaignReport
    :param format: ``markdown``, ``html`` or ``json``. An aspect's ``json`` is the bundle its page was computed from,
        as ``bundle.json`` holds it; a campaign's is code's readings across its aspects, the person's call and each
        aspect's bundle by its fingerprint and shape (:meth:`CampaignReport.json_document`).
    :ptype format: ReportFormat
    :return: The serialized report.
    :rtype: str
    """
    if isinstance(report, CampaignReport):
        match format:
            case "markdown":
                return campaign_report_markdown(report)
            case "html":
                return campaign_report_html(report)
            case "json":
                return json.dumps(report.json_document(), indent=2)
    match format:
        case "markdown":
            return report_markdown(report)
        case "html":
            return report_html(report)
        case "json":
            return report.bundle.to_json(indent=2)


__all__ = ["ReportFormat", "serialize_report"]
