"""The report: one versioned document an analysis is read through, and its three serializers.

- :mod:`model` — the :class:`Report` and its blocks (``text``, ``table``, ``chart``, ``disclosure``),
  with a published JSON Schema (``schema.json`` beside this module, :func:`report_json_schema`).
- :mod:`build` — :func:`build_report`, laying a stored analysis out as one, and
  :func:`build_code_only_report`, laying a campaign's evidence out as one when no analysis exists.
- :mod:`serialize_md` and :mod:`serialize_html` — Markdown for an agent, and HTML that reads without a
  script. JSON is the model's own dump (:meth:`Report.to_canonical_json`).
- :mod:`words` — the words a confidence, an evidence tier and an arm are read in, shared with the memo
  the reporter eval's judge reads.

Hosts import these from :mod:`threetears.evals.analysis`, the public root.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from threetears.evals.analysis.report.build import NO_ANALYSIS, build_code_only_report, build_report
from threetears.evals.analysis.report.model import (
    REPORT_VERSION,
    SECTION_TITLES,
    ChartBlock,
    DisclosureBlock,
    DisclosureSource,
    Fact,
    Report,
    ReportBasis,
    ReportBlock,
    ReportSection,
    ReportSource,
    TableBlock,
    TableColumn,
    TextBlock,
    TextRole,
    Verdict,
    VerdictKind,
    VerdictReason,
)
from threetears.evals.analysis.report.serialize_html import report_html
from threetears.evals.analysis.report.serialize_md import report_markdown

#: The published schema, generated from :class:`Report` and committed beside it so a host can read it
#: without importing the package. ``tests/test_report.py`` holds the file to the model. It states the shape and
#: every cross-field rule JSON Schema can express; the three that compare a value with a sibling's (a finding
#: position against ``finding_count``, ``total_rows`` against the rows shown, a row's keys against the columns)
#: only the model's validators hold — :mod:`model`'s docstring names them.
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.json"

#: The JSON Schema dialect the published schema declares.
_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def report_json_schema() -> dict[str, Any]:
    """The report's JSON Schema, as generated from the model — what ``schema.json`` must equal.

    Generated in serialization mode, because what it validates is a report's JSON dump.

    Returns:
        The schema.
    """
    return {"$schema": _DIALECT, **Report.model_json_schema(mode="serialization")}


def published_report_schema() -> dict[str, Any]:
    """The schema as published in ``schema.json``.

    Returns:
        The schema.
    """
    loaded: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return loaded


__all__ = [
    "NO_ANALYSIS",
    "REPORT_VERSION",
    "SCHEMA_PATH",
    "SECTION_TITLES",
    "ChartBlock",
    "DisclosureBlock",
    "DisclosureSource",
    "Fact",
    "Report",
    "ReportBasis",
    "ReportBlock",
    "ReportSection",
    "ReportSource",
    "TableBlock",
    "TableColumn",
    "TextBlock",
    "TextRole",
    "Verdict",
    "VerdictKind",
    "VerdictReason",
    "build_code_only_report",
    "build_report",
    "published_report_schema",
    "report_html",
    "report_json_schema",
    "report_markdown",
]
