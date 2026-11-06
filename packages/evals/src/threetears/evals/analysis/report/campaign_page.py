"""A campaign's page: its public names, each defined in the part of :mod:`threetears.evals.analysis.report.campaign`
that lays it out.

- :func:`build_campaign_report` lays the page out (:mod:`~threetears.evals.analysis.report.campaign.page`).
- :func:`campaign_warnings` is the loud lines it opens with, word for word, for whatever repeats them
  (:mod:`~threetears.evals.analysis.report.campaign.warnings`).
- :func:`timing_caveats` is its caveat lines on when its criteria were set, which every page and listing line says
  once (:mod:`~threetears.evals.analysis.report.campaign.provenance`).
- :func:`campaign_verdict`, :data:`READING_WORDS`, :data:`BAR_WORDS` and :data:`CHOICE_NOTE` are how it says what
  code read (:mod:`~threetears.evals.analysis.report.campaign.scorecard`).
- :class:`AspectNames` is what it calls each aspect, which those lines take
  (:mod:`~threetears.evals.analysis.report.campaign.names`).
- :data:`NO_RECOMMENDATION` and :data:`NO_CALL` are its lines where no memo or no call exists
  (:mod:`~threetears.evals.analysis.report.campaign.memo_blocks`).
"""

from __future__ import annotations

from threetears.evals.analysis.report.campaign.memo_blocks import NO_CALL, NO_RECOMMENDATION
from threetears.evals.analysis.report.campaign.names import AspectNames
from threetears.evals.analysis.report.campaign.page import build_campaign_report
from threetears.evals.analysis.report.campaign.provenance import timing_caveats
from threetears.evals.analysis.report.campaign.scorecard import BAR_WORDS, CHOICE_NOTE, READING_WORDS, campaign_verdict
from threetears.evals.analysis.report.campaign.warnings import campaign_warnings

__all__ = [
    "BAR_WORDS",
    "AspectNames",
    "CHOICE_NOTE",
    "NO_CALL",
    "NO_RECOMMENDATION",
    "READING_WORDS",
    "build_campaign_report",
    "campaign_verdict",
    "campaign_warnings",
    "timing_caveats",
]
