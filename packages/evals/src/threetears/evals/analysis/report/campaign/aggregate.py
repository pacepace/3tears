"""Figures a campaign's page reads across its aspects. None is read yet, so this part of the page is empty.

Two seams, each with one caller:

- :func:`equal_weight_cells` is each arm's figure across every aspect, which the scorecard
  (:mod:`~threetears.evals.analysis.report.campaign.scorecard`) places in its :data:`EQUAL_WEIGHT_ROW`; the scorecard
  lays the row out, and this module fills its cells.
- :func:`aggregate_blocks` is what the page says across aspects beside the scorecard, such as a chart, in the slot
  :mod:`~threetears.evals.analysis.report.campaign.page` gives it after the kind's own blocks.
"""

from __future__ import annotations

from collections.abc import Mapping

from threetears.evals.analysis.campaign_readings import CampaignReadings
from threetears.evals.analysis.report.model import Report, ReportBlock, TableCell
from threetears.evals.kernel.campaign import EvalCampaign

__all__ = ["EQUAL_WEIGHT_ROW", "aggregate_blocks", "equal_weight_cells"]

#: The scorecard row holding each arm's figure across every aspect, each aspect weighed alike.
EQUAL_WEIGHT_ROW = "All aspects (equal weight)"


def equal_weight_cells(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report]
) -> dict[str, TableCell]:
    """Each arm's cell in the scorecard's :data:`EQUAL_WEIGHT_ROW`, by its variant key: none yet.

    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :return: No cell, so the scorecard lays out no such row.
    :rtype: dict[str, TableCell]
    """
    return {}


def aggregate_blocks(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report]
) -> list[ReportBlock]:
    """What the page says across every aspect at once, beside the scorecard: nothing yet.

    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :return: No block.
    :rtype: list[ReportBlock]
    """
    return []
