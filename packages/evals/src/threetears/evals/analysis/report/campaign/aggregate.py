"""Figures a campaign's page reads across its aspects. None is read yet, so this part of the page is empty."""

from __future__ import annotations

from collections.abc import Mapping

from threetears.evals.analysis.campaign_readings import CampaignReadings
from threetears.evals.analysis.report.model import Report, ReportBlock
from threetears.evals.kernel.campaign import EvalCampaign

__all__ = ["aggregate_blocks"]


def aggregate_blocks(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report]
) -> list[ReportBlock]:
    """What the page says across every aspect at once: nothing yet.

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
