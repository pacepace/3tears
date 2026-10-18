"""What a campaign is — a measurement, a choice or a change — read off its arms, never declared.

One function, so every reader of a campaign derives its kind one way: the readings
(:mod:`threetears.evals.analysis.campaign_readings`), the writer's bundle
(:attr:`~threetears.evals.analysis.campaign_bundle.CampaignBundle.kind`) and the page laid out over a stored analysis
(:mod:`threetears.evals.analysis.report.build`). It reads nothing but the arms, so it sits below all three.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from threetears.evals.kernel.declaration import CampaignArm

#: What a campaign is, derived from its arms and never declared. ``measurement``: one arm, read on its own against
#: absolute bars. ``choice``: two or more arms and no control (nothing is shipped), read pair by pair. ``change``: a
#: control and one or more candidates, each read against the control.
CampaignKind = Literal["measurement", "choice", "change"]


def campaign_kind(arms: Sequence[CampaignArm]) -> CampaignKind:
    """What a campaign is, from its arms (the named and the derived, as
    :func:`~threetears.evals.analysis.campaigns.campaign_arms` reads them): one arm a measurement, a control with any
    candidate a change, two or more arms with no control a choice. Arms are counted by variant key.

    :param arms: The campaign's arms.
    :ptype arms: Sequence[CampaignArm]
    :return: The kind.
    :rtype: CampaignKind
    """
    keys = {arm.variant_key for arm in arms}
    if len(keys) <= 1:
        return "measurement"
    return "change" if any(arm.role == "control" for arm in arms) else "choice"


__all__ = ["CampaignKind", "campaign_kind"]
