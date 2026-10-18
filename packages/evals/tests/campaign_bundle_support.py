"""A campaign bundle built from aspect bundles a test already holds, with no campaign or store behind it.

The generator reads a :class:`~threetears.evals.analysis.campaign_bundle.CampaignBundle`, which a service assembles
from a stored campaign. A test that holds one aspect's assembled bundle (and perhaps rewrote part of it) builds the
campaign bundle around it here: the arms are read off the bundles' variant indexes, the control as the realized design
matched it, every one labelled as the arm table names it; code's readings carry whatever floors the test hands in.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from threetears.evals.analysis.arms import arm_names
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.campaign_readings import (
    AspectReading,
    BarStatus,
    CampaignReadings,
    FloorStatus,
    PairReading,
    campaign_kind,
)
from threetears.evals.kernel.campaign import CampaignDeclaration
from threetears.evals.kernel.declaration import Aspect, CampaignArm, Floor, resolve_control_arm

__all__ = ["arms_of", "campaign_bundle", "label_of", "memo"]


def arms_of(bundles: Sequence[AnalysisContextBundle]) -> list[CampaignArm]:
    """Every arm the bundles index, the control first, each labelled as the arm table names it.

    The control is the one the aspect declares, which names its cells under every identity predicate (the earlier
    one's included); with none declared, the arm the realized design matched; every other indexed variant is a
    candidate.

    :param bundles: The aspects' bundles.
    :ptype bundles: Sequence[AnalysisContextBundle]
    :return: The arms.
    :rtype: list[CampaignArm]
    """
    entries = {entry.variant_key: entry for bundle in bundles for entry in bundle.variant_index}
    names = arm_names(list(entries.values()))
    declared = next(
        (bundle.declared.control for bundle in bundles if bundle.declared is not None and bundle.declared.control),
        None,
    )
    realized = next(
        (bundle.design.control_arm.variant_key for bundle in bundles if bundle.design.control_arm is not None), None
    )
    arms: list[CampaignArm] = []
    covered: set[str] = set()
    if declared is not None:
        keys = {declared.variant_key, *resolve_control_arm(declared).variant_keys}
        shown = realized if realized in keys else declared.variant_key
        arms.append(declared.model_copy(update={"label": names.get(shown, declared.label)}))
        covered |= keys
    for key, entry in entries.items():
        if key in covered:
            continue
        arms.append(
            CampaignArm(
                label=names.get(key, key[:12]),
                role="control" if declared is None and key == realized else "candidate",
                variant_key=key,
                levers=dict(entry.levers) if entry.levels_unavailable is None else None,
            )
        )
    return sorted(arms, key=lambda arm: arm.role != "control")


def label_of(bundle: CampaignBundle, variant_key: str) -> str:
    """The label the campaign bundle gives the arm a variant key names.

    :param bundle: The campaign bundle.
    :ptype bundle: CampaignBundle
    :param variant_key: The arm's variant.
    :ptype variant_key: str
    :return: Its label.
    :rtype: str
    """
    return bundle.label_of(variant_key)


def campaign_bundle(
    *bundles: AnalysisContextBundle,
    arms: Sequence[CampaignArm] | None = None,
    floors: Sequence[FloorStatus] = (),
    readings: Sequence[AspectReading] = (),
    pairs: Sequence[PairReading] = (),
    bars: Sequence[BarStatus] = (),
    question: str = "",
) -> CampaignBundle:
    """A campaign bundle over ``bundles``, one aspect each, in order.

    :param bundles: The aspects' bundles, all of one campaign.
    :ptype bundles: AnalysisContextBundle
    :param arms: The campaign's arms; read off the bundles when None.
    :ptype arms: Sequence[CampaignArm] | None
    :param floors: Code's floor statuses.
    :ptype floors: Sequence[FloorStatus]
    :param readings: Code's aspect readings.
    :ptype readings: Sequence[AspectReading]
    :param pairs: Code's pair readings (a choice).
    :ptype pairs: Sequence[PairReading]
    :param bars: Code's bar statuses.
    :ptype bars: Sequence[BarStatus]
    :param question: The campaign's question.
    :ptype question: str
    :return: The campaign bundle.
    :rtype: CampaignBundle
    """
    first = bundles[0]
    held = list(arms) if arms is not None else arms_of(bundles)
    control = next((arm for arm in held if arm.role == "control"), None)
    declared_floors = [
        Floor(aspect=bundle.aspect, clause=floor.clause)
        for bundle in bundles
        if bundle.declared is not None
        for floor in bundle.declared.floors
    ]
    return CampaignBundle(
        campaign_id=first.campaign_id,
        scope_id=first.scope_id,
        name=first.campaign_id,
        subject_id=first.subject_id,
        subject_kind=first.subject_kind,
        behavior=first.behavior,
        declaration=CampaignDeclaration(
            question=question,
            declared_at=None,
            declared_by=None,
            arms=[arm for arm in held if arm.role == "control"],
            aspects=[Aspect(label=bundle.aspect, template_id=bundle.template_id) for bundle in bundles],
            floors=declared_floors,
        ),
        arms=held,
        readings=CampaignReadings(
            campaign_id=first.campaign_id,
            alpha=0.05,
            control=control.variant_key if control is not None else None,
            control_label=control.label if control is not None else "",
            aspects=[bundle.aspect for bundle in bundles],
            readings=list(readings),
            floors=list(floors),
            kind=campaign_kind(held),
            pairs=list(pairs),
            bars=list(bars),
        ),
        aspects=list(bundles),
    )


def memo(
    bundle: CampaignBundle | AnalysisContextBundle,
    findings: Sequence[dict[str, Any]] = (),
    *,
    means: str = "",
    arm: str | None = None,
    reasons: Sequence[str] = (),
    rests_on: Sequence[int] | None = None,
    next: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """An authored memo over ``bundle``: ``findings``, one recommendation and one summary per aspect.

    :param bundle: The campaign bundle, or one aspect's bundle read as a campaign of it.
    :ptype bundle: CampaignBundle | AnalysisContextBundle
    :param findings: The findings, as authored.
    :ptype findings: Sequence[dict[str, Any]]
    :param means: What each aspect's summary says it means for the question.
    :ptype means: str
    :param arm: The arm recommended, by label; the control (else the first arm) when None.
    :ptype arm: str | None
    :param reasons: The recommendation's reasons.
    :ptype reasons: Sequence[str]
    :param rests_on: The findings it rests on; the first, where there is one, when None.
    :ptype rests_on: Sequence[int] | None
    :param next: The next steps, as authored.
    :ptype next: Sequence[dict[str, Any]]
    :return: The payload.
    :rtype: dict[str, Any]
    """
    campaign = bundle if isinstance(bundle, CampaignBundle) else campaign_bundle(bundle)
    control = campaign.control()
    chosen = arm if arm is not None else (control.label if control is not None else campaign.arms[0].label)
    return {
        "findings": list(findings),
        "recommendation": [{
            "arm": chosen,
            "reasons": list(reasons),
            "confidence": "low",
            "rests_on": list(rests_on) if rests_on is not None else ([0] if findings else []),
            "floors_overridden": [],
            "behind_on": [],
            "bars_missed": [],
        }],
        "aspect_summaries": [
            {"aspect": label, "what_changed": "", "how_sure": "", "means": means} for label in campaign.labels
        ],
        "next": list(next),
    }
