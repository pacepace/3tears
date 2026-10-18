"""A campaign bundle built from aspect bundles a test already holds, with no campaign or store behind it.

The generator reads a :class:`~threetears.evals.analysis.campaign_bundle.CampaignBundle`, which a service assembles
from a stored campaign. A test that holds one aspect's assembled bundle (and perhaps rewrote part of it) builds the
campaign bundle around it here: the arms are read off the bundles' variant indexes, the control as the realized design
matched it, every one labelled as the arm table names it; code's readings carry whatever floors the test hands in.
"""

from __future__ import annotations

from collections.abc import Sequence

from threetears.evals.analysis.arms import arm_names
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.campaign_readings import AspectReading, CampaignReadings, FloorStatus
from threetears.evals.kernel.campaign import CampaignDeclaration
from threetears.evals.kernel.declaration import Aspect, CampaignArm, Floor

__all__ = ["arms_of", "campaign_bundle", "label_of"]


def arms_of(bundles: Sequence[AnalysisContextBundle]) -> list[CampaignArm]:
    """Every arm the bundles index, the control first, each labelled as the arm table names it.

    :param bundles: The aspects' bundles.
    :ptype bundles: Sequence[AnalysisContextBundle]
    :return: The arms.
    :rtype: list[CampaignArm]
    """
    entries = {entry.variant_key: entry for bundle in bundles for entry in bundle.variant_index}
    names = arm_names(list(entries.values()))
    controls = {bundle.design.control_arm.variant_key for bundle in bundles if bundle.design.control_arm is not None}
    ordered = sorted(entries, key=lambda key: key not in controls)
    return [
        CampaignArm(
            label=names.get(key, key[:12]),
            role="control" if key in controls else "candidate",
            variant_key=key,
            levers=dict(entries[key].levers) if entries[key].levels_unavailable is None else None,
        )
        for key in ordered
    ]


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
        ),
        aspects=list(bundles),
    )
