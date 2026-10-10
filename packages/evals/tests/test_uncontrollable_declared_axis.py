"""A stored campaign axis on an input the host cannot vary says why its row is unswept (#675).

``HostProfile.controllable`` refuses an axis declared on an apparatus or label input, but only the
authoring gate asked it. A design stored before the gate, or past it, still assembled, and the axis
showed only as a bare ``unswept`` coverage row: an apparatus input never enters the variant key, so the
run that moved it reads as a repeat of the control. Nothing said "this axis can never be an arm", so a
reader who did not author the campaign could not tell it from "this sweep did not happen".

The bundle now asks the same question of every declared axis and carries the host's reason on the row
(``cannot_be_an_arm``), and the code-only report says it once per such axis.
"""

from __future__ import annotations

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle, build_code_only_report
from threetears.evals.analysis.report import DisclosureBlock
from threetears.evals.contracts import SweptAxis
from threetears.evals.contracts.host import SweepableValue
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

#: An apparatus input of the toy host — the OCR engine is the rig, not a knob.
_APPARATUS = "ocr_engine_version"


def _bundle(*, with_apparatus_axis: bool) -> AnalysisContextBundle:
    """The toy campaign, as stored — optionally with a second declared axis on the apparatus input."""
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile)
    if with_apparatus_axis:
        design = campaign.declared_design
        assert design is not None
        axis = SweptAxis(
            axis_id=_APPARATUS,
            values=[SweepableValue.of(v, display=v) for v in ("tess-5.3.1", "tess-5.4.0")],
            rationale="does the newer OCR engine read the totals line more often",
        )
        # Written past the authoring gate, as a design stored before it was.
        campaign = campaign.model_copy(
            update={"declared_design": design.model_copy(update={"axes": [*design.axes, axis]})}
        )
    return assemble_context_bundle(campaign, storage=storage, profile=profile)


def _report_texts(bundle: AnalysisContextBundle) -> list[str]:
    report = build_code_only_report(
        bundle, measures=toyhost_profile().measures, assembled_at="2026-10-10T00:00:00+00:00"
    )
    return [block.text for block in report.blocks if isinstance(block, DisclosureBlock)]


def _bundle_reason(bundle: AnalysisContextBundle) -> str:
    (row,) = [row for row in bundle.coverage if row.name == _APPARATUS]
    assert row.cannot_be_an_arm is not None
    return row.cannot_be_an_arm.rstrip(".")


def test_an_apparatus_axis_row_names_the_reason_it_can_never_be_an_arm() -> None:
    bundle = _bundle(with_apparatus_axis=True)

    (row,) = [row for row in bundle.coverage if row.name == _APPARATUS]

    assert row.status == "unswept"
    expected = toyhost_profile().controllable(_APPARATUS)
    assert expected.state != "covered"
    assert row.cannot_be_an_arm == expected.reason
    assert "apparatus" in row.cannot_be_an_arm


def test_the_code_only_report_states_the_cause() -> None:
    bundle = _bundle(with_apparatus_axis=True)

    texts = _report_texts(bundle)

    (sentence,) = [text for text in texts if text.startswith(f"Declared axis {_APPARATUS} reads unswept")]
    assert _bundle_reason(bundle) in sentence


def test_a_campaign_whose_axes_are_all_levers_discloses_nothing() -> None:
    bundle = _bundle(with_apparatus_axis=False)

    assert bundle.coverage, "the toy campaign's lever axis has a row"
    assert all(row.cannot_be_an_arm is None for row in bundle.coverage)
    texts = _report_texts(bundle)
    assert not any(text.startswith("Declared axis ") for text in texts)
