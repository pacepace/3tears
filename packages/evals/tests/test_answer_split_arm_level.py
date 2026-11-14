"""An arm answered by two candidate models is read per model, and nothing is read for the arm.

A campaign's cells hold one answering model each (``analysis.cells.split_on_served_model``), so an arm whose runs
under one rig were answered by two models holds two cells and two contrasts against the control. Every arm-level
reader took the first contrast it met for the arm's variant key and set its change, interval and verdict beside counts
pooled over both models. Held here, through the doors a host reads:

- the campaign's readings hold no row and no floor status for the split arm, and name the split (``answer_splits``);
- the campaign's page reads the aspect as not compared, with one line naming the models, and no verdict for the arm;
- the report input gives the arm no figure on that aspect and says why, never that it ran under several rigs;
- the per-model contrasts stay in the aspect's bundle;
- the same campaign answered by one model throughout reads as it always did, with nothing withheld.

**Mutations these catch:** an arm-level reader taking the first cell's contrast; the report input keeping the first
cell's figures under the arm's name; the split line dropped from the page; a split control still read against.
"""

from __future__ import annotations

from collections.abc import Sequence

from threetears.evals.analysis import assemble_context_bundle, build_code_only_report, campaign_report_markdown
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.campaign_bundle import CampaignBundle, build_campaign_bundle
from threetears.evals.analysis.campaign_readings import CampaignReadings, campaign_readings
from threetears.evals.analysis.report.campaign.page import build_campaign_report
from threetears.evals.analysis.report.input_build import report_input
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import CampaignArm, RateDrop
from threetears.evals.schema.models import EvalResult, EvalRun, RoleUsage
from packages.evals.tests.campaign_support import ASKED_AT, QUESTION
from packages.evals.tests.factories import fixture_variant_key, make_aspect_campaign, minimal_aspect_design
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.run_facts_support import ASPECT, CASES, result_of, run_of

_CONTROL, _ALIAS = "pinned", "latest"
_MARCH, _JUNE = "m-03", "m-06"


def _results(run: EvalRun, served: str, passing: int) -> list[EvalResult]:
    """The run's trials, the first ``passing`` of them passing, each answered by ``served``."""
    usage = [RoleUsage(role="candidate", model=run.candidate_model, served_model=served, cost_usd=0.002, call_count=1)]
    return [
        result_of(
            run,
            index,
            usage=usage,
            goal_state_outcomes=[{"expression": "reply.total == case.total", "passed": index < passing}],
        )
        for index in range(CASES)
    ]


class _Read:
    """One campaign of a control and an alias arm of two runs, read through every door."""

    def __init__(self, alias_served: Sequence[str], *, control_served: Sequence[str] = ("pinned-1",)) -> None:
        trials: list[tuple[EvalRun, list[EvalResult]]] = []
        for arm, served_by, passing in ((_CONTROL, control_served, 5), (_ALIAS, alias_served, 1)):
            for index, served in enumerate(served_by):
                run = run_of(arm).model_copy(update={"id": f"run-{arm}-{index}"})
                # The alias's second run passes everything, so the two answering models read far apart.
                trials.append((run, _results(run, served, CASES if arm == _ALIAS and index else passing)))
        runs = [run for run, _ in trials]
        self.arms = [
            CampaignArm(
                label=label,
                role="control" if label == _CONTROL else "candidate",
                variant_key=fixture_variant_key(label),
                **({"named_at": ASKED_AT} if label == _CONTROL else {}),
            )
            for label in (_CONTROL, _ALIAS)
        ]
        self.campaign: EvalCampaign = make_aspect_campaign(
            id="campaign-answer-split",
            label=ASPECT,
            run_ids=[run.id for run in runs],
            design=minimal_aspect_design(merit_axes=[]),
            question=QUESTION,
            asked_at=ASKED_AT,
            control=self.arms[0],
            margin=10.0,
            arms=self.arms,
            floors=[RateDrop(max_points=10)],
            scope_id=runs[0].scope_id,
            name="answer split",
            subject_id=runs[0].subject_snapshot.subject_id,
            subject_kind="s",
            created_at=ASKED_AT,
        )
        self.bundle: AnalysisContextBundle = assemble_context_bundle(
            self.campaign,
            storage=ToyhostStorage(runs, {run.id: held for run, held in trials}),
            profile=toyhost_profile(),
        )
        self.readings: CampaignReadings = campaign_readings(self.campaign, {ASPECT: self.bundle}, arms=self.arms)
        self.whole: CampaignBundle = build_campaign_bundle(
            self.campaign, [self.bundle], arms=self.arms, readings=self.readings
        )

    def page(self) -> str:
        aspect = build_code_only_report(
            self.bundle, assembled_at="2026-10-09T00:00:00+00:00", arm_labels=self.readings.arm_labels()
        )
        return campaign_report_markdown(
            build_campaign_report(self.campaign, self.readings, {ASPECT: aspect}, arms=self.arms)
        )

    def first_screen(self) -> str:
        return self.page().split("\n## ", 1)[0]


def test_the_readings_hold_no_row_and_no_floor_for_an_arm_answered_by_two_models_and_name_the_split() -> None:
    read = _Read([_MARCH, _JUNE])

    assert read.readings.readings == [] and read.readings.floors == []
    (split,) = read.readings.answer_splits
    assert (split.aspect, split.arm_label, split.cells) == (ASPECT, _ALIAS, 2)
    assert (split.served_models, split.unrecorded) == ([_MARCH, _JUNE], False)
    # The per-model contrasts are still in the aspect's bundle, one per cell, each over one run's cases.
    alias = fixture_variant_key(_ALIAS)
    contrasts = [c for c in read.bundle.floor_contrasts if c.contrast.variant_key == alias and c.reading == "trial"]
    assert len({c.contrast.apparatus_class_id for c in contrasts}) == 2
    assert sorted(round(c.contrast.mean or 0.0, 3) for c in contrasts) == [round(1 / CASES, 3), 1.0]


def test_the_page_gives_the_split_arm_no_verdict_and_says_it_is_read_per_model() -> None:
    read = _Read([_MARCH, _JUNE])

    first = read.first_screen()

    assert f"**On {ASPECT}, not compared.**" in first
    assert (
        f"{_ALIAS}'s results on {ASPECT} were answered by 2 models ({_MARCH} and {_JUNE}), so they are read per model: "
        f"no change, interval or verdict is given for {_ALIAS} there."
    ) in first
    for decided in ("better", "worse", "no worse than", "Adopt", "Keep", "floor"):
        assert decided not in first, first


def test_the_report_input_gives_the_split_arm_no_figure_and_says_why() -> None:
    read = _Read([_MARCH, _JUNE])

    built = report_input(read.whole)

    (note,) = [line for line in built.caveats.unreported if _ALIAS in line]
    assert "answered by more than one candidate model" in note and "no figure is given" in note
    assert "rigs" not in note
    (aspect,) = built.aspects
    assert aspect.readings == []
    assert built.measures, "the control's figures must be there, or this asserts nothing"
    for name, measure in built.measures.items():
        assert _ALIAS not in measure.per_arm, f"{name} gives the arm a figure"
        assert _CONTROL in measure.per_arm, "the control, answered by one model, keeps its figures"


def test_a_control_answered_by_two_models_leaves_no_candidate_read_against_it() -> None:
    read = _Read([_MARCH], control_served=["pinned-1", "pinned-2"])

    assert read.readings.readings == [] and read.readings.floors == []
    (split,) = read.readings.answer_splits
    assert split.arm_label == _CONTROL
    assert f"no candidate is read against {_CONTROL} there" in read.first_screen()


def test_an_arm_answered_by_one_model_throughout_is_read_as_one_arm_with_nothing_withheld() -> None:
    read = _Read([_MARCH, _MARCH])

    (row,) = read.readings.readings
    assert (row.arm_label, row.arm_trials, row.control_trials) == (_ALIAS, 2 * CASES, CASES)
    assert len(read.readings.floors) == 1
    assert read.readings.answer_splits == [] and "answer_splits" not in read.readings.to_dict()
    built = report_input(read.whole)
    assert not [line for line in built.caveats.unreported if "answered by" in line]
    assert "read per model" not in read.page()
