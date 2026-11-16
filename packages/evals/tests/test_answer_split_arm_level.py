"""An arm answered by two candidate models is read per model, and nothing is read for the arm.

A campaign's cells hold one answering model each (``analysis.cells.split_on_served_model``), so an arm whose runs
under one rig were answered by two models holds two cells and two contrasts against the control. Every arm-level
reader took the first contrast it met for the arm's variant key and set its change, interval and verdict beside counts
pooled over both models. Held here, through the doors a host reads:

- the campaign's readings hold no row and no floor status for the split arm, and name the split (``answer_splits``);
- the campaign's page reads the aspect as not compared, with one line naming the models, and no verdict for the arm;
- the report input gives the arm no figure on that aspect and says why, never that it ran under several rigs;
- the per-model contrasts stay in the aspect's bundle;
- the aspect's page gives the split arm no pass count, failure count, cost or outcome bar, and its verdict line says
  it is not compared, and why; the report input's scoreboard row for it carries no passes, failures or cost;
- the same campaign answered by one model throughout reads as it always did, with nothing withheld.

**Mutations these catch:** an arm-level reader taking the first cell's contrast; the report input keeping the first
cell's figures under the arm's name; the split line dropped from the page; a split control still read against.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from threetears.evals.analysis import (
    assemble_context_bundle,
    build_code_only_report,
    campaign_report_markdown,
    report_markdown,
)
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.campaign_bundle import CampaignBundle, build_campaign_bundle
from threetears.evals.analysis.campaign_readings import CampaignReadings, campaign_readings
from threetears.evals.analysis.analyst.snapshot import analyst_input
from threetears.evals.analysis.errors import UnstatedFigure
from threetears.evals.analysis.report.campaign.page import build_campaign_report
from threetears.evals.analysis.report.figure_keys import resolve
from threetears.evals.analysis.report.model import ChartBlock, Report
from threetears.evals.analysis.caveats import caveat_list
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_input_build import report_input
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


def _results(run: EvalRun, served: str, passing: int, cost: float = 0.002, step: float = 0.0) -> list[EvalResult]:
    """The run's trials, the first ``passing`` of them passing, each answered by ``served``; trial ``i`` costs
    ``cost + i * step``."""
    return [
        result_of(
            run,
            index,
            usage=[
                RoleUsage(
                    role="candidate",
                    model=run.candidate_model,
                    served_model=served,
                    cost_usd=cost + index * step,
                    call_count=1,
                )
            ],
            goal_state_outcomes=[{"expression": "reply.total == case.total", "passed": index < passing}],
        )
        for index in range(CASES)
    ]


class _Read:
    """One campaign of a control and an alias arm of two runs, read through every door."""

    def __init__(
        self,
        alias_served: Sequence[str],
        *,
        control_served: Sequence[str] = ("pinned-1",),
        alias_costs: Sequence[float] | None = None,
    ) -> None:
        """``alias_costs``, one per alias run, also has every alias run pass as often as the control: the arms then
        differ on cost alone."""
        trials: list[tuple[EvalRun, list[EvalResult]]] = []
        for arm, served_by, passing in ((_CONTROL, control_served, 5), (_ALIAS, alias_served, 1)):
            for index, served in enumerate(served_by):
                run = run_of(arm).model_copy(update={"id": f"run-{arm}-{index}"})
                if arm == _ALIAS and alias_costs is not None:
                    # each trial's cost a little apart, so a cell's cost has a spread a contrast can read
                    trials.append((run, _results(run, served, 5, alias_costs[index], step=0.0001)))
                    continue
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

    def aspect(self) -> Report:
        return build_code_only_report(
            self.bundle, assembled_at="2026-10-09T00:00:00+00:00", arm_labels=self.readings.arm_labels()
        )

    def aspect_first_screen(self) -> str:
        """The aspect's own page, to its first section: the verdict line and the scoreboard."""
        return report_markdown(self.aspect()).split("\n## ", 1)[0]

    def bars(self) -> list[str]:
        """The arms the aspect's outcome bar draws, as the page names them."""
        return [bar.arm for block in self.aspect().blocks if isinstance(block, ChartBlock) for bar in block.bars]


def _split_caveats(built: ReportInput, *, stored: bool) -> list[tuple[object, ...]]:
    """The withheld-by-split caveats of a built input as the builder raised them with its bundle, or, ``stored``, of
    the same input written out without its list and read back, which raises them from its own parts alone."""
    if stored:
        written = built.model_copy(update={"caveat_list": None, "first_screen": None}).model_dump(
            mode="json", by_alias=True
        )
        entries = caveat_list(ReportInput.model_validate(written))
    else:
        assert built.caveat_list is not None
        entries = built.caveat_list
    return [
        (entry.state, entry.reason, entry.coordinate, entry.arms, entry.readings)
        for entry in entries
        if entry.reason == "split"
    ]


def _scoreboard_row(first_screen: str, arm: str) -> list[str]:
    """One arm's scoreboard row on an aspect's first screen, cell by cell."""
    (row,) = [line for line in first_screen.splitlines() if line.startswith(("| " + arm + " |", "| ▸ " + arm + " |"))]
    return [cell.strip() for cell in row.strip("|").split("|")]


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
    page = read.page()
    assert f"{ASPECT} is not compared: {_ALIAS}'s results there were answered by more than one model" in page
    assert "holds the runs of one side only" not in page, "both sides ran: the cause is the split"


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
    first = read.first_screen()
    assert (
        f"{_CONTROL} is the control, and its results on {ASPECT} were answered by 2 models (pinned-1 and pinned-2), so "
        f"they are read per model: the whole campaign is not compared, since no candidate is read against {_CONTROL} "
        "there."
    ) in first
    assert f"**On {ASPECT}, not compared.**" in first
    # The scorecard says why the aspect has no row: never that one side did not run.
    page = read.page()
    assert f"{ASPECT} is not compared: {_CONTROL}'s results there were answered by more than one model" in page
    assert "holds the runs of one side only" not in page
    # No contrast is read for the campaign: the candidate, answered by one model, has no figure against the control.
    assert read.readings.aggregate is None


def test_an_arm_answered_by_one_model_throughout_is_read_as_one_arm_with_nothing_withheld() -> None:
    read = _Read([_MARCH, _MARCH])

    (row,) = read.readings.readings
    assert (row.arm_label, row.arm_trials, row.control_trials) == (_ALIAS, 2 * CASES, CASES)
    assert len(read.readings.floors) == 1
    assert read.readings.answer_splits == [] and "answer_splits" not in read.readings.to_dict()
    built = report_input(read.whole)
    assert not [line for line in built.caveats.unreported if "answered by" in line]
    assert "read per model" not in read.page()


def test_a_one_arm_campaign_whose_arm_was_answered_by_two_models_reads_no_pooled_rate_and_says_why() -> None:
    """A measurement reads each arm's own figures, and a split arm has none: never a pass rate over both models."""
    trials = []
    for index, served in enumerate((_MARCH, _JUNE)):
        run = run_of(_ALIAS).model_copy(update={"id": f"run-solo-{index}"})
        trials.append((run, _results(run, served, CASES if index else 1)))
    runs = [run for run, _ in trials]
    arms = [CampaignArm(label=_ALIAS, role="candidate", variant_key=fixture_variant_key(_ALIAS))]
    campaign = make_aspect_campaign(
        id="campaign-answer-split-solo",
        label=ASPECT,
        run_ids=[run.id for run in runs],
        design=minimal_aspect_design(merit_axes=[]),
        question=QUESTION,
        asked_at=ASKED_AT,
        arms=arms,
        scope_id=runs[0].scope_id,
        name="answer split solo",
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="s",
        created_at=ASKED_AT,
    )
    bundle = assemble_context_bundle(
        campaign, storage=ToyhostStorage(runs, {run.id: held for run, held in trials}), profile=toyhost_profile()
    )
    assert len([cell for cell in bundle.cells if cell.rig_class_id is not None]) == 2

    readings = campaign_readings(campaign, {ASPECT: bundle}, arms=arms)

    assert readings.kind == "measurement"
    assert readings.measures == [], "no pass rate pooled over both models, no interval, no cell's figure as the arm's"
    (split,) = readings.answer_splits
    assert (split.arm_label, split.served_models) == (_ALIAS, [_MARCH, _JUNE])
    built = report_input(build_campaign_bundle(campaign, [bundle], arms=arms, readings=readings))
    (note,) = [line for line in built.caveats.unreported if _ALIAS in line]
    assert "answered by more than one candidate model" in note and "no figure is given" in note
    for name, measure in built.measures.items():
        assert _ALIAS not in measure.per_arm, f"{name} gives the arm a figure the note says is not given"
    aspect = build_code_only_report(bundle, assembled_at="2026-10-09T00:00:00+00:00", arm_labels=readings.arm_labels())
    first = campaign_report_markdown(build_campaign_report(campaign, readings, {ASPECT: aspect}, arms=arms)).split(
        "\n## ", 1
    )[0]
    assert (
        f"{_ALIAS}'s results on {ASPECT} were answered by 2 models ({_MARCH} and {_JUNE}), so they are read per model"
    ) in first
    own = report_markdown(aspect).split("\n## ", 1)[0]
    assert f"**{_ALIAS} has no one pass rate: its trials were answered by more than one model" in own
    assert "passed 7 of 12" not in own and "Cases that failed a trial" not in own
    assert _scoreboard_row(own, _ALIAS)[3] == "— (read per model)"
    assert not [block for block in aspect.blocks if isinstance(block, ChartBlock) and block.chart == "outcome_bar"]


def test_the_aspect_page_gives_an_arm_answered_by_two_models_no_count_no_cost_no_verdict_and_no_bar() -> None:
    read = _Read([_MARCH, _JUNE])
    first = read.aspect_first_screen()

    assert (
        f"{_ALIAS} is not compared with {_CONTROL}: its trials were answered by more than one model, or by one no "
        "response named, so they are read per model."
    ) in first
    assert "cannot tell" not in first and "more often" not in first, "a split arm is neither undecided nor ranked"
    # Ran, usable, passed, failed by check, failed by judge, cost: what ran stays; what two models answered does not.
    assert _scoreboard_row(first, _ALIAS)[1:] == ["12", "12", "— (read per model)", "—", "—", "—"]
    assert _scoreboard_row(first, _CONTROL)[3] == "5", "the control, answered by one model, keeps its count"
    assert read.bars() == [_CONTROL]


def test_an_arm_answered_by_two_models_that_cost_differently_is_not_compared_and_never_undecided() -> None:
    """Each cell's cost reads apart from the control's while no pass rate does: the line never says the evidence cannot
    tell the split arm from the control, and says it is not compared, and why."""
    read = _Read([_MARCH, _JUNE], alias_costs=[0.004, 0.008])
    first = read.aspect_first_screen()
    separated = {
        comparison.name
        for family in read.bundle.multiple_comparisons.families
        for comparison in family.comparisons
        if comparison.verdict != "not_separated"
    }
    assert separated == {"production_replicating_cost"}, "only cost reads apart, or this asserts nothing"

    assert "cannot tell" not in first, first
    assert (
        f"{_ALIAS} is not compared with {_CONTROL}: its trials were answered by more than one model, or by one no "
        "response named, so they are read per model."
    ) in first


def test_the_aspect_page_says_no_candidate_is_compared_with_a_control_answered_by_two_models() -> None:
    first = _Read([_MARCH], control_served=["pinned-1", "pinned-2"]).aspect_first_screen()

    # (The candidate's label is the readings', which name no arm a split control leaves unread: issue draft.)
    assert f" is not compared with {_CONTROL}: {_CONTROL}'s trials were answered by more than one model" in first
    assert "cannot tell" not in first
    assert _scoreboard_row(first, _CONTROL)[3] == "— (read per model)"


def test_the_report_input_scoreboard_row_of_an_arm_answered_by_two_models_has_no_passes_and_no_cost() -> None:
    built = report_input(_Read([_MARCH, _JUNE]).whole)

    split, control = built.scoreboard[_ALIAS], built.scoreboard[_CONTROL]
    assert (split.ran, split.usable, split.passed, split.failed, split.reply_cost_usd) == (
        2 * CASES,
        2 * CASES,
        None,
        {},
        None,
    )
    assert split.test_cost_usd is not None, "what the arm's trials cost in all is spend, not a reading"
    assert (control.passed, control.reply_cost_usd) == (5, 0.002)
    (note,) = [line for line in built.caveats.unreported if _ALIAS in line]
    assert "answered by more than one candidate model" in note


def test_an_arm_answered_by_one_model_throughout_withholds_nothing_on_its_aspect_page() -> None:
    read = _Read([_MARCH, _MARCH])
    first = read.aspect_first_screen()

    assert "not compared" not in first and "read per" not in first
    assert _scoreboard_row(first, _ALIAS)[3] == str(1 + CASES)
    assert read.bars() == [_CONTROL, _ALIAS]
    assert report_input(read.whole).scoreboard[_ALIAS].passed == 1 + CASES


def test_a_written_page_cites_no_failure_count_for_the_split_arm_and_the_controls_count_resolves() -> None:
    """The report input withholds the split arm's failures as an empty count, which no key reads as 0 failed."""
    built = report_input(_Read([_MARCH, _JUNE]).whole)

    held = built.scoreboard[_ALIAS].state
    assert held is not None and (held.state, held.reason, held.coordinate) == ("withheld", "split", "served_model")
    assert built.scoreboard[_CONTROL].state is None
    for key in (f"scoreboard.{_ALIAS}.failed", f"aspects.{ASPECT}.scoreboard.{_ALIAS}.failed"):
        with pytest.raises(UnstatedFigure):
            resolve(key, built)
    for key in (f"scoreboard.{_CONTROL}.failed", f"aspects.{ASPECT}.scoreboard.{_CONTROL}.failed"):
        assert resolve(key, built).value == CASES - 5


def test_an_input_that_states_no_passes_and_splits_no_arm_reads_its_failures_as_it_did() -> None:
    """A row's failures are refused because its typed state withholds them, never because ``passed`` is missing: an
    input that never stated passes, and keeps no arm apart (the report inputs written before the cells split an arm),
    reads its failures as it always did."""
    built = analyst_input(_Read([_MARCH]).whole)  # one answering model throughout: no arm is split
    assert all(row.state is None for row in built.scoreboard.values())
    unstated = built.model_copy(
        update={"scoreboard": {arm: row.model_copy(update={"passed": None}) for arm, row in built.scoreboard.items()}}
    )

    for arm, row in built.scoreboard.items():
        failed = sum(row.failed.values())
        assert failed > 0, "a held count, so reading it asserts something"
        for key in (f"scoreboard.{arm}.failed", f"aspects.{ASPECT}.scoreboard.{arm}.failed"):
            assert resolve(key, unstated).value == failed
        assert resolve(f"count.failures.{ASPECT}.{arm}", unstated).value == failed


def test_the_failures_key_refuses_the_split_arm_on_its_aspect_and_the_campaign_total_still_resolves() -> None:
    """`count.failures.<aspect>.<arm>` would read the input's failed trials for the split arm, which pool its cells:
    refused as `scoreboard.<arm>.failed` is; the control's resolves, and so does the campaign's total."""
    built = analyst_input(_Read([_MARCH, _JUNE]).whole)  # what a snapshot freezes for a writer's page

    assert built.failed_trials is not None
    pooled = [row for row in built.failed_trials if (row.aspect, row.arm) == (ASPECT, _ALIAS)]
    assert sum(row.failed for row in pooled) > 0, "a pooled count is held, so its refusal asserts something"
    for key in [f"count.failures.{ASPECT}.{_ALIAS}"] + [
        f"count.failures.{ASPECT}.{_ALIAS}.{row.stratum}" for row in pooled if row.stratum is not None
    ]:
        with pytest.raises(UnstatedFigure):
            resolve(key, built)
    assert resolve(f"count.failures.{ASPECT}.{_CONTROL}", built).value == CASES - 5
    assert resolve("count.failures", built).value == sum(row.failed for row in built.failed_trials)


_TONE = "tone"


def _across_two_aspects(read: _Read) -> CampaignReadings:
    """The campaign's readings with a second aspect, on which each arm was answered by one model throughout, beside its own."""
    trials: list[tuple[EvalRun, list[EvalResult]]] = []
    for arm, passing in ((_CONTROL, 5), (_ALIAS, 4)):
        run = run_of(arm).model_copy(update={"id": f"run-{_TONE}-{arm}"})
        trials.append((run, _results(run, _MARCH, passing)))
    runs = [run for run, _ in trials]
    tone = make_aspect_campaign(
        id=read.campaign.id,
        label=_TONE,
        run_ids=[run.id for run in runs],
        design=minimal_aspect_design(merit_axes=[]),
        question=QUESTION,
        asked_at=ASKED_AT,
        control=read.arms[0],
        margin=10.0,
        arms=read.arms,
        floors=[RateDrop(max_points=10)],
        scope_id=runs[0].scope_id,
        name=read.campaign.name,
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="s",
        created_at=ASKED_AT,
    )
    bundle = assemble_context_bundle(
        tone, storage=ToyhostStorage(runs, {run.id: held for run, held in trials}), profile=toyhost_profile()
    )
    campaign = read.campaign.model_copy(
        update={"aspects": [*read.campaign.aspects, *tone.aspects], "floors": [*read.campaign.floors, *tone.floors]}
    )
    return campaign_readings(campaign, {ASPECT: read.bundle, _TONE: bundle}, arms=read.arms)


def test_the_equal_weight_figures_withhold_the_pass_rate_of_an_arm_answered_by_two_models_on_an_aspect() -> None:
    """Its count on that aspect pools what its cells keep apart: the figure across aspects is withheld, naming it."""
    aggregate = _across_two_aspects(_Read([_MARCH, _JUNE])).aggregate

    assert aggregate is not None, "the arms are compared on the other aspect, or this asserts nothing"
    held = {figures.arm_label: figures for figures in aggregate.arms}
    assert (held[_ALIAS].pass_rate.value, held[_ALIAS].pass_rate.missing) == (None, [ASPECT])
    assert held[_CONTROL].pass_rate.value is not None and held[_CONTROL].pass_rate.missing == []


def test_the_equal_weight_figures_withhold_the_pass_rate_of_a_control_answered_by_two_models_on_an_aspect() -> None:
    aggregate = _across_two_aspects(_Read([_MARCH], control_served=["pinned-1", "pinned-2"])).aggregate

    assert aggregate is not None, "the arms are compared on the other aspect, or this asserts nothing"
    held = {figures.arm_label: figures for figures in aggregate.arms}
    assert (held[_CONTROL].pass_rate.value, held[_CONTROL].pass_rate.missing) == (None, [ASPECT])
    assert held[_ALIAS].pass_rate.value is not None and held[_ALIAS].pass_rate.missing == []


def test_the_figures_across_aspects_never_call_an_arm_answered_by_two_models_dearer_on_that_aspect() -> None:
    """Its cost per reply there pools both models': no aspect is named where it is cheaper or dearer than the control."""
    aggregate = _across_two_aspects(_Read([_MARCH, _JUNE], alias_costs=[0.004, 0.008])).aggregate

    assert aggregate is not None
    held = {figures.arm_label: figures for figures in aggregate.arms}
    assert (held[_ALIAS].cheaper_on, held[_ALIAS].dearer_on) == ([], [])


@pytest.mark.parametrize("stored", [False, True], ids=["built", "stored"])
def test_an_arm_answered_by_two_models_raises_one_split_caveat_on_the_served_model_from_its_scoreboard_row(
    stored: bool,
) -> None:
    """The withheld-by-split caveat is raised from the scoreboard row's typed state, so a stored input with no bundle
    raises it as the built one does, once."""
    built = report_input(_Read([_MARCH, _JUNE]).whole)

    assert _split_caveats(built, stored=stored) == [
        ("withheld", "split", "served_model", [_ALIAS], [f"{ASPECT}/passed"])
    ]


@pytest.mark.parametrize("stored", [False, True], ids=["built", "stored"])
def test_an_arm_answered_by_one_model_raises_no_split_caveat(stored: bool) -> None:
    built = report_input(_Read([_MARCH, _MARCH]).whole)

    assert all(row.state is None for row in built.scoreboard.values())
    assert _split_caveats(built, stored=stored) == []
