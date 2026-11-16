"""An arm read under two definitions of a host measure is read per definition, and nothing is read for the arm (#1187).

The mirror of ``test_answer_split_arm_level.py`` for the other coordinate the package reads. What a host measure
counted is the instrument's, so runs read under two definitions are under two rigs, and an arm holding them holds two
cells. Every arm-level reader asks one question first (``analysis.cells.unread_arms``), so none of them pools the
arm's counts over both definitions. Held here, through the doors a host reads:

- the campaign's readings hold no row and no floor for the split arm, and name the split with its coordinate;
- the campaign's page reads the aspect as not compared and says the arm's values are not one quantity;
- the report input gives the arm no figure on that aspect and says why;
- a one-arm campaign reads no pass rate pooled over both definitions;
- two one-run arms read under different definitions, or under none recorded, share no test setup: no floor, no row;
- the aspect's page gives a split arm no pass count, no verdict and no outcome bar, and says why; it never compares
  two arms that share no setup, at any case count; the report input's scoreboard row for a split arm has no passes;
- the same campaign read under one definition throughout reads as it always did, with nothing withheld;
- a frontier whose contestants were each read under one definition, but not the same one, says once, in its text and
  on the aspect page's method, that dominance and the cheapest pick are still tested between them, with no kept-out
  cell to bring the sentence; one read under one definition throughout does not.

**Mutations these catch:** an arm-level reader asking only about the served model; a floor or rate between arms that
share no rig; the report input keeping one cell's figures under the arm's name.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from threetears.evals.analysis import (
    assemble_context_bundle,
    build_code_only_report,
    campaign_report_markdown,
    frontier_text,
    report_markdown,
)
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis import lens_text as lens_text_module
from threetears.evals.analysis.lenses import frontier as frontier_module
from threetears.evals.analysis.report.aspect import method as method_module
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
from packages.evals.tests.fixtures.toyhost.kind import FIELD_ACCURACY
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.run_facts_support import ASPECT, CASES, result_of, run_of

_CONTROL, _CANDIDATE = "pinned", "next"
#: Two definitions of one host measure: the digest a host's registry gives it before and after a revision.
_BEFORE, _AFTER = {FIELD_ACCURACY: "a" * 64}, {FIELD_ACCURACY: "b" * 64}


def _results(run: EvalRun, passing: int, cases: int = CASES) -> list[EvalResult]:
    """The run's trials, the first ``passing`` of them passing, each answered by one named model and each holding a
    value of the host measure."""
    usage = [RoleUsage(role="candidate", model=run.candidate_model, served_model="m-1", cost_usd=0.002, call_count=1)]
    return [
        result_of(
            run,
            index,
            usage=usage,
            host_measures={FIELD_ACCURACY: 0.8},
            goal_state_outcomes=[{"expression": "reply.total == case.total", "passed": index < passing}],
        )
        for index in range(cases)
    ]


def _arms(labels: Sequence[str]) -> list[CampaignArm]:
    return [
        CampaignArm(
            label=label,
            role="control" if index == 0 else "candidate",
            variant_key=fixture_variant_key(label),
            **({"named_at": ASKED_AT} if index == 0 else {}),
        )
        for index, label in enumerate(labels)
    ]


def _campaign(runs: Sequence[EvalRun], arms: list[CampaignArm], name: str) -> EvalCampaign:
    return make_aspect_campaign(
        id=f"campaign-{name}",
        label=ASPECT,
        run_ids=[run.id for run in runs],
        design=minimal_aspect_design(merit_axes=[]),
        question=QUESTION,
        asked_at=ASKED_AT,
        control=arms[0],
        margin=10.0,
        arms=arms,
        floors=[RateDrop(max_points=10)] if len(arms) > 1 else [],
        scope_id=runs[0].scope_id,
        name=name,
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="s",
        created_at=ASKED_AT,
    )


class _Read:
    """One campaign: each arm's runs, each run recording the definitions given (None: recorded none)."""

    def __init__(self, definitions_by_arm: dict[str, Sequence[dict[str, str] | None]], *, cases: int = CASES) -> None:
        trials: list[tuple[EvalRun, list[EvalResult]]] = []
        for arm, each in definitions_by_arm.items():
            for index, definitions in enumerate(each):
                run = run_of(arm, measure_definitions=definitions).model_copy(update={"id": f"run-{arm}-{index}"})
                # The candidate's second run passes everything, so a rate pooled over both definitions reads apart.
                trials.append((run, _results(run, cases if arm == _CANDIDATE and index else 1, cases)))
        runs = [run for run, _ in trials]
        self.arms = _arms(list(definitions_by_arm))
        self.campaign = _campaign(runs, self.arms, "definition-split")
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


def test_an_arm_read_under_two_definitions_has_no_row_and_no_floor_and_the_split_names_its_coordinate() -> None:
    read = _Read({_CONTROL: [_AFTER], _CANDIDATE: [_BEFORE, _AFTER]})

    candidate = fixture_variant_key(_CANDIDATE)
    split_cells = [cell for cell in read.bundle.cells if cell.variant_key == candidate]
    assert len(split_cells) == 2 and all(cell.split_on == ["measure_definition"] for cell in split_cells)
    assert read.readings.readings == [] and read.readings.floors == []
    (split,) = read.readings.answer_splits
    assert (split.aspect, split.arm_label, split.cells, split.coordinate) == (
        ASPECT,
        _CANDIDATE,
        2,
        "measure_definition",
    )
    assert split.served_models == [], "the served model is not what splits it"


def test_the_page_says_the_split_arm_was_read_under_two_definitions_and_gives_it_no_verdict() -> None:
    page = _Read({_CONTROL: [_AFTER], _CANDIDATE: [_BEFORE, _AFTER]}).page()

    first = page.split("\n## ", 1)[0]
    assert f"**On {ASPECT}, not compared.**" in first
    assert f"{_CANDIDATE}'s runs on {ASPECT} were read under more than one definition of a host measure" in first
    assert f"no change, interval or verdict is given for {_CANDIDATE} there" in first
    assert "runs_attest, coordinate measure_definition" in first
    for decided in ("better", "worse", "no worse than", "Adopt", "Keep"):
        assert decided not in first, first
    assert f"{ASPECT} is not compared: {_CANDIDATE}'s runs there were not read under one recorded definition" in page
    assert "answered by more than one model" not in page, "the coordinate named is the one that split it"


def test_the_report_input_gives_the_split_arm_no_figure_and_says_why() -> None:
    read = _Read({_CONTROL: [_AFTER], _CANDIDATE: [_BEFORE, _AFTER]})

    built = report_input(read.whole)

    (note,) = [line for line in built.caveats.unreported if _CANDIDATE in line]
    assert "more than one definition of a host measure" in note and "no figure is given" in note
    (aspect,) = built.aspects
    assert aspect.readings == []
    assert built.measures, "the control's figures must be there, or this asserts nothing"
    for name, measure in built.measures.items():
        assert _CANDIDATE not in measure.per_arm, f"{name} gives the arm a figure"
        assert _CONTROL in measure.per_arm, "the control, read under one definition, keeps its figures"


def test_a_one_arm_campaign_read_under_two_definitions_reads_no_pooled_rate() -> None:
    read = _Read({_CANDIDATE: [_BEFORE, _AFTER]})

    assert read.readings.kind == "measurement"
    assert read.readings.measures == [], "no pass rate pooled over both definitions, and no one cell's as the arm's"
    (split,) = read.readings.answer_splits
    assert (split.arm_label, split.coordinate) == (_CANDIDATE, "measure_definition")


def test_two_one_run_arms_under_different_definitions_share_no_setup_and_read_no_floor_or_row() -> None:
    read = _Read({_CONTROL: [_BEFORE], _CANDIDATE: [_AFTER]})

    assert read.readings.readings == [] and read.readings.floors == []
    (held,) = read.readings.answer_splits
    assert (held.arm_label, held.cells, held.coordinate, held.unrecorded) == (
        _CANDIDATE,
        1,
        "measure_definition",
        False,
    )


def test_two_one_run_arms_whose_definitions_went_unrecorded_share_no_setup_either() -> None:
    """The unattested shape of every run stored before runs recorded definitions: a silence matches nothing."""
    read = _Read({_CONTROL: [None], _CANDIDATE: [None]})

    assert read.readings.readings == [] and read.readings.floors == []
    (held,) = read.readings.answer_splits
    assert (held.arm_label, held.coordinate, held.unrecorded) == (_CANDIDATE, "measure_definition", True)
    assert "under no recorded definition of a host measure" in read.page()


def test_a_campaign_read_under_one_definition_throughout_reads_as_it_always_did() -> None:
    read = _Read({_CONTROL: [_AFTER], _CANDIDATE: [_AFTER, _AFTER]})

    (row,) = read.readings.readings
    assert (row.arm_label, row.arm_trials, row.control_trials) == (_CANDIDATE, 2 * CASES, CASES)
    assert len(read.readings.floors) == 1
    assert read.readings.answer_splits == []
    assert all(cell.split_on == [] for cell in read.bundle.cells)
    assert "definition of a host measure" not in read.page()


#: The sentence a frontier says where its contestants' definitions differ (copied, not bound: the frontier's own words).
_DOMINANCE_POOLS = (
    "Dominance and the cheapest pick are still tested between contestants' headline cells whichever definitions of "
    "the host's measures read them."
)


@pytest.mark.parametrize(
    "definitions",
    [{_CONTROL: [_BEFORE], _CANDIDATE: [_AFTER]}, {_CONTROL: [None], _CANDIDATE: [None]}],
    ids=["two-definitions", "none-recorded"],
)
def test_a_frontier_whose_contestants_were_read_under_different_definitions_says_dominance_still_reads_across_them(
    definitions: dict[str, list[dict[str, str] | None]],
) -> None:
    read = _Read(definitions)
    frontier = read.bundle.frontier

    (subject,) = frontier.subjects
    assert len(subject.points) == 2, "two contestants, or there is nothing to dominate"
    assert all(len(point.pass_hat_k_cells) == 1 for point in subject.points), "no kept-out cell brings the sentence"
    text = frontier_text(frontier)
    assert text.count(_DOMINANCE_POOLS) == 1, text
    assert report_markdown(read.aspect()).count(_DOMINANCE_POOLS) == 1, "the aspect page's method says it too"


def test_a_frontier_read_under_one_definition_throughout_says_nothing_of_definitions() -> None:
    read = _Read({_CONTROL: [_AFTER], _CANDIDATE: [_AFTER, _AFTER]})
    frontier = read.bundle.frontier

    assert frontier.definitions_apart is False
    assert "definitions_apart" not in frontier.model_dump(), "it serializes, and digests, as it did"
    # Its candidate's two runs recorded no measurement context, so the cell sentence (which ends with the dominance
    # sentence) is said as it always was; what a frontier says of differing definitions is not.
    said = "Contestants here were read under different definitions of the host's measures"
    assert said not in frontier_text(frontier)
    assert said not in report_markdown(read.aspect())


def test_rewording_what_a_frontier_says_of_differing_definitions_moves_no_stored_bundle_and_no_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bundle stores the fact (``definitions_apart``); the sentence is rendered at read, so a wording change moves
    the text a reader sees and nothing stored: identity is over inputs and typed facts, never words."""
    definitions = {_CONTROL: [_BEFORE], _CANDIDATE: [_AFTER]}
    before = _Read(definitions)
    reworded = "Contestants were read under other definitions; dominance still reads across them."
    for module in (frontier_module, lens_text_module, method_module):
        monkeypatch.setattr(module, "DEFINITIONS_APART_READING", reworded)
    after = _Read(definitions)

    assert after.bundle.frontier.definitions_apart is True, "not vacuous: the fact is set"
    assert reworded in frontier_text(after.bundle.frontier), "not vacuous: the reader says the new words"
    assert reworded in report_markdown(after.aspect())
    stored = after.bundle.model_dump_json()
    assert stored == before.bundle.model_dump_json(), "no stored bundle byte moves"
    assert after.bundle.fingerprint() == before.bundle.fingerprint(), "no fingerprint moves"
    assert _DOMINANCE_POOLS not in stored and reworded not in stored, "the sentence is not stored, either wording"
    assert '"definitions_apart":true' in stored


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


def test_the_aspect_page_gives_an_arm_read_under_two_definitions_no_count_no_verdict_and_no_bar() -> None:
    read = _Read({_CONTROL: [_AFTER], _CANDIDATE: [_BEFORE, _AFTER]})
    first = read.aspect_first_screen()

    assert (
        f"{_CANDIDATE} is not compared with {_CONTROL}: its trials were read under more than one definition of a host "
        "measure, or under none recorded, so they are read per definition."
    ) in first
    assert "cannot tell" not in first and "more often" not in first, "a split arm is neither undecided nor ranked"
    # Ran, usable, passed, failed by check, failed by judge, cost: what ran stays; what a definition decided does not.
    assert _scoreboard_row(first, _CANDIDATE)[1:6] == ["12", "12", "— (read per definition)", "—", "—"]
    assert _scoreboard_row(first, _CANDIDATE)[6] != "—", "a definition reads what a reply scored, never its cost"
    assert _scoreboard_row(first, _CONTROL)[3] == "1", "the control, read under one definition, keeps its count"
    assert read.bars() == [_CONTROL], "no share of the split arm's trials is drawn as the arm's"


def test_the_aspect_page_of_a_one_arm_campaign_read_under_two_definitions_gives_no_pass_count() -> None:
    read = _Read({_CANDIDATE: [_BEFORE, _AFTER]})
    first = read.aspect_first_screen()

    assert f"**{_CANDIDATE} has no one pass rate: its trials were read under more than one definition" in first
    assert "passed 7 of 12" not in first and "Cases that failed a trial" not in first
    assert _scoreboard_row(first, _CANDIDATE)[3] == "— (read per definition)"
    assert read.bars() == []


def test_the_aspect_page_never_compares_two_arms_that_share_no_setup_at_any_case_count() -> None:
    said = (
        f"{_CANDIDATE} is not compared with {_CONTROL}: the two were read under different definitions of a host "
        "measure, or under none recorded, so they share no test setup."
    )
    for cases in (CASES, 3):  # 3 is under the case floor: the line that ranks arms by their passes
        first = _Read({_CONTROL: [_BEFORE], _CANDIDATE: [_AFTER]}, cases=cases).aspect_first_screen()
        assert said in first, first
        for compared in ("cannot tell", "more often", "less often", "as often"):
            assert compared not in first, (cases, first)
        # Each arm's own row is its one cell's: nothing pooled, so nothing withheld.
        assert "read per" not in first

    unrecorded = _Read({_CONTROL: [None], _CANDIDATE: [None]}).aspect_first_screen()
    assert said in unrecorded and "cannot tell" not in unrecorded


def test_the_report_input_scoreboard_row_of_an_arm_read_under_two_definitions_has_no_passes() -> None:
    built = report_input(_Read({_CONTROL: [_AFTER], _CANDIDATE: [_BEFORE, _AFTER]}).whole)

    split, control = built.scoreboard[_CANDIDATE], built.scoreboard[_CONTROL]
    assert (split.ran, split.usable, split.passed, split.failed) == (2 * CASES, 2 * CASES, None, {})
    assert split.reply_cost_usd is not None, "a definition reads what a reply scored, never what it cost"
    assert (control.passed, control.failed) == (1, {"check": CASES - 1})
    # The typed fact a reader withholds the row's counts by, naming the coordinate it is split on.
    assert split.state is not None
    assert (split.state.state, split.state.reason, split.state.coordinate) == (
        "withheld",
        "split",
        "measure_definition",
    )
    assert control.state is None
    (note,) = [line for line in built.caveats.unreported if _CANDIDATE in line]
    assert "more than one definition of a host measure" in note


def test_an_aspect_read_under_one_definition_throughout_withholds_nothing_on_its_page() -> None:
    read = _Read({_CONTROL: [_AFTER], _CANDIDATE: [_AFTER, _AFTER]})
    first = read.aspect_first_screen()

    assert "not compared" not in first and "read per" not in first
    assert _scoreboard_row(first, _CANDIDATE)[3] == str(1 + CASES)
    assert read.bars() == [_CONTROL, _CANDIDATE]
    assert report_input(read.whole).scoreboard[_CANDIDATE].passed == 1 + CASES


def test_a_written_page_cites_no_failure_count_for_the_split_arm_and_the_controls_count_resolves() -> None:
    """The report input withholds the split arm's failures as an empty count, which no key reads as 0 failed."""
    built = report_input(_Read({_CONTROL: [_AFTER], _CANDIDATE: [_BEFORE, _AFTER]}).whole)

    for key in (f"scoreboard.{_CANDIDATE}.failed", f"aspects.{ASPECT}.scoreboard.{_CANDIDATE}.failed"):
        with pytest.raises(UnstatedFigure):
            resolve(key, built)
    for key in (f"scoreboard.{_CONTROL}.failed", f"aspects.{ASPECT}.scoreboard.{_CONTROL}.failed"):
        assert resolve(key, built).value == CASES - 1


def test_the_failures_key_refuses_the_split_arm_on_its_aspect_and_the_campaign_total_still_resolves() -> None:
    """`count.failures.<aspect>.<arm>` would read the input's failed trials for the split arm, which pool its cells:
    refused as `scoreboard.<arm>.failed` is; the control's resolves, and so does the campaign's total."""
    built = analyst_input(
        _Read({_CONTROL: [_AFTER], _CANDIDATE: [_BEFORE, _AFTER]}).whole
    )  # what a snapshot freezes for a writer's page

    assert built.failed_trials is not None
    pooled = [row for row in built.failed_trials if (row.aspect, row.arm) == (ASPECT, _CANDIDATE)]
    assert sum(row.failed for row in pooled) > 0, "a pooled count is held, so its refusal asserts something"
    for key in [f"count.failures.{ASPECT}.{_CANDIDATE}"] + [
        f"count.failures.{ASPECT}.{_CANDIDATE}.{row.stratum}" for row in pooled if row.stratum is not None
    ]:
        with pytest.raises(UnstatedFigure):
            resolve(key, built)
    assert resolve(f"count.failures.{ASPECT}.{_CONTROL}", built).value == CASES - 1
    assert resolve("count.failures", built).value == sum(row.failed for row in built.failed_trials)


_TONE = "tone"


def _across_two_aspects(read: _Read) -> CampaignReadings:
    """The campaign's readings with a second aspect, on which each arm was read under one definition throughout, beside its own."""
    trials: list[tuple[EvalRun, list[EvalResult]]] = []
    for arm, passing in ((_CONTROL, 5), (_CANDIDATE, 4)):
        run = run_of(arm, measure_definitions=_AFTER).model_copy(update={"id": f"run-{_TONE}-{arm}"})
        trials.append((run, _results(run, passing)))
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


def test_the_equal_weight_figures_withhold_the_pass_rate_of_an_arm_read_under_two_definitions_on_an_aspect() -> None:
    """Its count on that aspect pools what its cells keep apart: the figure across aspects is withheld, naming it."""
    aggregate = _across_two_aspects(_Read({_CONTROL: [_AFTER], _CANDIDATE: [_BEFORE, _AFTER]})).aggregate

    assert aggregate is not None, "the arms are compared on the other aspect, or this asserts nothing"
    held = {figures.arm_label: figures for figures in aggregate.arms}
    assert (held[_CANDIDATE].pass_rate.value, held[_CANDIDATE].pass_rate.missing) == (None, [ASPECT])
    assert held[_CONTROL].pass_rate.value is not None and held[_CONTROL].pass_rate.missing == []


@pytest.mark.parametrize("stored", [False, True], ids=["built", "stored"])
def test_an_arm_read_under_two_definitions_raises_one_split_caveat_on_the_measure_definition_from_its_scoreboard_row(
    stored: bool,
) -> None:
    """The withheld-by-split caveat names the coordinate the row's typed state names, so a definition split is raised
    as one, from a stored input as from the built one."""
    built = report_input(_Read({_CONTROL: [_AFTER], _CANDIDATE: [_BEFORE, _AFTER]}).whole)

    assert _split_caveats(built, stored=stored) == [
        ("withheld", "split", "measure_definition", [_CANDIDATE], [f"{ASPECT}/passed"])
    ]


@pytest.mark.parametrize("stored", [False, True], ids=["built", "stored"])
def test_an_arm_read_under_one_definition_throughout_raises_no_split_caveat(stored: bool) -> None:
    built = report_input(_Read({_CONTROL: [_AFTER], _CANDIDATE: [_AFTER, _AFTER]}).whole)

    assert all(row.state is None for row in built.scoreboard.values())
    assert _split_caveats(built, stored=stored) == []
