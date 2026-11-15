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
- the same campaign read under one definition throughout reads as it always did, with nothing withheld.

**Mutations these catch:** an arm-level reader asking only about the served model; a floor or rate between arms that
share no rig; the report input keeping one cell's figures under the arm's name.
"""

from __future__ import annotations

from collections.abc import Sequence

from threetears.evals.analysis import assemble_context_bundle, build_code_only_report, campaign_report_markdown
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.campaign_bundle import CampaignBundle, build_campaign_bundle
from threetears.evals.analysis.campaign_readings import CampaignReadings, campaign_readings
from threetears.evals.analysis.report.campaign.page import build_campaign_report
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


def _results(run: EvalRun, passing: int) -> list[EvalResult]:
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
        for index in range(CASES)
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

    def __init__(self, definitions_by_arm: dict[str, Sequence[dict[str, str] | None]]) -> None:
        trials: list[tuple[EvalRun, list[EvalResult]]] = []
        for arm, each in definitions_by_arm.items():
            for index, definitions in enumerate(each):
                run = run_of(arm, measure_definitions=definitions).model_copy(update={"id": f"run-{arm}-{index}"})
                # The candidate's second run passes everything, so a rate pooled over both definitions reads apart.
                trials.append((run, _results(run, CASES if arm == _CANDIDATE and index else 1)))
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
