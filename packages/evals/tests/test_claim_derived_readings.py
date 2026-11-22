"""A host's own readings of a claim criterion's grade (``derived_from``), computed by its claim reader.

A host declares a measure with ``derived_from: <criterion>`` and registers a pure, synchronous claim reader for that
criterion on its profile. The engine hands the reader each cell's grade and claim evidence after the grade lands, and
again on a re-judge, on settling a row pending its trace and on a re-measure; the answer is stored on the criterion's
tally, read in the judged namespace under the criterion's key, ranked as the criterion's own readings are, and lost and
withheld with them. A reader that raises, or answers what its measures cannot hold, is the host's defect: it fails what
asked and is never read as a value or as the criterion's loss.

Two readers stand for the shapes the design names, both on invented text: a radio show's back-announce order, read
from the session claims against the play order a cell's ``session`` record holds, and a tabletop session's game-state
accuracy and character contradictions, read against its roll log and its roster. Every judge is scripted, so nothing is
spent.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.off_loop import off_loop
from threetears.evals.analysis.bundle.cell_reads import judged_measures
from threetears.evals.analysis.reporting import project_score_records
from threetears.evals.kernel.candidate_kind import CandidateOutput, CellSink, CellSpanWindow, VariantConfig
from threetears.evals.kernel.cassettes import CellCassettes
from threetears.evals.kernel.claims.derived import ClaimDerivation, claim_derivations, derive_claim_readings
from threetears.evals.kernel.claims.readings import claim_names_a_measure_holds, template_claim_readings
from threetears.evals.kernel.declaration import UnreadableBarName, resolve_bar_name
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.host import (
    ClaimReader,
    ClaimReaderDefect,
    HostProfile,
    MeasureRegistry,
    ProfileRegistrationError,
)
from threetears.evals.kernel.host.eval_host import EvalHost
from threetears.evals.kernel.host.measures import MeasureRegistrationError
from threetears.evals.kernel.identity import IDENTITY_VERSION, DerivedVariantIdentity, compute_variant_key
from threetears.evals.kernel.metrics import MeasureFamily, MetricDescriptor, measure_definition_digests
from threetears.evals.kernel.provider import withhold_failure_detail
from threetears.evals.run.judge import JUDGE_REQUEST_SETTINGS
from threetears.evals.run.judge_service import JudgeService
from threetears.evals.run.lifecycle import rejudge_result
from threetears.evals.run.remeasure import remeasure_run
from threetears.evals.run.runner import (
    RunnerOptions,
    execute_run,
    judge_witnessed_output,
    refuse_misfit_host_measures,
    run_one_result,
)
from threetears.evals.schema import ClaimEvidence, ClaimGrade, EvalResult, Segment
from threetears.evals.schema.models import (
    ClaimCriterion,
    ClaimTally,
    EvalTemplate,
    EvalTestCase,
    JudgedArtifact,
    PageBudget,
    RepairBudget,
)
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.values import SweepableValue

_KIND = "derived-single-shot"
_JUDGE = "scripted/derived-judge"
_SHOW = "show.claims"
_TABLE = "table.claims"

_ORDER = f"{_SHOW}_order_right_share"
_NAMED = f"{_SHOW}_tracks_named"
_STATE = f"{_TABLE}_state_accuracy"
_ABSENT = f"{_TABLE}_absent_characters"

# =============================================================================
# The two readers, and the measures they answer
# =============================================================================


def _session_claims(grade: ClaimGrade) -> list[Any]:
    # prose-canary: allow — `scope` is the closed Literal world/session the extractor checked, not free text
    return [claim for claim in grade.claims if claim.scope == "session"]


def back_announce_order(grade: ClaimGrade, evidence: ClaimEvidence) -> Mapping[str, float | bool]:
    """A radio show's back-announce order: the tracks its session claims name, in the order the text names them,
    against the play order the session record holds, most recent first. No session claim names a track: not carried."""
    played = list((evidence.session or {})["played"])
    named = [title for claim in _session_claims(grade) for title in played if title in claim.claim]
    if not named:
        return {_NAMED: 0.0}
    right = sum(said == heard for said, heard in zip(named, reversed(played), strict=False))
    return {_ORDER: right / len(named), _NAMED: float(len(named))}


def game_state(grade: ClaimGrade, evidence: ClaimEvidence) -> Mapping[str, float | bool]:
    """A tabletop session's game state: each session claim stating a roll or a hit-point count against the roll log
    and the sheet, and each naming a character the roster marks absent, by the segment's own tag."""
    session = evidence.session or {}
    stated: list[bool] = []
    absent = 0
    for claim in _session_claims(grade):
        numbers = [int(found) for found in re.findall(r"\d+", claim.claim)]
        if "rolled" in claim.claim and numbers:
            stated.append(numbers[0] == session["rolls"][-1])
        elif "hit points" in claim.claim and numbers:
            stated.append(numbers[0] == session["hit_points"])
        named = evidence.segments[claim.segment].tags.get("character")
        absent += named is not None and named in session["absent"]
    readings: dict[str, float | bool] = {_ABSENT: float(absent)}
    if stated:
        readings[_STATE] = sum(stated) / len(stated)
    return readings


def _derived(name: str, criterion: str, **fields: Any) -> MetricDescriptor:
    base: dict[str, Any] = {
        "name": name,
        "reader_name": name.replace("_", " "),
        "data_type": "numeric",
        "family": "claims",
        "transferability_class": "judge_mediated",
        "attribution_scope": "end_to_end",
        "description": f"A host's reading of {criterion}'s session claims against the session's own record.",
        "derived_from": criterion,
    }
    return MetricDescriptor(**{**base, **fields})


_DERIVED: tuple[MetricDescriptor, ...] = (
    _derived(_ORDER, _SHOW, higher_is_better=True, value_range=(0.0, 1.0)),
    _derived(_NAMED, _SHOW, diagnostic=True, nonnegative=True),
    _derived(_STATE, _TABLE, higher_is_better=True, value_range=(0.0, 1.0)),
    _derived(_ABSENT, _TABLE, higher_is_better=False, nonnegative=True),
)


def _profile(
    readers: Mapping[str, ClaimReader] | None = None,
    derived: tuple[MetricDescriptor, ...] = _DERIVED,
    families: tuple[MeasureFamily, ...] = (),
) -> HostProfile:
    base = toyhost_profile()
    return replace(
        base,
        measures=MeasureRegistry([*TOYHOST_MEASURES, *derived], families=(*base.measures.families, *families)),
        claim_readers=dict(readers if readers is not None else {_SHOW: back_announce_order, _TABLE: game_state}),
    )


# =============================================================================
# The registry and the profile refuse what no derived reading can be
# =============================================================================


@pytest.mark.parametrize(
    ("fields", "families", "refused"),
    [
        ({"family": "mechanical"}, (), "graded by code"),
        (
            {"family": "table_mechanics"},
            (MeasureFamily(name="table_mechanics", graded_by="code", description="x"),),
            "graded by code",
        ),
        ({"family": None}, (), "in no family"),
        ({"family": "rubric"}, (), "the engine's rubric family"),
        ({"name": "order_right_share"}, (), "not named under its key"),
        ({"data_type": "categorical", "categories": ("a", "b")}, (), "a reading of claims is a number"),
        ({"guardrail": True}, (), "itself a guardrail"),
        ({"merit_axis": "quality"}, (), "merit axis quality"),
        ({"shapes_run": True}, (), "shapes_run"),
    ],
    ids=["engine-code", "host-code", "no-family", "rubric", "off-key", "categorical", "guardrail", "axis", "shapes"],
)
def test_the_registry_refuses_a_derived_reading_that_is_not_a_judged_number_under_its_criterions_key(
    fields: dict[str, Any], families: tuple[MeasureFamily, ...], refused: str
) -> None:
    descriptor = _derived(_ORDER, _SHOW, higher_is_better=True, value_range=(0.0, 1.0)).model_copy(update=fields)

    with pytest.raises(MeasureRegistrationError, match=refused):
        MeasureRegistry([descriptor], families=families)


def test_a_derived_reading_in_a_judge_graded_host_family_is_admitted() -> None:
    family = MeasureFamily(name="table_judged", graded_by="judge", description="A judge's reading of the table.")
    descriptor = _derived(_ABSENT, _TABLE, family="table_judged", higher_is_better=False, nonnegative=True)

    assert MeasureRegistry([descriptor], families=(family,)).derived == (descriptor,)


def test_the_profile_refuses_a_derived_reading_with_no_reader_and_a_reader_with_no_reading() -> None:
    with pytest.raises(
        ProfileRegistrationError, match=r"derive from claim criterion 'table.claims', and no claim reader"
    ):
        _profile(readers={_SHOW: back_announce_order})
    with pytest.raises(ProfileRegistrationError, match=r"registered for criterion 'quiz.claims', and no measure"):
        _profile(readers={_SHOW: back_announce_order, _TABLE: game_state, "quiz.claims": game_state})


def test_a_descriptor_that_derives_nothing_keeps_the_definition_digest_every_stored_run_recorded() -> None:
    plain = TOYHOST_MEASURES[0]
    stored = MeasureRegistry([plain], families=toyhost_profile().measures.families).definition_digests()
    # The digest the build before derived readings recorded: the same fields, with no derived_from among them.
    fields = (
        "name",
        "data_type",
        "family",
        "unit",
        "value_range",
        "categories",
        "formula",
        "pass_fail",
        "contained_by",
        "definition_revision",
        "shapes_run",
    )
    assert stored[plain.name] == canonical_digest({name: getattr(plain, name) for name in fields})
    derived = plain.model_copy(update={"derived_from": "x.claims"})
    assert measure_definition_digests([derived])[plain.name] != stored[plain.name], (
        "the control: which criterion a reading derives from is what it is"
    )


# =============================================================================
# Where a template meets the host
# =============================================================================


def _criterion(name: str) -> ClaimCriterion:
    return ClaimCriterion(
        name=name,
        pages=PageBudget(per_document_chars=2000, total_chars=4000, window_chars=1000),
        repairs=RepairBudget(rounds=2, max_items_per_cell=4),
        judge="builtin",
    )


def _template(criterion: str = _SHOW) -> EvalTemplate:
    return EvalTemplate(
        scope_id="u",
        name="late show",
        intent="present the late show between tracks",
        candidate_kind=_KIND,
        claim_criteria=[_criterion(criterion)],
    )


def test_a_template_admits_the_readings_its_host_derives_from_its_criterion_and_refuses_a_clash() -> None:
    assert claim_names_a_measure_holds(_template(), _profile().measures) is None
    stolen = _derived(f"{_SHOW}_false_per_unit", _SHOW, higher_is_better=False, nonnegative=True)
    clash = claim_names_a_measure_holds(_template(), _profile(derived=(*_DERIVED, stolen)).measures)
    assert clash is not None and f"{_SHOW}_false_per_unit" in clash, "the criterion's own reading's name is its own"
    # A host measure under the criterion's key, derived from another criterion, takes a name the criterion owns.
    under = _derived(f"{_SHOW}_mood", _TABLE, higher_is_better=True, value_range=(0.0, 1.0))
    with pytest.raises(MeasureRegistrationError, match="not named under its key"):
        _profile(derived=(*_DERIVED, under))


def test_a_bar_on_a_derived_reading_is_refused_as_a_reading_that_only_ranks() -> None:
    measures = _profile().measures
    resolved = resolve_bar_name(
        _ORDER,
        rubric_dimensions={},
        goal_state_checks=[],
        measures=measures,
        claim_readings=template_claim_readings(_template(), measures),
    )
    assert isinstance(resolved, UnreadableBarName) and resolved.refusal == "ranks_only"
    # A template that declares another criterion offers none of this one's readings: the name is carried by no result.
    elsewhere = resolve_bar_name(
        _ORDER,
        rubric_dimensions={},
        goal_state_checks=[],
        measures=measures,
        claim_readings=template_claim_readings(_template(_TABLE), measures),
    )
    assert isinstance(elsewhere, UnreadableBarName) and elsewhere.refusal == "not_carried"


def test_a_kind_landing_a_derived_reading_on_its_host_measures_is_refused() -> None:
    with pytest.raises(ValueError, match="derived from claim criterion"):
        refuse_misfit_host_measures({_ORDER: 1.0}, _profile().measures)


# =============================================================================
# The readers, through the claims phase
# =============================================================================

_SHOW_SEGMENTS = (
    Segment(text="Good evening, this is the late show."),
    Segment(text="That was Lowland Air, off Second Tide.", unit=0, tags={"break": "1"}),
    Segment(text="And before that, Copper Bells.", unit=1, tags={"break": "1"}),
)
_PLAYED = {"played": ["Copper Bells", "Lowland Air"]}


def _announced(*titles: str) -> list[dict[str, Any]]:
    """The session claims a back-announce of ``titles`` lists, one per segment after the greeting."""
    excerpts = ("That was Lowland Air", "And before that, Copper Bells")
    return [
        {
            "segment": segment,
            "claim": f"{title} played",
            "excerpt": excerpts[segment - 1],
            "specific": True,
            "hedged": False,
            "attributed": False,
            "scope": "session",
            "restates": None,
        }
        for segment, title in enumerate(titles, start=1)
    ]


@dataclass(frozen=True)
class _Reply:
    """One completion, shaped as the engine's completion port describes it."""

    content: str
    stop_reason: str = "end_turn"
    model: str = _JUDGE
    served_model: str | None = _JUDGE
    input_tokens: int | None = 300
    output_tokens: int | None = 60
    reasoning_tokens: int | None = None
    cost_usd: float | None = 0.001
    price_source: str | None = "scripted"
    temperature: float | None = None
    sent_settings: Any = None


# parity-with: threetears.evals.schema.completion.BoundCompletionClient
@dataclass
class _FakeListingJudge:
    """Lists the session claims it is given, in turn (the last answers every later listing), or raises one given."""

    listings: list[Any] = field(default_factory=list)
    model_name: str = _JUDGE
    asked: list[str] = field(default_factory=list)

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        del system, user, response_format
        return None

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        del system, user
        name = (response_format or {}).get("json_schema", {}).get("name", "")
        self.asked.append(name)
        if name != "claim_listing":
            raise AssertionError(f"a session-claims cell asks no {name} call")
        answer = self.listings[min(self.asked.count(name), len(self.listings)) - 1]
        if isinstance(answer, BaseException):
            raise answer
        return _Reply(content=json.dumps({"claims": answer}))

    async def aclose(self) -> None:
        """Nothing to release."""


# parity-with: threetears.evals.kernel.candidate_kind.CandidateKind
class _FakeShowKind:
    """A single-shot kind that hands back one canned back-announce, its claim evidence and session record with it."""

    judged_artifact = JudgedArtifact.UNJUDGED

    def __init__(self, evidence: ClaimEvidence) -> None:
        self.evidence = evidence

    async def prepare(
        self,
        *,
        subject_snapshot: Any,
        variant_config: VariantConfig,
        world_seed: Any,
        span_window: CellSpanWindow,
        cassettes: CellCassettes | None,
        world: Any,
    ) -> object:
        return object()

    async def invoke(self, instance: object, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        return CandidateOutput(
            output=[{"said": " ".join(s.text for s in self.evidence.segments)}], claim_evidence=self.evidence
        )


_SHOW_EVIDENCE = ClaimEvidence(segmentation="lines-v1", segments=_SHOW_SEGMENTS, session=_PLAYED)
_MODEL_ONLY = {"model": SweepableValue.of("test/model", display="test/model")}


def _service(judge: Any) -> JudgeService:
    return JudgeService(
        client_factory=lambda model, temperature: judge, configs={}, failure_describer=withhold_failure_detail
    )


def _host(profile: HostProfile | None = None, judge: Any = None) -> EvalHost:
    clients = None if judge is None else (lambda role, model, *, temperature=None: judge)
    return toyhost_host(profile=profile or _profile(), clients=clients)


async def _cell(host: EvalHost, judge: Any, template: EvalTemplate, *, evidence: ClaimEvidence | None = None) -> Any:
    case = EvalTestCase(template_id=template.id, scope_id="u")
    shown = evidence or _SHOW_EVIDENCE
    return await run_one_result(
        host,
        template=template,
        test_case=case,
        subject_id="subj-1",
        model="test/model",
        k_iteration=1,
        eval_run_id="run-1",
        scope_id="u",
        judge_service=_service(judge),
        options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: _FakeShowKind(shown)}),
        variant=DerivedVariantIdentity(
            variant_key=compute_variant_key(_MODEL_ONLY), identity_version=IDENTITY_VERSION, levers=_MODEL_ONLY
        ),
    )


async def test_the_reader_reads_the_session_claims_against_the_session_record_once_the_grade_lands() -> None:
    seen: list[tuple[ClaimGrade, ClaimEvidence]] = []

    def reading(grade: ClaimGrade, evidence: ClaimEvidence) -> Mapping[str, float | bool]:
        seen.append((grade, evidence))
        return back_announce_order(grade, evidence)

    judge = _FakeListingJudge(listings=[_announced("Lowland Air", "Copper Bells")])
    result, trace = await _cell(_host(_profile({_SHOW: reading, _TABLE: game_state})), judge, _template())

    tally = result.claim_tallies[_SHOW]
    assert tally.derived == {_ORDER: 1.0, _NAMED: 2.0}, "most recent first, as the play order runs backwards"
    assert tally.session_claims == 2 and tally.by_judge == (), "session claims are no world claim"
    ((grade, evidence),) = seen
    assert grade == trace.claim_grades[_SHOW] and evidence == trace.claim_evidence == _SHOW_EVIDENCE
    assert judge.asked == ["claim_listing"], "no truth step reads a session claim"


async def test_a_shuffled_back_announce_moves_the_derived_reading_and_no_truth_reading() -> None:
    judge = _FakeListingJudge(listings=[_announced("Copper Bells", "Lowland Air")])

    result, _ = await _cell(_host(), judge, _template())

    tally = result.claim_tallies[_SHOW]
    assert tally.derived == {_ORDER: 0.0, _NAMED: 2.0}
    assert tally.units_with_false == 0 and tally.session_claims == 2, "a continuity slip reads as no false statement"


async def test_a_reader_that_raises_fails_the_cell_as_the_hosts_defect_never_as_a_loss() -> None:
    def broken(grade: ClaimGrade, evidence: ClaimEvidence) -> Mapping[str, float | bool]:
        raise KeyError("played")

    judge = _FakeListingJudge(listings=[_announced("Lowland Air", "Copper Bells")])

    with pytest.raises(ClaimReaderDefect, match=r"criterion 'show.claims' raised KeyError"):
        await _cell(_host(_profile({_SHOW: broken, _TABLE: game_state})), judge, _template())


async def test_a_run_whose_reader_raises_records_its_cell_with_what_it_spent_and_fails() -> None:
    def broken(grade: ClaimGrade, evidence: ClaimEvidence) -> Mapping[str, float | bool]:
        raise ZeroDivisionError("no tracks")

    judge = _FakeListingJudge(listings=[_announced("Lowland Air", "Copper Bells")])
    host = _host(_profile({_SHOW: broken, _TABLE: game_state}))
    template = _template()
    case = EvalTestCase(template_id=template.id, scope_id="u")
    run = make_eval_run(
        scope_id="u", template_id=template.id, candidate_kind=_KIND, test_case_ids=[case.id], judge_model=_JUDGE
    )
    off_loop(host.storage.save_eval_run, run)

    with pytest.raises(ClaimReaderDefect):
        await execute_run(
            host,
            run=run,
            template=template,
            test_cases=[case],
            judge_service=_service(judge),
            options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: _FakeShowKind(_SHOW_EVIDENCE)}),
        )

    (result,) = off_loop(host.storage.query_eval_results_by_run, run.id, run.scope_id)
    assert result.infra_error is not None and "ClaimReaderDefect" in result.infra_error
    assert result.claim_tallies == {}, "no value of the reader's is stored"
    (row,) = [row for row in result.usage if row.role == "judge"]
    assert row.call_count == 1, "the listing it read was paid for, and counted"


def _derivation(reader: ClaimReader) -> ClaimDerivation:
    derivations = claim_derivations(_profile({_SHOW: reader, _TABLE: game_state}))
    return derivations[_SHOW]


@pytest.mark.parametrize(
    ("answer", "refused"),
    [
        ({f"{_SHOW}_mood": 1.0}, "no measure derived from it"),
        ({_STATE: 1.0}, "no measure derived from it"),
        ({_ORDER: True}, "not a number"),
        ({_ORDER: 1.5}, "outside the measure's value range"),
        ({_ORDER: float("nan")}, "not finite"),
        ({_NAMED: -1.0}, "below zero"),
        ([(_ORDER, 1.0)], "not a mapping"),
    ],
    ids=["unknown", "another-criterions", "bool", "out-of-range", "nan", "negative", "not-a-mapping"],
)
async def test_a_reader_answering_what_its_measures_cannot_hold_is_the_hosts_defect(answer: Any, refused: str) -> None:
    judge = _FakeListingJudge(listings=[_announced("Lowland Air", "Copper Bells")])
    _, trace = await _cell(_host(), judge, _template())

    with pytest.raises(ClaimReaderDefect, match=refused):
        derive_claim_readings(_derivation(lambda grade, evidence: answer), trace.claim_grades[_SHOW], _SHOW_EVIDENCE)


_TABLE_SEGMENTS = (
    Segment(text="You rolled a 14, and the door gives way.", unit=0),
    Segment(text="You are at 9 hit points.", unit=1),
    Segment(text="Mira waves you through the gate.", unit=2, tags={"character": "Mira"}),
)
_TABLE_EVIDENCE = ClaimEvidence(
    segmentation="beats-v1",
    segments=_TABLE_SEGMENTS,
    session={"rolls": [3, 14], "hit_points": 7, "absent": ["Mira"]},
)


def _beat(segment: int, claim: str, excerpt: str) -> dict[str, Any]:
    """A session claim the narrator's beat ``segment`` states."""
    return {
        "segment": segment,
        "claim": claim,
        "excerpt": excerpt,
        "specific": True,
        "hedged": False,
        "attributed": False,
        "scope": "session",
        "restates": None,
    }


async def test_the_tabletop_reader_reads_game_state_claims_against_the_roll_log_and_the_roster() -> None:
    judge = _FakeListingJudge(
        listings=[
            [
                _beat(0, "the player rolled a 14", "You rolled a 14"),
                _beat(1, "the player has 9 hit points", "You are at 9 hit points"),
                _beat(2, "Mira is at the gate", "Mira waves you through the gate"),
            ]
        ]
    )

    result, _ = await _cell(_host(), judge, _template(_TABLE), evidence=_TABLE_EVIDENCE)

    assert result.claim_tallies[_TABLE].derived == {_ABSENT: 1.0, _STATE: 0.5}, (
        "the roll is right, the hit points wrong, and the roster says Mira is not there"
    )


# =============================================================================
# Ranked, lost and withheld as the criterion's own readings
# =============================================================================


def _tally(order: float) -> ClaimTally:
    return ClaimTally(
        criterion_fingerprint="c" * 64,
        documents_declared=False,
        units=2,
        units_with_false=0,
        session_claims=2,
        derived={_ORDER: order, _NAMED: 2.0},
    )


def _campaign(lost_by_arm: Mapping[str, int], order_by_arm: Mapping[str, float]) -> tuple[list[Any], list[EvalResult]]:
    """Two arms of twelve cases, two repeats each: each arm's first ``lost_by_arm`` cells lost to the criterion."""
    runs = [make_eval_run(status="completed", candidate_model=arm, template_id="tpl-show") for arm in lost_by_arm]
    results = []
    for run, (arm, lost) in zip(runs, lost_by_arm.items(), strict=True):
        for place in range(24):
            case, k = divmod(place, 2)
            results.append(
                make_eval_result(
                    id=f"{arm}-{case:02d}-{k}",
                    eval_run_id=run.id,
                    scope_id=run.scope_id,
                    model=arm,
                    test_case_id=f"tc-{case:02d}",
                    k_iteration=k + 1,
                    goal_state_outcomes=[],
                    rubric_scores=[],
                    cost_usd=None,
                    claim_tallies={} if place < lost else {_SHOW: _tally(order_by_arm[arm])},
                    claim_failures={_SHOW: "cut_short"} if place < lost else {},
                )
            )
    return runs, results


def _judged(lost_by_arm: Mapping[str, int], order_by_arm: Mapping[str, float]) -> tuple[list[Any], Any]:
    profile = _profile()
    runs, results = _campaign(lost_by_arm, order_by_arm)
    records = project_score_records(runs, results, archived_run_ids=None, profile=profile).records
    by_cell = {(arm, "rig"): [r for r in results if r.model == arm] for arm in lost_by_arm}
    return records, judged_measures(records, by_cell, None, tiers=[], registry=profile.measures)


def test_a_ranked_derived_reading_is_a_judged_reading_under_its_criterion_and_its_diagnostic_is_not() -> None:
    records, measured = _judged({"arm-a": 0, "arm-b": 0}, {"arm-a": 1.0, "arm-b": 0.5})

    rows = [record for record in records if record.rubric_dim == _ORDER]
    assert {row.instrument for row in rows} == {_SHOW} and {row.rubric_scale for row in rows} == {None}
    (order,) = [measure for measure in measured if measure.name == _ORDER]
    assert (order.instrument, order.reader_name, order.higher_is_better, order.value_range) == (
        _SHOW,
        _ORDER.replace("_", " "),
        True,
        (0.0, 1.0),
    )
    assert {arm.variant_key: arm.mean for arm in order.arms} == {"arm-a": 1.0, "arm-b": 0.5}
    assert any(record.rubric_dim == _NAMED for record in records), "the diagnostic is projected"
    assert _NAMED not in {measure.name for measure in measured}, "and, with no better end, never ranked"


def test_a_lost_criterion_withholds_its_derived_reading_and_counts_each_loss() -> None:
    records, measured = _judged({"arm-a": 0, "arm-b": 12}, {"arm-a": 1.0, "arm-b": 1.0})

    lost = [r for r in records if r.rubric_dim == _ORDER and r.outcome == "claim_lost"]
    assert len(lost) == 12 and {r.model for r in lost} == {"arm-b"} and {r.value for r in lost} == {None}
    assert not [r for r in records if r.rubric_dim == _NAMED and r.outcome == "claim_lost"], "a diagnostic counts none"
    (order,) = [measure for measure in measured if measure.name == _ORDER]
    assert {arm.variant_key: arm.n_claim_lost for arm in order.arms} == {"arm-a": 0, "arm-b": 12}
    assert order.withheld is not None and order.withheld.reason == "trials_lost"
    (_, control) = _judged({"arm-a": 1, "arm-b": 1}, {"arm-a": 1.0, "arm-b": 1.0})
    (held,) = [measure for measure in control if measure.name == _ORDER]
    assert held.withheld is None, "the control: one loss on each arm withholds nothing"


# =============================================================================
# Again on a re-judge, on settling a pending row, on a witnessed cell, and on a re-measure
# =============================================================================


async def _stored(
    host: EvalHost, judge: Any, template: EvalTemplate, *, trace_stored: bool = True, **run_fields: Any
) -> tuple[Any, Any, Any]:
    """One back-announce cell graded by ``judge`` and stored under a finished run, its trace too unless
    ``trace_stored`` is False."""
    off_loop(host.storage.save_template, template)
    run = make_eval_run(
        scope_id="u",
        template_id=template.id,
        candidate_kind=_KIND,
        status="completed",
        judge_model=_JUDGE,
        effective_judges={},
        **{"judge_asked_settings": JUDGE_REQUEST_SETTINGS, **run_fields},
    )
    case = EvalTestCase(template_id=template.id, scope_id="u")
    off_loop(host.storage.save_test_case, case)
    run = run.model_copy(update={"test_case_ids": [case.id]})
    off_loop(host.storage.save_eval_run, run)
    outcome = await run_one_result(
        host,
        template=template,
        test_case=case,
        subject_id="subj-1",
        model="test/model",
        k_iteration=1,
        eval_run_id=run.id,
        scope_id="u",
        judge_service=_service(judge),
        options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: _FakeShowKind(_SHOW_EVIDENCE)}),
        variant=DerivedVariantIdentity(
            variant_key=compute_variant_key(_MODEL_ONLY), identity_version=IDENTITY_VERSION, levers=_MODEL_ONLY
        ),
    )
    off_loop(host.storage.save_eval_result, outcome.result, outcome.trace if trace_stored else None)
    return run, outcome.result, outcome.trace


async def test_a_criterion_graded_again_on_a_re_judge_carries_its_derived_readings() -> None:
    judge = _FakeListingJudge(listings=[ConnectionError("unavailable"), _announced("Lowland Air", "Copper Bells")])
    host = _host(judge=judge)
    _, result, _ = await _stored(host, judge, _template())
    assert result.claim_failures == {_SHOW: "provider_error"}

    rejudged = await rejudge_result(host, result.id, "u")

    assert rejudged.claim_tallies[_SHOW].derived == {_ORDER: 1.0, _NAMED: 2.0}


async def test_a_row_pending_its_trace_is_settled_with_its_derived_readings() -> None:
    judge = _FakeListingJudge(listings=[_announced("Copper Bells", "Lowland Air")])
    host = _host(judge=judge)
    _, result, trace = await _stored(host, judge, _template())
    pending = result.model_copy(update={"claim_tallies": {}, "claim_failures": {_SHOW: "trace_pending"}})
    off_loop(host.storage.save_eval_result, pending, trace)

    rejudged = await rejudge_result(host, result.id, "u")

    assert rejudged.claim_tallies == result.claim_tallies and rejudged.claim_tallies[_SHOW].derived[_ORDER] == 0.0
    assert judge.asked == ["claim_listing"], "settling asks nothing"


async def test_a_witnessed_cell_is_read_by_the_reader_its_host_registers() -> None:
    judge = _FakeListingJudge(listings=[_announced("Lowland Air", "Copper Bells")])
    template = _template()

    judged, _ = await judge_witnessed_output(
        template=template,
        test_case=EvalTestCase(template_id=template.id, scope_id="u"),
        output=CandidateOutput(output=[{"said": "a back-announce"}], claim_evidence=_SHOW_EVIDENCE),
        goal_outcomes=[],
        judged_artifact=JudgedArtifact.UNJUDGED,
        judge_service=_service(judge),
        concurrency=1,
        eval_run_id="run-witnessed",
        derivations=claim_derivations(_profile()),
    )

    assert judged is not None and judged.claims is not None
    assert judged.claims.tallies[_SHOW].derived == {_ORDER: 1.0, _NAMED: 2.0}


def _stricter(grade: ClaimGrade, evidence: ClaimEvidence) -> Mapping[str, float | bool]:
    """The back-announce order under its next definition: right only where every track named is right."""
    read = dict(back_announce_order(grade, evidence))
    if _ORDER in read:
        read[_ORDER] = float(read[_ORDER] == 1.0)
    return read


async def test_a_re_measure_reads_a_moved_derived_reading_again_from_the_stored_grade() -> None:
    first = _profile()
    judge = _FakeListingJudge(listings=[_announced("Lowland Air", "Lowland Air")])
    host = _host(first, judge)
    run, result, _ = await _stored(host, judge, _template(), measure_definitions=first.measures.definition_digests())
    assert result.claim_tallies[_SHOW].derived == {_ORDER: 0.5, _NAMED: 2.0}
    revised = tuple(d.model_copy(update={"definition_revision": 1}) if d.name == _ORDER else d for d in _DERIVED)
    moved = toyhost_host(
        profile=_profile({_SHOW: _stricter, _TABLE: game_state}, derived=revised), storage=host.storage
    )

    report = await remeasure_run(moved, run.id, "u", apply=True)

    assert (report.moved, report.reread, report.values_changed) == ([_ORDER], [_ORDER], 1)
    stored = off_loop(moved.storage.load_eval_result, result.id, "u")
    assert stored is not None
    assert stored.claim_tallies[_SHOW].derived == {_ORDER: 0.0, _NAMED: 2.0}, "read again; the unmoved one kept"
    (reread,) = stored.measure_rereads
    assert reread.was == {_ORDER: 0.5} and set(reread.definitions) == {_ORDER}
    assert stored.host_measures == result.host_measures
    rerun = off_loop(moved.storage.load_eval_run, run.id, "u")
    assert rerun is not None and rerun.measure_definitions == moved.profile.measures.definition_digests()


async def test_a_re_measure_refuses_a_moved_derived_reading_with_no_stored_grade_to_read_it_from() -> None:
    first = _profile()
    judge = _FakeListingJudge(listings=[_announced("Lowland Air", "Copper Bells")])
    host = _host(first, judge)
    run, result, _ = await _stored(
        host, judge, _template(), trace_stored=False, measure_definitions=first.measures.definition_digests()
    )
    revised = tuple(d.model_copy(update={"definition_revision": 1}) if d.name == _ORDER else d for d in _DERIVED)
    moved = toyhost_host(profile=_profile(derived=revised), storage=host.storage)

    with pytest.raises(ValidationFailedError, match="no stored grade"):
        await remeasure_run(moved, run.id, "u", apply=True)
