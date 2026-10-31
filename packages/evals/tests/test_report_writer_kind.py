"""The report writer as a candidate kind: two arms that differ by one line of the skill, run as the engine runs any.

A run of the ``report_writer`` kind writes a report over each frozen writer case with the arm's skill, effort and
revisions, on scripted turns, and lands the run's own measures on each result. These tests drive two arms (the shipped
skill and a candidate with one line added) through :func:`~threetears.evals.run.runner.execute_run` over two cases on a
host whose store holds neither case's snapshot: each cell restores its case's own. They hold that both arms produce
results, keyed apart by the skill lever; that the measures land typed as their registry declares them; that a case's
withheld note never reaches the analyst; and that a skill the launch was not given, or a withheld note the skill
lacks, is refused before any turn is sent.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, get_args

import pytest

from packages.evals.tests.analyst_support import KAIRO, fixture_snapshots
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, ops_fixture
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.report_writer_support import (
    MODEL,
    FakeToolTurnClient,
    call,
    reply,
    review,
    submitted,
)
from threetears.evals.actions.report_writing import report_writer_kind
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.kernel.host.kinds import freeze
from threetears.evals.kernel.host.measures import MeasureRegistry
from threetears.evals.kernel.identity import LeverCoordinateError, resolve_variant_identity
from threetears.evals.negotiation.settings import EFFORT_LADDER
from threetears.evals.ops import OpsHost
from threetears.evals.ops.report_writer.tally import ReportBudget
from threetears.evals.ops.report_writer.writer_contract import (
    ENDING_MEASURE,
    FINISHED_MEASURE,
    MUST_FIX_OPEN_MEASURE,
    SHOULD_FIX_OPEN_MEASURE,
    SKILL_LEVER,
    TURNS_MEASURE,
    WRITER_KIND,
    WRITER_KIND_CONTRACT,
    WRITER_KIND_MEASURES,
    EffortWord,
    WriterOverlays,
    writer_kind_contract,
)
from threetears.evals.ops.report_writer.writer_kind import (
    WriterCase,
    writer_case_payload,
    writer_subject_components,
)
from threetears.evals.run.runner import NothingMeasuredError, RunnerOptions, execute_run
from threetears.evals.schema.models import EvalRun, EvalTemplate, EvalTestCase
from threetears.evals.schema.subject import SubjectSnapshot
from threetears.evals.schema.turns import ToolTurnClient, TurnText

#: The line only the candidate skill's analyst text holds.
_CANDIDATE_RULE = "Candidate rule: lead with the arm a reader would ship."

#: The note each case withholds, by the snapshot it is written over.
_KAIRO_NOTE = "kairo-round2.md"
_R1_NOTE = "r1-model-selection.md"

#: The fixture snapshot the second case is written over.
_R1 = "r1"


def _candidate() -> AnalystSkill:
    """The shipped skill with one line added to the analyst's role file."""
    files = dict(load_skill().files)
    files["SKILL.md"] = f"{files['SKILL.md']}\n\n{_CANDIDATE_RULE}\n"
    return AnalystSkill(files=files)


def _host() -> OpsHost:
    """The toy host with the writer's contract on its kinds and its measures on its registry, its store holding no
    writer snapshot."""
    profile = toyhost_profile()
    profile = replace(
        profile,
        kinds=(*profile.kinds, writer_kind_contract(host_seats=("reviewer_pool",))),
        measures=MeasureRegistry((*TOYHOST_MEASURES, *WRITER_KIND_MEASURES), families=profile.measures.families),
    )
    return ops_fixture(generation=False, profile=profile).host


def _cases() -> list[EvalTestCase]:
    """Kairo, cleared at its first review, and r1, whose analyst never submits a document; each frozen in another
    scope than the run's, withholding its own campaign's note."""
    frozen = fixture_snapshots("frozen-elsewhere")
    return [
        EvalTestCase(
            id=f"writer-{name}",
            template_id="writer-template",
            scope_id=CALLER.scope_id,
            host_payload=writer_case_payload(WriterCase.of(frozen[name], withhold=[note])),
        )
        for name, note in ((KAIRO, _KAIRO_NOTE), (_R1, _R1_NOTE))
    ]


def _snapshot_id(name: str) -> str:
    return fixture_snapshots("frozen-elsewhere")[name].id


def _kairo_sessions() -> dict[str, FakeToolTurnClient]:
    """An analyst that submits the Kairo document and revises it once; an editor whose first review raises a
    should_fix, which blocks in round 1, and whose second resolves it and raises another, which blocks nothing: the
    report is stored finished with one should_fix open."""
    kairo = _snapshot_id(KAIRO)
    first = {"id": "f1", "severity": "should_fix", "claim": "c1", "finding": "say which routing", "continues": "new"}
    second = {"id": "f2", "severity": "should_fix", "claim": "c1", "finding": "name the run", "continues": "new"}
    return {
        "analyst": FakeToolTurnClient(
            reply(call(1, "options_dominance", kairo)), reply(submitted()), reply(submitted())
        ),
        "editor-1": FakeToolTurnClient(reply(review(scores=(2, 2, 2, 2, 2, 2, 2), findings=[first]))),
        "editor-2": FakeToolTurnClient(
            reply(review(scores=(2, 2, 2, 2, 2, 2, 2), findings=[second], resolved=["f1.1"]))
        ),
    }


def _r1_sessions() -> dict[str, FakeToolTurnClient]:
    """An analyst whose every reply holds no document: bounced three times, then refused."""
    return {"analyst": FakeToolTurnClient(*(reply(TurnText("I have no report yet.")) for _ in range(4)))}


# parity-with: threetears.evals.ops.report_writer.loop.TurnModel
@dataclass
class _FakeCaseRoutedModel:
    """One cell's model: each session's scripted client, chosen by the snapshot the session reads."""

    effort: EffortWord | None
    model_name: str = MODEL
    sessions: dict[str, FakeToolTurnClient] = field(default_factory=dict)

    async def turn_pricing_refusal(self) -> str | None:
        return None

    def turn_session(self, name: str, *, snapshot: str) -> ToolTurnClient:
        if not self.sessions:
            self.sessions.update(_kairo_sessions() if snapshot == _snapshot_id(KAIRO) else _r1_sessions())
        return self.sessions[name]


@dataclass
class _Models:
    """The host's model factory: a fresh scripted model per cell, every one kept."""

    built: list[_FakeCaseRoutedModel] = field(default_factory=list)

    def __call__(self, effort: EffortWord | None) -> _FakeCaseRoutedModel:
        model = _FakeCaseRoutedModel(effort)
        self.built.append(model)
        return model


def _arm(skill: AnalystSkill, *, effort: EffortWord | None = None) -> EvalRun:
    """One arm's run, given one revision: its overlays frozen as a launch freezes them, its subject carrying its
    skill."""
    overlays = WRITER_KIND_CONTRACT.validate_overlays(
        {"skill": skill.digest, "reasoning_effort": effort, "revisions": 1}
    )
    return make_eval_run(
        scope_id=CALLER.scope_id,
        template_id="writer-template",
        candidate_kind=WRITER_KIND,
        candidate_model=MODEL,
        overlays=freeze(overlays),
        subject_snapshot=SubjectSnapshot(
            subject_id="report writer",
            subject_label="report writer",
            components=writer_subject_components(skill),
            state={},
        ),
        k_runs=1,
        test_case_ids=[case.id for case in _cases()],
    )


_TEMPLATE = EvalTemplate(
    id="writer-template",
    scope_id=CALLER.scope_id,
    name="Report writer",
    intent="Write the report.",
    candidate_kind=WRITER_KIND,
)


async def _execute(
    host: OpsHost,
    run: EvalRun,
    models: _Models,
    *,
    skills: dict[str, AnalystSkill],
    subject_skill: AnalystSkill | None = None,
) -> EvalRun:
    """Run one arm over both cases; the run, as it was executed."""
    overlays = WriterOverlays.model_validate(run.overlays)
    kind = report_writer_kind(
        host, CALLER, overlays=overlays, models=models, budget=ReportBudget(usd=5.0), skills=skills
    )
    if subject_skill is not None:
        run = run.model_copy(
            update={
                "subject_snapshot": run.subject_snapshot.model_copy(
                    update={"components": writer_subject_components(subject_skill)}
                )
            }
        )
    await execute_run(
        host.eval_host,
        run=run,
        template=_TEMPLATE,
        test_cases=_cases(),
        judge_service=None,
        options=RunnerOptions(candidate_kinds={WRITER_KIND: lambda _cell: kind}),
    )
    return run


def _results(host: OpsHost, run: EvalRun) -> list[Any]:
    """The run's stored results, by case."""
    results = off_loop(host.eval_host.storage.query_eval_results_by_run, run.id, run.scope_id)
    return sorted(results, key=lambda result: result.test_case_id)


def _analyst_system(models: _Models, snapshot: str) -> str:
    """The system text the analyst of the cell over ``snapshot`` was sent: one text for its whole session."""
    (model,) = (
        [model for model in models.built if model.sessions and "editor-1" in model.sessions]
        if snapshot == _snapshot_id(KAIRO)
        else [model for model in models.built if model.sessions and "editor-1" not in model.sessions]
    )
    (system,) = {request["system"] for request in model.sessions["analyst"].requests}
    return system


async def test_two_arms_differing_by_one_line_of_skill_produce_results_keyed_apart_with_their_measures() -> None:
    host, candidate = off_loop(_host), _candidate()
    shipped_models, candidate_models = _Models(), _Models()
    shipped_run, candidate_run = _arm(load_skill(), effort="low"), _arm(candidate, effort="low")

    await _execute(host, shipped_run, shipped_models, skills={})
    await _execute(host, candidate_run, candidate_models, skills={candidate.digest: candidate})

    shipped, tried = _results(host, shipped_run), _results(host, candidate_run)
    for results in (shipped, tried):
        assert [result.test_case_id for result in results] == ["writer-kairo round 2", "writer-r1"]
        kairo, r1 = results
        assert kairo.termination == "completed", kairo.runner_error
        assert kairo.host_measures[ENDING_MEASURE] == "stored"
        assert kairo.host_measures[FINISHED_MEASURE] is True
        assert kairo.host_measures[MUST_FIX_OPEN_MEASURE] == 0.0
        assert kairo.host_measures[SHOULD_FIX_OPEN_MEASURE] == 1.0
        assert kairo.host_measures[TURNS_MEASURE] == 5.0
        assert kairo.cost_usd == pytest.approx(0.05), "each scripted turn is priced at a cent"
        # r1 stored nothing: its ending and counts land, and no finding is reported open on a report that does not exist.
        assert r1.host_measures[ENDING_MEASURE] == "refused"
        assert r1.host_measures[FINISHED_MEASURE] is False
        assert MUST_FIX_OPEN_MEASURE not in r1.host_measures
        assert r1.host_measures[TURNS_MEASURE] == 4.0
    # Every measure is the type its registry declares: a boolean is a bool, a count a float, an ending text.
    declared = {descriptor.name: descriptor.data_type for descriptor in WRITER_KIND_MEASURES}
    for result in (*shipped, *tried):
        for name, value in result.host_measures.items():
            expected = {"boolean": bool, "numeric": float, "categorical": str}[declared[name]]
            assert type(value) is expected, (name, value)

    profile = host.eval_host.profile
    shipped_key = resolve_variant_identity(run=shipped_run, profile=profile)
    candidate_key = resolve_variant_identity(run=candidate_run, profile=profile)
    assert shipped_key.variant_key != candidate_key.variant_key
    assert shipped_key.levers[SKILL_LEVER] != candidate_key.levers[SKILL_LEVER]
    assert {result.variant_key for result in shipped} == {shipped_key.variant_key}
    assert {result.variant_key for result in tried} == {candidate_key.variant_key}

    # Each arm wrote with its own skill, and every cell's model was built at the arm's effort.
    assert _CANDIDATE_RULE in _analyst_system(candidate_models, _snapshot_id(KAIRO))
    assert _CANDIDATE_RULE not in _analyst_system(shipped_models, _snapshot_id(KAIRO))
    assert {model.effort for model in (*candidate_models.built, *shipped_models.built)} == {"low"}


async def test_a_cases_withheld_note_never_reaches_the_analyst_and_the_cell_stores_no_report_or_gap() -> None:
    host, models = off_loop(_host), _Models()

    run = _arm(load_skill())

    await _execute(host, run, models, skills={})

    kairo, r1 = _results(host, run)
    assert kairo.candidate_error is None
    assert "the writing ended refused with no report" in (r1.candidate_error or ""), "r1's analyst never submitted"
    kairo_system, r1_system = _analyst_system(models, _snapshot_id(KAIRO)), _analyst_system(models, _snapshot_id(_R1))
    assert f'<file path="exemplars/{_KAIRO_NOTE}">' not in kairo_system and f"`{_KAIRO_NOTE}`" not in kairo_system
    assert f'<file path="exemplars/{_R1_NOTE}">' in kairo_system, "only the case's own note is withheld"
    assert (
        f'<file path="exemplars/{_R1_NOTE}">' not in r1_system and f'<file path="exemplars/{_KAIRO_NOTE}">' in r1_system
    )
    storage = host.eval_host.storage
    assert off_loop(storage.list_report_docs, CALLER.scope_id) == []
    assert off_loop(lambda: storage.list_analyst_gaps(CALLER.scope_id, limit=10)) == ([], False)


async def test_an_arm_whose_skill_the_launch_was_not_given_is_refused_before_any_turn() -> None:
    host, models, candidate = off_loop(_host), _Models(), _candidate()

    run = _arm(candidate)

    with pytest.raises(NothingMeasuredError, match="which this launch was not given"):
        await _execute(host, run, models, skills={})

    results = _results(host, run)
    assert len(results) == 2 and {result.termination for result in results} == {"factory_failed"}
    assert all(
        f"the arm runs skill {candidate.digest}, which this launch was not given" in result.runner_error
        for result in results
    )
    assert models.built == [], "no model was built, so no turn was sent"


async def test_an_arm_whose_subject_carries_another_skill_than_its_lever_is_refused_before_any_turn() -> None:
    host, models, candidate = off_loop(_host), _Models(), _candidate()
    run = _arm(candidate)

    with pytest.raises(LeverCoordinateError, match=SKILL_LEVER):
        await _execute(host, run, models, skills={candidate.digest: candidate}, subject_skill=load_skill())

    assert _results(host, run) == []
    assert models.built == []


async def test_a_case_withholding_a_note_the_skill_lacks_is_excluded_before_any_turn() -> None:
    host, models = off_loop(_host), _Models()
    files = {path: text for path, text in load_skill().files.items() if path != f"exemplars/{_R1_NOTE}"}
    files["exemplars/INDEX.md"] = "\n".join(
        line for line in files["exemplars/INDEX.md"].splitlines() if f"`{_R1_NOTE}`" not in line
    )
    lacking = AnalystSkill(files=files)

    run = _arm(lacking)

    await _execute(host, run, models, skills={lacking.digest: lacking})

    r1 = next(result for result in _results(host, run) if result.test_case_id == "writer-r1")
    assert r1.candidate_error is None, "the case's fault is the harness's, never the candidate's"
    assert f"withholds a note skill {lacking.digest} lacks" in (r1.infra_error or "")
    assert f"no exemplar is named {_R1_NOTE}" in (r1.infra_error or "")
    routed = [model for model in models.built if model.sessions and "editor-1" not in model.sessions]
    assert routed == [], "the r1 cell opened no session"


def test_the_effort_lever_names_the_words_the_model_client_takes() -> None:
    assert get_args(EffortWord) == EFFORT_LADDER


def test_an_arm_naming_no_skill_records_the_shipped_skills_digest() -> None:
    assert WriterOverlays().skill == load_skill().digest
    assert freeze(WRITER_KIND_CONTRACT.validate_overlays({}))["skill"] == load_skill().digest
