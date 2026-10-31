"""The report writer as a candidate kind: two arms that differ by one line of the skill, run as the engine runs any.

A run of the ``report_writer`` kind writes a report over each frozen writer case with the arm's skill, effort and
revisions, on scripted turns, and lands the run's own measures on each result. These tests drive two arms (the shipped
skill and a candidate with one line added) through :func:`~threetears.evals.run.runner.execute_run` over two cases on a
host whose store holds neither case's snapshot: each cell restores its case's own. They hold that both arms produce
results, keyed apart by the skill lever; that the measures land typed as their registry declares them; that a case's
withheld note never reaches the analyst; and that a skill the launch was not given, or a withheld note the skill
lacks, is refused before any turn is sent.

The pages are scored by a fixed judge on the editor rubric's seven items, seeded into the template from the shipped
skill: every stored page carries the seven judged dimensions beside the run's measures, and an arm whose own
``editor.md`` rewords an item still has its page scored on the shipped wording. A template giving the judge nothing to
score, and a campaign whose arms were scored by another judge or rubric, are refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import re
from types import SimpleNamespace
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from packages.evals.tests.analyst_support import KAIRO, fixture_snapshots
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, ops_fixture
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.report_writer_support import (
    MODEL,
    FakeToolTurnClient,
    FakeTurnModel,
    call,
    reply,
    review,
    submitted,
)
from threetears.evals.actions.report_writing import report_writer_kind
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.analysis.judge_profiles import judges_tiers
from threetears.evals.analysis.reads import bisect_runs
from threetears.evals.kernel.candidate_kind import CandidatePreparationFailed, VariantConfig
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.host.kinds import freeze
from threetears.evals.kernel.host.measures import MeasureRegistry
from threetears.evals.kernel.identity import LeverCoordinateError, resolve_variant_identity
from threetears.evals.kernel.provider import withhold_failure_detail
from threetears.evals.negotiation.settings import EFFORT_LADDER
from threetears.evals.ops import OpsHost
from threetears.evals.ops.report_writer.rubric import shipped_rubric
from threetears.evals.ops.report_writer.loop import MOST_REVISIONS, WriterLimits
from threetears.evals.ops.report_writer.tally import BounceKind, ReportBudget, Tally
from threetears.evals.ops.report_writer.writer_contract import (
    BOUNCE_MEASURES,
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
    refuse_undeliverable_writer_template,
    refuse_unfixed_scoring_judge,
    writer_kind_contract,
    writer_scoring_rubric,
)
from threetears.evals.ops.report_writer.writer_kind import (
    WRITER_CASE_KEY,
    WriterCase,
    writer_case_payload,
    writer_subject_components,
)
from threetears.evals.run.definition_digests import measured_template_digest
from threetears.evals.run.judge_service import JudgeService
from threetears.evals.run.budget import AccountExhaustedError
from threetears.evals.run.runner import NothingMeasuredError, RunnerOptions, execute_run
from threetears.evals.schema.completion import FailureOwner, ProviderFailure
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


def _arm(skill: AnalystSkill, *, effort: EffortWord | None = None, judge_model: str | None = None) -> EvalRun:
    """One arm's run, given one revision: its overlays frozen as a launch freezes them, its subject carrying its
    skill, judged by ``judge_model`` when one is named."""
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
        judge_model=judge_model,
        # A judged arm runs the scored template, whose digest its launch freezes.
        template_digest=None if judge_model is None else measured_template_digest(_SCORED_TEMPLATE),
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
    template: EvalTemplate = _TEMPLATE,
    judge_service: JudgeService | None = None,
    cases: list[EvalTestCase] | None = None,
    overlays: WriterOverlays | None = None,
) -> EvalRun:
    """Run one arm over both cases, or over ``cases``; the run, as it was executed."""
    overlays = overlays if overlays is not None else WriterOverlays.model_validate(run.overlays)
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
        template=template,
        test_cases=cases if cases is not None else _cases(),
        judge_service=judge_service,
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


# =============================================================================
# The fixed scoring judge: the editor rubric's seven items, seeded from the shipped skill
# =============================================================================

#: The shipped wording of the rubric's first item, and the wording an arm's own editor.md gives it instead.
_SHIPPED_FIRST_SCREEN = "From the headline, decision block and first claim alone"
_EDITED_FIRST_SCREEN = "From the headline and lede alone"

#: The judge pin every scored arm records.
_JUDGE = "judge/flash"

#: The writer template as a host creates it: its rubric seeded from the shipped skill.
_SCORED_TEMPLATE = _TEMPLATE.model_copy(update={"rubric": writer_scoring_rubric()})


def _edited_editor() -> AnalystSkill:
    """The shipped skill with the editor rubric's first item reworded in its own editor.md."""
    files = dict(load_skill().files)
    assert _SHIPPED_FIRST_SCREEN in files["editor.md"]
    files["editor.md"] = files["editor.md"].replace(_SHIPPED_FIRST_SCREEN, _EDITED_FIRST_SCREEN)
    return AnalystSkill(files=files)


class _RecordingJudge:
    """A judge client that records every prompt it is sent and scores the dim its system prompt names 4, answering as
    ``model``."""

    def __init__(self, model: str = _JUDGE) -> None:
        self.model = model
        self.calls: list[tuple[str, str]] = []

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
        self.calls.append((system, user))
        match = re.search(r'the single key "(.+?)"', system)
        dim_id = match.group(1) if match else "?"
        return SimpleNamespace(
            temperature=None,
            served_model=self.model,
            stop_reason="end_turn",
            content=json.dumps({"reasoning": f"because {dim_id}", "criteria_scores": {dim_id: 4}}),
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
            model=self.model,
        )


def _scoring(judge: _RecordingJudge) -> JudgeService:
    return JudgeService(
        client_factory=lambda model, temperature: judge, configs={}, failure_describer=withhold_failure_detail
    )


def test_the_scoring_rubric_is_the_editor_rubrics_seven_items_in_the_shipped_wording() -> None:
    rubric, items = writer_scoring_rubric(), shipped_rubric().items

    assert len(rubric) == len(items) == 7
    assert [dim.name for dim in rubric][:2] == [f"{WRITER_KIND}.first_screen", f"{WRITER_KIND}.coherent_story"]
    for dim, item in zip(rubric, items, strict=True):
        assert dim.description == f"{item.name}. {item.definition}"
        assert dim.scale == "ordinal" and dim.judge == "builtin"
        assert dim.scoring_guide["5"] == "a reader is well served"
    # The item names the editor's answer is checked against come from the same text.
    assert "Evidence fits claims" in load_skill().editor().text and items[6].name == "Evidence fits claims"


async def test_a_scripted_campaign_scores_every_stored_page_on_all_seven_dims_in_the_shipped_wording() -> None:
    host, edited = off_loop(_host), _edited_editor()
    shipped_judge, edited_judge = _RecordingJudge(), _RecordingJudge()
    shipped_models, edited_models = _Models(), _Models()
    shipped_run, edited_run = _arm(load_skill(), judge_model=_JUDGE), _arm(edited, judge_model=_JUDGE)

    await _execute(
        host, shipped_run, shipped_models, skills={}, template=_SCORED_TEMPLATE, judge_service=_scoring(shipped_judge)
    )
    await _execute(
        host,
        edited_run,
        edited_models,
        skills={edited.digest: edited},
        template=_SCORED_TEMPLATE,
        judge_service=_scoring(edited_judge),
    )

    dims = [dim.name for dim in writer_scoring_rubric()]
    for run in (shipped_run, edited_run):
        kairo, r1 = _results(host, run)
        assert [score.dim for score in kairo.rubric_scores] == dims, kairo.runner_error
        assert {score.score for score in kairo.rubric_scores} == {4}
        # The judged dims sit beside the run's own measures, which land as before.
        assert kairo.host_measures[FINISHED_MEASURE] is True and kairo.host_measures[TURNS_MEASURE] == 5.0
        assert r1.rubric_scores == [], "r1 stored no page, so there is nothing to judge"
    # The arm's editor reviewed with its own wording; the judge asked the shipped wording of both arms' pages.
    (editor_system,) = {
        request["system"]
        for model in edited_models.built
        if "editor-1" in model.sessions
        for request in model.sessions["editor-1"].requests
    }
    assert _EDITED_FIRST_SCREEN in editor_system and _SHIPPED_FIRST_SCREEN not in editor_system
    for judge in (shipped_judge, edited_judge):
        assert len(judge.calls) == 7
        asked = "\n".join(system + user for system, user in judge.calls)
        assert _SHIPPED_FIRST_SCREEN in asked and _EDITED_FIRST_SCREEN not in asked
        assert all("Computed facts, from the snapshot" in user for _, user in judge.calls)
    # Nobody has rated a page, so each dim's judge reads as uncalibrated: undetermined, with no rating behind it.
    results = [*_results(host, shipped_run), *_results(host, edited_run)]
    tiers = off_loop(
        lambda: judges_tiers(
            host.eval_host.storage,
            CALLER.scope_id,
            [shipped_run, edited_run],
            results,
            templates={_SCORED_TEMPLATE.id: _SCORED_TEMPLATE},
        )
    )
    assert sorted(tier.rubric_dim for tier in tiers) == sorted(dims)
    assert {tier.tier for tier in tiers} == {"undetermined"}
    assert {tier.calibration.n for tier in tiers} == {0}
    # One judge on one rubric scored both arms, so the campaign may be read as a comparison of arms.
    refuse_unfixed_scoring_judge([shipped_run, edited_run])


def test_a_writer_template_whose_judge_has_nothing_to_score_is_refused() -> None:
    refuse_undeliverable_writer_template(_SCORED_TEMPLATE)

    with pytest.raises(ValidationFailedError, match="no rubric"):
        refuse_undeliverable_writer_template(_TEMPLATE)
    asking = _SCORED_TEMPLATE.model_copy(update={"tools_required": ["search"], "preconditions": ["a booked table"]})
    with pytest.raises(ValidationFailedError, match="preconditions.*tools"):
        refuse_undeliverable_writer_template(asking)


def _scored(run_id: str, **overrides: Any) -> EvalRun:
    """A writer arm's run as a launch records its judge and what its template measured."""
    fields: dict[str, Any] = {
        "id": run_id,
        "scope_id": CALLER.scope_id,
        "template_id": _SCORED_TEMPLATE.id,
        "template_digest": "a" * 64,
        "candidate_kind": WRITER_KIND,
        "judge_model": _JUDGE,
    }
    fields.update(overrides)
    return make_eval_run(**fields)


def test_a_campaign_scored_by_one_judge_on_one_rubric_is_admitted() -> None:
    refuse_unfixed_scoring_judge([_scored("arm-a"), _scored("arm-b")])


@pytest.mark.parametrize(
    ("moved", "named"),
    [
        ({"judge_model": "judge/other"}, "2 judges: model judge/flash"),
        ({"judge_temperature": 0.7}, "2 judges"),
        ({"judge_config_ids": {f"{WRITER_KIND}.first_screen": "cfg-2"}}, "2 judges"),
        ({"template_digest": "b" * 64}, "2 rubric revisions"),
        ({"template_id": "another-writer-template"}, "2 templates"),
        ({"template_digest": None}, "run arm-b recorded no template digest"),
        ({"judge_model": None}, "run arm-b recorded no judge"),
    ],
)
def test_a_campaign_whose_arms_were_not_scored_alike_is_refused(moved: dict[str, Any], named: str) -> None:
    with pytest.raises(ValidationFailedError, match=named):
        refuse_unfixed_scoring_judge([_scored("arm-a"), _scored("arm-b", **moved)])


async def test_the_engine_reads_a_judge_change_between_writer_arms_as_a_difference_of_the_rig() -> None:
    host = off_loop(_host)
    storage = host.eval_host.storage
    runs = [_arm(load_skill(), judge_model=_JUDGE), _arm(load_skill(), judge_model="judge/other")]
    for run in runs:
        off_loop(storage.save_eval_run, run)
        await _execute(
            host,
            run,
            _Models(),
            skills={},
            template=_SCORED_TEMPLATE,
            judge_service=_scoring(_RecordingJudge(run.judge_model or _JUDGE)),
        )
    a, b = runs

    bisected = off_loop(bisect_runs, storage, a.id, b.id, CALLER.scope_id, profile=host.eval_host.profile)

    # The writer's contract seats the judge, so the engine reports its change; the campaign read refuses it.
    assert "judge_model" in bisected["differs"], bisected["unknown"]
    with pytest.raises(ValidationFailedError, match="2 judges"):
        refuse_unfixed_scoring_judge(runs)


# =============================================================================
# Every bounce kind is a measure; the revisions cap has one source
# =============================================================================


async def test_every_bounce_kind_lands_as_a_count_and_r1s_unparsed_replies_are_counted() -> None:
    host, models = off_loop(_host), _Models()
    run = _arm(load_skill())

    await _execute(host, run, models, skills={})

    kairo, r1 = _results(host, run)
    for result in (kairo, r1):
        assert {name for name in BOUNCE_MEASURES.values()} <= set(result.host_measures), result.host_measures
    # r1's analyst answered with no document four times: three bounces as unparsed, then refused.
    assert r1.host_measures[BOUNCE_MEASURES["unparsed"]] == 3.0
    assert {r1.host_measures[BOUNCE_MEASURES[kind]] for kind in get_args(BounceKind) if kind != "unparsed"} == {0.0}
    assert {kairo.host_measures[name] for name in BOUNCE_MEASURES.values()} == {0.0}


def test_each_bounce_kind_the_analyst_records_has_a_declared_measure() -> None:
    assert tuple(BOUNCE_MEASURES) == get_args(BounceKind)
    assert set(BOUNCE_MEASURES.values()) <= {descriptor.name for descriptor in WRITER_KIND_MEASURES}
    tally = Tally()
    tally.record_bounce("analyst", 1, "shape")
    tally.record_bounce("analyst", 2, "shape")
    tally.record_bounce("analyst", 2, "max_tokens")
    assert tally.bounces() == {"unparsed": 0, "shape": 2, "validity": 0, "invariant": 0, "max_tokens": 1}


def test_an_arms_revisions_are_bounded_where_the_writing_loop_bounds_them() -> None:
    assert WriterOverlays().revisions == WriterLimits().revisions == MOST_REVISIONS
    WriterLimits(revisions=MOST_REVISIONS)
    with pytest.raises(ValidationError, match="less than or equal to"):
        WriterOverlays(revisions=MOST_REVISIONS + 1)


# =============================================================================
# prepare's refusals, invoke's exclusions, and a failed turn's routing
# =============================================================================


@dataclass
class _ScriptedFactory:
    """A model factory whose cells' models are built as a test says: renamed, unpriced, refusing, or failing."""

    name: str = MODEL
    unpriced: str | None = None
    refuses: str | None = None
    fails: Exception | None = None
    built: list[FakeTurnModel] = field(default_factory=list)

    def __call__(self, effort: EffortWord | None) -> FakeTurnModel:
        if self.refuses is not None:
            raise ValueError(self.refuses)
        failing = FakeToolTurnClient(*([self.fails] if self.fails is not None else []))
        model = FakeTurnModel(unpriced=self.unpriced, analyst=failing)
        model.model_name = self.name
        self.built.append(model)
        return model


@pytest.mark.parametrize(
    ("factory", "refused"),
    [
        (
            _ScriptedFactory(name="another-model"),
            f"recorded against model {MODEL!r} but the model factory built 'another-model'",
        ),
        (
            _ScriptedFactory(refuses="no route takes effort 'max'"),
            "the host's model factory refused the arm's settings: no route takes effort 'max'",
        ),
        (
            _ScriptedFactory(unpriced="report-writer-model has no rate"),
            "budget admits each turn on its price ceiling, and report-writer-model has no rate",
        ),
    ],
)
async def test_a_cell_whose_model_would_not_be_the_arms_or_could_not_be_priced_is_refused_before_any_turn(
    factory: _ScriptedFactory, refused: str
) -> None:
    host = off_loop(_host)
    run = _arm(load_skill())

    with pytest.raises(NothingMeasuredError, match=re.escape(refused)):
        await _execute(host, run, factory, skills={})  # type: ignore[arg-type]

    results = _results(host, run)
    assert {result.termination for result in results} == {"factory_failed"}
    assert all(refused in (result.runner_error or "") for result in results)
    assert all(not model.asked for model in factory.built), "no session was opened"


async def test_a_cell_whose_subject_carries_no_skill_is_refused_by_the_kind() -> None:
    host, models = off_loop(_host), _Models()
    kind = report_writer_kind(host, CALLER, overlays=WriterOverlays(), models=models, budget=ReportBudget(usd=5.0))
    bare = SubjectSnapshot(subject_id="report writer", subject_label="report writer", components={}, state={})

    with pytest.raises(CandidatePreparationFailed, match="its subject carries no skill"):
        await kind.prepare(
            subject_snapshot=bare,
            variant_config=VariantConfig(candidate_model=MODEL),
            world_seed=None,
            span_window=None,
            cassettes=None,
            world=None,
        )
    assert models.built == []


async def test_a_cell_whose_revisions_the_loop_does_not_run_is_refused_before_any_turn() -> None:
    host, models = off_loop(_host), _Models()
    beyond = WriterOverlays.model_construct(
        skill=load_skill().digest, reasoning_effort=None, revisions=MOST_REVISIONS + 1
    )
    kind = report_writer_kind(host, CALLER, overlays=beyond, models=models, budget=ReportBudget(usd=5.0))

    with pytest.raises(CandidatePreparationFailed, match="not ones the writing loop runs"):
        await kind.prepare(
            subject_snapshot=_arm(load_skill()).subject_snapshot,
            variant_config=VariantConfig(candidate_model=MODEL),
            world_seed=None,
            span_window=None,
            cassettes=None,
            world=None,
        )
    assert models.built == []


def _case(case_id: str, payload: dict[str, Any]) -> EvalTestCase:
    return EvalTestCase(id=case_id, template_id=_TEMPLATE.id, scope_id=CALLER.scope_id, host_payload=payload)


@pytest.mark.parametrize(
    ("payload", "excluded"),
    [
        ({}, "carries no writer case under host_payload['writer_case']"),
        ({WRITER_CASE_KEY: {"snapshot": "not a document"}}, "carries an unreadable writer case"),
        (
            {WRITER_CASE_KEY: {"snapshot": {"id": "report-snapshot:none"}, "withhold": []}},
            "frozen snapshot does not read under this build",
        ),
    ],
)
async def test_a_case_the_cell_cannot_write_over_is_excluded_as_the_harnesss_before_any_turn(
    payload: dict[str, Any], excluded: str
) -> None:
    host, models = off_loop(_host), _Models()
    run = _arm(load_skill()).model_copy(update={"test_case_ids": ["writer-broken"]})

    with pytest.raises(NothingMeasuredError):
        await _execute(host, run, models, skills={}, cases=[_case("writer-broken", payload)])

    (result,) = _results(host, run)
    assert result.candidate_error is None, "the case's fault is the harness's, never the candidate's"
    assert excluded in (result.infra_error or ""), result.infra_error
    assert all(not model.sessions for model in models.built), "no session was opened"


class _Overloaded(Exception):
    """What the host's client raised: a provider's answer the describer reads."""


def _describing(owner: FailureOwner) -> OpsHost:
    """The writer's host, its describer reading every failure as ``owner``'s."""
    host = _host()
    eval_host = replace(
        host.eval_host,
        failure_describer=lambda exc: ProviderFailure(f"{owner} said no", payload_withheld=False, owner=owner),
    )
    return replace(host, launch=replace(host.launch, eval_host=eval_host))


@pytest.mark.parametrize(
    ("owner", "infra", "candidate", "refused"),
    [
        (
            "account",
            "apparatus: report_writer: a turn was refused for the calling account: account said no",
            None,
            True,
        ),
        ("provider", "apparatus: report_writer: a turn was declined by the provider: provider said no", None, False),
        ("candidate", None, "report_writer: a turn failed: candidate said no", False),
    ],
)
async def test_a_turn_whose_client_failed_is_routed_by_whose_failure_the_host_says_it_was(
    owner: FailureOwner, infra: str | None, candidate: str | None, refused: bool
) -> None:
    host = off_loop(_describing, owner)
    factory = _ScriptedFactory(fails=_Overloaded("overloaded"))
    run = _arm(load_skill()).model_copy(update={"test_case_ids": ["writer-kairo round 2"]})

    stopped: type[Exception] | None = None
    try:
        await _execute(host, run, factory, skills={}, cases=_cases()[:1])  # type: ignore[arg-type]
    except (AccountExhaustedError, NothingMeasuredError) as raised:
        stopped = type(raised)

    (result,) = _results(host, run)
    assert (result.infra_error, result.candidate_error) == (infra, candidate)
    assert result.host_measures[ENDING_MEASURE] == "failed"
    assert (stopped is AccountExhaustedError) is refused, "only the account's refusal stops the run"
