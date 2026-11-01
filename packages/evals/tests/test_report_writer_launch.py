"""The report writer's launcher: a launch over a writer case set, each cell under the writer's own deadline.

These tests register :func:`~threetears.evals.actions.report_writing.launchable_report_writer_kind` on the toy host,
seed its writer bank from the package fixtures, and launch through :func:`~threetears.evals.run.start_run`, as a host
does. They hold that an arm over a named set runs exactly that set's cases, scored by the judge the launch names,
with every cell under the deadline the writer's limits and the host's request ceiling derive (far past the engine's
600 s default, which a real-speed report outlasts); and that a launch is refused before anything is priced when its
template gives the judge nothing to score, it names no judge, its judge is of the writer's family, its arm's skill
was not given, or its set lists a retired case. A model the host cannot place is launched, and the run records that
the family check was not made.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from packages.evals.tests.fixtures.toyhost.host import with_kinds
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER
from packages.evals.tests.report_writer_support import MODEL, FakeToolTurnClient, reply
from packages.evals.tests.writer_case_support import WRITER_TEMPLATE, seed_writer_cases, writer_host
from threetears.evals.actions.report_writing import launchable_report_writer_kind
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.ops import OpsHost
from threetears.evals.ops.report_writer.loop import WriterLimits
from threetears.evals.ops.report_writer.tally import ReportBudget
from threetears.evals.ops.report_writer.writer_bank import WriterSplit, writer_case_archive, writer_split
from threetears.evals.ops.report_writer.writer_contract import WRITER_KIND, EffortWord, writer_scoring_rubric
from threetears.evals.ops.report_writer.writer_kind import writer_cell_timeout_s
from threetears.evals.run import JUDGE_CALL_ATTEMPTS, JUDGE_MAX_TOKENS, start_run
from threetears.evals.run.runner import DEFAULT_CELL_TIMEOUT_S
from threetears.evals.schema.models import CaseSetRef, EvalRun
from threetears.evals.schema.turns import ToolTurnClient, TurnText

#: The scoring judge every launch names unless a test says otherwise, of another family than the writer.
_JUDGE = "google/gemini-flash"

#: Each model's family as the host places it; a model absent here is one it cannot place.
_FAMILIES = {MODEL: "anthropic", _JUDGE: "google", "anthropic/claude-opus": "anthropic"}

#: The host's ceiling for one request: every request may take half a minute, whatever its output cap.
_REQUEST_S = 30.0

#: Each report's budget: dollars, and a turn cap that bounds the report's wall clock.
_BUDGET = ReportBudget(usd=5.0, turns=40)

#: The writer template as a host creates it, its rubric seeded from the shipped skill.
_SCORED = WRITER_TEMPLATE.model_copy(update={"rubric": writer_scoring_rubric()})


# parity-with: threetears.evals.ops.report_writer.loop.TurnModel
@dataclass
class _FakeSilentModel:
    """One cell's model: an analyst whose every reply holds no document, so the report is refused before any review
    and the judge is asked nothing."""

    model_name: str = MODEL
    sessions: dict[str, FakeToolTurnClient] = field(default_factory=dict)

    async def turn_pricing_refusal(self) -> str | None:
        return None

    def turn_session(self, name: str, *, snapshot: str) -> ToolTurnClient:
        client = FakeToolTurnClient(*(reply(TurnText("I have no report yet.")) for _ in range(4)))
        self.sessions[name] = client
        return client


def _models(model: str) -> Any:
    """The host's model factory for an arm's candidate model: a fresh silent model per cell, at any effort."""

    def at(effort: EffortWord | None) -> _FakeSilentModel:
        return _FakeSilentModel(model_name=model)

    return at


def _launching(*, skills: dict[str, AnalystSkill] | None = None) -> OpsHost:
    """The toy host with the writer registered, its template scored, its caps unenforced (a scripted turn costs a
    cent)."""
    fixture = writer_host(settings=TOYHOST_LAUNCH_SETTINGS.model_copy(update={"enforcement_enabled": False}))
    fixture.host.eval_host.storage.save_template(_SCORED)
    holder: list[OpsHost] = []
    entry = launchable_report_writer_kind(
        lambda: holder[0],
        identity="report-writer-campaign",
        models=_models,
        budget=_BUDGET,
        request_s=lambda _max_tokens: _REQUEST_S,
        model_family=_FAMILIES.get,
        skills=skills or {},
    )
    host = OpsHost(launch=with_kinds(fixture.host.launch, {WRITER_KIND: entry}))
    holder.append(host)
    return host


async def _seeded(host: OpsHost) -> CaseSetRef:
    """Seed the bank, and name its held-out set at its latest version."""
    await seed_writer_cases(host)
    sets = off_loop(writer_split, host, _SCORED, CALLER.scope_id, split=WriterSplit())
    return sets.held_out[0].ref


async def _launch(host: OpsHost, case_set: CaseSetRef, **arguments: Any) -> EvalRun:
    """Launch one arm on :data:`MODEL` over ``case_set`` and wait for it; the run as stored."""
    launched: dict[str, Any] = {"judge_model": _JUDGE, **arguments}
    (run,) = await start_run(
        host.launch,
        template_id=_SCORED.id,
        scope_id=CALLER.scope_id,
        subject_id="the report writer",
        models=[MODEL],
        k_runs=1,
        case_set=case_set,
        **launched,
    )
    async with asyncio.timeout(20):
        while host.launch.job_manager.is_active(run.id):
            await asyncio.sleep(0.01)
    stored = off_loop(host.eval_host.storage.load_eval_run, run.id, CALLER.scope_id)
    assert stored is not None
    return stored


async def test_a_launch_over_a_named_set_runs_its_cases_each_under_the_writers_own_deadline() -> None:
    host = off_loop(_launching)
    case_set = await _seeded(host)

    run = await _launch(host, case_set)

    expected = writer_cell_timeout_s(
        WriterLimits(),
        request_s=lambda _max_tokens: _REQUEST_S,
        turns=_BUDGET.turns,
        judge_dims=len(writer_scoring_rubric()),
        judge_concurrency=TOYHOST_LAUNCH_SETTINGS.judge_concurrency,
        judge_call_attempts=JUDGE_CALL_ATTEMPTS,
        judge_max_tokens=JUDGE_MAX_TOKENS,
    )
    assert (run.cell_timeout_s, run.cell_timeout_s_origin) == (expected, "kind")
    assert expected == 40 * _REQUEST_S + 7 * JUDGE_CALL_ATTEMPTS * _REQUEST_S, "the turn cap, then each dim's calls"
    assert expected > DEFAULT_CELL_TIMEOUT_S, "a real-speed report outlasts the engine's default"
    results = off_loop(host.eval_host.storage.query_eval_results_by_run, run.id, CALLER.scope_id)
    assert sorted(result.test_case_id for result in results) == sorted(run.test_case_ids)
    sets = off_loop(writer_split, host, _SCORED, CALLER.scope_id, split=WriterSplit())
    assert sorted(run.test_case_ids) == sorted(sets.held_out[0].test_case_ids)
    assert {result.termination for result in results} == {"completed"}
    assert run.judge_model == _JUDGE
    assert run.host_payload == {WRITER_KIND: {"judge_family": "writer anthropic, judge google"}}


async def test_a_judge_the_host_cannot_place_is_launched_and_the_run_records_the_check_was_not_made() -> None:
    host = off_loop(_launching)
    case_set = await _seeded(host)

    run = await _launch(host, case_set, judge_model="local/unplaced")

    assert run.host_payload == {
        WRITER_KIND: {"judge_family": "unchecked: the host places no family for 'local/unplaced'"}
    }


@pytest.mark.parametrize(
    ("arguments", "refused"),
    [
        ({"judge_model": "anthropic/claude-opus"}, "of the writer 'report-writer-model''s family \\(anthropic\\)"),
        ({"judge_model": None}, "this launch names none"),
        ({"overlays": {"skill": "f" * 64}}, f"the arm runs skill {'f' * 64}, which this launch was not given"),
    ],
)
async def test_a_launch_the_writer_cannot_honour_is_refused_before_any_run(
    arguments: dict[str, Any], refused: str
) -> None:
    host = off_loop(_launching)
    case_set = await _seeded(host)
    before = off_loop(host.eval_host.storage.query_eval_runs, CALLER.scope_id)

    with pytest.raises(ValidationFailedError, match=refused):
        await _launch(host, case_set, **arguments)

    assert off_loop(host.eval_host.storage.query_eval_runs, CALLER.scope_id) == before, "no run was stored"


async def test_a_template_giving_the_judge_nothing_to_score_is_refused_at_launch() -> None:
    host = off_loop(_launching)
    case_set = await _seeded(host)
    off_loop(host.eval_host.storage.save_template, WRITER_TEMPLATE)

    with pytest.raises(ValidationFailedError, match="no rubric"):
        await _launch(host, case_set)


async def test_a_set_listing_a_retired_case_is_refused_at_launch() -> None:
    host = off_loop(_launching)
    case_set = await _seeded(host)
    sets = off_loop(writer_split, host, _SCORED, CALLER.scope_id, split=WriterSplit())
    retired = sets.held_out[0].test_case_ids[0]
    await writer_case_archive(host, retired, CALLER.scope_id, archived=True, reason="its campaign was re-run")

    with pytest.raises(ValidationFailedError, match=f"'{retired}' \\(retired\\)"):
        await _launch(host, case_set)


async def test_an_arm_on_a_skill_the_launch_was_given_runs_it() -> None:
    files = dict(load_skill().files)
    files["SKILL.md"] = f"{files['SKILL.md']}\n\nCandidate rule.\n"
    candidate = AnalystSkill(files=files)
    host = off_loop(_launching, skills={candidate.digest: candidate})
    case_set = await _seeded(host)

    run = await _launch(host, case_set, overlays={"skill": candidate.digest})

    assert run.overlays["skill"] == candidate.digest
    results = off_loop(host.eval_host.storage.query_eval_results_by_run, run.id, CALLER.scope_id)
    assert {result.termination for result in results} == {"completed"}
    assert {result.kind_payload["skill"] for result in results if result.kind_payload} == {candidate.digest}
