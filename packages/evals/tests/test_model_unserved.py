"""A candidate model the provider no longer serves is lost to the test setup, never judged, and stops its arm.

Pinned:

- **Classification**: the package's client reads "model not served" off each provider's structured answer (OpenRouter's
  404 and its 400 for a model id it does not know, Anthropic's 404 ``not_found_error``, OpenAI's ``model_not_found``)
  as the ``unserved`` owner; a 500 stays the candidate's, a 503 the provider's.
- **One trial**: a cell whose model is not served is excluded with its own cause, quoting the provider, and never
  judged, even when the kind went on to record a transcript of turns with no reply, as a conversing kind does.
- **The circuit**: cells that keep finding the model not served stop the arm ``exhausted``, naming the model and the
  provider's words, while the launch's other arm runs to its end; a served cell in between starts the count again,
  and a run shorter than the circuit whose every cell is unserved stops all the same.
- **Resume**: the stopped arm, taken up once the model is served, runs its unserved cells again and completes.
- **Other candidate errors stay the arm's**: a 500 is charged to the candidate, and the run completes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from threetears.evals import Eval
from threetears.evals.analysis import report_markdown
from threetears.evals.kernel import EvalStorage
from threetears.evals.kernel.candidate_kind import CandidateOutput, CellSink
from threetears.evals.kernel.failed_call import failed_call_output
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.result_condition import ResultOutcome, classify_result, trial_exclusion
from threetears.evals.ops import LaunchArguments, OpsHost, run_launch, run_job_id
from threetears.evals.providers import ProviderError, describe_failure
from threetears.evals.providers.failures import error_from_response
from threetears.evals.run import ModelUnservedError, RunnerOptions, execute_run, resumable_jobs, resume_run
from threetears.evals.run.judge import JUDGE_REQUEST_SETTINGS
from threetears.evals.run.launch import build_judge_service
from threetears.evals.schema import EvalResult, EvalRun
from threetears.evals.schema.models import (
    DEFAULT_JUDGE_TEMPERATURE,
    EvalTemplate,
    EvalTestCase,
    JudgedArtifact,
    JudgeEvidence,
)
from packages.evals.tests.clock_support import frozen_clock
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.judge import TOY_JUDGE_MODEL, ToyJudgeCompletion, toyhost_judged_template
from packages.evals.tests.fixtures.toyhost.kind import (
    TOY_EXTRACTOR_KIND,
    ExtractionResult,
    ScriptedExtractionClient,
    ToyExtractorKind,
)
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_run, toyhost_template, toyhost_test_cases
from packages.evals.tests.lease_support import ClockedStore, launched_run, other_process
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE, TOYHOST_SUBJECT, launched_runs, ops_fixture, settled

log = logging.getLogger(__name__)

#: The model the provider retired, and the one it still serves.
RETIRED, SERVED = RUN_MODELS[0], RUN_MODELS[1]

#: What OpenRouter answers for a retired model, as Discodon's production run met it.
NO_LONGER_AVAILABLE = f"{RETIRED} is no longer available on OpenRouter"

#: How many cells in a row the circuit waits for before it stops a run, as ``execute_run`` documents it.
CIRCUIT = 3

#: Repeats per case in a launched arm: three cases, nine cells, more than the circuit's run.
K = 3


def _retired() -> ProviderError:
    """OpenRouter's answer to a call naming the retired model, as the package's client raises it."""
    return error_from_response("openrouter", 404, {"error": {"code": 404, "message": NO_LONGER_AVAILABLE}})


def _server_error() -> ProviderError:
    """An upstream's 500, which says nothing about whether the model is served."""
    return error_from_response("openrouter", 500, {"error": {"code": 500, "message": "Internal Server Error"}})


# =============================================================================
# Classification: the package's describer reads each provider's structured answer
# =============================================================================


@pytest.mark.parametrize(
    ("provider", "status", "body", "owner"),
    [
        ("openrouter", 404, {"error": {"code": 404, "message": NO_LONGER_AVAILABLE}}, "unserved"),
        ("openrouter", 400, {"error": {"code": 400, "message": "acme/nope is not a valid model ID"}}, "unserved"),
        (
            "openrouter",
            404,
            {"error": {"code": 404, "message": "image gone", "metadata": {"error_type": "image_not_found"}}},
            "candidate",
        ),
        ("openrouter", 400, {"error": {"code": 400, "message": "max_tokens is too large"}}, "candidate"),
        ("anthropic", 404, {"type": "error", "error": {"type": "not_found_error", "message": "model: x"}}, "unserved"),
        ("openai", 404, {"error": {"message": "The model `x` does not exist", "code": "model_not_found"}}, "unserved"),
        ("openrouter", 500, {"error": {"code": 500, "message": "Internal Server Error"}}, "candidate"),
        ("openrouter", 503, {"error": {"code": 503, "message": "no provider available"}}, "provider"),
        ("openrouter", 402, {"error": {"code": 402, "message": "insufficient credits"}}, "account"),
    ],
)
def test_the_package_describer_reads_model_not_served_off_the_providers_structured_answer(
    provider: Any, status: int, body: dict[str, Any], owner: str
) -> None:
    failure = describe_failure(error_from_response(provider, status, body))

    assert failure.owner == owner
    assert failure.payload_withheld is False


def test_an_unserved_failure_is_excluded_as_the_setups_and_a_candidate_failure_stays_charged() -> None:
    unserved = describe_failure(_retired())
    words: dict[str, Any] = {
        "refused": "turn: refused for the account",
        "declined": "turn: declined by the provider",
        "failed": "turn: the candidate's call failed",
        "unserved": "turn: named a model the provider does not serve",
        "log": lambda what, level: log.log(level, what),
    }

    lost = failed_call_output(unserved.owner, unserved.description, **words)
    charged = failed_call_output(describe_failure(_server_error()).owner, "openrouter 500", **words)

    assert lost.model_unserved and not lost.candidate_errors and not lost.account_refused
    assert lost.infra_errors == [
        f"apparatus: turn: named a model the provider does not serve: openrouter 404: {NO_LONGER_AVAILABLE}"
    ]
    assert charged.candidate_errors and not charged.infra_errors and not charged.model_unserved


def test_model_unserved_without_its_infra_entry_is_refused() -> None:
    with pytest.raises(ValueError, match="model_unserved is set but infra_errors is empty"):
        CandidateOutput(model_unserved=True)


# =============================================================================
# One trial, through the real runner, judged
# =============================================================================


# parity-with: threetears.evals.schema.completion.CompletionClient
class FakeJudgeClient:
    """A judge client that scores every dim a 4 and counts what it was asked."""

    def __init__(self) -> None:
        """No call yet."""
        self.calls: list[str] = []

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> ToyJudgeCompletion:
        """Score the dim the prompt names a 4."""
        self.calls.append(system)
        dim = system.split('the single key "', 1)[1].split('"', 1)[0]
        return ToyJudgeCompletion(
            content=json.dumps({"reasoning": "read", "criteria_scores": {dim: 4}}),
            input_tokens=100,
            output_tokens=40,
            cost_usd=0.0004,
            model=TOY_JUDGE_MODEL,
            price_source="toyhost_script",
        )

    async def aclose(self) -> None:
        """Nothing to release."""


def _conversing_invoke(raises: Callable[[], ProviderError], *, served_every: int = 0) -> Any:
    """An ``invoke`` that stands in for a host's conversing kind: its candidate's call raises, the kind routes the
    failure through the package's describer, and goes on to record the round's transcript with a turn that has no
    reply, rendered for the judge, as a conversation that carries on past a failed call does.

    :param raises: The error the candidate's call raises.
    :param served_every: When set, every ``served_every``-th call is served instead (the toy extraction runs).
    """
    original = ToyExtractorKind.invoke
    calls = [0]

    async def invoke(self: ToyExtractorKind, instance: Any, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        calls[0] += 1
        if served_every and calls[0] % served_every == 0:
            return await original(self, instance, test_case, sink)
        try:
            raise raises()
        except ProviderError as raised:
            failure = describe_failure(raised)
        routed = failed_call_output(
            failure.owner,
            failure.description,
            refused="turn 1: refused for the calling account",
            declined="turn 1: declined by the provider",
            failed="turn 1: the candidate's call failed",
            unserved="turn 1: the candidate's call named a model the provider does not serve",
            log=lambda what, level: log.log(level, "%s: %s", what, failure.description),
        )
        transcript = [
            {"speaker": "user", "text": "Please read invoice A."},
            {"speaker": "candidate", "text": "[no reply]"},
        ]
        return routed.model_copy(
            update={
                "output": transcript,
                "judge_evidence": JudgeEvidence(case_material="Invoice A", artifact="user: ...\ncandidate: [no reply]"),
            }
        )

    return invoke


def _judged_run(
    monkeypatch: pytest.MonkeyPatch, invoke: Any, *, n_cases: int = 1, k: int = 1
) -> tuple[EvalHost, EvalRun, EvalTemplate, list[EvalTestCase], Any, FakeJudgeClient]:
    """A judged toy run whose kind's ``invoke`` is ``invoke``, launch-shaped, unstarted.

    :return: The host, the run, its template and cases, the runner options, and the judge client.
    """
    monkeypatch.setattr(ToyExtractorKind, "invoke", invoke)
    judge = FakeJudgeClient()
    host = toyhost_host(clients=lambda _role, _model, *, temperature=None: judge)
    world = host.profile.world
    assert world is not None
    template = toyhost_judged_template()
    kind = ToyExtractorKind(
        client=ScriptedExtractionClient(), world=world, judged=True, goal_checks=tuple(template.goal_state_checks)
    )
    cases = toyhost_test_cases(template)[:n_cases]
    host.storage.save_template(template)
    for case in cases:
        host.storage.save_test_case(case)
    run_judge = build_judge_service(host, template, TOY_JUDGE_MODEL, judged_artifact=JudgedArtifact.DOCUMENT)
    run = toyhost_run(model=RETIRED, template=template, kind=kind, world=world).model_copy(
        update={
            "k_runs": k,
            "test_case_ids": [case.id for case in cases],
            "judge_model": TOY_JUDGE_MODEL,
            "effective_judges": run_judge.effective_judges,
            "judge_config_ids": {dim: config.id for dim, config in run_judge.configs.items()},
            "judge_asked_settings": JUDGE_REQUEST_SETTINGS,
            "judge_temperature": DEFAULT_JUDGE_TEMPERATURE,
        }
    )
    host.storage.save_eval_run(run)
    options = RunnerOptions(candidate_kinds={TOY_EXTRACTOR_KIND: lambda _cell: kind}, max_concurrent_cells=1)
    return host, run, template, cases, (run_judge.service, options), judge


async def _execute(host: EvalHost, run: EvalRun, template: EvalTemplate, cases: list[EvalTestCase], wired: Any) -> None:
    service, options = wired
    await execute_run(host, run=run, template=template, test_cases=cases, judge_service=service, options=options)


def _results(storage: EvalStorage, run: EvalRun) -> list[EvalResult]:
    return off_loop(storage.query_eval_results_by_run, run.id, run.scope_id)


async def test_a_trial_whose_model_is_not_served_is_excluded_with_its_cause_and_never_judged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, run, template, cases, wired, judge = off_loop(_judged_run, monkeypatch, _conversing_invoke(_retired))

    with pytest.raises(ModelUnservedError):
        await _execute(host, run, template, cases, wired)

    (result,) = _results(host.storage, run)
    assert judge.calls == [], "a trial lost to the setup buys no judge call"
    assert result.model_unserved and trial_exclusion(result) == "model_unserved"
    assert classify_result(result) is ResultOutcome.INFRA_EXCLUDE
    assert result.infra_error is not None and NO_LONGER_AVAILABLE in result.infra_error
    assert result.candidate_error is None
    assert result.judge_scores() == [] and result.judge_error is None


async def test_a_server_error_is_still_charged_to_the_arm_and_the_run_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, run, template, cases, wired, judge = off_loop(
        _judged_run, monkeypatch, _conversing_invoke(_server_error), n_cases=3, k=2
    )

    await _execute(host, run, template, cases, wired)

    results = _results(host.storage, run)
    assert len(results) == 6, "no circuit: every cell ran"
    for result in results:
        assert classify_result(result) is ResultOutcome.CANDIDATE_FAIL
        assert trial_exclusion(result) is None and not result.model_unserved
        assert result.candidate_error is not None and "openrouter 500" in result.candidate_error


async def test_a_served_cell_between_unserved_ones_starts_the_circuits_count_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, run, template, cases, wired, judge = off_loop(
        _judged_run, monkeypatch, _conversing_invoke(_retired, served_every=CIRCUIT), n_cases=3, k=2
    )

    await _execute(host, run, template, cases, wired)

    results = _results(host.storage, run)
    assert len(results) == 6
    assert sorted(result.model_unserved for result in results) == [False, False, True, True, True, True]
    assert len(judge.calls) > 0 and all(result.judge_scores() for result in results if not result.model_unserved), (
        "the served cells are judged"
    )


async def test_a_run_shorter_than_the_circuit_whose_every_cell_is_unserved_stops_all_the_same(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, run, template, cases, wired, judge = off_loop(
        _judged_run, monkeypatch, _conversing_invoke(_retired), n_cases=2
    )

    with pytest.raises(ModelUnservedError) as stopped:
        await _execute(host, run, template, cases, wired)

    assert stopped.value.model == RETIRED and NO_LONGER_AVAILABLE in stopped.value.detail
    assert len(_results(host.storage, run)) == 2 and judge.calls == []


async def test_a_quick_eval_whose_model_is_not_served_stops_naming_it_and_never_judges_or_fails_a_case() -> None:
    async def retired(case: Mapping[str, Any]) -> int:
        raise _retired()

    def size(case: Mapping[str, Any], answer: Any) -> float:
        return float(answer)

    with frozen_clock():
        result = await Eval([{"n": 1}, {"n": 2}], retired, [size], k=1).run(store=":memory:", scope="unserved")

    summary = result.summary
    assert summary.status == "exhausted"
    assert (summary.n_scored, summary.n_candidate_failed, summary.n_excluded) == (0, 0, 2)
    page = report_markdown(result.report)
    assert "does not serve the candidate's model" in page and "top up the account" not in page


# =============================================================================
# The circuit, the other arm, and the take-up, through a launch
# =============================================================================


def _provider_retires(monkeypatch: pytest.MonkeyPatch) -> dict[str, bool]:
    """Answer every extraction naming :data:`RETIRED` as OpenRouter answers a retired model, until the returned flag's
    ``retired`` is cleared: the model served again."""
    extract = ScriptedExtractionClient.extract
    state = {"retired": True}

    async def answering(self: ScriptedExtractionClient, request: Any) -> ExtractionResult:
        if state["retired"] and request.model == RETIRED:
            raise _retired()
        return await extract(self, request)

    monkeypatch.setattr(ScriptedExtractionClient, "extract", answering)
    return state


def _run(host: OpsHost, run_id: str) -> EvalRun:
    run, _ = host.eval_host.storage.load_job_record(EvalRun, run_id, TOYHOST_SCOPE)
    assert run is not None
    return run


def _rows(host: OpsHost, run_id: str) -> list[EvalResult]:
    return off_loop(host.eval_host.storage.query_eval_results_by_run, run_id, TOYHOST_SCOPE)


async def test_the_circuit_stops_the_unserved_arm_naming_the_model_and_the_other_arm_runs_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _provider_retires(monkeypatch)
    host = off_loop(ops_fixture).host
    arguments = LaunchArguments(
        template_id=toyhost_template().id,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=[RETIRED, SERVED],
        k_runs=K,
    )

    run_ids = await launched_runs(host, await run_launch(host, arguments, TOYHOST_SCOPE))

    runs = {run.candidate_model: run for run in (off_loop(_run, host, run_id) for run_id in run_ids)}
    stopped, served = runs[RETIRED], runs[SERVED]
    assert (stopped.status, stopped.exhausted_because) == ("exhausted", "model_unserved")
    assert RETIRED in stopped.error_details[-1] and NO_LONGER_AVAILABLE in stopped.error_details[-1]
    lost = _rows(host, stopped.id)
    assert CIRCUIT <= len(lost) < 3 * K, "the arm stopped before the rest of its matrix"
    assert all(row.model_unserved and trial_exclusion(row) == "model_unserved" for row in lost)
    assert (served.status, served.exhausted_because) == ("completed", None)
    assert len(_rows(host, served.id)) == 3 * K


async def test_an_arm_its_model_stopped_is_taken_up_once_the_model_is_served_and_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider_retires(monkeypatch)
    holder = off_loop(ops_fixture, store=ClockedStore()).host
    run_id = await launched_run(holder, K)
    assert (await settled(holder, run_job_id(run_id))).status == "exhausted"
    unserved = _rows(holder, run_id)
    listed = off_loop(resumable_jobs, holder.eval_host.storage, [TOYHOST_SCOPE])
    offered = [
        job for job in [*listed.jobs, *(run for group in listed.groups for run in group.runs)] if job.run_id == run_id
    ]
    assert [job.ended for job in offered] == ["exhausted"]
    assert offered[0].reason is not None and RETIRED in offered[0].reason

    provider["retired"] = False
    resumer = other_process(holder)
    await resume_run(resumer.launch, run_id, TOYHOST_SCOPE)
    assert (await settled(resumer, run_job_id(run_id))).state == "completed"

    run = off_loop(_run, holder, run_id)
    assert (run.status, run.exhausted_because) == ("completed", None)
    after = {row.id: row for row in _rows(holder, run_id)}
    assert len(after) == 3 * K and not any(row.model_unserved for row in after.values())
    for row in unserved:
        assert (after[row.id].model_unserved, after[row.id].lease_epoch) == (False, 1), "run again, written over"
    first, second = run.lifetimes
    assert (first.status, second.status) == ("exhausted", "completed")
