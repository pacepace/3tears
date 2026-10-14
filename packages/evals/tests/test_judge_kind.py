"""The judge kind, end to end over the toy host: frozen judge cases, two judge configs as arms, three code-graded measures.

The toy host's judged variant scores each extraction's faithfulness with a scripted judge. Its stored results are
frozen into judge cases — the evidence the judge read, the criterion, and the person ratings given on the result —
and a judge campaign then asks two judges of the same model those cases again: the built-in prompt, and a "harsh"
config whose prompt the scripted judge answers one point lower and on one invoice cannot answer at all. What it
pins is the deliverable's done-condition (#628): the campaign produces, per criterion and judge, agreement with the
labels, agreement with the judge's own repeats and parse validity, and every cent it spent is judge spend.
"""

from __future__ import annotations

import types

import json
from collections.abc import Sequence
from typing import Any

import pytest

from threetears.evals.analysis import JudgeKey, judge_kind_readings
from threetears.evals.kernel import (
    JUDGE_CASE_KEY,
    AuthoredJudgeCase,
    JUDGE_KIND,
    JUDGE_LABEL_AGREEMENT_MEASURE,
    JUDGE_PARSE_VALID_MEASURE,
    JudgeKindOverlays,
    ValidationFailedError,
    judge_case_of,
    judge_trial_of,
    withhold_failure_detail,
)
from threetears.evals.kernel.host import EvalHost
from threetears.evals.run import (
    JUDGE_REQUEST_SETTINGS,
    JudgeKind,
    RunnerOptions,
    execute_run,
    freeze_judge_cases,
    judge_kind,
    rate_result,
)
from threetears.evals.run.judge_service import builtin_judge_config_ids, judge_config_ids_for
from threetears.evals.schema import (
    DEFAULT_JUDGE_TEMPERATURE,
    CompletionProtocolDefect,
    EvalResult,
    EvalRun,
    EvalTemplate,
    JudgeConfig,
    JudgedArtifact,
    RubricDim,
)
from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.scripted_table import FakeCellSink
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_INSTANT, TOYHOST_SCOPE
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.judge import (
    FAITHFULNESS_DIM,
    toyhost_rubric,
    TOY_JUDGE_MODEL,
    ScriptedJudgeClient,
    ToyJudgeCompletion,
    toyhost_judge_service,
    toyhost_judged_template,
)
from packages.evals.tests.fixtures.toyhost.run import RUN_K, execute_toyhost_run
from packages.evals.tests.off_loop import off_loop

#: The word the harsh config's prompt carries, which the arm-aware judge below reads.
_HARSH = "HARSH"

#: The invoice the harsh judge cannot answer about in its protocol: every reply about it is unreadable.
_UNREADABLE_INVOICE = "Invoice doc-02"

#: Who labels the toy results.
_RATER = "ana@example.com"


class _ArmAwareJudge(ScriptedJudgeClient):
    """The toy host's scripted judge, read through each arm's system prompt.

    Under the built-in prompt it is the toy judge exactly. Under a prompt carrying :data:`_HARSH` it scores one
    point lower, and about :data:`_UNREADABLE_INVOICE` it replies with text no parser reads — so the two arms differ
    in agreement and in parse validity, by construction rather than by chance.
    """

    def __init__(self) -> None:
        """Start with no calls, and record which role each client was built for."""
        super().__init__()
        self.roles: list[str] = []

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, str] | None = None
    ) -> ToyJudgeCompletion:
        """Score as the toy judge does, harsher under a harsh prompt."""
        completion = await super().generate(system=system, user=user, response_format=response_format)
        if _HARSH not in system:
            return completion
        if _UNREADABLE_INVOICE in user:
            return ToyJudgeCompletion(**{**completion.__dict__, "content": "I would rather not say."})
        reply = json.loads(completion.content)
        score = reply["criteria_scores"][FAITHFULNESS_DIM]
        if isinstance(score, int):
            reply["criteria_scores"][FAITHFULNESS_DIM] = max(1, score - 1)
        return ToyJudgeCompletion(**{**completion.__dict__, "content": json.dumps(reply)})


def _judge_template() -> EvalTemplate:
    """The judge template the cases are frozen into: the judge kind, no rubric — its grade is code."""
    return EvalTemplate(
        id="toyhost-judge-faithfulness",
        scope_id=TOYHOST_SCOPE,
        name="Faithfulness judge",
        intent="Score how faithfully an extraction reproduces its invoice, as a person would.",
        candidate_kind=JUDGE_KIND,
        created_at=TOYHOST_INSTANT,
        updated_at=TOYHOST_INSTANT,
    )


async def _judged_source(host: EvalHost) -> list[EvalResult]:
    """Drive the toy host's judged variant, stamp the judging its launch would have recorded, and label the results.

    Every result of the first extractor model's run is labelled with the score its judge gave, and the second
    model's with one point more, so a judge that answers as the toy judge does agrees with some labels and not
    others.
    """
    template = toyhost_judged_template()
    off_loop(host.storage.save_template, template)
    path = await execute_toyhost_run(
        host=host,
        template=template,
        judge_service=toyhost_judge_service(ScriptedJudgeClient()),
        judge_model=TOY_JUDGE_MODEL,
    )
    for case in path.test_cases:
        off_loop(host.storage.save_test_case, case)
    for run in path.runs:
        # What launch_run stamps beside the judge pin; a drive of execute_run directly stamps it here.
        off_loop(
            host.storage.save_eval_run,
            run.model_copy(
                update={
                    "effective_judges": {FAITHFULNESS_DIM: TOY_JUDGE_MODEL},
                    "judge_config_ids": {},
                    "judge_asked_settings": JUDGE_REQUEST_SETTINGS,
                    "judge_temperature": DEFAULT_JUDGE_TEMPERATURE,
                }
            ),
        )
    results = await path.results()
    for index, run in enumerate(path.runs):
        for result in off_loop(host.storage.query_eval_results_by_run, run.id, TOYHOST_SCOPE):
            score = result.judge_score(FAITHFULNESS_DIM)
            assert score is not None
            off_loop(
                rate_result,
                host.storage,
                result_id=result.id,
                scope_id=TOYHOST_SCOPE,
                rubric_dim=FAITHFULNESS_DIM,
                rater=_RATER,
                rater_kind="person",
                score=min(5, score.score + index),
                reason="read against the invoice",
            )
    return results


async def _judge_arm(
    host: EvalHost, template: EvalTemplate, case_ids: Sequence[str], *, run_id: str, overlays: JudgeKindOverlays
) -> EvalRun:
    """Run one judge arm over the frozen cases, through the engine's runner."""
    kind = off_loop(
        judge_kind,
        host.storage,
        host.completion_clients("a judge campaign"),
        model=TOY_JUDGE_MODEL,
        scope_id=TOYHOST_SCOPE,
        failure_describer=withhold_failure_detail,
        overlays=overlays,
    )
    run = make_eval_run(
        id=run_id,
        scope_id=TOYHOST_SCOPE,
        template_id=template.id,
        candidate_kind=JUDGE_KIND,
        candidate_model=TOY_JUDGE_MODEL,
        k_runs=RUN_K,
        test_case_ids=list(case_ids),
        overlays=overlays.model_dump(mode="json"),
    )
    off_loop(host.storage.save_eval_run, run)
    cases = off_loop(host.storage.load_test_cases_by_ids, list(case_ids), TOYHOST_SCOPE)
    async with kind:
        await execute_run(
            host,
            run=run,
            template=template,
            test_cases=cases,
            judge_service=None,
            options=RunnerOptions(candidate_kinds={JUDGE_KIND: lambda _cell: kind}),
        )
    return run


class _Campaign:
    """A judge campaign over the toy host: its frozen cases, its two arms, and everything they stored."""

    def __init__(self, host: EvalHost, judge: _ArmAwareJudge, harsh: JudgeConfig, case_ids: list[str]) -> None:
        self.host = host
        self.judge = judge
        self.harsh = harsh
        self.case_ids = case_ids
        self.results: list[EvalResult] = []


async def _campaign() -> _Campaign:
    judge = _ArmAwareJudge()

    def clients(role: str, model: str | None, *, temperature: float | None = None) -> Any:
        judge.roles.append(role)
        return judge

    host = toyhost_host(clients=clients)
    source = await _judged_source(host)
    template = _judge_template()
    off_loop(host.storage.save_template, template)
    report = off_loop(
        freeze_judge_cases,
        host.storage,
        template=template,
        run_ids=sorted({result.eval_run_id for result in source}),
        scope_id=TOYHOST_SCOPE,
        case_set="faithfulness",
    )
    assert report.case_set is not None and not report.skipped
    harsh = JudgeConfig(
        scope_id=TOYHOST_SCOPE,
        name="faithfulness-harsh",
        rubric_dim_id=FAITHFULNESS_DIM,
        prompt_template=f"{_HARSH}: score how faithfully the extraction reproduces the invoice. Be strict.",
    )
    off_loop(host.storage.save_judge_config, harsh)
    case_ids = [case.test_case_id for case in report.cases]
    campaign = _Campaign(host, judge, harsh, case_ids)
    judge.calls.clear()
    runs = [
        await _judge_arm(host, template, case_ids, run_id="judge-arm-builtin", overlays=JudgeKindOverlays()),
        await _judge_arm(
            host,
            template,
            case_ids,
            run_id="judge-arm-harsh",
            overlays=JudgeKindOverlays(config_ids={FAITHFULNESS_DIM: harsh.id}),
        ),
    ]
    campaign.results = [
        result for run in runs for result in off_loop(host.storage.query_eval_results_by_run, run.id, TOYHOST_SCOPE)
    ]
    return campaign


async def test_freezing_a_judged_run_makes_one_case_per_judged_output_carrying_what_its_judge_read() -> None:
    judge = _ArmAwareJudge()
    host = toyhost_host(clients=lambda role, model, *, temperature=None: judge)
    source = await _judged_source(host)
    template = _judge_template()
    off_loop(host.storage.save_template, template)
    run_ids = sorted({result.eval_run_id for result in source})

    report = off_loop(freeze_judge_cases, host.storage, template=template, run_ids=run_ids, scope_id=TOYHOST_SCOPE)

    assert len(report.cases) == len(source) and all(case.created for case in report.cases)
    by_result = {result.id: result for result in source}
    for frozen in report.cases:
        stored = off_loop(host.storage.load_test_case, frozen.test_case_id, TOYHOST_SCOPE)
        assert stored is not None and stored.template_id == template.id and stored.stratum == FAITHFULNESS_DIM
        case = judge_case_of(stored)
        assert case is not None
        result = by_result[case.source.result_id]
        trace = off_loop(host.storage.load_eval_trace, result.id, TOYHOST_SCOPE)
        assert trace is not None
        # Exactly what the first judge read, and what it answered, and what the person said.
        assert case.evidence == trace.judge_evidence
        assert case.criterion is not None and case.criterion.name == FAITHFULNESS_DIM
        first = result.judge_score(FAITHFULNESS_DIM)
        assert first is not None and case.source.first_score == first.score
        assert [label.rater for label in case.labels] == [_RATER]
    # Freezing the same outputs again names the stored cases rather than minting more.
    again = off_loop(freeze_judge_cases, host.storage, template=template, run_ids=run_ids, scope_id=TOYHOST_SCOPE)
    assert [case.test_case_id for case in again.cases] == [case.test_case_id for case in report.cases]
    assert not any(case.created for case in again.cases)


async def test_a_freeze_into_a_template_of_another_kind_is_refused() -> None:
    host = toyhost_host()
    with pytest.raises(ValidationFailedError, match="not 'judge'"):
        freeze_judge_cases(host.storage, template=toyhost_judged_template(), run_ids=["x"], scope_id=TOYHOST_SCOPE)


async def test_two_judge_configs_as_arms_produce_label_agreement_self_agreement_and_parse_validity_per_criterion() -> (
    None
):
    campaign = await _campaign()
    readings = judge_kind_readings(campaign.results)

    assert not readings.unread
    # The arm with no config is asked by the built-in prompt, which its trials record by its identity.
    builtin_id = builtin_judge_config_ids(toyhost_judged_template().rubric, judged_artifact=JudgedArtifact.DOCUMENT)[
        FAITHFULNESS_DIM
    ]
    builtin = readings.reading(JudgeKey(FAITHFULNESS_DIM, "ordinal", None, builtin_id, DEFAULT_JUDGE_TEMPERATURE))
    # The harsh arm is asked by its config, recorded with the reply format the engine appended to its prompt.
    configured = judge_config_ids_for(
        toyhost_judged_template().rubric, {FAITHFULNESS_DIM: campaign.harsh.id}, judged_artifact=JudgedArtifact.DOCUMENT
    )[FAITHFULNESS_DIM]
    harsh = readings.reading(JudgeKey(FAITHFULNESS_DIM, "ordinal", None, configured, DEFAULT_JUDGE_TEMPERATURE))
    assert builtin is not None and harsh is not None and len(readings.readings) == 2
    cases = len(campaign.case_ids)
    for reading in (builtin, harsh):
        assert reading.cases == cases and reading.trials == cases * RUN_K
        assert reading.case_set_fingerprint == builtin.case_set_fingerprint, "both arms read the same frozen cases"
    # Parse validity: the built-in prompt always answers in protocol; the harsh one never does about one invoice.
    assert builtin.parse_validity.rate == 1.0 and builtin.parse_validity.invalid == 0
    assert harsh.parse_validity.invalid > 0 and harsh.parse_validity.rate is not None
    assert harsh.parse_validity.rate < 1.0
    # Agreement with the labels: the built-in judge gives the first model's labels exactly; the harsh one never
    # matches a label it scored one point under.
    assert builtin.label_agreement is not None and harsh.label_agreement is not None
    assert builtin.label_agreement.results == cases
    assert builtin.label_agreement.exact_agreement > harsh.label_agreement.exact_agreement
    # Self-agreement: a scripted judge repeats itself exactly, over every case it scored, a case counted once.
    assert builtin.self_agreement is not None and builtin.self_agreement.exact_agreement == 1.0
    assert builtin.self_agreement.results == cases
    assert harsh.self_agreement is not None and harsh.self_agreement.results < cases


async def test_a_judge_campaign_meters_only_judge_spend_and_never_rebuys_the_candidate() -> None:
    campaign = await _campaign()

    assert campaign.results
    assert set(campaign.judge.roles) == {"judge"}, "every client a judge trial asked for is a judge client"
    # One judge call per trial (the unreadable replies buy their one parse retry), and no extractor call at all.
    assert len(campaign.judge.calls) >= len(campaign.results)
    for result in campaign.results:
        assert result.usage, "a judge trial's spend is metered"
        assert {row.role for row in result.usage} == {"judge"}, "no candidate role in a judge campaign's spend"
        assert result.cost_usd == pytest.approx(sum(row.cost_usd or 0.0 for row in result.usage))
        assert result.judge_model is None and not result.rubric_scores, "no judge phase ran over a judge trial"
        assert JUDGE_PARSE_VALID_MEASURE in result.host_measures
    labelled = [result for result in campaign.results if JUDGE_LABEL_AGREEMENT_MEASURE in result.host_measures]
    assert labelled, "a scored trial of a labelled case lands its agreement with the label"


async def test_a_trial_on_a_case_with_no_judge_case_is_a_harness_fault() -> None:
    campaign = await _campaign()
    stored = off_loop(campaign.host.storage.load_test_case, campaign.case_ids[0], TOYHOST_SCOPE)
    assert stored is not None
    bare = stored.model_copy(update={"id": "not-a-judge-case", "host_payload": {}})
    kind = JudgeKind(
        clients=lambda role, model, *, temperature=None: campaign.judge,
        model=TOY_JUDGE_MODEL,
        failure_describer=withhold_failure_detail,
    )

    class _Window:
        def identity(self) -> Any:
            import contextlib

            return contextlib.nullcontext()

        collecting = identity

    from threetears.evals.kernel import VariantConfig

    prepared = await kind.prepare(
        subject_snapshot=None,
        variant_config=VariantConfig(candidate_model=TOY_JUDGE_MODEL),
        world_seed=None,
        span_window=_Window(),
        cassettes=None,
        world=None,
    )
    output = await kind.invoke(prepared, bare, sink=None)  # type: ignore[arg-type]
    assert output.infra_errors and JUDGE_CASE_KEY in output.infra_errors[0]
    assert not output.telemetry.usage, "nothing was asked, so nothing was spent"


async def test_a_judge_trial_holds_its_call_against_the_runs_cap_and_counts_what_it_cost() -> None:
    campaign = await _campaign()
    stored = off_loop(campaign.host.storage.load_test_case, campaign.case_ids[0], TOYHOST_SCOPE)
    assert stored is not None
    kind = JudgeKind(
        clients=lambda role, model, *, temperature=None: campaign.judge,
        model=TOY_JUDGE_MODEL,
        failure_describer=withhold_failure_detail,
    )

    class _Window:
        def identity(self) -> Any:
            import contextlib

            return contextlib.nullcontext()

        collecting = identity

    from threetears.evals.kernel import VariantConfig

    prepared = await kind.prepare(
        subject_snapshot=None,
        variant_config=VariantConfig(candidate_model=TOY_JUDGE_MODEL),
        world_seed=None,
        span_window=_Window(),
        cassettes=None,
        world=None,
    )

    room = FakeCellSink(cap_usd=10.0)
    answered = await kind.invoke(prepared, stored, sink=room)
    spent = sum(row.cost_usd or 0.0 for row in answered.telemetry.usage)
    assert spent > 0
    assert room.spent == {"candidate": pytest.approx(spent)}, "the trial's call is counted the moment it returns"
    assert room.held == [], "its reservation was settled"

    full = FakeCellSink(cap_usd=0.0001)
    full.add_spend("candidate", 1.0)
    calls = len(campaign.judge.calls)
    refused = await kind.invoke(prepared, stored, sink=full)
    assert len(campaign.judge.calls) == calls, "a call the cap has no room for is not sent"
    assert not refused.output and not refused.candidate_errors and not refused.telemetry.usage


class _Untempered:
    """Hands back the judge's replies stripped of ``temperature``: a host completion type short of the protocol."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        reply = await self.inner.generate(system=system, user=user, response_format=response_format)
        kept = {name: getattr(reply, name) for name in dir(reply) if not name.startswith("_") and name != "temperature"}
        return types.SimpleNamespace(**{name: value for name, value in kept.items() if not callable(value)})


async def test_a_trial_whose_completion_carries_no_temperature_fails_the_run_and_counts_what_it_was_billed() -> None:
    """The host's completion type breaks the protocol: every trial would, so it is not folded into this one.

    It used to be recorded as the trial's apparatus error, its spend counted, and the campaign finished with every
    trial failed the same way. A protocol defect is the host's code, so it is raised through ``invoke`` to fail the
    run, as a kind's defect is — with what the call was billed still counted.
    """
    campaign = await _campaign()
    stored = off_loop(campaign.host.storage.load_test_case, campaign.case_ids[0], TOYHOST_SCOPE)
    assert stored is not None
    untempered = _Untempered(campaign.judge)
    kind = JudgeKind(
        clients=lambda role, model, *, temperature=None: untempered,
        model=TOY_JUDGE_MODEL,
        failure_describer=withhold_failure_detail,
    )

    class _Window:
        def identity(self) -> Any:
            import contextlib

            return contextlib.nullcontext()

        collecting = identity

    from threetears.evals.kernel import VariantConfig

    prepared = await kind.prepare(
        subject_snapshot=None,
        variant_config=VariantConfig(candidate_model=TOY_JUDGE_MODEL),
        world_seed=None,
        span_window=_Window(),
        cassettes=None,
        world=None,
    )
    room = FakeCellSink(cap_usd=10.0)

    with pytest.raises(CompletionProtocolDefect, match="check_completion_conformance"):
        await kind.invoke(prepared, stored, sink=room)

    # Raised, but what the call was billed is counted: settled against the cap and reported as the cell's spend,
    # which the runner records on the excluded cell before it fails the run.
    assert room.held == [], "nothing is left held against the cap"
    assert room.progress is not None
    spent = sum(row.cost_usd or 0.0 for row in room.progress().telemetry.usage)
    assert spent > 0 and room.spent == {"candidate": pytest.approx(spent)}


def test_a_config_pinning_another_model_than_the_arm_is_refused() -> None:
    config = JudgeConfig(
        scope_id=TOYHOST_SCOPE, name="other", rubric_dim_id=FAITHFULNESS_DIM, prompt_template="x", model="other/model"
    )
    with pytest.raises(ValueError, match="pins model 'other/model'"):
        JudgeKind(
            clients=lambda role, model, *, temperature=None: None,
            model=TOY_JUDGE_MODEL,
            failure_describer=withhold_failure_detail,
            configs={FAITHFULNESS_DIM: config},
        )


async def test_the_freeze_operation_loads_the_template_in_scope_and_renders_its_receipt() -> None:
    from threetears.evals.actions import engine_actions
    from threetears.evals.ops import JudgeCasesFreeze, judge_cases_freeze

    host = toyhost_host()
    source = await _judged_source(host)
    template = _judge_template()
    off_loop(host.storage.save_template, template)
    freeze = JudgeCasesFreeze(
        template_id=template.id,
        judged_run_ids=sorted({result.eval_run_id for result in source}),
        criteria=[FAITHFULNESS_DIM],
        into_case_set="faithfulness",
    )

    report = off_loop(judge_cases_freeze, host, freeze, TOYHOST_SCOPE)

    assert report.case_set is not None and report.case_set.version == 1 and len(report.cases) == len(source)
    action = next(action for action in engine_actions() if action.name == "judge_cases_freeze")
    assert action.permission == "write"
    text = action.render(report)
    lines = text.splitlines()
    assert {f"template_id: {template.id}", f"cases ({len(source)}):", "  name: faithfulness", "  version: 1"} <= set(lines)
    # The same freeze again answers with the version already listing exactly these cases.
    assert off_loop(judge_cases_freeze, host, freeze, TOYHOST_SCOPE).case_set == report.case_set


async def test_a_label_follows_a_byte_identical_output_into_its_case_and_never_counts_twice() -> None:
    host = toyhost_host()
    template = toyhost_judged_template()
    off_loop(host.storage.save_template, template)
    path = await execute_toyhost_run(
        host=host,
        template=template,
        judge_service=toyhost_judge_service(ScriptedJudgeClient()),
        judge_model=TOY_JUDGE_MODEL,
    )
    for case in path.test_cases:
        off_loop(host.storage.save_test_case, case)
    for run in path.runs:
        off_loop(
            host.storage.save_eval_run,
            run.model_copy(
                update={
                    "effective_judges": {FAITHFULNESS_DIM: TOY_JUDGE_MODEL},
                    "judge_config_ids": {},
                    "judge_asked_settings": JUDGE_REQUEST_SETTINGS,
                    "judge_temperature": DEFAULT_JUDGE_TEMPERATURE,
                }
            ),
        )
    results = [
        result
        for run in path.runs
        for result in off_loop(host.storage.query_eval_results_by_run, run.id, TOYHOST_SCOPE)
    ]
    keys = {}
    for result in results:
        score = result.judge_score(FAITHFULNESS_DIM)
        assert score is not None and score.label_key is not None, "the judge stamps what it read"
        keys[result.id] = score.label_key
    # One result rated, by one person: its output's twins carry the label too, found by what was read.
    rated = next(result for result in results if sum(key == keys[result.id] for key in keys.values()) > 1)
    score = rated.judge_score(FAITHFULNESS_DIM)
    assert score is not None
    off_loop(
        rate_result,
        host.storage,
        result_id=rated.id,
        scope_id=TOYHOST_SCOPE,
        rubric_dim=FAITHFULNESS_DIM,
        rater=_RATER,
        rater_kind="person",
        score=score.score,
        reason="read against the invoice",
    )
    judge_template = _judge_template()
    off_loop(host.storage.save_template, judge_template)

    report = off_loop(
        freeze_judge_cases,
        host.storage,
        template=judge_template,
        run_ids=[run.id for run in path.runs],
        scope_id=TOYHOST_SCOPE,
    )

    labels = {case.source_result_id: case.labels for case in report.cases}
    twins = {result_id for result_id, key in keys.items() if key == keys[rated.id]}
    assert labels[rated.id] == 1, "found by its result and by what was read, it is one label"
    assert all(labels[result_id] == 1 for result_id in twins), "a byte-identical output carries the label"
    assert all(labels[result_id] == 0 for result_id in set(keys) - twins), "another output carries none"
    assert len(twins) > 1
    # The same person rating a twin too gives one opinion of one output: the later rating is the case's label.
    twin = next(result_id for result_id in sorted(twins) if result_id != rated.id)
    later = off_loop(
        rate_result,
        host.storage,
        result_id=twin,
        scope_id=TOYHOST_SCOPE,
        rubric_dim=FAITHFULNESS_DIM,
        rater=_RATER,
        rater_kind="person",
        score=max(1, score.score - 1),
        reason="read it again, more strictly",
    )
    again = off_loop(
        freeze_judge_cases,
        host.storage,
        template=judge_template,
        run_ids=[run.id for run in path.runs],
        scope_id=TOYHOST_SCOPE,
        result_ids=sorted(twins),
    )
    for frozen in again.cases:
        stored = off_loop(host.storage.load_test_case, frozen.test_case_id, TOYHOST_SCOPE)
        assert stored is not None
        case = judge_case_of(stored)
        assert case is not None
        assert [label.rating_id for label in case.labels] == [later.id]


class _ServedJudge(_ArmAwareJudge):
    """The arm-aware toy judge, whose replies name the model that served them, as a profile needs to match a judge."""

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, str] | None = None
    ) -> ToyJudgeCompletion:
        """Answer as the toy judge does, naming :data:`TOY_JUDGE_MODEL` as the served model."""
        completion = await super().generate(system=system, user=user, response_format=response_format)
        return ToyJudgeCompletion(**{**completion.__dict__, "served_model": TOY_JUDGE_MODEL})


async def test_a_label_outlives_its_evidence_and_still_calibrates_a_judge_after_a_core_reset() -> None:
    """Rate, delete every run and reset the core, then freeze from the labels alone and profile a judge on them.

    The labels are the one irreplaceable input to calibration and are durable; everything else is core. So after
    a reset the freeze reads no run, result or trace, and the judge campaign over its cases still measures the
    judge against what people said, on the very evidence they read.
    """
    from threetears.evals.analysis import judge_profiles_of
    from threetears.evals.run import delete_run

    judge = _ServedJudge()
    host = toyhost_host(clients=lambda role, model, *, temperature=None: judge)
    source = await _judged_source(host)
    ratings = off_loop(host.storage.query_calibration_ratings, TOYHOST_SCOPE)
    assert ratings and all(rating.rated is not None for rating in ratings), "every label carries what was read"
    read = {}
    for result in source:
        trace = off_loop(host.storage.load_eval_trace, result.id, TOYHOST_SCOPE)
        assert trace is not None and trace.judge_evidence is not None
        read[result.id] = trace.judge_evidence
    for run_id in sorted({result.eval_run_id for result in source}):
        run = off_loop(host.storage.load_eval_run, run_id, TOYHOST_SCOPE)
        assert run is not None
        off_loop(delete_run, host.storage, run, TOYHOST_SCOPE, confirm=run.id)
    off_loop(host.storage.reset_core, [TOYHOST_SCOPE])
    assert not off_loop(host.storage.query_eval_runs, TOYHOST_SCOPE)
    assert all(off_loop(host.storage.load_eval_trace, result_id, TOYHOST_SCOPE) is None for result_id in read)
    template = _judge_template()
    off_loop(host.storage.save_template, template)

    report = off_loop(
        freeze_judge_cases,
        host.storage,
        template=template,
        scope_id=TOYHOST_SCOPE,
        from_labels=True,
        case_set="faithfulness",
    )

    assert not report.skipped
    assert len(report.cases) == len({rating.label_key for rating in ratings})
    for frozen in report.cases:
        stored = off_loop(host.storage.load_test_case, frozen.test_case_id, TOYHOST_SCOPE)
        assert stored is not None
        case = judge_case_of(stored)
        assert case is not None and case.labels
        assert case.evidence == read[case.source.result_id], "the case is the evidence the rater read"
        assert case.source.first_score is None and case.source.test_case_id is None
    run = await _judge_arm(
        host,
        template,
        [case.test_case_id for case in report.cases],
        run_id="judge-arm-after-reset",
        overlays=JudgeKindOverlays(),
    )
    results = off_loop(host.storage.query_eval_results_by_run, run.id, TOYHOST_SCOPE)

    drafts = judge_profiles_of(judge_kind_readings(results), scope_id=TOYHOST_SCOPE)

    (profile,) = drafts.profiles
    assert profile.judge_model == TOY_JUDGE_MODEL
    assert profile.label_agreement is not None and profile.label_agreement.results == len(report.cases)


async def test_a_rating_of_a_result_whose_template_was_edited_since_is_stored_saying_why_it_carries_nothing() -> None:
    host = toyhost_host()
    source = await _judged_source(host)
    template = off_loop(host.storage.load_template, toyhost_judged_template().id, TOYHOST_SCOPE)
    assert template is not None
    off_loop(host.storage.save_template, template.model_copy(update={"updated_at": "2099-01-01T00:00:00+00:00"}))
    score = source[0].judge_score(FAITHFULNESS_DIM)
    assert score is not None

    rating = off_loop(
        rate_result,
        host.storage,
        result_id=source[0].id,
        scope_id=TOYHOST_SCOPE,
        rubric_dim=FAITHFULNESS_DIM,
        rater="bo@example.com",
        rater_kind="person",
        score=score.score,
        reason="read it after the template changed",
    )

    assert rating.rated is None and rating.rated_withheld is not None
    assert rating.rated_withheld.startswith("what the judge read cannot be rebuilt from the run")
    assert "edited" in rating.rated_withheld


async def test_a_freeze_from_labels_skips_by_name_a_label_that_carries_no_content() -> None:
    """A label written before ratings carried their content cannot be frozen, and the receipt names it."""
    host = toyhost_host()
    source = await _judged_source(host)
    ratings = off_loop(host.storage.query_calibration_ratings, TOYHOST_SCOPE)
    withheld = ratings[0].model_copy(update={"rated": None, "rated_withheld": "written before labels carried it"})
    off_loop(host.storage.save_calibration_rating, withheld)
    alone = [rating for rating in ratings if rating.label_key == withheld.label_key]
    template = _judge_template()
    off_loop(host.storage.save_template, template)

    report = off_loop(freeze_judge_cases, host.storage, template=template, scope_id=TOYHOST_SCOPE, from_labels=True)

    if len(alone) == 1:
        (skip,) = report.skipped
        assert skip.rating_id == withheld.id and skip.result_id == withheld.result_id
        assert "carries no rated content: written before labels carried it" in skip.reason
    assert len(report.cases) == len({rating.label_key for rating in ratings}) - (len(alone) == 1)
    assert source
    with pytest.raises(ValidationFailedError, match="reads no run or result"):
        freeze_judge_cases(host.storage, template=template, run_ids=["x"], scope_id=TOYHOST_SCOPE, from_labels=True)


def test_a_result_whose_trial_this_build_cannot_read_is_left_out_saying_why() -> None:
    """An unreadable trial read as "carries no judge trial": the validation error behind it was swallowed."""
    unreadable = make_eval_result(id="res-unreadable", kind_payload={"test_case_id": "c-1", "dim": FAITHFULNESS_DIM})
    absent = make_eval_result(id="res-absent", kind_payload=None)

    readings = judge_kind_readings([unreadable, absent])

    reasons = {entry.result_id: entry.reason for entry in readings.unread}
    assert reasons["res-absent"] == "it carries no judge trial"
    assert reasons["res-unreadable"].startswith("its kind_payload is not a judge trial this build can read")
    assert "scale" in reasons["res-unreadable"], "the reason names what failed to validate"


# --- authored cases: the hard negatives no run stored ----------------------------------------------------------

#: The invoice every authored case is judged against, rendered as the toy kind renders one.
_AUTHORED_INVOICE = "\n".join(
    [
        "Invoice doc-authored",
        "invoice_number: INV-0042",
        "invoice_date: 2026-03-01",
        "total_amount: 0.00",
        "vendor_name: Acme Ltd",
    ]
)

#: Who writes the authored cases.
_AUTHOR = "cy@example.com"


def _authored(fields: dict[str, str], expected: int | str, note: str, **overrides: Any) -> AuthoredJudgeCase:
    """One hand-written case on the toy criterion, judged against :data:`_AUTHORED_INVOICE`."""
    (criterion,) = toyhost_rubric()
    return AuthoredJudgeCase(
        criterion=overrides.pop("criterion", criterion),
        output=json.dumps(fields, sort_keys=True),
        case_material=_AUTHORED_INVOICE,
        intent=toyhost_judged_template().intent,
        expected=expected,
        author=_AUTHOR,
        note=note,
        **overrides,
    )


def _hard_cases() -> list[AuthoredJudgeCase]:
    """A hard negative a lenient judge passes, a plain negative, and an output that looks wrong and must pass."""
    right = {
        "invoice_number": "INV-0042",
        "invoice_date": "2026-03-01",
        "total_amount": "0.00",
        "vendor_name": "Acme Ltd",
    }
    return [
        # Three of four fields right, so the toy judge scores it 4; the swapped date is the whole point: it fails.
        _authored(
            {**right, "invoice_date": "2026-01-03"}, 1, "the date is day-month swapped, which bills a month early"
        ),
        _authored({name: "?" for name in right}, 1, "nothing on it is the invoice's"),
        # A zero total reads like an extraction that dropped the amount; the invoice says zero, so it is right.
        _authored(right, 5, "the invoice really is for 0.00: a credit-note reissue"),
    ]


async def test_authored_cases_are_judged_and_read_out_as_labelled_cases_whose_label_is_the_verdict() -> None:
    from threetears.evals.analysis import judge_profiles_of

    judge = _ServedJudge()
    host = toyhost_host(clients=lambda role, model, *, temperature=None: judge)
    template = _judge_template()
    off_loop(host.storage.save_template, template)

    report = off_loop(
        freeze_judge_cases,
        host.storage,
        template=template,
        scope_id=TOYHOST_SCOPE,
        authored=_hard_cases(),
        case_set="faithfulness-hard",
    )

    assert report.case_set is not None and len(report.cases) == 3 and not report.skipped
    assert all(case.authored_by == _AUTHOR and case.source_result_id is None for case in report.cases)
    for frozen, written in zip(report.cases, _hard_cases(), strict=True):
        stored = off_loop(host.storage.load_test_case, frozen.test_case_id, TOYHOST_SCOPE)
        assert stored is not None and stored.stratum == FAITHFULNESS_DIM
        case = judge_case_of(stored)
        assert case is not None and case.source.authored is not None and case.source.run_id is None
        assert case.evidence.artifact == written.output and case.criterion == written.criterion
        (label,) = case.labels
        assert (label.rater, label.score, label.reason, label.rating_id) == (
            _AUTHOR,
            written.expected,
            written.note,
            None,
        )
    # Authoring the same cases again names the stored ones.
    again = off_loop(
        freeze_judge_cases, host.storage, template=template, scope_id=TOYHOST_SCOPE, authored=_hard_cases()
    )
    assert [case.test_case_id for case in again.cases] == [case.test_case_id for case in report.cases]
    assert not any(case.created for case in again.cases)

    run = await _judge_arm(
        host,
        template,
        [case.test_case_id for case in report.cases],
        run_id="judge-arm-hard",
        overlays=JudgeKindOverlays(),
    )
    results = off_loop(host.storage.query_eval_results_by_run, run.id, TOYHOST_SCOPE)

    trials = [trial for result in results if (trial := judge_trial_of(result.kind_payload)) is not None]
    assert trials and all(trial.source_result_id is None and trial.labels[0].rater == _AUTHOR for trial in trials)
    # The lenient toy judge passes the swapped date (4 against a 1): the hard negative catches it, and only it.
    agreement = {result.test_case_id: result.host_measures[JUDGE_LABEL_AGREEMENT_MEASURE] for result in results}
    assert sorted(set(agreement.values())) == [0.0, 1.0]
    readings = judge_kind_readings(results)
    (reading,) = readings.readings
    assert reading.label_agreement is not None and reading.label_agreement.results == 3
    (profile,) = judge_profiles_of(readings, scope_id=TOYHOST_SCOPE).profiles
    assert profile.label_agreement is not None and profile.label_agreement.results == 3


async def test_authored_cases_freeze_beside_stored_ones_into_one_case_set() -> None:
    host = toyhost_host()
    source = await _judged_source(host)
    template = _judge_template()
    off_loop(host.storage.save_template, template)

    report = off_loop(
        freeze_judge_cases,
        host.storage,
        template=template,
        run_ids=sorted({result.eval_run_id for result in source}),
        scope_id=TOYHOST_SCOPE,
        authored=_hard_cases(),
        case_set="faithfulness",
    )

    assert report.case_set is not None and len(report.cases) == len(source) + 3
    assert [case.authored_by for case in report.cases[:3]] == [_AUTHOR] * 3
    assert all(case.authored_by is None and case.source_result_id is not None for case in report.cases[3:])


@pytest.mark.parametrize(
    ("scale", "verdict", "named"),
    [
        ("ordinal", "fail", "'fail'"),
        ("ordinal", 7, "verdict 7"),
        ("pass_fail", 3, "verdict 3"),
        ("ordinal", 0, "verdict 0"),
    ],
)
def test_an_authored_verdict_its_criterion_s_scale_cannot_hold_is_refused_naming_both(
    scale: str, verdict: int | str, named: str
) -> None:
    (criterion,) = toyhost_rubric()
    if scale == "pass_fail":
        criterion = RubricDim(
            name=FAITHFULNESS_DIM, description=criterion.description, scale="pass_fail", judge="builtin"
        )
    with pytest.raises(
        ValueError, match=rf"criterion '{FAITHFULNESS_DIM}' is answered on the {scale} scale"
    ) as refused:
        _authored({}, verdict, "out of range", criterion=criterion)
    assert named in str(refused.value)


def test_an_authored_pass_fail_verdict_is_stored_as_the_score_it_stands_for() -> None:
    (criterion,) = toyhost_rubric()
    criterion = RubricDim(name=FAITHFULNESS_DIM, description=criterion.description, scale="pass_fail", judge="builtin")
    must_fail = _authored({}, "fail", "a hard negative", criterion=criterion)
    must_pass = _authored({}, "pass", "looks wrong, is right", criterion=criterion)

    assert [must_fail.expected_score, must_pass.expected_score] == [0, 1]
    from threetears.evals.kernel import JudgeCase

    assert JudgeCase.from_authored(must_fail).labels[0].score == 0


def test_a_freeze_naming_no_run_no_labels_and_no_authored_case_is_refused() -> None:
    host = toyhost_host()
    with pytest.raises(ValidationFailedError, match="no judged run, no labels and no authored case"):
        freeze_judge_cases(host.storage, template=_judge_template(), scope_id=TOYHOST_SCOPE)


async def test_the_author_operation_writes_a_case_on_a_template_s_dimension_as_the_caller() -> None:
    from threetears.evals.actions import engine_actions
    from threetears.evals.ops import JudgeCaseAuthor, judge_case_author

    host = toyhost_host()
    subject = toyhost_judged_template()
    off_loop(host.storage.save_template, subject)
    template = _judge_template()
    off_loop(host.storage.save_template, template)
    hard = JudgeCaseAuthor(
        template_id=template.id,
        criterion_template_id=subject.id,
        criterion=FAITHFULNESS_DIM,
        output=json.dumps({"invoice_number": "INV-0042"}),
        case_material=_AUTHORED_INVOICE,
        expected=1,
        note="three of four fields missing is not faithful",
    )

    report = off_loop(judge_case_author, host, hard, TOYHOST_SCOPE, author=_AUTHOR)

    (frozen,) = report.cases
    stored = off_loop(host.storage.load_test_case, frozen.test_case_id, TOYHOST_SCOPE)
    assert stored is not None and stored.template_id == template.id
    case = judge_case_of(stored)
    assert case is not None and case.criterion == subject.rubric[0] and case.intent == subject.intent
    assert case.labels[0].rater == _AUTHOR and frozen.authored_by == _AUTHOR
    action = next(action for action in engine_actions() if action.name == "judge_case_author")
    assert action.permission == "write"
    (line,) = [line for line in action.render(report).splitlines() if line.startswith("- test_case_id=")]
    assert f"dim={FAITHFULNESS_DIM}" in line and f"authored_by={_AUTHOR}" in line
    # Authoring it again names the stored case.
    assert off_loop(judge_case_author, host, hard, TOYHOST_SCOPE, author=_AUTHOR).cases[0].created is False
    # A dimension the template does not have, and a verdict its scale cannot hold, are each refused by name.
    with pytest.raises(ValidationFailedError, match="has no rubric dimension 'extraction.tone'"):
        off_loop(
            judge_case_author,
            host,
            hard.model_copy(update={"criterion": "extraction.tone"}),
            TOYHOST_SCOPE,
            author=_AUTHOR,
        )
    with pytest.raises(
        ValidationFailedError, match=rf"criterion '{FAITHFULNESS_DIM}' .* cannot hold the verdict 'fail'"
    ):
        off_loop(
            judge_case_author,
            host,
            JudgeCaseAuthor.model_validate({**hard.model_dump(), "expected": "fail"}),
            TOYHOST_SCOPE,
            author=_AUTHOR,
        )
