"""The judge kind, end to end over the toy host: frozen judge cases, two judge configs as arms, three code-graded measures.

The toy host's judged variant scores each extraction's faithfulness with a scripted judge. Its stored results are
frozen into judge cases — the evidence the judge read, the criterion, and the person ratings given on the result —
and a judge campaign then asks two judges of the same model those cases again: the built-in prompt, and a "harsh"
config whose prompt the scripted judge answers one point lower and on one invoice cannot answer at all. What it
pins is the deliverable's done-condition (#628): the campaign produces, per criterion and judge, agreement with the
labels, agreement with the judge's own repeats and parse validity, and every cent it spent is judge spend.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from threetears.evals.analysis import JudgeKey, judge_kind_readings
from threetears.evals.kernel import (
    JUDGE_CASE_KEY,
    JUDGE_KIND,
    JUDGE_LABEL_AGREEMENT_MEASURE,
    JUDGE_PARSE_VALID_MEASURE,
    JudgeKindOverlays,
    ValidationFailedError,
    judge_case_of,
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
from threetears.evals.schema import DEFAULT_JUDGE_TEMPERATURE, EvalResult, EvalRun, EvalTemplate, JudgeConfig
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_INSTANT, TOYHOST_SCOPE
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.judge import (
    FAITHFULNESS_DIM,
    TOY_JUDGE_MODEL,
    ScriptedJudgeClient,
    ToyJudgeCompletion,
    toyhost_judge_service,
    toyhost_judged_template,
)
from packages.evals.tests.fixtures.toyhost.run import RUN_K, execute_toyhost_run

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
    host.storage.save_template(template)
    path = await execute_toyhost_run(
        host=host,
        template=template,
        judge_service=toyhost_judge_service(ScriptedJudgeClient()),
        judge_model=TOY_JUDGE_MODEL,
    )
    for case in path.test_cases:
        host.storage.save_test_case(case)
    for run in path.runs:
        # What launch_run stamps beside the judge pin; a drive of execute_run directly stamps it here.
        host.storage.save_eval_run(
            run.model_copy(
                update={
                    "effective_judges": {FAITHFULNESS_DIM: TOY_JUDGE_MODEL},
                    "judge_config_ids": {},
                    "judge_request_settings": JUDGE_REQUEST_SETTINGS,
                    "judge_temperature": DEFAULT_JUDGE_TEMPERATURE,
                }
            )
        )
    results = path.results
    for index, run in enumerate(path.runs):
        for result in host.storage.query_eval_results_by_run(run.id, TOYHOST_SCOPE):
            score = result.judge_score(FAITHFULNESS_DIM)
            assert score is not None
            rate_result(
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
    kind = judge_kind(
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
    host.storage.save_eval_run(run)
    cases = host.storage.load_test_cases_by_ids(list(case_ids), TOYHOST_SCOPE)
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
    host.storage.save_template(template)
    report = freeze_judge_cases(
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
    host.storage.save_judge_config(harsh)
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
        result for run in runs for result in host.storage.query_eval_results_by_run(run.id, TOYHOST_SCOPE)
    ]
    return campaign


async def test_freezing_a_judged_run_makes_one_case_per_judged_output_carrying_what_its_judge_read() -> None:
    judge = _ArmAwareJudge()
    host = toyhost_host(clients=lambda role, model, *, temperature=None: judge)
    source = await _judged_source(host)
    template = _judge_template()
    host.storage.save_template(template)
    run_ids = sorted({result.eval_run_id for result in source})

    report = freeze_judge_cases(host.storage, template=template, run_ids=run_ids, scope_id=TOYHOST_SCOPE)

    assert len(report.cases) == len(source) and all(case.created for case in report.cases)
    by_result = {result.id: result for result in source}
    for frozen in report.cases:
        stored = host.storage.load_test_case(frozen.test_case_id, TOYHOST_SCOPE)
        assert stored is not None and stored.template_id == template.id and stored.stratum == FAITHFULNESS_DIM
        case = judge_case_of(stored)
        assert case is not None
        result = by_result[case.source.result_id]
        trace = host.storage.load_eval_trace(result.id, TOYHOST_SCOPE)
        assert trace is not None
        # Exactly what the first judge read, and what it answered, and what the person said.
        assert case.evidence == trace.judge_evidence
        assert case.criterion is not None and case.criterion.name == FAITHFULNESS_DIM
        first = result.judge_score(FAITHFULNESS_DIM)
        assert first is not None and case.source.first_score == first.score
        assert [label.rater for label in case.labels] == [_RATER]
    # Freezing the same outputs again names the stored cases rather than minting more.
    again = freeze_judge_cases(host.storage, template=template, run_ids=run_ids, scope_id=TOYHOST_SCOPE)
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
    builtin = readings.reading(JudgeKey(FAITHFULNESS_DIM, "ordinal", None, None, DEFAULT_JUDGE_TEMPERATURE))
    harsh = readings.reading(JudgeKey(FAITHFULNESS_DIM, "ordinal", None, campaign.harsh.id, DEFAULT_JUDGE_TEMPERATURE))
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
    stored = campaign.host.storage.load_test_case(campaign.case_ids[0], TOYHOST_SCOPE)
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
