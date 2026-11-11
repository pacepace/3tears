"""``Eval`` with a judge: a rubric scored by the engine's judge, its evidence and spend recorded as any judged run's.

The judge here is scripted: it reads the dimension it was asked for off the system prompt and the
answer off the user prompt, and scores the answer by what it says, so every score below is a function
of the evidence the callable kind rendered and the engine's judge service forwarded. The tests that
count calls pass ``k=1``, so each count is one per case per dimension. The rubric is scored 1 to 5,
stated on each dimension (a rubric given as a mapping is pass/fail).
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import pytest

from threetears.evals import Eval, Example, Judge
from threetears.evals.providers.failures import error_from_response
from threetears.evals.schema import JudgedArtifact, RubricDim, StopReason
from threetears.evals.testing import RecordedCompletions
from threetears.evals.kernel.host import EvalHost, KindContract, TemplateChecks
from threetears.evals.quick.one_call import (
    CALLABLE_KIND_CONTRACT,
    JUDGED_CALLABLE_KIND,
    JUDGED_CALLABLE_KIND_CONTRACT,
    callable_host,
)
from threetears.evals.analysis import campaign_report, campaign_report_markdown, summarize_run
from threetears.evals.run import (
    get_result_trace,
    get_template,
    list_results,
    list_runs,
    list_templates,
    update_template,
)
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.report_support import page_violations
from packages.evals.tests.clock_support import frozen_clock


@pytest.fixture(autouse=True)
def _frozen_clock() -> Iterator[None]:
    """Every stamp reads a frozen clock, not the wall's, that steps on with each read: a template edited after its run
    is stamped after it."""
    with frozen_clock(step_seconds=0.001):
        yield


SCOPE = "eval-judged-tests"
CASES = [{"question": "What is two plus two?"}, {"question": "What is the capital of France?"}]
ANSWERS = {"What is two plus two?": "4", "What is the capital of France?": "I am not sure."}
RUBRIC = {"helpful": "The answer resolves the question.", "honest": "The answer claims nothing it cannot support."}
JUDGE_MODEL = "scripted/judge"
JUDGE_COST_USD = 0.0003

#: Where an eval with no host of its own keeps its runs in these tests: memory, for the test alone.
MEMORY = ":memory:"


async def answer(case: Mapping[str, Any]) -> str:
    """Answer a question in a few words."""
    return ANSWERS[case["question"]]


def short(case: Mapping[str, Any], answer: Any) -> bool:
    return len(str(answer).split()) <= 5


@dataclass(frozen=True)
class _Reply:
    """One scripted judge reply, in the attribute names the engine reads."""

    content: str
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    price_source: str | None
    model: str = JUDGE_MODEL
    served_model: str | None = "scripted/judge-2026"
    reasoning_tokens: int | None = None
    stop_reason: StopReason = "end_turn"
    temperature: float | None = None


# parity-with: threetears.evals.schema.CompletionClient
class _FakeJudgeClient:
    """A judge that gives a committed answer 5 and a hedge 2, or a reply set up per test."""

    def __init__(self, *, priced: bool = True, reply: str | None = None, stop_reason: StopReason = "end_turn") -> None:
        self.model_name = JUDGE_MODEL
        self.calls: list[tuple[str, str]] = []
        self.aclose_calls = 0
        self._priced = priced
        self._reply = reply
        self._stop_reason = stop_reason

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        self.calls.append((system, user))
        match = re.search(r'single key "([^"]+)"', system)
        assert match is not None, "the judge's system prompt names no dimension"
        output = user.split("# Output under review\n", 1)[1]
        score = 2 if "not sure" in output else 5
        content = self._reply or json.dumps({"reasoning": f"read {output!r}", "criteria_scores": {match[1]: score}})
        return _Reply(
            content=content,
            input_tokens=len(user) // 4,
            output_tokens=30,
            cost_usd=JUDGE_COST_USD if self._priced else None,
            price_source="scripted rate" if self._priced else None,
            stop_reason=self._stop_reason,
        )

    async def aclose(self) -> None:
        self.aclose_calls += 1

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


def _ordinal(rubric: Mapping[str, str]) -> list[RubricDim]:
    """A rubric's dimensions scored 1 to 5, each named under ``answer.`` as a Judge names a bare key."""
    return [RubricDim(name=f"answer.{name}", description=said, scale="ordinal") for name, said in rubric.items()]


def _judge(client: _FakeJudgeClient, **overrides: Any) -> Judge:
    rubric = overrides.pop("rubric", RUBRIC)
    return Judge(client, _ordinal(rubric) if isinstance(rubric, Mapping) else rubric, **overrides)


async def _summary(cases: Any, runner: Any, graders: list[Any], **run: Any) -> Any:
    """The one arm's summary of an eval run with ``k=1``, in memory unless a host is named."""
    if "host" not in run:
        run["store"] = MEMORY
    return (await Eval(cases, runner, graders, k=1).run(scope=SCOPE, **run)).summary


async def test_each_dimension_is_scored_by_the_engines_judge_and_summarised_beside_the_scorers() -> None:
    client = _FakeJudgeClient()
    host = callable_host([short])
    summary = await _summary(CASES, answer, [short, _judge(client)], host=host)

    assert (summary.status, summary.n_scored, summary.n_excluded) == ("completed", 2, 0)
    assert [measure.name for measure in summary.measures] == ["short"]
    by_name = {dimension.name: dimension for dimension in summary.judged}
    assert list(by_name) == ["answer.helpful", "answer.honest"]
    for dimension in by_name.values():
        assert (dimension.scale, dimension.n, dimension.minimum, dimension.maximum) == ("ordinal", 2, 2.0, 5.0)
        assert dimension.mean == pytest.approx(3.5) and dimension.cannot_tell == 0
    assert len(client.calls) == 4, "one call per case per dimension, and no conversation axis"
    assert summary.role_spend("judge").calls == 4
    assert summary.role_spend("judge").cost_usd == pytest.approx(4 * JUDGE_COST_USD)

    rendered = summary.render()
    assert "  answer helpful (judged), scored 1-5: mean 3.5 (n=2, min 2, max 5)" in rendered
    assert "  answer honest (judged), scored 1-5: mean 3.5 (n=2, min 2, max 5)" in rendered
    assert "  judge spend: $0.0012 over 4 call(s)" in rendered  # 4 x $0.0003

    (run,) = off_loop(list_runs, host, SCOPE)
    assert run.judge_model == JUDGE_MODEL
    template = off_loop(get_template, host, summary.template_id or "", SCOPE)
    assert template.candidate_kind == JUDGED_CALLABLE_KIND
    assert [dim.name for dim in template.rubric] == ["answer.helpful", "answer.honest"]
    for result in off_loop(list_results, host.storage, summary.run_id, SCOPE):
        assert result.judge_model == JUDGE_MODEL and result.judge_error is None
        assert {score.served_model for score in result.rubric_scores} == {"scripted/judge-2026"}
        (judge_row,) = [row for row in result.usage if row.role == "judge"]
        assert judge_row.cost_usd == pytest.approx(2 * JUDGE_COST_USD)
        assert judge_row.price_source == "scripted rate"


async def test_the_judge_reads_the_answer_against_the_case_material_and_the_trace_keeps_it() -> None:
    client = _FakeJudgeClient()
    host = callable_host()
    judge = _judge(client, case_material=lambda case: f"Reference: arithmetic.\nQuestion: {case['question']}")
    summary = await _summary(CASES[:1], answer, [judge], host=host)

    _, user = client.calls[0]
    assert "Reference: arithmetic.\nQuestion: What is two plus two?" in user
    assert user.endswith("# Output under review\n4")
    (result,) = off_loop(list_results, host.storage, summary.run_id, SCOPE)
    trace = off_loop(get_result_trace, host.storage, result)
    assert trace is not None and trace.judged_artifact is JudgedArtifact.DOCUMENT
    assert trace.judge_evidence is not None
    assert trace.judge_evidence.case_material == "Reference: arithmetic.\nQuestion: What is two plus two?"
    assert trace.judge_evidence.artifact == "4"


async def test_without_case_material_the_judge_reads_the_case_as_json_and_a_non_string_answer_as_json() -> None:
    client = _FakeJudgeClient()

    async def structured(case: Mapping[str, Any]) -> dict[str, Any]:
        return {"answer": 4}

    await _summary(CASES[:1], structured, [_judge(client)])
    _, user = client.calls[0]
    assert '{\n  "question": "What is two plus two?"\n}' in user
    assert user.endswith('# Output under review\n{\n  "answer": 4\n}')


async def test_without_case_material_a_classifier_s_judge_never_reads_the_answer_key() -> None:
    """The field holding the expected label is left out of what the judge reads, and the trace keeps what was sent."""
    client = _FakeJudgeClient()
    host = callable_host()
    cases = [{"ticket": "I was charged twice.", "queue": "billing", "channel": "email"}]

    async def classify(case: Mapping[str, Any]) -> str:
        return "billing"

    summary = await _summary(cases, classify, ["queue", _judge(client)], host=host)
    sent = {user.split("# Output under review\n", 1)[0] for _, user in client.calls}
    assert len(sent) == 1 and len(client.calls) == 2
    (material_sent,) = sent
    assert '"ticket": "I was charged twice."' in material_sent and '"channel": "email"' in material_sent
    assert '"queue"' not in material_sent, "the judge read the expected label it is grading without"
    (result,) = off_loop(list_results, host.storage, summary.run_id, SCOPE)
    trace = off_loop(get_result_trace, host.storage, result)
    assert trace is not None and trace.judge_evidence is not None
    assert trace.judge_evidence.case_material == '{\n  "channel": "email",\n  "ticket": "I was charged twice."\n}'
    assert trace.judge_evidence.case_material in material_sent, "the trace records exactly the material sent"


async def test_a_reference_answer_reaches_the_judge_only_through_case_material() -> None:
    client = _FakeJudgeClient()
    cases = [{"ticket": "I was charged twice.", "queue": "billing"}]

    async def classify(case: Mapping[str, Any]) -> str:
        return "billing"

    judge = _judge(client, case_material=lambda case: f"Ticket: {case['ticket']}\nReference queue: {case['queue']}")
    await _summary(cases, classify, ["queue", judge])
    assert all("Reference queue: billing" in user for _, user in client.calls)


async def test_the_judges_client_is_lent_never_closed_so_one_client_serves_two_runs() -> None:
    client = _FakeJudgeClient()
    host = callable_host()
    judge = _judge(client)
    first = await _summary(CASES, answer, [judge], host=host)
    second = await _summary(CASES, {"again": answer}, [judge], host=host)
    assert client.aclose_calls == 0
    assert first.template_id == second.template_id
    assert first.n_scored == second.n_scored == 2


async def test_a_judge_that_prices_nothing_leaves_its_spend_unknown_never_zero() -> None:
    summary = await _summary(CASES, answer, [_judge(_FakeJudgeClient(priced=False))])
    assert summary.role_spend("judge").calls == 4 and summary.role_spend("judge").cost_usd is None
    assert "  judge spend: unpriced over 4 call(s): its client reported no price for them" in summary.render()


async def test_a_judge_reply_that_will_not_parse_excludes_the_cell_and_says_so() -> None:
    summary = await _summary(CASES[:1], answer, [short, _judge(_FakeJudgeClient(reply="no json here"))])
    assert (summary.n_scored, summary.n_candidate_failed, summary.n_excluded) == (0, 0, 1)
    assert summary.errors and all(": judge: " in error for error in summary.errors)
    # Lost to the judge's call alone, which a re-judge recovers: the run completes, never "measured nothing".
    assert summary.status == "completed"
    # Every attempt the judge made was paid for, parsed or not, so its spend is still counted.
    assert (
        summary.role_spend("judge").calls >= 2
        and summary.role_spend("judge").cost_usd is not None
        and summary.role_spend("judge").cost_usd > 0
    )


async def test_a_judge_that_cannot_tell_is_counted_apart_and_excludes_nothing() -> None:
    reply = json.dumps({"reasoning": "the answer is not shown", "criteria_scores": {"answer.helpful": "cannot_tell"}})
    summary = await _summary(CASES[:1], answer, [_judge(_FakeJudgeClient(reply=reply), rubric={"helpful": "Helps."})])
    (dimension,) = summary.judged
    assert (dimension.name, dimension.n, dimension.mean, dimension.cannot_tell) == ("answer.helpful", 0, None, 1)
    assert summary.n_excluded == 0
    assert "answer helpful (judged): no result carries a score, the judge could not tell on 1" in summary.render()


async def test_a_run_the_judge_could_not_tell_on_anywhere_still_completes() -> None:
    """A judge that could not tell read evidence the run did measure: no cell is an infra loss, so nothing fails."""
    reply = json.dumps({"reasoning": "the answer is not shown", "criteria_scores": {"answer.helpful": "cannot_tell"}})
    summary = await _summary(CASES, answer, [_judge(_FakeJudgeClient(reply=reply), rubric={"helpful": "Helps."})])
    (dimension,) = summary.judged
    assert dimension.n == 0 and dimension.cannot_tell == len(CASES)
    assert summary.status == "completed" and summary.n_excluded == 0
    assert not any("the run measured nothing" in error for error in summary.errors)


async def test_a_run_whose_every_judge_reply_was_cut_off_at_its_cap_still_completes() -> None:
    """A cut-off judge reply is the judge's request running out of room over evidence the run did measure."""
    summary = await _summary(CASES, answer, [_judge(_FakeJudgeClient(stop_reason="max_tokens"))])
    assert summary.status == "completed"
    assert (summary.n_scored, summary.n_excluded) == (0, len(CASES))
    assert not any("the run measured nothing" in error for error in summary.errors)


async def test_a_run_with_one_cell_declined_and_the_other_judge_cut_off_still_completes() -> None:
    """One cell lost to the provider, one to a cut-off judge reply: the second measured the candidate, so it completes."""
    declined = error_from_response("anthropic", 529, json.dumps({"error": {"message": "Overloaded"}}))

    async def runner(case: Mapping[str, Any]) -> str:
        if case == CASES[0]:
            raise declined
        return await answer(case)

    summary = await _summary(CASES, runner, [_judge(_FakeJudgeClient(stop_reason="max_tokens"))])
    assert summary.status == "completed"
    assert (summary.n_scored, summary.n_excluded) == (0, 2)
    assert not any("the run measured nothing" in error for error in summary.errors)


async def test_a_rubric_of_engine_dimensions_keeps_their_names_and_scales() -> None:
    client = _FakeJudgeClient(
        reply=json.dumps({"reasoning": "fine", "criteria_scores": {"faq.correct": "pass"}}),
    )
    dims = [RubricDim(name="faq.correct", description="The answer is correct.", scale="pass_fail")]
    summary = await _summary(CASES[:1], answer, [_judge(client, rubric=dims)])
    (dimension,) = summary.judged
    assert (dimension.name, dimension.scale, dimension.mean) == ("faq.correct", "pass_fail", 1.0)
    assert "faq correct (judged), scored pass/fail: mean 1" in summary.render()


async def test_a_judged_eval_and_an_unjudged_one_over_the_same_cases_are_two_templates() -> None:
    host = callable_host([short])
    unjudged = await _summary(CASES, answer, [short], host=host)
    judged = await _summary(CASES, answer, [short, _judge(_FakeJudgeClient())], host=host)
    other_rubric = _judge(_FakeJudgeClient(), rubric={"helpful": "Helps."})
    reworded = await _summary(CASES, answer, [short, other_rubric], host=host)
    assert len({unjudged.template_id, judged.template_id, reworded.template_id}) == 3
    assert unjudged.judged == [] and "judge spend" not in unjudged.render()


# --- the intent the judge reads -------------------------------------------------------------------


def _intents_read(client: _FakeJudgeClient) -> set[str]:
    """The intent line of every prompt the judge was sent."""
    return {line for _, user in client.calls for line in user.splitlines() if line.startswith("**Intent:**")}


async def test_the_judge_reads_the_runners_docstring_as_the_intent_and_the_summary_names_it() -> None:
    client = _FakeJudgeClient()
    host = callable_host()
    summary = await _summary(CASES, answer, [_judge(client)], host=host)

    assert _intents_read(client) == {"**Intent:** Answer a question in a few words."}
    assert off_loop(get_template, host, summary.template_id or "", SCOPE).intent == "Answer a question in a few words."
    assert summary.intent_source == "from answer's docstring"
    assert "  intent (from answer's docstring): Answer a question in a few words." in summary.render()


async def test_with_no_docstring_the_summary_says_the_intent_is_a_generic_default() -> None:
    client = _FakeJudgeClient()

    async def undocumented(case: Mapping[str, Any]) -> str:
        return ANSWERS[case["question"]]

    summary = await _summary(CASES, undocumented, [_judge(client)])
    default = "Answer each case so that the judge and every scorer grade the answer well."
    assert _intents_read(client) == {f"**Intent:** {default}"}
    assert f"  intent (a generic default: undocumented has no docstring): {default}" in summary.render()


async def test_a_summary_read_back_keeps_the_intent_the_judge_read_but_not_where_it_came_from() -> None:
    host = callable_host()
    run = await _summary(CASES, answer, [_judge(_FakeJudgeClient())], host=host)
    read_back = off_loop(summarize_run, host, run.run_id, SCOPE)
    assert (read_back.intent, read_back.intent_source) == ("Answer a question in a few words.", None)
    assert "  intent: Answer a question in a few words." in read_back.render()


async def test_two_judged_runs_stating_different_intents_each_keep_the_template_their_judge_read() -> None:
    host = callable_host()

    async def reworded(case: Mapping[str, Any]) -> str:
        """Reworded."""
        return ANSWERS[case["question"]]

    first = await _summary(CASES, answer, [_judge(_FakeJudgeClient())], host=host)
    second = await _summary(CASES, reworded, [_judge(_FakeJudgeClient())], host=host)
    assert first.template_id != second.template_id
    assert off_loop(summarize_run, host, first.run_id, SCOPE).intent == "Answer a question in a few words."
    assert off_loop(summarize_run, host, second.run_id, SCOPE).intent == "Reworded."


async def test_a_template_edited_since_the_run_no_longer_says_what_its_judge_read_so_no_intent_is_shown() -> None:
    host = callable_host()
    run = await _summary(CASES, answer, [_judge(_FakeJudgeClient())], host=host)
    off_loop(
        update_template,
        dataclasses.replace(
            host,
            template_checks=TemplateChecks(
                require_known_tools_allowed=lambda _tools: None,
                refuse_undeclared_world_seed=lambda _template: None,
                refuse_undeliverable_template=lambda _template: None,
                kinds={},
                refuse_inadmissible_template_case=lambda _case: None,
            ),
        ),
        run.template_id or "",
        SCOPE,
        {"intent": "Edited after the run."},
    )
    assert off_loop(summarize_run, host, run.run_id, SCOPE).intent is None


async def test_an_unjudged_run_shows_no_intent_since_nothing_that_grades_it_reads_one() -> None:
    summary = await _summary(CASES, answer, [short])
    assert summary.intent is None and "intent" not in summary.render()


# --- refusals ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "said"),
    [
        ({"model_name": " "}, "a client that names its model"),
        ({"rubric": {}}, "at least one rubric dimension"),
        ({"rubric": {"a.b.c": "Too many contexts."}}, "a.b.c"),
        ({"rubric": {"helpful": ""}}, "description"),
        ({"rubric": {"helpful": "Helps.", "answer.helpful": "Helps again."}}, "answer.helpful more than once"),
    ],
    ids=["a blank model", "no dimension", "a doubly namespaced name", "a blank description", "one name twice"],
)
def test_a_judge_no_run_could_record_is_refused_where_it_is_built(overrides: dict[str, Any], said: str) -> None:
    client = _FakeJudgeClient()
    client.model_name = overrides.pop("model_name", JUDGE_MODEL)
    with pytest.raises(ValueError, match=said):
        Judge(client, overrides.get("rubric", RUBRIC))


def _host_whose_judged_contract_is(contract: KindContract | None) -> EvalHost:
    host = callable_host()
    kinds = (CALLABLE_KIND_CONTRACT,) if contract is None else (CALLABLE_KIND_CONTRACT, contract)
    return dataclasses.replace(host, profile=dataclasses.replace(host.profile, kinds=kinds))


@pytest.mark.parametrize(
    ("contract", "said"),
    [
        (None, "has no contract for the 'callable-judged' kind"),
        (KindContract(JUDGED_CALLABLE_KIND), "a contract that declares no seats"),
        (KindContract(JUDGED_CALLABLE_KIND, seats=frozenset()), "does not seat the judge"),
        (KindContract(JUDGED_CALLABLE_KIND, seats=frozenset({"judge", "simulator"})), "seats simulator"),
    ],
    ids=["no contract", "no seats", "the judge unseated", "the simulator seated"],
)
async def test_a_callers_host_that_has_not_declared_the_judged_kinds_rig_is_refused(
    contract: KindContract | None, said: str
) -> None:
    host = _host_whose_judged_contract_is(contract)
    with pytest.raises(ValueError, match=said):
        await Eval(CASES, answer, [_judge(_FakeJudgeClient())]).run(host=host, scope=SCOPE)
    assert off_loop(list_templates, host.storage, SCOPE) == []


async def test_a_host_seating_the_judge_by_its_pins_is_accepted() -> None:
    pins = frozenset(
        {
            "judge_model",
            "judge_temperature",
            "judge_request_settings",
            "judge_sent_settings",
            "judge_dim_divergence",
            "judge_config_ids",
        }
    )
    host = _host_whose_judged_contract_is(KindContract(JUDGED_CALLABLE_KIND, seats=pins))
    summary = await _summary(CASES, answer, [_judge(_FakeJudgeClient())], host=host)
    assert summary.status == "completed"


def test_the_judged_contract_seats_the_judge_and_nothing_else() -> None:
    assert JUDGED_CALLABLE_KIND_CONTRACT.seats == frozenset({"judge"})
    assert {contract.kind for contract in callable_host().profile.kinds} == {"callable", JUDGED_CALLABLE_KIND}


async def test_the_judges_client_factory_lends_only_the_judge_role_on_its_own_model() -> None:
    clients = _judge(_FakeJudgeClient()).clients()
    lent = clients("judge", JUDGE_MODEL)
    assert lent.model_name == JUDGE_MODEL and lent.price_ceiling(system="", user="") is None
    await lent.aclose()
    with pytest.raises(ValueError, match="calls no model in the 'simulator' role"):
        clients("simulator", None)
    with pytest.raises(ValueError, match="this judge's client calls 'scripted/judge'"):
        clients("judge", "another/model")


async def test_a_judge_s_examples_are_seeded_as_authored_cases_its_calibration_starts_from() -> None:
    from threetears.evals.kernel import judge_case_of

    client = _FakeJudgeClient()
    host = callable_host([short])
    examples = [
        Example("helpful", "Probably 4, but check.", 2, "a hedge on arithmetic does not resolve it", case=CASES[0]),
        Example("honest", "I am not sure.", 5, "saying so is the honest answer it looks like a failure of"),
    ]

    summary = await _summary(CASES, answer, [_judge(client, examples=examples)], host=host)

    assert len(client.calls) == 4 + 2, "one call per case per dimension, and one per example per repeat to grade it"
    seeded = f"{summary.template_id}-judge"
    (case_set,) = off_loop(host.storage.query_case_sets, SCOPE, name=seeded)
    assert case_set.template_id == seeded and len(case_set.test_case_ids) == 2
    stored = off_loop(host.storage.load_test_cases_by_ids, case_set.test_case_ids, SCOPE)
    cases = [judge_case_of(test_case) for test_case in stored]
    by_dim = {case.dim: case for case in cases if case is not None}
    assert set(by_dim) == {"answer.helpful", "answer.honest"}
    helpful = by_dim["answer.helpful"]
    template = off_loop(get_template, host, summary.template_id or "", SCOPE)
    assert helpful.intent == template.intent and helpful.criterion == template.rubric[0]
    assert helpful.evidence.artifact == "Probably 4, but check." and "two plus two" in helpful.evidence.case_material
    assert [(label.score, label.rating_id) for label in helpful.labels] == [(2, None)]
    # Seeding the same examples again names the stored cases, and the set's version stands.
    await _summary(CASES, answer, [_judge(client, examples=examples)], host=host)
    assert [entry.version for entry in off_loop(host.storage.query_case_sets, SCOPE, name=seeded)] == [1]


def test_an_example_its_dimension_cannot_hold_or_that_names_none_is_refused_when_the_judge_is_built() -> None:
    with pytest.raises(
        ValueError,
        match=r"judge example 0: criterion 'answer.helpful' is answered on the ordinal scale .* cannot hold the verdict 'fail'",
    ):
        _judge(_FakeJudgeClient(), examples=[Example("helpful", "4", "fail", "must fail")])
    with pytest.raises(ValueError, match="judge example 0 is graded on 'tone', which is no dimension of the rubric"):
        _judge(_FakeJudgeClient(), examples=[Example("tone", "4", 1, "no such dimension")])
    pass_fail = [RubricDim(name="answer.correct", description="The answer is right.", scale="pass_fail")]
    assert _judge(_FakeJudgeClient(), rubric=pass_fail, examples=[Example("correct", "5", "fail", "it is 4")])


# --- a judge's examples, graded in the run ---------------------------------------------------------------------------

#: Two examples on the ordinal rubric: a hedge the scripted judge scores 2, as it must, and an honest "not sure" it
#: must score 5 and scores 2: a miss.
EXAMPLES = [
    Example("helpful", "Probably 4, but I am not sure.", 2, "a hedge on arithmetic does not resolve it", case=CASES[0]),
    Example("honest", "I am not sure.", 5, "saying so is the honest answer", case=CASES[1]),
]


async def test_each_example_is_graded_k_times_through_the_call_the_run_s_answers_are_graded_by() -> None:
    client = _FakeJudgeClient()
    result = await Eval(CASES, answer, [_judge(client, examples=EXAMPLES)], k=2).run(store=MEMORY, scope=SCOPE)

    run_calls, example_calls = client.calls[:8], client.calls[8:]
    assert len(run_calls) == 2 * 2 * 2 and len(example_calls) == 2 * 2, "each example once per repeat, after the run"
    # The run's own prompt for the case and dimension, with the example's answer in the reply's place: the same
    # instructions, criterion, scale and intent, the case as its variation, and its material rendered as a run's is.
    for (system, user), example in zip(example_calls[::2], EXAMPLES, strict=True):
        dim = f"answer.{example.dimension}"
        (ran,) = {
            (run_system, run_user)
            for run_system, run_user in run_calls
            if f'"{dim}"' in run_system
            and json.dumps(example.case["question"]) in run_user
            and run_user.endswith(f"# Output under review\n{ANSWERS[example.case['question']]}")
        }
        assert system == ran[0]
        assert user == ran[1].removesuffix(ANSWERS[example.case["question"]]) + example.answer
    assert example_calls[0] == example_calls[1] and example_calls[2] == example_calls[3]

    read = result.judge_examples
    assert read is not None and read.repeats == 2
    assert [(grade.dimension, grade.expected, grade.verdicts) for grade in read.grades] == [
        ("answer.helpful", "2", ("2", "2")),
        ("answer.honest", "5", ("2", "2")),
    ]
    assert [grade.as_it_must for grade in read.grades] == [True, False]
    assert (read.as_they_must, read.misses, read.unread) == (1, (read.grades[1],), ())


async def test_a_miss_is_said_with_its_verdicts_in_the_print_and_on_the_page_beside_the_judge_s_calibration() -> None:
    result = await Eval(CASES, answer, [_judge(_FakeJudgeClient(), examples=EXAMPLES)], k=2).run(
        store=MEMORY, scope=SCOPE
    )

    miss = '"I am not sure." graded 2 on honest, must 5 (saying so is the honest answer)'
    lines = str(result).splitlines()
    said = "judge: 1 of 2 examples graded as they must be on every one of 2 repeats ($0.0012 over 4 call(s))"
    at = lines.index(said)
    assert lines[at - 1].startswith("judge: 0 labels: judge uncalibrated"), "read beside its labels, never pooled"
    assert lines[at + 1] == f"  missed: {miss}"
    (block,) = [block for block in result.page.blocks if block.anchor == "judge-examples"]
    assert block.section == "provenance" and block.source == "calibration"
    assert block.text == (
        "The judge graded 1 of 2 of its examples as they must be on every one of 2 repeats ($0.0012 over 4 call(s)). "
        f"Missed: {miss}."
    ), "the page states what the examples cost, as the print does"
    # Laid out after the test setup table, whose row says how far the judge's scores can be read.
    provenance = [block for block in result.page.blocks if block.section == "provenance"]
    assert provenance[-1] is block and any(getattr(table, "name", None) == "test_setup" for table in provenance)
    assert block.text in result.render(full=True)
    assert page_violations(result.report) == []


async def test_the_examples_spend_is_the_judge_s_ledgered_out_of_run_and_beside_every_arm_s_results() -> None:
    result = await Eval(CASES, answer, [_judge(_FakeJudgeClient(), examples=EXAMPLES)], k=1).run(
        store=MEMORY, scope=SCOPE
    )
    read = result.judge_examples
    assert read is not None
    assert (read.spend.role, read.spend.calls, read.stopped) == ("judge", 2, None)
    assert read.spend.cost_usd == pytest.approx(2 * JUDGE_COST_USD)
    assert result.summary.role_spend("judge").calls == 4, "the arm's results hold their own calls and no example's"

    # Made outside any run, so each is a row of the out-of-run ledger, under the judge's purpose and the template.
    rows = off_loop(result.host.storage.query_out_of_run_spend, SCOPE, purpose="judge")
    assert len(rows) == 2 and all(row.outcome == "completed" for row in rows)
    assert {(row.model, row.template_id, row.run_id) for row in rows} == {(JUDGE_MODEL, result.template_id, None)}
    assert [row.cost_usd for row in rows] == [JUDGE_COST_USD] * 2
    assert math.fsum(row.cost_usd or 0.0 for row in rows) == pytest.approx(read.spend.cost_usd)


async def test_the_examples_are_held_to_the_eval_s_cap_on_landed_spend_so_the_call_that_crosses_it_goes_over() -> None:
    """The eval's cap is read before each example's call against the spend landed so far, as a run's cap is read
    before each cell: the call that crosses it lands whole, so the eval ends over its cap, and the reading says so."""
    # The arms land 4 calls ($0.0012) of a $0.0013 cap: the first example's call is made inside what is left, lands the
    # eval at $0.0015, over the cap, and no other is made.
    capped = Eval(CASES, answer, [_judge(_FakeJudgeClient(), examples=EXAMPLES)], k=1, max_cost_usd=0.0013)
    result = await capped.run(store=MEMORY, scope=SCOPE)
    assert result.summary.role_spend("judge").calls == 4, "the arm's results hold their own calls and no example's"
    read = result.judge_examples
    over = "the eval's cost cap stopped the grading, the eval having landed $0.0015, over its $0.0013 cap"
    assert read is not None and read.stopped == over and read.spend.calls == 1
    assert [grade.verdicts for grade in read.grades] == [("2",), ()]
    assert read.lines()[0] == (
        f"judge: 1 of 2 examples graded as they must be; 1 not graded in full, since {over} ($0.00030 over 1 call(s))"
    )
    assert len(off_loop(result.host.storage.query_out_of_run_spend, SCOPE, purpose="judge")) == 1


async def test_under_a_cap_arms_whose_spend_nobody_priced_stop_the_examples_and_the_line_says_that_not_the_cap() -> (
    None
):
    """A cap cannot bound spend nobody priced, so an enforcing run stops at it; the examples stop at it too, and the
    reading names it, never the cap as though the cap had been reached."""
    capped = Eval(CASES, answer, [_judge(_FakeJudgeClient(priced=False), examples=EXAMPLES)], k=1, max_cost_usd=1.0)
    read = (await capped.run(store=MEMORY, scope=SCOPE)).judge_examples
    assert read is not None and read.spend.calls == 0
    assert read.stopped == "the arms spent what nobody priced, which the eval's cost cap cannot bound"
    assert read.lines() == [
        "judge: 2 examples not graded, since the arms spent what nobody priced, which the eval's cost cap cannot bound"
    ]
    assert "cost cap stopped" not in read.sentence()

    # Uncapped, nothing bounds the grading, and the unpriced calls are said as such.
    read = (
        await Eval(CASES, answer, [_judge(_FakeJudgeClient(priced=False), examples=EXAMPLES)], k=1).run(
            store=MEMORY, scope=SCOPE
        )
    ).judge_examples
    assert read is not None and read.stopped is None and read.spend.calls == 2 and read.spend.cost_usd is None


async def test_an_example_the_judge_s_call_failed_on_is_not_graded_in_full_never_a_miss() -> None:
    # parity-with: threetears.evals.schema.CompletionClient
    class _FakeRefusingJudge(_FakeJudgeClient):
        """A judge whose call fails on one example's answer, as a recording that does not hold it fails."""

        async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
            if user.endswith("\nI am not sure."):
                raise RuntimeError("not recorded")
            return await super().generate(system=system, user=user, response_format=response_format)

    read = (
        await Eval(CASES[:1], answer, [_judge(_FakeRefusingJudge(), examples=EXAMPLES)], k=1).run(
            store=MEMORY, scope=SCOPE
        )
    ).judge_examples
    assert read is not None
    (_, failed) = read.grades
    assert failed.verdicts == () and len(failed.errors) == 1 and "RuntimeError" in failed.errors[0]
    assert (read.as_they_must, read.misses, read.unread) == (1, (), (failed,))
    assert read.lines()[0].startswith(
        "judge: 1 of 2 examples graded as they must be; 1 not graded in full, since the judge's call failed: "
    )


async def test_a_judge_with_no_examples_makes_no_extra_call_and_its_result_and_page_are_as_they_were() -> None:
    client = _FakeJudgeClient()
    result = await Eval(CASES, answer, [_judge(client)], k=1).run(store=MEMORY, scope=SCOPE)
    assert len(client.calls) == 4 and result.judge_examples is None
    assert not [line for line in str(result).splitlines() if "examples graded" in line]
    # The oracle is the engine's own page for the campaign, read from the store: the result's is it, byte for byte.
    stored = off_loop(campaign_report, result.host, result.campaign_id, result.scope_id)
    assert campaign_report_markdown(result.page) == campaign_report_markdown(stored)


async def test_a_judge_with_examples_adds_one_line_to_the_engine_s_page_and_nothing_else() -> None:
    result = await Eval(CASES, answer, [_judge(_FakeJudgeClient(), examples=EXAMPLES)], k=1).run(
        store=MEMORY, scope=SCOPE
    )
    stored = off_loop(campaign_report, result.host, result.campaign_id, result.scope_id)
    added = [block for block in result.page.blocks if block not in stored.blocks]
    assert [block.anchor for block in added] == ["judge-examples"]
    without = result.page.model_copy(update={"blocks": [block for block in result.page.blocks if block not in added]})
    assert campaign_report_markdown(without) == campaign_report_markdown(stored)


# --- a judge's recording, replayed ------------------------------------------------------------------------------------


async def other_answer(case: Mapping[str, Any]) -> str:
    """Answer a question in a few words, as :func:`answer` does."""
    return ANSWERS[case["question"]]


async def test_two_arms_sending_the_judge_one_prompt_replay_the_verdicts_they_were_recorded_with_every_time(
    tmp_path: Path,
) -> None:
    """Two arms giving one answer send the judge one prompt, so their repeats share each recorded call's index. A run's
    cells start in an order shuffled by its fresh id, several at once: a replay that served the first asker the first
    call swapped the arms' verdicts between replays. Each cell is served the call recorded for it."""

    # parity-with: threetears.evals.schema.CompletionClient
    class _FakeDrawingJudge(_FakeJudgeClient):
        """A judge whose every call scores the next of 1 to 5 in turn, so no two neighbouring calls agree."""

        async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
            reply = await super().generate(system=system, user=user, response_format=response_format)
            match = re.search(r'single key "([^"]+)"', system)
            assert match is not None
            score = len(self.calls) % 5 + 1
            return dataclasses.replace(
                reply, content=json.dumps({"reasoning": "drawn", "criteria_scores": {match[1]: score}})
            )

    recording = tmp_path / "judge.jsonl"

    async def verdicts(client: Any) -> dict[str, list[tuple[str, int, list[tuple[str, int]]]]]:
        twin = Eval(CASES, {"a": answer, "b": other_answer}, [_judge(client)], k=2, name="twin")
        result = await twin.run(store=MEMORY, scope=SCOPE)
        return {
            arm: sorted(
                (row.case_id, row.repeat, [(grade.dimension, grade.score) for grade in row.judged])
                for row in result.results(arm)
            )
            for arm in ("a", "b")
        }

    recorded = await verdicts(RecordedCompletions.record(_FakeDrawingJudge(), recording, settings=None))
    assert len({str(rows) for rows in recorded.values()}) == 2, "the arms were recorded with different verdicts"
    for _ in range(2):
        assert await verdicts(RecordedCompletions.replay(recording, settings=None)) == recorded


# --- an example is judged as a run's answer is ------------------------------------------------------------------------


async def test_on_a_classifier_eval_an_example_s_judge_never_reads_the_label_its_case_expects() -> None:
    """A cell's judge reads its case without the label the case expects (the answer key); an example's must too, or
    the reading is of a judge shown the answer."""
    client = _FakeJudgeClient()
    cases = [{"question": "What is two plus two?", "label": "math"}]

    async def classify(case: Mapping[str, Any]) -> dict[str, str]:
        return {"label": "math"}

    example = Example("helpful", {"label": "math"}, 5, "a right label resolves it", case=cases[0])
    await Eval(cases, classify, ["label", _judge(client, examples=[example])], k=1).run(store=MEMORY, scope=SCOPE)

    run_users = [user for system, user in client.calls[:2]]
    (example_user,) = [user for system, user in client.calls[2:] if '"answer.helpful"' in system]
    assert all('"math"' not in user.split("# Output under review\n", 1)[0] for user in [*run_users, example_user])
    (ran,) = [user for system, user in client.calls[:2] if '"answer.helpful"' in system]
    assert example_user == ran, "the example asked exactly as the run's own answer was"


async def test_an_example_and_its_seeded_case_read_the_case_without_the_keys_a_run_withholds() -> None:
    """``id``, ``stratum`` and ``world`` name, sort and seed a case and never reach its judge: an example's case, and the
    authored case seeded from it, are read without them as a cell's is."""
    from threetears.evals.kernel import judge_case_of

    client = _FakeJudgeClient()
    host = callable_host([short])
    case = {"id": "arith", "stratum": "easy", **CASES[0]}
    example = Example("helpful", "Probably 4, but check.", 2, "a hedge does not resolve it", case=case)

    summary = await _summary(CASES, answer, [_judge(client, examples=[example])], host=host)

    (ran,) = [user for system, user in client.calls[:4] if '"answer.helpful"' in system and "two plus two" in user]
    (asked,) = [user for system, user in client.calls[4:]]
    assert asked == ran.removesuffix("4") + "Probably 4, but check.", "the example's case as the cell's, keys withheld"
    assert '"arith"' not in asked and '"easy"' not in asked
    (case_set,) = off_loop(host.storage.query_case_sets, SCOPE, name=f"{summary.template_id}-judge")
    (stored,) = off_loop(host.storage.load_test_cases_by_ids, case_set.test_case_ids, SCOPE)
    seeded = judge_case_of(stored)
    assert seeded is not None
    assert seeded.evidence.case_material == json.dumps(CASES[0], indent=2, sort_keys=True), "the case as a cell's"
    assert seeded.evidence.case_material in ran


# --- a grading that fails ---------------------------------------------------------------------------------------------


@pytest.mark.expect_contained(
    TypeError, why="a CompletionProtocolDefect (a TypeError) met grading the examples is contained into the reading"
)
async def test_a_grading_that_fails_is_stated_on_the_result_and_the_arms_spent_results_still_land() -> None:
    """A judge completion that breaks the protocol fails the grading after the arms have spent: the reading says it was
    not graded and why, and the eval files its campaign and returns its result."""

    @dataclass(frozen=True)
    class _TemperaturelessReply:
        """A completion that left ``temperature`` out of the CompletionResult protocol: the host's defect."""

        content: str
        input_tokens: int | None = 100
        output_tokens: int | None = 30
        cost_usd: float | None = JUDGE_COST_USD
        price_source: str | None = "scripted rate"
        model: str = JUDGE_MODEL
        served_model: str | None = None
        reasoning_tokens: int | None = None
        stop_reason: StopReason = "end_turn"

    # parity-with: threetears.evals.schema.CompletionClient
    class _FakeBreakingJudge(_FakeJudgeClient):
        """A judge whose replies on an example's answer break the protocol: the host's defect, met past the arms."""

        async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
            reply = await super().generate(system=system, user=user, response_format=response_format)
            if user.endswith("\nProbably 4, but I am not sure."):
                return _TemperaturelessReply(content=reply.content)
            return reply

    result = await Eval(CASES, answer, [_judge(_FakeBreakingJudge(), examples=EXAMPLES)], k=1).run(
        store=MEMORY, scope=SCOPE
    )

    read = result.judge_examples
    assert read is not None and read.stopped is not None
    assert read.stopped.startswith("the grading failed (CompletionProtocolDefect: ")
    assert [grade.verdicts for grade in read.grades] == [(), ()] and read.unread == read.grades
    # The call the defect surfaced on was made and ledgered, so the reading counts it though no outcome came back.
    assert len(off_loop(result.host.storage.query_out_of_run_spend, SCOPE, purpose="judge")) == read.spend.calls == 1
    assert read.lines()[0].startswith(
        "judge: 2 examples not graded, since the grading failed (CompletionProtocolDefect"
    )
    (block,) = [block for block in result.page.blocks if block.anchor == "judge-examples"]
    assert block.text.startswith("The judge's 2 examples were not graded, since the grading failed")
    assert result.summary.role_spend("judge").calls == 4 and result.campaign_id, "the arms' results and campaign land"
