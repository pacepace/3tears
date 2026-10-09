"""``run_eval`` with a judge: a rubric scored by the engine's judge, its evidence and spend recorded as any judged run's.

The judge here is scripted: it reads the dimension it was asked for off the system prompt and the
answer off the user prompt, and scores the answer by what it says, so every score below is a function
of the evidence the callable kind rendered and the engine's judge service forwarded. The tests that
count calls pass ``k=1``, so each count is one per case per dimension.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

import pytest

from threetears.evals.contracts import JudgedArtifact, RubricDim, StopReason
from threetears.evals.contracts.host import EvalHost, KindContract
from threetears.evals.quick import (
    CALLABLE_KIND_CONTRACT,
    JUDGED_CALLABLE_KIND,
    JUDGED_CALLABLE_KIND_CONTRACT,
    Judge,
    callable_host,
    run_eval,
    summarize_run,
)
from threetears.evals.run import (
    get_result_trace,
    get_template,
    list_results,
    list_runs,
    list_templates,
    update_template,
)

SCOPE = "run-eval-judged-tests"
CASES = [{"question": "What is two plus two?"}, {"question": "What is the capital of France?"}]
ANSWERS = {"What is two plus two?": "4", "What is the capital of France?": "I am not sure."}
RUBRIC = {"helpful": "The answer resolves the question.", "honest": "The answer claims nothing it cannot support."}
JUDGE_MODEL = "scripted/judge"
JUDGE_COST_USD = 0.0003


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


# parity-with: threetears.evals.contracts.CompletionClient
class _FakeJudgeClient:
    """A judge that gives a committed answer 5 and a hedge 2, or a reply set up per test."""

    def __init__(self, *, priced: bool = True, reply: str | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.aclose_calls = 0
        self._priced = priced
        self._reply = reply

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
        )

    async def aclose(self) -> None:
        self.aclose_calls += 1

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


def _judge(client: _FakeJudgeClient, **overrides: Any) -> Judge:
    return Judge(client=client, model=JUDGE_MODEL, rubric=overrides.pop("rubric", RUBRIC), **overrides)


async def test_each_dimension_is_scored_by_the_engines_judge_and_summarised_beside_the_scorers() -> None:
    client = _FakeJudgeClient()
    host = callable_host([short])
    summary = await run_eval(CASES, answer, [short], judge=_judge(client), scope_id=SCOPE, host=host, k=1)

    assert (summary.status, summary.n_scored, summary.n_excluded) == ("completed", 2, 0)
    assert [measure.name for measure in summary.measures] == ["short"]
    by_name = {dimension.name: dimension for dimension in summary.judged}
    assert list(by_name) == ["answer.helpful", "answer.honest"]
    for dimension in by_name.values():
        assert (dimension.scale, dimension.n, dimension.minimum, dimension.maximum) == ("ordinal", 2, 2.0, 5.0)
        assert dimension.mean == pytest.approx(3.5) and dimension.cannot_tell == 0
    assert len(client.calls) == 4, "one call per case per dimension, and no conversation axis"
    assert summary.judge_calls == 4
    assert summary.judge_cost_usd == pytest.approx(4 * JUDGE_COST_USD)

    rendered = summary.render()
    assert "  answer.helpful (judged 1-5): mean 3.5 (n=2, min 2, max 5)" in rendered
    assert "  answer.honest (judged 1-5): mean 3.5 (n=2, min 2, max 5)" in rendered
    assert "  judge spend: $0.00120 over 4 call(s)" in rendered  # 4 x $0.0003

    (run,) = list_runs(host, SCOPE)
    assert run.judge_model == JUDGE_MODEL
    template = get_template(host, summary.template_id or "", SCOPE)
    assert template.candidate_kind == JUDGED_CALLABLE_KIND
    assert [dim.name for dim in template.rubric] == ["answer.helpful", "answer.honest"]
    for result in list_results(host.storage, summary.run_id, SCOPE):
        assert result.judge_model == JUDGE_MODEL and result.judge_error is None
        assert {score.served_model for score in result.rubric_scores} == {"scripted/judge-2026"}
        (judge_row,) = [row for row in result.usage if row.role == "judge"]
        assert judge_row.cost_usd == pytest.approx(2 * JUDGE_COST_USD)
        assert judge_row.price_source == "scripted rate"


async def test_the_judge_reads_the_answer_against_the_case_material_and_the_trace_keeps_it() -> None:
    client = _FakeJudgeClient()
    host = callable_host()
    judge = _judge(client, case_material=lambda case: f"Reference: arithmetic.\nQuestion: {case['question']}")
    summary = await run_eval(CASES[:1], answer, judge=judge, scope_id=SCOPE, host=host, k=1)

    _, user = client.calls[0]
    assert "Reference: arithmetic.\nQuestion: What is two plus two?" in user
    assert user.endswith("# Output under review\n4")
    (result,) = list_results(host.storage, summary.run_id, SCOPE)
    trace = get_result_trace(host.storage, result)
    assert trace is not None and trace.judged_artifact is JudgedArtifact.DOCUMENT
    assert trace.judge_evidence is not None
    assert trace.judge_evidence.case_material == "Reference: arithmetic.\nQuestion: What is two plus two?"
    assert trace.judge_evidence.artifact == "4"


async def test_without_case_material_the_judge_reads_the_case_as_json_and_a_non_string_answer_as_json() -> None:
    client = _FakeJudgeClient()

    async def structured(case: Mapping[str, Any]) -> dict[str, Any]:
        return {"answer": 4}

    await run_eval(CASES[:1], structured, judge=_judge(client), scope_id=SCOPE, k=1)
    _, user = client.calls[0]
    assert '{\n  "question": "What is two plus two?"\n}' in user
    assert user.endswith('# Output under review\n{\n  "answer": 4\n}')


async def test_the_judges_client_is_lent_never_closed_so_one_client_serves_two_runs() -> None:
    client = _FakeJudgeClient()
    host = callable_host()
    judge = _judge(client)
    first = await run_eval(CASES, answer, judge=judge, scope_id=SCOPE, host=host, k=1)
    second = await run_eval(CASES, answer, judge=judge, scope_id=SCOPE, host=host, k=1, model="again")
    assert client.aclose_calls == 0
    assert first.template_id == second.template_id
    assert first.n_scored == second.n_scored == 2


async def test_a_judge_that_prices_nothing_leaves_its_spend_unknown_never_zero() -> None:
    summary = await run_eval(CASES, answer, judge=_judge(_FakeJudgeClient(priced=False)), scope_id=SCOPE, k=1)
    assert summary.judge_calls == 4 and summary.judge_cost_usd is None
    assert "  judge spend: unknown: a judge call went unpriced over 4 call(s)" in summary.render()


async def test_a_judge_reply_that_will_not_parse_excludes_the_cell_and_says_so() -> None:
    summary = await run_eval(
        CASES[:1], answer, [short], judge=_judge(_FakeJudgeClient(reply="no json here")), scope_id=SCOPE, k=1
    )
    assert (summary.n_scored, summary.n_candidate_failed, summary.n_excluded) == (0, 0, 1)
    assert summary.errors and all(": judge: " in error for error in summary.errors)
    # Every attempt the judge made was paid for, parsed or not, so its spend is still counted.
    assert summary.judge_calls >= 2 and summary.judge_cost_usd is not None and summary.judge_cost_usd > 0


async def test_a_judge_that_cannot_tell_is_counted_apart_and_excludes_nothing() -> None:
    reply = json.dumps({"reasoning": "the answer is not shown", "criteria_scores": {"answer.helpful": "cannot_tell"}})
    summary = await run_eval(
        CASES[:1],
        answer,
        judge=_judge(_FakeJudgeClient(reply=reply), rubric={"helpful": "Helps."}),
        scope_id=SCOPE,
        k=1,
    )
    (dimension,) = summary.judged
    assert (dimension.name, dimension.n, dimension.mean, dimension.cannot_tell) == ("answer.helpful", 0, None, 1)
    assert summary.n_excluded == 0
    assert "answer.helpful (judged): no result carries a score, the judge could not tell on 1" in summary.render()


async def test_a_rubric_of_engine_dimensions_keeps_their_names_and_scales() -> None:
    client = _FakeJudgeClient(
        reply=json.dumps({"reasoning": "fine", "criteria_scores": {"faq.correct": "pass"}}),
    )
    dims = [RubricDim(name="faq.correct", description="The answer is correct.", scale="pass_fail")]
    summary = await run_eval(CASES[:1], answer, judge=_judge(client, rubric=dims), scope_id=SCOPE, k=1)
    (dimension,) = summary.judged
    assert (dimension.name, dimension.scale, dimension.mean) == ("faq.correct", "pass_fail", 1.0)
    assert "faq.correct (judged pass/fail): mean 1" in summary.render()


async def test_a_judged_call_and_an_unjudged_one_over_the_same_cases_are_two_templates() -> None:
    host = callable_host([short])
    unjudged = await run_eval(CASES, answer, [short], scope_id=SCOPE, host=host, k=1)
    judged = await run_eval(CASES, answer, [short], judge=_judge(_FakeJudgeClient()), scope_id=SCOPE, host=host, k=1)
    other_rubric = _judge(_FakeJudgeClient(), rubric={"helpful": "Helps."})
    reworded = await run_eval(CASES, answer, [short], judge=other_rubric, scope_id=SCOPE, host=host, k=1)
    assert len({unjudged.template_id, judged.template_id, reworded.template_id}) == 3
    assert unjudged.judged == [] and "judge spend" not in unjudged.render()


# --- the intent the judge reads -------------------------------------------------------------------


def _intents_read(client: _FakeJudgeClient) -> set[str]:
    """The intent line of every prompt the judge was sent."""
    return {line for _, user in client.calls for line in user.splitlines() if line.startswith("**Intent:**")}


async def test_an_explicit_intent_is_what_the_judge_reads_and_the_summary_says_so() -> None:
    client = _FakeJudgeClient()
    host = callable_host()
    stated = "Answer a quiz question correctly, or say you do not know."
    summary = await run_eval(CASES, answer, judge=_judge(client), intent=stated, scope_id=SCOPE, host=host, k=1)

    assert _intents_read(client) == {f"**Intent:** {stated}"}
    assert get_template(host, summary.template_id or "", SCOPE).intent == stated
    assert (summary.intent, summary.intent_source) == (stated, None)
    assert f"\n  intent: {stated}\n" in summary.render()


async def test_with_no_intent_the_judge_reads_the_candidates_docstring_and_the_summary_names_it() -> None:
    client = _FakeJudgeClient()
    summary = await run_eval(CASES, answer, judge=_judge(client), scope_id=SCOPE, k=1)

    assert _intents_read(client) == {"**Intent:** Answer a question in a few words."}
    assert "  intent (from answer's docstring): Answer a question in a few words." in summary.render()


async def test_with_no_intent_and_no_docstring_the_summary_says_the_intent_is_a_generic_default() -> None:
    client = _FakeJudgeClient()

    async def undocumented(case: Mapping[str, Any]) -> str:
        return ANSWERS[case["question"]]

    summary = await run_eval(CASES, undocumented, judge=_judge(client), scope_id=SCOPE, k=1)
    default = "Answer each case so that the judge and every scorer grade the answer well."
    assert _intents_read(client) == {f"**Intent:** {default}"}
    assert f"  intent (a generic default: no intent=, and undocumented has no docstring): {default}" in summary.render()


async def test_a_summary_read_back_keeps_the_intent_the_judge_read_but_not_where_it_came_from() -> None:
    host = callable_host()
    run = await run_eval(CASES, answer, judge=_judge(_FakeJudgeClient()), scope_id=SCOPE, host=host, k=1)
    read_back = summarize_run(host, run.run_id, SCOPE)
    assert (read_back.intent, read_back.intent_source) == ("Answer a question in a few words.", None)
    assert "  intent: Answer a question in a few words." in read_back.render()


async def test_two_judged_runs_stating_different_intents_each_keep_the_template_their_judge_read() -> None:
    host = callable_host()
    first = await run_eval(CASES, answer, judge=_judge(_FakeJudgeClient()), scope_id=SCOPE, host=host, k=1)
    second = await run_eval(
        CASES, answer, judge=_judge(_FakeJudgeClient()), intent="Reworded.", scope_id=SCOPE, host=host, k=1
    )
    assert first.template_id != second.template_id
    assert summarize_run(host, first.run_id, SCOPE).intent == "Answer a question in a few words."
    assert summarize_run(host, second.run_id, SCOPE).intent == "Reworded."


async def test_a_template_edited_since_the_run_no_longer_says_what_its_judge_read_so_no_intent_is_shown() -> None:
    host = callable_host()
    run = await run_eval(CASES, answer, judge=_judge(_FakeJudgeClient()), scope_id=SCOPE, host=host, k=1)
    update_template(
        host,
        run.template_id or "",
        SCOPE,
        {"intent": "Edited after the run."},
        require_known_tools_allowed=lambda _tools: None,
        refuse_undeclared_world_seed=lambda _template: None,
        refuse_undeliverable_template=lambda _template: None,
    )
    assert summarize_run(host, run.run_id, SCOPE).intent is None


async def test_an_unjudged_run_shows_no_intent_since_nothing_that_grades_it_reads_one() -> None:
    summary = await run_eval(CASES, answer, [short], intent="Answer briefly.", scope_id=SCOPE, k=1)
    assert summary.intent is None and "intent" not in summary.render()


@pytest.mark.parametrize("intent", ["", "   ", 3])
async def test_an_intent_that_is_not_a_non_blank_string_is_refused_before_anything_runs(intent: Any) -> None:
    client = _FakeJudgeClient()
    with pytest.raises(ValueError, match="intent= is the sentence the judge reads"):
        await run_eval(CASES, answer, judge=_judge(client), intent=intent, scope_id=SCOPE, k=1)
    assert client.calls == []


# --- refusals ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "said"),
    [
        ({"model": " "}, "names the model its client calls"),
        ({"rubric": {}}, "at least one rubric dimension"),
        ({"rubric": {"a.b.c": "Too many contexts."}}, "a.b.c"),
        ({"rubric": {"helpful": ""}}, "description"),
        ({"rubric": {"helpful": "Helps.", "answer.helpful": "Helps again."}}, "answer.helpful more than once"),
    ],
    ids=["a blank model", "no dimension", "a doubly namespaced name", "a blank description", "one name twice"],
)
def test_a_judge_no_run_could_record_is_refused_where_it_is_built(overrides: dict[str, Any], said: str) -> None:
    fields: dict[str, Any] = {"client": _FakeJudgeClient(), "model": JUDGE_MODEL, "rubric": RUBRIC, **overrides}
    with pytest.raises(ValueError, match=said):
        Judge(**fields)


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
        await run_eval(CASES, answer, judge=_judge(_FakeJudgeClient()), scope_id=SCOPE, host=host)
    assert list_templates(host.storage, SCOPE) == []


async def test_a_host_seating_the_judge_by_its_pins_is_accepted() -> None:
    pins = frozenset({"judge_model", "judge_request_settings", "judge_dim_divergence", "judge_config_ids"})
    host = _host_whose_judged_contract_is(KindContract(JUDGED_CALLABLE_KIND, seats=pins))
    summary = await run_eval(CASES, answer, judge=_judge(_FakeJudgeClient()), scope_id=SCOPE, host=host, k=1)
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
