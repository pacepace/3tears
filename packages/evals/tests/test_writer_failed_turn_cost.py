"""A writer turn the provider failed: priced at what the turns before it cost, sent again once, and no stop of the run.

The incident (2026-11-04, a writer campaign on Sonnet 5.5): one turn in each arm was answered ``anthropic
520: the response body is not JSON``. The writer kind booked the turn as spend nothing could price, so each result
was stored with ``cost_usd`` null, and the capped run stopped after one result ("1 result(s) spent what could not be
priced"), leaving the pairwise pass no common cases. A 5xx carries no completion and no usage, and is not billed: the
turn is an unanswered call, the result's cost is the sum of the turns priced before it, and the run goes on. The
client now sends such a turn again once (a 5xx, or a connection never made), and the reply names what each resend
followed, so a recording, and a replay of it, says the turn was sent twice.

Every request goes through ``httpx.MockTransport`` and every wait is a test's that waits for nothing; nothing here
reaches a live API. The writer is a placeholder model priced by a test pricer at fixed rates ($2 and $10 a million
input and output tokens, the incident model's), so each expected cost is computed here by hand and no shipped rate
table moves it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from packages.evals.tests.analyst_support import KAIRO, fixture_snapshots
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, ops_fixture
from packages.evals.tests.report_writer_support import FakeToolTurnClient, FakeTurnModel, call, reply
from threetears.evals.actions.report_writing import report_writer_kind
from threetears.evals.analysis.out_of_run_spend import OutOfRunSpendTotals
from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.kernel.host.kinds import freeze
from threetears.evals.kernel.host.measures import MeasureRegistry
from threetears.evals.ops import OpsHost
from threetears.evals.ops.report_writer.tally import ReportBudget
from threetears.evals.ops.report_writer.writer_contract import (
    ENDING_MEASURE,
    TURNS_MEASURE,
    UNPRICED_TURNS_MEASURE,
    WRITER_KIND,
    WRITER_KIND_CONTRACT,
    WRITER_KIND_MEASURES,
    EffortWord,
    WriterOverlays,
    writer_kind_contract,
)
from threetears.evals.ops.report_writer.writer_kind import WriterCase, writer_case_payload, writer_subject_components
from threetears.evals.providers import DeclinedRetry, Model, ProviderError, describe_failure
from threetears.evals.providers.failures import Provider
from threetears.evals.providers.pricing import CallPrice, HostRates, ModelRates
from threetears.evals.providers.wire import usage_counts
from threetears.evals.run.budget import BudgetStoppedError, EvalRunCostCap
from threetears.evals.run.runner import RunnerOptions, execute_run
from threetears.evals.schema.models import EvalResult, EvalRun, EvalTemplate, EvalTestCase
from threetears.evals.schema.out_of_run_spend import OutOfRunSpend
from threetears.evals.schema.subject import SubjectSnapshot
from threetears.evals.schema.turns import TurnMessage, TurnReply
from threetears.evals.testing import RecordedTurns

#: The writer: a placeholder id, which only the test pricer prices.
MODEL: Final = "writer-under-test"

#: The writer's rates, dollars a million tokens: input, output (Sonnet 5.5's published ones, the incident's model).
INPUT_RATE: Final = 2.0
OUTPUT_RATE: Final = 10.0

#: The rates the test pricer answers: input and output as above, no cache in any reply here.
RATES: Final = ModelRates(
    base_input=Decimal(2),
    cache_write_5m=Decimal("2.5"),
    cache_write_1h=Decimal(4),
    cache_read=Decimal("0.2"),
    output=Decimal(10),
)


# parity-with: threetears.evals.providers.Pricer
class FakeFixedPricer:
    """The host's port at fixed, guaranteed rates for any model; a reply that reported no usage is unpriced."""

    def rates_for(self, model: str, *, provider: Provider) -> HostRates | None:
        return HostRates(RATES, guaranteed=True)

    def price(
        self, model: str, *, provider: Provider, usage: Mapping[str, Any], served_provider: str | None
    ) -> CallPrice:
        counts = usage_counts(provider, usage)
        if counts.input_tokens is None or counts.output_tokens is None:
            return CallPrice(usd=None, source=None, unpriced_reason="the reply reported no usage")
        dollars, why = RATES.cost(input_tokens=counts.input_tokens, output_tokens=counts.output_tokens)
        assert dollars is not None, why
        return CallPrice(usd=float(dollars), source="test-rates")


#: What Anthropic's edge answers when the origin failed: a Cloudflare page, not JSON.
EDGE_520: Final = b"<html><head><title>520: Web server is returning an unknown error</title></head></html>"

#: The 520 as the client raises it, and as a resend names what it followed.
SAID_520: Final = "anthropic 520: the response body is not JSON"

#: The analyst reads the snapshot: a tool call, 1000 tokens in and 200 out.
READS: Final[dict[str, Any]] = {
    "model": f"{MODEL}-20261001",
    "content": [
        {"type": "text", "text": "Reading the overview first."},
        {"type": "tool_use", "id": "toolu_01A", "name": "evals_analyst", "input": {"action": "campaign_overview"}},
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 1000, "output_tokens": 200},
}

#: What the read cost, by hand: 1000 x $2/M + 200 x $10/M.
READS_USD: Final = (1000 * INPUT_RATE + 200 * OUTPUT_RATE) / 1e6  # 0.004

#: An answer with no tool call, 3000 tokens in and 500 out.
ANSWERS: Final[dict[str, Any]] = {
    "model": f"{MODEL}-20261001",
    "content": [{"type": "text", "text": "Here is what I found."}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 3000, "output_tokens": 500},
}

#: What the answer cost, by hand: 3000 x $2/M + 500 x $10/M.
ANSWERS_USD: Final = (3000 * INPUT_RATE + 500 * OUTPUT_RATE) / 1e6  # 0.011

#: An answer whose reply reported no usage: sent, maybe billed, and nothing the client can price.
UNREPORTED: Final[dict[str, Any]] = {key: value for key, value in ANSWERS.items() if key != "usage"}


Answer = tuple[int, bytes | Mapping[str, Any]] | Exception


class Messages:
    """Anthropic's Messages API: each request answered in turn with a status and a body, or a transport failure."""

    def __init__(self, *answers: Answer) -> None:
        self._answers = list(answers)
        self.sent = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.sent += 1
        if not self._answers:
            raise AssertionError(f"request {self.sent} came, and the script holds no answer for it")
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        status, body = answer
        if isinstance(body, bytes):
            return httpx.Response(status, content=body, headers={"content-type": "text/html"})
        return httpx.Response(status, json=body)


class Waits:
    """The retry's sleep, waiting for nothing and keeping each wait it was asked for."""

    def __init__(self) -> None:
        self.asked: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.asked.append(seconds)


def sonnet(messages: Messages, waits: Waits, **options: Any) -> Model:
    """The writer on Anthropic's route, priced by :class:`FakeFixedPricer`, sending through ``messages`` and waiting
    through ``waits``."""
    retry = DeclinedRetry(sleep=waits, **options.pop("retry", {}))
    http = httpx.AsyncClient(transport=httpx.MockTransport(messages), timeout=options.pop("timeout", 600.0))
    return Model(
        MODEL, provider="anthropic", api_key="test-key", http=http, retry=retry, pricer=FakeFixedPricer(), **options
    )


async def _turn(model: Any) -> TurnReply:
    reply_: TurnReply = await model.turn(
        system="s", messages=[TurnMessage("user", "Write the report.")], tools=(), max_tokens=1000
    )
    return reply_


# =============================================================================
# The client: a writer turn's 5xx, and a connection never made, sent again once
# =============================================================================


async def test_a_turn_answered_520_then_answered_is_returned_priced_naming_the_resend() -> None:
    messages, waits = Messages((520, EDGE_520), (200, ANSWERS)), Waits()

    answered = await _turn(sonnet(messages, waits))

    assert messages.sent == 2 and waits.asked == [1.0], "one resend, after the first backoff wait"
    assert answered.resent_after == (SAID_520,)
    assert answered.cost_usd == pytest.approx(ANSWERS_USD), "priced as the answer it is; the 520 billed nothing"
    assert TurnReply.from_json(answered.to_json()) == answered, "the resend survives the reply's JSON"


async def test_a_turn_failed_on_both_attempts_raises_the_last_520_as_the_candidates_failure() -> None:
    messages, waits = Messages((520, EDGE_520), (520, EDGE_520)), Waits()

    with pytest.raises(ProviderError) as raised:
        await _turn(sonnet(messages, waits))

    assert messages.sent == 2 and waits.asked == [1.0]
    assert (raised.value.status, raised.value.attempts, raised.value.owner) == (520, 2, "candidate")
    assert str(raised.value) == f"{SAID_520} (after 2 attempts)"


@pytest.mark.parametrize(
    "never", [httpx.ConnectError("connection refused"), httpx.ConnectTimeout("connect timed out")], ids=str
)
async def test_a_turn_whose_connection_was_never_made_is_sent_again(never: Exception) -> None:
    messages, waits = Messages(never, (200, ANSWERS)), Waits()

    answered = await _turn(sonnet(messages, waits))

    assert messages.sent == 2 and waits.asked == [1.0]
    assert answered.resent_after == (f"the connection was never made ({type(never).__name__})",)


@pytest.mark.parametrize("status", [504, 524], ids=["a 504", "Cloudflare's 524"])
async def test_a_turn_a_gateway_timed_out_is_never_sent_again_since_it_may_have_been_served(status: int) -> None:
    """The gateway gave up waiting; the provider may still have served and billed the turn, as a turn cut in flight."""
    messages, waits = Messages((status, EDGE_520), (200, ANSWERS)), Waits()

    with pytest.raises(ProviderError) as raised:
        await _turn(sonnet(messages, waits))

    assert messages.sent == 1 and waits.asked == []
    assert (raised.value.status, raised.value.attempts, raised.value.owner) == (status, 1, "candidate")


async def test_a_turn_cut_in_flight_is_never_sent_again_since_it_may_have_been_served() -> None:
    cut = httpx.ReadTimeout("read timed out")
    messages, waits = Messages(cut, (200, ANSWERS)), Waits()

    with pytest.raises(httpx.ReadTimeout):
        await _turn(sonnet(messages, waits))

    assert messages.sent == 1 and waits.asked == []


async def test_a_connection_never_made_twice_raises_the_first_from_the_second() -> None:
    first, second = httpx.ConnectError("connection refused"), httpx.ConnectError("connection refused again")
    messages, waits = Messages(first, second), Waits()

    with pytest.raises(httpx.ConnectError) as raised:
        await _turn(sonnet(messages, waits))

    assert messages.sent == 2 and waits.asked == [1.0]
    assert raised.value is first and raised.value.__cause__ is second


async def test_a_520_whose_resend_would_wait_past_the_requests_timeout_is_raised_unresent() -> None:
    """The first backoff wait (1 s) would end past a 0.5 s request timeout counted from the first attempt."""
    messages, waits = Messages((520, EDGE_520), (200, ANSWERS)), Waits()

    with pytest.raises(ProviderError) as raised:
        await _turn(sonnet(messages, waits, timeout=0.5))

    assert messages.sent == 1 and waits.asked == [] and raised.value.attempts == 1


async def test_one_turn_failure_attempt_raises_the_first_520() -> None:
    messages, waits = Messages((520, EDGE_520)), Waits()

    with pytest.raises(ProviderError) as raised:
        await _turn(sonnet(messages, waits, retry={"turn_failure_attempts": 1}))

    assert messages.sent == 1 and waits.asked == [] and raised.value.attempts == 1


async def test_a_completion_answered_520_is_never_sent_again() -> None:
    """The resend is the tool turn's: a completion's failure stays the call's outcome, as before."""
    messages, waits = Messages((520, EDGE_520)), Waits()

    with pytest.raises(ProviderError, match=f"^{SAID_520}$"):
        await sonnet(messages, waits).generate(system="", user="x")

    assert messages.sent == 1 and waits.asked == []


async def test_a_decline_then_a_520_then_an_answer_names_both_resends() -> None:
    overloaded = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    messages, waits = Messages((529, overloaded), (520, EDGE_520), (200, ANSWERS)), Waits()

    answered = await _turn(sonnet(messages, waits))

    assert messages.sent == 3 and waits.asked == [1.0, 1.0], "each kind's first backoff wait"
    assert answered.resent_after == ("anthropic 529: Overloaded", f"{SAID_520} (after 2 attempts)")


def test_a_policy_with_no_turn_failure_attempt_is_refused() -> None:
    with pytest.raises(ValueError, match="DeclinedRetry.turn_failure_attempts 0"):
        DeclinedRetry(turn_failure_attempts=0)


async def test_a_recorded_turn_keeps_its_resend_and_its_replay_says_so(tmp_path: Path) -> None:
    recording = tmp_path / "turns.jsonl"
    messages, waits = Messages((520, EDGE_520), (200, ANSWERS)), Waits()
    live = sonnet(messages, waits, turn_recording=recording).turn_session("analyst", snapshot="snap-1")
    await live.refresh_prices()
    live.turn_price_ceiling(system="s", messages=[TurnMessage("user", "Write the report.")], tools=(), max_tokens=1000)
    recorded = await _turn(live)

    (line,) = [json.loads(text) for text in recording.read_text().splitlines() if '"kind": "reply"' in text]
    assert line["reply"]["resent_after"] == [SAID_520]

    replay = RecordedTurns.replay(recording).turn_session("analyst", snapshot="snap-1")
    replay.turn_price_ceiling(system="s", messages=[], tools=(), max_tokens=1)
    replayed = await _turn(replay)
    assert replayed.resent_after == recorded.resent_after == (SAID_520,)


def test_a_reply_written_before_resends_were_kept_reads_as_answered_at_once() -> None:
    written = reply().to_json()
    assert "resent_after" not in written, "a reply answered at once is written as before"
    assert TurnReply.from_json(written).resent_after == ()
    with pytest.raises(ValueError, match="resent_after"):
        TurnReply.from_json({**written, "resent_after": []})


# =============================================================================
# The run: a failed turn's result priced at the turns before it, and the run goes on
# =============================================================================


def _host() -> OpsHost:
    """The toy host with the writer's contract and measures, describing a failure as the shipped client does."""
    profile = toyhost_profile()
    profile = replace(
        profile,
        kinds=(*profile.kinds, writer_kind_contract(host_seats=("reviewer_pool",))),
        measures=MeasureRegistry((*TOYHOST_MEASURES, *WRITER_KIND_MEASURES), families=profile.measures.families),
    )
    host = ops_fixture(generation=False, profile=profile).host
    eval_host = replace(host.eval_host, failure_describer=describe_failure)
    return replace(host, launch=replace(host.launch, eval_host=eval_host))


def _cases() -> list[EvalTestCase]:
    """Two cases, Kairo and r1, each frozen in another scope than the run's."""
    frozen = fixture_snapshots("frozen-elsewhere")
    return [
        EvalTestCase(
            id=f"writer-{name}",
            template_id="writer-template",
            scope_id=CALLER.scope_id,
            host_payload=writer_case_payload(WriterCase.of(frozen[name])),
        )
        for name in (KAIRO, "r1")
    ]


_TEMPLATE = EvalTemplate(
    id="writer-template",
    scope_id=CALLER.scope_id,
    name="Report writer",
    intent="Write the report.",
    candidate_kind=WRITER_KIND,
    created_at="2026-08-01T00:00:00+00:00",
    updated_at="2026-08-01T00:00:00+00:00",
)


@dataclass
class _Models:
    """The host's model factory: each cell's model built by ``build``, in the order the cells ask."""

    build: Any  # Any: a callable taking no argument and answering a TurnModel
    built: list[Any] = field(default_factory=list)

    def __call__(self, effort: EffortWord | None) -> Any:
        model = self.build()
        self.built.append(model)
        return model


async def _capped_run(models: _Models, *, cap_usd: float = 5.0) -> tuple[list[EvalResult], BudgetStoppedError | None]:
    """One arm over both cases, one cell at a time, under an enforcing cost cap: its stored results, by case, and the
    stop the cap raised, if it did."""
    host = off_loop(_host)
    skill = load_skill()
    overlays = WRITER_KIND_CONTRACT.validate_overlays({"skill": skill.digest, "revisions": 1})
    run: EvalRun = make_eval_run(
        scope_id=CALLER.scope_id,
        template_id=_TEMPLATE.id,
        candidate_kind=WRITER_KIND,
        candidate_model=MODEL,
        overlays=freeze(overlays),
        subject_snapshot=SubjectSnapshot(
            subject_id="report writer",
            subject_label="report writer",
            components=writer_subject_components(skill),
            state={},
        ),
        test_case_ids=[case.id for case in _cases()],
        max_cost_usd=cap_usd,
    )
    kind = report_writer_kind(
        host,
        CALLER,
        overlays=WriterOverlays.model_validate(run.overlays),
        models=models,
        budget=ReportBudget(usd=1.0),
        skills={},
    )
    cap = EvalRunCostCap(run.id, cap_usd, enabled=True)
    stopped: BudgetStoppedError | None = None
    try:
        await execute_run(
            host.eval_host,
            run=run,
            template=_TEMPLATE,
            test_cases=_cases(),
            judge_service=None,
            options=RunnerOptions(candidate_kinds={WRITER_KIND: lambda _cell: kind}, max_concurrent_cells=1),
            budget_gate=cap.check,
            on_cost=cap.record,
        )
    except BudgetStoppedError as raised:
        stopped = raised
    results = off_loop(host.eval_host.storage.query_eval_results_by_run, run.id, run.scope_id)
    return sorted(results, key=lambda result: result.test_case_id), stopped


async def test_a_turn_failed_by_a_520_prices_its_result_at_the_turns_before_it_and_the_run_goes_on() -> None:
    """Each cell: the analyst reads once (priced), then its next turn is answered 520 twice and the writing fails."""
    waits = Waits()
    models = _Models(lambda: sonnet(Messages((200, READS), (520, EDGE_520), (520, EDGE_520)), waits))

    results, stopped = await _capped_run(models)

    assert stopped is None, "a failed turn is no unpriced spend, so the cap never stops the run"
    assert len(results) == 2, "the second cell ran after the first one's failure"
    for result in results:
        assert result.cost_usd == pytest.approx(READS_USD), "the read's price; the two 520s billed nothing"
        assert result.candidate_error == f"report_writer: a turn failed: {SAID_520} (after 2 attempts)"
        assert result.turns_delivered == 1
        (row,) = result.usage
        assert (row.call_count, row.cost_usd, row.unanswered_lost, row.unpriced_reasons) == (
            1,
            pytest.approx(READS_USD),
            1,
            None,
        ), "the failed turn sits beside the priced read as one unanswered call"
        assert result.host_measures[ENDING_MEASURE] == "failed"
        assert (result.host_measures[TURNS_MEASURE], result.host_measures[UNPRICED_TURNS_MEASURE]) == (2.0, 0.0)
    assert waits.asked == [1.0, 1.0], "one resend a cell"


async def test_a_turn_failed_first_prices_its_result_at_zero_and_the_run_goes_on() -> None:
    """Nothing was delivered before the failure: the priced sum is of no turns, $0, never unknown."""
    models = _Models(lambda: sonnet(Messages((520, EDGE_520), (520, EDGE_520)), Waits()))

    results, stopped = await _capped_run(models)

    assert stopped is None and len(results) == 2
    assert [result.cost_usd for result in results] == [0.0, 0.0]
    assert all(result.candidate_error is not None and result.turns_delivered == 0 for result in results)


async def test_a_520_answered_on_its_resend_lets_the_writing_go_on_and_is_priced_as_the_turns_answered() -> None:
    """The analyst reads, its next turn is answered 520 then answered, and the read after it reaches the script's end:
    two priced turns before the failure that ends the writing, and the resend the turn between them made."""
    waits = Waits()
    models = _Models(
        lambda: sonnet(Messages((200, READS), (520, EDGE_520), (200, READS), (520, EDGE_520), (520, EDGE_520)), waits)
    )

    results, stopped = await _capped_run(models)

    assert stopped is None and len(results) == 2
    for result in results:
        assert result.cost_usd == pytest.approx(2 * READS_USD)
        assert result.turns_delivered == 2
    assert waits.asked == [1.0, 1.0, 1.0, 1.0], "the resent turn's wait, then the failed turn's, in each cell"


async def test_a_reply_nothing_could_price_still_stops_the_capped_run_after_its_cell() -> None:
    """A reply that reported no usage was answered, so may have been billed: unpriced spend, and the cap stops."""
    models = _Models(lambda: sonnet(Messages((200, READS), (200, UNREPORTED)), Waits()))

    results, stopped = await _capped_run(models)

    assert stopped is not None, "unpriced spend stops an enforcing run"
    assert "1 result(s) spent what could not be priced" in str(stopped)
    (result,) = results
    assert result.cost_usd is None
    assert result.host_measures[UNPRICED_TURNS_MEASURE] == 1.0


def _scripted(*replies: TurnReply | Exception) -> FakeTurnModel:
    """A scripted writer whose analyst answers ``replies`` in turn."""
    model = FakeTurnModel(analyst=FakeToolTurnClient(*replies))
    model.model_name = MODEL
    return model


async def test_a_scripted_unpriced_reply_is_unpriced_spend_and_a_failed_turn_is_not() -> None:
    """The same rule off the shipped client: a reply with no cost unprices the result; a turn cut in flight, whose
    reply never came, does not."""
    reads = reply(call(1, "campaign_overview", "s"), cost=0.25, model=MODEL)
    cut = httpx.ReadTimeout("read timed out")
    unpriced = _Models(lambda: _scripted(reads, reply(call(2, "campaign_overview", "s"), cost=None, model=MODEL), cut))
    failed = _Models(lambda: _scripted(reads, cut))

    (priced_none,), stopped = await _capped_run(unpriced)
    assert priced_none.cost_usd is None and stopped is not None

    results, stopped = await _capped_run(failed)
    assert stopped is None and [result.cost_usd for result in results] == [0.25, 0.25]


# =============================================================================
# The out-of-run ledger agrees with the result: a raised turn is no unpriced spend there either
# =============================================================================


def test_a_ledger_row_counts_a_raised_turn_apart_from_its_unpriced_replies_and_prices_the_rest() -> None:
    """A report of three turns, two priced at $0.004 and one whose reply never came: the row's total is $0.008."""
    fields: dict[str, Any] = {
        "scope_id": "s",
        "purpose": "analysis",
        "model": MODEL,
        "outcome": "completed",
        "calls": 3,
    }

    row = OutOfRunSpend(**fields, unpriced_calls=0, raised_calls=1, priced_usd=2 * READS_USD, cost_usd=2 * READS_USD)
    assert OutOfRunSpendTotals.of([row]).model_dump(include={"n_calls", "n_raised", "n_unpriced", "priced_usd"}) == {
        "n_calls": 3,
        "n_raised": 1,
        "n_unpriced": 0,
        "priced_usd": pytest.approx(2 * READS_USD),
    }
    with pytest.raises(ValueError, match="unpriced replies and raised turns are calls apart"):
        OutOfRunSpend(**fields, unpriced_calls=2, raised_calls=2, priced_usd=0.0, cost_usd=None)
