"""The shipped client's price ceiling, the catalogue it reads, and the client a model opens for itself.

A price ceiling is what an enforced cost cap holds a call against before it is made, so it is a true upper bound or
``None``: every price a catalogue row lists is bounded (a reasoning rate, a per-request fee, a long-context tier, a
cache write, an output image or audio), the prompt is counted after NFKC normalisation, and rows older than
``Catalogue.ceiling_stale_after`` bound nothing. The rows below are OpenRouter's, as ``GET /api/v1/models`` listed them
on 2026-10-15 (trimmed to the fields the client reads), except where a test says it writes one.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from datetime import timedelta
from decimal import Decimal
from typing import Any

import httpx
import pytest

from threetears.evals.kernel import AdmittedCall, OutOfRunBudget, PlannedCall, ValidationFailedError
from threetears.evals.ops import await_job, start_paid_call
from threetears.evals.providers import CATALOGUE_FETCH_TIMEOUT_S, Catalogue, Clients, Model, PriceRefresh
from threetears.evals.schema import BoundCompletionClient
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.testing import CompletionGuard
from packages.evals.tests.jobs_conformance_support import CONFORMANCE_SCOPE, conformance_host
from packages.evals.tests.off_loop import off_loop

#: Anthropic's Haiku: a long-context tier from 100,000 prompt tokens, cache reads and writes (five-minute and
#: one-hour), and a web-search price for searches the client never asks for.
HAIKU: dict[str, Any] = {
    "id": "anthropic/claude-haiku-5.5",
    "pricing": {
        "prompt": "0.0000001",
        "completion": "0.0000005",
        "web_search": "0.01",
        "input_cache_read": "0.00000001",
        "input_cache_write": "0.000000125",
        "input_cache_write_1h": "0.0000002",
        "overrides": [
            {
                "min_prompt_tokens": 100000,
                "prompt": "0.0000005",
                "completion": "0.0000025",
                "input_cache_read": "0.00000005",
                "input_cache_write": "0.000000625",
                "input_cache_write_1h": "0.000001",
            }
        ],
    },
    "supported_parameters": ["max_tokens", "reasoning", "response_format", "structured_outputs"],
}

#: A model priced by the time of day: the dearer tier is the bound, since a call's billing time is not known.
TIME_OF_DAY: dict[str, Any] = {
    "id": "tencent/hy4-preview",
    "pricing": {
        "prompt": "0.0000007506",
        "completion": "0.0000022509",
        "input_cache_read": "0.0000000378",
        "overrides": [
            {"utc_start": 0, "utc_end": 1600, "prompt": "0.000000834", "completion": "0.000002501"},
            {"utc_start": 1600, "utc_end": 0, "prompt": "0.0000007506", "completion": "0.0000022509"},
        ],
    },
    "supported_parameters": ["max_tokens", "reasoning", "response_format"],
}

#: A model that searches the web by itself (it takes ``web_search_options``), each search priced.
SONAR: dict[str, Any] = {
    "id": "perplexity/sonar",
    "pricing": {"prompt": "0.000001", "completion": "0.000001", "web_search": "0.005"},
    "supported_parameters": ["max_tokens", "temperature", "web_search_options"],
}

#: An audio model: audio input the client never sends, and audio output above the completion rate.
AUDIO: dict[str, Any] = {
    "id": "openai/gpt-audio",
    "pricing": {"prompt": "0.0000025", "completion": "0.00001", "audio": "0.000032", "audio_output": "0.000064"},
    "supported_parameters": ["max_tokens", "response_format", "structured_outputs"],
}

#: Written for these tests: a reasoning rate above the completion rate, and a per-request fee (OpenRouter lists both
#: as `internal_reasoning` and `request`; no row of the recorded catalogue carries either above the base).
REASONER: dict[str, Any] = {
    "id": "vendor/reasoner",
    "pricing": {"prompt": "0.000001", "completion": "0.000002", "internal_reasoning": "0.000008", "request": "0.005"},
    "supported_parameters": ["max_tokens", "reasoning"],
}

ROWS: list[dict[str, Any]] = [HAIKU, TIME_OF_DAY, SONAR, AUDIO, REASONER]


def catalogue(*rows: Mapping[str, Any], **options: Any) -> Catalogue:
    """A catalogue of ``rows`` (:data:`ROWS` by default) over a scripted fetch."""
    payload = {"data": list(rows or ROWS)}
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    return Catalogue(http=httpx.AsyncClient(transport=transport), **options)


def model(model_id: str, shared: Catalogue, *, max_tokens: int = 1000, **options: Any) -> Model:
    """A model on ``shared`` that never sends (the ceiling needs no call)."""
    never = httpx.MockTransport(lambda _: httpx.Response(500))
    return Model(
        model_id,
        api_key="test-key",
        max_tokens=max_tokens,
        catalogue=shared,
        http=httpx.AsyncClient(transport=never),
        **options,
    )


def prompt_bound(*texts: str) -> int:
    """The prompt tokens the ceiling counts for ``texts``: one per byte, and 2048 for framing."""
    return sum(len(text.encode()) for text in texts) + 2048


# --- every price dimension ---------------------------------------------------------------------------------------


async def test_a_long_prompt_is_priced_at_the_tier_it_can_reach_where_the_old_ceiling_was_below_the_bill() -> None:
    shared = catalogue()
    await shared.load()
    haiku = model("anthropic/claude-haiku-5.5", shared, max_tokens=8192)
    user = "7" * 150_000  # digits: as many as one token a byte, so 150,000 prompt tokens is a bill it can run up
    ceiling = haiku.price_ceiling(system="", user=user)

    tokens = prompt_bound(user)
    old_ceiling = tokens * 0.0000001 + 8192 * 0.0000005  # .29: the base prompt and completion rates alone
    bill = 150_000 * 0.0000005 + 8192 * 0.0000025  # what the provider bills it: the 100,000-token tier
    assert old_ceiling < bill, "the case the old ceiling got wrong"
    assert ceiling is not None and ceiling >= bill
    # The tier's dearest prompt-token rate (a one-hour cache write) and its completion rate.
    assert ceiling == pytest.approx(tokens * 0.000001 + 8192 * 0.0000025)


async def test_a_short_prompt_is_priced_below_the_tier_and_at_the_dearest_cache_rate() -> None:
    shared = catalogue()
    await shared.load()
    ceiling = model("anthropic/claude-haiku-5.5", shared).price_ceiling(system="s", user="u")
    assert ceiling == pytest.approx(prompt_bound("s", "u") * 0.0000002 + 1000 * 0.0000005)


async def test_reasoning_tokens_are_priced_at_the_reasoning_rate_and_the_request_fee_is_added() -> None:
    shared = catalogue()
    await shared.load()
    ceiling = model("vendor/reasoner", shared, max_tokens=4000).price_ceiling(system="", user="hi")
    bill = 2 * 0.000001 + 4000 * 0.000008 + 0.005  # every output token a reasoning one, within max_tokens
    assert ceiling is not None and ceiling >= bill
    assert ceiling == pytest.approx(prompt_bound("hi") * 0.000001 + 4000 * 0.000008 + 0.005)


async def test_a_time_of_day_tier_is_taken_to_apply() -> None:
    shared = catalogue()
    await shared.load()
    ceiling = model("tencent/hy4-preview", shared).price_ceiling(system="", user="hi")
    assert ceiling == pytest.approx(prompt_bound("hi") * 0.000000834 + 1000 * 0.000002501)


async def test_output_audio_is_priced_and_input_audio_the_client_never_sends_is_not() -> None:
    shared = catalogue()
    await shared.load()
    ceiling = model("openai/gpt-audio", shared).price_ceiling(system="", user="hi")
    assert ceiling == pytest.approx(prompt_bound("hi") * 0.0000025 + 1000 * 0.000064)


async def test_the_prompt_is_counted_after_nfkc_where_that_is_longer() -> None:
    shared = catalogue()
    await shared.load()
    reasoner = model("vendor/reasoner", shared, max_tokens=1)
    # U+FDFA is three bytes as written and 33 after NFKC: a normalising tokenizer reads the 33.
    ligatures = reasoner.price_ceiling(system="", user="ﷺ" * 100)
    assert ligatures == pytest.approx((3300 + 2048) * 0.000001 + 0.000008 + 0.005)


async def test_the_directive_is_counted_too() -> None:
    shared = catalogue()
    await shared.load()
    reasoner = model("vendor/reasoner", shared, max_tokens=1)
    directive = {"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": {"type": "object"}}}
    bare = reasoner.price_ceiling(system="", user="hi")
    framed = reasoner.price_ceiling(system="", user="hi", response_format=directive)
    assert bare is not None and framed is not None and framed > bare


@pytest.mark.parametrize(
    ("row", "why"),
    [
        (SONAR, "can search the web unasked (it takes web_search_options) at 0.005 a search"),
        (
            {"id": "vendor/x", "pricing": {"prompt": "0.000001", "completion": "0.000001", "video_output": "0.1"}},
            "lists a `video_output` price (0.1) the client does not know the unit of",
        ),
        (
            {"id": "vendor/x", "pricing": {"prompt": "0.000001", "completion": "0.000001", "request": "a dime"}},
            "its `request` price ('a dime') is not a price",
        ),
        (
            {"id": "vendor/x", "pricing": {"prompt": "0.000001", "completion": "0.000001", "overrides": {"a": 1}}},
            "its price tiers (`overrides`) are not a list of objects",
        ),
        (
            {
                "id": "vendor/x",
                "pricing": {
                    "prompt": "0.000001",
                    "completion": "0.000001",
                    "overrides": [{"min_prompt_tokens": 10, "completion": "-1"}],
                },
            },
            "its `completion` price ('-1') is not a price",
        ),
    ],
)
async def test_a_price_the_client_cannot_bound_answers_no_ceiling_saying_why(
    row: Mapping[str, Any], why: str, caplog: pytest.LogCaptureFixture
) -> None:
    shared = catalogue(row)
    await shared.load()
    unbounded = model(str(row["id"]), shared)
    with caplog.at_level(logging.WARNING):
        assert unbounded.price_ceiling(system="", user="hi") is None
        assert unbounded.price_ceiling(system="", user="hi again") is None
    warned = [record.getMessage() for record in caplog.records if "price_ceiling cannot bound" in record.getMessage()]
    assert len(warned) == 1, "each reason is logged once"
    assert why in warned[0]
    assert shared.ceiling(str(row["id"]), prompt_tokens=10, output_tokens=10)[1] is not None


async def test_a_zero_price_of_any_name_bounds_nothing_away() -> None:
    row = {"id": "vendor/x", "pricing": {"prompt": "0.000001", "completion": "0.000001", "video_output": "0"}}
    shared = catalogue(row)
    await shared.load()
    assert shared.ceiling("vendor/x", prompt_tokens=10, output_tokens=10) == (Decimal("0.000020"), None)


async def test_a_tier_whose_threshold_cannot_be_read_is_taken_to_apply() -> None:
    row = {
        "id": "vendor/x",
        "pricing": {
            "prompt": "0.000001",
            "completion": "0.000001",
            "overrides": [{"min_prompt_tokens": "lots", "completion": "0.00001"}],
        },
    }
    shared = catalogue(row)
    await shared.load()
    assert shared.ceiling("vendor/x", prompt_tokens=10, output_tokens=10) == (Decimal("0.00011"), None)


async def test_no_ceiling_before_the_catalogue_is_read_and_for_an_unlisted_model() -> None:
    shared = catalogue()
    assert model("anthropic/claude-haiku-5.5", shared).price_ceiling(system="", user="hi") is None
    assert "has not been read" in (shared.ceiling(HAIKU["id"], prompt_tokens=1, output_tokens=1)[1] or "")
    await shared.load()
    assert model("vendor/unlisted", shared).price_ceiling(system="", user="hi") is None
    assert "no priced row" in (shared.ceiling("vendor/unlisted", prompt_tokens=1, output_tokens=1)[1] or "")


# --- staleness ---------------------------------------------------------------------------------------------------


def test_the_staleness_bound_is_refresh_after_unless_given_and_never_below_it() -> None:
    assert Catalogue().ceiling_stale_after == timedelta(hours=6)
    assert Catalogue(refresh_after=timedelta(hours=1)).ceiling_stale_after == timedelta(hours=1)
    later = Catalogue(refresh_after=timedelta(hours=1), ceiling_stale_after=timedelta(hours=6))
    assert later.ceiling_stale_after == timedelta(hours=6)
    with pytest.raises(ValueError, match="ceiling_stale_after .* is below refresh_after"):
        Catalogue(ceiling_stale_after=timedelta(hours=2))  # rows past two hours would not be due a refresh until six


class CatalogueServer:
    """A catalogue endpoint answering each fetch in turn (the last answer repeating), counting the fetches; each
    answer waits a turn of the loop first, so loads made together overlap."""

    def __init__(self, *answers: tuple[int, Mapping[str, Any]]) -> None:
        self._answers = list(answers)
        self.fetches = 0

    async def __call__(self, _: httpx.Request) -> httpx.Response:
        self.fetches += 1
        status, payload = self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]
        await asyncio.sleep(0)
        return httpx.Response(status, json=payload)

    def catalogue(self, **options: Any) -> Catalogue:
        return Catalogue(http=httpx.AsyncClient(transport=httpx.MockTransport(self)), **options)


ANTHROPIC_REPLY: dict[str, Any] = {
    "model": "claude-haiku-5-5-20260901",
    "content": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 4},
}


async def test_rows_stale_past_the_bound_price_the_call_but_bound_none(caplog: pytest.LogCaptureFixture) -> None:
    server = CatalogueServer((200, {"data": ROWS}), (503, {"error": "down"}))
    shared = server.catalogue(refresh_after=timedelta(0))  # every load refreshes; the bound follows it, to zero
    haiku = Model(
        "claude-haiku-5-5",
        provider="anthropic",
        api_key="test-key",
        catalogue=shared,
        http=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=ANTHROPIC_REPLY))),
    )
    await haiku.generate(system="", user="x")
    stale = await haiku.generate(system="", user="x")

    assert stale.cost_usd == pytest.approx(10 * 0.0000001 + 4 * 0.0000005), "a completion still prices"
    assert stale.pricing_stale_since is not None and shared.stale_since is not None
    with caplog.at_level(logging.WARNING):
        assert haiku.price_ceiling(system="", user="x") is None, "an enforced cap never holds a call on old prices"
    assert any("ceiling_stale_after" in record.getMessage() for record in caplog.records)


async def test_a_bound_above_refresh_after_keeps_the_ceiling_through_a_failed_refresh() -> None:
    server = CatalogueServer((200, {"data": ROWS}), (503, {"error": "down"}))
    shared = server.catalogue(refresh_after=timedelta(0), ceiling_stale_after=timedelta(hours=1))
    await shared.load()
    await shared.load()
    assert server.fetches == 2 and shared.stale_since is not None
    assert model("anthropic/claude-haiku-5.5", shared).price_ceiling(system="", user="x") is not None


# --- an idle host heals on its next out-of-run call ---------------------------------------------------------------


#: How long the rows are kept, and so how old they may be for a ceiling, in the idle tests: short enough to wait out.
_BRIEFLY = timedelta(milliseconds=50)


class _Draft(EvalBaseModel):
    """What the paid call's job returns."""

    text: str


async def _paid_call(host: Any, budget: OutOfRunBudget) -> None:
    """One out-of-run call on the shipped client, through ``start_paid_call``, awaited to the end of its job."""

    async def make(client: BoundCompletionClient, admitted: AdmittedCall) -> _Draft:
        await client.aclose()
        return _Draft(text="drafted")

    started = await start_paid_call(
        host,
        CONFORMANCE_SCOPE,
        action="rubric_propose",
        role="proposer",
        model=HAIKU["id"],
        purpose="proposer",
        budget=budget,
        call=PlannedCall(system="Draft a rubric.", user="the persona"),
        make=make,
        label="a capability rubric",
        budget_s=60.0,
    )
    await await_job(host, started, CONFORMANCE_SCOPE)


def _idle_host(server: CatalogueServer, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, OutOfRunBudget, Catalogue]:
    """A host on the shipped client over ``server``'s catalogue, its out-of-run cap enforced."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    shared = server.catalogue(refresh_after=_BRIEFLY)
    never = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500)))
    built = off_loop(conformance_host, CompletionGuard(), Clients(catalogue=shared, http=never))
    eval_host = built.host.eval_host
    budget = OutOfRunBudget(eval_host.storage, CONFORMANCE_SCOPE, 1.0, blocking_executor=eval_host.blocking_executor)
    return built.host, budget, shared


async def test_an_idle_hosts_next_out_of_run_call_refreshes_the_rows_and_is_priced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = CatalogueServer((200, {"data": ROWS}))
    host, budget, shared = _idle_host(server, monkeypatch)
    await shared.load()
    await asyncio.sleep(2 * _BRIEFLY.total_seconds())  # idle past ceiling_stale_after: no call refreshed the rows

    await _paid_call(host, budget)

    assert server.fetches == 2, "the call refreshed the rows before pricing"
    assert budget.committed_usd > 0, "and was admitted on a ceiling from them"


async def test_rows_a_failed_refresh_left_past_the_bound_refuse_the_out_of_run_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = CatalogueServer((200, {"data": ROWS}), (503, {"error": "down"}))
    host, budget, shared = _idle_host(server, monkeypatch)
    await shared.load()
    await asyncio.sleep(2 * _BRIEFLY.total_seconds())

    with pytest.raises(ValidationFailedError, match="price_ceiling returned None"):
        await _paid_call(host, budget)

    assert server.fetches == 2 and shared.stale_since is not None
    assert budget.committed_usd == 0 and budget.recorded == ()


# --- one load in flight ------------------------------------------------------------------------------------------


async def test_loads_made_together_during_a_failing_first_fetch_share_one_fetch_and_its_error() -> None:
    server = CatalogueServer((503, {"error": "down"}), (200, {"data": ROWS}))
    shared = server.catalogue()
    outcomes = await asyncio.gather(*(shared.load() for _ in range(5)), return_exceptions=True)
    assert server.fetches == 1, "five callers, one fetch"
    assert all(isinstance(outcome, httpx.HTTPStatusError) for outcome in outcomes)
    assert not shared.loaded

    await asyncio.gather(*(shared.load() for _ in range(5)))
    assert server.fetches == 2 and shared.loaded, "a load after the failure fetches again, once"


async def test_a_cancelled_caller_does_not_cancel_the_load_the_others_await() -> None:
    server = CatalogueServer((200, {"data": ROWS}))
    shared = server.catalogue()
    first = asyncio.create_task(shared.load())
    second = asyncio.create_task(shared.load())
    await asyncio.sleep(0)
    first.cancel()
    await second
    assert server.fetches == 1 and shared.loaded


# --- one request, one catalogue fetch ----------------------------------------------------------------------------


def _replying_haiku(shared: Catalogue) -> Model:
    """Haiku, called directly over a scripted reply, priced from ``shared``."""
    reply = httpx.MockTransport(lambda _: httpx.Response(200, json=ANTHROPIC_REPLY))
    return Model(
        "claude-haiku-5-5",
        provider="anthropic",
        api_key="test-key",
        catalogue=shared,
        http=httpx.AsyncClient(transport=reply),
    )


async def _request(client: Any) -> Any:  # Any: a Model, or a lease Clients hands the engine
    """One request as the engine makes it: the client's prices brought up to date, then its call."""
    await client.refresh_prices()
    return await client.generate(system="", user="x")


async def test_a_request_whose_first_read_fails_makes_one_fetch_and_the_next_request_tries_again() -> None:
    server = CatalogueServer((503, {"error": "down"}))
    haiku = _replying_haiku(server.catalogue())

    completion = await _request(haiku)
    assert server.fetches == 1, "the call took its refresh's failed read as its own"
    assert completion.cost_usd is None and "could not be read" in completion.pricing_failure

    await _request(haiku)
    assert server.fetches == 2, "a request later reads again"


async def test_a_request_whose_refresh_fails_makes_one_fetch_and_bounds_nothing_on_the_stale_rows() -> None:
    server = CatalogueServer((200, {"data": ROWS}), (503, {"error": "down"}))
    shared = server.catalogue(refresh_after=timedelta(0))  # every load is due a refresh, so a second would fetch
    await shared.load()
    haiku = _replying_haiku(shared)

    completion = await _request(haiku)
    assert server.fetches == 2, "boot's read, then the request's one refresh"
    assert completion.cost_usd is not None and completion.pricing_stale_since is not None
    assert haiku.price_ceiling(system="", user="x") is None, "only a failed refresh leaves the rows past the bound"


async def test_a_call_made_without_a_refresh_still_reads_for_itself() -> None:
    server = CatalogueServer((503, {"error": "down"}))
    haiku = _replying_haiku(server.catalogue())
    await _request(haiku)
    await haiku.generate(system="", user="x")
    assert server.fetches == 2, "the refresh's read went to the call made for it, not to every call after"


def _counted_reads(shared: Catalogue, monkeypatch: pytest.MonkeyPatch) -> list[PriceRefresh]:
    """Every read of ``shared`` made from here on, as it is made."""
    reads: list[PriceRefresh] = []
    read = shared.read

    async def counted() -> PriceRefresh:
        found = await read()
        reads.append(found)
        return found

    monkeypatch.setattr(shared, "read", counted)
    return reads


async def test_calls_made_together_after_one_refresh_take_its_read_once(monkeypatch: pytest.MonkeyPatch) -> None:
    shared = CatalogueServer((200, {"data": ROWS})).catalogue()
    haiku = _replying_haiku(shared)
    await haiku.refresh_prices()
    reads = _counted_reads(shared, monkeypatch)

    await asyncio.gather(haiku.generate(system="", user="x"), haiku.generate(system="", user="y"))
    assert len(reads) == 1, "one call took the refresh's read, the other read for itself"


async def test_an_estimate_spends_the_refresh_so_a_later_call_reads_for_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    shared = CatalogueServer((200, {"data": ROWS})).catalogue()
    haiku = _replying_haiku(shared)
    await haiku.refresh_prices()
    reads = _counted_reads(shared, monkeypatch)

    haiku.price_ceiling(system="", user="x")  # the estimate, which spends the read
    await haiku.generate(system="", user="x")
    assert len(reads) == 1


async def test_a_failed_refresh_is_not_reused_by_a_second_call_in_a_task_started_after_it() -> None:
    server = CatalogueServer((503, {"error": "down"}))
    haiku = _replying_haiku(server.catalogue())
    await haiku.refresh_prices()

    first = await asyncio.create_task(haiku.generate(system="", user="x"))
    second = await haiku.generate(system="", user="x")
    assert server.fetches == 2, "the task took the refresh's failed read; the call after it tried again"
    assert first.cost_usd is None and second.cost_usd is None


async def test_concurrent_requests_still_share_one_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    server = CatalogueServer((503, {"error": "down"}))
    reply = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=ANTHROPIC_REPLY)))
    template = Model("claude-haiku-5-5", provider="anthropic", http=reply)
    clients = Clients(default=template, catalogue=server.catalogue())
    judge, analysis = clients("judge", None), clients("analysis", None)

    completions = await asyncio.gather(_request(judge), _request(judge), _request(analysis))
    assert server.fetches == 1, "three requests over one catalogue, one fetch"
    assert all(completion.cost_usd is None for completion in completions)


# --- what a refresh found -----------------------------------------------------------------------------------------


async def test_a_refresh_whose_first_read_fails_says_so_and_the_model_keeps_it() -> None:
    haiku = _replying_haiku(CatalogueServer((503, {"error": "down"})).catalogue())
    assert haiku.last_price_refresh is None, "nothing refreshed yet"

    found = await haiku.refresh_prices()
    assert found == PriceRefresh(refreshed=False, fetched_at=None, failure=found.failure)
    assert found.failure is not None and "HTTPStatusError" in found.failure and "503" in found.failure
    assert haiku.last_price_refresh == found


async def test_a_refresh_that_succeeds_says_when_and_a_failed_one_after_it_keeps_that_time() -> None:
    server = CatalogueServer((200, {"data": ROWS}), (503, {"error": "down"}))
    haiku = _replying_haiku(server.catalogue(refresh_after=timedelta(0)))

    fresh = await haiku.refresh_prices()
    assert fresh.refreshed and fresh.fetched_at is not None and fresh.failure is None

    stale = await haiku.refresh_prices()
    assert not stale.refreshed and stale.fetched_at == fresh.fetched_at, "the rows read before are kept"
    assert stale.failure is not None and "could not be refreshed" in stale.failure
    assert haiku.last_price_refresh == stale


async def test_clients_refreshes_its_one_catalogue_once_for_every_role(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    server = CatalogueServer((200, {"data": ROWS}))
    clients = Clients(default="vendor/judge", catalogue=server.catalogue())

    found = await clients.refresh_prices()
    assert server.fetches == 1 and found.refreshed and found.failure is None
    assert await clients("judge", None).refresh_prices() == PriceRefresh(  # type: ignore[attr-defined]
        refreshed=False, fetched_at=found.fetched_at, failure=None
    ), "a role's lease reads the same rows, young enough to keep"


# --- the client a model opens for itself -------------------------------------------------------------------------


class Opened:
    """Stands in for ``httpx.AsyncClient``: builds the real one over a scripted transport, keeping what it was asked
    for and each request's headers and timeout."""

    def __init__(self, answer: Mapping[str, Any]) -> None:
        self._real = httpx.AsyncClient
        self._answer = answer
        self.asked: list[dict[str, Any]] = []
        self.requests: list[httpx.Request] = []

    def __call__(self, **options: Any) -> httpx.AsyncClient:
        if options.get("transport") is not None:  # a test's own scripted client, or one opened on a transport given
            return self._real(**options)
        options.pop("transport", None)
        self.asked.append(options)
        return self._real(transport=httpx.MockTransport(self._answered), **options)

    def _answered(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=self._answer)


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> Opened:
    stand_in = Opened(ANTHROPIC_REPLY)
    monkeypatch.setattr(httpx, "AsyncClient", stand_in)
    return stand_in


async def test_a_model_opens_its_client_with_the_timeout_and_headers_it_was_given(opened: Opened) -> None:
    shared = catalogue()
    claude = Model(
        "claude-haiku-5-5",
        provider="anthropic",
        api_key="test-key",
        catalogue=shared,
        timeout=45.0,
        headers={"x-title": "my-evals", "x-api-key": "not-this-one"},
    )
    await claude.generate(system="", user="x")
    await claude.aclose()
    sent = opened.requests[-1]
    assert sent.extensions["timeout"] == {"connect": 45.0, "read": 45.0, "write": 45.0, "pool": 45.0}
    assert sent.headers["x-title"] == "my-evals"
    assert sent.headers["x-api-key"] == "test-key", "the request's own key wins over a header of the same name"


async def test_a_model_given_no_timeout_opens_its_client_with_the_default(opened: Opened) -> None:
    claude = Model("claude-haiku-5-5", provider="anthropic", api_key="test-key", catalogue=catalogue())
    await claude.generate(system="", user="x")
    await claude.aclose()
    assert opened.requests[-1].extensions["timeout"] == {"connect": 20.0, "read": 600.0, "write": 600.0, "pool": 600.0}


def test_transport_timeout_or_headers_beside_a_client_handed_in_is_refused() -> None:
    http = httpx.AsyncClient(timeout=7.0, headers={"x-title": "mine"})
    for options in ({"timeout": 5.0}, {"headers": {"x-title": "theirs"}}, {"transport": CountingTransport({})}):
        with pytest.raises(ValueError, match="set them on that client instead"):
            Model("vendor/x", api_key="test-key", http=http, **options)
    Model("vendor/x", api_key="test-key", http=http)
    assert http.timeout.read == 7.0 and http.headers["x-title"] == "mine", "a client handed in is used as built"


async def test_derive_keeps_the_timeout_and_headers(opened: Opened) -> None:
    template = Model("claude-haiku-5-5", provider="anthropic", api_key="test-key", timeout=9.0, headers={"x-a": "b"})
    derived = template.derive(temperature=None, catalogue=catalogue())
    await derived.generate(system="", user="x")
    await derived.aclose()
    assert opened.requests[-1].extensions["timeout"]["read"] == 9.0 and opened.requests[-1].headers["x-a"] == "b"


class CountingTransport(httpx.MockTransport):
    """A transport answering every request with ``answer``, counting the requests and its closes."""

    def __init__(self, answer: Mapping[str, Any]) -> None:
        super().__init__(self._answered)
        self._answer = answer
        self.requests = 0
        self.closes = 0

    def _answered(self, _: httpx.Request) -> httpx.Response:
        self.requests += 1
        return httpx.Response(200, json=self._answer)

    async def aclose(self) -> None:
        self.closes += 1


def _haiku(**options: Any) -> Model:
    """Haiku, called directly, priced from :data:`ROWS`."""
    return Model("claude-haiku-5-5", provider="anthropic", api_key="test-key", catalogue=catalogue(), **options)


async def test_a_model_opens_its_client_on_the_transport_it_was_given_and_closing_it_closes_that() -> None:
    transport = CountingTransport(ANTHROPIC_REPLY)
    claude = _haiku(transport=transport, timeout=45.0, headers={"x-title": "my-evals"})
    await claude.generate(system="", user="x")
    assert transport.requests == 1 and transport.closes == 0
    await claude.aclose()
    await claude.aclose()
    assert transport.closes == 1, "closed with the client the model opened on it, once"


@pytest.mark.parametrize("own_http", [False, True])
async def test_a_client_handed_in_is_closed_by_the_model_only_when_it_owns_it(own_http: bool) -> None:
    http = httpx.AsyncClient(transport=CountingTransport(ANTHROPIC_REPLY))
    claude = _haiku(http=http, own_http=own_http)
    await claude.generate(system="", user="x")
    await claude.aclose()
    assert http.is_closed is own_http


def test_own_http_without_a_client_is_refused() -> None:
    with pytest.raises(ValueError, match="own_http=True says the model closes"):
        Model("vendor/x", api_key="test-key", own_http=True)
    with pytest.raises(ValueError, match="own_http=True says the factory closes"):
        Clients(default="vendor/x", own_http=True)


async def test_a_derived_model_sends_through_the_templates_client_and_never_closes_it() -> None:
    http = httpx.AsyncClient(transport=CountingTransport(ANTHROPIC_REPLY))
    template = _haiku(http=http, own_http=True)
    derived = template.derive(temperature=0.5, catalogue=catalogue())
    await derived.generate(system="", user="x")
    await derived.aclose()
    assert not http.is_closed, "the template owns it"
    await template.aclose()
    assert http.is_closed


async def test_a_derived_model_opens_its_own_client_on_the_templates_transport() -> None:
    transport = CountingTransport(ANTHROPIC_REPLY)
    derived = _haiku(transport=transport).derive(temperature=None, catalogue=catalogue())
    await derived.generate(system="", user="x")
    await derived.aclose()
    assert transport.requests == 1 and transport.closes == 1


@pytest.mark.parametrize("own_http", [False, True])
async def test_clients_closes_the_client_handed_in_after_its_models_only_when_it_owns_it(
    own_http: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    http = httpx.AsyncClient(transport=CountingTransport(ANTHROPIC_REPLY))
    clients = Clients(default="vendor/judge", catalogue=catalogue(), http=http, own_http=own_http)
    lease = clients("judge", None)
    await lease.aclose()
    assert not http.is_closed, "closing a lease releases nothing"
    await clients.aclose()
    assert http.is_closed is own_http


async def test_a_catalogue_opens_its_fetch_with_the_exported_timeout(opened: Opened) -> None:
    assert CATALOGUE_FETCH_TIMEOUT_S == 30.0
    with pytest.raises(ValueError, match="holds no `data` list"):
        await Catalogue().load()  # the stand-in answers a reply, not a catalogue: the fetch is what is read here
    timeout = CATALOGUE_FETCH_TIMEOUT_S
    assert opened.requests[-1].extensions["timeout"] == {
        "connect": timeout,
        "read": timeout,
        "write": timeout,
        "pool": timeout,
    }


# --- a completion priced as fully as the ceiling bounds it --------------------------------------------------------

#: Written for these tests: a row listing a price the client does not know the unit of.
ODD: dict[str, Any] = {
    "id": "vendor/odd",
    "pricing": {"prompt": "0.000001", "completion": "0.000001", "video_output": "0.1"},
    "supported_parameters": ["max_tokens"],
}


def _replying(answer: Mapping[str, Any]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=answer)))


async def _anthropic_completion(usage: Mapping[str, Any]) -> Any:
    """Haiku, called directly, answering with ``usage``, priced by :data:`HAIKU`."""
    reply = {**ANTHROPIC_REPLY, "usage": usage}
    haiku = Model("claude-haiku-5-5", provider="anthropic", api_key="k", catalogue=catalogue(), http=_replying(reply))
    return await haiku.generate(system="", user="x")


async def _compatible_completion(model_id: str, usage: Mapping[str, Any]) -> Any:
    """``model_id`` on an OpenAI-compatible endpoint, answering with ``usage``, priced by its catalogue row."""
    reply = {"model": model_id, "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": usage}
    compatible = Model(
        model_id,
        base_url="https://compatible.test/v1",
        api_key="k",
        catalogue=catalogue(*ROWS, ODD),
        http=_replying(reply),
    )
    return await compatible.generate(system="", user="x")


async def test_a_prompt_past_a_tier_is_priced_at_the_tier_where_the_old_price_was_below_the_bill() -> None:
    priced = await _anthropic_completion({"input_tokens": 150_000, "output_tokens": 1000})
    old_price = 150_000 * 0.0000001 + 1000 * 0.0000005  # .29: the base rates
    bill = 150_000 * 0.0000005 + 1000 * 0.0000025  # the 100,000-token tier
    assert old_price < bill
    assert priced.cost_usd == pytest.approx(bill)
    assert priced.price_source == "openrouter-catalogue:anthropic/claude-haiku-5.5"


async def test_a_prompt_below_the_tier_is_priced_at_the_base_rates() -> None:
    priced = await _anthropic_completion({"input_tokens": 10, "output_tokens": 4})
    assert priced.cost_usd == pytest.approx(10 * 0.0000001 + 4 * 0.0000005)


async def test_reasoning_tokens_are_priced_at_the_reasoning_rate_and_the_request_fee_added() -> None:
    usage = {"prompt_tokens": 10, "completion_tokens": 100, "completion_tokens_details": {"reasoning_tokens": 80}}
    priced = await _compatible_completion("reasoner", usage)
    old_price = 10 * 0.000001 + 100 * 0.000002
    bill = 10 * 0.000001 + 80 * 0.000008 + 20 * 0.000002 + 0.005
    assert old_price < bill
    assert priced.cost_usd == pytest.approx(bill)
    assert priced.price_source == "openrouter-catalogue:vendor/reasoner"


async def test_reasoning_the_reply_does_not_split_out_leaves_the_call_unpriced() -> None:
    priced = await _compatible_completion("reasoner", {"prompt_tokens": 10, "completion_tokens": 100})
    assert priced.cost_usd is None and priced.price_source is None
    assert "did not say how many of its output tokens were reasoning" in (priced.pricing_failure or "")


async def test_cache_reads_and_writes_are_priced_at_their_rates_when_the_reply_splits_them() -> None:
    usage = {
        "input_tokens": 100,
        "output_tokens": 4,
        "cache_read_input_tokens": 1000,
        "cache_creation_input_tokens": 500,
        "cache_creation": {"ephemeral_5m_input_tokens": 300, "ephemeral_1h_input_tokens": 200},
    }
    priced = await _anthropic_completion(usage)
    bill = 100 * 0.0000001 + 1000 * 0.00000001 + 300 * 0.000000125 + 200 * 0.0000002 + 4 * 0.0000005
    assert priced.cost_usd == pytest.approx(bill)
    assert priced.price_source == "openrouter-catalogue:anthropic/claude-haiku-5.5", "every share at its own rate"


async def test_cache_writes_not_split_by_lifetime_are_priced_at_the_dearer_rate_and_marked_approximate() -> None:
    priced = await _anthropic_completion({"input_tokens": 100, "output_tokens": 4, "cache_creation_input_tokens": 500})
    assert priced.cost_usd == pytest.approx(100 * 0.0000001 + 500 * 0.0000002 + 4 * 0.0000005)
    assert priced.price_source == "openrouter-catalogue:anthropic/claude-haiku-5.5;cache-approximate"


async def test_a_cache_read_on_a_row_listing_no_cache_rate_is_priced_plain_and_marked_approximate() -> None:
    usage = {"prompt_tokens": 100, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 60}}
    priced = await _compatible_completion("gpt-audio", usage)
    assert priced.cost_usd == pytest.approx(100 * 0.0000025 + 4 * 0.00001)
    assert priced.price_source == "openrouter-catalogue:openai/gpt-audio;cache-approximate"


async def test_a_row_listing_a_price_the_client_cannot_read_leaves_the_call_unpriced() -> None:
    priced = await _compatible_completion("odd", {"prompt_tokens": 10, "completion_tokens": 4})
    assert priced.cost_usd is None
    assert "lists a `video_output` price (0.1) the client does not know the unit of" in (priced.pricing_failure or "")
