"""A pooled chat model failing over between keys, by the classifier's verdict (KP-H, KP-E4).

A refusal that is the key's moves the call to the next key, once per key; any other failure propagates
unchanged; a stream that has handed the caller a chunk is never retried. A rate limit means retry
(owner, 2026-11-26): the SDK retries it on the same key, or the pool does where the SDK never does
(OpenRouter), and then it reaches the caller unchanged, with nothing about the key changed. A rate limit
asking a wait past the pool's threshold moves on as exhaustion. Each failover is counted by reason
(KP-K1). Every refusal is a recorded provider response, served to the real provider model on loopback.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import anthropic
import httpx
import openrouter.errors
import pytest
from langchain_core.messages import HumanMessage
from prometheus_client import CollectorRegistry

from threetears.models.capabilities import ModelCapabilities, register_capabilities
from threetears.models.enums import ModelTier, ModelType
from threetears.models.factory import create_chat_model
from threetears.models.key_pool import (
    InMemoryKeyPoolStore,
    KeyPool,
    KeyPoolMetrics,
    KeySpec,
    Secret,
    KeyState,
    KeyStatus,
    MoveReason,
    NoKeyAvailableError,
    reservation_estimate,
)
from threetears.models.pricing import cost_of_usage
from threetears.models.tracking import UsageTracker, estimate_input_tokens

from . import key_error_fixtures as fx
from .key_pool_support import (
    NOW,
    POOL,
    PROVIDER,
    FakeFlakyStore,
    ManualClock,
    RecordingSleep,
    make_pool,
    secret_for,
    spec,
    tripping_registry,
)
from .pooled_chat_support import (
    CAPABILITIES,
    MODEL_ID,
    KeyedWire,
    key_status,
    partial_then,
    pooled_model,
    refusal,
)
from .provider_wire import ChatCompletionsWire, serve_http_handler


def _reservation(prompt: str = "hello") -> Decimal:
    tokens = estimate_input_tokens([HumanMessage(content=prompt)])
    return reservation_estimate(CAPABILITIES, input_tokens=tokens)


def _cost() -> Decimal:
    return cost_of_usage(CAPABILITIES, input_tokens=3, output_tokens=2)


def _forbidden_for_the_model(status: KeyStatus) -> bool:
    return MODEL_ID in status.forbidden_models


def _in(state: KeyState) -> Callable[[KeyStatus], bool]:
    return lambda status: status.state is state


def _exhausted_until(until: datetime) -> Callable[[KeyStatus], bool]:
    return lambda status: (status.state, status.until) == (KeyState.EXHAUSTED, until)


def _quick(recorded: fx.RecordedError) -> fx.RecordedError:
    """``recorded`` with a one-millisecond ``retry-after-ms``, which the SDK sleeps on instead of its backoff."""
    return fx.RecordedError(recorded.status, recorded.body, headers={**recorded.headers, "retry-after-ms": "1"})


class TestKeyMovingFailures:
    """A refusal that is the key's is recorded on it, released, and the call served by the next key."""

    @pytest.mark.parametrize(
        ("recorded", "recorded_as"),
        [
            pytest.param(fx.ANTHROPIC_BILLING_402, _in(KeyState.EXHAUSTED), id="exhausted-402"),
            pytest.param(fx.ANTHROPIC_CREDIT_TOO_LOW_400, _in(KeyState.EXHAUSTED), id="exhausted-credit"),
            pytest.param(fx.ANTHROPIC_SPEND_CAP_429, _in(KeyState.EXHAUSTED), id="exhausted-spend-cap"),
            pytest.param(fx.ANTHROPIC_CONTRACT_SPEND_LIMIT_429, _in(KeyState.EXHAUSTED), id="exhausted-contract"),
            pytest.param(
                fx.ANTHROPIC_RATE_LIMIT_429_LONG_WAIT,
                _exhausted_until(NOW + timedelta(hours=1)),
                id="exhausted-until-a-long-waits-end",
            ),
            pytest.param(fx.ANTHROPIC_AUTHENTICATION_401, _in(KeyState.REVOKED), id="revoked"),
            pytest.param(fx.ANTHROPIC_PERMISSION_403, _forbidden_for_the_model, id="forbidden"),
        ],
    )
    def test_the_call_moves_to_the_next_key(
        self, recorded: fx.RecordedError, recorded_as: Callable[[KeyStatus], bool]
    ) -> None:
        pool = make_pool(spec("a"), spec("b"))
        wire = KeyedWire("a", "b")
        wire.script("a", refusal(recorded))
        with wire.serve() as url:
            answer = pooled_model(pool, url).invoke("hello")
        assert answer.content == "ok"
        assert wire.labels_sent() == ["a", "b"]
        a, b = asyncio.run(key_status(pool, "a")), asyncio.run(key_status(pool, "b"))
        assert recorded_as(a)
        # refused before any answer: nothing billed, so a's reservation was released
        assert (a.spent, a.reserved) == (Decimal(0), Decimal(0))
        assert b.active and b.spent > 0

    async def test_every_key_is_tried_at_most_once(self) -> None:
        pool = make_pool(spec("a"), spec("b"))
        wire = KeyedWire("a", "b")
        wire.script("a", refusal(fx.ANTHROPIC_AUTHENTICATION_401))
        wire.script("b", refusal(fx.ANTHROPIC_PERMISSION_403))
        with wire.serve() as url, pytest.raises(NoKeyAvailableError) as raised:
            await pooled_model(pool, url).ainvoke("hello")
        assert wire.labels_sent() == ["a", "b"]
        assert isinstance(raised.value.__cause__, anthropic.PermissionDeniedError)
        assert [(key.label, key.state) for key in raised.value.keys] == [
            ("a", KeyState.REVOKED),
            ("b", KeyState.FORBIDDEN),
        ]

    def test_a_key_whose_breaker_refused_the_call_is_released_and_skipped(self) -> None:
        """Two calls both chose ``a`` while its breaker was due a probe: the second is refused by the
        breaker, which sent nothing, so it is released and served by ``b``."""
        registry, monotonic = tripping_registry()
        registry.get(PROVIDER, credential=secret_for("a")).record_failure()
        monotonic.now += 31  # past the recovery window: one probe is due
        store = FakeFlakyStore()  # yields on every store call, so both calls reserve before either runs
        pool = make_pool(spec("a"), spec("b"), store=store, breaker_registry=registry)
        wire = KeyedWire("a", "b")

        async def both() -> list[str]:
            with wire.serve() as url:
                model = pooled_model(pool, url)
                answers = await asyncio.gather(model.ainvoke("hello"), model.ainvoke("hello"))
            return [str(answer.content) for answer in answers]

        assert asyncio.run(both()) == ["ok", "ok"]
        assert sorted(wire.labels_sent()) == ["a", "b"]
        # the refused lease was released -- the one release here; each served call settled
        assert store.calls.count("release") == 1
        assert store.calls.count("settle") == 2
        a = asyncio.run(key_status(pool, "a"))
        assert (a.spent, a.reserved) == (_cost(), Decimal(0))


class TestFailuresThatAreNotTheKeys:
    """Not the key's: propagated unchanged, on the key it ran on, closed as the classifier says."""

    def test_a_server_error_propagates_and_keeps_its_reservation(self) -> None:
        pool = make_pool(spec("a"), spec("b"))
        wire = KeyedWire("a", "b")
        wire.script("a", refusal(fx.ANTHROPIC_API_ERROR_500))
        with wire.serve() as url, pytest.raises(anthropic.InternalServerError):
            pooled_model(pool, url).invoke("hello")
        assert wire.labels_sent() == ["a"]
        a = asyncio.run(key_status(pool, "a"))
        assert (a.state, a.active, a.spent) == (KeyState.OK, True, _reservation())

    async def test_a_refused_request_propagates_and_is_released(self) -> None:
        pool = make_pool(spec("a"), spec("b"))
        wire = KeyedWire("a", "b")
        wire.script("a", refusal(fx.ANTHROPIC_INVALID_REQUEST_400))
        with wire.serve() as url, pytest.raises(anthropic.BadRequestError):
            await pooled_model(pool, url).ainvoke("hello")
        assert wire.labels_sent() == ["a"]
        a = await key_status(pool, "a")
        assert (a.state, a.spent, a.reserved) == (KeyState.OK, Decimal(0), Decimal(0))


class TestASettleThatFails:
    """A served call whose settle fails still returns its answer; its reservation stays open, settled
    in full when it expires, and the store is asked to settle it once (KP-G5, KP-E5)."""

    async def test_the_answer_is_returned_and_the_reservation_left_open(self) -> None:
        store = FakeFlakyStore()
        pool = make_pool(spec("a"), store=store)
        store.failing["settle"] = ConnectionError("the ledger's database is unreachable")
        with KeyedWire("a").serve() as url:
            answer = await pooled_model(pool, url).ainvoke("hello")
        assert answer.content == "ok"
        assert store.calls.count("settle") == 1
        store.failing.clear()
        assert (await key_status(pool, "a")).reserved == _reservation()

    def test_a_streamed_answer_is_returned_and_the_reservation_left_open(self) -> None:
        store = FakeFlakyStore()
        pool = make_pool(spec("a"), store=store)
        store.failing["settle"] = ConnectionError("the ledger's database is unreachable")
        with KeyedWire("a").serve() as url:
            text = "".join(str(chunk.content) for chunk in pooled_model(pool, url).stream("hello"))
        assert text == "ok"
        assert store.calls.count("settle") == 1


class TestStreams:
    """A stream fails over only before its first chunk (KP-H5)."""

    def test_a_refusal_before_the_first_chunk_fails_over(self) -> None:
        pool = make_pool(spec("a"), spec("b"))
        wire = KeyedWire("a", "b")
        wire.script("a", refusal(fx.ANTHROPIC_BILLING_402))
        with wire.serve() as url:
            text = "".join(str(chunk.content) for chunk in pooled_model(pool, url).stream("hello"))
        assert text == "ok"
        assert wire.labels_sent() == ["a", "b"]

    def test_nothing_is_retried_once_a_chunk_reached_the_caller(self) -> None:
        pool = make_pool(spec("a"), spec("b"))
        wire = KeyedWire("a", "b")
        wire.script("a", partial_then("rate_limit_error", "Rate limited"))
        received: list[str] = []
        with wire.serve() as url, pytest.raises(anthropic.APIStatusError, match="rate_limit_error"):
            for chunk in pooled_model(pool, url).stream("hello"):
                received.append(str(chunk.content))
        assert "par" in received
        assert wire.labels_sent() == ["a"]
        a = asyncio.run(key_status(pool, "a"))
        # a rate limit is no state of the key, and a stream that began may have been billed: in full
        assert (a.state, a.active, a.spent, a.reserved) == (KeyState.OK, True, _reservation(), Decimal(0))

    async def test_nothing_is_retried_once_an_async_chunk_reached_the_caller(self) -> None:
        pool = make_pool(spec("a"), spec("b"))
        wire = KeyedWire("a", "b")
        wire.script("a", partial_then("rate_limit_error", "Rate limited"))
        received: list[str] = []
        with wire.serve() as url, pytest.raises(anthropic.APIStatusError, match="rate_limit_error"):
            async for chunk in pooled_model(pool, url).astream("hello"):
                received.append(str(chunk.content))
        assert "par" in received
        assert wire.labels_sent() == ["a"]
        assert (await key_status(pool, "a")).state is KeyState.OK

    def test_a_stream_its_caller_stops_reading_settles_its_reservation(self) -> None:
        pool = make_pool(spec("a"))
        wire = KeyedWire("a")
        with wire.serve() as url:
            chunks = pooled_model(pool, url).stream("hello")
            next(chunks)
            chunks.close()
        a = asyncio.run(key_status(pool, "a"))
        assert (a.spent, a.reserved) == (_reservation(), Decimal(0))


class TestRateLimits:
    """A rate limit means retry, on the same key; it never moves the key or changes its state."""

    def test_after_the_sdks_retries_a_rate_limit_reaches_the_caller_unchanged(self) -> None:
        registry = CollectorRegistry()
        pool = make_pool(spec("a"), spec("b"), observer=KeyPoolMetrics(registry))
        wire = KeyedWire("a", "b")
        wire.script("a", *[refusal(_quick(fx.ANTHROPIC_RATE_LIMIT_429))] * 3)
        with wire.serve() as url, pytest.raises(anthropic.RateLimitError) as raised:
            pooled_model(pool, url, max_retries=2).invoke("hello")
        assert raised.value.body == fx.ANTHROPIC_RATE_LIMIT_429.body
        assert wire.labels_sent() == ["a", "a", "a"]  # the SDK's two retries, on the same key
        a = asyncio.run(key_status(pool, "a"))
        assert (a.state, a.until, a.active, a.spent, a.reserved) == (KeyState.OK, None, True, Decimal(0), Decimal(0))
        assert not pool.counters.moves
        assert _no_failovers(pool, registry)
        with wire.serve() as url:
            assert pooled_model(pool, url).invoke("hello").content == "ok"
        assert wire.labels_sent()[-1] == "a"

    async def test_the_sdk_retry_that_answers_serves_the_call_on_the_same_key(self) -> None:
        pool = make_pool(spec("a"), spec("b"))
        wire = KeyedWire("a", "b")
        wire.script("a", refusal(_quick(fx.ANTHROPIC_RATE_LIMIT_429)))
        with wire.serve() as url:
            answer = await pooled_model(pool, url, max_retries=2).ainvoke("hello")
        assert answer.content == "ok"
        assert wire.labels_sent() == ["a", "a"]
        assert (await key_status(pool, "a")).spent == _cost()

    def test_a_quota_429_is_not_retried_by_the_sdk_and_moves_to_the_next_key(self) -> None:
        pool = make_pool(spec("a"), spec("b"))
        wire = KeyedWire("a", "b")
        wire.script("a", *[refusal(_quick(fx.ANTHROPIC_SPEND_CAP_429))] * 3)
        with wire.serve() as url:
            assert pooled_model(pool, url, max_retries=2).invoke("hello").content == "ok"
        assert wire.labels_sent() == ["a", "b"]
        assert asyncio.run(key_status(pool, "a")).state is KeyState.EXHAUSTED

    def test_a_long_wait_is_not_slept_on_and_holds_the_key_until_it_ends(self) -> None:
        clock = ManualClock()
        # a one-second threshold, so an SDK that slept on the 2 s wait would cost seconds, not an hour
        pool = make_pool(
            spec("a", priority=1),
            spec("b", priority=2),
            clock=clock,
            rate_limit_exhaustion_wait=timedelta(seconds=1),
        )
        wire = KeyedWire("a", "b")
        long_wait = fx.RecordedError(429, fx.ANTHROPIC_RATE_LIMIT_429.body, headers={"retry-after": "2"})
        wire.script("a", refusal(long_wait))
        with wire.serve() as url:
            model = pooled_model(pool, url, max_retries=2)
            assert model.invoke("hello").content == "ok"
            assert wire.labels_sent() == ["a", "b"]  # the SDK did not sleep on a and retry it
            assert pool.counters.failovers == {MoveReason.EXHAUSTED: 1}
            a = asyncio.run(key_status(pool, "a"))
            assert (a.state, a.until) == (KeyState.EXHAUSTED, NOW + timedelta(seconds=2))
            clock.advance(seconds=2)
            model.invoke("hello")
        assert wire.labels_sent()[-1] == "a"  # back on the preferred key once the wait is over

    def test_a_one_key_pool_keeps_the_sdks_retry_after_wait(self) -> None:
        pool = make_pool(spec("only"))
        wire = KeyedWire("only")
        wire.script("only", refusal(_quick(fx.ANTHROPIC_RATE_LIMIT_429)))
        with wire.serve() as url:
            assert pooled_model(pool, url, max_retries=2).invoke("hello").content == "ok"
        assert wire.labels_sent() == ["only", "only"]


# -- openrouter: the pool's own retries, an oversized request, a 403 budget -------------------------

#: A model only these tests register: priced, with a bounded output, served by openrouter.
_OR_MODEL = "pool-test-openrouter/scripted"

register_capabilities(
    _OR_MODEL,
    ModelCapabilities(
        model_name=_OR_MODEL,
        model_type=ModelType.CHAT,
        model_tier=ModelTier.SMALL,
        provider_name="openrouter",
        max_output_tokens=1000,
        cost_per_input_token=Decimal("0.000001"),
        cost_per_output_token=Decimal("0.000002"),
    ),
)


def _or_secret(label: str) -> str:
    return f"sk-or-v1-{label}-0123456789abcdef"


@dataclass
class _OpenRouterWire:
    """The OpenRouter chat API, answering each key's requests from that key's script, in order; a key
    whose script has run out gets an answer. Every request is kept, by the key that sent it."""

    labels: tuple[str, ...]
    scripts: dict[str, list[fx.RecordedError]] = field(default_factory=dict)
    sent: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def script(self, label: str, *answers: fx.RecordedError) -> None:
        self.scripts.setdefault(label, []).extend(answers)

    def handle(self, request: httpx.Request) -> httpx.Response:
        token = request.headers["authorization"].removeprefix("Bearer ")
        label = next(label for label in self.labels if _or_secret(label) == token)
        with self._lock:
            self.sent.append(label)
            script = self.scripts.get(label, [])
            recorded = script.pop(0) if script else None
        if recorded is not None:
            return recorded.response()
        json.loads(request.content)  # a real request body, as the SDK sent it
        return ChatCompletionsWire().handle(request)

    @contextmanager
    def serve(self) -> Iterator[str]:
        with serve_http_handler(self.handle) as base_url:
            yield base_url


def _or_pool(*labels: str, clock: ManualClock, sleep: RecordingSleep) -> KeyPool:
    keys = [KeySpec(Secret(_or_secret(label)), label, Decimal("10")) for label in labels]
    return KeyPool("openrouter", POOL, keys, InMemoryKeyPoolStore(), clock=clock, sleep=sleep)


def _or_model(pool: KeyPool, url: str) -> object:
    return create_chat_model(_OR_MODEL, key_pool=pool, tracker=UsageTracker(), base_url=url)


def _or_status(pool: KeyPool, label: str) -> KeyStatus:
    return asyncio.run(key_status(pool, label))


class TestOpenRouter:
    """OpenRouter's SDK never retries a 429, so the pool does: twice on the same key, then the error."""

    def test_a_rate_limit_is_retried_twice_on_the_same_key_then_reaches_the_caller(self) -> None:
        clock = ManualClock()
        sleep = RecordingSleep(clock)
        pool = _or_pool("a", "b", clock=clock, sleep=sleep)
        wire = _OpenRouterWire(("a", "b"))
        limited = fx.RecordedError(429, fx.OPENROUTER_RATE_LIMIT_429.body, headers={"retry-after": "3"})
        wire.script("a", limited, limited, limited)
        with wire.serve() as url, pytest.raises(openrouter.errors.TooManyRequestsResponseError):
            _or_model(pool, url).invoke("hello")  # type: ignore[attr-defined]
        assert wire.sent == ["a", "a", "a"]
        assert sleep.waits == [3.0, 3.0]
        a = _or_status(pool, "a")
        assert (a.state, a.active, a.spent, a.reserved) == (KeyState.OK, True, Decimal(0), Decimal(0))
        assert not pool.counters.moves and not pool.counters.failovers

    def test_a_retry_that_answers_serves_the_call(self) -> None:
        clock = ManualClock()
        sleep = RecordingSleep(clock)
        pool = _or_pool("a", "b", clock=clock, sleep=sleep)
        wire = _OpenRouterWire(("a", "b"))
        wire.script("a", fx.OPENROUTER_RATE_LIMIT_429_RESET_ONLY)  # X-RateLimit-Reset: 12:00:45
        with wire.serve() as url:
            answer = _or_model(pool, url).invoke("hello")  # type: ignore[attr-defined]
        assert answer.content == "ok"
        assert wire.sent == ["a", "a"]
        assert sleep.waits == [45.0]
        assert not pool.counters.failovers

    def test_a_request_too_large_for_the_key_skips_it_for_this_call_only(self) -> None:
        clock = ManualClock()
        pool = _or_pool("a", "b", clock=clock, sleep=RecordingSleep(clock))
        wire = _OpenRouterWire(("a", "b"))
        wire.script("a", fx.OPENROUTER_WEIGHT_EXCEEDS_BUDGET_402)
        with wire.serve() as url:
            model = _or_model(pool, url)
            assert model.invoke("hello").content == "ok"  # type: ignore[attr-defined]
            assert wire.sent == ["a", "b"]
            a = _or_status(pool, "a")
            assert (a.state, a.active, a.spent) == (KeyState.OK, True, Decimal(0))
            assert not pool.counters.moves
            assert pool.counters.failovers == {MoveReason.LOW_ROOM: 1}
            model.invoke("hello")  # type: ignore[attr-defined]
        assert wire.sent[-1] == "a"  # the next call starts on a again

    def test_a_403_budget_holds_the_key_until_its_reset(self) -> None:
        clock = ManualClock()
        pool = _or_pool("a", "b", clock=clock, sleep=RecordingSleep(clock))
        wire = _OpenRouterWire(("a", "b"))
        wire.script("a", fx.OPENROUTER_KEY_BUDGET_403)
        with wire.serve() as url:
            assert _or_model(pool, url).invoke("hello").content == "ok"  # type: ignore[attr-defined]
        assert wire.sent == ["a", "b"]
        a = _or_status(pool, "a")
        assert (a.state, a.until) == (KeyState.EXHAUSTED, datetime(2026, 11, 27, tzinfo=UTC))
        assert pool.counters.moves == {("a", "b", MoveReason.EXHAUSTED): 1}

    @pytest.mark.parametrize(
        "recorded",
        [
            pytest.param(fx.OPENROUTER_CONTENT_POLICY_403, id="content_policy_violation"),
            pytest.param(fx.OPENROUTER_REFUSAL_403, id="refusal"),
        ],
    )
    def test_a_content_refusal_reaches_the_caller_and_leaves_the_key(self, recorded: fx.RecordedError) -> None:
        clock = ManualClock()
        pool = _or_pool("a", "b", clock=clock, sleep=RecordingSleep(clock))
        wire = _OpenRouterWire(("a", "b"))
        wire.script("a", recorded)
        with wire.serve() as url, pytest.raises(openrouter.errors.OpenRouterError):
            _or_model(pool, url).invoke("hello")  # type: ignore[attr-defined]
        assert wire.sent == ["a"]
        assert _or_status(pool, "a").state is KeyState.OK


def _failovers(registry: CollectorRegistry, reason: MoveReason) -> float | None:
    return registry.get_sample_value("threetears_key_pool_failovers_total", {"pool": POOL, "reason": str(reason)})


def _no_failovers(pool: KeyPool, registry: CollectorRegistry) -> bool:
    return not pool.counters.failovers and all(_failovers(registry, reason) in (None, 0.0) for reason in MoveReason)


class TestFailoverCounter:
    """KP-K1: each failover inside a call is counted once, by why the key refused it, in the pool's
    counters and in ``threetears_key_pool_failovers_total``; nothing else is."""

    @pytest.mark.parametrize(
        ("recorded", "reason"),
        [
            pytest.param(fx.ANTHROPIC_BILLING_402, MoveReason.EXHAUSTED, id="exhausted"),
            pytest.param(fx.ANTHROPIC_AUTHENTICATION_401, MoveReason.REVOKED, id="revoked"),
            pytest.param(fx.ANTHROPIC_PERMISSION_403, MoveReason.FORBIDDEN, id="forbidden"),
            pytest.param(fx.ANTHROPIC_RATE_LIMIT_429_LONG_WAIT, MoveReason.EXHAUSTED, id="long-wait"),
        ],
    )
    async def test_each_key_moving_verdict_counts_one_failover_by_its_kind(
        self, recorded: fx.RecordedError, reason: MoveReason
    ) -> None:
        registry = CollectorRegistry()
        pool = make_pool(spec("a"), spec("b"), observer=KeyPoolMetrics(registry))
        wire = KeyedWire("a", "b")
        wire.script("a", refusal(recorded))
        with wire.serve() as url:
            await pooled_model(pool, url).ainvoke("hello")
        assert wire.labels_sent() == ["a", "b"]
        assert pool.counters.failovers == {reason: 1}
        assert _failovers(registry, reason) == 1

    def test_a_breaker_that_refused_the_call_counts_a_tripped_failover(self) -> None:
        breakers, monotonic = tripping_registry()
        breakers.get(PROVIDER, credential=secret_for("a")).record_failure()
        monotonic.now += 31  # one probe is due: the first call takes it, the second is refused by the breaker
        registry = CollectorRegistry()
        pool = make_pool(
            spec("a"), spec("b"), store=FakeFlakyStore(), breaker_registry=breakers, observer=KeyPoolMetrics(registry)
        )
        wire = KeyedWire("a", "b")

        async def both() -> None:
            with wire.serve() as url:
                model = pooled_model(pool, url)
                await asyncio.gather(model.ainvoke("hello"), model.ainvoke("hello"))

        asyncio.run(both())
        assert pool.counters.failovers == {MoveReason.TRIPPED: 1}
        assert _failovers(registry, MoveReason.TRIPPED) == 1

    async def test_two_refusals_in_one_call_count_two_failovers(self) -> None:
        registry = CollectorRegistry()
        pool = make_pool(spec("a"), spec("b"), spec("c"), observer=KeyPoolMetrics(registry))
        wire = KeyedWire("a", "b", "c")
        wire.script("a", refusal(fx.ANTHROPIC_BILLING_402))
        wire.script("b", refusal(fx.ANTHROPIC_AUTHENTICATION_401))
        with wire.serve() as url:
            await pooled_model(pool, url).ainvoke("hello")
        assert wire.labels_sent() == ["a", "b", "c"]
        assert pool.counters.failovers == {MoveReason.EXHAUSTED: 1, MoveReason.REVOKED: 1}
        assert _failovers(registry, MoveReason.EXHAUSTED) == _failovers(registry, MoveReason.REVOKED) == 1

    async def test_a_failure_that_is_not_the_keys_counts_no_failover(self) -> None:
        registry = CollectorRegistry()
        pool = make_pool(spec("a"), spec("b"), observer=KeyPoolMetrics(registry))
        wire = KeyedWire("a", "b")
        wire.script("a", refusal(fx.ANTHROPIC_API_ERROR_500))
        with wire.serve() as url, pytest.raises(anthropic.InternalServerError):
            await pooled_model(pool, url).ainvoke("hello")
        assert wire.labels_sent() == ["a"]
        assert _no_failovers(pool, registry)

    def test_a_partial_stream_failure_counts_no_failover(self) -> None:
        registry = CollectorRegistry()
        pool = make_pool(spec("a"), spec("b"), observer=KeyPoolMetrics(registry))
        wire = KeyedWire("a", "b")
        wire.script("a", partial_then("rate_limit_error", "Rate limited"))
        with wire.serve() as url, pytest.raises(anthropic.APIStatusError):
            for _ in pooled_model(pool, url).stream("hello"):
                pass
        assert _no_failovers(pool, registry)

    async def test_a_refusal_that_leaves_no_key_counts_no_failover(self) -> None:
        registry = CollectorRegistry()
        pool = make_pool(spec("a"), observer=KeyPoolMetrics(registry))
        wire = KeyedWire("a")
        wire.script("a", refusal(fx.ANTHROPIC_AUTHENTICATION_401))
        with wire.serve() as url, pytest.raises(NoKeyAvailableError):
            await pooled_model(pool, url).ainvoke("hello")
        assert _no_failovers(pool, registry)

    def test_a_rate_limit_retried_on_the_same_key_is_not_a_failover(self) -> None:
        registry = CollectorRegistry()
        pool = make_pool(spec("a"), spec("b"), observer=KeyPoolMetrics(registry))
        wire = KeyedWire("a", "b")
        wire.script("a", refusal(_quick(fx.ANTHROPIC_RATE_LIMIT_429)))
        with wire.serve() as url:
            pooled_model(pool, url, max_retries=2).invoke("hello")
        assert wire.labels_sent() == ["a", "a"]
        assert _no_failovers(pool, registry)

    def test_a_rate_limit_that_reaches_the_caller_is_not_a_failover(self) -> None:
        registry = CollectorRegistry()
        pool = make_pool(spec("a"), spec("b"), observer=KeyPoolMetrics(registry))
        wire = KeyedWire("a", "b")
        wire.script("a", refusal(fx.ANTHROPIC_RATE_LIMIT_429))
        with wire.serve() as url, pytest.raises(anthropic.RateLimitError):
            pooled_model(pool, url).invoke("hello")
        assert wire.labels_sent() == ["a"]
        assert _no_failovers(pool, registry)
