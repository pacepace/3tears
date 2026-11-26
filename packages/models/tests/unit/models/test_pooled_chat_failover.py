"""A pooled chat model failing over between keys, by the classifier's verdict (KP-H, KP-E4, Q4).

A refusal that is the key's moves the call to the next key, once per key; any other failure propagates
unchanged; a stream that has handed the caller a chunk is never retried; and a call whose keys are only
rate-limited waits for the earliest hold, up to the pool's cap, under an injected clock. Each failover is
counted by reason (KP-K1). Every refusal is a recorded provider response, served to the real provider
model on loopback.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import timedelta
from decimal import Decimal

import anthropic
import pytest
from langchain_core.messages import HumanMessage
from prometheus_client import CollectorRegistry

from threetears.models.key_pool import (
    KeyPool,
    KeyPoolMetrics,
    KeyState,
    KeyStatus,
    MoveReason,
    NoKeyAvailableError,
    reservation_estimate,
)
from threetears.models.pricing import cost_of_usage
from threetears.models.tracking import estimate_input_tokens

from . import key_error_fixtures as fx
from .key_pool_support import (
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


def _reservation(prompt: str = "hello") -> Decimal:
    tokens = estimate_input_tokens([HumanMessage(content=prompt)])
    return reservation_estimate(CAPABILITIES, input_tokens=tokens)


def _cost() -> Decimal:
    return cost_of_usage(CAPABILITIES, input_tokens=3, output_tokens=2)


def _forbidden_for_the_model(status: KeyStatus) -> bool:
    return MODEL_ID in status.forbidden_models


def _in(state: KeyState) -> Callable[[KeyStatus], bool]:
    return lambda status: status.state is state


class TestKeyMovingFailures:
    """A refusal that is the key's is recorded on it, released, and the call served by the next key."""

    @pytest.mark.parametrize(
        ("recorded", "recorded_as"),
        [
            pytest.param(fx.ANTHROPIC_BILLING_402, _in(KeyState.EXHAUSTED), id="exhausted-402"),
            pytest.param(fx.ANTHROPIC_CREDIT_TOO_LOW_400, _in(KeyState.EXHAUSTED), id="exhausted-credit"),
            pytest.param(fx.ANTHROPIC_SPEND_CAP_429, _in(KeyState.EXHAUSTED), id="exhausted-spend-cap"),
            pytest.param(fx.ANTHROPIC_AUTHENTICATION_401, _in(KeyState.REVOKED), id="revoked"),
            pytest.param(fx.ANTHROPIC_PERMISSION_403, _forbidden_for_the_model, id="forbidden"),
            pytest.param(fx.ANTHROPIC_RATE_LIMIT_429, _in(KeyState.RATE_LIMITED), id="rate-limited"),
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
        wire.script("a", refusal(fx.ANTHROPIC_RATE_LIMIT_429))
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
        # the key's state is still recorded, and a stream that began may have been billed: in full
        assert (a.state, a.spent, a.reserved) == (KeyState.RATE_LIMITED, _reservation(), Decimal(0))

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
        assert (await key_status(pool, "a")).state is KeyState.RATE_LIMITED

    def test_a_stream_its_caller_stops_reading_settles_its_reservation(self) -> None:
        pool = make_pool(spec("a"))
        wire = KeyedWire("a")
        with wire.serve() as url:
            chunks = pooled_model(pool, url).stream("hello")
            next(chunks)
            chunks.close()
        a = asyncio.run(key_status(pool, "a"))
        assert (a.spent, a.reserved) == (_reservation(), Decimal(0))


class TestRateLimitWait:
    """Q4: a call whose keys are only rate-limited waits for the earliest hold, up to the pool's cap."""

    def test_with_the_default_cap_of_zero_the_call_raises_at_once(self) -> None:
        clock = ManualClock()
        sleep = RecordingSleep(clock)
        pool = make_pool(spec("a"), clock=clock, sleep=sleep)
        wire = KeyedWire("a")
        wire.script("a", refusal(fx.ANTHROPIC_RATE_LIMIT_429))
        with wire.serve() as url, pytest.raises(NoKeyAvailableError) as raised:
            pooled_model(pool, url).invoke("hello")
        assert wire.labels_sent() == ["a"]
        assert sleep.waits == []
        assert isinstance(raised.value.__cause__, anthropic.RateLimitError)
        assert raised.value.keys[0].state is KeyState.RATE_LIMITED

    def test_under_a_positive_cap_the_call_waits_out_the_hold_on_the_same_key(self) -> None:
        clock = ManualClock()
        sleep = RecordingSleep(clock)
        pool = make_pool(spec("a"), clock=clock, sleep=sleep, max_rate_limit_wait=timedelta(seconds=60))
        wire = KeyedWire("a")
        wire.script("a", refusal(fx.ANTHROPIC_RATE_LIMIT_429))  # retry-after: 17
        with wire.serve() as url:
            answer = pooled_model(pool, url).invoke("hello")
        assert answer.content == "ok"
        assert wire.labels_sent() == ["a", "a"]
        assert sleep.waits == [17.0]

    async def test_a_key_rate_limited_again_after_its_wait_is_not_tried_a_third_time(self) -> None:
        clock = ManualClock()
        sleep = RecordingSleep(clock)
        pool = make_pool(spec("a"), clock=clock, sleep=sleep, max_rate_limit_wait=timedelta(seconds=60))
        wire = KeyedWire("a")
        wire.script("a", refusal(fx.ANTHROPIC_RATE_LIMIT_429), refusal(fx.ANTHROPIC_RATE_LIMIT_429))
        with wire.serve() as url, pytest.raises(NoKeyAvailableError):
            await pooled_model(pool, url).ainvoke("hello")
        assert wire.labels_sent() == ["a", "a"]
        assert sleep.waits == [17.0]

    def test_a_hold_longer_than_the_cap_raises_without_waiting(self) -> None:
        clock = ManualClock()
        sleep = RecordingSleep(clock)
        pool = make_pool(spec("a"), clock=clock, sleep=sleep, max_rate_limit_wait=timedelta(seconds=10))
        wire = KeyedWire("a")
        wire.script("a", refusal(fx.ANTHROPIC_RATE_LIMIT_429))
        with wire.serve() as url, pytest.raises(NoKeyAvailableError):
            pooled_model(pool, url).invoke("hello")
        assert wire.labels_sent() == ["a"]
        assert sleep.waits == []

    def test_another_key_serves_before_any_wait(self) -> None:
        clock = ManualClock()
        sleep = RecordingSleep(clock)
        pool = make_pool(spec("a"), spec("b"), clock=clock, sleep=sleep, max_rate_limit_wait=timedelta(seconds=60))
        wire = KeyedWire("a", "b")
        wire.script("a", refusal(fx.ANTHROPIC_RATE_LIMIT_429))
        with wire.serve() as url:
            pooled_model(pool, url).invoke("hello")
        assert wire.labels_sent() == ["a", "b"]
        assert sleep.waits == []


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
            pytest.param(fx.ANTHROPIC_RATE_LIMIT_429, MoveReason.RATE_LIMITED, id="rate-limited"),
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
        wire.script("a", refusal(fx.ANTHROPIC_RATE_LIMIT_429))
        wire.script("b", refusal(fx.ANTHROPIC_AUTHENTICATION_401))
        with wire.serve() as url:
            await pooled_model(pool, url).ainvoke("hello")
        assert wire.labels_sent() == ["a", "b", "c"]
        assert pool.counters.failovers == {MoveReason.RATE_LIMITED: 1, MoveReason.REVOKED: 1}
        assert _failovers(registry, MoveReason.RATE_LIMITED) == _failovers(registry, MoveReason.REVOKED) == 1

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

    def test_a_retry_on_the_same_key_after_its_hold_is_not_a_failover(self) -> None:
        clock = ManualClock()
        registry = CollectorRegistry()
        pool = make_pool(
            spec("a"),
            clock=clock,
            sleep=RecordingSleep(clock),
            max_rate_limit_wait=timedelta(seconds=60),
            observer=KeyPoolMetrics(registry),
        )
        wire = KeyedWire("a")
        wire.script("a", refusal(fx.ANTHROPIC_RATE_LIMIT_429))
        with wire.serve() as url:
            pooled_model(pool, url).invoke("hello")
        assert wire.labels_sent() == ["a", "a"]
        assert _no_failovers(pool, registry)
