"""KP-B3: every key holds a recognisable sentinel secret; driven through a pooled chat model's every state
and error path, no secret, nor any 8-character piece of one, reaches a surface an operator or a sink sees.

The surfaces: log records at every level (message, arguments, extra data, tracebacks), every exception a
caller or a callback receives (``str``, ``repr``, args, attributes and chained causes), ``status()``, the
metric exposition of a private registry, the spans the usage tracker exports, the usage records a sink
receives, what LangChain hands a caller's callbacks (its ``serialized`` payload included), and everything
the store was sent or holds (account names, reservations and the pool's record).

The search looks for every 8-character window of each secret, so a prefix, a suffix or any slice of 8 or
more characters is caught as well as the whole. The test checks each surface captured something, so a
capture that broke can't pass it by finding nothing.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import traceback
from collections.abc import Awaitable
from typing import Any

import anthropic
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import CollectorRegistry, generate_latest

from threetears.models.circuit_breaker import CircuitBreakerRegistry
from threetears.models.key_pool import (
    KeyPool,
    KeyPoolMetrics,
    KeyPoolStoreError,
    KeySpec,
    MoveReason,
    NoKeyAvailableError,
)
from threetears.models.tracking import UsageTracker

from . import key_error_fixtures as fx
from .key_pool_support import PROVIDER, FakeFlakyStore, ManualClock, make_pool, spec, tripping_registry
from .pooled_chat_support import KeyedWire, RecordingSink, partial_then, pooled_model, refusal

#: each key's secret: its first 8 characters name the key, and no part of it is text anything else holds
_SENTINELS = {
    "a": "Qv7SENTa-x9Lk2Pq8Vw4Rt6Ny3Bz5Hm1",
    "b": "Qv7SENTb-j4Fs8Dg2Kh6Lz0Xc3Vb7Nm9",
    "c": "Qv7SENTc-w2Er5Ty8Ui1Op4As7Df0Gh3",
}

#: the shortest piece of a secret that counts as leaked
_WINDOW = 8


def _pieces(secret: str) -> set[str]:
    return {secret[start : start + _WINDOW] for start in range(len(secret) - _WINDOW + 1)}


def _key(label: str, budget: str = "10", **settings: Any) -> KeySpec:  # noqa: ANN401 -- spec()'s settings
    return spec(label, budget, secret=_SENTINELS[label], **settings)


def _exception_texts(exc: BaseException) -> list[str]:
    """Everything an exception shows: its text, args and attributes, its traceback, and each exception
    chained to it, as cause or context."""
    texts: list[str] = []
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        texts += [str(current), repr(current), repr(current.args), *(repr(arg) for arg in current.args)]
        texts += [repr(vars(current)), "".join(traceback.format_exception(current))]
        pending += [chained for chained in (current.__cause__, current.__context__) if chained is not None]
        if isinstance(current, BaseExceptionGroup):
            pending += list(current.exceptions)
    return texts


def _log_texts(records: list[logging.LogRecord]) -> list[str]:
    """Every log record's message, its whole ``__dict__`` (arguments and extra data), and its
    exception, walked as :func:`_exception_texts` walks one."""
    texts: list[str] = []
    for record in records:
        texts += [record.getMessage(), repr(vars(record))]
        if record.exc_info and record.exc_info[1] is not None:
            texts += _exception_texts(record.exc_info[1])
    return texts


class _CallerCallback(BaseCallbackHandler):
    """A caller's callback, keeping what LangChain hands it: each start's ``serialized`` payload and
    invocation details, each answer, and each error."""

    run_inline = True

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] = []

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any) -> None:
        self.seen += [repr(serialized), repr(kwargs)]

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # noqa: ANN401 -- LangChain's LLMResult
        self.seen += [repr(response), repr(kwargs)]

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.seen += [*_exception_texts(error), repr(kwargs)]


class TestNoSecretReachesAnySurface:
    async def test_every_path_leaves_no_piece_of_any_secret(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        caplog.set_level(logging.DEBUG)
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        # UsageTracker takes its tracer from trace.get_tracer when built; OTel's global provider can be
        # set once per process, so the lookup is substituted for this test only
        monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
        registry = CollectorRegistry()
        metrics = KeyPoolMetrics(registry)
        sink = RecordingSink()
        tracker = UsageTracker(audit_sink=sink, prom_registry=registry)
        store = FakeFlakyStore()
        clock = ManualClock()
        caller = _CallerCallback()
        config: RunnableConfig = {"callbacks": [caller]}
        rendered: list[str] = []
        raised: list[str] = []

        def build(name: str, *keys: KeySpec, breakers: CircuitBreakerRegistry | None = None) -> KeyPool:
            pool = make_pool(
                *keys,
                store=store,
                clock=clock,
                name=name,
                observer=metrics,
                breaker_registry=breakers if breakers is not None else CircuitBreakerRegistry(),
            )
            rendered.extend([repr(pool), repr(pool.keys), str(pool.keys)])
            return pool

        async def expect(call: Awaitable[object], error: type[BaseException]) -> None:
            with pytest.raises(error) as caught:
                await call
            raised.extend(_exception_texts(caught.value))

        async def answer(model: Any, prompt: str = "hello") -> None:  # noqa: ANN401 -- the pooled model
            rendered.append(repr(await model.ainvoke(prompt, config)))

        async def read_stream(model: Any) -> None:  # noqa: ANN401 -- the pooled model
            async for chunk in model.astream("hello", config):
                rendered.append(repr(chunk))

        wire = KeyedWire("a", "b", "c", secrets=_SENTINELS)
        with wire.serve() as url:
            # -- one pool through each verdict, a breaker's trip and pull-back, and each failure -----
            breakers, monotonic = tripping_registry()
            main = build(
                "secrecy-main", _key("a", priority=0), _key("b", priority=1), _key("c", priority=2), breakers=breakers
            )
            model = pooled_model(main, url, tracker=tracker)
            rendered.append(repr(model))
            await answer(model)  # served on a, and settled
            rendered.append(repr(model.invoke("hello", config)))  # the synchronous path
            await read_stream(model)
            for recorded in (
                fx.ANTHROPIC_BILLING_402,  # exhausted
                fx.ANTHROPIC_AUTHENTICATION_401,  # revoked
                fx.ANTHROPIC_PERMISSION_403,  # forbidden
                fx.ANTHROPIC_RATE_LIMIT_429,  # rate-limited
            ):
                wire.script("a", refusal(recorded))
                await answer(model)  # a refuses: recorded, moved off, and the call fails over to b
                await main.reset("a")  # the operator's reset moves the pool back onto a
            breakers.get(PROVIDER, credential=_SENTINELS["a"]).record_failure()
            await answer(model)  # a's breaker is open: the pool moves off it, tripped
            monotonic.now += 31
            await answer(model)  # a's breaker lets a probe through: the pool pulls back to a
            wire.script("a", partial_then("rate_limit_error", "Rate limited"))
            await expect(read_stream(model), anthropic.APIStatusError)  # failed after a chunk: no retry
            await main.reset("a")
            wire.script("a", refusal(fx.ANTHROPIC_API_ERROR_500))
            await expect(model.ainvoke("hello", config), anthropic.InternalServerError)  # not the key's
            await main.reset("a")
            for label in ("a", "b", "c"):
                wire.script(label, refusal(fx.ANTHROPIC_AUTHENTICATION_401))
            await expect(model.ainvoke("hello", config), NoKeyAvailableError)  # every key revoked in turn
            for label in ("a", "b", "c"):
                await main.reset(label)
            store.down = True
            await expect(model.ainvoke("hello", config), KeyPoolStoreError)  # the store is unreachable
            await expect(metrics.refresh(main), KeyPoolStoreError)
            store.down = False
            store.failing["settle"] = ConnectionError("the ledger's database is unreachable")
            await answer(model)  # served, and its failed settle logged
            store.failing.clear()
            rendered.append(repr(await metrics.refresh(main)))

            # -- a key too small for the call: refused for room, served on the next ------------------
            room = build("secrecy-room", _key("a", "0.012"), _key("b"))
            await answer(pooled_model(room, url, tracker=tracker))
            rendered.append(repr(await metrics.refresh(room)))

            # -- a breaker that refuses the call after the pool chose its key ------------------------
            breakers, monotonic = tripping_registry()
            trip = build("secrecy-trip", _key("a"), _key("b"), breakers=breakers)
            breakers.get(PROVIDER, credential=_SENTINELS["a"]).record_failure()
            monotonic.now += 31  # one probe is due: one call takes it, the other is refused by a's breaker
            trip_model = pooled_model(trip, url, tracker=tracker)
            both = await asyncio.gather(trip_model.ainvoke("hello", config), trip_model.ainvoke("hello", config))
            rendered += [repr(item) for item in both]
            rendered.append(repr(await metrics.refresh(trip)))
        await tracker.drain()

        # every path above ran
        assert {reason for _, _, reason in main.counters.moves} >= {
            MoveReason.EXHAUSTED,
            MoveReason.REVOKED,
            MoveReason.FORBIDDEN,
            MoveReason.RATE_LIMITED,
            MoveReason.TRIPPED,
            MoveReason.PULL_BACK,
            MoveReason.OPERATOR_RESET,
        }
        assert set(main.counters.failovers) == {
            MoveReason.EXHAUSTED,
            MoveReason.REVOKED,
            MoveReason.FORBIDDEN,
            MoveReason.RATE_LIMITED,
        }
        assert main.counters.no_key_available >= 1
        assert main.counters.store_failures["settle"] == 1 and sum(main.counters.store_failures.values()) >= 3
        assert room.counters.room_refusals == {"a": 1}
        assert trip.counters.failovers == {MoveReason.TRIPPED: 1}
        rendered += [repr(vars(pool.counters)) for pool in (main, room, trip)]

        stored = list(store.seen)
        stored += [repr(await store.inner.read_record(name)) for name in sorted(store.records)]
        stored += [repr(sorted(store.accounts)), repr(await store.inner.snapshot(sorted(store.accounts)))]
        spans = exporter.get_finished_spans()
        surfaces = {
            "logs": _log_texts(caplog.records),
            "exceptions": raised,
            "status, reprs and answers": rendered,
            "metric exposition": [generate_latest(registry).decode()],
            "span attributes": [
                repr((span.name, dict(span.attributes or {}), [dict(event.attributes or {}) for event in span.events]))
                for span in spans
            ],
            "usage records": [repr(dataclasses.asdict(record)) for record in sink.records],
            "callbacks": caller.seen,
            "store": stored,
        }
        for where, texts in surfaces.items():
            assert texts, f"nothing was captured from {where}"
        assert not [text for text in surfaces["usage records"] if "'credential_label': None" in text]
        assert any("llm.credential_label" in text for text in surfaces["span attributes"])
        assert "threetears_key_pool_failovers_total" in surfaces["metric exposition"][0]

        leaks = sorted(
            {
                f"{where}: a piece of key {label!r}'s secret, {piece!r}"
                for where, texts in surfaces.items()
                for label, secret in _SENTINELS.items()
                for piece in _pieces(secret)
                if any(piece in text for text in texts)
            }
        )
        assert not leaks, "\n".join(leaks)
