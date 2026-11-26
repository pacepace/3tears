"""A pooled chat model, ``create_chat_model(key_pool=...)``: every call path, every re-binding, and the
callbacks of the key that served the call (KP-J, KP-E1, KP-E3, KP-L1).

Each test drives the real provider model of each key against one scripted Anthropic Messages API on
loopback, which answers each key from its own script; the pool runs on an in-memory store and an
injected clock. Failover by verdict is ``test_pooled_chat_failover.py``.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import anthropic
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from threetears.models.capabilities import ModelCapabilities, register_capabilities
from threetears.models.circuit_breaker import CircuitBreakerRegistry, CircuitState
from threetears.models.enums import ModelTier, ModelType
from threetears.models.factory import create_chat_model
from threetears.models.key_pool import KeyPool, NoKeyAvailableError, reservation_estimate
from threetears.models.pricing import UnpricedModelError, cost_of_usage
from threetears.models.tracking import UsageTracker, estimate_input_tokens

from . import key_error_fixtures as fx
from .key_pool_support import PROVIDER, make_pool, secret_for, spec, tripping_registry
from .pooled_chat_support import (
    CAPABILITIES,
    MODEL_ID,
    EventLog,
    KeyedWire,
    RecordingSink,
    key_status,
    pooled_model,
    refusal,
    reply,
    tool_call,
)


class _Answer(BaseModel):
    """The shape a structured call asks for."""

    text: str


class _Write(BaseTool):
    """A tool a caller binds."""

    name: str = "write_file"
    description: str = "writes a file"

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError


def _cost(input_tokens: int = 3, output_tokens: int = 2, **cache: int) -> Decimal:
    return cost_of_usage(CAPABILITIES, input_tokens=input_tokens, output_tokens=output_tokens, **cache)


def _reservation(prompt: str, max_tokens: int | None = None) -> Decimal:
    tokens = estimate_input_tokens([HumanMessage(content=prompt)])
    return reservation_estimate(CAPABILITIES, input_tokens=tokens, max_tokens=max_tokens)


def _spent(pool: KeyPool, label: str) -> Decimal:
    return asyncio.run(key_status(pool, label)).spent


# -- every call path ------------------------------------------------------------------------------


class TestCallPaths:
    """Each path reserves on the active key, runs there, and settles to the reported cost (KP-E3)."""

    def test_invoke_settles_the_reported_cost(self) -> None:
        pool = make_pool(spec("a"), spec("b"))
        wire = KeyedWire("a", "b")
        with wire.serve() as url:
            answer = pooled_model(pool, url).invoke("hello")
        assert answer.content == "ok"
        assert wire.labels_sent() == ["a"]
        status = asyncio.run(key_status(pool, "a"))
        assert (status.spent, status.reserved) == (_cost(), Decimal(0))

    async def test_ainvoke_settles_the_reported_cost(self) -> None:
        pool = make_pool(spec("a"), spec("b"))
        wire = KeyedWire("a", "b")
        with wire.serve() as url:
            answer = await pooled_model(pool, url).ainvoke("hello")
        assert answer.content == "ok"
        status = await key_status(pool, "a")
        assert (status.spent, status.reserved) == (_cost(), Decimal(0))

    def test_stream_yields_the_answer_and_settles(self) -> None:
        pool = make_pool(spec("a"))
        wire = KeyedWire("a")
        with wire.serve() as url:
            chunks = list(pooled_model(pool, url).stream("hello"))
        assert "".join(str(chunk.content) for chunk in chunks) == "ok"
        assert wire.exchanges[0].body["stream"] is True
        status = asyncio.run(key_status(pool, "a"))
        assert (status.spent, status.reserved) == (_cost(), Decimal(0))

    async def test_astream_yields_the_answer_and_settles(self) -> None:
        pool = make_pool(spec("a"))
        wire = KeyedWire("a")
        with wire.serve() as url:
            chunks = [chunk async for chunk in pooled_model(pool, url).astream("hello")]
        assert "".join(str(chunk.content) for chunk in chunks) == "ok"
        status = await key_status(pool, "a")
        assert (status.spent, status.reserved) == (_cost(), Decimal(0))

    def test_batch_runs_each_input_as_its_own_call(self) -> None:
        pool = make_pool(spec("a"))
        wire = KeyedWire("a")
        wire.script("a", reply("one"), reply("two"))
        with wire.serve() as url:
            answers = pooled_model(pool, url).batch(["1", "2"], config={"max_concurrency": 1})
        assert [answer.content for answer in answers] == ["one", "two"]
        assert _spent(pool, "a") == 2 * _cost()

    async def test_abatch_runs_each_input_as_its_own_call(self) -> None:
        pool = make_pool(spec("a"))
        wire = KeyedWire("a")
        with wire.serve() as url:
            answers = await pooled_model(pool, url).abatch(["1", "2", "3"])
        assert [answer.content for answer in answers] == ["ok", "ok", "ok"]
        assert (await key_status(pool, "a")).spent == 3 * _cost()

    async def test_a_synchronous_call_from_inside_a_running_loop_is_served(self) -> None:
        """The sync path drives the pool on its own loop, so a caller already on one doesn't deadlock."""
        pool = make_pool(spec("a"))
        wire = KeyedWire("a")
        with wire.serve() as url:
            answer = pooled_model(pool, url).invoke("hello")
        assert answer.content == "ok"
        assert (await key_status(pool, "a")).spent == _cost()

    def test_cache_reads_and_writes_are_settled_at_their_own_rates(self) -> None:
        pool = make_pool(spec("a"))
        wire = KeyedWire("a")
        wire.script("a", reply(input_tokens=3, cache_read_input_tokens=100, cache_creation_input_tokens=50))
        with wire.serve() as url:
            pooled_model(pool, url).invoke("hello")
        # LangChain reports the whole prompt as input, cache tokens included
        expected = _cost(input_tokens=153, cache_read_tokens=100, cache_write_tokens=50)
        assert expected != _cost(input_tokens=153)
        assert _spent(pool, "a") == expected


# -- the reservation (KP-E1) ----------------------------------------------------------------------


class TestReservation:
    """A call reserves its input estimate plus its max_tokens, else the model's max_output_tokens."""

    def test_a_call_whose_output_bound_does_not_fit_is_refused_before_any_request(self) -> None:
        budget = _reservation("hello", max_tokens=100) * 2
        assert budget < _reservation("hello")
        pool = make_pool(spec("a", str(budget)))
        wire = KeyedWire("a")
        with wire.serve() as url, pytest.raises(NoKeyAvailableError):
            pooled_model(pool, url).invoke("hello")
        assert wire.exchanges == []

    @pytest.mark.parametrize("where", ["bind", "call", "factory"])
    def test_the_call_s_max_tokens_bounds_the_reservation(self, where: str) -> None:
        budget = _reservation("hello", max_tokens=100) * 2
        pool = make_pool(spec("a", str(budget)))
        wire = KeyedWire("a")
        with wire.serve() as url:
            if where == "factory":
                pooled_model(pool, url, max_tokens=100).invoke("hello")
            elif where == "bind":
                pooled_model(pool, url).bind(max_tokens=100).invoke("hello")
            else:
                pooled_model(pool, url).invoke("hello", max_tokens=100)
        assert wire.exchanges[0].body["max_tokens"] == 100
        assert _spent(pool, "a") == _cost()

    def test_a_failure_the_provider_may_have_billed_keeps_the_reservation(self) -> None:
        pool = make_pool(spec("a"))
        wire = KeyedWire("a")
        wire.script("a", refusal(fx.ANTHROPIC_API_ERROR_500))
        with wire.serve() as url, pytest.raises(anthropic.InternalServerError):
            pooled_model(pool, url).invoke("hello")
        assert _spent(pool, "a") == _reservation("hello")


# -- re-binding (KP-J2) ---------------------------------------------------------------------------


class TestRebinding:
    """A re-binding reaches whichever key's model serves the call."""

    def test_bind_tools_reaches_the_key_that_fails_over_to(self) -> None:
        pool = make_pool(spec("a"), spec("b"))
        wire = KeyedWire("a", "b")
        wire.script("a", refusal(fx.ANTHROPIC_AUTHENTICATION_401))
        wire.script("b", tool_call("write_file", {"path": "x"}))
        with wire.serve() as url:
            answer = pooled_model(pool, url).bind_tools([_Write()]).invoke("write it")
        assert [call["name"] for call in answer.tool_calls] == ["write_file"]
        assert wire.labels_sent() == ["a", "b"]
        assert all([tool["name"] for tool in e.body["tools"]] == ["write_file"] for e in wire.exchanges)

    async def test_structured_output_is_parsed_from_the_key_that_serves(self) -> None:
        pool = make_pool(spec("a"), spec("b"))
        wire = KeyedWire("a", "b")
        wire.script("a", refusal(fx.ANTHROPIC_BILLING_402))
        wire.script("b", tool_call("_Answer", {"text": "hi"}))
        with wire.serve() as url:
            answer = await pooled_model(pool, url).with_structured_output(_Answer).ainvoke("answer")
        assert answer == _Answer(text="hi")
        assert wire.labels_sent() == ["a", "b"]
        assert (await key_status(pool, "b")).spent == _cost(5, 4)

    def test_a_structured_answer_that_fails_to_parse_settles_its_reported_cost(self) -> None:
        """The provider answered and billed; the parse after it failing is not the key's."""
        pool = make_pool(spec("a"), spec("b"))
        wire = KeyedWire("a", "b")
        wire.script("a", tool_call("_Answer", {"wrong": 1}))
        with wire.serve() as url, pytest.raises(Exception, match="text"):
            pooled_model(pool, url).with_structured_output(_Answer).invoke("answer")
        assert wire.labels_sent() == ["a"]
        assert _spent(pool, "a") == _cost(5, 4)

    def test_with_config_and_bind_tools_compose(self) -> None:
        pool = make_pool(spec("a"), spec("b"))
        wire = KeyedWire("a", "b")
        wire.script("a", refusal(fx.ANTHROPIC_AUTHENTICATION_401))
        wire.script("b", tool_call("write_file", {"path": "x"}))
        log = EventLog()
        with wire.serve() as url:
            model = pooled_model(pool, url).with_config(callbacks=[log]).bind_tools([_Write()])
            answer = model.invoke("write it")
        assert [call["name"] for call in answer.tool_calls] == ["write_file"]
        assert log.events == ["start", "error:AnthropicAuthenticationError", "start", "end"]

    def test_with_retry_retries_inside_the_key_s_attempt(self) -> None:
        pool = make_pool(spec("a"), spec("b"))
        wire = KeyedWire("a", "b")
        wire.script("a", refusal(fx.ANTHROPIC_OVERLOADED_529))
        with wire.serve() as url:
            model = pooled_model(pool, url).with_retry(stop_after_attempt=2, wait_exponential_jitter=False)
            answer = model.invoke("hello")
        assert answer.content == "ok"
        assert wire.labels_sent() == ["a", "a"]
        assert _spent(pool, "a") == _cost()

    def test_attributes_are_the_chat_model_s(self) -> None:
        pool = make_pool(spec("a"), spec("b"))
        with KeyedWire("a", "b").serve() as url:
            model = pooled_model(pool, url, max_tokens=321)
        assert model.max_tokens == 321
        assert model.model == MODEL_ID


# -- the serving key's callbacks (KP-J3, J4) ------------------------------------------------------


class TestServingKeyCallbacks:
    async def test_the_tracker_records_the_serving_key_and_settles_what_it_prices(self) -> None:
        sink = RecordingSink()
        tracker = UsageTracker(audit_sink=sink)
        pool = make_pool(spec("a"), spec("b"))
        wire = KeyedWire("a", "b")
        wire.script("a", refusal(fx.ANTHROPIC_AUTHENTICATION_401))
        with wire.serve() as url:
            await pooled_model(pool, url, tracker=tracker).ainvoke("hello")
        await tracker.drain()
        assert [(record.credential_label, record.cost_usd) for record in sink.records] == [("b", _cost())]
        # the settlement and the tracker price through one function (KP-J5)
        assert (await key_status(pool, "b")).spent == sink.records[0].cost_usd

    def test_the_failure_counts_on_the_serving_key_s_breaker_only(self) -> None:
        registry, _ = tripping_registry()
        pool = make_pool(spec("a"), spec("b"), breaker_registry=registry)
        wire = KeyedWire("a", "b")
        wire.script("a", refusal(fx.ANTHROPIC_API_ERROR_500))
        with wire.serve() as url, pytest.raises(anthropic.InternalServerError):
            pooled_model(pool, url).invoke("hello")
        breaker_a = registry.peek(PROVIDER, credential=secret_for("a"))
        breaker_b = registry.peek(PROVIDER, credential=secret_for("b"))
        assert breaker_a is not None and breaker_a.state is CircuitState.OPEN
        assert breaker_b is None or breaker_b.state is CircuitState.CLOSED

    def test_the_caller_s_callbacks_see_every_attempt(self) -> None:
        pool = make_pool(spec("a"), spec("b"))
        wire = KeyedWire("a", "b")
        wire.script("a", refusal(fx.ANTHROPIC_AUTHENTICATION_401))
        log = EventLog()
        extra = EventLog()
        with wire.serve() as url:
            pooled_model(pool, url, extra_callbacks=[extra]).invoke("hello", {"callbacks": [log]})
        assert log.events == ["start", "error:AnthropicAuthenticationError", "start", "end"]
        assert extra.events == log.events


# -- building one (KP-J1, A4) ---------------------------------------------------------------------


class TestBuilding:
    def test_both_a_key_and_a_pool_is_refused(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            create_chat_model(MODEL_ID, api_key=secret_for("a"), key_pool=make_pool(spec("a")))

    def test_neither_a_key_nor_a_pool_is_refused(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            create_chat_model(MODEL_ID)

    def test_a_pool_of_another_provider_is_refused(self) -> None:
        pool = KeyPool("openai", "openai-pool", [spec("a", secret="sk-proj-0123456789abcdef")])
        with pytest.raises(ValueError, match="openai keys"):
            create_chat_model(MODEL_ID, key_pool=pool)

    @pytest.mark.parametrize(
        ("input_price", "output_price"),
        [(None, None), (Decimal("0.000003"), None), (None, Decimal("0.000015"))],
        ids=["both-missing", "output-missing", "input-missing"],
    )
    def test_a_model_without_both_prices_is_refused_at_build(
        self, input_price: Decimal | None, output_price: Decimal | None
    ) -> None:
        unpriced = f"pool-test-unpriced-{input_price is None}-{output_price is None}"
        register_capabilities(
            unpriced,
            ModelCapabilities(
                model_name=unpriced,
                model_type=ModelType.CHAT,
                model_tier=ModelTier.SMALL,
                provider_name="anthropic",
                cost_per_input_token=input_price,
                cost_per_output_token=output_price,
            ),
        )
        with pytest.raises(UnpricedModelError):
            create_chat_model(unpriced, key_pool=make_pool(spec("a")))

    def test_an_unregistered_model_is_refused_at_build(self) -> None:
        with pytest.raises(UnpricedModelError):
            create_chat_model("pool-test-unregistered", provider="anthropic", key_pool=make_pool(spec("a")))

    def test_a_breaker_setting_is_refused(self) -> None:
        with pytest.raises(ValueError, match="breaker_registry"):
            create_chat_model(MODEL_ID, key_pool=make_pool(spec("a")), breaker_registry=CircuitBreakerRegistry())

    def test_the_retry_mode_is_the_pool_s(self) -> None:
        with pytest.raises(ValueError, match="unretried_key_errors"):
            create_chat_model(MODEL_ID, key_pool=make_pool(spec("a")), unretried_key_errors=False)

    def test_its_repr_names_no_secret(self) -> None:
        model = create_chat_model(MODEL_ID, key_pool=make_pool(spec("a")))
        assert secret_for("a") not in repr(model)
