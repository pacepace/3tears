"""Usage that knows who it belongs to.

``UsageRecord`` has always had tenant fields -- customer, user, conversation, agent -- and a sink
protocol to store it, but nothing could fill them: ``UsageTrackingCallback`` built every record from
the model name alone, and ``create_chat_model`` gave each model a fresh tracker with no sinks. So
every multi-tenant consumer (metallm, scriob) wrote a second metering path beside it. These tests pin
what closes that gap:

- a ``usage_scope`` around the calls it covers, or run metadata, attributes each record;
- a record says whether its tokens were reported, estimated or unavailable;
- ``UsageAccumulator`` totals one run for per-turn metering;
- ``attach_callbacks`` adds handlers to a factory-built model;
- ``set_default_usage_tracker`` points every factory-built model at the consumer's sinks.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from threetears.models import (
    LlmPurpose,
    UsageAccumulator,
    UsageRecord,
    UsageTracker,
    attach_callbacks,
    create_chat_model,
    current_usage_scope,
    set_default_usage_tracker,
    usage_scope,
)
from threetears.models.tracking import UsageAuditSink


class _Sink(UsageAuditSink):
    def __init__(self) -> None:
        self.records: list[UsageRecord] = []

    async def record(self, record: UsageRecord) -> None:
        self.records.append(record)


def _reply(text: str = "hello", *, usage: dict[str, Any] | None = None) -> AIMessage:
    return AIMessage(content=text, usage_metadata=usage)  # type: ignore[arg-type]


_USAGE = {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17}


def _model(sink: _Sink, *replies: AIMessage) -> tuple[Any, UsageTracker]:
    tracker = UsageTracker(audit_sink=sink)
    callback = tracker.make_callback(model_name="fake", provider_name="fake")
    model = GenericFakeChatModel(messages=iter(replies)).with_config(callbacks=[callback])
    return model, tracker


async def _settle() -> None:
    """The tracker hands records to its sinks on background tasks."""
    for _ in range(5):
        await asyncio.sleep(0)


async def test_a_usage_scope_attributes_the_calls_inside_it() -> None:
    sink = _Sink()
    model, _ = _model(sink, _reply(usage=_USAGE))
    customer, user, conversation = uuid4(), uuid4(), uuid4()
    with usage_scope(customer_id=customer, user_id=user, conversation_id=conversation, category="chat.turn"):
        await model.ainvoke([HumanMessage(content="hi")])
    await _settle()
    [record] = sink.records
    assert (record.customer_id, record.user_id, record.conversation_id) == (customer, user, conversation)
    assert record.category == "chat.turn"


async def test_outside_any_scope_the_record_is_unattributed() -> None:
    sink = _Sink()
    model, _ = _model(sink, _reply(usage=_USAGE))
    await model.ainvoke([HumanMessage(content="hi")])
    await _settle()
    assert sink.records[0].customer_id is None


async def test_run_metadata_attributes_and_wins_over_the_scope() -> None:
    sink = _Sink()
    model, _ = _model(sink, _reply(usage=_USAGE))
    scoped, direct = uuid4(), uuid4()
    with usage_scope(user_id=scoped, customer_id=scoped):
        await model.ainvoke(
            [HumanMessage(content="hi")],
            config={"metadata": {"threetears.usage.user_id": str(direct), "threetears.usage.purpose": "summarization"}},
        )
    await _settle()
    [record] = sink.records
    assert record.user_id == direct, "metadata names the payer for this one call"
    assert record.customer_id == scoped, "the scope still fills what the metadata does not"
    assert record.purpose is LlmPurpose.SUMMARIZATION


def test_scopes_nest_and_restore() -> None:
    outer, inner = uuid4(), uuid4()
    with usage_scope(customer_id=outer, user_id=outer):
        with usage_scope(user_id=inner):
            assert current_usage_scope() == {"customer_id": outer, "user_id": inner}
        assert current_usage_scope() == {"customer_id": outer, "user_id": outer}
    assert current_usage_scope() == {}


def test_an_unknown_scope_field_is_refused() -> None:
    with pytest.raises(TypeError, match="customer"):
        with usage_scope(customer=uuid4()):  # type: ignore[call-arg]
            pass


async def test_reported_usage_is_marked_reported() -> None:
    sink = _Sink()
    model, _ = _model(sink, _reply(usage=_USAGE))
    await model.ainvoke([HumanMessage(content="hi")])
    await _settle()
    record = sink.records[0]
    assert (record.input_tokens, record.output_tokens, record.token_source) == (12, 5, "reported")


async def test_missing_usage_is_estimated_rather_than_recorded_as_zero() -> None:
    sink = _Sink()
    model, _ = _model(sink, _reply("a reply long enough to count several tokens"))
    await model.ainvoke([HumanMessage(content="hi")])
    await _settle()
    record = sink.records[0]
    assert record.token_source == "estimated"
    assert record.output_tokens > 0


def _result(*messages: AIMessage) -> LLMResult:
    return LLMResult(generations=[[ChatGeneration(message=m) for m in messages]])


def test_every_generation_is_counted_and_cache_tokens_are_read() -> None:
    sink = _Sink()
    tracker = UsageTracker(audit_sink=sink)
    recorded: list[UsageRecord] = []
    tracker.record = recorded.append  # type: ignore[method-assign]
    callback = tracker.make_callback(model_name="fake", provider_name="fake")
    cached = {**_USAGE, "input_token_details": {"cache_read": 4, "cache_creation": 2}}
    callback.on_llm_end(_result(_reply(usage=cached), _reply(usage=_USAGE)), run_id=uuid4())
    [record] = recorded
    assert (record.input_tokens, record.output_tokens) == (24, 10)
    assert (record.cache_read_tokens, record.cache_creation_tokens) == (4, 2)


def test_no_usage_and_no_text_is_unavailable() -> None:
    tracker = UsageTracker()
    recorded: list[UsageRecord] = []
    tracker.record = recorded.append  # type: ignore[method-assign]
    tracker.make_callback(model_name="fake", provider_name="fake").on_llm_end(_result(_reply("")), run_id=uuid4())
    assert recorded[0].token_source == "unavailable"
    assert (recorded[0].input_tokens, recorded[0].output_tokens) == (0, 0)


async def test_the_accumulator_totals_a_run() -> None:
    accumulator = UsageAccumulator(cost_per_input_token=Decimal("0.001"), cost_per_output_token=Decimal("0.002"))
    model = GenericFakeChatModel(messages=iter([_reply(usage=_USAGE), _reply("no usage reported")]))
    bound = attach_callbacks(model, accumulator)
    await bound.ainvoke([HumanMessage(content="one")])
    await bound.ainvoke([HumanMessage(content="two")])
    assert accumulator.input_tokens > 12, "the second call reported nothing, so its input is estimated from its prompt"
    assert accumulator.output_tokens > 5
    assert accumulator.calls == 2
    assert accumulator.token_source == "estimated", "one estimated call makes the total an estimate"
    assert isinstance(accumulator.cost_usd, Decimal)
    assert (
        accumulator.cost_usd
        == Decimal("0.001") * accumulator.input_tokens + Decimal("0.002") * accumulator.output_tokens
    )


class _Heard(BaseCallbackHandler):
    def __init__(self) -> None:
        self.ends = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.ends += 1


async def test_attach_callbacks_adds_to_a_factory_built_model_and_keeps_its_own() -> None:
    sink = _Sink()
    model, _ = _model(sink, _reply(usage=_USAGE))  # a RunnableBinding carrying the tracker callback
    heard = _Heard()
    await attach_callbacks(model, heard).ainvoke([HumanMessage(content="hi")])
    await _settle()
    assert heard.ends == 1
    assert len(sink.records) == 1, "the model's own tracker callback survived the attach"


def test_factory_models_use_the_default_tracker_unless_given_one() -> None:
    default = UsageTracker()
    explicit = UsageTracker()
    try:
        set_default_usage_tracker(default)
        built = create_chat_model("gpt-4o-mini", api_key="sk-test", provider="openai")
        chosen = create_chat_model("gpt-4o-mini", api_key="sk-test", provider="openai", tracker=explicit)
    finally:
        set_default_usage_tracker(None)
    assert _tracker_of(built) is default
    assert _tracker_of(chosen) is explicit


def _tracker_of(model: Any) -> UsageTracker:
    callbacks = model.config["callbacks"]
    [tracking] = [c for c in callbacks if type(c).__name__ == "UsageTrackingCallback"]
    return tracking._tracker  # type: ignore[no-any-return]  # noqa: SLF001


def test_uuid_metadata_is_parsed_and_junk_is_ignored() -> None:
    tracker = UsageTracker()
    recorded: list[UsageRecord] = []
    tracker.record = recorded.append  # type: ignore[method-assign]
    callback = tracker.make_callback(model_name="fake", provider_name="fake")
    run = uuid4()
    user = uuid4()
    callback.on_chat_model_start(
        {}, [[]], run_id=run, metadata={"threetears.usage.user_id": str(user), "threetears.usage.customer_id": "junk"}
    )
    callback.on_llm_end(_result(_reply(usage=_USAGE)), run_id=run)
    assert recorded[0].user_id == user
    assert recorded[0].customer_id is None
    assert isinstance(recorded[0].user_id, UUID)
