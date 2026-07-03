"""Behavioral tests for :class:`MemoryInjectionMiddleware`.

Verifies the ``awrap_model_call`` seam: retrieved memories are FOLDED into
``request.system_message`` under :data:`_MEMORY_CONTEXT_PREFIX` (never appended as a
second, non-consecutive ``SystemMessage`` -- the pattern that crashes LangChain's
Anthropic binding), the surfaced ids are deduped into the ``surfaced_memory_ids``
metadata ledger and persisted via a returned
:class:`~langchain.agents.middleware.types.ExtendedModelResponse` ``Command``; a
missing integration, a missing verified ``call_context``, an empty query, and no
surfaced memory all pass the call through un-merged; a retrieval fault soft-fails to a
pass-through; and the sync mirror passes through. A REAL ``create_agent`` + capturing
fake model regression proves the model receives exactly one leading ``SystemMessage``
(the crash guard the stub-model unit tests cannot see).

The integration + verified identity are read off ``config["configurable"]``; the
direct-drive tests set the runnable-config contextvar so the middleware's
``get_config`` sees them, and the real-agent test passes them via ``ainvoke``'s
``config``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid7

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables.config import var_child_runnable_config

from threetears.agent.memory.middleware import (
    _MEMORY_CONTEXT_PREFIX,
    _SURFACED_MEMORY_IDS_KEY,
    MemoryInjectionMiddleware,
    MemoryInjectionState,
)


class _StubRetriever:
    """opaque memory retriever standing in for the real one.

    exposes the duck-typed ``retrieve`` surface
    :func:`threetears.agent.memory.integration.retrieve_memories` calls; returns the
    configured ``context`` string (falsy -> no memory surfaced).
    """

    def __init__(self, context: str) -> None:
        self._context = context

    async def retrieve(
        self,
        user_id: Any,
        query: Any,
        *,
        agent_id: Any,
        customer_id: Any,
        caller_user_id: Any,
        caller_agent_id: Any,
    ) -> str:
        return self._context


class _StubIntegration:
    """opaque memory integration exposing the ``retriever`` the seam reads."""

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever
        self.extractor = None


class _StubIntegrationRaising:
    """integration whose ``retriever`` access raises (soft-fail path)."""

    @property
    def retriever(self) -> Any:
        raise RuntimeError("retriever boom")


def _call_context() -> Any:
    """build a verified per-call identity exposing agent/customer/user ids."""
    return SimpleNamespace(agent_id=uuid7(), customer_id=uuid7(), user_id=uuid7())


@contextmanager
def _configured(configurable: dict[str, Any]) -> Iterator[None]:
    """set the runnable-config contextvar so ``get_config`` sees ``configurable``."""
    token = var_child_runnable_config.set({"configurable": configurable})
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


def _configurable(context: str = "user prefers dark mode") -> dict[str, Any]:
    """build a configurable with a memory integration + verified call_context."""
    integration = _StubIntegration(_StubRetriever(context))
    return {"memory_integration": integration, "call_context": _call_context()}


def _request(
    system: SystemMessage | None,
    *,
    messages: list[BaseMessage] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ModelRequest:
    """build a minimal ``ModelRequest`` for the direct-drive tests.

    ``metadata`` seeds ``request.state["metadata"]`` so the cross-turn dedup path can be
    exercised (the middleware unions newly surfaced ids over the ids already on state).
    """
    return ModelRequest(
        model=cast("BaseChatModel", SimpleNamespace()),
        messages=messages if messages is not None else [HumanMessage(content="hi")],
        system_message=system,
        state=cast("Any", {"messages": [], "metadata": metadata or {}}),
    )


def _drive(
    mw: MemoryInjectionMiddleware,
    request: ModelRequest,
    configurable: dict[str, Any],
) -> tuple[ModelRequest, Any]:
    """drive ``awrap_model_call``; return the request the handler saw + the result."""
    captured: dict[str, ModelRequest] = {}

    async def _handler(req: ModelRequest) -> Any:
        captured["req"] = req
        return SimpleNamespace(result=[AIMessage(content="ok")])

    async def _run() -> Any:
        with _configured(configurable):
            return await mw.awrap_model_call(request, _handler)

    out = asyncio.run(_run())
    return captured["req"], out


class TestInjection:
    def test_folds_memory_into_system_message(self) -> None:
        req, _out = _drive(
            MemoryInjectionMiddleware(),
            _request(SystemMessage(content="base")),
            _configurable("user prefers dark mode"),
        )
        assert req.system_message is not None
        content = req.system_message.content
        assert isinstance(content, str)
        # folded into the SINGLE system message, base first then the memory block.
        assert content.startswith("base\n\n" + _MEMORY_CONTEXT_PREFIX)
        assert "- user prefers dark mode" in content

    def test_returns_extended_response_with_surfaced_ids(self) -> None:
        _req, out = _drive(
            MemoryInjectionMiddleware(),
            _request(SystemMessage(content="base")),
            _configurable("user prefers dark mode"),
        )
        assert isinstance(out, ExtendedModelResponse)
        assert out.command is not None
        update = out.command.update
        assert isinstance(update, dict)
        assert update["metadata"][_SURFACED_MEMORY_IDS_KEY] == ["user prefers dark mode"]

    def test_dedupes_already_surfaced_memory(self) -> None:
        # the memory was already surfaced on a prior turn (on request.state metadata) ->
        # the union keeps the ledger deduped.
        req = _request(
            SystemMessage(content="base"),
            metadata={_SURFACED_MEMORY_IDS_KEY: ["user prefers dark mode"]},
        )
        _req, out = _drive(MemoryInjectionMiddleware(), req, _configurable("user prefers dark mode"))
        assert isinstance(out, ExtendedModelResponse)
        assert out.command is not None
        assert out.command.update["metadata"][_SURFACED_MEMORY_IDS_KEY] == ["user prefers dark mode"]

    def test_no_base_system_message_becomes_block(self) -> None:
        req, out = _drive(MemoryInjectionMiddleware(), _request(None), _configurable("user prefers dark mode"))
        assert isinstance(out, ExtendedModelResponse)
        assert req.system_message is not None
        assert isinstance(req.system_message.content, str)
        assert req.system_message.content.startswith(_MEMORY_CONTEXT_PREFIX)


class TestNoop:
    def test_noop_without_integration(self) -> None:
        req_in = _request(SystemMessage(content="base"))
        req_out, out = _drive(MemoryInjectionMiddleware(), req_in, {})
        assert req_out is req_in
        assert not isinstance(out, ExtendedModelResponse)

    def test_noop_without_call_context(self) -> None:
        # integration present but no verified identity -> pass-through.
        integration = _StubIntegration(_StubRetriever("user prefers dark mode"))
        req_in = _request(SystemMessage(content="base"))
        req_out, out = _drive(MemoryInjectionMiddleware(), req_in, {"memory_integration": integration})
        assert req_out is req_in
        assert not isinstance(out, ExtendedModelResponse)

    def test_noop_when_no_human_message(self) -> None:
        # no HumanMessage -> empty query -> retrieval skipped -> pass-through.
        req_in = _request(SystemMessage(content="base"), messages=[AIMessage(content="prior assistant turn")])
        req_out, out = _drive(MemoryInjectionMiddleware(), req_in, _configurable("user prefers dark mode"))
        assert req_out is req_in
        assert not isinstance(out, ExtendedModelResponse)

    def test_noop_when_no_memory_surfaced(self) -> None:
        # retriever returns an empty context -> no memory -> pass-through.
        req_in = _request(SystemMessage(content="base"))
        req_out, out = _drive(MemoryInjectionMiddleware(), req_in, _configurable(""))
        assert req_out is req_in
        assert not isinstance(out, ExtendedModelResponse)

    def test_noop_outside_runnable_context(self) -> None:
        # no contextvar set -> get_config raises -> soft-fail to pass-through.
        mw = MemoryInjectionMiddleware()
        req = _request(SystemMessage(content="base"))
        captured: dict[str, ModelRequest] = {}

        async def _handler(r: ModelRequest) -> Any:
            captured["req"] = r
            return SimpleNamespace(result=[])

        out = asyncio.run(mw.awrap_model_call(req, _handler))
        assert captured["req"] is req
        assert not isinstance(out, ExtendedModelResponse)


class TestSoftFail:
    def test_soft_fail_on_retrieval_error(self) -> None:
        # the integration faults while retrieving -> the seam swallows it and proceeds
        # on the un-merged request rather than failing the turn.
        configurable = {"memory_integration": _StubIntegrationRaising(), "call_context": _call_context()}
        req_in = _request(SystemMessage(content="base"))
        req_out, out = _drive(MemoryInjectionMiddleware(), req_in, configurable)
        assert req_out is req_in
        assert not isinstance(out, ExtendedModelResponse)


class TestSyncMirror:
    def test_sync_wrap_model_call_passes_through(self) -> None:
        mw = MemoryInjectionMiddleware()
        req = _request(SystemMessage(content="base"))
        captured: dict[str, ModelRequest] = {}

        def _handler(r: ModelRequest) -> Any:
            captured["req"] = r
            return SimpleNamespace(result=[])

        with _configured(_configurable("user prefers dark mode")):
            out = mw.wrap_model_call(req, _handler)
        assert captured["req"] is req  # un-merged
        assert not isinstance(out, ExtendedModelResponse)


class TestShape:
    def test_is_agent_middleware(self) -> None:
        m = MemoryInjectionMiddleware()
        assert isinstance(m, AgentMiddleware)
        assert m.name == "MemoryInjectionMiddleware"
        assert hasattr(m, "awrap_model_call")
        assert hasattr(m, "wrap_model_call")

    def test_state_schema_declares_metadata_channel(self) -> None:
        # the surfaced-ids ledger only persists if the state schema declares the channel.
        assert MemoryInjectionMiddleware().state_schema is MemoryInjectionState
        assert "metadata" in MemoryInjectionState.__annotations__


# --------------------------------------------------------------------------- #
# real create_agent regression: the crash guard the stub-model tests can't see
# --------------------------------------------------------------------------- #
class _CapturingChatModel(BaseChatModel):
    """fake chat model that records the messages every model call receives.

    Drives a REAL ``create_agent`` so the prompt-assembly path
    (``[request.system_message, *messages]``) is exercised end to end -- the path
    the direct-drive stub tests never reach.

    :ivar sink: accumulates one message-list snapshot per model call.
    """

    sink: list[list[BaseMessage]]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Record the received messages and return a fixed no-tool-call reply.

        :param messages: the assembled prompt for this model call.
        :ptype messages: list[BaseMessage]
        :param stop: stop sequences (unused).
        :ptype stop: list[str] | None
        :param run_manager: callback manager (unused).
        :ptype run_manager: Any
        :return: a single-generation result ending the agent loop.
        :rtype: ChatResult
        """
        self.sink.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    @property
    def _llm_type(self) -> str:
        """Return the model type discriminator.

        :return: fixed type string.
        :rtype: str
        """
        return "capturing-fake"


class TestRealAgentRegression:
    def test_model_receives_exactly_one_leading_system_message(self) -> None:
        model = _CapturingChatModel(sink=[])
        agent = create_agent(
            model=model,
            tools=[],
            system_prompt="base prompt",
            middleware=[MemoryInjectionMiddleware()],
            state_schema=MemoryInjectionState,
        )
        final = asyncio.run(
            agent.ainvoke(
                {"messages": [HumanMessage(content="who are the active users")]},
                config={"configurable": _configurable("user prefers dark mode")},
            )
        )
        # the model was called; inspect the FIRST call's assembled prompt.
        assert model.sink, "the model was never called"
        first_call = model.sink[0]
        systems = [m for m in first_call if isinstance(m, SystemMessage)]
        # EXACTLY ONE system message, and it is FIRST -> no non-consecutive system
        # message (which langchain_anthropic would reject).
        assert len(systems) == 1
        assert isinstance(first_call[0], SystemMessage)
        assert "Relevant memories about this user" in str(systems[0].content)
        assert "base prompt" in str(systems[0].content)
        # the ledger survived onto the metadata channel via the Command update.
        assert final.get("metadata", {}).get(_SURFACED_MEMORY_IDS_KEY) == ["user prefers dark mode"]
