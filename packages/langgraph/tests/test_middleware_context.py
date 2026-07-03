"""Behavioral tests for :class:`ContextMergeMiddleware`.

Verifies the merge through the real ``awrap_model_call`` seam: the accumulated
context is folded into ``request.system_message`` (never appended as a second
message), the seam is a byte-for-byte no-op with no manager / empty context, and a
manager that raises soft-fails to the un-merged request.

The context manager is read off ``config["configurable"]`` via
:func:`langgraph.config.get_config` (the ``Runtime`` on a ``ModelRequest`` does not
carry ``config`` for the model-call seam), so tests drive it by setting the
runnable-config contextvar rather than stubbing ``runtime.config``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import var_child_runnable_config

from threetears.langgraph.middleware_context import (
    ContextMergeMiddleware,
    _fold_context_into_system,
)


class _Manager:
    """context provider returning a fixed context block."""

    def __init__(self, text: str) -> None:
        self._text = text

    def build_conversation_context(self) -> str:
        return self._text


class _RaisingManager:
    """context provider whose render raises (exercises the soft-fail path)."""

    def build_conversation_context(self) -> str:
        raise RuntimeError("boom")


@contextmanager
def _with_configurable(configurable: dict[str, Any]) -> Iterator[None]:
    """Set the runnable-config contextvar so the middleware's ``get_config`` sees it."""
    token = var_child_runnable_config.set({"configurable": configurable})
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


def _request(system: SystemMessage | None) -> ModelRequest:
    return ModelRequest(
        model=cast("BaseChatModel", SimpleNamespace()),
        messages=[HumanMessage(content="hi")],
        system_message=system,
    )


def _drive(request: ModelRequest, configurable: dict[str, Any]) -> ModelRequest:
    captured: dict[str, ModelRequest] = {}

    async def _handler(req: ModelRequest) -> Any:
        captured["req"] = req
        return SimpleNamespace(result=[AIMessage(content="ok")])

    async def _run() -> None:
        with _with_configurable(configurable):
            await ContextMergeMiddleware().awrap_model_call(request, _handler)

    asyncio.run(_run())
    return captured["req"]


class TestFold:
    def test_appends_to_string_system(self) -> None:
        out = _fold_context_into_system(SystemMessage(content="base"), "ctx")
        assert out.content == "base\n\nctx"

    def test_none_system_becomes_context(self) -> None:
        out = _fold_context_into_system(None, "ctx")
        assert out.content == "ctx"

    def test_appends_text_part_to_structured_content(self) -> None:
        base = SystemMessage(
            content=[{"type": "text", "text": "base", "cache_control": {"type": "ephemeral"}}],
        )
        out = _fold_context_into_system(base, "ctx")
        assert isinstance(out.content, list)
        assert out.content[-1] == {"type": "text", "text": "ctx"}
        # the pre-existing cache_control part is preserved untouched
        assert out.content[0]["cache_control"] == {"type": "ephemeral"}


class TestMiddleware:
    def test_folds_manager_context_into_system(self) -> None:
        seen = _drive(_request(SystemMessage(content="base")), {"context_manager": _Manager("CTX")})
        assert seen.system_message is not None
        assert seen.system_message.content == "base\n\nCTX"

    def test_noop_without_manager(self) -> None:
        req_in = _request(SystemMessage(content="base"))
        assert _drive(req_in, {}) is req_in  # same object -> byte-for-byte no-op

    def test_noop_on_empty_context(self) -> None:
        req_in = _request(SystemMessage(content="base"))
        assert _drive(req_in, {"context_manager": _Manager("")}) is req_in

    def test_soft_fails_on_manager_error(self) -> None:
        req_in = _request(SystemMessage(content="base"))
        assert _drive(req_in, {"context_manager": _RaisingManager()}) is req_in  # un-merged

    def test_no_system_message_uses_context(self) -> None:
        seen = _drive(_request(None), {"context_manager": _Manager("CTX")})
        assert seen.system_message is not None
        assert seen.system_message.content == "CTX"


class TestShape:
    def test_is_agent_middleware(self) -> None:
        m = ContextMergeMiddleware()
        assert isinstance(m, AgentMiddleware)
        assert m.name == "ContextMergeMiddleware"
        assert hasattr(m, "awrap_model_call")
        assert hasattr(m, "wrap_model_call")
