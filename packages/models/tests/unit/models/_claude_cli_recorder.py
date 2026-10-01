"""What a subscription chat model sends the Claude CLI, read at the CLI's door.

The subscription suites used to call the model's private helpers (``_build_options``,
``_convert_messages``, ``_wrap_langchain_tool``, ...) and assert on what they returned. These
helpers drive the model through its public entry points instead -- ``ainvoke`` on a model built
the ordinary way, ``bind_tools`` for tools -- and hand back what reached the CLI: the launch
options the SDK client was built with, the query text it was sent, and the in-process tool server
the CLI lists and calls tools on. A helper renamed, inlined or bypassed cannot fake a pass here;
only a change to what the CLI receives can fail one.

Only the Claude Agent SDK client is replaced. Every binding of it a call can reach is patched, and
a tripwire on the real class's ``__init__`` fails a test that would otherwise start a real, billed
CLI. Pooling is turned off for the duration, so a call takes the one-off client path; the pooled
path is read through :func:`pooled_launch`, which stands a recording pool in for the process-wide
one.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

from threetears.models import DEFAULT_CHAT_MODEL, claude_cli_pool
from threetears.models.claude_cli_pool import ClaudeCliPoolExhausted

__all__ = [
    "AFTER_CLI_IDENTITY",
    "TOKEN",
    "CliInput",
    "PooledLaunch",
    "advertised_tools",
    "call_tool",
    "pooled_launch",
    "recording_cli",
    "sent_to_cli",
    "subscription_model",
]

TOKEN = "sk-ant-oat01-faketokenfortest"

#: What the model puts before a caller's system prompt so it starts after the CLI's own identity
#: line (see ``test_claude_cli_api_parity``). Every system prompt the CLI receives begins with it.
AFTER_CLI_IDENTITY = "\n\n"


@dataclass
class CliInput:
    """One call as the CLI received it.

    :ivar options: the ``ClaudeAgentOptions`` the SDK client was built with
    :ivar query: the query text sent to the CLI
    """

    options: Any
    query: str

    @property
    def system_prompt(self) -> str | None:
        """the caller's system prompt as the CLI received it, without the identity separator.

        :return: the prompt, or ``None`` when the call had none
        :rtype: str | None
        :raises AssertionError: when a prompt arrived without the separator every one carries
        """
        prompt = self.options.system_prompt
        if prompt is None:
            return None
        assert prompt.startswith(AFTER_CLI_IDENTITY), f"the system prompt lost its separator: {prompt!r}"
        stripped: str = prompt[len(AFTER_CLI_IDENTITY) :]
        return stripped


# parity-exempt: records the options and query of one call and answers with one plain reply; a one-off call reaches nothing else
class _FakeSDKClient:
    """Stands in for ``claude_agent_sdk.ClaudeSDKClient`` on the one-off path: records, then answers."""

    received: list[CliInput] = []

    def __init__(self, options: Any = None) -> None:
        self._options = options

    async def __aenter__(self) -> _FakeSDKClient:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    async def query(self, prompt: str, session_id: str = "default") -> None:
        del session_id
        _FakeSDKClient.received.append(CliInput(options=self._options, query=prompt))

    async def receive_response(self) -> AsyncIterator[Any]:
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock  # noqa: PLC0415

        yield AssistantMessage(content=[TextBlock(text="ok")], model=DEFAULT_CHAT_MODEL)
        yield ResultMessage(
            subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=1, session_id="s"
        )


def _tripwire_init(self: Any, *_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("the REAL ClaudeSDKClient ran; a patch binding was missed")


@contextmanager
def recording_cli() -> Iterator[list[CliInput]]:
    """every SDK client binding patched to record what each call sends; pooling off meanwhile.

    A model must be BUILT inside the block as well as called there: the subscription model class
    captures ``ClaudeSDKClient`` when it is built.

    :return: the calls the CLI received, in order, filled in as they arrive
    :rtype: Iterator[list[CliInput]]
    """
    received: list[CliInput] = []
    _FakeSDKClient.received = received
    claude_cli_pool.configure_claude_cli_pool(enabled=False)
    try:
        with (
            patch("claude_agent_sdk.ClaudeSDKClient", _FakeSDKClient),
            patch("langchain_claude_code.claude_chat_model.ClaudeSDKClient", _FakeSDKClient),
            patch("claude_agent_sdk.client.ClaudeSDKClient.__init__", _tripwire_init),
        ):
            yield received
    finally:
        claude_cli_pool.configure_claude_cli_pool(enabled=True)
        _FakeSDKClient.received = []


def subscription_model(token: str = TOKEN, **model_kwargs: Any) -> BaseChatModel:
    """a subscription chat model built the way a caller builds one.

    :param token: the subscription token
    :ptype token: str
    :param model_kwargs: constructor keyword arguments
    :ptype model_kwargs: Any
    :return: the model
    :rtype: BaseChatModel
    """
    from threetears.models.providers._claude_cli import create_subscription_chat  # noqa: PLC0415

    return create_subscription_chat(DEFAULT_CHAT_MODEL, token, **model_kwargs)


def sent_to_cli(
    messages: list[BaseMessage],
    *,
    build: Callable[[], Runnable[Any, Any]] = subscription_model,
    **call_kwargs: Any,
) -> CliInput:
    """invoke a model once and return what the CLI received.

    :param messages: the conversation
    :ptype messages: list[BaseMessage]
    :param build: builds the model (or bound model) to invoke, inside the recording
    :ptype build: Callable[[], Runnable[Any, Any]]
    :param call_kwargs: per-call keyword arguments for ``ainvoke``
    :ptype call_kwargs: Any
    :return: the call as the CLI received it
    :rtype: CliInput
    """
    with recording_cli() as received:
        model = build()
        asyncio.run(model.ainvoke(messages, **call_kwargs))
    [call] = received
    return call


async def _list_tools(server: Any) -> list[Any]:
    """the tools an in-process MCP server lists to the CLI.

    :param server: the server instance
    :ptype server: Any
    :return: the listed tools
    :rtype: list[Any]
    """
    from mcp.types import ListToolsRequest  # noqa: PLC0415

    result = await server.request_handlers[ListToolsRequest](ListToolsRequest(method="tools/list"))
    tools: list[Any] = result.root.tools
    return tools


def _tool_server(options: Any) -> Any:
    """the in-process tool server a call's launch options carry.

    :param options: the call's ``ClaudeAgentOptions``
    :ptype options: Any
    :return: the server instance
    :rtype: Any
    """
    return options.mcp_servers["langchain-tools"]["instance"]


def advertised_tools(build: Callable[[], Runnable[Any, Any]]) -> dict[str, dict[str, Any]]:
    """the tools a bound model's CLI is shown, by name, each as its listed input schema.

    :param build: builds the bound model, inside the recording
    :ptype build: Callable[[], Runnable[Any, Any]]
    :return: listed tool name to input schema
    :rtype: dict[str, dict[str, Any]]
    """
    from langchain_core.messages import HumanMessage  # noqa: PLC0415

    call = sent_to_cli([HumanMessage(content="hi")], build=build)
    listed = asyncio.run(_list_tools(_tool_server(call.options)))
    return {tool.name: tool.inputSchema for tool in listed}


def call_tool(build: Callable[[], Runnable[Any, Any]], name: str, arguments: dict[str, Any]) -> Any:
    """call a bound model's tool the way the CLI does, through its in-process server.

    :param build: builds the bound model, inside the recording
    :ptype build: Callable[[], Runnable[Any, Any]]
    :param name: the tool's listed name
    :ptype name: str
    :param arguments: the arguments the model filled in
    :ptype arguments: dict[str, Any]
    :return: the MCP ``CallToolResult``
    :rtype: Any
    """
    from langchain_core.messages import HumanMessage  # noqa: PLC0415
    from mcp.types import CallToolRequest, CallToolRequestParams  # noqa: PLC0415

    call = sent_to_cli([HumanMessage(content="hi")], build=build)
    handler = _tool_server(call.options).request_handlers[CallToolRequest]
    request = CallToolRequest(method="tools/call", params=CallToolRequestParams(name=name, arguments=arguments))
    return asyncio.run(handler(request)).root


@dataclass
class PooledLaunch:
    """What a call asked the process-wide pool for.

    :ivar options: the launch options a pooled CLI for the call starts with
    :ivar fallback: the options the call's own CLI was then built with, when the pool refused it
    """

    options: Any
    fallback: Any


def pooled_launch(
    messages: list[BaseMessage],
    *,
    build: Callable[[], Runnable[Any, Any]] = subscription_model,
    **call_kwargs: Any,
) -> PooledLaunch:
    """invoke a model once against a pool that records the launch it is asked for, then refuses it.

    The refusal sends the call to a CLI of its own, as every exhausted pool does, so the options
    that call then runs with are read too.

    :param messages: the conversation
    :ptype messages: list[BaseMessage]
    :param build: builds the model to invoke, inside the recording
    :ptype build: Callable[[], Runnable[Any, Any]]
    :param call_kwargs: per-call keyword arguments for ``ainvoke``
    :ptype call_kwargs: Any
    :return: what the pool was asked for, and what the call then ran with
    :rtype: PooledLaunch
    """
    asked: list[Any] = []

    class _RefusingPool:
        @asynccontextmanager
        async def checkout(self, options: Any, **_kwargs: Any) -> AsyncIterator[Any]:
            asked.append(options)
            raise ClaudeCliPoolExhausted("every Claude CLI session is busy")
            yield  # pragma: no cover

    from threetears.models.providers import _claude_cli  # noqa: PLC0415

    with recording_cli() as received, patch.object(_claude_cli, "claude_cli_pool", lambda: _RefusingPool()):
        model = build()
        asyncio.run(model.ainvoke(messages, **call_kwargs))
    [launch] = asked
    [own] = received
    return PooledLaunch(options=launch, fallback=own.options)
