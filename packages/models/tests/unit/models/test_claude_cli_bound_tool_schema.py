"""What a subscription model's bound tool looks like to the CLI, and that its handler runs nothing.

The model hands every tool call back to the caller (see ``_claude_cli``'s module docstring), so the
handler the CLI calls between the model's tool use and its one-turn limit must not run the tool.
What still matters on the CLI side is what the model is SHOWN: the tool's wire name and schema.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import BaseTool

pytest.importorskip("langchain_claude_code")
pytest.importorskip("claude_agent_sdk")

from threetears.models import DEFAULT_CHAT_MODEL
from threetears.models.providers import _claude_cli as claude_cli
from threetears.models.providers._claude_cli import create_subscription_chat


class _EchoTool(BaseTool):
    """A trivial :class:`BaseTool` that returns its args, for tool-event tests."""

    name: str = "echo"
    description: str = "echoes its input"

    def _run(self, **kwargs: Any) -> str:
        return f"echo:{kwargs}"

    async def _arun(self, **kwargs: Any) -> str:
        return f"echo:{kwargs}"


class _DottedNameTool(BaseTool):
    """A :class:`BaseTool` with a canonical dotted 3tears name, like every real builtin
    (``threetears.web_search``, ``threetears.calculator``, ...) -- see
    :meth:`threetears.agent.tools.base_tool.BaseAgentTool.mcp_name`."""

    name: str = "threetears.web_search"
    description: str = "search the web"

    def _run(self, **kwargs: Any) -> str:
        return f"result:{kwargs}"

    async def _arun(self, **kwargs: Any) -> str:
        return f"result:{kwargs}"


def _wrap(tool: BaseTool):
    """Build the subscription model and wrap ``tool``, returning the raw ``SdkMcpTool``."""
    model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
    schema = {"properties": {"x": {"type": "string"}}, "required": []}
    return model._wrap_langchain_tool(tool, schema)  # noqa: SLF001 -- the method under test  # type: ignore[attr-defined]


class _MustNotRunTool(BaseTool):
    """A tool whose running is the failure."""

    name: str = "threetears.send_email"
    description: str = "sends an email"

    def _run(self, **kwargs: Any) -> str:
        raise AssertionError("the CLI's handler ran the tool; tool calls belong to the caller")

    async def _arun(self, **kwargs: Any) -> str:
        raise AssertionError("the CLI's handler ran the tool; tool calls belong to the caller")


def _wrapped_handler(tool: BaseTool):
    """Build the subscription model, wrap ``tool``, and return the SDK tool's raw async handler."""
    return _wrap(tool).handler


class TestTheHandlerRunsNothing:
    async def test_the_handler_answers_without_running_the_tool(self) -> None:
        result = await _wrapped_handler(_MustNotRunTool())({"to": "someone"})
        assert result == {"content": [{"type": "text", "text": claude_cli._HANDED_BACK}]}  # noqa: SLF001

    async def test_a_dotted_tool_is_registered_under_its_wire_name_and_still_not_run(self) -> None:
        from threetears.models.tool_name_translation import build_name_translation

        [wire_tool], _reverse_map = build_name_translation([_MustNotRunTool()])
        sdk_tool = _wrap(wire_tool)
        assert sdk_tool.name == "threetears_send_email"
        assert "is_error" not in await sdk_tool.handler({})


class TestDottedToolNamePermissionFix:
    """Every 3tears builtin's canonical name is dotted (``threetears.web_search``, per
    ``BaseAgentTool.mcp_name()``). Live-observed bug: every real call to a bound dotted tool was
    silently DENIED under a subscription turn ("Claude requested permissions to use ... but you
    haven't granted it yet"), with nothing logged anywhere, while the same tool worked fine on
    every other backend. Root cause: the base class's own ``bind_tools`` already tries to
    auto-approve bound tools by deriving ``allowed_tools`` from each tool's raw (dotted) ``.name``,
    but the SDK/CLI normalizes dots out of tool identities on the wire, so that entry never matched
    -- the auto-approval attempt silently failed to match its own target. ``bind_tools`` here
    substitutes each dotted tool for a ``NameMangledToolProxy`` (the same translation
    ``anthropic.py``/``openrouter.py`` already apply for the identical Anthropic tool-name
    constraint) before the base class ever derives ``allowed_tools``, so the entry matches.
    """

    def test_allowed_tools_matches_the_underscored_wire_identity(self) -> None:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        bound = model.bind_tools([_DottedNameTool()])
        assert bound.bound.allowed_tools == ["mcp__langchain-tools__threetears_web_search"]  # type: ignore[attr-defined]

    def test_dotless_tool_name_is_unaffected(self) -> None:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        bound = model.bind_tools([_EchoTool()])
        assert bound.bound.allowed_tools == ["mcp__langchain-tools__echo"]  # type: ignore[attr-defined]

    def test_permission_mode_is_left_at_the_default(self) -> None:
        """The fix is a precise allowed_tools match, not a blanket permission bypass."""
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        bound = model.bind_tools([_DottedNameTool()])
        assert bound.bound.permission_mode == "default"  # type: ignore[attr-defined]


class TestOptionalParameterSchemaFix:
    """Live-observed bug, two compounding root causes on the same bound tool
    (``memory_search``'s real schema, lifted from
    ``MemorySearchInput.model_json_schema()`` in ``threetears.agent.memory.tools``):

    1. A ``list``-typed Optional parameter (``ids: list[str] | None``) arrived at the handler as
       the literal string ``"[]"``, failing pydantic validation ("Input should be a valid list")
       every time the model tried to pass it. Pydantic renders an ``X | None`` field as
       ``anyOf: [{type: X}, {type: null}]`` with no top-level ``type`` key, and the naive
       ``prop.get("type", "string")`` conversion used to default every such field to ``string``.
    2. Every OTHER optional filter (``alias``, ...) arrived populated with an empty string on
       every call instead of being omitted. ``claude_agent_sdk.create_sdk_mcp_server``'s own
       schema builder marks every key in a bare ``{name: type}`` map as ``required`` -- forcing
       the model to invent a value for filters it had nothing to fill in. The fix now hands the
       SDK a full ``{"type": "object", "properties": ..., "required": [...]}`` schema so its
       required-everything fallback never triggers.
    """

    _MEMORY_SEARCH_LIKE_SCHEMA = {
        "properties": {
            "query": {"type": "string"},
            "ids": {
                "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}],
                "default": None,
            },
            "limit": {"type": "integer", "default": 10},
            "alias": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        },
        "required": ["query"],
    }

    def test_optional_list_parameter_resolves_to_array_not_string(self) -> None:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        sdk_tool = model._wrap_langchain_tool(  # noqa: SLF001 -- the method under test  # type: ignore[attr-defined]
            _EchoTool(), self._MEMORY_SEARCH_LIKE_SCHEMA
        )
        assert sdk_tool.input_schema["properties"]["ids"]["type"] == "array"

    def test_optional_string_parameter_still_resolves_to_string(self) -> None:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        sdk_tool = model._wrap_langchain_tool(  # noqa: SLF001 -- the method under test  # type: ignore[attr-defined]
            _EchoTool(), self._MEMORY_SEARCH_LIKE_SCHEMA
        )
        assert sdk_tool.input_schema["properties"]["alias"]["type"] == "string"

    def test_plain_typed_parameters_are_unaffected(self) -> None:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        sdk_tool = model._wrap_langchain_tool(  # noqa: SLF001 -- the method under test  # type: ignore[attr-defined]
            _EchoTool(), self._MEMORY_SEARCH_LIKE_SCHEMA
        )
        assert sdk_tool.input_schema["properties"]["query"]["type"] == "string"
        assert sdk_tool.input_schema["properties"]["limit"]["type"] == "integer"

    def test_only_the_genuinely_required_field_is_marked_required(self) -> None:
        """The SDK's own required-everything fallback must never trigger: ``ids``/``limit``/
        ``alias`` all have defaults in the source schema and must NOT appear in ``required``,
        even though they're present in ``properties``."""
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        sdk_tool = model._wrap_langchain_tool(  # noqa: SLF001 -- the method under test  # type: ignore[attr-defined]
            _EchoTool(), self._MEMORY_SEARCH_LIKE_SCHEMA
        )
        assert sdk_tool.input_schema["required"] == ["query"]

    async def test_final_advertised_schema_survives_create_sdk_mcp_server_unmodified(self) -> None:
        """End-to-end: create_sdk_mcp_server's own schema builder must pass our full schema
        through VERBATIM (its required-everything fallback only fires for a bare {name: type}
        map) -- this is what the model actually sees when a turn starts."""
        from claude_agent_sdk import create_sdk_mcp_server
        from mcp.types import ListToolsRequest

        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        sdk_tool = model._wrap_langchain_tool(  # noqa: SLF001 -- the method under test  # type: ignore[attr-defined]
            _EchoTool(), self._MEMORY_SEARCH_LIKE_SCHEMA
        )
        server = create_sdk_mcp_server("test", tools=[sdk_tool])
        handler = server["instance"].request_handlers[ListToolsRequest]
        result = await handler(ListToolsRequest(method="tools/list"))

        advertised = result.root.tools[0].inputSchema
        assert advertised["required"] == ["query"]
        assert advertised["properties"]["ids"]["type"] == "array"
