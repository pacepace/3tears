"""Tests for :mod:`threetears.models.tool_name_translation`'s ``NameMangledToolProxy``.

Scoped to the ``canonical_name`` public accessor added alongside the claude-cli dotted-tool-name
permission fix (see ``test_claude_cli_tool_events.py``), and to the ``config``-propagation bug
found live the same day: every REAL 3tears builtin tool (a ``StructuredTool`` built by
:func:`~threetears.agent.tools.langchain_adapter.to_langchain_tool`, whose own ``_arun``/``_run``
REQUIRE a ``RunnableConfig``) raised ``TypeError: ... missing 1 required keyword-only argument:
'config'`` the instant it was actually invoked through this proxy -- on ANY provider that uses it,
not just one. See the module docstring's "config propagation bug" section for the full root cause.
The module's other primitives are already exercised indirectly via the ``anthropic``/``openrouter``
provider test files.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool, ToolException

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.builtin.calculator import create_calculator_tool
from threetears.agent.tools.langchain_adapter import to_langchain_tool
from threetears.models.tool_name_translation import NameMangledToolProxy, build_name_translation, mangle_tool_name


class _DottedTool(BaseTool):
    name: str = "threetears.web_search"
    description: str = "search the web"

    def _run(self, **kwargs: Any) -> str:
        return "r"

    async def _arun(self, **kwargs: Any) -> str:
        return "r"


class TestCanonicalNameAccessor:
    def test_returns_the_delegates_original_dotted_name(self) -> None:
        delegate = _DottedTool()
        proxy = NameMangledToolProxy(delegate=delegate, mangled_name=mangle_tool_name(delegate.name))

        assert proxy.name == "threetears_web_search"
        assert proxy.canonical_name == "threetears.web_search"


class TestConfigPropagationToDelegate:
    """Live-observed bug: a real 3tears builtin tool (a ``StructuredTool``, whose ``_arun``/``_run``
    require ``config: RunnableConfig`` with no default) crashed with a ``TypeError`` on every real
    invocation through this proxy -- LangChain's ``BaseTool.arun``/``run`` only forward ``config``
    to a method whose OWN signature declares a ``RunnableConfig``-typed parameter, and the proxy's
    ``_arun``/``_run`` never declared one, so it never received a ``config`` to forward to the
    delegate it calls directly."""

    async def test_real_builtin_tool_executes_successfully_through_the_proxy(self) -> None:
        """The exact live failure: threetears.calculator, proxied, actually invoked."""
        real_calculator = create_calculator_tool({}, "Evaluate a math expression.")
        [wire_tool], _reverse_map = build_name_translation([real_calculator])

        assert isinstance(wire_tool, NameMangledToolProxy)
        content = await wire_tool.ainvoke({"expression": "47 * 89"})
        assert content == "4183"
        assert content == await real_calculator.ainvoke({"expression": "47 * 89"})

    def test_real_builtin_tool_executes_successfully_through_the_proxy_sync(self) -> None:
        real_calculator = create_calculator_tool({}, "Evaluate a math expression.")
        [wire_tool], _reverse_map = build_name_translation([real_calculator])

        content = wire_tool.invoke({"expression": "47 * 89"})
        assert content == "4183"
        assert content == real_calculator.invoke({"expression": "47 * 89"})

    async def test_delegate_that_does_not_want_config_is_unaffected(self) -> None:
        """A plain custom BaseTool (like the tests above this class use) never declared a
        config param before this fix and must keep working identically -- the forwarding is
        conditional on the DELEGATE's own signature, never forced."""
        delegate = _DottedTool()
        proxy = NameMangledToolProxy(delegate=delegate, mangled_name=mangle_tool_name(delegate.name))

        result = await proxy.ainvoke({})
        assert result == "r"


def _plan(**kwargs: Any) -> tuple[str, dict[str, Any]]:
    return f"planned {sorted(kwargs)}", {"kwargs": kwargs}


def _refuse(**_: Any) -> str:
    raise ToolException("the plan was refused")


_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"shots": {"type": "array", "items": {"$ref": "#/$defs/Shot"}}},
    "required": ["shots"],
    "$defs": {"Shot": {"type": "object", "properties": {"prompt": {"type": "string"}}}},
}


class TestTheProxyIsTheSameToolUnderAnotherName:
    """A proxy changes the name the provider sees and nothing else. It copied only the name,
    description and a pydantic ``args_schema``, so a tool carrying its schema as a JSON Schema dict
    could not be proxied at all, a tool returning ``(content, artifact)`` came back through the proxy
    as a bare tuple, and a tool that handles its own ``ToolException`` raised through the proxy."""

    def _dict_schema_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=_plan,
            name="studio.plan",
            description="plans shots",
            args_schema=_PLAN_SCHEMA,
            response_format="content_and_artifact",
        )

    def test_a_json_schema_tool_is_proxied_with_its_schema_unchanged(self) -> None:
        [wire_tool], _reverse_map = build_name_translation([self._dict_schema_tool()])

        assert wire_tool.name == "studio_plan"
        assert wire_tool.tool_call_schema == self._dict_schema_tool().tool_call_schema

    async def test_a_tool_call_through_the_proxy_answers_as_the_tool_itself_does(self) -> None:
        tool = self._dict_schema_tool()
        [wire_tool], _reverse_map = build_name_translation([tool])
        call = {"type": "tool_call", "id": "c1", "name": "studio.plan", "args": {"shots": [{"prompt": "dawn"}]}}

        direct = await tool.ainvoke(call)
        proxied = await wire_tool.ainvoke(call)

        assert isinstance(proxied, ToolMessage)
        assert proxied.content == direct.content == "planned ['shots']"
        assert proxied.artifact == direct.artifact == {"kwargs": {"shots": [{"prompt": "dawn"}]}}

    async def test_a_tool_that_handles_its_own_errors_still_does_through_the_proxy(self) -> None:
        tool = StructuredTool.from_function(
            func=_refuse, name="studio.refuse", description="refuses", handle_tool_error=True
        )
        [wire_tool], _reverse_map = build_name_translation([tool])

        message = await wire_tool.ainvoke({"type": "tool_call", "id": "c2", "name": "studio.refuse", "args": {}})

        assert isinstance(message, ToolMessage)
        assert message.status == "error"
        assert message.content == "the plan was refused"


class _Refusing(TearsTool):
    """a TearsTool that fails with a typed record in its metadata."""

    async def execute(self, **kwargs: Any) -> ToolResult:
        """fail, naming why in prose and in structure.

        :param kwargs: ignored
        :ptype kwargs: Any
        :return: the failure
        :rtype: ToolResult
        """
        return ToolResult(success=False, content="", error="the upstream refused", metadata={"failure": "upstream"})

    def mcp_schema(self) -> MCPToolDefinition:
        """the tool's definition.

        :return: an empty-object schema
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name="studio.refusing",
            version="1.0",
            description="refuses",
            input_schema={"type": "object", "properties": {}},
        )

    def mcp_name(self) -> str:
        """the canonical dotted name.

        :return: the name
        :rtype: str
        """
        return "studio.refusing"

    def mcp_version(self) -> str:
        """the version.

        :return: the version
        :rtype: str
        """
        return "1.0"


class _GatedRefusing(_Refusing):
    """the same tool, gated behind a person's approval."""

    requires_confirmation = True


class TestAProxiedTearsToolKeepsWhatItCarries:
    """the proxy is the tool under another name: its confirmation gate and a failure's artifact
    must come through it as they come from the tool."""

    def test_the_confirmation_gate_survives_the_proxy(self) -> None:
        """a gate reading the bound tool list sees the flag on the proxy.

        :return: none
        :rtype: None
        """
        [gated], _ = build_name_translation([to_langchain_tool(_GatedRefusing())])
        [ungated], _ = build_name_translation([to_langchain_tool(_Refusing())])
        assert getattr(gated, "requires_confirmation", False) is True
        assert getattr(ungated, "requires_confirmation", None) is False
