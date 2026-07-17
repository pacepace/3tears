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

from langchain_core.tools import BaseTool

from threetears.agent.tools.builtin.calculator import create_calculator_tool
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
        content, _artifact = await wire_tool.ainvoke({"expression": "47 * 89"})
        assert content == "4183"

    def test_real_builtin_tool_executes_successfully_through_the_proxy_sync(self) -> None:
        real_calculator = create_calculator_tool({}, "Evaluate a math expression.")
        [wire_tool], _reverse_map = build_name_translation([real_calculator])

        content, _artifact = wire_tool.invoke({"expression": "47 * 89"})
        assert content == "4183"

    async def test_delegate_that_does_not_want_config_is_unaffected(self) -> None:
        """A plain custom BaseTool (like the tests above this class use) never declared a
        config param before this fix and must keep working identically -- the forwarding is
        conditional on the DELEGATE's own signature, never forced."""
        delegate = _DottedTool()
        proxy = NameMangledToolProxy(delegate=delegate, mangled_name=mangle_tool_name(delegate.name))

        result = await proxy.ainvoke({})
        assert result == "r"
