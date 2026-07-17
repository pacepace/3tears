"""Tests for :mod:`threetears.models.tool_name_translation`'s ``NameMangledToolProxy``.

Scoped to the ``canonical_name`` public accessor added alongside the claude-cli dotted-tool-name
permission fix (see ``test_claude_cli_tool_events.py``); the module's other primitives are already
exercised indirectly via the ``anthropic``/``openrouter`` provider test files.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from threetears.models.tool_name_translation import NameMangledToolProxy, mangle_tool_name


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
