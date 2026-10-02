"""Shared test helpers for the provider name-translation wrappers.

The OpenAI / OpenRouter / Anthropic provider-wrapper suites all need the same
dotted-named tool to exercise dot<->underscore translation. Define it once here
and import it (see the sibling ``test_provider_*.py``) rather than copying it
into each file.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool


class DottedTool(BaseTool):
    """Minimal :class:`BaseTool` whose ``.name`` carries a dot, for tests."""

    name: str = "threetears.calculator"
    description: str = "test calculator"
    invoked_with: list[dict[str, Any]] = []

    def _run(self, **kwargs: Any) -> str:
        self.invoked_with.append(dict(kwargs))
        return "ok"

    async def _arun(self, **kwargs: Any) -> str:
        self.invoked_with.append(dict(kwargs))
        return "ok"
