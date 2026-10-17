"""Pin the exact schema block a model reads for a column's free-text description.

A column description is warehouse documentation of any number of lines. It must stay on
its column's bullet line, joined into one line, so no continuation line lands at the
margin where it would read as another column or a new section.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import asyncio

from langchain.agents.middleware import ModelRequest
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import var_child_runnable_config

from threetears.langgraph.middleware_schema import SchemaPrimingMiddleware


class _Integration:
    """Duck-typed schema-priming integration handing over fixed tables."""

    datasource_names = ["ds-1"]

    def __init__(self, tables: list[Any]) -> None:
        self._tables = tables

    async def documented_schema(self) -> list[Any]:
        return self._tables


def _block(columns: list[dict[str, Any]]) -> str:
    """The documented-schema block primed for one table holding ``columns``."""
    table = {"schema": "public", "table": "orders", "description": "customer orders", "columns": columns}
    request = ModelRequest(
        model=cast("BaseChatModel", SimpleNamespace()), messages=[HumanMessage(content="hi")], system_message=None
    )

    async def handler(_req: ModelRequest) -> Any:
        return SimpleNamespace(result=[AIMessage(content="ok")])

    async def run() -> Any:
        token = var_child_runnable_config.set({"configurable": {"schema_priming_integration": _Integration([table])}})
        try:
            return await SchemaPrimingMiddleware().awrap_model_call(request, handler)
        finally:
            var_child_runnable_config.reset(token)

    out = asyncio.run(run())
    assert isinstance(out, ExtendedModelResponse)
    assert out.command is not None
    block = out.command.update["metadata"]["documented_schema_block"]
    assert isinstance(block, str)
    return block


def test_a_many_line_column_description_is_joined_onto_the_column_line() -> None:
    block = _block(
        [
            {"name": "status", "type": "text", "description": "Order state.\n\n  One of:\n    open\n    shipped\n"},
            {"name": "id", "type": "int", "description": "pk"},
        ]
    )

    assert block.endswith(
        "### public.orders\ncustomer orders\n- status (text) -- Order state. One of: open shipped\n- id (int) -- pk"
    )


def test_a_column_with_a_blank_description_still_shows_its_column_line() -> None:
    block = _block([{"name": "status", "type": "text", "description": " \n\n "}, {"name": "id", "type": "int"}])

    assert block.endswith("### public.orders\ncustomer orders\n- status (text)\n- id (int)")
