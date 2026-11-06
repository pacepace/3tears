"""read an MCP SDK model by its wire field names, the same on mcp 1.x and 2.x.

mcp 1.x names the fields as on the wire (``inputSchema``, ``readOnlyHint``); mcp 2.x names them
in snake_case and keeps the wire name as each field's alias, and FastMCP 4 deprecates reading the
old attribute names. ``model_dump(by_alias=True)`` answers with the wire names on both, so a test
reads one spelling and runs under ``scripts/test-mcp-v1.sh`` and the locked 2.x alike.
"""

from __future__ import annotations

from typing import Any


def wire(model: Any) -> dict[str, Any]:
    """the model's fields keyed by their MCP wire names.

    :param model: an MCP SDK pydantic model, such as a listed ``Tool`` or its ``ToolAnnotations``
    :ptype model: Any
    :return: field values by wire name (``inputSchema``, ``readOnlyHint``, ...)
    :rtype: dict[str, Any]
    """
    fields: dict[str, Any] = model.model_dump(by_alias=True)
    return fields
