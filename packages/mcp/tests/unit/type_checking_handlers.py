"""handlers whose module imports its annotation types only for type checking.

their annotations stay text at runtime, so :class:`~threetears.mcp.tool.McpTool` reads them by this
module's imports. ``threetears.core`` exports a ``CallContext`` of its own (a tool dispatch's identity
envelope); a handler annotated with that one must not be taken for one asking for the MCP call's
context, and an alias of the MCP one must still be found.
"""

from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING, Annotated, Literal, Optional

if TYPE_CHECKING:
    from threetears.core.tool_wire import CallContext
    from threetears.core.tool_wire import CallRequest as Row
    import threetears.mcp.context as mcp_context
    from threetears import mcp as mcp_package
    from threetears.mcp import CallContext as McpCallContext


async def through_the_package_alias(*, ctx: mcp_package.CallContext | None = None) -> str:
    """the MCP CallContext through ``from threetears import mcp as mcp_package``."""
    return "ok"


async def through_the_module_alias(*, ctx: mcp_context.CallContext | None = None) -> str:
    """the MCP CallContext through ``import threetears.mcp.context as mcp_context``."""
    return "ok"


async def core_context(*, call: CallContext | None = None) -> str:
    """annotated with threetears.core's CallContext, imported only for type checking."""
    return "ok"


async def core_context_list(*, calls: list[CallContext] | None = None) -> str:
    """a list of threetears.core's CallContext: no mention of the MCP one."""
    return "ok"


async def aliased_mcp_context(*, ctx: McpCallContext | None = None) -> str:
    """the MCP CallContext under an alias, imported only for type checking."""
    return "ok"


async def annotated_aliased_context(*, ctx: Annotated[McpCallContext | None, "meta"] = None) -> str:
    """the MCP CallContext inside Annotated, its text unevaluable here."""
    return "ok"


async def quoted_aliased_context_list(*, ctxs: list["McpCallContext"] | None = None) -> str:  # noqa: UP037
    """a list of the MCP CallContext, quoted, in an annotation that does not evaluate here."""
    return "ok"


async def quoted_optional_alias(*, ctx: Optional["McpCallContext"] = None) -> str:  # noqa: UP037, UP045
    """the MCP CallContext or None, its name quoted, in an annotation that does not evaluate here."""
    return "ok"


async def annotated_with_prose_metadata(*, row: Annotated[Row, "the row id"] | None = None) -> str:
    """a type imported for type checking, with prose metadata; it names nothing unaccounted for."""
    return "ok"


async def bracket_in_metadata_context(*, ctx: Annotated[McpCallContext, "x]"] | None = None) -> str:
    """the MCP CallContext in Annotated whose metadata holds a closing bracket."""
    return "ok"


async def bracket_in_metadata_row(*, row: Annotated[Row, "a]b"] | None = None) -> str:
    """a type imported for type checking, in Annotated whose metadata holds a closing bracket."""
    return "ok"


async def literal_naming_it(*, mode: Literal["CallContext", "other"] = "other") -> str:
    """a Literal whose value happens to be the class's name."""
    return "ok"


async def module_qualified_annotated_context(*, ctx: t.Annotated[McpCallContext, "meta"]) -> str:
    """the MCP CallContext inside ``Annotated`` reached through a module's name (``t.Annotated``), as text."""
    return "ok"


async def module_qualified_optional_context(*, ctx: t.Optional[McpCallContext] = None) -> str:  # noqa: UP045
    """the MCP CallContext or None through a module's name (``t.Optional``), as text."""
    return "ok"


async def module_qualified_literal_naming_it(*, mode: t.Literal["CallContext", "other"] = "other") -> str:
    """a Literal reached through a module's name whose value happens to be the class's name."""
    return "ok"
