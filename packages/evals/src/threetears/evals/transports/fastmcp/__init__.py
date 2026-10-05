"""The FastMCP transport: mounts the action catalogue as tools on a FastMCP server.

Install with the ``fastmcp`` extra (``3tears-evals[fastmcp]``). The adapter is thin by construction:
each tool's name, description, input schema and behaviour hints are read off a
:class:`~threetears.evals.actions.MountedTool`, and a call is handed to
:meth:`~threetears.evals.actions.MountedTool.call` whole — selecting the action, refusing an undeclared
parameter, generating help and rendering the result all happen in the catalogue, so this transport and
any other answer a call identically.

A host mounts it with its own server, its catalogue (the engine's actions plus its own), its
:class:`~threetears.evals.ops.OpsHost` and a caller resolver — who is calling and the scope they act in,
asked once per call::

    from fastmcp import FastMCP
    from threetears.evals.actions import Caller, eval_catalogue
    from threetears.evals.transports.fastmcp import mount_fastmcp

    server = FastMCP("myapp")
    mount_fastmcp(server, eval_catalogue(my_actions), host=ops_host, caller=lambda: Caller(scope_id="dev", identity="me"))

A refused call is an MCP error result (``isError``) carrying the refusal's teaching text; a result
carries its text and the result model's JSON as structured content.

**This module is a public root.** A host imports from here and from no module below it.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastmcp.tools.base import ToolResult
from pydantic import ConfigDict, Field

from threetears.evals.actions import ActionCatalogue, Caller, MountedTool, ToolSpec, standard_tools
from threetears.evals.ops import OpsHost

#: Resolves who is calling, and their scope, for one call — sync or async. The host reads its own
#: request context (an access token, a session) to answer it.
CallerResolver = Callable[[], Caller | Awaitable[Caller]]


class ToolBinding:
    """What a :class:`CatalogueTool` hands each call to: the mounted tool, the host and the caller resolver.

    A plain class rather than a model or a dataclass, so the tool holds it by identity and FastMCP never
    walks the host's types to describe it.
    """

    def __init__(self, mounted: MountedTool, host: OpsHost, caller: CallerResolver) -> None:
        """Bind a mounted tool to the host and caller resolver its calls run under.

        Args:
            mounted: The tool cut from the catalogue.
            host: The host its actions work in.
            caller: Resolves who is calling, per call.
        """
        self.mounted = mounted
        self.host = host
        self.caller = caller


class CatalogueTool(Tool):
    """One mounted tool, as FastMCP serves it: every call handed to the catalogue.

    Attributes:
        binding: What each call is handed to. Excluded from the tool's serialized form: a client is shown
            what the catalogue declares — name, description, schema, hints — and nothing of the host.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    binding: ToolBinding = Field(exclude=True)

    @classmethod
    def over(cls, mounted: MountedTool, *, host: OpsHost, caller: CallerResolver) -> CatalogueTool:
        """The FastMCP tool for one mounted tool.

        Args:
            mounted: The tool cut from the catalogue.
            host: The host its actions work in.
            caller: Resolves who is calling, per call.

        Returns:
            The tool, ready for :meth:`fastmcp.FastMCP.add_tool`.
        """
        hints = mounted.hints
        # Validated rather than constructed, so the hints are the MCP annotation model FastMCP declares
        # without this adapter importing the SDK beneath it.
        return cls.model_validate(
            {
                "name": mounted.name,
                "description": mounted.description,
                "parameters": mounted.input_schema(),
                "annotations": {
                    "readOnlyHint": hints.read_only,
                    "destructiveHint": hints.destructive,
                    "idempotentHint": False,
                    "openWorldHint": hints.open_world,
                },
                "binding": ToolBinding(mounted, host, caller),
            }
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Carry out one call through the catalogue.

        Args:
            arguments: The call's arguments, ``action`` among them.

        Returns:
            The result: text always, the result model's JSON when the call succeeded, and the error
            flag when it was refused.
        """
        resolved = self.binding.caller()
        caller = await resolved if inspect.isawaitable(resolved) else resolved
        outcome = await self.binding.mounted.call(arguments, host=self.binding.host, caller=caller)
        return ToolResult(content=outcome.text, structured_content=outcome.structured, is_error=outcome.is_error)


def mount_fastmcp(
    server: FastMCP,
    catalogue: ActionCatalogue,
    *,
    host: OpsHost,
    caller: CallerResolver,
    tools: Sequence[ToolSpec] | None = None,
) -> tuple[str, ...]:
    """Add the catalogue's tools to a FastMCP server.

    Args:
        server: The host's server.
        catalogue: The actions to serve — :func:`~threetears.evals.actions.eval_catalogue`, with the host's own.
        host: The host every action works in.
        caller: Resolves who is calling and their scope, once per call.
        tools: The tools to cut; ``None`` for :func:`~threetears.evals.actions.standard_tools` under the
            default prefix. Pass :func:`~threetears.evals.actions.read_only_tools` for an agent that may
            only read, or either with the host's own prefix.

    Returns:
        The names of the tools added.

    Raises:
        ValueError: The catalogue refuses a tool (see :meth:`~threetears.evals.actions.ActionCatalogue.mount`).
    """
    mounted = catalogue.mount_all(tools if tools is not None else standard_tools())
    for tool in mounted:
        server.add_tool(CatalogueTool.over(tool, host=host, caller=caller))
    return tuple(tool.name for tool in mounted)


__all__ = ["CallerResolver", "CatalogueTool", "ToolBinding", "mount_fastmcp"]
