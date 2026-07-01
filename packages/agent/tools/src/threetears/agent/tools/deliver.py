"""General deliver tool: resolve a stored object by id -> presigned download URL.

The first real object-store *consumer* (Path-2 P2.3). A consuming pod registers
this tool; the agent calls it with an object id (e.g. one a prior produce
returned) and the tool resolves that id to its stored key -- tenant-safely, via
the hub -- then mints a presigned GET URL so a human downloads the bytes
directly from the store. The bytes never cross NATS or the agent.

The tool is *general*: the resolver + presign are framework primitives
(:mod:`threetears.agent.tools.consume`), and this tool only wires them. Its
registered name is constructor-parameterised so a consuming pod names it in its
own namespace (e.g. a pentest pod registers
``DeliverObjectTool(name="pentest.deliver_object")``), mirroring
:class:`~threetears.agent.tools.report.ReportTool`.

Fail-closed: a malformed id, an object the verified customer does not own, or an
object that cannot be resolved / presigned returns a clean refusal, never a
disclosure. The object id is untrusted (an LLM arg) -- only the hub, which owns
the object catalog, authorises it against the verified customer.
"""

from __future__ import annotations

import os
from typing import Any
from uuid import UUID

from threetears.observe import get_logger

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.consume import ConsumeObjectError, presigned_object_url, resolve_object
from threetears.agent.tools.object_resolver import ResolveObjectError

__all__ = ["DeliverObjectTool"]

_log = get_logger(__name__)

#: default registered name; a consuming pod overrides it for its namespace.
_DEFAULT_NAME = "threetears.deliver_object"

#: presigned delivery-URL validity; override with DELIVER_PRESIGN_TTL_SECONDS.
_DEFAULT_PRESIGN_TTL = 3600
_PRESIGN_TTL_ENV = "DELIVER_PRESIGN_TTL_SECONDS"

#: expected max resolve+presign time (a NATS round-trip + a presign, no
#: rendering); override with DELIVER_TIMEOUT_SECONDS.
_DEFAULT_TIMEOUT_SECONDS = 30.0
_TIMEOUT_ENV = "DELIVER_TIMEOUT_SECONDS"

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "object_id": {
            "type": "string",
            "description": (
                "The id of a previously stored object (e.g. a report or capture a "
                "prior tool produced, as returned in that tool's result). Returns a "
                "time-limited download URL for it. Must be an object owned by this "
                "customer; unknown or foreign ids are refused."
            ),
        },
    },
    "required": ["object_id"],
}


def _timeout_seconds() -> float:
    """Resolve the tool's expected-max resolve+presign time from the environment.

    :return: timeout in seconds (falls back to the default on unset/invalid)
    :rtype: float
    """
    raw = os.environ.get(_TIMEOUT_ENV)
    if raw:
        try:
            value = float(raw)
        except ValueError:
            _log.warning("invalid %s=%r; using default", _TIMEOUT_ENV, raw)
        else:
            if value > 0:
                return value
    return _DEFAULT_TIMEOUT_SECONDS


def _presign_ttl() -> int:
    """Resolve the presigned-URL validity (seconds) from the environment.

    :return: TTL in seconds (falls back to the default on unset/invalid)
    :rtype: int
    """
    raw = os.environ.get(_PRESIGN_TTL_ENV)
    if raw:
        try:
            value = int(raw)
        except ValueError:
            _log.warning("invalid %s=%r; using default", _PRESIGN_TTL_ENV, raw)
        else:
            if value > 0:
                return value
    return _DEFAULT_PRESIGN_TTL


class DeliverObjectTool(TearsTool):
    """Resolve a stored object by id and return a presigned download URL.

    Takes an ``object_id``, resolves it to its stored key tenant-safely (the hub
    verifies the invoking agent's identity + confirms the verified customer owns
    the object), and mints a presigned GET URL for it. The bytes are never
    returned inline -- the URL points at the object store.

    :param name: the registered MCP tool name; defaults to
        ``threetears.deliver_object``. A consuming pod passes its own namespace
        (e.g. ``pentest.deliver_object``).
    :ptype name: str
    """

    def __init__(self, name: str = _DEFAULT_NAME) -> None:
        """Initialize the tool with its registered name.

        :param name: registered MCP tool name (namespace-qualified)
        :ptype name: str
        """
        super().__init__()
        self._name = name

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Resolve an object id and deliver a presigned URL for it.

        :param kwargs: tool input (``object_id`` required)
        :ptype kwargs: Any
        :return: the presigned download URL + object metadata, or a fail-closed
            refusal (bad id / not owned / unresolvable / no store)
        :rtype: ToolResult
        """
        raw_id = kwargs.get("object_id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            return ToolResult(
                success=False,
                content="",
                error="provide 'object_id' (the id of a stored object to deliver)",
            )
        try:
            object_id = UUID(raw_id.strip())
        except ValueError:
            return ToolResult(success=False, content="", error=f"'object_id' is not a valid id: {raw_id!r}")

        try:
            handle = await resolve_object(object_id)
        except ConsumeObjectError as exc:
            # structural refusal: no scope / no resolver / no verified customer /
            # no identity token / the resolved key is not owned by the customer.
            _log.warning("deliver refused: cannot resolve", extra={"extra_data": {"error": str(exc)}})
            return ToolResult(success=False, content="", error=f"refused: {exc}")
        except ResolveObjectError as exc:
            # the hub rejected the resolve (identity unverified, or the customer
            # does not own the object) or it failed in transit.
            _log.warning("deliver refused: hub rejected resolve", extra={"extra_data": {"error": str(exc)}})
            return ToolResult(success=False, content="", error=f"could not deliver object {object_id}: {exc}")

        ttl = _presign_ttl()
        try:
            url = await presigned_object_url(handle.s3_key, expires_in=ttl)
        except ConsumeObjectError as exc:
            # unlike ReportTool (where presign is best-effort ON TOP of a produce),
            # the URL IS the deliverable here, so a presign failure is a refusal.
            _log.warning("deliver refused: cannot presign", extra={"extra_data": {"error": str(exc)}})
            return ToolResult(success=False, content="", error=f"refused: cannot deliver the object ({exc})")

        return ToolResult(
            success=True,
            content=(
                f"Object {object_id} ({handle.size_bytes} bytes, {handle.mime_type}) is ready. "
                f"Download link (expires in {ttl}s): {url}"
            ),
            metadata={
                "object_id": str(object_id),  # convert at border: ToolResult.metadata JSON field (crosses NATS)
                "mime_type": handle.mime_type,
                "size_bytes": handle.size_bytes,
                "delivery_url": url,
            },
        )

    def mcp_schema(self) -> MCPToolDefinition:
        """Return the MCP-compatible tool definition.

        :return: the tool definition
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description=(
                "Deliver a previously stored object (e.g. a report or capture a prior tool "
                "produced) as a time-limited download link. Provide 'object_id' -- the id from "
                "that tool's result. Returns a presigned URL; the object bytes are not returned "
                "inline. Unknown or foreign object ids are refused."
            ),
            input_schema=_INPUT_SCHEMA,
            timeout_seconds=_timeout_seconds(),
        )

    def mcp_name(self) -> str:
        """Return the namespaced tool name.

        :return: the registered tool name
        :rtype: str
        """
        return self._name

    def mcp_version(self) -> str:
        """Return the tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"
