"""shared builders for catalog entries whose endpoints each carry their OWN tool definition.

A catalog entry no longer holds a description or a schema: every endpoint (one pod's copy of a
tool) carries the definitions that pod announced. The registry tests build entries in many shapes,
and an endpoint with no live definition is visible to nobody, so each builder here gives every
endpoint a definition announced NOW unless the test says otherwise.

not a ``test_*`` module, so pytest does not collect it; the registry test package makes it
importable as a sibling.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from threetears.registry.catalog import AnnouncedDefinition, CatalogEntry, ToolDefinition, ToolEndpoint

__all__ = [
    "announced",
    "definition",
    "endpoint",
    "entry",
    "uniform_entry",
]


def definition(
    description: str = "a test tool",
    *,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    timeout_seconds: float | None = None,
    requires_confirmation: bool = False,
) -> ToolDefinition:
    """one tool definition, defaulting to an empty object schema.

    :param description: human-readable description
    :ptype description: str
    :param input_schema: JSON schema for the input, or ``None`` for an empty object schema
    :ptype input_schema: dict[str, Any] | None
    :param output_schema: JSON schema for the output, or ``None``
    :ptype output_schema: dict[str, Any] | None
    :param timeout_seconds: the tool's declared timeout, or ``None``
    :ptype timeout_seconds: float | None
    :param requires_confirmation: whether a call must be approved by a person
    :ptype requires_confirmation: bool
    :return: the definition
    :rtype: ToolDefinition
    """
    return ToolDefinition(
        description=description,
        input_schema=input_schema if input_schema is not None else {"type": "object", "properties": {}},
        output_schema=output_schema,
        timeout_seconds=timeout_seconds,
        requires_confirmation=requires_confirmation,
    )


def announced(
    tool_definition: ToolDefinition,
    *,
    first: datetime | None = None,
    last: datetime | None = None,
) -> dict[str, AnnouncedDefinition]:
    """the ``definitions`` mapping of an endpoint that announced one definition.

    :param tool_definition: the definition announced
    :ptype tool_definition: ToolDefinition
    :param first: when it was first announced; now when omitted
    :ptype first: datetime | None
    :param last: when it was last announced; ``first`` when omitted
    :ptype last: datetime | None
    :return: the mapping keyed by the definition's digest
    :rtype: dict[str, AnnouncedDefinition]
    """
    first_at = first if first is not None else datetime.now(UTC)
    last_at = last if last is not None else first_at
    return {
        tool_definition.digest: AnnouncedDefinition(
            definition=tool_definition,
            first_announced=first_at,
            last_announced=last_at,
        )
    }


def endpoint(
    pod_id: str,
    status: str = "available",
    *,
    tool_definition: ToolDefinition | None = None,
    first: datetime | None = None,
    verified_publisher: bool = False,
) -> ToolEndpoint:
    """one pod's copy of a tool, carrying one definition announced at ``first``.

    :param pod_id: the serving pod
    :ptype pod_id: str
    :param status: the endpoint's lifecycle status
    :ptype status: str
    :param tool_definition: the definition this copy announced; a default one when omitted
    :ptype tool_definition: ToolDefinition | None
    :param first: when the definition was first announced; now when omitted
    :ptype first: datetime | None
    :param verified_publisher: whether a verified publisher registered this copy
    :ptype verified_publisher: bool
    :return: the endpoint
    :rtype: ToolEndpoint
    """
    chosen = tool_definition if tool_definition is not None else definition()
    return ToolEndpoint(
        pod_id=pod_id,
        status=status,
        definitions=announced(chosen, first=first),
        verified_publisher=verified_publisher,
    )


def entry(
    tool_name: str = "threetears.calculator",
    tool_version: str = "1.0.0",
    *endpoints: ToolEndpoint,
) -> CatalogEntry:
    """a catalog entry holding the given copies.

    :param tool_name: namespaced tool name
    :ptype tool_name: str
    :param tool_version: tool version
    :ptype tool_version: str
    :param endpoints: the copies serving it
    :ptype endpoints: ToolEndpoint
    :return: the entry
    :rtype: CatalogEntry
    """
    return CatalogEntry(
        tool_name=tool_name,
        tool_version=tool_version,
        full_name=f"{tool_name}@{tool_version}",
        endpoints=list(endpoints),
    )


def uniform_entry(
    *,
    tool_name: str,
    tool_version: str,
    full_name: str | None = None,
    description: str,
    input_schema: dict[str, Any],
    output_schema: dict[str, Any] | None = None,
    timeout_seconds: float | None = None,
    requires_confirmation: bool = False,
    endpoints: Iterable[ToolEndpoint] = (),
    date_registered: datetime | None = None,
) -> CatalogEntry:
    """a catalog entry every one of whose copies announced the SAME definition, now.

    The shape most routing and dispatch tests need: they are about where a call goes, not about
    copies disagreeing. An endpoint that already carries definitions keeps them.

    :param tool_name: namespaced tool name
    :ptype tool_name: str
    :param tool_version: tool version
    :ptype tool_version: str
    :param full_name: ``name@version``; derived when omitted
    :ptype full_name: str | None
    :param description: the shared description
    :ptype description: str
    :param input_schema: the shared input schema
    :ptype input_schema: dict[str, Any]
    :param output_schema: the shared output schema
    :ptype output_schema: dict[str, Any] | None
    :param timeout_seconds: the shared declared timeout
    :ptype timeout_seconds: float | None
    :param requires_confirmation: whether every copy requires confirmation
    :ptype requires_confirmation: bool
    :param endpoints: the copies
    :ptype endpoints: Iterable[ToolEndpoint]
    :param date_registered: when the entry was registered; now when omitted
    :ptype date_registered: datetime | None
    :return: the entry
    :rtype: CatalogEntry
    """
    shared = ToolDefinition(
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        timeout_seconds=timeout_seconds,
        requires_confirmation=requires_confirmation,
    )
    copies = list(endpoints)
    for copy in copies:
        if not copy.definitions:
            copy.definitions = announced(shared)
    result = CatalogEntry(
        tool_name=tool_name,
        tool_version=tool_version,
        full_name=full_name if full_name is not None else f"{tool_name}@{tool_version}",
        endpoints=copies,
    )
    if date_registered is not None:
        result.date_registered = date_registered
    return result
