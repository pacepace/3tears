"""path-template binding, shared by the outbound and inbound HTTP faces, under its agent-tools name.

:class:`RestAffordance`, the :class:`PathTemplateBinding` it shares with the
outbound :class:`~threetears.agent.tools.builtin.http_api_tool.HttpOperationDescriptor`,
and the names beside them are defined in :mod:`threetears.core.tool_wire.rest`:
a tool's manifest entry carries its REST declaration, and the registry reads
the manifest as well as the pod. this module re-exports the same objects.
"""

from __future__ import annotations

from threetears.core.tool_wire.rest import (
    CACHEABLE_METHODS,
    MAX_POINTER_AGE_SECONDS,
    QUERY_METHODS,
    HttpMethod,
    ParameterBinding,
    PathTemplateBinding,
    RestAffordance,
    RestAffordanceError,
)

__all__ = [
    "CACHEABLE_METHODS",
    "MAX_POINTER_AGE_SECONDS",
    "QUERY_METHODS",
    "HttpMethod",
    "ParameterBinding",
    "PathTemplateBinding",
    "RestAffordance",
    "RestAffordanceError",
]
