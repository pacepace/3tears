"""3tears-registry: MCP-compatible tool registry for 3tears tool system."""

from __future__ import annotations

__version__ = "0.5.0"

from threetears.registry.auth import (
    AgentToolAuthorizer,
    AllowAllAuthorizer,
    DenyAllAuthorizer,
    ToolPodAuth,
    ToolPodAuthenticator,
)
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.discovery import DiscoveryHandler
from threetears.registry.health import HeartbeatMonitor, PodStatus
from threetears.registry.proxy import CallProxy
from threetears.registry.rbac_authorizer import (
    NamespaceByNameResolver,
    RbacEvaluatorAuthorizer,
    ToolNamespaceRow,
)
from threetears.registry.registration import RegistrationHandler, RegistrationResponse
from threetears.registry.routing import LeastConnectionsStrategy, RoutingStrategy
from threetears.registry.server import RegistryServer

__all__ = [
    "AgentToolAuthorizer",
    "AllowAllAuthorizer",
    "CallProxy",
    "CatalogEntry",
    "DenyAllAuthorizer",
    "DiscoveryHandler",
    "HeartbeatMonitor",
    "LeastConnectionsStrategy",
    "NamespaceByNameResolver",
    "PodStatus",
    "RbacEvaluatorAuthorizer",
    "RegistrationHandler",
    "RegistrationResponse",
    "RegistryServer",
    "RoutingStrategy",
    "ToolCatalog",
    "ToolEndpoint",
    "ToolNamespaceRow",
    "ToolPodAuth",
    "ToolPodAuthenticator",
]
