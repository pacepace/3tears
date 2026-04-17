"""3tears-registry: MCP-compatible tool registry for 3tears tool system."""

from __future__ import annotations

__version__ = "0.5.0"

from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.discovery import DiscoveryHandler
from threetears.registry.health import HeartbeatMonitor, PodStatus
from threetears.registry.proxy import CallProxy
from threetears.registry.registration import RegistrationHandler, RegistrationResponse
from threetears.registry.routing import LeastConnectionsStrategy, RoutingStrategy
from threetears.registry.server import RegistryServer

__all__ = [
    "CallProxy",
    "CatalogEntry",
    "DiscoveryHandler",
    "HeartbeatMonitor",
    "LeastConnectionsStrategy",
    "PodStatus",
    "RegistrationHandler",
    "RegistrationResponse",
    "RegistryServer",
    "RoutingStrategy",
    "ToolCatalog",
    "ToolEndpoint",
]
