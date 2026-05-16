"""3tears-registry: MCP-compatible tool registry for 3tears tool system."""

from __future__ import annotations

__version__ = "0.7.0"

from threetears.registry.auth import (
    AgentToolAuthorizer,
    AllowAllAuthorizer,
    DenyAllAuthorizer,
    ToolPodAuth,
    ToolPodAuthenticator,
)
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.discovery import DiscoveryHandler
from threetears.registry.entities import HeartbeatEntity
from threetears.registry.health import HeartbeatSubscriber
from threetears.registry.heartbeat_collection import HeartbeatCollection
from threetears.registry.l1_cache import (
    REGISTRY_L1_METADATA,
    REGISTRY_L1_TABLE_NAMES,
    create_registry_l1_backend,
    pod_heartbeats_table,
)
from threetears.registry.proxy import CallProxy
from threetears.registry.rbac_authorizer import RbacEvaluatorAuthorizer
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
    "HeartbeatCollection",
    "HeartbeatEntity",
    "HeartbeatSubscriber",
    "LeastConnectionsStrategy",
    "REGISTRY_L1_METADATA",
    "REGISTRY_L1_TABLE_NAMES",
    "RbacEvaluatorAuthorizer",
    "RegistrationHandler",
    "RegistrationResponse",
    "RegistryServer",
    "RoutingStrategy",
    "ToolCatalog",
    "ToolEndpoint",
    "ToolPodAuth",
    "ToolPodAuthenticator",
    "create_registry_l1_backend",
    "pod_heartbeats_table",
]
