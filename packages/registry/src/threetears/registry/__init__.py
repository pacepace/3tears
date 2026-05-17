"""3tears-registry: MCP-compatible tool registry for 3tears tool system."""

from __future__ import annotations

# Version derived from pyproject.toml so the metadata is the single
# source of truth -- a future release that bumps pyproject without
# updating ``__init__.py`` can't drift the runtime ``__version__``.
# The except guard handles the rare case where the package isn't
# installed via importlib.metadata (e.g. running directly from a
# checked-out source tree without ``uv sync``); the fallback keeps
# imports working but reports ``unknown`` rather than crashing.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("3tears-registry")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

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
