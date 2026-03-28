"""3tears-registry: MCP-compatible tool registry for 3tears tool system."""

from __future__ import annotations

__version__ = "0.5.0"

from threetears.registry.catalog import CatalogEntry, ToolCatalog
from threetears.registry.health import HeartbeatMonitor, PodStatus
from threetears.registry.registration import RegistrationHandler, RegistrationResponse

__all__ = [
    "CatalogEntry",
    "HeartbeatMonitor",
    "PodStatus",
    "RegistrationHandler",
    "RegistrationResponse",
    "ToolCatalog",
]
