"""tool catalog for centralized tool registration and discovery.

maintains in-memory catalog of registered tools with NATS KV
persistence. provides lookup by full name, search by name/version,
and availability tracking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from threetears.core.logging import get_logger

_logger = get_logger(__name__)


def _sanitize_kv_key(full_name: str) -> str:
    """convert full_name to NATS KV-safe key.

    NATS KV keys cannot contain dots or @ characters.
    replaces dots with underscores and @ with _AT_.

    :param full_name: tool full_name as name@version
    :ptype full_name: str
    :return: KV-safe key string
    :rtype: str
    """
    return full_name.replace(".", "_").replace("@", "_AT_")


@dataclass
class CatalogEntry:
    """single tool entry in catalog.

    :param tool_name: namespaced tool name (e.g., 'threetears.calculator')
    :ptype tool_name: str
    :param tool_version: semver-compatible version string
    :ptype tool_version: str
    :param full_name: composite key as name@version
    :ptype full_name: str
    :param pod_id: identifier of pod serving this tool
    :ptype pod_id: str
    :param description: human-readable tool description
    :ptype description: str
    :param input_schema: JSON Schema for tool input parameters
    :ptype input_schema: dict[str, Any]
    :param output_schema: optional JSON Schema for tool output
    :ptype output_schema: dict[str, Any] | None
    :param status: availability status ('available' or 'unavailable')
    :ptype status: str
    :param date_registered: timestamp when tool was registered
    :ptype date_registered: datetime
    :param date_last_heartbeat: timestamp of last heartbeat from serving pod
    :ptype date_last_heartbeat: datetime
    """

    tool_name: str
    tool_version: str
    full_name: str
    pod_id: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    status: str = "available"
    date_registered: datetime = field(default_factory=lambda: datetime.now(UTC))
    date_last_heartbeat: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """serialize entry to dictionary for KV storage.

        :return: dictionary representation of catalog entry
        :rtype: dict[str, Any]
        """
        result = {
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "full_name": self.full_name,
            "pod_id": self.pod_id,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "status": self.status,
            "date_registered": self.date_registered.isoformat(),
            "date_last_heartbeat": self.date_last_heartbeat.isoformat(),
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CatalogEntry:
        """deserialize entry from dictionary loaded from KV storage.

        :param data: dictionary representation of catalog entry
        :ptype data: dict[str, Any]
        :return: reconstructed catalog entry
        :rtype: CatalogEntry
        """
        result = cls(
            tool_name=data["tool_name"],
            tool_version=data["tool_version"],
            full_name=data["full_name"],
            pod_id=data["pod_id"],
            description=data["description"],
            input_schema=data["input_schema"],
            output_schema=data.get("output_schema"),
            status=data.get("status", "unavailable"),
            date_registered=datetime.fromisoformat(data["date_registered"]),
            date_last_heartbeat=datetime.fromisoformat(data["date_last_heartbeat"]),
        )
        return result


class ToolCatalog:
    """centralized catalog of registered tools.

    maintains in-memory dictionary of catalog entries keyed by
    full_name (name@version). supports NATS KV persistence for
    recovery after restart, with all entries marked unavailable
    on load until heartbeats confirm liveness.
    """

    def __init__(self) -> None:
        """initialize empty tool catalog."""
        self._entries: dict[str, CatalogEntry] = {}
        self._kv: Any | None = None

    async def load_from_kv(self, kv: Any) -> None:
        """load catalog entries from NATS KV store.

        loads all entries and marks them as unavailable until
        heartbeats confirm liveness. stores KV reference for
        subsequent write operations.

        :param kv: NATS KV store instance
        :ptype kv: Any
        """
        self._kv = kv
        try:
            keys = await kv.keys()
        except Exception:
            keys = []
        for key in keys:
            kv_entry = await kv.get(key)
            data = json.loads(kv_entry.value.decode("utf-8"))
            entry = CatalogEntry.from_dict(data)
            entry.status = "unavailable"
            self._entries[entry.full_name] = entry
        _logger.info(
            "loaded catalog from KV",
            extra={"extra_data": {"entry_count": len(self._entries)}},
        )

    async def register(self, entry: CatalogEntry) -> None:
        """register tool in catalog and persist to KV.

        :param entry: catalog entry to register
        :ptype entry: CatalogEntry
        """
        self._entries[entry.full_name] = entry
        if self._kv is not None:
            kv_key = _sanitize_kv_key(entry.full_name)
            await self._kv.put(
                kv_key,
                json.dumps(entry.to_dict()).encode("utf-8"),
            )
        _logger.info(
            "registered tool in catalog",
            extra={"extra_data": {
                "full_name": entry.full_name,
                "pod_id": entry.pod_id,
            }},
        )

    async def deregister(self, full_name: str) -> None:
        """remove tool from catalog and delete from KV.

        :param full_name: composite key as name@version
        :ptype full_name: str
        """
        self._entries.pop(full_name, None)
        if self._kv is not None:
            try:
                kv_key = _sanitize_kv_key(full_name)
                await self._kv.delete(kv_key)
            except Exception:
                pass
        _logger.info(
            "deregistered tool from catalog",
            extra={"extra_data": {"full_name": full_name}},
        )

    async def deregister_pod(self, pod_id: str) -> list[str]:
        """remove all tools registered by specified pod.

        :param pod_id: identifier of pod whose tools should be removed
        :ptype pod_id: str
        :return: list of full_name values that were removed
        :rtype: list[str]
        """
        to_remove = [
            full_name
            for full_name, entry in self._entries.items()
            if entry.pod_id == pod_id
        ]
        for full_name in to_remove:
            await self.deregister(full_name)
        result = to_remove
        return result

    def get(self, full_name: str) -> CatalogEntry | None:
        """look up tool by full name.

        :param full_name: composite key as name@version
        :ptype full_name: str
        :return: catalog entry if found, None otherwise
        :rtype: CatalogEntry | None
        """
        result = self._entries.get(full_name)
        return result

    def search(
        self,
        name: str | None = None,
        version: str | None = None,
    ) -> list[CatalogEntry]:
        """search catalog by tool name and/or version.

        :param name: filter by tool name (substring match)
        :ptype name: str | None
        :param version: filter by exact version string
        :ptype version: str | None
        :return: list of matching catalog entries
        :rtype: list[CatalogEntry]
        """
        results: list[CatalogEntry] = []
        for entry in self._entries.values():
            if name is not None and name not in entry.tool_name:
                continue
            if version is not None and entry.tool_version != version:
                continue
            results.append(entry)
        return results

    def list_available(self) -> list[CatalogEntry]:
        """list all tools with available status.

        :return: list of available catalog entries
        :rtype: list[CatalogEntry]
        """
        result = [
            entry
            for entry in self._entries.values()
            if entry.status == "available"
        ]
        return result

    def mark_available(self, full_name: str) -> bool:
        """mark tool as available.

        :param full_name: composite key as name@version
        :ptype full_name: str
        :return: True if entry was found and updated, False otherwise
        :rtype: bool
        """
        entry = self._entries.get(full_name)
        if entry is None:
            return False
        entry.status = "available"
        return True

    def mark_unavailable(self, full_name: str) -> bool:
        """mark tool as unavailable.

        :param full_name: composite key as name@version
        :ptype full_name: str
        :return: True if entry was found and updated, False otherwise
        :rtype: bool
        """
        entry = self._entries.get(full_name)
        if entry is None:
            return False
        entry.status = "unavailable"
        return True
