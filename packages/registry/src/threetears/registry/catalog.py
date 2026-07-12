"""tool catalog for centralized tool registration and discovery.

maintains in-memory catalog of registered tools with NATS KV
persistence. supports multiple endpoints per tool for horizontal
scaling with load-balanced routing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from threetears.observe import get_logger

__all__ = [
    "CatalogEntry",
    "ToolCatalog",
    "ToolEndpoint",
]

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


def _replace_endpoint_status(
    endpoint: ToolEndpoint,
    pod_id: str,
    status: str,
) -> ToolEndpoint:
    """return a new ToolEndpoint reflecting a projected status change.

    used by ``mark_ready`` to build the KV snapshot that would be
    persisted if the transition succeeded, without mutating the
    in-memory endpoint ahead of the write. when ``endpoint.pod_id``
    does not match ``pod_id``, returns a copy preserving the current
    status.

    :param endpoint: source endpoint to copy
    :ptype endpoint: ToolEndpoint
    :param pod_id: identifier of pod whose endpoint status should be set
    :ptype pod_id: str
    :param status: status value to apply when pod_id matches
    :ptype status: str
    :return: new ToolEndpoint instance with projected status
    :rtype: ToolEndpoint
    """
    projected_status = status if endpoint.pod_id == pod_id else endpoint.status
    return ToolEndpoint(
        pod_id=endpoint.pod_id,
        status=projected_status,
        in_flight=endpoint.in_flight,
        date_last_heartbeat=endpoint.date_last_heartbeat,
    )


@dataclass
class ToolEndpoint:
    """single pod endpoint serving a tool.

    tracks liveness and in-flight call count for
    load-balanced routing across multiple pods. status values
    follow a three-phase lifecycle: 'pending' on initial
    registration (before reachability probe round-trips),
    'available' after probe confirms reachability or heartbeat
    refreshes liveness, and 'unavailable' after missed
    heartbeats. calls route only to 'available' endpoints.

    :param pod_id: identifier of pod serving this tool
    :ptype pod_id: str
    :param status: availability status ('pending', 'available', or 'unavailable')
    :ptype status: str
    :param in_flight: number of currently in-flight calls to this endpoint
    :ptype in_flight: int
    :param date_last_heartbeat: timestamp of last heartbeat from this pod
    :ptype date_last_heartbeat: datetime
    """

    pod_id: str
    status: str = "pending"
    in_flight: int = 0
    date_last_heartbeat: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """serialize endpoint to dictionary for KV storage.

        :return: dictionary representation of endpoint
        :rtype: dict[str, Any]
        """
        result = {
            "pod_id": self.pod_id,
            "status": self.status,
            "date_last_heartbeat": self.date_last_heartbeat.isoformat(),
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolEndpoint:
        """deserialize endpoint from dictionary loaded from KV storage.

        in_flight is always reset to zero on load since pending calls
        cannot survive a restart.

        :param data: dictionary representation of endpoint
        :ptype data: dict[str, Any]
        :return: reconstructed endpoint
        :rtype: ToolEndpoint
        """
        result = cls(
            pod_id=data["pod_id"],
            status=data.get("status", "unavailable"),
            in_flight=0,
            date_last_heartbeat=datetime.fromisoformat(data["date_last_heartbeat"]),
        )
        return result


@dataclass
class CatalogEntry:
    """single tool entry in catalog.

    represents a tool definition (schema, description) with one or
    more endpoints (pods) that serve it. availability is derived
    from endpoint statuses.

    :param tool_name: namespaced tool name (e.g., 'threetears.calculator')
    :ptype tool_name: str
    :param tool_version: semver-compatible version string
    :ptype tool_version: str
    :param full_name: composite key as name@version
    :ptype full_name: str
    :param description: human-readable tool description
    :ptype description: str
    :param input_schema: JSON Schema for tool input parameters
    :ptype input_schema: dict[str, Any]
    :param output_schema: optional JSON Schema for tool output
    :ptype output_schema: dict[str, Any] | None
    :param endpoints: list of pod endpoints serving this tool
    :ptype endpoints: list[ToolEndpoint]
    :param date_registered: timestamp when tool was first registered
    :ptype date_registered: datetime
    :param requires_confirmation: whether calls to the tool must be gated
        behind human-in-the-loop approval; mirrors
        :attr:`TearsTool.requires_confirmation`. Defaults to ``False``.
    :ptype requires_confirmation: bool
    """

    tool_name: str
    tool_version: str
    full_name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    timeout_seconds: float | None = None
    requires_confirmation: bool = False
    endpoints: list[ToolEndpoint] = field(default_factory=list)
    date_registered: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def status(self) -> str:
        """aggregate availability status from endpoints.

        pending endpoints (awaiting probe confirmation) do not
        count toward availability -- only fully-confirmed endpoints
        do. an entry whose endpoints are all pending aggregates
        to 'unavailable' so callers treat it as not-yet-routable.

        :return: 'available' if any endpoint is available, 'unavailable' otherwise
        :rtype: str
        """
        for endpoint in self.endpoints:
            if endpoint.status == "available":
                return "available"
        return "unavailable"

    def get_endpoint(self, pod_id: str) -> ToolEndpoint | None:
        """look up endpoint by pod_id.

        :param pod_id: identifier of pod to find
        :ptype pod_id: str
        :return: endpoint if found, None otherwise
        :rtype: ToolEndpoint | None
        """
        for endpoint in self.endpoints:
            if endpoint.pod_id == pod_id:
                return endpoint
        return None

    def add_endpoint(self, endpoint: ToolEndpoint) -> None:
        """add or update endpoint for pod.

        if pod already has an endpoint, replaces it.
        otherwise appends new endpoint.

        :param endpoint: endpoint to add or update
        :ptype endpoint: ToolEndpoint
        """
        for i, existing in enumerate(self.endpoints):
            if existing.pod_id == endpoint.pod_id:
                self.endpoints[i] = endpoint
                return
        self.endpoints.append(endpoint)

    def remove_endpoint(self, pod_id: str) -> bool:
        """remove endpoint for specified pod.

        :param pod_id: identifier of pod whose endpoint to remove
        :ptype pod_id: str
        :return: True if endpoint was found and removed, False otherwise
        :rtype: bool
        """
        for i, endpoint in enumerate(self.endpoints):
            if endpoint.pod_id == pod_id:
                self.endpoints.pop(i)
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """serialize entry to dictionary for KV storage.

        :return: dictionary representation of catalog entry
        :rtype: dict[str, Any]
        """
        result = {
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "full_name": self.full_name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "timeout_seconds": self.timeout_seconds,
            "requires_confirmation": self.requires_confirmation,
            "endpoints": [ep.to_dict() for ep in self.endpoints],
            "date_registered": self.date_registered.isoformat(),
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
        endpoints = [ToolEndpoint.from_dict(ep_data) for ep_data in data.get("endpoints", [])]
        result = cls(
            tool_name=data["tool_name"],
            tool_version=data["tool_version"],
            full_name=data["full_name"],
            description=data["description"],
            input_schema=data["input_schema"],
            output_schema=data.get("output_schema"),
            timeout_seconds=data.get("timeout_seconds"),
            requires_confirmation=data.get("requires_confirmation", False),
            endpoints=endpoints,
            date_registered=datetime.fromisoformat(data["date_registered"]),
        )
        return result


class ToolCatalog:
    """centralized catalog of registered tools.

    maintains in-memory dictionary of catalog entries keyed by
    full_name (name@version). each entry can have multiple endpoints
    (pods) serving it. supports NATS KV persistence for recovery
    after restart, with all endpoints marked unavailable on load
    until heartbeats confirm liveness.
    """

    def __init__(self) -> None:
        """initialize empty tool catalog."""
        self._entries: dict[str, CatalogEntry] = {}
        self._kv: Any | None = None

    async def load_from_kv(self, kv: Any) -> None:
        """load catalog entries from NATS KV store.

        loads all entries and marks all endpoints as unavailable
        until heartbeats confirm liveness. stores KV reference for
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
            for endpoint in entry.endpoints:
                endpoint.status = "unavailable"
            self._entries[entry.full_name] = entry
        _logger.info(
            "loaded catalog from KV",
            extra={"extra_data": {"entry_count": len(self._entries)}},
        )

    async def register(self, entry: CatalogEntry) -> None:
        """register tool in catalog and persist to KV.

        if tool already exists, merges endpoints from new entry
        into existing entry and updates tool definition (description,
        schema). if tool is new, stores entry as-is.

        :param entry: catalog entry to register
        :ptype entry: CatalogEntry
        """
        existing = self._entries.get(entry.full_name)
        if existing is not None:
            existing.description = entry.description
            existing.input_schema = entry.input_schema
            existing.output_schema = entry.output_schema
            existing.timeout_seconds = entry.timeout_seconds
            existing.requires_confirmation = entry.requires_confirmation
            for endpoint in entry.endpoints:
                existing.add_endpoint(endpoint)
            target = existing
        else:
            self._entries[entry.full_name] = entry
            target = entry
        if self._kv is not None:
            kv_key = _sanitize_kv_key(target.full_name)
            await self._kv.put(
                kv_key,
                json.dumps(target.to_dict()).encode("utf-8"),
            )
        _logger.info(
            "registered tool in catalog",
            extra={
                "extra_data": {
                    "full_name": target.full_name,
                    "endpoint_count": len(target.endpoints),
                }
            },
        )

    async def deregister(self, full_name: str) -> None:
        """remove tool entirely from catalog and delete from KV.

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
        """remove all endpoints for specified pod.

        removes pod's endpoint from each tool it serves. if a tool
        has no endpoints remaining after removal, removes the entire
        catalog entry. if other endpoints remain, persists the
        updated entry to KV.

        :param pod_id: identifier of pod whose endpoints to remove
        :ptype pod_id: str
        :return: list of full_name values that were affected
        :rtype: list[str]
        """
        affected: list[str] = []
        to_remove: list[str] = []
        for full_name, entry in self._entries.items():
            removed = entry.remove_endpoint(pod_id)
            if not removed:
                continue
            affected.append(full_name)
            if not entry.endpoints:
                to_remove.append(full_name)
            elif self._kv is not None:
                kv_key = _sanitize_kv_key(full_name)
                await self._kv.put(
                    kv_key,
                    json.dumps(entry.to_dict()).encode("utf-8"),
                )
        for full_name in to_remove:
            await self.deregister(full_name)
        result = affected
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
        """list all tools with at least one available endpoint.

        :return: list of available catalog entries
        :rtype: list[CatalogEntry]
        """
        result = [entry for entry in self._entries.values() if entry.status == "available"]
        return result

    def mark_available(self, full_name: str, pod_id: str) -> bool:
        """mark specific endpoint as available.

        :param full_name: composite key as name@version
        :ptype full_name: str
        :param pod_id: identifier of pod whose endpoint to mark
        :ptype pod_id: str
        :return: True if endpoint was found and updated, False otherwise
        :rtype: bool
        """
        entry = self._entries.get(full_name)
        if entry is None:
            return False
        endpoint = entry.get_endpoint(pod_id)
        if endpoint is None:
            return False
        endpoint.status = "available"
        return True

    def mark_unavailable(self, full_name: str, pod_id: str) -> bool:
        """mark specific endpoint as unavailable.

        :param full_name: composite key as name@version
        :ptype full_name: str
        :param pod_id: identifier of pod whose endpoint to mark
        :ptype pod_id: str
        :return: True if endpoint was found and updated, False otherwise
        :rtype: bool
        """
        entry = self._entries.get(full_name)
        if entry is None:
            return False
        endpoint = entry.get_endpoint(pod_id)
        if endpoint is None:
            return False
        endpoint.status = "unavailable"
        return True

    def mark_pod_endpoints_available(self, pod_id: str) -> list[str]:
        """mark all endpoints for specified pod as available.

        scans entire catalog for endpoints belonging to pod_id
        and sets their status to available.

        :param pod_id: identifier of pod whose endpoints to mark
        :ptype pod_id: str
        :return: list of full_name values that were marked available
        :rtype: list[str]
        """
        marked: list[str] = []
        for full_name, entry in self._entries.items():
            endpoint = entry.get_endpoint(pod_id)
            if endpoint is None:
                continue
            endpoint.status = "available"
            marked.append(full_name)
        result = marked
        return result

    def mark_pod_endpoints_unavailable(self, pod_id: str) -> list[str]:
        """mark all endpoints for specified pod as unavailable.

        scans entire catalog for endpoints belonging to pod_id and sets
        their status to unavailable WITHOUT removing them. used to
        quarantine a pod whose heartbeats have lapsed but whose
        consecutive-miss count has not yet reached the eviction
        threshold: routing stops selecting the pod (only 'available'
        endpoints are routable) while the endpoints stay in the catalog
        so a returning heartbeat can revive them via
        :meth:`mark_pod_endpoints_available` -- no full re-registration.

        :param pod_id: identifier of pod whose endpoints to mark
        :ptype pod_id: str
        :return: list of full_name values that were marked unavailable
        :rtype: list[str]
        """
        marked: list[str] = []
        for full_name, entry in self._entries.items():
            endpoint = entry.get_endpoint(pod_id)
            if endpoint is None:
                continue
            endpoint.status = "unavailable"
            marked.append(full_name)
        result = marked
        return result

    async def mark_ready(self, pod_id: str) -> list[str]:
        """transition pending endpoints for pod to available and persist.

        scans catalog for endpoints belonging to pod_id whose status is
        'pending' (freshly registered, probe not yet confirmed). writes
        all updated entries to KV first with the transition applied,
        and only flips in-memory state after every KV write succeeds.

        **partial-failure semantics**: if a mid-loop KV write fails,
        earlier KV entries have already advanced to 'available' while
        in-memory state is still 'pending'. the exception propagates
        and in-memory stays untouched so the caller can safely retry
        the whole transition. next ``load_from_kv`` forces every
        endpoint back to 'unavailable', which reconverges both tiers
        once heartbeat-driven revival runs. this KV-vs-memory drift
        window only opens under a genuine KV outage; steady-state
        operation never enters it.

        endpoints already 'available' or 'unavailable' are skipped;
        heartbeat-driven revival continues to use
        ``mark_pod_endpoints_available``.

        :param pod_id: identifier of pod whose pending endpoints to promote
        :ptype pod_id: str
        :return: list of full_name values that were promoted to available
        :rtype: list[str]
        :raises Exception: if any KV write fails; in-memory state unchanged
        """
        targets: list[tuple[str, CatalogEntry, ToolEndpoint]] = []
        for full_name, entry in self._entries.items():
            endpoint = entry.get_endpoint(pod_id)
            if endpoint is None:
                continue
            if endpoint.status != "pending":
                continue
            targets.append((full_name, entry, endpoint))

        if self._kv is not None:
            for full_name, entry, endpoint in targets:
                kv_key = _sanitize_kv_key(full_name)
                projected_endpoints = [_replace_endpoint_status(ep, pod_id, "available") for ep in entry.endpoints]
                projected_entry = CatalogEntry(
                    tool_name=entry.tool_name,
                    tool_version=entry.tool_version,
                    full_name=entry.full_name,
                    description=entry.description,
                    input_schema=entry.input_schema,
                    output_schema=entry.output_schema,
                    timeout_seconds=entry.timeout_seconds,
                    requires_confirmation=entry.requires_confirmation,
                    endpoints=projected_endpoints,
                    date_registered=entry.date_registered,
                )
                await self._kv.put(
                    kv_key,
                    json.dumps(projected_entry.to_dict()).encode("utf-8"),
                )

        promoted: list[str] = []
        for full_name, _entry, endpoint in targets:
            endpoint.status = "available"
            promoted.append(full_name)
        result = promoted
        return result
