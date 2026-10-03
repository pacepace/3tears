"""tool catalog for centralized tool registration and discovery.

maintains in-memory catalog of registered tools with NATS KV
persistence. supports multiple endpoints per tool for horizontal
scaling with load-balanced routing.

**each endpoint is one pod's COPY of a tool and carries that pod's own definition** --
description, input and output schema, timeout, confirmation gate. no registration can change
another pod's copy, and :meth:`CatalogEntry.select_copies` is the one function that decides
what a caller is shown and where its call may be routed, for discovery and the proxy alike.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import UUID

from threetears.core.serialization import json_datetime
from threetears.nats import Subjects
from threetears.observe import get_logger
from threetears.registry.config import get_definition_ttl
from threetears.registry.routing import endpoints_callable_by

__all__ = [
    "PERSISTED_SHAPE",
    "AnnouncedDefinition",
    "CatalogEntry",
    "CopySelection",
    "CopyStatus",
    "ToolCatalog",
    "ToolDefinition",
    "ToolEndpoint",
]

_logger = get_logger(__name__)

#: the shape every catalog entry is written to KV in. an entry carrying no ``shape`` predates
#: per-copy definitions and is translated once, by :meth:`ToolCatalog.load_from_kv`, rather than
#: read by a second code path.
PERSISTED_SHAPE = 2

#: the entry-level definition fields the pre-shape-2 KV value carried. named once so the
#: translation drops exactly these and a test can assert none survive.
_LEGACY_ENTRY_DEFINITION_FIELDS = (
    "description",
    "input_schema",
    "output_schema",
    "timeout_seconds",
    "requires_confirmation",
)


def _canonical_json(value: Any) -> str:
    """compact JSON with recursively sorted keys, the one spelling a digest is taken over.

    :param value: a JSON-serializable value
    :ptype value: Any
    :return: the canonical text
    :rtype: str
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_hex(text: str) -> str:
    """hex SHA-256 of ``text`` encoded as UTF-8.

    :param text: the text to digest
    :ptype text: str
    :return: the lowercase hex digest
    :rtype: str
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _translate_persisted_entry(data: dict[str, Any]) -> dict[str, Any]:
    """bring one persisted catalog value to the current shape; the ONLY reader of older shapes.

    **The old entry-level definition is dropped, not handed to the copies.** Before shape 2 an
    entry held one description, schema, timeout and confirmation flag, rewritten by whichever
    pod registered last -- which may have been a pod that had no business defining the tool.
    That value's provenance is unknown, so giving it to any copy would launder a possible
    stray's overwrite into a copy's own definition. Each endpoint comes back with NO
    definition instead: shown to nobody and routed to by nobody until its pod's next manifest
    re-announces what it actually serves, which a live pod does within one heartbeat.

    :param data: the value as read from KV
    :ptype data: dict[str, Any]
    :return: the value in the current shape
    :rtype: dict[str, Any]
    """
    if data.get("shape") == PERSISTED_SHAPE:
        return data
    dropped = sorted(name for name in _LEGACY_ENTRY_DEFINITION_FIELDS if name in data)
    endpoints = [
        {
            "pod_id": endpoint["pod_id"],
            "status": endpoint.get("status", "unavailable"),
            "date_last_heartbeat": endpoint["date_last_heartbeat"],
            "verified_publisher": False,
            "definitions": [],
        }
        for endpoint in data.get("endpoints", [])
    ]
    _logger.info(
        "translated a catalog entry persisted before per-copy definitions; its entry-level "
        "definition is dropped and each copy is shown to nobody until its pod announces again",
        extra={
            "extra_data": {
                "full_name": data.get("full_name"),
                "dropped_fields": dropped,
                "endpoint_count": len(endpoints),
            }
        },
    )
    return {
        "shape": PERSISTED_SHAPE,
        "tool_name": data["tool_name"],
        "tool_version": data["tool_version"],
        "full_name": data["full_name"],
        "endpoints": endpoints,
        "date_registered": data["date_registered"],
    }


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
    status. every other field -- the copy's announced definitions and
    whether a verified publisher registered it -- is carried unchanged,
    so the snapshot never persists a copy stripped of what it serves.

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
    return replace(endpoint, status=projected_status, definitions=dict(endpoint.definitions))


def _default_ttl() -> timedelta:
    """the configured definition time-to-live.

    :return: how long an announced definition stays live without re-announcement
    :rtype: timedelta
    """
    return timedelta(seconds=get_definition_ttl())


@dataclass(frozen=True)
class ToolDefinition:
    """what one pod says a tool IS: the five fields a caller reads before calling it.

    A copy of a tool is one pod's endpoint, and each copy keeps the definition its own pod
    announced. Nothing another pod announces can change it.

    :param description: human-readable tool description
    :ptype description: str
    :param input_schema: JSON Schema for the tool's input
    :ptype input_schema: dict[str, Any]
    :param output_schema: JSON Schema for the tool's output, or ``None``
    :ptype output_schema: dict[str, Any] | None
    :param timeout_seconds: the tool's declared execution ceiling, or ``None`` for the default
    :ptype timeout_seconds: float | None
    :param requires_confirmation: whether a call must be approved by a person first
    :ptype requires_confirmation: bool
    """

    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    timeout_seconds: float | None = None
    requires_confirmation: bool = False

    def __hash__(self) -> int:
        """hash by :attr:`digest`, which two equal definitions always share.

        the generated field hash would fail on the schema dicts, so two copies' definitions could
        not be collected into a set.

        :return: the hash
        :rtype: int
        """
        return hash(self.digest)

    @property
    def schema_digest(self) -> str:
        """hex SHA-256 of the canonical input schema.

        What a caller hands back to say "route me only where this schema is served". Key order
        does not change it; any change to the schema does.

        :return: the input-schema digest
        :rtype: str
        """
        return _sha256_hex(_canonical_json(self.input_schema))

    @property
    def digest(self) -> str:
        """hex SHA-256 over all five fields; the identity of one definition.

        :return: the definition digest
        :rtype: str
        """
        return _sha256_hex(_canonical_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        """serialize for KV storage.

        :return: the five fields
        :rtype: dict[str, Any]
        """
        return {
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "timeout_seconds": self.timeout_seconds,
            "requires_confirmation": self.requires_confirmation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolDefinition:
        """deserialize from KV storage.

        :param data: the five fields
        :ptype data: Mapping[str, Any]
        :return: the definition
        :rtype: ToolDefinition
        """
        return cls(
            description=data["description"],
            input_schema=data["input_schema"],
            output_schema=data["output_schema"],
            timeout_seconds=data["timeout_seconds"],
            requires_confirmation=data["requires_confirmation"],
        )


@dataclass
class AnnouncedDefinition:
    """one definition one pod announced, and when.

    :param definition: the definition
    :ptype definition: ToolDefinition
    :param first_announced: when THIS registry first saw the pod announce it; decides which
        definition is shown when copies differ
    :ptype first_announced: datetime
    :param last_announced: when the pod last announced it; decides whether it is still live
    :ptype last_announced: datetime
    """

    definition: ToolDefinition
    first_announced: datetime
    last_announced: datetime

    def is_live(self, now: datetime, ttl: timedelta) -> bool:
        """whether the pod has re-announced this definition within ``ttl``.

        :param now: the reference instant
        :ptype now: datetime
        :param ttl: the liveness window
        :ptype ttl: timedelta
        :return: true while re-announced within the window
        :rtype: bool
        """
        return now - self.last_announced <= ttl

    def to_dict(self) -> dict[str, Any]:
        """serialize for KV storage.

        :return: the definition and both instants
        :rtype: dict[str, Any]
        """
        return {
            "definition": self.definition.to_dict(),
            "first_announced": json_datetime(self.first_announced, field="first_announced"),
            "last_announced": json_datetime(self.last_announced, field="last_announced"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AnnouncedDefinition:
        """deserialize from KV storage.

        :param data: the serialized announcement
        :ptype data: Mapping[str, Any]
        :return: the announcement
        :rtype: AnnouncedDefinition
        """
        return cls(
            definition=ToolDefinition.from_dict(data["definition"]),
            first_announced=datetime.fromisoformat(data["first_announced"]),
            last_announced=datetime.fromisoformat(data["last_announced"]),
        )


@dataclass
class ToolEndpoint:
    """single pod endpoint serving a tool: one COPY of it, with the definitions that pod announced.

    tracks liveness and in-flight call count for
    load-balanced routing across multiple pods. status values
    follow a three-phase lifecycle: 'pending' on initial
    registration (before reachability probe round-trips),
    'available' after probe confirms reachability or heartbeat
    refreshes liveness, and 'unavailable' after missed
    heartbeats. calls route only to 'available' endpoints.

    ``definitions`` is keyed by :attr:`ToolDefinition.digest`. One pod id can carry more than
    one live definition, because every replica of a Deployment shares its pod id and queue
    group, so mid-rollout the old and new replicas both announce under it.

    :param pod_id: identifier of pod serving this tool
    :ptype pod_id: str
    :param status: availability status ('pending', 'available', or 'unavailable')
    :ptype status: str
    :param in_flight: number of currently in-flight calls to this endpoint
    :ptype in_flight: int
    :param date_last_heartbeat: timestamp of last heartbeat from this pod
    :ptype date_last_heartbeat: datetime
    :param definitions: the definitions this pod announced, by digest
    :ptype definitions: dict[str, AnnouncedDefinition]
    :param verified_publisher: whether the registration that last wrote this copy came from a
        publisher whose identity was verified; a pod id that has registered verified may not be
        re-registered by an unverified manifest
    :ptype verified_publisher: bool
    """

    pod_id: str
    status: str = "pending"
    in_flight: int = 0
    date_last_heartbeat: datetime = field(default_factory=lambda: datetime.now(UTC))
    definitions: dict[str, AnnouncedDefinition] = field(default_factory=dict)
    verified_publisher: bool = False

    def announce(self, definition: ToolDefinition, now: datetime) -> None:
        """record that this pod announced ``definition`` at ``now``.

        a definition already held keeps its ``first_announced`` -- re-announcement on every
        heartbeat refreshes liveness and nothing else -- so which definition is shown does not
        churn while nothing changes.

        :param definition: the announced definition
        :ptype definition: ToolDefinition
        :param now: the announcement instant
        :ptype now: datetime
        :return: nothing
        :rtype: None
        """
        digest = definition.digest
        held = self.definitions.get(digest)
        if held is None:
            self.definitions[digest] = AnnouncedDefinition(
                definition=definition, first_announced=now, last_announced=now
            )
        elif now > held.last_announced:
            held.last_announced = now

    def merge_announcements(self, incoming: Mapping[str, AnnouncedDefinition]) -> None:
        """fold another view of this copy's announcements into this one.

        :param incoming: announcements keyed by digest
        :ptype incoming: Mapping[str, AnnouncedDefinition]
        :return: nothing
        :rtype: None
        """
        for digest, announcement in incoming.items():
            held = self.definitions.get(digest)
            if held is None:
                self.definitions[digest] = AnnouncedDefinition(
                    definition=announcement.definition,
                    first_announced=announcement.first_announced,
                    last_announced=announcement.last_announced,
                )
            else:
                held.first_announced = min(held.first_announced, announcement.first_announced)
                held.last_announced = max(held.last_announced, announcement.last_announced)

    def live_definitions(self, now: datetime, ttl: timedelta) -> list[AnnouncedDefinition]:
        """the definitions re-announced within ``ttl`` of ``now``.

        :param now: the reference instant
        :ptype now: datetime
        :param ttl: the liveness window
        :ptype ttl: timedelta
        :return: live announcements, in insertion order
        :rtype: list[AnnouncedDefinition]
        """
        return [announcement for announcement in self.definitions.values() if announcement.is_live(now, ttl)]

    def prune_definitions(self, now: datetime, ttl: timedelta) -> list[str]:
        """drop every definition not re-announced within ``ttl``.

        :param now: the reference instant
        :ptype now: datetime
        :param ttl: the liveness window
        :ptype ttl: timedelta
        :return: digests dropped
        :rtype: list[str]
        """
        dropped = [digest for digest, held in self.definitions.items() if not held.is_live(now, ttl)]
        for digest in dropped:
            del self.definitions[digest]
        return dropped

    def to_dict(self) -> dict[str, Any]:
        """serialize endpoint to dictionary for KV storage.

        :return: dictionary representation of endpoint
        :rtype: dict[str, Any]
        """
        result = {
            "pod_id": self.pod_id,
            "status": self.status,
            "date_last_heartbeat": json_datetime(self.date_last_heartbeat, field="date_last_heartbeat"),
            "verified_publisher": self.verified_publisher,
            "definitions": [announcement.to_dict() for announcement in self.definitions.values()],
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolEndpoint:
        """deserialize endpoint from dictionary loaded from KV storage.

        in_flight is always reset to zero on load since pending calls
        cannot survive a restart. each definition is re-keyed by its digest as computed NOW,
        so a stored key can never name a definition other than the one beside it.

        :param data: dictionary representation of endpoint
        :ptype data: dict[str, Any]
        :return: reconstructed endpoint
        :rtype: ToolEndpoint
        """
        announcements = [AnnouncedDefinition.from_dict(item) for item in data["definitions"]]
        result = cls(
            pod_id=data["pod_id"],
            status=data.get("status", "unavailable"),
            in_flight=0,
            date_last_heartbeat=datetime.fromisoformat(data["date_last_heartbeat"]),
            definitions={announcement.definition.digest: announcement for announcement in announcements},
            verified_publisher=data["verified_publisher"],
        )
        return result


class CopyStatus(StrEnum):
    """the state of one pod's OWN copy of a tool, as a polling pod reads it.

    :cvar AVAILABLE: probed, and holding a live definition -- routable
    :cvar PENDING: registered, reachability probe not yet confirmed
    :cvar UNAVAILABLE: known, but not routable: heartbeats lapsed or no live definition
    :cvar ABSENT: the registry holds no copy under that pod id -- never registered, refused,
        or withdrawn
    """

    AVAILABLE = "available"
    PENDING = "pending"
    UNAVAILABLE = "unavailable"
    ABSENT = "absent"


@dataclass(frozen=True)
class CopySelection:
    """what one caller sees of one tool, decided once by :meth:`CatalogEntry.select_copies`.

    :param visible: copies the caller may be routed to that are available and hold a live
        definition
    :ptype visible: tuple[ToolEndpoint, ...]
    :param requires_confirmation: whether ANY live definition on ANY visible copy requires a
        person's approval; one ungated copy cannot switch the gate off
    :ptype requires_confirmation: bool
    :param own_tier: whether the caller is being served from its own in-process copies rather
        than from the copies that serve everyone
    :ptype own_tier: bool
    :param shown: the definition the caller is shown, or ``None`` when nothing is visible
    :ptype shown: ToolDefinition | None
    :param routable: copies in the tier that serve the routed input schema
    :ptype routable: tuple[ToolEndpoint, ...]
    :param routed_definitions: for each routable copy, by pod id, the definition a call to it
        runs under -- its most recently announced live definition of the routed schema
    :ptype routed_definitions: Mapping[str, ToolDefinition]
    """

    visible: tuple[ToolEndpoint, ...]
    requires_confirmation: bool
    own_tier: bool
    shown: ToolDefinition | None
    routable: tuple[ToolEndpoint, ...]
    routed_definitions: Mapping[str, ToolDefinition]

    @property
    def definition_changed(self) -> bool:
        """whether copies are visible but none serves the schema the caller named.

        :return: true when the caller's view of the tool is stale
        :rtype: bool
        """
        return bool(self.visible) and not self.routable


@dataclass
class CatalogEntry:
    """single tool entry in catalog: one ``name@version`` and every copy serving it.

    the entry holds NO definition of its own. each copy (endpoint) carries the definitions its
    pod announced, and :meth:`select_copies` decides what any one caller is shown and where it
    may be routed.

    :param tool_name: namespaced tool name (e.g., 'threetears.calculator')
    :ptype tool_name: str
    :param tool_version: semver-compatible version string
    :ptype tool_version: str
    :param full_name: composite key as name@version
    :ptype full_name: str
    :param endpoints: list of pod endpoints serving this tool
    :ptype endpoints: list[ToolEndpoint]
    :param date_registered: timestamp when tool was first registered
    :ptype date_registered: datetime
    """

    tool_name: str
    tool_version: str
    full_name: str
    endpoints: list[ToolEndpoint] = field(default_factory=list)
    date_registered: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def status(self) -> str:
        """aggregate availability status from endpoints, for persistence and observability only.

        pending endpoints (awaiting probe confirmation) do not
        count toward availability -- only fully-confirmed endpoints
        do. an entry whose endpoints are all pending aggregates
        to 'unavailable'.

        **this ignores the caller, so it does not say a tool is routable for anyone.** an agent's
        in-process endpoint serves only that agent, so an entry can be 'available' here while
        every caller but one is refused. whether a tool is available TO A CALLER is
        :meth:`available_to`; the catalog's listing of what a caller may use is
        :meth:`ToolCatalog.list_available`.

        :return: 'available' if any endpoint is available, 'unavailable' otherwise
        :rtype: str
        """
        for endpoint in self.endpoints:
            if endpoint.status == "available":
                return "available"
        return "unavailable"

    def endpoints_for(self, caller_id: UUID | None) -> list[ToolEndpoint]:
        """this entry's endpoints one caller may be routed to, whatever their status.

        the catalog's single door onto :func:`~threetears.registry.routing.endpoints_callable_by`,
        so routing, discovery and anything that lists tools ask the same question the same way.

        :param caller_id: the calling agent's id, or ``None`` when the caller names no agent,
            which leaves it only the Tool Pod endpoints
        :ptype caller_id: UUID | None
        :return: the caller's endpoints, in catalog order
        :rtype: list[ToolEndpoint]
        """
        return endpoints_callable_by(self.endpoints, caller_id)

    def select_copies(
        self,
        caller_id: UUID | None,
        schema_digest: str | None = None,
        *,
        now: datetime | None = None,
        ttl: timedelta | None = None,
    ) -> CopySelection:
        """the ONE decision of what a caller sees of this tool and where it may be routed.

        Discovery and the call proxy both ask this, so what an agent is shown and where its call
        lands cannot disagree.

        1. **visible** -- the caller's endpoints (:meth:`endpoints_for`) that are available and
           hold at least one live definition;
        2. **requires_confirmation** -- OR across every live definition on every visible copy;
        3. **tier** -- the caller's own in-process copies when it has any visible, else the
           copies that serve everyone. An agent's own tool answers from its own state, so it is
           preferred over a shared copy of the same name;
        4. **shown** -- within the tier, the live definition first announced most recently, the
           lexically smaller digest breaking a tie so every replica picks the same one;
        5. **routable** -- the tier's copies holding a live definition of the routed schema:
           ``schema_digest`` when the caller names one, else the shown definition's.

        :param caller_id: the calling principal, or ``None`` when it names no agent
        :ptype caller_id: UUID | None
        :param schema_digest: the input-schema digest the caller was shown, to route only to
            copies still serving it; ``None`` routes by the shown definition
        :ptype schema_digest: str | None
        :param now: the reference instant; the wall clock when omitted
        :ptype now: datetime | None
        :param ttl: the liveness window; ``THREETEARS_REGISTRY_DEFINITION_TTL`` when omitted
        :ptype ttl: timedelta | None
        :return: the selection
        :rtype: CopySelection
        """
        at = now if now is not None else datetime.now(UTC)
        window = ttl if ttl is not None else _default_ttl()
        live_by_pod: dict[str, list[AnnouncedDefinition]] = {}
        visible: list[ToolEndpoint] = []
        for endpoint in self.endpoints_for(caller_id):
            live = endpoint.live_definitions(at, window)
            if endpoint.status == "available" and live:
                visible.append(endpoint)
                live_by_pod[endpoint.pod_id] = live
        requires_confirmation = any(
            announcement.definition.requires_confirmation for live in live_by_pod.values() for announcement in live
        )
        own = [endpoint for endpoint in visible if Subjects.agent_inprocess_owner_id(endpoint.pod_id) is not None]
        tier = own if own else visible
        tier_announcements = [announcement for endpoint in tier for announcement in live_by_pod[endpoint.pod_id]]
        shown: ToolDefinition | None = None
        if tier_announcements:
            newest = max(announcement.first_announced for announcement in tier_announcements)
            shown = min(
                (
                    announcement.definition
                    for announcement in tier_announcements
                    if announcement.first_announced == newest
                ),
                key=lambda candidate: candidate.digest,
            )
        routed_schema = schema_digest if schema_digest is not None else (shown.schema_digest if shown else None)
        routable: list[ToolEndpoint] = []
        routed_definitions: dict[str, ToolDefinition] = {}
        if routed_schema is not None:
            for endpoint in tier:
                matching = [
                    announcement
                    for announcement in live_by_pod[endpoint.pod_id]
                    if announcement.definition.schema_digest == routed_schema
                ]
                if matching:
                    routable.append(endpoint)
                    newest_match = max(
                        matching,
                        key=lambda announcement: (announcement.first_announced, announcement.definition.digest),
                    )
                    routed_definitions[endpoint.pod_id] = newest_match.definition
        return CopySelection(
            visible=tuple(visible),
            requires_confirmation=requires_confirmation,
            own_tier=bool(own),
            shown=shown,
            routable=tuple(routable),
            routed_definitions=routed_definitions,
        )

    def available_to(self, caller_id: UUID | None) -> bool:
        """whether one caller could be routed to this tool right now.

        :param caller_id: the calling agent's id, or ``None`` when the caller names no agent
        :ptype caller_id: UUID | None
        :return: true when :meth:`select_copies` leaves the caller a routable copy; a pending one,
            or one with no live definition, is not routable
        :rtype: bool
        """
        return bool(self.select_copies(caller_id).routable)

    def copy_status(
        self,
        pod_id: str,
        *,
        now: datetime | None = None,
        ttl: timedelta | None = None,
    ) -> CopyStatus:
        """the state of ONE pod's own copy, whatever any caller would be shown.

        :param pod_id: the pod asking about its own copy
        :ptype pod_id: str
        :param now: the reference instant; the wall clock when omitted
        :ptype now: datetime | None
        :param ttl: the liveness window; the configured one when omitted
        :ptype ttl: timedelta | None
        :return: the copy's state
        :rtype: CopyStatus
        """
        at = now if now is not None else datetime.now(UTC)
        window = ttl if ttl is not None else _default_ttl()
        endpoint = self.get_endpoint(pod_id)
        result = CopyStatus.ABSENT
        if endpoint is not None:
            if endpoint.status == "pending":
                result = CopyStatus.PENDING
            elif endpoint.status == "available" and endpoint.live_definitions(at, window):
                result = CopyStatus.AVAILABLE
            else:
                result = CopyStatus.UNAVAILABLE
        return result

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
        """serialize entry to dictionary for KV storage, in the current :data:`PERSISTED_SHAPE`.

        :return: dictionary representation of catalog entry
        :rtype: dict[str, Any]
        """
        result = {
            "shape": PERSISTED_SHAPE,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "full_name": self.full_name,
            "endpoints": [ep.to_dict() for ep in self.endpoints],
            "date_registered": json_datetime(self.date_registered, field="date_registered"),
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CatalogEntry:
        """deserialize entry from dictionary in the current :data:`PERSISTED_SHAPE`.

        :param data: dictionary representation of catalog entry
        :ptype data: dict[str, Any]
        :return: reconstructed catalog entry
        :rtype: CatalogEntry
        :raises ValueError: when ``data`` is not in the current shape; an older value is brought
            forward by :meth:`ToolCatalog.load_from_kv` and nowhere else
        """
        if data.get("shape") != PERSISTED_SHAPE:
            raise ValueError(
                f"catalog entry {data.get('full_name')!r} is persisted in shape {data.get('shape')!r}, "
                f"expected {PERSISTED_SHAPE}; an older value is translated only by ToolCatalog.load_from_kv"
            )
        endpoints = [ToolEndpoint.from_dict(ep_data) for ep_data in data["endpoints"]]
        result = cls(
            tool_name=data["tool_name"],
            tool_version=data["tool_version"],
            full_name=data["full_name"],
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
        subsequent write operations. an entry this catalog already holds
        is kept, never replaced: a load that lands after the registry began
        serving -- its bucket was unreachable at start -- must not overwrite
        a live registration with a persisted copy marked unavailable.

        **the one place an older persisted shape is read.** every value passes through
        :func:`_translate_persisted_entry` before it becomes a :class:`CatalogEntry`; an entry
        written before per-copy definitions loses its entry-level definition there, and its
        copies are shown to nobody until their pods announce again.

        :param kv: NATS KV store instance
        :ptype kv: Any
        """
        self._kv = kv
        try:
            keys = await kv.keys()
        except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- an empty bucket answers "no keys" by raising, and the raw client's error type is not importable past the NATS wrapper; the catalog starts empty and fills from the next heartbeat's manifests, and the reason is logged so a real listing failure is not mistaken for an empty bucket
            _logger.warning(
                "catalog KV listed no keys; starting with an empty catalog that fills from the next manifests",
                extra={"extra_data": {"reason": type(exc).__name__, "detail": str(exc)}},
            )
            keys = []
        for key in keys:
            kv_entry = await kv.get(key)
            data = json.loads(kv_entry.value.decode("utf-8"))
            entry = CatalogEntry.from_dict(_translate_persisted_entry(data))
            for endpoint in entry.endpoints:
                endpoint.status = "unavailable"
            self._entries.setdefault(entry.full_name, entry)
        _logger.info(
            "loaded catalog from KV",
            extra={"extra_data": {"entry_count": len(self._entries)}},
        )

    async def restore_to_kv(self, kv: Any) -> list[str]:
        """bind ``kv`` for every later write and write every entry this catalog holds into it.

        what a bucket that came back empty after a NATS restart needs: the in-memory catalog is the
        one the registry routes from, and the bucket is its persisted copy, read only to warm-load a
        starting registry. each entry is written the way a registration writes it; one that fails
        is logged and named in the result, and the rest are still written.

        :param kv: raw nats-py KeyValue handle of the declared bucket
        :ptype kv: Any
        :return: the full names of the entries that could not be written; empty when all were
        :rtype: list[str]
        """
        self._kv = kv
        failed: list[str] = []
        for full_name in list(self._entries):
            # read at the moment of writing, not from a snapshot: a tool deregistered while an earlier
            # write was awaited has had its key deleted, and writing it back would advertise a tool
            # no replica holds to every registry that warm-loads the bucket.
            entry = self._entries.get(full_name)
            if entry is None:
                continue
            try:
                await self._persist(entry)
            except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- one entry's write must not strand the rest; each failure is logged and returned, and the caller retries
                failed.append(full_name)
                _logger.warning(
                    "writing a catalog entry back into its bucket failed",
                    extra={"extra_data": {"full_name": full_name, "error": f"{type(exc).__name__}: {exc}"}},
                )
        _logger.info(
            "catalog written back into its bucket",
            extra={"extra_data": {"entry_count": len(self._entries), "failed": len(failed)}},
        )
        return failed

    async def _persist(self, entry: CatalogEntry) -> None:
        """write one entry to KV when a bucket is bound.

        :param entry: the entry to write
        :ptype entry: CatalogEntry
        :return: nothing
        :rtype: None
        """
        if self._kv is not None:
            await self._kv.put(
                _sanitize_kv_key(entry.full_name),
                json.dumps(entry.to_dict()).encode("utf-8"),
            )

    async def register(self, entry: CatalogEntry) -> None:
        """register tool in catalog and persist to KV, merging PER COPY.

        each endpoint in ``entry`` is folded into the held entry's copy for the same pod id: its
        status, liveness and publisher mark are taken from the incoming copy, and its announced
        definitions are merged into that copy's own. **No other copy is touched** -- one pod's
        registration cannot change what another pod's copy says the tool is. definitions on the
        registering copy that have not been re-announced within the TTL are pruned; another
        copy's stale definitions are that pod's to refresh, and are left alone.

        :param entry: catalog entry to register
        :ptype entry: CatalogEntry
        """
        now = datetime.now(UTC)
        ttl = _default_ttl()
        existing = self._entries.get(entry.full_name)
        if existing is None:
            for endpoint in entry.endpoints:
                endpoint.prune_definitions(now, ttl)
            self._entries[entry.full_name] = entry
            target = entry
        else:
            for incoming in entry.endpoints:
                held = existing.get_endpoint(incoming.pod_id)
                if held is None:
                    incoming.prune_definitions(now, ttl)
                    existing.add_endpoint(incoming)
                    continue
                # in_flight is NOT taken from the incoming copy: the proxy counts calls on the held
                # object, and a registration landing mid-call must not reset that count.
                held.status = incoming.status
                held.date_last_heartbeat = incoming.date_last_heartbeat
                held.verified_publisher = incoming.verified_publisher
                held.merge_announcements(incoming.definitions)
                held.prune_definitions(now, ttl)
            target = existing
        await self._persist(target)
        _logger.info(
            "registered tool in catalog",
            extra={
                "extra_data": {
                    "full_name": target.full_name,
                    "endpoint_count": len(target.endpoints),
                }
            },
        )

    async def remove_copy(self, full_name: str, pod_id: str) -> bool:
        """withdraw ONE pod's copy of one tool, leaving every other copy as it was.

        used when a registration refuses a tool its publisher previously held: the refused
        publisher's prior copy goes, and the incumbent it was competing with stays. an entry left
        with no copy is removed entirely.

        :param full_name: composite key as name@version
        :ptype full_name: str
        :param pod_id: the pod whose copy is withdrawn
        :ptype pod_id: str
        :return: whether a copy was removed
        :rtype: bool
        """
        entry = self._entries.get(full_name)
        removed = entry is not None and entry.remove_endpoint(pod_id)
        if entry is not None and removed:
            if entry.endpoints:
                await self._persist(entry)
            else:
                await self.deregister(full_name)
            _logger.info(
                "withdrew one pod's copy of a tool",
                extra={"extra_data": {"full_name": full_name, "pod_id": pod_id}},
            )
        return removed

    def pod_has_verified_copy(self, pod_id: str) -> bool:
        """whether any copy under ``pod_id`` was last registered by a verified publisher.

        :param pod_id: the pod id to look for
        :ptype pod_id: str
        :return: true when such a copy is held
        :rtype: bool
        """
        return any(
            endpoint.verified_publisher
            for entry in self._entries.values()
            for endpoint in entry.endpoints
            if endpoint.pod_id == pod_id
        )

    async def deregister(self, full_name: str) -> bool:
        """remove tool entirely from catalog and delete from KV.

        :param full_name: composite key as name@version
        :ptype full_name: str
        :return: ``True`` when the tool is gone everywhere; ``False`` when only the LOCAL entry was
            dropped and the shared KV copy survived. A caller that ignores this gets the previous
            behaviour, but it can now retry -- which matters because ``load_from_kv`` repopulates
            ``_entries`` from KV, so a node that fails the KV delete re-learns the tool it just
            deregistered on its next restart, and every other node never stopped advertising it.
        :rtype: bool
        """
        self._entries.pop(full_name, None)
        kv_deleted = True
        if self._kv is not None:
            try:
                kv_key = _sanitize_kv_key(full_name)
                await self._kv.delete(kv_key)
            except Exception as exc:  # noqa: BLE001 -- the local drop above already succeeded
                # The local entry is gone but the shared KV copy is not, so every other node still
                # discovers this tool. Deregistration is not complete, and saying so at info would
                # have read as a clean removal.
                kv_deleted = False
                _logger.warning(
                    "deregistered tool locally but failed to remove its shared KV entry",
                    extra={"extra_data": {"full_name": full_name, "error": str(exc)}},
                )
        _logger.info(
            "deregistered tool from catalog",
            extra={"extra_data": {"full_name": full_name, "kv_deleted": kv_deleted}},
        )
        return kv_deleted

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
        #: tools whose SHARED copy still advertises this pod after the sweep.
        stale_shared: list[str] = []
        for full_name, entry in self._entries.items():
            removed = entry.remove_endpoint(pod_id)
            if not removed:
                continue
            affected.append(full_name)
            if not entry.endpoints:
                to_remove.append(full_name)
            elif self._kv is not None:
                kv_key = _sanitize_kv_key(full_name)
                try:
                    await self._kv.put(
                        kv_key,
                        json.dumps(entry.to_dict()).encode("utf-8"),
                    )
                except Exception as exc:  # noqa: BLE001 -- one tool's KV write must not strand the rest
                    # The in-memory endpoint is ALREADY removed at this point. Letting this
                    # propagate abandoned the loop mid-way with `_entries` mutated and `affected`
                    # discarded, so the caller learned nothing about the pods that had been
                    # processed -- the same partial-failure shape as the deregister below, in the
                    # same method. The shared copy keeps advertising this endpoint until a later
                    # write succeeds; that is worth a warning, not a lost sweep.
                    stale_shared.append(full_name)
                    _logger.warning(
                        "removed a pod's endpoint locally but failed to update its shared KV entry",
                        extra={"extra_data": {"full_name": full_name, "pod_id": pod_id, "error": str(exc)}},
                    )
        for full_name in to_remove:
            if not await self.deregister(full_name):
                stale_shared.append(full_name)
        if stale_shared:
            _logger.warning(
                "pod deregistration left shared KV entries stale",
                extra={"extra_data": {"pod_id": pod_id, "full_names": sorted(set(stale_shared))}},
            )
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

    def list_available(self, caller_id: UUID | None) -> list[CatalogEntry]:
        """list every tool one caller could be routed to right now.

        the caller is required: availability depends on who asks, because an agent's
        in-process endpoint serves only that agent. a listing that did not ask would offer one
        agent's in-process tools to every other agent as if callable.

        :param caller_id: the calling agent's id, or ``None`` when the caller names no agent,
            which lists only tools a Tool Pod serves
        :ptype caller_id: UUID | None
        :return: the entries available to that caller
        :rtype: list[CatalogEntry]
        """
        result = [entry for entry in self._entries.values() if entry.available_to(caller_id)]
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
