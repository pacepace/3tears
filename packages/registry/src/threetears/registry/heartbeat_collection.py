"""L1+L2 HeartbeatCollection for per-pod tool-pod liveness state.

namespace-task-01 phase 8.5l-3 retires :class:`HeartbeatMonitor`'s
bespoke :class:`SQLiteBackend` wrapper in favour of a proper
:class:`BaseCollection`-backed surface. the surface is L1+L2 only:

- L1 (SQLite in-memory, registry-process-local) for fast in-process
  lookups. the L1 table ``pod_heartbeats`` is defined in
  :mod:`threetears.registry.l1_cache`.
- L2 (NATS KV, shared bucket) for cross-registry coherence. when a
  registry process marks a pod unresponsive and writes the updated
  row, peer registries receive the
  :data:`~threetears.core.collections.registry.INVALIDATION_SUBJECT`
  envelope on the shared NATS connection and evict their own L1
  copy. on the next local access the collection pulls the fresh row
  from L2 into L1.
- L3 is intentionally unwired. the ``l3_pool`` attribute is
  deliberately left ``None``; the three-tier base class's
  :meth:`fetch_from_postgres` / :meth:`save_to_postgres` /
  :meth:`delete_from_postgres` are overridden to raise loudly so
  accidental mis-wiring surfaces on the first call. heartbeats are
  transient by construction -- a restarted pod re-emits its
  heartbeat within seconds -- so durable L3 storage of every
  heartbeat would be ops cost for no operational benefit.

the orchestration around the persistent state (NATS subscription,
health-check loop, tool-catalog invalidation) lives in
:class:`~threetears.registry.health.HeartbeatSubscriber` and consumes
this Collection via constructor injection. split out under phase
8.5l-3 so persistent-state concerns (this module) and orchestration
concerns (``health.py``) each have one home.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from threetears.core.collections.base import BaseCollection
from threetears.core.collections.flush import WriteBuffer
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig
from threetears.observe import get_logger

from threetears.registry.entities import HeartbeatEntity

__all__ = ["HeartbeatCollection"]

log = get_logger(__name__)


class HeartbeatCollection(BaseCollection[HeartbeatEntity]):
    """L1+L2 :class:`BaseCollection` for tool-pod heartbeat state.

    keys are opaque pod ids; values are :class:`HeartbeatEntity`
    rows carrying ``date_last_heartbeat`` / ``tools`` / ``status`` /
    ``consecutive_misses``. the Collection has NO L3 pool binding:
    every method that would otherwise cross the L3 border is either
    overridden to operate on L1+L2 only or raises to surface
    mis-wiring.

    cross-registry coherence is on the L2 path: every write publishes
    the invalidation envelope
    (:data:`~threetears.core.collections.registry.INVALIDATION_SUBJECT`,
    ``{"table": "pod_heartbeats", "ids": [<pod_id>]}``) so peer
    registry processes evict their L1 copy. the base class ``get``
    path then refills L1 from L2 on the next read.
    """

    primary_key_column: str = "pod_id"

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        nats_client: Any = None,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        """wire the Collection against the registry and force L3 off.

        :param registry: shared collection registry; ``l1_backend`` is
            resolved through
            :meth:`CollectionRegistry.get_l1_backend` identically to
            every other pod-side Collection. the registry's default
            ``l3_pool`` is deliberately ignored: heartbeats NEVER
            round-trip to L3 regardless of registry configuration.
        :ptype registry: CollectionRegistry
        :param config: core configuration governing flush behaviour;
            the :class:`BaseCollection` write path is short-circuited
            before the flush strategy is consulted so the value is
            effectively ignored, but the constructor signature matches
            the sibling Collections for DI symmetry
        :ptype config: CoreConfig
        :param nats_client: NATS client for L2 cache coherence. may
            be ``None`` when the Collection is running L1-only (unit
            tests, single-registry deployments); every L2 hop guards
            on ``self._nats_client is None``
        :ptype nats_client: Any
        :param write_buffer: unused; heartbeat writes bypass the
            deferred-flush path entirely (no L3)
        :ptype write_buffer: WriteBuffer | None
        """
        super().__init__(registry, config, nats_client, write_buffer)
        # override whatever L3 pool the registry snapped on us: this
        # Collection is L1+L2 only by design. assigning ``None`` here
        # makes the base-class L3 guards (``if self.l3_pool is None:
        # return``) fire on every fetch / save / delete_from_postgres
        # path that might leak through before the method-level
        # overrides can raise.
        self.l3_pool = None

    @property
    def table_name(self) -> str:
        """return L1 table name holding heartbeat rows.

        :return: table name
        :rtype: str
        """
        return "pod_heartbeats"

    @property
    def entity_class(self) -> type[HeartbeatEntity]:
        """return entity class for this collection.

        :return: :class:`HeartbeatEntity`
        :rtype: type[HeartbeatEntity]
        """
        return HeartbeatEntity

    async def get(self, entity_id: Any) -> HeartbeatEntity | None:
        """read a heartbeat from L1 first, pull-through L2 on miss.

        overrides the three-tier read path. the base-class
        :meth:`BaseCollection.get` falls through to
        :meth:`fetch_from_postgres` on L1+L2 miss; for this
        Collection that is a raise trigger by design. instead, an
        L1+L2 miss resolves to ``None`` so the :class:`HeartbeatSubscriber`
        can distinguish "this pod's row is gone, peer registry evicted
        it" from "something is mis-wired".

        :param entity_id: pod id
        :ptype entity_id: Any
        :return: hydrated :class:`HeartbeatEntity` or ``None`` on miss
        :rtype: HeartbeatEntity | None
        """
        row: dict[str, Any] | None = self.get_row_sync(entity_id)
        if row is None:
            l2_data = await self._get_from_l2(entity_id)
            if l2_data is not None:
                if self._l1 is not None:
                    self._l1.upsert(
                        self.table_name, l2_data, self.primary_key_column,
                    )
                row = l2_data
        result: HeartbeatEntity | None
        if row is None:
            result = None
        else:
            result = self.entity_class(row, is_new=False, collection=self)
            result.original_date_updated = row.get("date_updated")
        return result

    async def save_entity(self, entity: Any) -> None:
        """persist heartbeat entity to L1 + L2 (no L3).

        overrides the three-tier write path. the base-class
        :meth:`BaseCollection.save_entity` starts with
        ``save_to_postgres``; that raises for this Collection by
        design. the write here lays down L1 first, then L2, then
        publishes the cross-pod invalidation so peer registries
        refresh on next read.

        :param entity: :class:`HeartbeatEntity` to persist
        :ptype entity: Any
        :return: nothing
        :rtype: None
        """
        data = entity.to_dict()
        now = datetime.now(UTC)
        if entity.is_new:
            data.setdefault("date_created", now)
        data["date_updated"] = now
        # Convert aware -> naive at the L1/L2 border (TIMESTAMP columns)
        for key, val in list(data.items()):
            if isinstance(val, datetime) and val.tzinfo is not None:
                data[key] = val.replace(tzinfo=None)
        if self._l1 is not None:
            self._l1.upsert(self.table_name, data, self.primary_key_column)
        await self._save_to_l2(entity.id, data)
        await self._publish_invalidation(entity.id)
        entity.mark_clean()
        entity.original_date_updated = data.get("date_updated")

    async def delete(self, entity_id: Any) -> bool:
        """remove heartbeat row from L1 and L2.

        overrides the three-tier delete path. the base-class
        :meth:`BaseCollection.delete` calls
        ``delete_from_postgres`` unconditionally; for the L1+L2
        Collection that is a raise trigger. publishes an invalidation
        envelope so peer registries evict their own L1 copy.

        :param entity_id: pod id
        :ptype entity_id: Any
        :return: ``True`` unconditionally -- deletes are idempotent
        :rtype: bool
        """
        if self._l1 is not None:
            self._l1.delete_by_id(
                self.table_name, str(entity_id), self.primary_key_column,
            )
        await self._delete_from_l2(entity_id)
        await self._publish_invalidation(entity_id)
        return True

    async def fetch_from_postgres(
        self, entity_id: Any,
    ) -> dict[str, Any] | None:
        """unreachable on the L1+L2 Collection.

        :param entity_id: ignored; kept for signature symmetry
        :ptype entity_id: Any
        :return: never returns
        :rtype: dict[str, Any] | None
        :raises RuntimeError: always; L3 is not wired for heartbeats
        """
        raise RuntimeError(
            "HeartbeatCollection is L1+L2 only; fetch_from_postgres must "
            "never be reached (no L3 pool bound for 'pod_heartbeats')",
        )

    async def save_to_postgres(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """unreachable on the L1+L2 Collection.

        signature mirrors :meth:`BaseCollection.save_to_postgres`
        verbatim (including the keyword-only ``conn`` parameter) so an
        accidental framework invocation surfaces the documented
        ``RuntimeError`` rather than a confusing ``TypeError``.

        :param data: ignored; kept for signature symmetry
        :ptype data: dict[str, Any]
        :param original_timestamp: ignored
        :ptype original_timestamp: datetime | None
        :param conn: ignored; kept for LSP parity with base class
        :ptype conn: Any
        :return: never returns
        :rtype: int
        :raises RuntimeError: always; L3 is not wired for heartbeats
        """
        raise RuntimeError(
            "HeartbeatCollection is L1+L2 only; save_to_postgres must "
            "never be reached (save_entity is overridden to skip L3)",
        )

    async def delete_from_postgres(self, entity_id: Any) -> None:
        """unreachable on the L1+L2 Collection.

        :param entity_id: ignored; kept for signature symmetry
        :ptype entity_id: Any
        :return: never returns
        :rtype: None
        :raises RuntimeError: always; L3 is not wired for heartbeats
        """
        raise RuntimeError(
            "HeartbeatCollection is L1+L2 only; delete_from_postgres must "
            "never be reached (delete is overridden to skip L3)",
        )

    def serialize(self, data: dict[str, Any]) -> bytes:
        """serialize row dict to JSON bytes for L2 storage.

        :param data: row data
        :ptype data: dict[str, Any]
        :return: JSON-encoded bytes
        :rtype: bytes
        """
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """deserialize JSON bytes from L2 back into row dict.

        hydrates ``date_last_heartbeat`` / ``date_created`` /
        ``date_updated`` ISO strings back to naive
        :class:`datetime` so L1 pull-through writes typed values.

        :param data: JSON-encoded bytes
        :ptype data: bytes
        :return: row dict with typed datetime columns
        :rtype: dict[str, Any]
        """
        raw: dict[str, Any] = json.loads(data.decode("utf-8"))
        for field_name in ("date_last_heartbeat", "date_created", "date_updated"):
            value = raw.get(field_name)
            if isinstance(value, str):
                parsed = datetime.fromisoformat(value)
                if parsed.tzinfo is not None:
                    parsed = parsed.replace(tzinfo=None)
                raw[field_name] = parsed
        tools = raw.get("tools")
        if isinstance(tools, str):
            try:
                raw["tools"] = json.loads(tools)
            except (json.JSONDecodeError, ValueError):
                raw["tools"] = []
        return raw

    async def get_pods(self, pod_ids: list[str]) -> list[HeartbeatEntity]:
        """hydrate multiple pods by id through the normal tier path.

        convenience wrapper around :meth:`get`. callers (the
        :class:`HeartbeatSubscriber` health-check loop) maintain their
        own set of known pod ids; when the sweep runs it reads every
        known pod through this method, which honours the L1 -> L2 ->
        (raise, because L3 is off) pull-through contract.

        ids that resolve to a miss in every wired tier are silently
        dropped (returned list is shorter than input); callers treat
        a dropped pod as "the other registry already evicted this one
        and our L1 saw the invalidation, so it is gone".

        :param pod_ids: pod ids to hydrate
        :ptype pod_ids: list[str]
        :return: list of hydrated entities in the order their pods
            resolved to a hit; misses are omitted
        :rtype: list[HeartbeatEntity]
        """
        result: list[HeartbeatEntity] = []
        for pod_id in pod_ids:
            entity = await self.get(pod_id)
            if entity is not None:
                result.append(entity)
        return result
