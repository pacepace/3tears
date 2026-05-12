"""entity definitions for registry persistent state.

namespace-task-01 phase 8.5l-3: the sole entity in this module is
:class:`HeartbeatEntity`, persisting per-pod heartbeat state for
:class:`~threetears.registry.heartbeat_collection.HeartbeatCollection`.
the former :class:`HeartbeatMonitor` bundled NATS subscription,
persistent state, health-check loop, and tool-catalog invalidation into
one class; phase 8.5l-3 splits the persistent-state portion out into a
proper :class:`BaseCollection`-backed surface. this file carries only
the entity. the Collection lives in ``heartbeat_collection.py``; the
orchestration (NATS subscription + health-check loop) lives in
``health.py`` as :class:`HeartbeatSubscriber`.
"""

from __future__ import annotations

from threetears.core.entities.base import BaseEntity

__all__ = ["HeartbeatEntity"]


class HeartbeatEntity(BaseEntity):
    """persistent state for a single tool pod's heartbeat record.

    fields mirror the ``pod_heartbeats`` L1 table managed by
    :class:`HeartbeatCollection`. the collection is L1+L2-only: pod
    heartbeats are transient (a restarted pod re-registers within
    seconds and re-emits its heartbeat), so the platform does not
    durably persist them through L3. the L2 tier gives cross-registry
    coherence: when one registry process marks a pod unresponsive, a
    peer registry sees the invalidation and refreshes its own L1.

    :cvar primary_key_field: pk column name on the backing table;
        pod ids are opaque strings supplied by tool pods, not UUIDs
    :ivar pod_id: opaque pod identifier (primary key)
    :ivar date_last_heartbeat: timestamp of most recent heartbeat from
        this pod
    :ivar tools: list of tool full_names this pod currently serves
        (for informational purposes and pod-local audit; the catalog
        is the source of truth for routing)
    :ivar tools_count: count of tools this pod is serving; carried
        alongside :attr:`tools` so downstream metrics can read a
        stable integer without re-walking the list
    :ivar status: pod liveness label; one of ``"healthy"``,
        ``"draining"``, or ``"unresponsive"``. the subscriber owns the
        transitions: initial heartbeat -> ``"healthy"``, timeout
        exceeded -> ``"unresponsive"``
    :ivar consecutive_misses: count of successive health-check sweeps
        in which this pod's heartbeat fell outside the liveness
        timeout; reset to zero on every received heartbeat
    :ivar date_created: timestamp when this pod was first observed
    :ivar date_updated: timestamp of the most recent save; also the
        optimistic-lock CAS token for :meth:`BaseCollection.save_entity`
    """

    primary_key_field: str = "pod_id"
