"""entity definitions for the channels presence/membership state layer.

channels-task-01 lands the cross-pod, tenancy-ready presence state
layer that replaces the racy single-pod dict ``ConnectionRegistry``.
two pk-keyed entity types model the state, both stored in the
L1+L2-only :class:`~threetears.channels.presence.collection.PresenceCollection`
(modelled on :class:`~threetears.registry.heartbeat_collection.HeartbeatCollection`):

- :class:`PresenceConnectionEntity` (``pk = connection_id``) carries
  one live connection's room membership + identity + the
  ``date_last_heartbeat`` that drives self-heal. heartbeats refresh
  THIS entry only, so a busy heartbeat path never contends on the
  room-index CAS.
- :class:`RoomIndexEntity` (``pk = room_id``) carries the set of
  member ``connection_id`` s for one ``{customer}:{story}:{branch}:{file}``
  room. updated only on join/leave (low churn).

both are L1+L2-only: presence is transient by construction (a dropped
pod's connections re-join within seconds of client reconnect, and the
sweep prunes the stragglers), so the platform does not durably persist
presence through L3.
"""

from __future__ import annotations

from threetears.core.entities.base import BaseEntity

__all__ = ["PresenceConnectionEntity", "RoomIndexEntity"]


class PresenceConnectionEntity(BaseEntity):
    """per-connection presence row for one live websocket connection.

    fields mirror the ``presence_connections`` L1 table managed by
    :class:`~threetears.channels.presence.collection.PresenceCollection`.

    :cvar primary_key_field: pk column name; opaque connection ids
        minted per-socket by the handler, not UUIDs
    :ivar connection_id: opaque per-socket identifier (primary key)
    :ivar room_id: ``{customer}:{story}:{branch}:{file}`` room this
        connection is a member of
    :ivar user_id: authenticated principal owning the connection
    :ivar pod_id: id of the pod hosting the live socket; lets the
        fanout (task-02) resolve which pod holds a member's handle
    :ivar customer_id: tenant id; first-class so multi-tenancy is a
        config change later, never a re-architecture
    :ivar date_last_heartbeat: timestamp of the connection's most
        recent heartbeat; the sweep evicts connections whose value
        falls outside the liveness window
    :ivar date_created: timestamp when the connection first joined
    :ivar date_updated: timestamp of the most recent save; also the
        optimistic-lock CAS token for
        :meth:`~threetears.core.collections.base.BaseCollection.save_entity`
    """

    primary_key_field: str = "connection_id"


class RoomIndexEntity(BaseEntity):
    """room-index row holding the member connection-id set for one room.

    fields mirror the ``presence_rooms`` L1 table managed by
    :class:`~threetears.channels.presence.collection.PresenceCollection`.
    the member set is stored as a JSON list of ``connection_id`` strings
    (the L1 column is JSONB; the L2 codec round-trips it as a JSON
    array). updated only on join/leave under optimistic-concurrency CAS.

    :cvar primary_key_field: pk column name; the room id is the opaque
        ``{customer}:{story}:{branch}:{file}`` string
    :ivar room_id: ``{customer}:{story}:{branch}:{file}`` room key
        (primary key)
    :ivar customer_id: tenant id owning the room; denormalised onto the
        index row so a sweep / admin read can scope by tenant without
        re-parsing the composite room id
    :ivar members: list of member ``connection_id`` strings; the
        authoritative cross-pod membership set for this room
    :ivar date_created: timestamp when the room index was first written
    :ivar date_updated: timestamp of the most recent save; also the
        optimistic-lock CAS token
    """

    primary_key_field: str = "room_id"
