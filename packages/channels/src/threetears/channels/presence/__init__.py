"""channels presence/membership state layer (channels-task-01).

the concurrency-safe, cross-pod, tenancy-ready foundation that replaces
the racy single-pod dict ``ConnectionRegistry``. presence/membership
state lives in L1+L2 :class:`~threetears.core.collections.base.BaseCollection`
instances (never dicts); the only pod-local structure is the
synchronized ``connection_id → live socket`` map on
:class:`RoomState`.

public surface:

- :class:`PresenceCollection` — facade over the two pk-keyed L1+L2
  Collections (per-connection + room-index).
- :class:`RoomState` — the reshaped registry: socket map +
  join/leave/heartbeat/members/local_sockets.
- :class:`PresenceSweeper` — heartbeat-driven self-heal.
- :class:`RoomMember` — a resolved cross-pod room member.
- the entities + the L1 schema factory for direct wiring.
"""

from __future__ import annotations

from threetears.channels.presence.collection import (
    PresenceCollection,
    PresenceConnectionCollection,
    RoomIndexCollection,
)
from threetears.channels.presence.entities import (
    PresenceConnectionEntity,
    RoomIndexEntity,
)
from threetears.channels.presence.l1_cache import (
    PRESENCE_L1_METADATA,
    PRESENCE_L1_TABLE_NAMES,
    create_presence_l1_backend,
)
from threetears.channels.presence.room_state import RoomMember, RoomState
from threetears.channels.presence.sweeper import PresenceSweeper

__all__ = [
    "PRESENCE_L1_METADATA",
    "PRESENCE_L1_TABLE_NAMES",
    "PresenceCollection",
    "PresenceConnectionCollection",
    "PresenceConnectionEntity",
    "PresenceSweeper",
    "RoomIndexCollection",
    "RoomIndexEntity",
    "RoomMember",
    "RoomState",
    "create_presence_l1_backend",
]
