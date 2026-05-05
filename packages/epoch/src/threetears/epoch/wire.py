"""typed wire envelope for config-epoch broadcasts.

publishers emit one :class:`EpochBumpMessage` per
:meth:`threetears.epoch.client.EpochClient.bump` call after the
``config_epochs`` row has committed. every other pod subscribed to the
matching :class:`~threetears.nats.subjects.Subject` receives the message
via :meth:`threetears.nats.NatsClient.subscribe_typed` with
``message_type=EpochBumpMessage`` -- validation failures deadletter via
the standard typed-NATS path and never reach the listener callback.

field set is frozen by an enforcement test: additions are additive
(default-None) and removals fail. mirror's
:class:`threetears.core.collections.registry.CacheInvalidationMessage`
discipline -- typed Pydantic envelope, ``model_dump_json`` serialization,
opaque to the framework after parsing.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = [
    "EpochBumpMessage",
]


class EpochBumpMessage(BaseModel):
    """typed wire envelope for one config-epoch bump broadcast.

    :ivar subject_path: namespaced NATS subject the bump targets;
        matches the ``config_epochs.subject_path`` row PK and the
        path string of the :class:`Subject` the publisher used
    :ivar epoch: new strictly-monotonic epoch returned by the
        atomic ``INSERT ... ON CONFLICT DO UPDATE`` statement
    :ivar payload: opaque hint for the consumer's reload callback;
        framework never inspects, consumers parse if useful
    """

    model_config = ConfigDict(frozen=True)

    subject_path: str
    epoch: int
    payload: dict[str, Any] | None = None
