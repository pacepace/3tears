"""durable record of when a replay ledger first existed, so a wipe can be told from a first run.

:class:`~threetears.core.coordination.replay_guard.ReplayGuard` keeps its nonces in a
memory-backed KV bucket, so a broker restart empties it. The guard cannot then tell two
situations apart: it has never run before, or it ran and forgot. It assumes the worse one and
refuses every artifact issued within the verifier's tolerance of the bucket's creation time.

That assumption is right after a wipe and wrong on a first run. On a first run no artifact was
ever recorded, so no replay is possible and the refusal protects nothing while denying every
artifact for the length of the window -- on a fresh deployment, and in any test whose bucket is
younger than the tolerance.

An anchor settles it with one durable fact: the moment this ledger first existed. Anchor absent
means the ledger is new, so nothing can have been forgotten. Anchor older than the bucket means
the bucket is a replacement and the watermark applies exactly as before.

**The anchor must outlive the thing it describes**, so it cannot live in the bucket. It is a row
in the coordination tables, written once and never expiring, read once per guard at its first
record -- never on the per-artifact path.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from threetears.observe import get_logger

from threetears.core.coordination.tables import CoordinationRedemptionsCollection, coordination_collection

if TYPE_CHECKING:
    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import CoreConfig

__all__ = ["ANCHOR_KEY", "CollectionReplayAnchor", "ReplayAnchor"]

log = get_logger(__name__)

#: the row key every anchor uses. The purpose segment carries which ledger it anchors, so one
#: key here keeps the row addressable without a second naming scheme to keep in step.
ANCHOR_KEY = "ledger"

#: prefix on the anchor row's purpose, so an anchor can never collide with a redemption sharing
#: the ledger's own purpose string.
_PURPOSE_PREFIX = "replay-anchor"


@runtime_checkable
class ReplayAnchor(Protocol):
    """the durable first-existence record a :class:`ReplayGuard` consults once."""

    async def first_existed(self, purpose: str, *, now: datetime) -> datetime:
        """the moment ``purpose``'s ledger first existed, recording ``now`` if nothing is.

        Claim-or-read: the first caller across every replica records ``now`` and gets it back;
        every later caller gets what that first one recorded, whenever this process started.

        :param purpose: which ledger is being anchored -- the guard's bucket name
        :ptype purpose: str
        :param now: this process's clock, recorded when no anchor exists yet
        :ptype now: datetime
        :return: the aware UTC moment the ledger first existed
        :rtype: datetime
        :raises Exception: on a storage failure. The guard treats ANY failure as "cannot tell"
            and falls back to the watermark, so an implementation must raise rather than guess
        """
        ...


class CollectionReplayAnchor:
    """a :class:`ReplayAnchor` over the coordination tables.

    The anchor is a redemption in the literal sense the table already models: the first caller
    to claim "this ledger now exists" wins, and every later one reads that claim back. It is
    written with no expiry, so the expiry sweep -- which matches only ``expires_at IS NOT NULL``
    -- never removes it. A row that expired would read as a ledger that had never existed, which
    is the one wrong answer this class must not give.
    """

    def __init__(self, registry: "CollectionRegistry", *, config: "CoreConfig | None" = None) -> None:
        """configure the anchor over its registry's coordination tables.

        :param registry: the collection registry this anchor reads and writes through
        :ptype registry: CollectionRegistry
        :param config: core config forwarded when this process first builds the collection
        :ptype config: CoreConfig | None
        """
        self._collection = coordination_collection(registry, CoordinationRedemptionsCollection, config)
        # The claim has the same exactly-once shape a redemption does: two replicas booting
        # together must not both be told they recorded the first anchor, or the later one's
        # clock would silently become the ledger's birth time.
        self._collection.require_l2_fence("CollectionReplayAnchor")

    async def first_existed(self, purpose: str, *, now: datetime) -> datetime:
        """claim or read ``purpose``'s first-existence moment.

        :param purpose: which ledger is being anchored
        :ptype purpose: str
        :param now: this process's clock, recorded when no anchor exists yet
        :ptype now: datetime
        :return: the aware UTC moment the ledger first existed
        :rtype: datetime
        :raises threetears.nats.KvError: on an L2 failure
        :raises threetears.core.exceptions.DataLayerUnavailableError: on an L3 failure
        :raises RuntimeError: when the row cannot be read back after being written, which would
            otherwise return a first-existence time no tier actually holds
        """
        row_id = (f"{_PURPOSE_PREFIX}:{purpose}", ANCHOR_KEY)

        def _claim_if_absent(
            current: dict[str, Any] | None,
        ) -> tuple[Literal["upsert", "delete", "noop"], dict[str, Any] | None]:
            if current is None:
                # No expiry, deliberately: this row outlives every artifact it helps judge, and
                # a swept anchor reads as a ledger that never existed.
                #
                # `date_created` is written rather than left to the collection's own stamp: it
                # IS the value this class returns, so a caller that passed a moment and got a
                # different one back would have no way to notice.
                return "upsert", {
                    "purpose": row_id[0],
                    "key": ANCHOR_KEY,
                    "expires_at": None,
                    "date_created": now,
                }
            return "noop", None

        await self._collection.l2_cas_mutate(row_id, _claim_if_absent)
        row = await self._collection.get(row_id)
        if row is None:
            raise RuntimeError(
                f"replay anchor for {purpose!r} is absent immediately after being claimed; "
                "the guard cannot tell a wiped ledger from a new one without it"
            )
        # The winning write's moment, so every replica reads one birth time rather than each
        # keeping its own.
        first_existed: datetime = row.date_created
        return first_existed
