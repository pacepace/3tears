"""agent-audit v003 (CONTRACT, held): the index no read uses is dropped.

- ``idx_audit_events_customer_time`` (v001, ``(customer_id, timestamp DESC)``): no read in 3tears or a consumer pins
  a customer's events here. The hub's audit pages read its own ``platform_audit.audit_events`` through its v164
  ``audit_events_page_by_customer_timestamp``; this package's reads are retention's prune (``idx_audit_events_time_asc``)
  and an erasure by actor (``idx_audit_events_actor``). Found unread by the index tool's run with the hub as consumer
  (o96, 2026-11-14); dropped under Pace's rule (b) (2026-11-13: a non-unique index no read uses).

A pod of an older release may still plan through a dropped index mid-rollout, so the step is held:
:func:`~threetears.agent.audit.migrations.register` attaches it only when asked (``held_contract=True``: the test
schemas now, every deployment from the release after this one).

``DROP INDEX IF EXISTS``, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["drop_unread_audit_index"]

log = get_logger(__name__)

_DROP_CUSTOMER_TIME_INDEX_SQL = "DROP INDEX IF EXISTS idx_audit_events_customer_time"


async def drop_unread_audit_index(store: DataStore) -> None:
    """Drop ``idx_audit_events_customer_time`` (v003, contract).

    :param store: the migration's data store
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_DROP_CUSTOMER_TIME_INDEX_SQL)
    log.info("agent-audit v003 (contract): idx_audit_events_customer_time dropped")
