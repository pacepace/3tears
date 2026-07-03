"""
conversations v003: add ``name`` column to ``conversations`` table.

display-label support for human-facing agent UIs: every consumer
exposing conversations to end users (chat clients, admin consoles)
needs a renameable title independent of the rolling ``summary``.
some consumers carried this column locally pre-migration; promoted to the
canonical entity here so downstream consumers do not
re-invent it in their own schemas.

nullable -- conversations get titled lazily (often by an LLM after the
first user turn) so a fresh row predates the title.  no backfill is
needed because the column is nullable; the helper still emits the
ALTER as ``ADD COLUMN IF NOT EXISTS`` for replay safety.

yugabyte-safe shape: the helper splits the ALTER from any UPDATE so
DDL and DML never share a transaction (here only the ALTER fires
because no backfill is requested).
"""

from __future__ import annotations

from threetears.core.data.migrations.helpers import add_column_with_backfill
from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_name_column",
]

log = get_logger(__name__)


async def add_name_column(store: DataStore) -> None:
    """
    add ``name TEXT`` (nullable) to ``conversations``.

    yugabyte-safe via :func:`add_column_with_backfill`.  the column is
    added with ``ADD COLUMN IF NOT EXISTS`` for idempotency; no
    backfill UPDATE is emitted because the column is nullable and
    consumers populate it lazily.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("adding name column to conversations")
    await add_column_with_backfill(
        store,
        table="conversations",
        column="name",
        column_type="TEXT",
        default=None,
        not_null=False,
        backfill_value_sql=None,
        backfill_predicate=None,
        backfill_replay_guard=False,
    )
