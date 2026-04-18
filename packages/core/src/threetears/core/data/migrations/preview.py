"""
preview-mode DataStore wrapper for the migration runner.

a :class:`PreviewStore` wraps a "real" DataStore-shape and intercepts
every ``execute`` and ``query`` call so the runner can answer the
question "what DDL would you run if I asked you to apply pending
migrations?" without actually mutating the target database.

design contract
---------------

- bookkeeping queries the runner issues against ``_schema_migrations``
  must still return realistic answers, so the wrapper delegates the
  ``SELECT`` paths through to the underlying store. that lets the
  preview run against a database that already has migrations applied
  and surface only the *pending* DDL.
- ``CREATE TABLE IF NOT EXISTS _schema_migrations`` and the
  ``INSERT``/``DELETE`` statements the runner uses for bookkeeping are
  recognized by SQL prefix and recorded into the captured stream as
  ``BOOKKEEPING`` entries instead of being executed. this avoids
  writing rows that would skew a subsequent real apply, while still
  producing a complete record of what the runner would do.
- every other ``execute`` call is captured into the captured stream
  and not delegated to the underlying store. this is the migration
  body DDL the operator wants to review.

callers consume the captured stream via :meth:`captured_statements`
and render it for human review (see ``aibots.hub.migrations.__main__``
``upgrade --sql``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CapturedStatement",
    "PreviewStore",
]

# SQL fragments below match the literals MigrationRunner issues. they
# are kept here (rather than imported from runner.py) to keep the
# preview wrapper independent of internal changes; the test
# ``test_preview_recognizes_bookkeeping_prefixes`` pins the contract.

_BOOKKEEPING_PREFIXES = (
    "CREATE TABLE IF NOT EXISTS _SCHEMA_MIGRATIONS",
    "INSERT INTO _SCHEMA_MIGRATIONS",
    "DELETE FROM _SCHEMA_MIGRATIONS",
)


@dataclass
class CapturedStatement:
    """
    one captured statement the runner would have issued.

    :ivar sql: SQL text as the runner produced it
    :ivar params: positional parameters bound to the statement
    :ivar kind: one of ``"DDL"`` (migration body) or ``"BOOKKEEPING"``
        (``_schema_migrations`` insert/delete/create)
    """

    sql: str
    params: tuple[Any, ...]
    kind: str


@dataclass
class PreviewStore:
    """
    DataStore-shape wrapper that captures executes instead of running them.

    the wrapper presents the same ``execute`` / ``query`` surface the
    runner consumes. it delegates every ``query`` and every bookkeeping
    ``SELECT``/``COALESCE`` against ``_schema_migrations`` through to
    the underlying store so the runner sees the database's real
    applied-version state. every non-bookkeeping ``execute`` is
    captured.

    :ivar underlying: the wrapped DataStore-shape used for read-only
        delegation of bookkeeping queries
    :ivar captured: list of captured statements, in the order the
        runner would have issued them
    """

    underlying: Any
    captured: list[CapturedStatement] = field(default_factory=list)

    def captured_statements(self) -> list[CapturedStatement]:
        """
        return the ordered list of captured statements.

        the returned list is the live mutable list; callers that need
        a snapshot should ``list(...)`` it themselves. preview output
        is one-shot; mutating after the apply call is not supported.

        :return: ordered list of captured statements
        :rtype: list[CapturedStatement]
        """
        result = self.captured
        return result

    def captured_ddl(self) -> list[str]:
        """
        return the SQL text of just the DDL captures (no bookkeeping).

        this is what an operator wants to read or pipe to a file. the
        ``BOOKKEEPING`` entries are interesting for debugging the
        runner itself but noise for routine ops review.

        :return: list of DDL SQL strings in apply order
        :rtype: list[str]
        """
        result = [s.sql for s in self.captured if s.kind == "DDL"]
        return result

    async def execute(self, sql: str, *params: Any) -> str:
        """
        capture the statement instead of running it; classify it first.

        :param sql: SQL text the runner produced
        :ptype sql: str
        :param params: positional parameters bound to the statement
        :ptype params: Any
        :return: synthetic asyncpg-style status tag
        :rtype: str
        """
        normalized = " ".join(sql.split()).upper()
        is_bookkeeping = any(normalized.startswith(p) for p in _BOOKKEEPING_PREFIXES)
        kind = "BOOKKEEPING" if is_bookkeeping else "DDL"
        self.captured.append(
            CapturedStatement(sql=sql, params=tuple(params), kind=kind)
        )
        result = "PREVIEW"
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """
        delegate query through to the underlying store.

        bookkeeping reads must reflect what is actually applied so the
        preview can subtract already-applied migrations from the captured
        DDL. delegating every query is simpler and just as correct since
        the runner only issues SELECTs against ``_schema_migrations``.

        :param sql: SQL text the runner produced
        :ptype sql: str
        :param params: positional parameters bound to the query
        :ptype params: Any
        :return: rows from the underlying store
        :rtype: list[dict[str, Any]]
        """
        result: list[dict[str, Any]] = await self.underlying.query(sql, *params)
        return result
