"""a pooled asyncpg connection that sends ``SET search_path`` and ``SET statement_timeout`` only when they change.

the cost
--------

A platform pool on YugabyteDB cannot reset a released connection with asyncpg's default ``RESET ALL``
(it breaks the JSONB codecs), so it restores what its borrowers change: the hub's reset query was
``SET search_path TO <platform schema>`` on EVERY release, and its query proxy sends
``SET search_path`` and ``SET statement_timeout`` before every proxied statement. On cobalt dev on
2026-10-29 that was about 133 ``SET search_path`` and 17 ``SET statement_timeout`` round trips a
second, most of them setting what the session already held: the YSQL statement mix was mostly
SETs.

the rule
--------

:func:`session_settings_connection_class` builds a connection class that remembers, per connection,
the value it last set for each of the two settings, and:

- sends a plain ``SET <setting> TO <value>`` outside a transaction only when the value differs from
  the one it holds; one sent INSIDE a transaction is sent and the value forgotten, since a rollback
  undoes it;
- forgets both values when any statement it sends names either setting another way (``SET LOCAL``,
  ``set_config``, ``RESET``, ``DISCARD``), so a value it holds is never one the session may not;
- resets a released connection's ``search_path`` to the pool's own only when it may have moved.

A value forgotten is a value not known, never a value assumed: the next ``SET`` is sent and the next
release resets. A connection starts with both unknown, so its first release resets once.
"""

from __future__ import annotations

import re
from typing import Any, Final

import asyncpg

__all__ = ["session_settings_connection_class"]

#: the two settings a borrower changes on a pooled platform connection
_TRACKED: Final = ("search_path", "statement_timeout")

#: a plain session ``SET`` of a tracked setting, the one statement a value is learned from; one statement only, so
#: skipping it never skips another
_PLAIN_SET_RE: Final = re.compile(
    r"\s*SET\s+(?:SESSION\s+)?(search_path|statement_timeout)\s*(?:TO|=)\s*([^;]+?)\s*;?\s*", re.IGNORECASE | re.DOTALL
)

#: any other mention of a tracked setting, or a statement that resets session state, after which nothing is known
_CHANGES_SESSION_RE: Final = re.compile(r"search_path|statement_timeout|\bRESET\b|\bDISCARD\b", re.IGNORECASE)


def session_settings_connection_class(*, search_path: str) -> type[asyncpg.Connection]:
    """a connection class whose ``search_path`` returns to ``search_path`` on release, sent only when it moved.

    Pass it as ``connection_class`` to ``asyncpg.create_pool``; it replaces asyncpg's ``RESET ALL``
    reset, which YugabyteDB's JSONB codecs do not survive, so session state other than the two settings
    is the borrower's to restore, as with any reset that is not ``RESET ALL``.

    :param search_path: the pool's own ``search_path``, as a ``SET search_path TO`` takes it
    :ptype search_path: str
    :return: the connection class
    :rtype: type[asyncpg.Connection]
    """
    home = search_path.strip()
    reset_query = f"SET search_path TO {home}"

    class _SessionSettingsConnection(asyncpg.Connection):  # type: ignore[misc]
        """an asyncpg connection that holds what it last set ``search_path`` and ``statement_timeout`` to."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._held: dict[str, str | None] = dict.fromkeys(_TRACKED)

        def get_reset_query(self) -> str:
            """``SET search_path`` back to the pool's, only when it may have moved.

            :return: the reset statement, or ``""`` when the session holds the pool's own
            :rtype: str
            """
            return "" if self._held["search_path"] == home else reset_query

        def _forget(self, query: str) -> None:
            """forget both values when ``query`` may change either another way than a plain SET.

            :param query: the statement about to be sent
            :ptype query: str
            :return: nothing
            :rtype: None
            """
            if _CHANGES_SESSION_RE.search(query):
                self._held = dict.fromkeys(_TRACKED)

        async def execute(self, query: str, *args: Any, timeout: float | None = None) -> str:
            """send ``query``, unless it is a plain SET of a value this session already holds.

            :param query: the statement
            :ptype query: str
            :param args: its arguments
            :ptype args: Any
            :param timeout: asyncpg's timeout
            :ptype timeout: float | None
            :return: the status
            :rtype: str
            """
            plain = None if args else _PLAIN_SET_RE.fullmatch(query)
            if plain is None:
                self._forget(query)
                status: str = await super().execute(query, *args, timeout=timeout)
                return status
            setting, value = plain.group(1).lower(), plain.group(2).strip()
            if not self.is_in_transaction() and self._held[setting] == value:
                return "SET"
            # unknown until it has been sent, and still unknown after it when a rollback may undo it
            self._held[setting] = None
            status = await super().execute(query, timeout=timeout)
            if not self.is_in_transaction():
                self._held[setting] = value
            return status

        async def fetch(self, query: str, *args: Any, **kwargs: Any) -> list[Any]:
            """asyncpg's ``fetch``, forgetting the settings when the statement names one.

            :param query: the statement
            :ptype query: str
            :param args: its arguments
            :ptype args: Any
            :param kwargs: asyncpg's options
            :ptype kwargs: Any
            :return: the rows
            :rtype: list[Any]
            """
            self._forget(query)
            rows: list[Any] = await super().fetch(query, *args, **kwargs)
            return rows

        async def fetchrow(self, query: str, *args: Any, **kwargs: Any) -> Any:
            """asyncpg's ``fetchrow``, forgetting the settings when the statement names one.

            :param query: the statement
            :ptype query: str
            :param args: its arguments
            :ptype args: Any
            :param kwargs: asyncpg's options
            :ptype kwargs: Any
            :return: the row
            :rtype: Any
            """
            self._forget(query)
            return await super().fetchrow(query, *args, **kwargs)

        async def fetchval(self, query: str, *args: Any, **kwargs: Any) -> Any:
            """asyncpg's ``fetchval``, forgetting the settings when the statement names one.

            :param query: the statement
            :ptype query: str
            :param args: its arguments
            :ptype args: Any
            :param kwargs: asyncpg's options
            :ptype kwargs: Any
            :return: the value
            :rtype: Any
            """
            self._forget(query)
            return await super().fetchval(query, *args, **kwargs)

        async def executemany(self, command: str, args: Any, *, timeout: float | None = None) -> None:
            """asyncpg's ``executemany``, forgetting the settings when the statement names one.

            :param command: the statement
            :ptype command: str
            :param args: its argument rows
            :ptype args: Any
            :param timeout: asyncpg's timeout
            :ptype timeout: float | None
            :return: nothing
            :rtype: None
            """
            self._forget(command)
            await super().executemany(command, args, timeout=timeout)

    return _SessionSettingsConnection
