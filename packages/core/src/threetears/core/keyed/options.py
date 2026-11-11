"""a connection whose every statement carries the caller's keyword arguments, for the keyed helpers.

The keyed helpers send ``conn.fetch(query, *params)`` / ``conn.execute(query, *params)``. A connection that routes a
statement by a keyword argument (a proxy's ``namespace=``, a driver's ``timeout=``) takes it per statement, which a
helper call has no place for; :func:`with_options` wraps the connection so every statement carries them, and every
helper reads through it unchanged::

    row = await read_row(with_options(conn, namespace=name), "workspace_file_versions", by={...})

Every other attribute (``transaction``, ``tx_id``, the dialect marker) is the wrapped connection's.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Final

from threetears.core.keyed._filters import binds_native_arrays

__all__ = ["with_options"]

#: the methods that send a statement, each sent with the options
_STATEMENT_METHODS: Final = frozenset({"fetch", "fetchrow", "fetchval", "execute", "executemany", "fetch_arrow"})


class _WithOptions:
    """``conn`` with ``options`` added to every statement's keyword arguments (a statement's own win over them)."""

    def __init__(self, conn: Any, options: dict[str, Any]) -> None:
        self._conn = conn
        self._options = options
        # asyncpg's own connection binds typed arrays natively; known by its type, which this wrapper is not
        self.binds_native_arrays = binds_native_arrays(conn)

    def __getattr__(self, name: str) -> Any:
        found = getattr(self._conn, name)
        if name in _STATEMENT_METHODS:
            return partial(found, **self._options)
        return found


def with_options(conn: Any, **options: Any) -> Any:
    """``conn``, its every statement sent with ``options`` as keyword arguments; ``conn`` itself when there are none.

    A statement method the connection lacks the wrapper lacks too, so a check of what the connection can do (an Arrow
    reader's ``fetch_arrow``) answers for it as before. A reader asked whether it answers from YugabyteDB (a listing's
    page) is asked once per wrapper: keep one wrapper for a run of pages.

    :param conn: a connection with asyncpg-shaped ``fetch`` / ``fetchrow`` / ``fetchval`` / ``execute`` /
        ``executemany`` (and ``fetch_arrow`` where it has one), each taking the keyword arguments given
    :ptype conn: Any
    :param options: the keyword arguments every statement carries (``namespace=``, ``timeout=``)
    :ptype options: Any
    :return: the connection to hand a keyed helper
    :rtype: Any
    """
    if not options:
        return conn
    return _WithOptions(conn, options)
