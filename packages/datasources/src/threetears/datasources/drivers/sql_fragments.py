"""shared driver utilities: placeholder translation across SQL dialects.

every concrete driver in :mod:`threetears.datasources.drivers` reuses
:func:`translate_placeholders` instead of reimplementing the regex
dance. centralizing the edge-case handling (``$10`` vs ``$1``, escaped
``$$``, string-literal ``'$1'``) prevents per-driver bugs.

target styles:

- ``"asyncpg"`` -- no-op; ``$1`` is already asyncpg's native style
- ``"pyformat"`` -- ``$1`` -> ``%s`` (psycopg2 / redshift_connector
  pyformat-style binding; positional placeholders are all ``%s``)
- ``"numeric"`` -- ``$1`` -> ``:1`` (Oracle-style numeric placeholders;
  position preserved)
- ``"named-at"`` -- ``$1`` -> ``@p1`` (BigQuery named query parameters;
  the at-prefixed name encodes the original ordinal)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from threetears.core.fingerprint import relation_key_expression
from threetears.core.keyed import SQL_TOKEN, each_placeholder
from threetears.core.sql_fragments import (
    FilterValue,
    as_written,
    equality_conditions,
    filter_conditions,
    quote_identifier,
)

__all__ = [
    "PlaceholderStyle",
    "build_equality_filter",
    "build_filter",
    "build_relation_key_expression",
    "build_reset_statement_timeout_sql",
    "build_search_path_value",
    "build_set_local_statement_timeout_sql",
    "build_set_search_path_sql",
    "build_set_statement_timeout_sql",
    "translate_placeholders",
]

PlaceholderStyle = Literal["asyncpg", "pyformat", "numeric", "named-at"]


#: SESSION-scoped ``statement_timeout``. Postgres / Redshift take
#: milliseconds. TWO call sites are legitimate, both in the driver:
#: connection open (establish the datasource ceiling) and the
#: release path (restore that ceiling). ANY other use is the leak the
#: whole per-statement override exists to prevent -- a session-scoped
#: SET persists on a cached connection and hands the previous
#: borrower's bound to whoever draws it next.
_SET_STATEMENT_TIMEOUT_SQL_TEMPLATE = "SET statement_timeout TO {ms:d}"


#: TRANSACTION-scoped ``statement_timeout``: the per-statement override.
#: ``SET LOCAL`` unwinds when the transaction ends, whether it commits
#: or rolls back, which makes the leak structurally impossible rather
#: than merely unlikely. both Postgres and Redshift document ``SET
#: [ SESSION | LOCAL ] parameter TO value``; ``SET LOCAL`` outside a
#: transaction block is a no-op with a warning, so callers MUST have a
#: block open (both drivers do -- see their override paths).
_SET_LOCAL_STATEMENT_TIMEOUT_SQL_TEMPLATE = "SET LOCAL statement_timeout TO {ms:d}"


#: restore ``statement_timeout`` to the connection's session default.
#: used by the asyncpg release path, where the "default" is whatever
#: the server / startup packet established rather than a driver-held
#: number.
_RESET_STATEMENT_TIMEOUT_SQL = "RESET statement_timeout"


def _validate_timeout_seconds(timeout_seconds: int) -> int:
    """validate a timeout override and convert it to milliseconds.

    the value is formatted INLINE into the SQL because neither Postgres
    nor Redshift accepts a bind parameter in a ``SET`` statement (the
    parser rejects ``SET x = $1`` with a syntax error). validating here
    -- a positive ``int``, never a string -- is what keeps that inline
    format off the injection surface.

    :param timeout_seconds: per-statement timeout in seconds
    :ptype timeout_seconds: int
    :return: the timeout in milliseconds
    :rtype: int
    :raises ValueError: when the value is not a positive integer
    """
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
        raise ValueError(f"statement timeout must be a positive int, got {type(timeout_seconds).__name__}")
    if timeout_seconds <= 0:
        raise ValueError(f"statement timeout must be a positive number of seconds, got {timeout_seconds}")
    return timeout_seconds * 1000


def build_set_local_statement_timeout_sql(timeout_seconds: int) -> str:
    """build the TRANSACTION-scoped ``statement_timeout`` override.

    the single place the millisecond conversion and the ``LOCAL``
    scoping live, so the two drivers cannot drift into one of them
    emitting a session-scoped ``SET``. callers MUST be inside a
    transaction block -- ``SET LOCAL`` outside one is a no-op with a
    warning, which would silently leave the statement on the
    connection's ceiling.

    :param timeout_seconds: per-statement timeout in seconds
    :ptype timeout_seconds: int
    :return: the ``SET LOCAL statement_timeout`` statement
    :rtype: str
    :raises ValueError: when the value is not a positive integer
    """
    return _SET_LOCAL_STATEMENT_TIMEOUT_SQL_TEMPLATE.format(ms=_validate_timeout_seconds(timeout_seconds))


def build_set_statement_timeout_sql(timeout_seconds: int) -> str:
    """build the SESSION-scoped ``statement_timeout`` for the datasource ceiling.

    legitimate at exactly two driver call sites: connection open, and
    the release path that restores the ceiling after a per-statement
    override. never use this for a per-statement bound -- see
    :func:`build_set_local_statement_timeout_sql`.

    :param timeout_seconds: connection-level ceiling in seconds
    :ptype timeout_seconds: int
    :return: the ``SET statement_timeout`` statement
    :rtype: str
    :raises ValueError: when the value is not a positive integer
    """
    return _SET_STATEMENT_TIMEOUT_SQL_TEMPLATE.format(ms=_validate_timeout_seconds(timeout_seconds))


def build_reset_statement_timeout_sql() -> str:
    """build the ``RESET statement_timeout`` release-path statement.

    used where the session default is server-established rather than
    driver-held (asyncpg sends its session settings in the STARTUP
    packet, so ``RESET`` restores exactly that value).

    :return: the ``RESET statement_timeout`` statement
    :rtype: str
    """
    return _RESET_STATEMENT_TIMEOUT_SQL


def build_search_path_value(schemas: list[str]) -> str | None:
    """build the VALUE portion of a ``search_path`` setting for ``schemas``.

    returns the comma-separated, identifier-quoted form Postgres /
    Yugabyte / Redshift accept as the ``search_path`` value, e.g.::

        "reporting_prod", "audit"

    suitable for either:

    - asyncpg's :class:`create_pool` ``server_settings={"search_path": ...}``
      kwarg (sent in the STARTUP packet so the value survives
      ``DISCARD ALL`` / ``RESET ALL`` issued during pool release; this
      is the only safe asyncpg approach because the pool's reset
      otherwise wipes session-level ``SET`` state between acquires)
    - any caller that wants to interpolate the value clause without
      the ``SET search_path TO`` SQL prefix

    each schema name is identifier-quoted via :func:`threetears.core.sql_fragments.quote_identifier`
    so callers can pass arbitrary names without SQL-injection risk.
    order is preserved: leftmost-wins semantics for unqualified-name
    resolution, which matches both Postgres' documented behaviour and
    what callers expect when threading from a ``DatasourceConfig.schemas``
    list authored in priority order. returns ``None`` when ``schemas``
    is empty so callers can branch on "do not configure search_path".

    :param schemas: ordered list of schema names
    :ptype schemas: list[str]
    :return: comma-separated quoted value, or ``None`` when empty
    :rtype: str | None
    """
    if not schemas:
        return None
    return ", ".join(quote_identifier(s) for s in schemas)


def build_set_search_path_sql(schemas: list[str]) -> str | None:
    """build a ``SET search_path TO ...`` SQL statement for ``schemas``.

    convenience wrapper over :func:`build_search_path_value` that
    prepends the ``SET search_path TO `` SQL prefix; used by drivers
    whose underlying client library does NOT expose a server-settings
    startup hook (e.g. :mod:`redshift_connector`), which need to
    issue the SET via ``cursor.execute`` after connect.

    drivers whose client library DOES support startup-time server
    settings (e.g. asyncpg) should call :func:`build_search_path_value`
    directly so the search_path survives connection reset on pool
    release.

    :param schemas: ordered list of schema names to set
    :ptype schemas: list[str]
    :return: the SET statement, or ``None`` when ``schemas`` is empty
    :rtype: str | None
    """
    value = build_search_path_value(schemas)
    if value is None:
        return None
    return "SET search_path TO " + value


def _format_style(sql: str) -> str:
    """``sql`` in the DB-API ``format`` style Redshift's driver reads: each ``$n`` written ``%s``, and each ``%`` the
    driver would read as its own written ``%%``.

    The driver (``redshift_connector.core.convert_paramstyle``) reads a statement for ``%`` only when it is sent with
    parameters, and then reads one outside a quoted literal, a quoted name or a comment as a placeholder or an escape:
    a lone ``%`` there (a modulo, a ``LIKE`` pattern written unquoted) is refused ("Only %s and %% are supported"), so
    it is doubled. Inside a literal, a quoted name or a comment the driver leaves ``%`` as it is, and so does this. A
    dollar-quoted body the driver does not know as quoted: its ``%`` is doubled too. A statement naming no placeholder
    is sent with no parameters (the drivers' ``execute(sql)``), so it is left whole.

    :param sql: the statement, ``$n`` placeholders
    :ptype sql: str
    :return: the statement, ``%s`` placeholders
    :rtype: str
    """
    tokens = list(SQL_TOKEN.finditer(sql))
    if not any(token.group("n") is not None for token in tokens):
        return sql
    out: list[str] = []
    at = 0
    for token in tokens:
        out.append(sql[at : token.start()].replace("%", "%%"))
        if token.group("n") is not None:
            out.append("%s")
        elif token.group("tag") is not None:
            out.append(token.group().replace("%", "%%"))
        else:
            out.append(token.group())
        at = token.end()
    out.append(sql[at:].replace("%", "%%"))
    return "".join(out)


def translate_placeholders(sql: str, target_style: PlaceholderStyle) -> str:
    """translate ``$N``-style placeholders to ``target_style``.

    edge cases handled:

    - ``$1`` and ``$10`` in the same SQL parse as distinct positions
      (longest-match regex on the digit run)
    - a dollar-quoted string (``$$...$$``, ``$tag$...$tag$``) is preserved
      verbatim, its body with it
    - ``'$1'`` or ``"$1"`` inside a string literal or quoted name, and a
      ``$1`` in a comment, are preserved verbatim
    - mixed ``$1`` outside a literal AND ``'$1'`` inside is correctly
      handled: the statement is read a token at a time as the keyed
      builders read it (:data:`threetears.interfaces.keyed.filters.SQL_TOKEN`)

    :param sql: SQL text with ``$N``-style placeholders
    :ptype sql: str
    :param target_style: target placeholder dialect
    :ptype target_style: PlaceholderStyle
    :return: SQL with placeholders translated
    :rtype: str
    :raises ValueError: if ``target_style`` is not a recognised style
    """
    if target_style == "asyncpg":
        return sql
    if target_style == "pyformat":
        return _format_style(sql)
    if target_style == "numeric":
        return each_placeholder(sql, lambda number: f":{number}")
    if target_style == "named-at":
        return each_placeholder(sql, lambda number: f"@p{number}")
    raise ValueError(f"unknown placeholder style: {target_style!r}")


def build_equality_filter(where: Mapping[str, FilterValue] | None) -> tuple[str, list[FilterValue]]:
    """a `` WHERE`` fragment keeping only the rows whose columns equal ``where``'s values.

    Placeholders are ``$1..$n``, the canonical style; a driver translates them for its engine
    with :func:`translate_placeholders`. Values are bound, never interpolated; the column
    names are interpolated and must be TRUSTED identifiers, as the request model enforces.

    :param where: column -> value; empty or ``None`` filters nothing
    :ptype where: Mapping[str, str] | None
    :return: the fragment (empty when there are no filters) and its values in order
    :rtype: tuple[str, list[str]]
    """
    # unquoted, as the statement's relation and key columns are: every name in it folds case alike
    clause, values = equality_conditions(where, quote=as_written)
    return (f" WHERE {clause}" if clause else ""), values


def build_filter(
    where: Mapping[str, FilterValue] | None, where_in: Mapping[str, Sequence[FilterValue]] | None = None
) -> tuple[str, list[FilterValue]]:
    """a `` WHERE`` fragment keeping the rows whose columns equal ``where``'s values and are among ``where_in``'s.

    As :func:`build_equality_filter`, with set membership too: ``column IN ($n, ...)``, each value
    bound (:func:`~threetears.core.sql_fragments.filter_conditions`). An empty set keeps no row.

    :param where: column -> value
    :ptype where: Mapping[str, FilterValue] | None
    :param where_in: column -> the values it may hold
    :ptype where_in: Mapping[str, Sequence[FilterValue]] | None
    :return: the fragment (empty when there are no filters) and its values in order
    :rtype: tuple[str, list[FilterValue]]
    """
    clause, values = filter_conditions(where, where_in, quote=as_written)
    return (f" WHERE {clause}" if clause else ""), values


#: render the ordering key of one row as a single text value, NULLs distinguished: the
#: dialect-independent half of a relation fingerprint. One rule for every engine and for the
#: L1 copies proven against them, so it lives in :mod:`threetears.core.fingerprint`.
build_relation_key_expression = relation_key_expression
