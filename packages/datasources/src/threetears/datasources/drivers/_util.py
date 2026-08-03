"""shared driver utilities: placeholder translation across SQL dialects.

every concrete driver in :mod:`threetears.datasources.drivers` reuses
:func:`_translate_placeholders` instead of reimplementing the regex
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

import re
from typing import Literal

__all__ = [
    "PlaceholderStyle",
    "build_reset_statement_timeout_sql",
    "build_search_path_value",
    "build_set_local_statement_timeout_sql",
    "build_set_search_path_sql",
    "build_set_statement_timeout_sql",
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


def _quote_pg_identifier(name: str) -> str:
    """quote a single Postgres / Redshift identifier safely.

    wraps the name in double quotes and escapes any internal double
    quote by doubling it, matching the SQL standard identifier-quoting
    rules accepted by Postgres, Yugabyte, and Redshift. callers MUST
    use this when interpolating user-controllable identifiers into a
    SQL fragment (e.g. schema names threaded from an agent's
    ``allowed_schemas`` config); parameter placeholders are not
    accepted for identifiers in any of these backends.

    :param name: identifier to quote
    :ptype name: str
    :return: quoted identifier (with the wrapping double quotes)
    :rtype: str
    """
    return '"' + name.replace('"', '""') + '"'


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

    each schema name is identifier-quoted via :func:`_quote_pg_identifier`
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
    return ", ".join(_quote_pg_identifier(s) for s in schemas)


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


# match ``$N`` where N is a positive integer AND it isn't part of an
# escaped ``$$`` sequence. the lookbehind ``(?<!\$)`` rejects the
# second ``$`` in ``$$`` (so ``$$1`` parses as ``$ + $1`` -- but we
# also need to skip the WHOLE ``$$`` sequence; the string-replace
# pass below handles that). string-literal protection is handled by
# splitting the SQL on quoted segments first.
_PLACEHOLDER_RE = re.compile(r"\$(\d+)")


def _split_preserving_literals(sql: str) -> list[tuple[str, bool]]:
    """split sql into ``(segment, is_literal)`` pairs.

    string literals (single-quoted and double-quoted) are NOT touched
    by placeholder translation. SQL standard string-literal escaping
    uses doubled quotes (``''`` -> literal ``'``); we honour that.

    :param sql: SQL text to split
    :ptype sql: str
    :return: list of (segment, is_string_literal) pairs in order
    :rtype: list[tuple[str, bool]]
    """
    segments: list[tuple[str, bool]] = []
    i = 0
    n = len(sql)
    buf: list[str] = []
    while i < n:
        ch = sql[i]
        if ch in ("'", '"'):
            # flush the non-literal buffer
            if buf:
                segments.append(("".join(buf), False))
                buf = []
            quote = ch
            lit = [ch]
            i += 1
            while i < n:
                ch2 = sql[i]
                lit.append(ch2)
                if ch2 == quote:
                    # check for SQL doubled-quote escape
                    if i + 1 < n and sql[i + 1] == quote:
                        lit.append(sql[i + 1])
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            segments.append(("".join(lit), True))
        else:
            buf.append(ch)
            i += 1
    if buf:
        segments.append(("".join(buf), False))
    return segments


def _translate_non_literal_segment(segment: str, target_style: PlaceholderStyle) -> str:
    """translate placeholders in a non-literal segment.

    handles ``$$`` (escaped, do not touch) by splitting the segment on
    ``$$`` first, translating each chunk, and rejoining. ``$10`` and
    ``$1`` are correctly distinguished because the regex matches the
    longest run of digits.

    :param segment: SQL segment outside any string literal
    :ptype segment: str
    :param target_style: target placeholder style
    :ptype target_style: PlaceholderStyle
    :return: translated segment
    :rtype: str
    """
    # split on $$ so we don't translate inside the escape
    parts = segment.split("$$")
    translated_parts: list[str] = []
    for part in parts:
        if target_style == "asyncpg":
            translated_parts.append(part)
        elif target_style == "pyformat":
            translated_parts.append(_PLACEHOLDER_RE.sub("%s", part))
        elif target_style == "numeric":
            translated_parts.append(_PLACEHOLDER_RE.sub(lambda m: f":{m.group(1)}", part))
        elif target_style == "named-at":
            translated_parts.append(_PLACEHOLDER_RE.sub(lambda m: f"@p{m.group(1)}", part))
        else:
            raise ValueError(f"unknown placeholder style: {target_style!r}")
    return "$$".join(translated_parts)


def _translate_placeholders(sql: str, target_style: PlaceholderStyle) -> str:
    """translate ``$N``-style placeholders to ``target_style``.

    edge cases handled:

    - ``$1`` and ``$10`` in the same SQL parse as distinct positions
      (longest-match regex on the digit run)
    - escaped ``$$`` is preserved verbatim (SQL dollar-quote opener
      stays an opener; the body is treated as a string literal by
      most engines but the translator just doesn't touch the dollar
      sequence)
    - ``'$1'`` or ``"$1"`` inside a string literal is preserved
      verbatim
    - mixed ``$1`` outside a literal AND ``'$1'`` inside is correctly
      handled per-segment

    :param sql: SQL text with ``$N``-style placeholders
    :ptype sql: str
    :param target_style: target placeholder dialect
    :ptype target_style: PlaceholderStyle
    :return: SQL with placeholders translated
    :rtype: str
    :raises ValueError: if ``target_style`` is not a recognised style
    """
    segments = _split_preserving_literals(sql)
    out: list[str] = []
    for segment, is_literal in segments:
        if is_literal:
            out.append(segment)
        else:
            out.append(_translate_non_literal_segment(segment, target_style))
    return "".join(out)
