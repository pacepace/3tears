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
    "build_search_path_value",
    "build_set_search_path_sql",
]

PlaceholderStyle = Literal["asyncpg", "pyformat", "numeric", "named-at"]


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
