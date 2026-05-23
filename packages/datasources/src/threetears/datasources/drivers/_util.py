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
]

PlaceholderStyle = Literal["asyncpg", "pyformat", "numeric", "named-at"]


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
