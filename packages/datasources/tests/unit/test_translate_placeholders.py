"""tests for :func:`_translate_placeholders`.

every supported style:

- ``"asyncpg"`` -- no-op
- ``"pyformat"`` -- ``$N`` -> ``%s``
- ``"numeric"`` -- ``$N`` -> ``:N``
- ``"named-at"`` -- ``$N`` -> ``@pN``

edge cases:

- ``$10`` and ``$1`` distinguished (longest-match)
- escaped ``$$`` preserved verbatim
- ``'$N'`` inside single-quoted string literal preserved
- ``"$N"`` inside double-quoted identifier preserved
- doubled-quote escape inside literal (``'it''s'``) doesn't trip the
  literal tracker
"""

from __future__ import annotations

import pytest

from threetears.datasources.drivers._util import _translate_placeholders


class TestAsyncpgStyle:
    """asyncpg style is a no-op; verifies the dispatch path runs cleanly."""

    def test_simple_passthrough(self) -> None:
        sql = "SELECT * FROM t WHERE a = $1 AND b = $2"
        assert _translate_placeholders(sql, "asyncpg") == sql

    def test_dollar_one_unchanged(self) -> None:
        assert _translate_placeholders("$1", "asyncpg") == "$1"


class TestPyformatStyle:
    """``$N`` -> ``%s`` (positional pyformat)."""

    def test_simple(self) -> None:
        sql = "SELECT * FROM t WHERE a = $1 AND b = $2"
        expected = "SELECT * FROM t WHERE a = %s AND b = %s"
        assert _translate_placeholders(sql, "pyformat") == expected

    def test_repeated_position(self) -> None:
        # pyformat doesn't reuse positions; each $N becomes its own %s
        sql = "SELECT $1, $1"
        assert _translate_placeholders(sql, "pyformat") == "SELECT %s, %s"


class TestNumericStyle:
    """``$N`` -> ``:N``."""

    def test_simple(self) -> None:
        sql = "SELECT * FROM t WHERE a = $1 AND b = $2"
        expected = "SELECT * FROM t WHERE a = :1 AND b = :2"
        assert _translate_placeholders(sql, "numeric") == expected


class TestNamedAtStyle:
    """``$N`` -> ``@pN`` (BigQuery)."""

    def test_simple(self) -> None:
        sql = "SELECT * FROM t WHERE a = $1 AND b = $2"
        expected = "SELECT * FROM t WHERE a = @p1 AND b = @p2"
        assert _translate_placeholders(sql, "named-at") == expected


class TestEdgeCaseDoubleDigit:
    """``$10`` and ``$1`` in the same SQL parse as distinct positions."""

    def test_pyformat(self) -> None:
        sql = "SELECT $1, $10, $2"
        assert _translate_placeholders(sql, "pyformat") == "SELECT %s, %s, %s"

    def test_numeric_preserves_position(self) -> None:
        sql = "SELECT $1, $10, $2"
        assert _translate_placeholders(sql, "numeric") == "SELECT :1, :10, :2"

    def test_named_at_preserves_position(self) -> None:
        sql = "SELECT $1, $10, $2"
        assert _translate_placeholders(sql, "named-at") == "SELECT @p1, @p10, @p2"


class TestEdgeCaseDoubleDollar:
    """``$$`` sequence is preserved verbatim (SQL dollar-quote opener)."""

    def test_double_dollar_not_touched_pyformat(self) -> None:
        sql = "SELECT $$body$$, $1"
        assert _translate_placeholders(sql, "pyformat") == "SELECT $$body$$, %s"

    def test_double_dollar_not_touched_numeric(self) -> None:
        sql = "SELECT $$body$$, $1"
        assert _translate_placeholders(sql, "numeric") == "SELECT $$body$$, :1"

    def test_double_dollar_with_no_placeholders(self) -> None:
        sql = "SELECT $$dollar-quoted$$"
        assert _translate_placeholders(sql, "pyformat") == "SELECT $$dollar-quoted$$"


class TestEdgeCaseStringLiteral:
    """``'$N'`` / ``"$N"`` inside literals preserved verbatim."""

    def test_single_quoted_literal_preserved(self) -> None:
        sql = "SELECT '$1' AS literal, $1 AS bound"
        assert _translate_placeholders(sql, "pyformat") == "SELECT '$1' AS literal, %s AS bound"

    def test_double_quoted_identifier_preserved(self) -> None:
        sql = 'SELECT "$1" AS "col", $1 AS bound'
        assert _translate_placeholders(sql, "pyformat") == 'SELECT "$1" AS "col", %s AS bound'

    def test_doubled_quote_escape_in_literal(self) -> None:
        """``'it''s'`` is a single literal containing one apostrophe."""
        sql = "SELECT 'it''s a $1', $1"
        # the literal is everything between the outermost quotes;
        # inside-literal $1 stays; outside-literal $1 translates
        assert _translate_placeholders(sql, "pyformat") == "SELECT 'it''s a $1', %s"

    def test_numeric_inside_literal_not_touched(self) -> None:
        sql = "SELECT '$10', $10"
        assert _translate_placeholders(sql, "numeric") == "SELECT '$10', :10"


class TestUnknownStyle:
    """unknown style raises ValueError."""

    def test_raises_on_unknown_style(self) -> None:
        with pytest.raises(ValueError, match="unknown placeholder style"):
            _translate_placeholders("SELECT $1", "bogus")  # type: ignore[arg-type]


class TestNoPlaceholders:
    """SQL with no placeholders passes through unchanged for every style."""

    @pytest.mark.parametrize("style", ["asyncpg", "pyformat", "numeric", "named-at"])
    def test_no_placeholder_passthrough(self, style: str) -> None:
        sql = "SELECT 1"
        assert _translate_placeholders(sql, style) == sql  # type: ignore[arg-type]
