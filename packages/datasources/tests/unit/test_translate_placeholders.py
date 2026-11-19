"""tests for :func:`translate_placeholders`.

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

from threetears.datasources.drivers.sql_fragments import translate_placeholders


class TestAsyncpgStyle:
    """asyncpg style is a no-op; verifies the dispatch path runs cleanly."""

    def test_simple_passthrough(self) -> None:
        sql = "SELECT * FROM t WHERE a = $1 AND b = $2"
        assert translate_placeholders(sql, "asyncpg") == sql

    def test_dollar_one_unchanged(self) -> None:
        assert translate_placeholders("$1", "asyncpg") == "$1"


class TestPyformatStyle:
    """``$N`` -> ``%s`` (positional pyformat)."""

    def test_simple(self) -> None:
        sql = "SELECT * FROM t WHERE a = $1 AND b = $2"
        expected = "SELECT * FROM t WHERE a = %s AND b = %s"
        assert translate_placeholders(sql, "pyformat") == expected

    def test_repeated_position(self) -> None:
        # pyformat doesn't reuse positions; each $N becomes its own %s
        sql = "SELECT $1, $1"
        assert translate_placeholders(sql, "pyformat") == "SELECT %s, %s"


class TestNumericStyle:
    """``$N`` -> ``:N``."""

    def test_simple(self) -> None:
        sql = "SELECT * FROM t WHERE a = $1 AND b = $2"
        expected = "SELECT * FROM t WHERE a = :1 AND b = :2"
        assert translate_placeholders(sql, "numeric") == expected


class TestNamedAtStyle:
    """``$N`` -> ``@pN`` (BigQuery)."""

    def test_simple(self) -> None:
        sql = "SELECT * FROM t WHERE a = $1 AND b = $2"
        expected = "SELECT * FROM t WHERE a = @p1 AND b = @p2"
        assert translate_placeholders(sql, "named-at") == expected


class TestEdgeCaseDoubleDigit:
    """``$10`` and ``$1`` in the same SQL parse as distinct positions."""

    def test_pyformat(self) -> None:
        sql = "SELECT $1, $10, $2"
        assert translate_placeholders(sql, "pyformat") == "SELECT %s, %s, %s"

    def test_numeric_preserves_position(self) -> None:
        sql = "SELECT $1, $10, $2"
        assert translate_placeholders(sql, "numeric") == "SELECT :1, :10, :2"

    def test_named_at_preserves_position(self) -> None:
        sql = "SELECT $1, $10, $2"
        assert translate_placeholders(sql, "named-at") == "SELECT @p1, @p10, @p2"


class TestEdgeCaseDoubleDollar:
    """``$$`` sequence is preserved verbatim (SQL dollar-quote opener)."""

    def test_double_dollar_not_touched_pyformat(self) -> None:
        sql = "SELECT $$body$$, $1"
        assert translate_placeholders(sql, "pyformat") == "SELECT $$body$$, %s"

    def test_double_dollar_not_touched_numeric(self) -> None:
        sql = "SELECT $$body$$, $1"
        assert translate_placeholders(sql, "numeric") == "SELECT $$body$$, :1"

    def test_double_dollar_with_no_placeholders(self) -> None:
        sql = "SELECT $$dollar-quoted$$"
        assert translate_placeholders(sql, "pyformat") == "SELECT $$dollar-quoted$$"

    def test_a_dollar_quoted_body_and_a_comment_keep_their_dollar_numbers(self) -> None:
        """Flips if a ``$1`` inside a dollar-quoted string or a comment is translated as a placeholder."""
        sql = "SELECT $$ costs $1 $$, $tag$ $2 $tag$, $1 -- not $2\n/* nor $3 */"
        assert translate_placeholders(sql, "pyformat") == (
            "SELECT $$ costs $1 $$, $tag$ $2 $tag$, %s -- not $2\n/* nor $3 */"
        )

    def test_a_race_named_with_a_dollar_amount(self) -> None:
        sql = "SELECT * FROM races WHERE name = 'Proposition: 40 - Impose one-time wealth tax over $1 billion' AND id = $1"
        assert translate_placeholders(sql, "numeric") == sql.replace("id = $1", "id = :1")


class TestEdgeCaseStringLiteral:
    """``'$N'`` / ``"$N"`` inside literals preserved verbatim."""

    def test_single_quoted_literal_preserved(self) -> None:
        sql = "SELECT '$1' AS literal, $1 AS bound"
        assert translate_placeholders(sql, "pyformat") == "SELECT '$1' AS literal, %s AS bound"

    def test_double_quoted_identifier_preserved(self) -> None:
        sql = 'SELECT "$1" AS "col", $1 AS bound'
        assert translate_placeholders(sql, "pyformat") == 'SELECT "$1" AS "col", %s AS bound'

    def test_doubled_quote_escape_in_literal(self) -> None:
        """``'it''s'`` is a single literal containing one apostrophe."""
        sql = "SELECT 'it''s a $1', $1"
        # the literal is everything between the outermost quotes;
        # inside-literal $1 stays; outside-literal $1 translates
        assert translate_placeholders(sql, "pyformat") == "SELECT 'it''s a $1', %s"

    def test_numeric_inside_literal_not_touched(self) -> None:
        sql = "SELECT '$10', $10"
        assert translate_placeholders(sql, "numeric") == "SELECT '$10', :10"


class TestUnknownStyle:
    """unknown style raises ValueError."""

    def test_raises_on_unknown_style(self) -> None:
        with pytest.raises(ValueError, match="unknown placeholder style"):
            translate_placeholders("SELECT $1", "bogus")  # type: ignore[arg-type]


class TestNoPlaceholders:
    """SQL with no placeholders passes through unchanged for every style."""

    @pytest.mark.parametrize("style", ["asyncpg", "pyformat", "numeric", "named-at"])
    def test_no_placeholder_passthrough(self, style: str) -> None:
        sql = "SELECT 1"
        assert translate_placeholders(sql, style) == sql  # type: ignore[arg-type]


class TestAPercentUnderTheFormatStyle:
    """``pyformat`` (Redshift's driver, ``%s``) reads a ``%`` outside a quoted literal, a quoted name or a comment as
    its own: a lone one is refused ("Only %s and %% are supported"), so it is doubled there; inside a literal the driver
    leaves it as it is, and so does this. A statement naming no placeholder is sent with no parameters, which the
    driver does not read for ``%`` at all: it is left whole."""

    def test_a_percent_outside_a_literal_is_doubled_and_one_inside_is_kept(self) -> None:
        """Flips if a modulo or a LIKE pattern's ``%`` reaches the driver as a lone ``%``, or one inside a literal is
        changed (the driver would then match ``%%``)."""
        sql = "SELECT g % 2, name FROM races WHERE name LIKE 'Proposition: 40%' AND state = $1 AND share > 50 % 3"
        assert translate_placeholders(sql, "pyformat") == (
            "SELECT g %% 2, name FROM races WHERE name LIKE 'Proposition: 40%' AND state = %s AND share > 50 %% 3"
        )

    def test_quoted_names_comments_and_escaped_literals_keep_their_percent(self) -> None:
        sql = "SELECT \"pct%\" -- 50% off\n, E'it\\'s 9%' /* 1% */ FROM t WHERE a = $1"
        assert translate_placeholders(sql, "pyformat") == (
            "SELECT \"pct%\" -- 50% off\n, E'it\\'s 9%' /* 1% */ FROM t WHERE a = %s"
        )

    def test_a_statement_naming_no_placeholder_is_left_whole(self) -> None:
        """sent with no parameters, the driver reads no ``%`` in it: doubled, one would reach the warehouse as two."""
        sql = "SELECT g % 2 FROM t WHERE name LIKE 'CA%'"
        assert translate_placeholders(sql, "pyformat") == sql

    def test_the_other_styles_leave_a_percent_alone(self) -> None:
        sql = "SELECT g % 2 FROM t WHERE a = $1"
        assert translate_placeholders(sql, "numeric") == "SELECT g % 2 FROM t WHERE a = :1"
        assert translate_placeholders(sql, "named-at") == "SELECT g % 2 FROM t WHERE a = @p1"
