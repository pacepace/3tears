"""the one spelling of an identifier and of an equality filter, which every tier builds SQL with."""

from __future__ import annotations

from threetears.core.sql_fragments import as_written, equality_conditions, quote_identifier


def test_an_identifier_is_quoted_with_any_quote_in_it_doubled() -> None:
    assert quote_identifier("% of Exp. In") == '"% of Exp. In"'
    assert quote_identifier('a"b') == '"a""b"'


def test_equality_conditions_quote_each_column_and_number_placeholders_from_first() -> None:
    assert equality_conditions({"State": "TX", "geo": 3}, first=2) == ('"State" = $2 AND "geo" = $3', ["TX", 3])


def test_equality_conditions_can_leave_columns_as_written_for_an_unquoted_statement() -> None:
    assert equality_conditions({"state": "TX"}, quote=as_written) == ("state = $1", ["TX"])


def test_no_filters_is_no_condition() -> None:
    assert equality_conditions(None) == ("", [])
