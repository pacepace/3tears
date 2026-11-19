"""the walk's and the rewriter's one reading of a statement's placeholders (``no_whole_reads.placeholders``)."""

from __future__ import annotations

import pytest

from threetears.enforcement.no_whole_reads.placeholders import (
    placeholder_numbers,
    renumbered_with,
    without_hidden_dollars,
)

#: text holding a ``$`` that is no placeholder, each as SQL writes it
_HIDING = ("'over $1 billion'", '"col$1"', "$$ $1 $$", "-- $1\n", "/* $1 */", "E'it\\'s $1'", "'40%'")


@pytest.mark.parametrize("text", _HIDING)
def test_a_dollar_inside_a_literal_a_name_or_a_comment_is_no_placeholder(text: str) -> None:
    sql = f"a = $2 AND b = {text} AND c = $3"
    assert placeholder_numbers(sql) == [2, 3]
    assert renumbered_with(sql, lambda number: number - 1) == f"a = $1 AND b = {text} AND c = $2"
    assert "$1" not in without_hidden_dollars(sql).replace("$2", "").replace("$3", "")
    assert without_hidden_dollars(sql).startswith("a = $2 AND b = ") and without_hidden_dollars(sql).endswith("c = $3")
