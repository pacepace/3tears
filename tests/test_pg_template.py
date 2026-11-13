"""``tests.pg_template``'s emptying order: a kept schema's tables are emptied children first, so no delete trips a
foreign key, and a cycle of foreign keys (no such order) is refused, so the schema is migrated afresh instead."""

from __future__ import annotations

import pytest

from tests.pg_template import emptying_order


def test_each_table_comes_before_every_table_it_refers_to() -> None:
    references = [("children", "parents"), ("grandchildren", "children"), ("notes", "parents")]
    order = emptying_order(["parents", "children", "grandchildren", "notes", "alone"], references)
    assert sorted(order) == ["alone", "children", "grandchildren", "notes", "parents"]
    for child, parent in references:
        assert order.index(child) < order.index(parent), order


def test_a_table_referring_to_itself_needs_no_order() -> None:
    assert emptying_order(["tree"], [("tree", "tree")]) == ["tree"]


def test_a_reference_to_a_table_not_emptied_is_no_constraint() -> None:
    """a table the migrations wrote rows into is checked, never emptied: a reference to it orders nothing."""
    assert emptying_order(["children"], [("children", "applied")]) == ["children"]


def test_a_cycle_of_foreign_keys_is_refused() -> None:
    with pytest.raises(ValueError, match="cycle"):
        emptying_order(["a", "b"], [("a", "b"), ("b", "a")])
