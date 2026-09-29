"""unit tests for :func:`threetears.core.data.gin.gin_filter`."""

from __future__ import annotations

import pytest

from threetears.core.data import gin_filter


class TestGinFilter:
    """an any-of GIN predicate is rendered so the planner cannot serve it from the index."""

    def test_the_predicate_becomes_a_boolean_test(self) -> None:
        """``IS TRUE`` is not an index-matchable clause, so the predicate filters rows instead."""
        assert (
            gin_filter("search_vector @@ websearch_to_tsquery('english', $3)")
            == "(search_vector @@ websearch_to_tsquery('english', $3)) IS TRUE"
        )

    def test_surrounding_whitespace_is_not_carried_into_the_fragment(self) -> None:
        assert gin_filter("  tags ?| $4::text[]  ") == "(tags ?| $4::text[]) IS TRUE"

    @pytest.mark.parametrize("predicate", ["", "   "])
    def test_an_empty_predicate_is_refused(self, predicate: str) -> None:
        """an empty fragment would render ``() IS TRUE``, which is a syntax error far from its cause."""
        with pytest.raises(ValueError, match="predicate"):
            gin_filter(predicate)
