"""unit tests for :mod:`threetears.agent.memory.retrieval_scope`.

the builder is pure -- conditions, params, ordering -- and the empty-is-
not-absent rule is a security property (an empty allow-list must never
read as "no constraint"), so both get pinned without a database.
"""

from __future__ import annotations

from uuid import uuid4

from threetears.agent.memory.retrieval_scope import (
    RetrievalScope,
    build_scope_conditions,
    scope_matches_nothing,
)


class TestScopeMatchesNothing:
    """empty is not absent -- the fail-closed rule, named."""

    def test_none_scope_matches_everything(self) -> None:
        assert scope_matches_nothing(None) is False

    def test_default_scope_matches_everything(self) -> None:
        assert scope_matches_nothing(RetrievalScope()) is False

    def test_empty_allow_list_matches_nothing(self) -> None:
        assert scope_matches_nothing(RetrievalScope(restrict_to_ids=frozenset())) is True

    def test_empty_tag_tuples_match_nothing(self) -> None:
        assert scope_matches_nothing(RetrievalScope(tags_any=())) is True
        assert scope_matches_nothing(RetrievalScope(tags_all=())) is True

    def test_empty_exclude_list_excludes_nothing(self) -> None:
        """excluding zero rows constrains nothing -- NOT empty-eligibility."""
        assert scope_matches_nothing(RetrievalScope(exclude_ids=frozenset())) is False


class TestBuildScopeConditions:
    """deterministic fragments, params and numbering."""

    def test_none_scope_builds_nothing(self) -> None:
        conditions, params, next_param = build_scope_conditions(None, next_param=4)
        assert conditions == []
        assert params == []
        assert next_param == 4

    def test_every_field_renders_in_fixed_order_with_sequential_params(self) -> None:
        keep = uuid4()
        drop = uuid4()
        scope = RetrievalScope(
            tags_any=("a", "b"),
            tags_all=("c",),
            restrict_to_ids=frozenset({keep}),
            exclude_ids=frozenset({drop}),
        )
        conditions, params, next_param = build_scope_conditions(scope, next_param=7)
        assert conditions == [
            "tags ?| $7::text[]",
            "tags @> $8::jsonb",
            "memory_id = ANY($9::uuid[])",
            "NOT (memory_id = ANY($10::uuid[]))",
        ]
        assert params == [["a", "b"], ["c"], [keep], [drop]]
        assert next_param == 11

    def test_a_single_field_takes_the_first_free_slot(self) -> None:
        conditions, params, next_param = build_scope_conditions(RetrievalScope(tags_all=("x",)), next_param=5)
        assert conditions == ["tags @> $5::jsonb"]
        assert params == [["x"]]
        assert next_param == 6
