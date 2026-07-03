"""Tests for the shared ``metadata`` state-channel reducer :func:`merge_metadata`.

Pins the compose-not-clobber contract the before-model injection middleware
(memory / schema / knowledge) rely on: disjoint top-level keys survive, either side
may be ``None``, and the result is a fresh dict (the inputs are not mutated).
"""

from __future__ import annotations

from threetears.langgraph import merge_metadata


class TestMergeMetadata:
    def test_composes_disjoint_keys(self) -> None:
        existing = {"surfaced_memory_ids": ["m1"]}
        update = {"governed_knowledge_block": "block"}
        merged = merge_metadata(existing, update)
        assert merged == {"surfaced_memory_ids": ["m1"], "governed_knowledge_block": "block"}

    def test_update_wins_on_key_collision(self) -> None:
        assert merge_metadata({"k": "old"}, {"k": "new"}) == {"k": "new"}

    def test_none_existing_treated_as_empty(self) -> None:
        assert merge_metadata(None, {"k": "v"}) == {"k": "v"}

    def test_none_update_treated_as_empty(self) -> None:
        assert merge_metadata({"k": "v"}, None) == {"k": "v"}

    def test_both_none_is_empty(self) -> None:
        assert merge_metadata(None, None) == {}

    def test_does_not_mutate_inputs(self) -> None:
        existing = {"a": 1}
        update = {"b": 2}
        merged = merge_metadata(existing, update)
        assert existing == {"a": 1}
        assert update == {"b": 2}
        assert merged is not existing
        assert merged is not update
