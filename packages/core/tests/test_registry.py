"""Tests for CollectionRegistry."""

from __future__ import annotations

from unittest.mock import MagicMock

from threetears.core.collections.registry import CollectionRegistry


def _make_mock_collection(table_name: str) -> MagicMock:
    coll = MagicMock()
    coll.table_name = table_name
    return coll


class TestCollectionRegistry:
    """Tests for CollectionRegistry."""

    def test_register_and_lookup(self) -> None:
        registry = CollectionRegistry()
        coll = _make_mock_collection("users")
        registry.register(coll)

        assert registry.get_collection("users") is coll

    def test_get_collection_returns_none_for_unregistered(self) -> None:
        registry = CollectionRegistry()

        assert registry.get_collection("nonexistent") is None

    def test_configure_sets_defaults(self) -> None:
        registry = CollectionRegistry()
        l1 = MagicMock()
        l2 = MagicMock()
        l3 = MagicMock()

        registry.configure(l1_backend=l1, l2_client=l2, l3_pool=l3)

        assert registry.get_l1_backend("any_table") is l1
        assert registry.get_l2_client("any_table") is l2
        assert registry.get_l3_pool("any_table") is l3

    def test_get_l1_backend_returns_default(self) -> None:
        registry = CollectionRegistry()
        default_l1 = MagicMock()
        registry.configure(l1_backend=default_l1)

        assert registry.get_l1_backend("messages") is default_l1

    def test_per_collection_override(self) -> None:
        registry = CollectionRegistry()
        default_l1 = MagicMock()
        override_l1 = MagicMock()
        registry.configure(l1_backend=default_l1)

        coll = _make_mock_collection("special_table")
        registry.register(coll, l1_backend=override_l1)

        assert registry.get_l1_backend("special_table") is override_l1
        assert registry.get_l1_backend("other_table") is default_l1

    def test_per_collection_override_l2(self) -> None:
        registry = CollectionRegistry()
        default_l2 = MagicMock()
        override_l2 = MagicMock()
        registry.configure(l2_client=default_l2)

        coll = _make_mock_collection("cached_table")
        registry.register(coll, l2_client=override_l2)

        assert registry.get_l2_client("cached_table") is override_l2
        assert registry.get_l2_client("other_table") is default_l2

    def test_per_collection_override_l3(self) -> None:
        registry = CollectionRegistry()
        default_l3 = MagicMock()
        override_l3 = MagicMock()
        registry.configure(l3_pool=default_l3)

        coll = _make_mock_collection("sharded_table")
        registry.register(coll, l3_pool=override_l3)

        assert registry.get_l3_pool("sharded_table") is override_l3
        assert registry.get_l3_pool("other_table") is default_l3

    def test_clear_removes_all(self) -> None:
        registry = CollectionRegistry()
        coll1 = _make_mock_collection("t1")
        coll2 = _make_mock_collection("t2")
        registry.register(coll1, l1_backend=MagicMock())
        registry.register(coll2)

        registry.clear()

        assert registry.get_collection("t1") is None
        assert registry.get_collection("t2") is None
        # Defaults are NOT cleared by clear()
        # Overrides are cleared
        assert registry._overrides == {}

    def test_configure_partial_update(self) -> None:
        """Calling configure multiple times only updates provided fields."""
        registry = CollectionRegistry()
        l1 = MagicMock()
        l2 = MagicMock()

        registry.configure(l1_backend=l1)
        registry.configure(l2_client=l2)

        assert registry.get_l1_backend("any") is l1
        assert registry.get_l2_client("any") is l2

    def test_defaults_are_none_initially(self) -> None:
        registry = CollectionRegistry()

        assert registry.get_l1_backend("any") is None
        assert registry.get_l2_client("any") is None
        assert registry.get_l3_pool("any") is None
