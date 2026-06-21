"""Tests for CollectionRegistry."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.registry import CollectionRegistry


def _make_mock_collection(table_name: str) -> MagicMock:
    coll = MagicMock()
    coll.table_name = table_name
    return coll


def _underlying_l3(resolved: Any) -> Any:
    """unwrap a resolved L3 backend to the raw pool it was configured with.

    ``configure`` / ``bind_table`` / ``register`` normalize a raw L3 transport (a
    bare pool) to a :class:`SqlL3Backend` so the resolved backend exposes the
    structured ``DurableStore`` ops the collection CRUD lifecycle needs (L3B-03). A
    backend that already satisfies ``DurableStore`` passes through un-wrapped. These
    routing/isolation/override tests assert WHICH pool reaches WHICH table; this
    helper peels the wrapper so the identity assertion targets the configured pool.

    :param resolved: the value returned by ``get_l3_pool``.
    :ptype resolved: Any
    :return: the raw pool the backend wraps, or ``resolved`` unchanged.
    :rtype: Any
    """
    if isinstance(resolved, SqlL3Backend):
        return resolved._pool  # noqa: SLF001 -- peel the wrapper to the configured raw pool
    return resolved


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
        assert _underlying_l3(registry.get_l3_pool("any_table")) is l3

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

        assert _underlying_l3(registry.get_l3_pool("sharded_table")) is override_l3
        assert _underlying_l3(registry.get_l3_pool("other_table")) is default_l3

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
        # Overrides are cleared: t1's per-table l1 override registered
        # above must no longer win over the (absent) default; since no
        # default is set, the public get_l1_backend returns None.
        assert registry.get_l1_backend("t1") is None

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
        assert _underlying_l3(registry.get_l3_pool("any")) is None


class TestBindTable:
    """tests for :meth:`CollectionRegistry.bind_table` (Phase C2)."""

    def test_bind_table_l3_pool_overrides_default(self) -> None:
        """per-table l3 override wins over the registry default."""
        registry = CollectionRegistry()
        default_l3 = MagicMock()
        override_l3 = MagicMock()
        registry.configure(l3_pool=default_l3)

        registry.bind_table("groups", l3_pool=override_l3)

        assert _underlying_l3(registry.get_l3_pool("groups")) is override_l3
        assert _underlying_l3(registry.get_l3_pool("conversations")) is default_l3

    def test_bind_table_accepts_pool_without_instance(self) -> None:
        """bind_table pins a pool BEFORE any collection is constructed."""
        registry = CollectionRegistry()
        pool = MagicMock()

        registry.bind_table("roles", l3_pool=pool)

        # subsequent register() calls merge with the earlier binding
        # rather than overwriting it
        coll = _make_mock_collection("roles")
        registry.register(coll)
        assert _underlying_l3(registry.get_l3_pool("roles")) is pool

    def test_bind_table_layers_l1_and_l3_independently(self) -> None:
        """l1 and l3 bindings on the same table are independent."""
        registry = CollectionRegistry()
        l1_override = MagicMock()
        l3_override = MagicMock()

        registry.bind_table("namespaces", l1_backend=l1_override)
        registry.bind_table("namespaces", l3_pool=l3_override)

        assert registry.get_l1_backend("namespaces") is l1_override
        assert _underlying_l3(registry.get_l3_pool("namespaces")) is l3_override

    def test_bind_table_no_op_when_every_arg_none(self) -> None:
        """bind_table with no overrides leaves existing overrides untouched."""
        registry = CollectionRegistry()
        pool = MagicMock()
        registry.bind_table("roles", l3_pool=pool)

        registry.bind_table("roles")

        assert _underlying_l3(registry.get_l3_pool("roles")) is pool

    def test_bind_table_isolates_to_named_table(self) -> None:
        """per-table binding never leaks onto an unrelated table."""
        registry = CollectionRegistry()
        default_l3 = MagicMock()
        registry.configure(l3_pool=default_l3)
        rbac_pool = MagicMock()

        registry.bind_table("groups", l3_pool=rbac_pool)

        assert _underlying_l3(registry.get_l3_pool("groups")) is rbac_pool
        assert _underlying_l3(registry.get_l3_pool("workspace_files")) is default_l3
        assert _underlying_l3(registry.get_l3_pool("memories")) is default_l3
