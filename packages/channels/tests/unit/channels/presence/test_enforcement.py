"""enforcement tests for the presence state layer's hard constraints.

these guard the design decisions that are easy to reflexively violate:

- membership/presence state must live in the L1+L2 collection, NOT in a
  dict — the ONLY dict-typed instance attribute permitted in the
  presence modules is :class:`RoomState`'s synchronized ``_sockets``
  live-handle map (and the lock-management helpers). a new dict holding
  room/membership/presence state is the racy thing being replaced.
- the presence collections stay PK-keyed — no secondary-field query
  surface (``BaseCollection`` has none; the KV wrapper has no listing;
  ``SchemaBackedCollection`` is L3-bound). a ``WHERE`` / ``query`` /
  ``filter`` / ``scan`` method on a presence collection would mean a
  secondary scan crept in.
"""

from __future__ import annotations

import ast
import inspect

from threetears.channels.presence import collection as collection_mod
from threetears.channels.presence import room_state as room_state_mod
from threetears.channels.presence import sweeper as sweeper_mod

# the one sanctioned in-process dict: RoomState's live socket-handle map.
_ALLOWED_DICT_ATTRS = {"_sockets"}


def _assigned_dict_attributes(module: object) -> set[str]:
    """collect ``self.<attr> = {...}`` / ``self.<attr>: dict = ...`` names.

    walks every method body in the module for an attribute assigned a
    dict literal or annotated as a dict — the shape that would hold
    shared state.
    """
    source = inspect.getsource(module)  # type: ignore[arg-type]
    tree = ast.parse(source)
    found: set[str] = set()

    for node in ast.walk(tree):
        # self.attr = {...}
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    found.add(target.attr)
        # self.attr: dict[...] = ...
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Attribute):
            target = node.target
            if isinstance(target.value, ast.Name) and target.value.id == "self":
                annotation = ast.unparse(node.annotation) if node.annotation is not None else ""
                if annotation.startswith("dict"):
                    found.add(target.attr)
    return found


class TestNoDictForSharedState:
    """no dict-typed instance attribute holds room/membership/presence state."""

    def test_room_state_only_socket_map_is_a_dict(self) -> None:
        offenders = _assigned_dict_attributes(room_state_mod) - _ALLOWED_DICT_ATTRS
        assert offenders == set(), (
            f"RoomState declares dict-typed instance attribute(s) {offenders}; "
            "membership/presence state belongs in the PresenceCollection, "
            "only the live-socket map (_sockets) may be a dict"
        )

    def test_collection_holds_no_state_dict(self) -> None:
        offenders = _assigned_dict_attributes(collection_mod) - _ALLOWED_DICT_ATTRS
        assert offenders == set(), (
            f"presence collection module declares dict-typed instance attribute(s) {offenders}; "
            "state must live in L1+L2, not a dict"
        )

    def test_sweeper_holds_no_state_dict(self) -> None:
        offenders = _assigned_dict_attributes(sweeper_mod) - _ALLOWED_DICT_ATTRS
        assert offenders == set(), (
            f"presence sweeper declares dict-typed instance attribute(s) {offenders}; "
            "tracked-id state is a set, presence state lives in L1+L2"
        )


class TestNoSecondaryQuery:
    """the presence collections expose no secondary-field query surface."""

    def test_no_secondary_query_methods(self) -> None:
        from threetears.channels.presence.collection import (
            PresenceConnectionCollection,
            RoomIndexCollection,
        )

        banned = {"where", "query", "filter", "scan", "find", "search", "list_keys", "keys", "all"}
        for cls in (PresenceConnectionCollection, RoomIndexCollection):
            method_names = {name for name in dir(cls) if not name.startswith("__")}
            offenders = method_names & banned
            assert offenders == set(), (
                f"{cls.__name__} exposes secondary-query method(s) {offenders}; "
                "presence collections must stay strictly pk-keyed"
            )

    def test_room_membership_is_a_pk_get(self) -> None:
        """``members`` resolves the room set by pk-get, never a scan.

        the room-index is itself pk-keyed; ``members`` reads it via the
        single-key ``get`` path. assert the collections declare exactly
        a single-column pk so no composite/secondary key shape crept in.
        """
        from threetears.channels.presence import create_presence_l1_backend
        from threetears.channels.presence.collection import PresenceCollection
        from threetears.core.collections.registry import CollectionRegistry
        from threetears.core.config import DefaultCoreConfig

        registry = CollectionRegistry()
        registry.configure(l1_backend=create_presence_l1_backend())
        collection = PresenceCollection(registry, DefaultCoreConfig(collection_flush="ALWAYS"))
        assert collection.connections.primary_key_columns == ("connection_id",)
        assert collection.rooms.primary_key_columns == ("room_id",)
