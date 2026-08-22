"""tests for the registry-specific half of the shared AST vocabulary.

The generic spelling helpers (``dotted`` and friends) moved to ``ast_helpers`` and are
tested there; what stays here is the part that knows what a ``CollectionRegistry`` is.

These helpers back two gates (L2 scope, invalidation listener), so a change here moves
both. That is the point of sharing them, and the reason they need coverage of their own
rather than only through whichever domain happens to exercise a shape.
"""

from __future__ import annotations

import ast

from threetears.enforcement.common import (
    CLIENT_SPELLINGS,
    constructed_registries,
    constructed_registry_lines,
    l2_live_registries,
    names_a_live_client,
)


def _call(source: str) -> ast.Call:
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.Expr)
    assert isinstance(node.value, ast.Call)
    return node.value


class TestNamesALiveClient:
    """the heuristic half, and the one that grows."""

    def test_a_client_keyword_counts(self) -> None:
        assert names_a_live_client(_call("configure(l2_client=x)")) is True

    def test_the_per_collection_keyword_counts(self) -> None:
        """``nats_client=`` WINS over the registry default, so it must count."""
        assert names_a_live_client(_call("Collection(registry=r, nats_client=x)")) is True

    def test_an_explicit_none_does_not(self) -> None:
        """``nats_client=None`` is the deliberate opt-out, not a live client."""
        assert names_a_live_client(_call("Collection(registry=r, nats_client=None)")) is False

    def test_a_positional_client_counts_by_spelling(self) -> None:
        assert names_a_live_client(_call("configure(nc)")) is True

    def test_a_dotted_positional_client_matches_on_its_last_segment(self) -> None:
        assert names_a_live_client(_call("configure(self._nc)")) is True

    def test_an_unrelated_argument_does_not(self) -> None:
        assert names_a_live_client(_call("configure(pool)")) is False

    def test_every_declared_spelling_is_recognised(self) -> None:
        """the constant and the matcher must not drift apart."""
        for spelling in CLIENT_SPELLINGS:
            assert names_a_live_client(_call(f"configure({spelling})")) is True, spelling


class TestConstructedRegistries:
    def test_a_plain_assignment_is_tracked(self) -> None:
        tree = ast.parse("registry = CollectionRegistry()\n")

        assert constructed_registries(tree) == frozenset({"registry"})

    def test_an_annotated_assignment_is_tracked(self) -> None:
        tree = ast.parse("registry: CollectionRegistry = CollectionRegistry()\n")

        assert constructed_registries(tree) == frozenset({"registry"})

    def test_the_construction_line_is_recorded(self) -> None:
        """violations report against the construction site, which stays put."""
        tree = ast.parse("import os\n\nregistry = CollectionRegistry()\n")

        assert constructed_registry_lines(tree) == {"registry": 3}

    def test_the_first_construction_wins_when_a_name_is_rebound(self) -> None:
        tree = ast.parse("registry = CollectionRegistry()\nregistry = CollectionRegistry()\n")

        assert constructed_registry_lines(tree) == {"registry": 1}

    def test_something_else_entirely_is_not_a_registry(self) -> None:
        tree = ast.parse("registry = SomethingElse()\n")

        assert constructed_registries(tree) == frozenset()


class TestL2LiveRegistries:
    """the question both gates ask."""

    def test_a_configured_client_makes_it_live(self) -> None:
        tree = ast.parse("r = CollectionRegistry()\nr.configure(l2_client=nc)\n")

        assert l2_live_registries(tree) == frozenset({"r"})

    def test_bind_table_makes_it_live(self) -> None:
        tree = ast.parse("r = CollectionRegistry()\nr.bind_table('t', l2_client=nc)\n")

        assert l2_live_registries(tree) == frozenset({"r"})

    def test_a_collection_taking_the_client_directly_makes_it_live(self) -> None:
        """the shape a ``configure(l2_client=)`` sweep cannot see."""
        tree = ast.parse("r = CollectionRegistry()\nc = Coll(registry=r, nats_client=nc)\n")

        assert l2_live_registries(tree) == frozenset({"r"})

    def test_an_l3_only_registry_is_not_live(self) -> None:
        tree = ast.parse("r = CollectionRegistry()\nr.configure(l3_pool=pool)\n")

        assert l2_live_registries(tree) == frozenset()

    def test_a_registry_never_constructed_here_is_not_reported(self) -> None:
        """a module that only USES an injected registry owns no wiring decision."""
        tree = ast.parse("r.configure(l2_client=nc)\n")

        assert l2_live_registries(tree) == frozenset()
