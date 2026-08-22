"""tests for the shared registry-detection AST vocabulary.

These helpers back two gates (L2 scope, invalidation listener), so a change here moves
both. That is the point of sharing them, and the reason they need coverage of their own
rather than only through whichever domain happens to exercise a shape.
"""

from __future__ import annotations

import ast

from threetears.enforcement.common import (
    CLIENT_SPELLINGS,
    argument_spellings,
    callee_names,
    constructed_registries,
    constructed_registry_lines,
    dotted,
    l2_live_registries,
    names_a_live_client,
    receiver,
)


def _call(source: str) -> ast.Call:
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.Expr)
    assert isinstance(node.value, ast.Call)
    return node.value


class TestDotted:
    """the spelling function every other helper is built on."""

    def test_a_bare_name_resolves(self) -> None:
        assert dotted(ast.parse("registry").body[0].value) == "registry"  # type: ignore[attr-defined]

    def test_an_attribute_chain_resolves(self) -> None:
        assert dotted(ast.parse("self._registry").body[0].value) == "self._registry"  # type: ignore[attr-defined]

    def test_a_deep_chain_resolves(self) -> None:
        assert dotted(ast.parse("a.b.c.d").body[0].value) == "a.b.c.d"  # type: ignore[attr-defined]

    def test_a_non_name_rooted_expression_does_not(self) -> None:
        """``build()[0]`` has no static name, and guessing one would be worse than None."""
        assert dotted(ast.parse("build()[0]").body[0].value) is None  # type: ignore[attr-defined]


class TestCalleeAndReceiver:
    def test_a_bare_call_has_a_callee_and_no_receiver(self) -> None:
        call = _call("configure(x)")

        assert callee_names(call) == frozenset({"configure"})
        assert receiver(call) is None

    def test_a_method_call_has_both(self) -> None:
        call = _call("self._registry.configure(x)")

        assert callee_names(call) == frozenset({"configure"})
        assert receiver(call) == "self._registry"


class TestArgumentSpellings:
    def test_positional_and_keyword_values_are_both_collected(self) -> None:
        call = _call("configure(nc, l3_pool=pool)")

        assert argument_spellings(call) == frozenset({"nc", "pool"})

    def test_unspellable_arguments_are_dropped_rather_than_guessed(self) -> None:
        call = _call("configure(build()[0], nc)")

        assert argument_spellings(call) == frozenset({"nc"})


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
