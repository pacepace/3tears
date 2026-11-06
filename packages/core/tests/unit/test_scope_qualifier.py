"""the scope qualifier the hub and a tool both compute, so a qualified edge token and URL name one scope."""

from __future__ import annotations

from threetears.core.http_cache import SCOPE_QUALIFIER_SEPARATOR, qualified_scope, scope_qualifier


def test_the_qualifier_ignores_order_and_repeats() -> None:
    assert scope_qualifier(["results", "turnout"]) == scope_qualifier(["turnout", "results", "results"])


def test_different_children_qualify_differently() -> None:
    assert scope_qualifier(["results"]) != scope_qualifier(["results", "turnout"])


def test_the_qualifier_never_carries_a_url_scope_separator() -> None:
    qualifier = scope_qualifier(["a-b", "c"])
    assert "-" not in qualifier and SCOPE_QUALIFIER_SEPARATOR not in qualifier and len(qualifier) == 12


def test_a_reader_of_every_child_keeps_the_scope() -> None:
    assert qualified_scope("va", None) == "va"
    assert qualified_scope("va", "3f9a1c2b7e04") == "va.3f9a1c2b7e04"
