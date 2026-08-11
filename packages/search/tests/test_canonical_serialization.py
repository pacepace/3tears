"""Canonical serialization -- the D26 keying rule, pinned (SR-F1, SR-F8).

Explicitly-set parameters only; absent and defaulted canonically identical;
stable form; criteria order canonically irrelevant. Two consumers must agree
on this form (the replay key and eval run identity), so these are contract
pins, not implementation tests.
"""

from __future__ import annotations

import hashlib
import json

from threetears.search.contracts import Criterion, SearchRequest, canonical_digest, canonicalize


def test_unset_fields_never_appear() -> None:
    """a query-only request canonicalizes to the query alone."""
    assert SearchRequest(query="capybara").canonical_form() == '{"query":"capybara"}'


def test_absent_and_defaulted_are_identical() -> None:
    """explicitly passing a field's default value changes nothing (D26)."""
    bare = SearchRequest(query="capybara")
    spelled_out = SearchRequest(query="capybara", criteria=(), fidelity=None, record=False, budget_scope_tags=())
    assert spelled_out.canonical_form() == bare.canonical_form()
    assert spelled_out.canonical_digest() == bare.canonical_digest()


def test_adding_a_defaulted_parameter_shifts_no_key() -> None:
    """the rule that makes recordings outlive contract growth: a request that
    never set a parameter has the same form whether or not the parameter
    exists with a default -- so its digest is stable across additive minors."""
    request = SearchRequest(query="capybara", record=True)
    assert json.loads(request.canonical_form()) == {"query": "capybara", "record": True}


def test_criteria_order_is_canonically_irrelevant() -> None:
    """two requests differing only in criteria order are one search."""
    a = Criterion.language("en")
    b = Criterion.domains_include(["example.org", "example.net"])
    forward = SearchRequest(query="q", criteria=(a, b))
    backward = SearchRequest(query="q", criteria=(b, a))
    assert forward.canonical_form() == backward.canonical_form()
    assert forward.canonical_digest() == backward.canonical_digest()


def test_budget_scope_tag_order_is_canonically_irrelevant() -> None:
    """scope tags name sets of scopes; order carries no meaning."""
    forward = SearchRequest(query="q", budget_scope_tags=("a", "b"))
    backward = SearchRequest(query="q", budget_scope_tags=("b", "a"))
    assert forward.canonical_form() == backward.canonical_form()


def test_different_explicit_values_diverge() -> None:
    """the form distinguishes what actually differs."""
    assert SearchRequest(query="q", record=True).canonical_digest() != SearchRequest(query="q").canonical_digest()
    assert SearchRequest(query="q").canonical_digest() != SearchRequest(query="r").canonical_digest()


def test_form_is_stable_json() -> None:
    """sorted keys, compact separators -- byte-stable for equal inputs."""
    form = SearchRequest(
        query="q",
        criteria=(Criterion.language("en"),),
        budget_scope_tags=("run:1",),
    ).canonical_form()
    parsed = json.loads(form)
    assert form == json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def test_digest_is_sha256_of_form() -> None:
    """the digest is exactly the hex SHA-256 of the canonical form."""
    request = SearchRequest(query="q", fidelity="content")
    expected = hashlib.sha256(request.canonical_form().encode("utf-8")).hexdigest()
    assert request.canonical_digest() == expected


def test_generic_canonicalize_serves_parameter_types() -> None:
    """adapters' own parameter models reuse the same function (SR-F1)."""
    criterion = Criterion.language("en")
    assert canonicalize(criterion) == '{"key":"language","value":"en"}'
    assert canonical_digest(criterion) == hashlib.sha256(canonicalize(criterion).encode("utf-8")).hexdigest()
