"""Canonical serialization -- the D26 keying rule, pinned (SR-F1, SR-F8).

Explicitly-set semantic parameters only; absent and defaulted canonically
identical; stable form; criteria order canonically irrelevant; operational
fields (``record``, ``budget_scope_tags``) never participate. Two consumers
must agree on this form (the replay key and eval run identity), so these are
contract pins, not implementation tests.
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
    request = SearchRequest(query="capybara", fidelity="content")
    assert json.loads(request.canonical_form()) == {"query": "capybara", "fidelity": "content"}


def test_operational_fields_never_participate() -> None:
    """the Gate A re-cut: ``record`` and ``budget_scope_tags`` are facts about
    one invocation, not about which search this is. A recording is made with
    record=True by definition (SR-F6) and replayed without it (SR-F7), and
    scope tags carry per-run identity (SR-D2) -- so neither may key the
    digest, or replay and eval attributability both break structurally."""
    bare = SearchRequest(query="capybara")
    recorded = SearchRequest(query="capybara", record=True)
    assert recorded.canonical_form() == '{"query":"capybara"}'
    assert recorded.canonical_digest() == bare.canonical_digest()

    run_one = SearchRequest(query="capybara", budget_scope_tags=("run:1",))
    run_two = SearchRequest(query="capybara", budget_scope_tags=("run:2", "persona:capy"))
    assert run_one.canonical_digest() == bare.canonical_digest()
    assert run_two.canonical_digest() == bare.canonical_digest()


def test_criteria_order_is_canonically_irrelevant() -> None:
    """two requests differing only in criteria order are one search."""
    a = Criterion.language("en")
    b = Criterion.domains_include(["example.org", "example.net"])
    forward = SearchRequest(query="q", criteria=(a, b))
    backward = SearchRequest(query="q", criteria=(b, a))
    assert forward.canonical_form() == backward.canonical_form()
    assert forward.canonical_digest() == backward.canonical_digest()


def test_different_explicit_values_diverge() -> None:
    """the form distinguishes what actually differs -- semantically."""
    assert (
        SearchRequest(query="q", fidelity="content").canonical_digest() != SearchRequest(query="q").canonical_digest()
    )
    assert SearchRequest(query="q").canonical_digest() != SearchRequest(query="r").canonical_digest()


def test_form_is_stable_json() -> None:
    """sorted keys, compact separators -- byte-stable for equal inputs, and
    the operational scope tags leave no trace in the form."""
    form = SearchRequest(
        query="q",
        criteria=(Criterion.language("en"),),
        budget_scope_tags=("run:1",),
    ).canonical_form()
    parsed = json.loads(form)
    assert form == json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert "budget_scope_tags" not in parsed


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
