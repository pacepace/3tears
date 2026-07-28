"""Mapping identity-provider claims onto local group memberships."""

from __future__ import annotations

import pytest

from threetears.iam.claim_mapping import ClaimBinding, ClaimMappingRule, resolve_claim_grants


def _rule(*bindings: ClaimBinding) -> ClaimMappingRule:
    return ClaimMappingRule(bindings=bindings)


def test_a_matching_scalar_claim_grants_its_references() -> None:
    rule = _rule(ClaimBinding(claim="hd", value="acme.com", group_refs=("acme-staff",)))
    assert resolve_claim_grants(rule, {"hd": "acme.com"}) == frozenset({"acme-staff"})


def test_a_matching_list_claim_grants_its_references() -> None:
    # An Entra `groups` claim asserts a list; a scalar and a list must take one code path.
    rule = _rule(ClaimBinding(claim="groups", value="admins", group_refs=("platform-admins",)))
    assert resolve_claim_grants(rule, {"groups": ["users", "admins"]}) == frozenset({"platform-admins"})


def test_no_rule_grants_nothing() -> None:
    # Default-deny by construction: no mapping configured is not an error, it is no grant.
    assert resolve_claim_grants(None, {"groups": ["admins"]}) == frozenset()


def test_a_rule_with_no_bindings_grants_nothing() -> None:
    assert resolve_claim_grants(_rule(), {"groups": ["admins"]}) == frozenset()


def test_an_absent_claim_grants_nothing() -> None:
    rule = _rule(ClaimBinding(claim="groups", value="admins", group_refs=("platform-admins",)))
    assert resolve_claim_grants(rule, {}) == frozenset()


def test_a_null_claim_grants_nothing() -> None:
    rule = _rule(ClaimBinding(claim="groups", value="admins", group_refs=("platform-admins",)))
    assert resolve_claim_grants(rule, {"groups": None}) == frozenset()


def test_a_non_matching_value_grants_nothing() -> None:
    rule = _rule(ClaimBinding(claim="hd", value="acme.com", group_refs=("acme-staff",)))
    assert resolve_claim_grants(rule, {"hd": "evil.com"}) == frozenset()


def test_matching_is_exact_not_a_prefix() -> None:
    # A prefix match would grant "acme.com.attacker.net" everything "acme.com" gets.
    rule = _rule(ClaimBinding(claim="hd", value="acme.com", group_refs=("acme-staff",)))
    assert resolve_claim_grants(rule, {"hd": "acme.com.attacker.net"}) == frozenset()
    assert resolve_claim_grants(rule, {"hd": "notacme.com"}) == frozenset()


def test_matching_is_case_sensitive() -> None:
    rule = _rule(ClaimBinding(claim="hd", value="acme.com", group_refs=("acme-staff",)))
    assert resolve_claim_grants(rule, {"hd": "ACME.COM"}) == frozenset()


def test_a_string_claim_is_not_iterated_character_by_character() -> None:
    # A string IS a sequence. Iterating one would match a single CHARACTER against a binding
    # value, so "a" would match the claim "acme.com".
    rule = _rule(ClaimBinding(claim="hd", value="a", group_refs=("wrong",)))
    assert resolve_claim_grants(rule, {"hd": "acme.com"}) == frozenset()


def test_several_bindings_union_their_grants() -> None:
    rule = _rule(
        ClaimBinding(claim="groups", value="admins", group_refs=("platform-admins",)),
        ClaimBinding(claim="hd", value="acme.com", group_refs=("acme-staff", "everyone")),
    )
    assert resolve_claim_grants(rule, {"groups": ["admins"], "hd": "acme.com"}) == frozenset(
        {"platform-admins", "acme-staff", "everyone"}
    )


def test_overlapping_bindings_are_not_a_conflict() -> None:
    rule = _rule(
        ClaimBinding(claim="groups", value="a", group_refs=("shared",)),
        ClaimBinding(claim="groups", value="b", group_refs=("shared",)),
    )
    assert resolve_claim_grants(rule, {"groups": ["a", "b"]}) == frozenset({"shared"})


def test_only_the_matching_binding_grants() -> None:
    rule = _rule(
        ClaimBinding(claim="groups", value="admins", group_refs=("platform-admins",)),
        ClaimBinding(claim="groups", value="auditors", group_refs=("auditors",)),
    )
    assert resolve_claim_grants(rule, {"groups": ["auditors"]}) == frozenset({"auditors"})


def test_the_result_is_deterministic_and_order_independent() -> None:
    first = ClaimBinding(claim="groups", value="a", group_refs=("x",))
    second = ClaimBinding(claim="hd", value="acme.com", group_refs=("y",))
    claims = {"groups": ["a"], "hd": "acme.com"}
    assert resolve_claim_grants(_rule(first, second), claims) == resolve_claim_grants(_rule(second, first), claims)


@pytest.mark.parametrize("asserted", [["admins"], ("admins",), {"admins"}, frozenset({"admins"})])
def test_every_collection_shape_is_matched(asserted: object) -> None:
    rule = _rule(ClaimBinding(claim="groups", value="admins", group_refs=("platform-admins",)))
    assert resolve_claim_grants(rule, {"groups": asserted}) == frozenset({"platform-admins"})


def test_non_string_claim_values_are_stringified_for_matching() -> None:
    rule = _rule(ClaimBinding(claim="level", value="3", group_refs=("tier-3",)))
    assert resolve_claim_grants(rule, {"level": 3}) == frozenset({"tier-3"})
