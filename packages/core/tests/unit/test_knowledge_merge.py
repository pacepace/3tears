"""unit tests for the playbook-entry effective/layered merge (D4 / KNW-11).

covers: additive union of unlinked entries; single-shadow at each scope
pair; shadow-of-shadow (three-scope chain); cross-scope chains; unlinked
entries pass through; two contradictory unlinked entries are BOTH
emitted (no origin link -> no suppression); the ``always_inject``
invariant flag rides through shadow resolution (KNW-17 / D9); and the
write-time cycle rejection. also covers the pure scope-derivation
contract (D1).

the merge + scope vocabulary live in :mod:`threetears.knowledge` so the
hub fingerprint builder and the SDK retrieval node share one
implementation; these tests own the pure-function coverage there.
"""

from __future__ import annotations

from uuid import uuid7

import pytest

from threetears.knowledge import (
    MAX_SHADOW_CHAIN_DEPTH,
    EntrySnapshot,
    OriginCycleError,
    Scope,
    assert_no_origin_cycle,
    derive_scope,
    merge_entry_views,
)


def _entry(
    scope: Scope,
    *,
    origin: object = None,
    title: str = "t",
    body: str = "b",
    always_inject: bool = False,
) -> EntrySnapshot:
    return EntrySnapshot(
        id=uuid7(),
        scope=scope,
        origin_entry_id=origin,  # type: ignore[arg-type]
        title=title,
        body=body,
        always_inject=always_inject,
    )


class TestDeriveScope:
    """pure scope derivation from (customer_id, user_id) nullability (D1)."""

    def test_both_null_is_platform(self) -> None:
        assert derive_scope(customer_id=None, user_id=None) is Scope.PLATFORM

    def test_customer_set_user_null_is_customer(self) -> None:
        assert derive_scope(customer_id=uuid7(), user_id=None) is Scope.CUSTOMER

    def test_both_set_is_user(self) -> None:
        assert derive_scope(customer_id=uuid7(), user_id=uuid7()) is Scope.USER

    def test_user_without_customer_rejected(self) -> None:
        with pytest.raises(ValueError, match="user always belongs to a customer"):
            derive_scope(customer_id=None, user_id=uuid7())


class TestUnion:
    """additive union of unlinked entries (KNW-11)."""

    def test_three_unlinked_entries_all_emitted(self) -> None:
        platform = _entry(Scope.PLATFORM, title="p")
        customer = _entry(Scope.CUSTOMER, title="c")
        user = _entry(Scope.USER, title="u")
        effective, layered = merge_entry_views([platform, customer, user])
        assert len(effective) == 3
        assert len(layered) == 3
        # no shadows anywhere; every entry passes through.
        assert all(e.shadows_scope is None for e in effective)
        emitted_titles = {e.entry.title for e in effective}
        assert emitted_titles == {"p", "c", "u"}

    def test_output_order_is_platform_then_customer_then_user(self) -> None:
        user = _entry(Scope.USER, title="u")
        platform = _entry(Scope.PLATFORM, title="p")
        customer = _entry(Scope.CUSTOMER, title="c")
        effective, _layered = merge_entry_views([user, platform, customer])
        assert [e.entry.scope for e in effective] == [
            Scope.PLATFORM, Scope.CUSTOMER, Scope.USER,
        ]

    def test_two_contradictory_unlinked_entries_both_pass_through(self) -> None:
        # two entries with NO origin link contradicting each other are
        # both emitted -- the author did not declare a shadow, so neither
        # is suppressed.
        a = _entry(Scope.USER, title="exclude test counties", body="A")
        b = _entry(Scope.USER, title="include test counties", body="B")
        effective, _layered = merge_entry_views([a, b])
        assert len(effective) == 2
        assert {e.entry.body for e in effective} == {"A", "B"}


class TestSingleShadow:
    """one entry shadowing one ancestor, at each scope pair (D4)."""

    def test_customer_shadows_platform(self) -> None:
        platform = _entry(Scope.PLATFORM, title="p", body="platform body")
        customer = _entry(
            Scope.CUSTOMER, origin=platform.id, title="c", body="customer body",
        )
        effective, layered = merge_entry_views([platform, customer])
        assert len(effective) == 1
        winner = effective[0]
        assert winner.entry.body == "customer body"
        assert winner.shadows_scope is Scope.PLATFORM
        # layered retains the shadowed platform ancestor.
        assert len(layered[0].shadowed) == 1
        assert layered[0].shadowed[0].body == "platform body"

    def test_user_shadows_customer(self) -> None:
        customer = _entry(Scope.CUSTOMER, body="customer body")
        user = _entry(Scope.USER, origin=customer.id, body="user body")
        effective, layered = merge_entry_views([customer, user])
        assert len(effective) == 1
        assert effective[0].entry.body == "user body"
        assert effective[0].shadows_scope is Scope.CUSTOMER
        assert layered[0].shadowed[0].body == "customer body"

    def test_user_shadows_platform(self) -> None:
        platform = _entry(Scope.PLATFORM, body="platform body")
        user = _entry(Scope.USER, origin=platform.id, body="user body")
        effective, _layered = merge_entry_views([platform, user])
        assert len(effective) == 1
        assert effective[0].entry.body == "user body"
        assert effective[0].shadows_scope is Scope.PLATFORM


class TestShadowOfShadow:
    """three-scope chain: user shadows customer shadows platform."""

    def test_nearest_scope_wins_whole_chain(self) -> None:
        platform = _entry(Scope.PLATFORM, body="platform body")
        customer = _entry(
            Scope.CUSTOMER, origin=platform.id, body="customer body",
        )
        user = _entry(Scope.USER, origin=customer.id, body="user body")
        effective, layered = merge_entry_views([platform, customer, user])
        assert len(effective) == 1
        winner = effective[0]
        assert winner.entry.body == "user body"
        # nearest shadowed ancestor is the customer entry.
        assert winner.shadows_scope is Scope.CUSTOMER
        # layered retains BOTH ancestors, nearest-first.
        shadowed_bodies = [s.body for s in layered[0].shadowed]
        assert shadowed_bodies == ["customer body", "platform body"]

    def test_whole_entry_replace_no_body_splicing(self) -> None:
        # the winner body is the user body VERBATIM -- the platform /
        # customer bodies are not concatenated or spliced in (D4).
        platform = _entry(Scope.PLATFORM, body="PLATFORM")
        customer = _entry(Scope.CUSTOMER, origin=platform.id, body="CUSTOMER")
        user = _entry(Scope.USER, origin=customer.id, body="USER")
        effective, _layered = merge_entry_views([platform, customer, user])
        assert effective[0].entry.body == "USER"
        assert "PLATFORM" not in effective[0].entry.body
        assert "CUSTOMER" not in effective[0].entry.body


class TestCrossScopeChainsMixed:
    """a shadowed chain plus unrelated unlinked entries together."""

    def test_chain_collapses_unlinked_pass_through(self) -> None:
        platform = _entry(Scope.PLATFORM, body="platform body")
        user = _entry(Scope.USER, origin=platform.id, body="user override")
        unrelated_platform = _entry(Scope.PLATFORM, body="unrelated")
        unrelated_user = _entry(Scope.USER, body="standalone user")
        effective, _layered = merge_entry_views(
            [platform, user, unrelated_platform, unrelated_user],
        )
        bodies = sorted(e.entry.body for e in effective)
        # the chain collapses to the user override; the two unlinked
        # entries pass through. three effective entries total.
        assert bodies == ["standalone user", "unrelated", "user override"]


class TestAlwaysInjectRidesThroughShadow:
    """KNW-17 / D9: the invariant flag is not special-cased by merge."""

    def test_user_non_invariant_shadows_platform_invariant(self) -> None:
        # platform entry is an INVARIANT; the user shadows it with a
        # NON-invariant override. the winner's flag governs -> the
        # effective entry is NOT an invariant.
        platform = _entry(Scope.PLATFORM, body="inv", always_inject=True)
        user = _entry(
            Scope.USER, origin=platform.id, body="override",
            always_inject=False,
        )
        effective, _layered = merge_entry_views([platform, user])
        assert len(effective) == 1
        assert effective[0].entry.body == "override"
        assert effective[0].always_inject is False

    def test_user_invariant_shadows_platform_non_invariant(self) -> None:
        # the reverse: platform non-invariant, user invariant override.
        # the winner (user) governs -> effective entry IS an invariant.
        platform = _entry(Scope.PLATFORM, body="situational", always_inject=False)
        user = _entry(
            Scope.USER, origin=platform.id, body="hard rule",
            always_inject=True,
        )
        effective, _layered = merge_entry_views([platform, user])
        assert len(effective) == 1
        assert effective[0].always_inject is True

    def test_unlinked_invariant_preserves_flag(self) -> None:
        inv = _entry(Scope.PLATFORM, always_inject=True)
        situational = _entry(Scope.PLATFORM, always_inject=False)
        effective, _layered = merge_entry_views([inv, situational])
        flags = {e.entry.id: e.always_inject for e in effective}
        assert flags[inv.id] is True
        assert flags[situational.id] is False


class TestCycleRejectionAtWrite:
    """origin_entry_id cycles rejected at write time."""

    def test_self_link_rejected(self) -> None:
        entry_id = uuid7()
        with pytest.raises(OriginCycleError):
            assert_no_origin_cycle(
                entry_id=entry_id,
                origin_entry_id=entry_id,
                origin_of={entry_id: None},
            )

    def test_two_node_cycle_rejected(self) -> None:
        a = uuid7()
        b = uuid7()
        # b -> a already persisted; now writing a -> b closes a cycle.
        origin_of = {a: None, b: a}
        with pytest.raises(OriginCycleError):
            assert_no_origin_cycle(
                entry_id=a, origin_entry_id=b, origin_of=origin_of,
            )

    def test_acyclic_chain_accepted(self) -> None:
        platform = uuid7()
        customer = uuid7()
        new_user = uuid7()
        origin_of = {platform: None, customer: platform}
        # new user entry shadowing customer -> acyclic, accepted.
        assert_no_origin_cycle(
            entry_id=new_user, origin_entry_id=customer, origin_of=origin_of,
        )

    def test_none_origin_is_trivially_acyclic(self) -> None:
        assert_no_origin_cycle(
            entry_id=uuid7(), origin_entry_id=None, origin_of={},
        )

    def test_overlong_chain_rejected(self) -> None:
        # build a chain longer than the depth bound.
        ids = [uuid7() for _ in range(MAX_SHADOW_CHAIN_DEPTH + 3)]
        origin_of: dict = {ids[0]: None}
        for i in range(1, len(ids)):
            origin_of[ids[i]] = ids[i - 1]
        new_id = uuid7()
        with pytest.raises(OriginCycleError):
            assert_no_origin_cycle(
                entry_id=new_id,
                origin_entry_id=ids[-1],
                origin_of=origin_of,
            )


class TestMergeDeterminism:
    """two merges of the same input produce the same order."""

    def test_stable_ordering(self) -> None:
        entries = [
            _entry(Scope.USER, title="u1"),
            _entry(Scope.PLATFORM, title="p1"),
            _entry(Scope.CUSTOMER, title="c1"),
            _entry(Scope.PLATFORM, title="p2"),
        ]
        first, _ = merge_entry_views(entries)
        second, _ = merge_entry_views(entries)
        assert [e.entry.id for e in first] == [e.entry.id for e in second]
