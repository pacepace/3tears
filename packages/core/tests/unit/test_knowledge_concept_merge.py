"""unit tests for the concept effective/layered merge (D4 / D5 / KNW-21).

covers: additive union of unlinked concepts; single-shadow at each scope
pair; shadow-of-shadow (three-scope chain); cross-scope chains; unbound
concepts (no datasource_table_id / sql_fragment); the ``always_inject``
invariant flag rides through shadow resolution (KNW-25 / D9); and the
concept-specific AMBIGUITY flag (Note 2): same-name unlinked concepts at
different scopes both survive and are both flagged ``ambiguous``, while a
same-name SHADOW collapses to one non-ambiguous winner.

the concept merge lives in :mod:`threetears.knowledge` so the hub
fingerprint builder and the SDK retrieval node share one implementation;
these tests own the pure-function coverage there. the generic
chain-resolution walk is shared with the entry merge (covered by
``test_knowledge_merge.py``); these tests focus on the concept-shaped
wrapper + ambiguity.
"""

from __future__ import annotations

from uuid import uuid7

from threetears.knowledge import (
    ConceptSnapshot,
    Scope,
    build_table_ref,
    merge_concept_views,
)


class TestBuildTableRef:
    """schema.table ref composition is the single source of truth.

    the SDK retrieval path and the hub knowledge tool both render concept
    bindings; the composed ref the agent reads must match the ``schema.table``
    its datasource tools accept, so the format lives in ONE helper.
    """

    def test_both_parts_compose(self) -> None:
        """schema + table compose to the dotted ref."""
        assert (
            build_table_ref("reporting_prod", "report_geofacts_joined_data")
            == "reporting_prod.report_geofacts_joined_data"
        )

    def test_missing_schema_yields_none(self) -> None:
        """a missing schema is not a usable ref (no malformed None.table)."""
        assert build_table_ref(None, "t") is None

    def test_missing_table_yields_none(self) -> None:
        """a missing table is not a usable ref (no malformed schema.None)."""
        assert build_table_ref("s", None) is None

    def test_both_missing_yields_none(self) -> None:
        """an unbound concept has no ref."""
        assert build_table_ref(None, None) is None


def _concept(
    scope: Scope,
    *,
    name: str = "active users",
    origin: object = None,
    definition: str = "def",
    datasource_table_id: object = None,
    sql_fragment: object = None,
    always_inject: bool = False,
) -> ConceptSnapshot:
    return ConceptSnapshot(
        id=uuid7(),
        scope=scope,
        origin_concept_id=origin,  # type: ignore[arg-type]
        name=name,
        definition=definition,
        datasource_table_id=datasource_table_id,  # type: ignore[arg-type]
        sql_fragment=sql_fragment,  # type: ignore[arg-type]
        always_inject=always_inject,
    )


class TestUnion:
    """additive union of unlinked concepts (distinct names)."""

    def test_three_unlinked_distinct_names_all_emitted(self) -> None:
        platform = _concept(Scope.PLATFORM, name="active users")
        customer = _concept(Scope.CUSTOMER, name="early vote")
        user = _concept(Scope.USER, name="churn")
        effective, layered = merge_concept_views([platform, customer, user])
        assert len(effective) == 3
        assert len(layered) == 3
        assert all(e.shadows_scope is None for e in effective)
        # distinct names -> none ambiguous.
        assert all(e.ambiguous is False for e in effective)
        names = {e.concept.name for e in effective}
        assert names == {"active users", "early vote", "churn"}

    def test_output_order_is_platform_then_customer_then_user(self) -> None:
        user = _concept(Scope.USER, name="u")
        platform = _concept(Scope.PLATFORM, name="p")
        customer = _concept(Scope.CUSTOMER, name="c")
        effective, _layered = merge_concept_views([user, platform, customer])
        assert [e.concept.scope for e in effective] == [
            Scope.PLATFORM,
            Scope.CUSTOMER,
            Scope.USER,
        ]


class TestSingleShadow:
    """one concept shadowing one ancestor, at each scope pair (D4)."""

    def test_customer_shadows_platform(self) -> None:
        platform = _concept(
            Scope.PLATFORM,
            definition="platform def",
            datasource_table_id=uuid7(),
            sql_fragment="is_test = false",
        )
        customer = _concept(
            Scope.CUSTOMER,
            origin=platform.id,
            definition="customer def",
        )
        effective, layered = merge_concept_views([platform, customer])
        assert len(effective) == 1
        winner = effective[0]
        assert winner.concept.definition == "customer def"
        assert winner.shadows_scope is Scope.PLATFORM
        # a declared shadow is NEVER ambiguous (one winner).
        assert winner.ambiguous is False
        assert len(layered[0].shadowed) == 1
        assert layered[0].shadowed[0].definition == "platform def"

    def test_user_shadows_customer(self) -> None:
        customer = _concept(Scope.CUSTOMER, definition="customer def")
        user = _concept(
            Scope.USER,
            origin=customer.id,
            definition="user def",
        )
        effective, layered = merge_concept_views([customer, user])
        assert len(effective) == 1
        assert effective[0].concept.definition == "user def"
        assert effective[0].shadows_scope is Scope.CUSTOMER
        assert effective[0].ambiguous is False
        assert layered[0].shadowed[0].definition == "customer def"


class TestShadowOfShadow:
    """three-scope chain: user shadows customer shadows platform."""

    def test_nearest_scope_wins_whole_chain(self) -> None:
        platform = _concept(Scope.PLATFORM, definition="platform def")
        customer = _concept(
            Scope.CUSTOMER,
            origin=platform.id,
            definition="customer def",
        )
        user = _concept(
            Scope.USER,
            origin=customer.id,
            definition="user def",
        )
        effective, layered = merge_concept_views([platform, customer, user])
        assert len(effective) == 1
        winner = effective[0]
        assert winner.concept.definition == "user def"
        assert winner.shadows_scope is Scope.CUSTOMER
        assert winner.ambiguous is False
        shadowed_defs = [s.definition for s in layered[0].shadowed]
        assert shadowed_defs == ["customer def", "platform def"]

    def test_whole_entry_replace_no_definition_splicing(self) -> None:
        platform = _concept(Scope.PLATFORM, definition="PLATFORM")
        customer = _concept(
            Scope.CUSTOMER,
            origin=platform.id,
            definition="CUSTOMER",
        )
        user = _concept(Scope.USER, origin=customer.id, definition="USER")
        effective, _layered = merge_concept_views([platform, customer, user])
        assert effective[0].concept.definition == "USER"
        assert "PLATFORM" not in effective[0].concept.definition
        assert "CUSTOMER" not in effective[0].concept.definition


class TestUnboundConcept:
    """a concept with no binding (definition + caveats only) passes through."""

    def test_unbound_concept_emitted(self) -> None:
        unbound = _concept(
            Scope.CUSTOMER,
            name="churn",
            definition="left in 90 days",
            datasource_table_id=None,
            sql_fragment=None,
        )
        effective, _layered = merge_concept_views([unbound])
        assert len(effective) == 1
        assert effective[0].concept.datasource_table_id is None
        assert effective[0].concept.sql_fragment is None
        assert effective[0].ambiguous is False


class TestAlwaysInjectRidesThroughShadow:
    """KNW-25 / D9: the invariant flag is not special-cased by merge."""

    def test_user_non_invariant_shadows_platform_invariant(self) -> None:
        platform = _concept(
            Scope.PLATFORM,
            definition="inv",
            always_inject=True,
        )
        user = _concept(
            Scope.USER,
            origin=platform.id,
            definition="override",
            always_inject=False,
        )
        effective, _layered = merge_concept_views([platform, user])
        assert len(effective) == 1
        assert effective[0].concept.definition == "override"
        assert effective[0].always_inject is False

    def test_user_invariant_shadows_platform_non_invariant(self) -> None:
        platform = _concept(
            Scope.PLATFORM,
            definition="situational",
            always_inject=False,
        )
        user = _concept(
            Scope.USER,
            origin=platform.id,
            definition="hard rule",
            always_inject=True,
        )
        effective, _layered = merge_concept_views([platform, user])
        assert len(effective) == 1
        assert effective[0].always_inject is True


class TestAmbiguity:
    """Note 2 / D5: same-name unlinked concepts flag ambiguous; disclose."""

    def test_same_name_different_scopes_no_link_both_ambiguous(self) -> None:
        # two governed definitions of "active users" at different scopes
        # with NO declared shadow -> both survive AND both flag ambiguous
        # so the agent/footer disclose the competing definitions (D5).
        platform = _concept(
            Scope.PLATFORM,
            name="active users",
            definition="trailing 28d",
        )
        customer = _concept(
            Scope.CUSTOMER,
            name="Active Users",
            definition="trailing 7d",
        )
        effective, _layered = merge_concept_views([platform, customer])
        assert len(effective) == 2
        # case-insensitive collision -> BOTH flagged.
        assert all(e.ambiguous is True for e in effective)

    def test_same_name_with_shadow_link_not_ambiguous(self) -> None:
        # the SAME collision but WITH an origin link is a declared shadow:
        # collapses to one winner, never ambiguous.
        platform = _concept(
            Scope.PLATFORM,
            name="active users",
            definition="trailing 28d",
        )
        customer = _concept(
            Scope.CUSTOMER,
            name="active users",
            origin=platform.id,
            definition="trailing 7d",
        )
        effective, _layered = merge_concept_views([platform, customer])
        assert len(effective) == 1
        assert effective[0].ambiguous is False
        assert effective[0].shadows_scope is Scope.PLATFORM

    def test_distinct_name_alongside_ambiguous_pair_not_flagged(self) -> None:
        # a distinct-named concept sharing the result set with an
        # ambiguous pair is itself NOT flagged.
        a = _concept(Scope.PLATFORM, name="active users", definition="a")
        b = _concept(Scope.CUSTOMER, name="active users", definition="b")
        c = _concept(Scope.CUSTOMER, name="early vote", definition="c")
        effective, _layered = merge_concept_views([a, b, c])
        by_name: dict[str, bool] = {}
        for e in effective:
            by_name.setdefault(e.concept.name, False)
            by_name[e.concept.name] = by_name[e.concept.name] or e.ambiguous
        # both "active users" winners ambiguous; "early vote" is not.
        ambiguous_flags = {e.concept.definition: e.ambiguous for e in effective}
        assert ambiguous_flags["a"] is True
        assert ambiguous_flags["b"] is True
        assert ambiguous_flags["c"] is False

    def test_shadow_winner_also_ambiguous_vs_independent_winner(self) -> None:
        # MIXED-AMBIGUITY edge: a shadow chain resolves to ONE winner
        # named "X" (user shadows platform), but a SEPARATE unlinked
        # customer concept is ALSO named "X". the shadow collapse does not
        # immunise the winner: name_counts counts every surviving winner,
        # so the shadow winner AND the independent concept are BOTH flagged
        # ambiguous (the winner competes by name with a separate unlinked
        # definition the agent/footer must disclose).
        platform = _concept(Scope.PLATFORM, name="X", definition="platform X")
        user = _concept(
            Scope.USER,
            name="X",
            origin=platform.id,
            definition="user X",
        )
        independent = _concept(
            Scope.CUSTOMER,
            name="x",
            definition="independent X",
        )
        effective, _layered = merge_concept_views(
            [platform, user, independent],
        )
        # shadow chain collapses to the user winner; the independent
        # concept survives on its own -> two surviving winners named "x".
        assert len(effective) == 2
        by_def = {e.concept.definition: e for e in effective}
        assert set(by_def) == {"user X", "independent X"}
        # the shadow WINNER is ambiguous (shares name with the independent
        # winner) yet ALSO carries its shadow disclosure.
        shadow_winner = by_def["user X"]
        assert shadow_winner.ambiguous is True
        assert shadow_winner.shadows_scope is Scope.PLATFORM
        # the independent winner is likewise flagged.
        assert by_def["independent X"].ambiguous is True

    def test_blank_name_not_treated_as_collision(self) -> None:
        # two blank-named concepts are not a name collision (defensive:
        # the hub requires a name, but the pure merge must not flag empties).
        a = _concept(Scope.PLATFORM, name="", definition="a")
        b = _concept(Scope.CUSTOMER, name="", definition="b")
        effective, _layered = merge_concept_views([a, b])
        assert all(e.ambiguous is False for e in effective)


class TestMergeDeterminism:
    """two merges of the same input produce the same order."""

    def test_stable_ordering(self) -> None:
        concepts = [
            _concept(Scope.USER, name="u1"),
            _concept(Scope.PLATFORM, name="p1"),
            _concept(Scope.CUSTOMER, name="c1"),
            _concept(Scope.PLATFORM, name="p2"),
        ]
        first, _ = merge_concept_views(concepts)
        second, _ = merge_concept_views(concepts)
        assert [e.concept.id for e in first] == [e.concept.id for e in second]
