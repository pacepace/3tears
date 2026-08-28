"""unit tests for the ``tools.*`` namespace-name grammar.

the grammar is a BUILDER and a PARSER that must agree exactly, because
the same string is written into ``namespaces.name`` by one
process and rebuilt for comparison by several others. a disagreement
between the two spellings does not raise -- it makes a row stop being
addressable, which is a silence.

the corpus below is the live one. every entry is either a name read off
a running stack or an mcp name and version read off production, so a
change that round-trips here round-trips against real data rather than
against invented data shaped to pass.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from threetears.core.namespaces import (
    NAMESPACE_NAME_SEPARATOR,
    PLURAL_PREFIX_HITL,
    PLURAL_PREFIX_TOOL,
    ToolNamespaceName,
    build_hitl_namespace_name,
    build_namespace_name,
    build_tool_namespace_name,
    build_tool_namespace_name_or_none,
    build_tool_provider_node_name,
    namespace_contains,
    parse_tool_namespace_name,
    sanitize_segment,
)

# The census corpus: ``(mcp_name, version)`` pairs observed in the wild.
#
# Two version SHAPES are present and both are load-bearing. The
# ``aibots`` / ``threetears`` / ``datasource`` families carry a
# two-part ``1.0``; the ``addrnorm`` family, which arrives through the
# api-import capability-source path rather than through an
# ``agent.yaml``, carries a three-part ``1.0.0``. A grammar that
# assumed one shape would recover the wrong version for the other, and
# a corpus drawn from a local stack alone contains only the first.
CENSUS_CORPUS: tuple[tuple[str, str], ...] = (
    # read off a running stack
    ("aibots.knowledge_drafts", "1.0"),
    ("datasource.central-reporting.read", "1.0"),
    ("datasource.central-reporting.schema", "1.0"),
    ("datasource.relations.lookup", "1.0"),
    ("knowledge.concept_lookup", "1.0"),
    ("knowledge.lookup", "1.0"),
    ("threetears.calculator", "1.0"),
    ("threetears.current_date", "1.0"),
    ("threetears.dictionary", "1.0"),
    ("threetears.image_prep", "1.0"),
    ("threetears.parse_document", "1.0"),
    ("threetears.timezone_converter", "1.0"),
    ("threetears.unit_converter", "1.0"),
    ("threetears.web_fetch", "1.0"),
    ("threetears.web_search", "1.0"),
    # read off production, three-part version
    ("addrnorm.geocode_reverse_batch", "1.0.0"),
    ("addrnorm.geocode", "1.0.0"),
    # read off production, other families
    ("survey_admin.close_collector", "1.0"),
    ("pentest.sqlmap", "1.0"),
    ("aibots.admin.list_agents", "1.0"),
    ("aibots.set_active_engagement", "1.0"),
    ("ripple.audience_define", "1.0"),
    # a single-segment mcp name: no dot at all, so the name it builds
    # has exactly the two components the parser needs and not one more
    ("standalone_tool", "1.0"),
)


class TestBuildToolNamespaceName:
    """the shape the builder emits."""

    def test_the_mcp_name_keeps_its_dots(self) -> None:
        # THE POINT OF THE GRAMMAR. a flattened name is not under
        # ``tools.pentest`` by any segment-aware rule, so subtree
        # containment has nothing to bite on.
        assert build_tool_namespace_name("pentest.sqlmap", "1.0") == "tools.pentest.sqlmap.1-0"

    def test_the_version_is_still_sanitized_and_still_last(self) -> None:
        # what keeps the parse unambiguous right-to-left: the version
        # is the one segment guaranteed to carry no separator.
        assert build_tool_namespace_name("a.b.c", "1.0.0") == "tools.a.b.c.1-0-0"

    def test_a_single_segment_mcp_name_builds_two_components(self) -> None:
        assert build_tool_namespace_name("standalone_tool", "1.0") == "tools.standalone_tool.1-0"

    def test_a_hyphen_in_the_mcp_name_survives_as_a_hyphen(self) -> None:
        # the old flattening made a real hyphen and a rewritten dot
        # indistinguishable, so ``a.b`` and ``a-b`` collapsed onto one
        # row. they no longer can.
        assert build_tool_namespace_name("datasource.central-reporting.read", "1.0") == (
            "tools.datasource.central-reporting.read.1-0"
        )

    def test_a_dot_and_a_hyphen_no_longer_collide(self) -> None:
        assert build_tool_namespace_name("a.b", "1.0") != build_tool_namespace_name("a-b", "1.0")

    def test_an_already_rooted_mcp_name_is_refused(self) -> None:
        # never ``tools.tools.*``: the prefix is added by exactly one
        # layer, and a caller passing an already-built name is passing
        # the wrong thing rather than asking for a nested one.
        with pytest.raises(ValueError, match="already rooted"):
            build_tool_namespace_name("tools.pentest.sqlmap", "1.0")

    def test_the_bare_prefix_as_an_mcp_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="already rooted"):
            build_tool_namespace_name(PLURAL_PREFIX_TOOL, "1.0")

    def test_an_empty_mcp_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="mcp_name"):
            build_tool_namespace_name("", "1.0")

    def test_an_empty_version_is_refused(self) -> None:
        with pytest.raises(ValueError, match="version"):
            build_tool_namespace_name("pentest.sqlmap", "")

    def test_an_empty_component_in_the_mcp_name_is_refused(self) -> None:
        # a doubled separator renders a name whose version is no longer
        # recoverable as "the thing after the last dot" for a reader
        # counting components. refuse rather than mint it.
        with pytest.raises(ValueError, match="empty component"):
            build_tool_namespace_name("pentest..sqlmap", "1.0")

    def test_a_leading_separator_in_the_mcp_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty component"):
            build_tool_namespace_name(".pentest", "1.0")

    def test_a_trailing_separator_in_the_mcp_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty component"):
            build_tool_namespace_name("pentest.", "1.0")

    def test_every_built_name_sits_under_the_tools_prefix(self) -> None:
        for mcp_name, version in CENSUS_CORPUS:
            built = build_tool_namespace_name(mcp_name, version)
            assert namespace_contains(PLURAL_PREFIX_TOOL, built) is True

    def test_no_built_name_is_doubly_rooted(self) -> None:
        doubled = f"{PLURAL_PREFIX_TOOL}{NAMESPACE_NAME_SEPARATOR}{PLURAL_PREFIX_TOOL}{NAMESPACE_NAME_SEPARATOR}"
        for mcp_name, version in CENSUS_CORPUS:
            assert not build_tool_namespace_name(mcp_name, version).startswith(doubled)


class TestParseToolNamespaceName:
    """recovering the components, right to left."""

    def test_it_recovers_the_mcp_name_and_the_version_segment(self) -> None:
        parsed = parse_tool_namespace_name("tools.pentest.sqlmap.1-0")
        assert parsed == ToolNamespaceName(mcp_name="pentest.sqlmap", version_segment="1-0")

    def test_a_three_segment_mcp_name_still_yields_the_last_segment_as_the_version(self) -> None:
        # the ambiguity the grammar spends its invariant on: with dots
        # inside the name the separator is overloaded, and the parse is
        # sound only because the version is ALWAYS last.
        parsed = parse_tool_namespace_name("tools.a.b.c.1-0-0")
        assert parsed.mcp_name == "a.b.c"
        assert parsed.version_segment == "1-0-0"

    def test_a_three_part_version_is_recovered_whole(self) -> None:
        parsed = parse_tool_namespace_name("tools.addrnorm.geocode.1-0-0")
        assert parsed.mcp_name == "addrnorm.geocode"
        assert parsed.version_segment == "1-0-0"

    def test_a_single_segment_mcp_name_parses(self) -> None:
        parsed = parse_tool_namespace_name("tools.standalone_tool.1-0")
        assert parsed.mcp_name == "standalone_tool"
        assert parsed.version_segment == "1-0"

    def test_a_name_outside_the_tools_prefix_is_refused(self) -> None:
        with pytest.raises(ValueError, match="tools."):
            parse_tool_namespace_name("agents.3f2504e0-4f89-41d3-9a0c-0305e82c3301")

    def test_a_sibling_prefix_is_refused(self) -> None:
        # segment-aware, so the imposter cannot be parsed as a tool name.
        with pytest.raises(ValueError, match="tools."):
            parse_tool_namespace_name("toolsimposter.thing.1-0")

    def test_the_bare_prefix_is_refused(self) -> None:
        with pytest.raises(ValueError, match="tools."):
            parse_tool_namespace_name(PLURAL_PREFIX_TOOL)

    def test_a_name_with_no_version_component_is_refused(self) -> None:
        # ``tools.pentest`` is a PROVIDER node, not a tool namespace
        # name, and reading it as one would hand back a version of
        # ``pentest`` and an empty mcp name.
        with pytest.raises(ValueError, match="two components"):
            parse_tool_namespace_name("tools.pentest")

    def test_an_empty_component_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty component"):
            parse_tool_namespace_name("tools.pentest..1-0")

    def test_an_empty_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="tools."):
            parse_tool_namespace_name("")

    def test_the_result_is_frozen(self) -> None:
        parsed = parse_tool_namespace_name("tools.pentest.sqlmap.1-0")
        with pytest.raises(Exception):  # noqa: B017 -- FrozenInstanceError is not exported by dataclasses as a builtin here
            parsed.mcp_name = "other"  # type: ignore[misc]


class TestTheBuilderAndTheParserAgree:
    """the round trip, over the census corpus.

    a builder and a parser that disagree do not raise; they make a row
    unaddressable. so the agreement is asserted over real names rather
    than described in a docstring.
    """

    @pytest.mark.parametrize(("mcp_name", "version"), CENSUS_CORPUS)
    def test_every_census_entry_round_trips(self, mcp_name: str, version: str) -> None:
        built = build_tool_namespace_name(mcp_name, version)
        parsed = parse_tool_namespace_name(built)
        assert parsed.mcp_name == mcp_name
        assert parsed.version_segment == sanitize_segment(version)

    @pytest.mark.parametrize(("mcp_name", "version"), CENSUS_CORPUS)
    def test_rebuilding_from_the_parse_reproduces_the_name(self, mcp_name: str, version: str) -> None:
        built = build_tool_namespace_name(mcp_name, version)
        parsed = parse_tool_namespace_name(built)
        assert build_tool_namespace_name(parsed.mcp_name, parsed.version_segment) == built

    def test_the_corpus_names_are_distinct(self) -> None:
        # the collision the old flattening admitted: two distinct mcp
        # names rendering one row. if this ever fails, two tools are
        # sharing a namespace row and one of them is not addressable.
        built = [build_tool_namespace_name(m, v) for m, v in CENSUS_CORPUS]
        assert len(set(built)) == len(built)

    def test_the_version_segment_is_not_the_natural_version(self) -> None:
        # stated as a test because it is the one lossy half, and a
        # caller that treated the recovered segment as a semver string
        # would be wrong for every three-part version.
        parsed = parse_tool_namespace_name(build_tool_namespace_name("addrnorm.geocode", "1.0.0"))
        assert parsed.version_segment == "1-0-0"
        assert parsed.version_segment != "1.0.0"


class TestTheGrammarUnlocksSubtreeContainment:
    """what the un-flattening is FOR.

    a flattened name is not under its provider node by any
    segment-aware rule, so a subtree grant on ``tools.pentest`` reached
    nothing. these assertions are the reason the invariant that the
    separator is never overloaded was worth spending.
    """

    def test_a_tool_sits_under_its_provider_node(self) -> None:
        built = build_tool_namespace_name("pentest.sqlmap", "1.0")
        assert namespace_contains("tools.pentest", built) is True

    def test_a_tool_sits_under_a_multi_segment_provider_node(self) -> None:
        built = build_tool_namespace_name("aibots.admin.list_agents", "1.0")
        assert namespace_contains("tools.aibots.admin", built) is True

    def test_the_flattened_form_did_not(self) -> None:
        # the A/B that shows the change is load-bearing rather than
        # cosmetic: the old spelling fails this containment.
        flattened = build_namespace_name(PLURAL_PREFIX_TOOL, "pentest.sqlmap", "1.0")
        assert flattened == "tools.pentest-sqlmap.1-0"
        assert namespace_contains("tools.pentest", flattened) is False

    def test_a_provider_node_does_not_reach_an_imposter(self) -> None:
        built = build_tool_namespace_name("pentestimposter.sqlmap", "1.0")
        assert namespace_contains("tools.pentest", built) is False


class TestTheProviderNodeBuilderAgreesWithTheSubjectLayer:
    """the node name is built in TWO packages, and the two must never drift.

    ``threetears.nats.subject_permissions`` mints a tool pod's human-in-the-loop grants
    from the stems on its ``tool_pods`` row, and it cannot import this module -- this
    module imports :mod:`threetears.nats.subjects`, so the dependency runs core -> nats
    and not back. The rooting rule therefore lives in ``Subjects.tool_provider_node`` and
    :func:`build_tool_provider_node_name` delegates to it.

    That leaves one literal in the lower package (``"tools"``) and one constant in this
    one (:data:`PLURAL_PREFIX_TOOL`). This class is the pin between them. A drift would
    not raise anywhere: the grant would name one digest, the pod would subscribe another,
    and an ungranted subscription receives nothing forever.
    """

    def test_the_node_builder_uses_this_packages_tool_prefix(self) -> None:
        """the delegate's literal and ``PLURAL_PREFIX_TOOL`` are the same string.

        :return: none
        :rtype: None
        """
        assert build_tool_provider_node_name("pentest") == f"{PLURAL_PREFIX_TOOL}{NAMESPACE_NAME_SEPARATOR}pentest"

    def test_a_multi_segment_stem_roots_whole(self) -> None:
        """a stem's dots are segment boundaries and survive rooting.

        :return: none
        :rtype: None
        """
        built = build_tool_provider_node_name("aibots.admin")
        assert built == "tools.aibots.admin"

    def test_the_node_contains_the_tools_beneath_it(self) -> None:
        """the property the whole shape exists for, asserted through the builder.

        :return: none
        :rtype: None
        """
        node = build_tool_provider_node_name("pentest")
        assert namespace_contains(node, build_tool_namespace_name("pentest.sqlmap", "1.0")) is True
        assert namespace_contains(node, build_tool_namespace_name("pentestimposter.sqlmap", "1.0")) is False

    def test_an_already_rooted_stem_is_not_doubled(self) -> None:
        """both spellings reach this builder, so rooting must be idempotent.

        the mint holds the bare stem off the ``tool_pods`` row; a pod holds the canonical
        name its registration reply carried back. doubling one of them mints
        ``tools.tools.pentest``, a name nothing resolves and nothing complains about.

        :return: none
        :rtype: None
        """
        assert build_tool_provider_node_name("tools.pentest") == "tools.pentest"

    def test_the_bare_prefix_is_not_a_node(self) -> None:
        """``tools`` is the whole tree, and a family keyed on it would be shared by all.

        :return: none
        :rtype: None
        """
        with pytest.raises(ValueError, match="names the whole tool tree"):
            build_tool_provider_node_name(PLURAL_PREFIX_TOOL)


class TestHitlDerivationOverAnUnflattenedName:
    """``build_hitl_namespace_name`` over a name with N components.

    the hitl builder lifts the tool components out of the tool
    namespace name and splats them between the ``hitl`` prefix and the
    customer. under the flattened shape that was always exactly two
    components; it is now one per mcp segment plus the version, so the
    derivation is exercised at several arities to show that nothing
    counted them.

    a live ``hitl.*`` row does not exist and cannot: the
    ``namespaces_namespace_type_ck`` CHECK admits fifteen values and
    ``hitl`` is not one of them, and the builder has no production
    caller. so these assertions pin the derivation for the day it gains
    one, not a migration hazard.
    """

    CUSTOMER = UUID("7f3c9a1d-1111-4111-8111-000000000001")

    @pytest.mark.parametrize(
        ("mcp_name", "expected_components"),
        [
            ("standalone_tool", 1),
            ("pentest.sqlmap", 2),
            ("aibots.admin.list_agents", 3),
            ("a.b.c.d", 4),
        ],
    )
    def test_the_customer_stays_last_at_every_arity(self, mcp_name: str, expected_components: int) -> None:
        tool_ns = build_tool_namespace_name(mcp_name, "1.0.0")
        name = build_hitl_namespace_name(tool_ns, self.CUSTOMER)
        components = name.split(NAMESPACE_NAME_SEPARATOR)
        # prefix + mcp segments + version + customer
        assert len(components) == expected_components + 3
        assert components[0] == PLURAL_PREFIX_HITL
        assert components[-1] == self.CUSTOMER.hex

    def test_the_tool_half_is_the_tool_name_with_its_prefix_swapped(self) -> None:
        tool_ns = build_tool_namespace_name("aibots.admin.list_agents", "1.0")
        assert tool_ns == "tools.aibots.admin.list_agents.1-0"
        assert build_hitl_namespace_name(tool_ns, self.CUSTOMER) == (
            f"hitl.aibots.admin.list_agents.1-0.{self.CUSTOMER.hex}"
        )

    def test_two_tools_that_used_to_collapse_now_derive_distinct_hitl_names(self) -> None:
        # ``a.b`` and ``a-b`` rendered ONE tool namespace name under the
        # flattened shape, so their hitl rows were one row and an
        # operator entitled to either reached both.
        first = build_hitl_namespace_name(build_tool_namespace_name("a.b", "1.0"), self.CUSTOMER)
        second = build_hitl_namespace_name(build_tool_namespace_name("a-b", "1.0"), self.CUSTOMER)
        assert first != second

    def test_a_provider_node_is_not_a_tool_namespace_name(self) -> None:
        # ``tools.pentest`` is an interior node with no version, so it
        # names no display session; the hitl builder must not mint one
        # whose "version" is a provider.
        name = build_hitl_namespace_name("tools.pentest", self.CUSTOMER)
        # it does not raise -- the hitl builder's contract is only that
        # the value is a strict descendant of the prefix -- so the
        # guard that matters is the tool grammar's own, asserted here
        # so the difference is visible rather than assumed.
        assert name == f"hitl.pentest.{self.CUSTOMER.hex}"
        with pytest.raises(ValueError, match="two components"):
            parse_tool_namespace_name("tools.pentest")


class TestBuildingOrNone:
    """the non-raising door, for callers holding untrusted input.

    :func:`build_tool_namespace_name` REFUSES a malformed pair, which is
    right where the pair is authored -- an emitter, a seed, a migration.
    It is wrong where the pair comes off a URL path segment, a dispatch
    envelope or an operator-typed column: there a refusal has to read as
    "no such tool", because raising turns a 404 into a 500 and turns a
    DENY on an authorization hot path into an exception.

    So the catch is spelled ONCE, here, rather than at each call site
    where one of them would eventually be written as a bare ``except``
    or forgotten entirely.
    """

    def test_a_well_formed_pair_builds(self) -> None:
        assert build_tool_namespace_name_or_none("pentest.sqlmap", "1.0") == "tools.pentest.sqlmap.1-0"

    def test_it_agrees_with_the_raising_builder_on_every_census_entry(self) -> None:
        for mcp_name, version in CENSUS_CORPUS:
            assert build_tool_namespace_name_or_none(mcp_name, version) == build_tool_namespace_name(mcp_name, version)

    @pytest.mark.parametrize(
        ("mcp_name", "version"),
        [
            # every shape reachable from a URL path segment through
            # ``tool_address_candidates``, which enumerates splits and
            # so can compose a doubled or leading separator out of an
            # address a caller typed.
            ("a..geocode", "1.0"),
            (".op", "a.b"),
            ("pentest.", "1.0"),
            ("", "1.0"),
            ("pentest.sqlmap", ""),
            ("tools.pentest.sqlmap", "1.0"),
            ("tools", "1.0"),
        ],
    )
    def test_a_malformed_pair_is_none_rather_than_a_raise(self, mcp_name: str, version: str) -> None:
        assert build_tool_namespace_name_or_none(mcp_name, version) is None

    def test_it_never_returns_an_empty_string(self) -> None:
        # an empty name would be a lookup that matches whatever an
        # empty-name row is, and ``namespace_contains`` treats an empty
        # node as containing nothing -- two different readings of one
        # value. ``None`` is the only unambiguous "no name".
        assert build_tool_namespace_name_or_none("", "") is None
