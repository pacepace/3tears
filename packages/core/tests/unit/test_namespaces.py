"""unit tests for :mod:`threetears.core.namespaces`.

covers the builder + sanitizer pair that ships the canonical
namespace-name shape under namespace-task-01 phase 9.5.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from uuid import UUID

import pytest

from threetears.core.namespaces import (
    HITL_NAMESPACE_TYPE,
    PLURAL_PREFIX_BY_NAMESPACE_TYPE,
    PLURAL_PREFIX_HITL,
    PLURAL_PREFIX_TOOL,
    HitlSessionNamespace,
    build_hitl_namespace_name,
    build_namespace_name,
    namespace_contains,
    sanitize_segment,
)

# Two customers whose first eight hex characters are IDENTICAL. The
# eight-character form is what ``memories.`` and ``intentions.`` names
# use, so it is the plausible wrong spelling here -- and under it these
# two tenants would land on one namespace row. Every cross-tenant
# assertion below is written against this pair so that the wrong
# spelling cannot pass it.
CUSTOMER_X = UUID("7f3c9a1d-1111-4111-8111-000000000001")
CUSTOMER_Z = UUID("7f3c9a1d-2222-4222-8222-000000000002")

# Zone alpha and zone beta at one version, plus zone alpha at a second
# version differing only in its patch component. A name that dropped
# the tool segment collapses the first pair; one that dropped or
# truncated the version collapses the second.
TOOL_NS_ALPHA = build_namespace_name(PLURAL_PREFIX_TOOL, "scrape.zone_alpha", "1.0.0")
TOOL_NS_BETA = build_namespace_name(PLURAL_PREFIX_TOOL, "scrape.zone_beta", "1.0.0")
TOOL_NS_ALPHA_NEXT = build_namespace_name(PLURAL_PREFIX_TOOL, "scrape.zone_alpha", "1.0.1")


class TestSanitizeSegment:
    """tests for :func:`sanitize_segment`."""

    def test_replaces_single_dot(self) -> None:
        assert sanitize_segment("a.b") == "a-b"

    def test_replaces_multiple_dots(self) -> None:
        assert sanitize_segment("claude-sonnet-4.5") == "claude-sonnet-4-5"

    def test_passes_through_no_dot(self) -> None:
        assert sanitize_segment("anthropic") == "anthropic"

    def test_passes_through_empty_string(self) -> None:
        assert sanitize_segment("") == ""

    def test_leaves_hyphens_intact(self) -> None:
        assert sanitize_segment("foo-bar-baz") == "foo-bar-baz"

    def test_leaves_underscores_intact(self) -> None:
        assert sanitize_segment("foo_bar") == "foo_bar"


class TestBuildNamespaceName:
    """tests for :func:`build_namespace_name`."""

    def test_single_segment(self) -> None:
        assert build_namespace_name("datasources", "my_db") == "datasources.my_db"

    def test_two_segments(self) -> None:
        assert build_namespace_name("memories", "aaaaaaaa", "bbbbbbbb") == "memories.aaaaaaaa.bbbbbbbb"

    def test_three_segments(self) -> None:
        assert build_namespace_name("channels", "slack", "12345678", "extra") == "channels.slack.12345678.extra"

    def test_no_segments_yields_prefix_only(self) -> None:
        assert build_namespace_name("system") == "system"

    def test_sanitizes_segment_with_dot(self) -> None:
        assert build_namespace_name("models", "anthropic", "claude-sonnet-4.5") == "models.anthropic.claude-sonnet-4-5"

    def test_sanitizes_every_segment_independently(self) -> None:
        assert build_namespace_name("models", "v1.0", "model.2.3") == "models.v1-0.model-2-3"

    def test_prefix_with_dots_is_not_sanitized(self) -> None:
        # the prefix argument is supplied by the caller from the
        # PLURAL_PREFIX_* constants; it is not sanitized. guard the
        # contract so callers cannot accidentally introduce a dotted
        # prefix.
        assert build_namespace_name("sys.tem", "a") == "sys.tem.a"


class TestPluralPrefixMapping:
    """tests pinning the ``namespace_type`` -> plural prefix contract."""

    @pytest.mark.parametrize(
        ("namespace_type", "expected"),
        [
            ("agent", "agents"),
            ("api_key", "api_keys"),
            ("audit", "audits"),
            ("channel", "channels"),
            ("conversation", "conversations"),
            ("customer", "customers"),
            ("datasource", "datasources"),
            ("hitl", "hitl"),
            ("knowledge", "knowledge"),
            ("memory", "memories"),
            ("model", "models"),
            ("shared", "shared"),
            ("shared_agent", "shared_agents"),
            ("system", "system"),
            ("tool", "tools"),
            ("workspace", "workspaces"),
        ],
    )
    def test_namespace_type_maps_to_plural_prefix(self, namespace_type: str, expected: str) -> None:
        assert PLURAL_PREFIX_BY_NAMESPACE_TYPE[namespace_type] == expected

    def test_mapping_closed_set_size(self) -> None:
        # pins the closed set; adding a new namespace_type requires
        # updating the hub namespaces CHECK constraint + this mapping.
        # namespace-task-01 phase 11 adds ``audit``; phase 12 adds
        # ``customer`` + ``api_key``; knowledge-task-01 adds
        # ``knowledge`` (mass noun, non-pluralized prefix).
        assert len(PLURAL_PREFIX_BY_NAMESPACE_TYPE) == 16


class TestBuildHitlNamespaceName:
    """tests for :func:`build_hitl_namespace_name`."""

    def test_matches_the_documented_shape(self) -> None:
        assert TOOL_NS_ALPHA == "tools.scrape-zone_alpha.1-0-0"
        assert build_hitl_namespace_name(TOOL_NS_ALPHA, CUSTOMER_X) == f"hitl.scrape-zone_alpha.1-0-0.{CUSTOMER_X.hex}"

    def test_components_are_the_tool_components_between_prefix_and_customer(self) -> None:
        name = build_hitl_namespace_name(TOOL_NS_ALPHA, CUSTOMER_X)
        assert name.split(".") == [PLURAL_PREFIX_HITL, *TOOL_NS_ALPHA.split(".")[1:], CUSTOMER_X.hex]

    def test_two_zones_at_one_version_do_not_collide(self) -> None:
        # the customer-only shape passes every cross-tenant assertion
        # and fails this one: an operator entitled to this customer on
        # one network would reach its display on another.
        alpha = build_hitl_namespace_name(TOOL_NS_ALPHA, CUSTOMER_X)
        beta = build_hitl_namespace_name(TOOL_NS_BETA, CUSTOMER_X)
        assert alpha != beta

    def test_two_versions_of_one_zone_do_not_collide(self) -> None:
        # entitlement is per version, because tools.<mcp>.<version> is
        # its own row with its own assignments.
        first = build_hitl_namespace_name(TOOL_NS_ALPHA, CUSTOMER_X)
        second = build_hitl_namespace_name(TOOL_NS_ALPHA_NEXT, CUSTOMER_X)
        assert first != second

    def test_two_customers_of_one_zone_do_not_collide(self) -> None:
        assert build_hitl_namespace_name(TOOL_NS_ALPHA, CUSTOMER_X) != build_hitl_namespace_name(
            TOOL_NS_ALPHA, CUSTOMER_Z
        )

    def test_carries_the_full_customer_hex(self) -> None:
        # the two constants share their first eight hex characters, so
        # a truncated spelling would put both tenants on one row.
        assert CUSTOMER_X.hex[:8] == CUSTOMER_Z.hex[:8]
        assert build_hitl_namespace_name(TOOL_NS_ALPHA, CUSTOMER_X).endswith(f".{CUSTOMER_X.hex}")

    def test_a_dot_in_a_tool_name_cannot_add_a_component(self) -> None:
        # nothing validates a tool's mcp name, so this input is a
        # property to hold rather than a convention to follow. the dots
        # were already mapped to ``-`` when the tool namespace name was
        # built, which is what keeps the customer the last component.
        hostile_tool_ns = build_namespace_name(PLURAL_PREFIX_TOOL, "scrape zone_alpha.*.>", "1.0.0")
        name = build_hitl_namespace_name(hostile_tool_ns, CUSTOMER_X)
        assert len(name.split(".")) == 4
        assert name.split(".")[-1] == CUSTOMER_X.hex
        assert name != build_hitl_namespace_name(TOOL_NS_ALPHA, CUSTOMER_X)

    def test_a_hostile_tool_name_keeps_its_characters(self) -> None:
        # DELIBERATE: a namespace name is a database row value, not a
        # subject token. the digesting that makes an unvalidated tool
        # name safe happens where subjects are built; doing it here as
        # well would cost the legibility the readable segment exists
        # for. only ``.`` is touched, by the shared sanitizer.
        name = build_hitl_namespace_name(
            build_namespace_name(PLURAL_PREFIX_TOOL, "scrape zone_alpha *>", "1.0.0"),
            CUSTOMER_X,
        )
        assert name == f"hitl.scrape zone_alpha *>.1-0-0.{CUSTOMER_X.hex}"

    @pytest.mark.parametrize(
        "not_a_tool_name",
        [
            "hitl.scrape-zone_alpha.1-0-0",
            "memories.aaaaaaaa.bbbbbbbb",
            "tools",
            "toolsx.a.b",
            "",
        ],
    )
    def test_rejects_a_name_that_is_not_a_tool_namespace_name(self, not_a_tool_name: str) -> None:
        with pytest.raises(ValueError, match="must start with"):
            build_hitl_namespace_name(not_a_tool_name, CUSTOMER_X)

    @pytest.mark.parametrize("degenerate", ["tools.", "tools.scrape-zone_alpha.", "tools..1-0-0"])
    def test_rejects_an_empty_component(self, degenerate: str) -> None:
        with pytest.raises(ValueError, match="empty component"):
            build_hitl_namespace_name(degenerate, CUSTOMER_X)


class TestHitlSessionNamespace:
    """tests for the :class:`HitlSessionNamespace` seam."""

    def test_states_the_name_the_builder_renders(self) -> None:
        session = HitlSessionNamespace(tool_namespace_name=TOOL_NS_ALPHA, customer_id=CUSTOMER_X)
        assert session.namespace_name == build_hitl_namespace_name(TOOL_NS_ALPHA, CUSTOMER_X)

    def test_states_the_namespace_type_of_the_row_it_names(self) -> None:
        session = HitlSessionNamespace(tool_namespace_name=TOOL_NS_ALPHA, customer_id=CUSTOMER_X)
        assert session.namespace_type == HITL_NAMESPACE_TYPE
        assert PLURAL_PREFIX_BY_NAMESPACE_TYPE[session.namespace_type] == PLURAL_PREFIX_HITL

    def test_validates_at_construction_rather_than_at_attach_time(self) -> None:
        with pytest.raises(ValueError, match="must start with"):
            HitlSessionNamespace(tool_namespace_name="scrape-zone_alpha.1-0-0", customer_id=CUSTOMER_X)

    def test_is_frozen(self) -> None:
        session = HitlSessionNamespace(tool_namespace_name=TOOL_NS_ALPHA, customer_id=CUSTOMER_X)
        with pytest.raises(FrozenInstanceError):
            session.tool_namespace_name = TOOL_NS_BETA  # type: ignore[misc]


class TestNamespaceContains:
    """tests for :func:`namespace_contains`, the ONE containment rule.

    the rule is ``name == node or name.startswith(node + separator)``,
    which is segment-aware BY CONSTRUCTION: the only character that may
    follow the node is the separator itself, so a sibling whose name
    merely begins with the node's characters can never match.
    """

    def test_a_node_contains_its_child(self) -> None:
        assert namespace_contains("tools.dipp", "tools.dipp.thing") is True

    def test_a_node_contains_a_deeper_descendant(self) -> None:
        assert namespace_contains("tools.dipp", "tools.dipp.thing.inner") is True

    def test_a_node_does_not_contain_a_sibling_sharing_its_prefix(self) -> None:
        # the bug class: a raw prefix test admits this, and it is why
        # every allowed_namespaces value used to carry a trailing dot.
        assert namespace_contains("tools.dipp", "tools.dippX.thing") is False

    def test_pentest_does_not_reach_pentestimposter(self) -> None:
        assert namespace_contains("pentest", "pentestimposter.sqlmap") is False
        assert namespace_contains("pentest", "pentestimposter") is False

    def test_a_node_contains_itself(self) -> None:
        assert namespace_contains("tools.dipp", "tools.dipp") is True

    def test_an_empty_node_contains_nothing(self) -> None:
        # NOT everything: an empty node under a raw prefix test matches
        # every name in the system, which is the widest possible grant
        # arriving from the emptiest possible value.
        assert namespace_contains("", "tools.dipp") is False
        assert namespace_contains("", "") is False

    def test_an_empty_name_is_contained_by_nothing(self) -> None:
        assert namespace_contains("tools.dipp", "") is False

    def test_case_does_not_widen(self) -> None:
        assert namespace_contains("tools.DIPP", "tools.dipp.thing") is False
        assert namespace_contains("tools.dipp", "TOOLS.DIPP.thing") is False

    def test_trailing_whitespace_does_not_widen(self) -> None:
        # a node carrying stray whitespace grants NOTHING rather than
        # being silently stripped into a node that grants a subtree.
        assert namespace_contains("tools.dipp ", "tools.dipp.thing") is False
        assert namespace_contains("tools.dipp ", "tools.dipp") is False

    def test_leading_whitespace_does_not_widen(self) -> None:
        assert namespace_contains(" tools.dipp", "tools.dipp.thing") is False

    def test_a_node_written_with_a_trailing_separator_is_not_a_convention(self) -> None:
        # the trailing-dot workaround is GONE, not accommodated: a value
        # written the old way matches nothing, which is a visible
        # failure rather than a silent one.
        assert namespace_contains("tools.dipp.", "tools.dipp.thing") is False

    def test_a_child_does_not_contain_its_parent(self) -> None:
        assert namespace_contains("tools.dipp.thing", "tools.dipp") is False

    def test_the_top_prefix_contains_every_name_under_it(self) -> None:
        assert namespace_contains(PLURAL_PREFIX_TOOL, TOOL_NS_ALPHA) is True
        assert namespace_contains(PLURAL_PREFIX_TOOL, "toolsimposter.x") is False

    def test_underscores_and_hyphens_are_ordinary_characters(self) -> None:
        # sanitize_segment leaves both intact, so a real name carries
        # them and containment must not treat either as a boundary.
        assert namespace_contains("tools.scrape-zone_alpha", "tools.scrape-zone_alpha.1-0-0") is True
        assert namespace_contains("tools.scrape-zone", "tools.scrape-zone_alpha.1-0-0") is False
