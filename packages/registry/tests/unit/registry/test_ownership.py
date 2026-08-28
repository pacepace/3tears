"""unit -- the one rule that decides whether a pod may register a tool name.

Registration used to ask three different questions on three different paths: a
token-bearing pod's tools were prefix-filtered against a text column, a tokenless
pod's were not filtered at all, and a registry with no authenticator filtered
nothing either. :mod:`threetears.registry.ownership` replaces all three with one
question asked of the namespace GRAPH -- who owns the most specific provider node
that contains this name.

Every refusal here is paired with an admitted twin. A rule that refuses everything
passes a refusal test for the wrong reason, and the pairs are what separate "this
name is refused" from "nothing gets through".
"""

from __future__ import annotations

import pytest

from threetears.registry.ownership import (
    most_specific_container,
    tool_is_registrable,
)

__all__: list[str] = []


class TestMostSpecificContainer:
    """which node decides, when more than one contains the name."""

    def test_returns_the_only_containing_node(self) -> None:
        """one container, and it is the answer."""
        assert most_specific_container(("tools.pentest", "tools.dipp"), "tools.pentest.sqlmap") == "tools.pentest"

    def test_returns_none_when_no_node_contains_the_name(self) -> None:
        """unowned territory is a real answer, not an error."""
        assert most_specific_container(("tools.pentest",), "tools.dipp.thing") is None

    def test_a_sibling_sharing_a_prefix_does_not_contain(self) -> None:
        """``tools.pentest`` must never reach ``tools.pentestimposter``.

        The paired admission below is what shows the comparison is live rather
        than uniformly negative.
        """
        assert most_specific_container(("tools.pentest",), "tools.pentestimposter.sqlmap") is None
        assert most_specific_container(("tools.pentest",), "tools.pentest.sqlmap") == "tools.pentest"

    def test_the_deeper_node_wins_over_its_parent(self) -> None:
        """most-specific owner wins, and it is decided rather than left to order.

        Both orderings are asserted: a rule that merely returned the first match
        would pass one of these and fail the other.
        """
        nodes = ("tools.aibots", "tools.aibots.admin")
        assert most_specific_container(nodes, "tools.aibots.admin.list_pods") == "tools.aibots.admin"
        assert most_specific_container(tuple(reversed(nodes)), "tools.aibots.admin.list_pods") == "tools.aibots.admin"

    def test_the_parent_still_wins_a_name_the_child_does_not_contain(self) -> None:
        """specificity, not blanket preference for the longer node."""
        nodes = ("tools.aibots", "tools.aibots.admin")
        assert most_specific_container(nodes, "tools.aibots.other.thing") == "tools.aibots"

    def test_a_node_contains_itself(self) -> None:
        """the node's own name is inside it, per the containment rule."""
        assert most_specific_container(("tools.pentest",), "tools.pentest") == "tools.pentest"

    def test_an_empty_directory_contains_nothing(self) -> None:
        """no graph, no container -- the open-mode answer."""
        assert most_specific_container((), "tools.pentest.sqlmap") is None


class TestToolIsRegistrable:
    """the decision the three registration paths now share."""

    def test_an_owner_may_register_beneath_its_own_node(self) -> None:
        """the admitted twin for every refusal below."""
        assert tool_is_registrable(
            tool_name="dipp.thing",
            owned_nodes=("tools.dipp",),
            provider_nodes=("tools.dipp", "tools.other"),
        )

    def test_an_owner_may_not_register_beneath_another_owners_node(self) -> None:
        """the refusal this chunk exists for."""
        assert not tool_is_registrable(
            tool_name="other.thing",
            owned_nodes=("tools.dipp",),
            provider_nodes=("tools.dipp", "tools.other"),
        )

    def test_an_owner_may_register_its_own_node_exactly(self) -> None:
        """a tool named exactly as the node is inside it."""
        assert tool_is_registrable(
            tool_name="dipp",
            owned_nodes=("tools.dipp",),
            provider_nodes=("tools.dipp",),
        )

    def test_a_bound_pod_may_not_claim_unowned_territory(self) -> None:
        """a pod that owns nodes stays inside them.

        This preserves what the old positive filter did: a token-bearing pod
        offering a name under no provider node at all was refused, and still is.
        Paired with the admission of a name that IS under its node.
        """
        assert not tool_is_registrable(
            tool_name="brandnew.thing",
            owned_nodes=("tools.dipp",),
            provider_nodes=("tools.dipp",),
        )
        assert tool_is_registrable(
            tool_name="dipp.thing",
            owned_nodes=("tools.dipp",),
            provider_nodes=("tools.dipp",),
        )

    def test_an_unbound_pod_may_claim_unowned_territory(self) -> None:
        """the agent-owned in-process pod: filtered, not exempt.

        It owns no provider node, so its own tools -- which sit under no
        provider -- are admitted, exactly as before.
        """
        assert tool_is_registrable(
            tool_name="myagent.summarize",
            owned_nodes=(),
            provider_nodes=("tools.dipp",),
        )

    def test_an_unbound_pod_may_not_encroach_on_an_owned_node(self) -> None:
        """what was previously unfiltered, now refused."""
        assert not tool_is_registrable(
            tool_name="dipp.thing",
            owned_nodes=(),
            provider_nodes=("tools.dipp",),
        )

    def test_an_empty_graph_admits_an_unbound_pod(self) -> None:
        """open mode: a registry with no view of the graph enforces nothing.

        Deliberate and stated: there is no ownership data to decide with, and
        refusing everything would break a registry running without an
        authenticator. The path is the same one, and its answer differs only
        because the graph is empty.
        """
        assert tool_is_registrable(tool_name="anything.at.all", owned_nodes=(), provider_nodes=())

    def test_the_deeper_owner_wins_against_the_parents_owner(self) -> None:
        """parent-vs-child precedence, at the decision rather than in the helper."""
        nodes = ("tools.aibots", "tools.aibots.admin")
        assert tool_is_registrable(
            tool_name="aibots.admin.list_pods",
            owned_nodes=("tools.aibots.admin",),
            provider_nodes=nodes,
        )
        assert not tool_is_registrable(
            tool_name="aibots.admin.list_pods",
            owned_nodes=("tools.aibots",),
            provider_nodes=nodes,
        )

    def test_a_rooted_manifest_name_is_refused_even_from_the_provider_owner(self) -> None:
        """a namespace name offered where an mcp name belongs is a category error.

        Refused for the node's OWNER, which is the case that would otherwise slip
        through, and refused for a pod owning nothing, which is the evasion it
        would otherwise open. Paired with the bare mcp name that is admitted.
        """
        assert not tool_is_registrable(
            tool_name="tools.dipp.thing",
            owned_nodes=("tools.dipp",),
            provider_nodes=("tools.dipp",),
        )
        assert not tool_is_registrable(
            tool_name="tools.dipp.thing",
            owned_nodes=(),
            provider_nodes=("tools.dipp",),
        )
        assert tool_is_registrable(
            tool_name="dipp.thing",
            owned_nodes=("tools.dipp",),
            provider_nodes=("tools.dipp",),
        )

    @pytest.mark.parametrize("unusable", ["", "tools", "   "])
    def test_a_name_that_composes_no_node_is_refused(self, unusable: str) -> None:
        """an empty name, or the bare tree root, names no provider and is refused.

        ``tools`` is refused because it is the whole tool tree rather than one
        provider; admitting it would let one pod's manifest sit above every
        node in the graph. Paired with the admission on the same directory.
        """
        assert not tool_is_registrable(
            tool_name=unusable,
            owned_nodes=("tools.dipp",),
            provider_nodes=("tools.dipp",),
        )
        assert tool_is_registrable(
            tool_name="dipp.thing",
            owned_nodes=("tools.dipp",),
            provider_nodes=("tools.dipp",),
        )

    def test_a_malformed_owned_node_grants_nothing(self) -> None:
        """``dipp.`` and ``dipp.*`` are not nodes and match nothing.

        The old column carried both shapes. Neither can win the containment
        comparison, so a pod holding only those is bound and reaches nothing --
        which is the fail-closed direction, and is paired with the well-formed
        node that does reach its tool.
        """
        assert not tool_is_registrable(
            tool_name="dipp.thing",
            owned_nodes=("tools.dipp.", "tools.dipp.*"),
            provider_nodes=("tools.dipp",),
        )
        assert tool_is_registrable(
            tool_name="dipp.thing",
            owned_nodes=("tools.dipp",),
            provider_nodes=("tools.dipp",),
        )
