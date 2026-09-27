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

from threetears.agent.tools.server import FINAL_REFUSAL_CODES
from threetears.registry.ownership import (
    CopyAudience,
    PublisherStanding,
    RefusalCode,
    admit_copy,
    audience_of,
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


class TestAdmitCopy:
    """the per-copy verdict: who may publish a copy, and to whom it may be served.

    Every refusal is paired with the admitted twin that differs in one input, so a
    rule that refused everything would fail a twin.
    """

    _GRAPH = ("tools.pentest", "tools.aibots.admin")

    @staticmethod
    def _standing(
        *, verified: bool = True, platform_shared: bool = False, owned: tuple[str, ...] = ()
    ) -> PublisherStanding:
        return PublisherStanding(verified=verified, platform_shared=platform_shared, owned_nodes=owned)

    def test_a_serve_everyone_copy_under_a_node_needs_its_owner(self) -> None:
        """OWNED_ELSEWHERE for anybody else; admitted for the owner."""
        stray = self._standing()
        owner = self._standing(owned=("tools.pentest",))
        assert (
            admit_copy(
                tool_name="pentest.sqlmap", audience=CopyAudience.EVERYONE, standing=stray, provider_nodes=self._GRAPH
            )
            is RefusalCode.OWNED_ELSEWHERE
        )
        assert (
            admit_copy(
                tool_name="pentest.sqlmap", audience=CopyAudience.EVERYONE, standing=owner, provider_nodes=self._GRAPH
            )
            is None
        )

    def test_a_serve_everyone_copy_under_no_node_needs_the_platform(self) -> None:
        """NOT_PLATFORM_SHARED for a verified pod owning nothing; admitted for the shared pod."""
        stray = self._standing()
        shared = self._standing(platform_shared=True)
        assert (
            admit_copy(
                tool_name="threetears.calculator",
                audience=CopyAudience.EVERYONE,
                standing=stray,
                provider_nodes=self._GRAPH,
            )
            is RefusalCode.NOT_PLATFORM_SHARED
        )
        assert (
            admit_copy(
                tool_name="threetears.calculator",
                audience=CopyAudience.EVERYONE,
                standing=shared,
                provider_nodes=self._GRAPH,
            )
            is None
        )

    def test_a_pod_that_owns_a_node_is_not_the_platform_outside_it(self) -> None:
        """owning pentest buys nothing under no node."""
        owner = self._standing(owned=("tools.pentest",))
        assert (
            admit_copy(
                tool_name="brandnew.thing", audience=CopyAudience.EVERYONE, standing=owner, provider_nodes=self._GRAPH
            )
            is RefusalCode.NOT_PLATFORM_SHARED
        )

    def test_the_platform_does_not_reach_into_an_owned_node(self) -> None:
        """the shared pod is refused under somebody's node like anyone else."""
        shared = self._standing(platform_shared=True)
        assert (
            admit_copy(
                tool_name="pentest.sqlmap", audience=CopyAudience.EVERYONE, standing=shared, provider_nodes=self._GRAPH
            )
            is RefusalCode.OWNED_ELSEWHERE
        )

    def test_an_unverified_publisher_may_not_serve_everyone(self) -> None:
        """UNVERIFIED_PUBLISHER even where a verified pod would be admitted."""
        unverified = self._standing(verified=False, platform_shared=True)
        assert (
            admit_copy(
                tool_name="threetears.calculator",
                audience=CopyAudience.EVERYONE,
                standing=unverified,
                provider_nodes=self._GRAPH,
            )
            is RefusalCode.UNVERIFIED_PUBLISHER
        )

    def test_an_agent_scoped_copy_is_admitted_under_no_node(self) -> None:
        """an agent's own copy serves only that agent, so it needs no platform standing."""
        agent = self._standing()
        assert (
            admit_copy(
                tool_name="threetears.calculator",
                audience=CopyAudience.AGENT,
                standing=agent,
                provider_nodes=self._GRAPH,
            )
            is None
        )

    def test_an_agent_scoped_copy_is_refused_under_somebodys_node(self) -> None:
        """the existing ownership rule still holds for an agent's own copy."""
        agent = self._standing()
        assert (
            admit_copy(
                tool_name="pentest.sqlmap", audience=CopyAudience.AGENT, standing=agent, provider_nodes=self._GRAPH
            )
            is RefusalCode.OWNED_ELSEWHERE
        )

    def test_an_unsigned_agent_scoped_copy_is_admitted_during_the_rollout(self) -> None:
        """0.55.0 admits an unsigned agent's own copy -- still only where a signed one would be."""
        unsigned = self._standing(verified=False)
        assert (
            admit_copy(
                tool_name="threetears.calculator",
                audience=CopyAudience.AGENT,
                standing=unsigned,
                provider_nodes=self._GRAPH,
            )
            is None
        )
        assert (
            admit_copy(
                tool_name="pentest.sqlmap", audience=CopyAudience.AGENT, standing=unsigned, provider_nodes=self._GRAPH
            )
            is RefusalCode.OWNED_ELSEWHERE
        )

    @pytest.mark.parametrize("audience", [CopyAudience.AGENT, CopyAudience.EVERYONE])
    def test_a_name_that_composes_no_node_is_invalid_for_any_audience(self, audience: CopyAudience) -> None:
        """a rooted or empty name is refused before anything else is asked."""
        shared = self._standing(platform_shared=True)
        for bad in ("tools.pentest.sqlmap", ""):
            assert (
                admit_copy(tool_name=bad, audience=audience, standing=shared, provider_nodes=self._GRAPH)
                is RefusalCode.INVALID_TOOL_NAME
            )

    def test_open_mode_enforces_nothing_but_the_name(self) -> None:
        """no authenticator: every audience admitted, a malformed name still refused."""
        standing = PublisherStanding.unenforced()
        assert (
            admit_copy(
                tool_name="anything.at.all", audience=CopyAudience.EVERYONE, standing=standing, provider_nodes=()
            )
            is None
        )
        assert (
            admit_copy(tool_name="tools.x", audience=CopyAudience.EVERYONE, standing=standing, provider_nodes=())
            is RefusalCode.INVALID_TOOL_NAME
        )


class TestAudienceOf:
    """who a copy serves is read from its pod id, the one source routing reads too."""

    def test_an_agents_in_process_pod_serves_that_agent(self) -> None:
        """a composite id names the owning agent."""
        from uuid import UUID

        from threetears.nats import Subjects

        pod_id = Subjects.agent_inprocess_pod_id(UUID("01948a00-aaaa-7000-8000-00000000000a"), "inst")
        assert audience_of(pod_id) is CopyAudience.AGENT

    def test_a_tool_pod_serves_everyone(self) -> None:
        """a single-token id is a Tool Pod's."""
        assert audience_of("builtin-tool-server") is CopyAudience.EVERYONE


class TestThePodsFinalRefusalsAreRegistryCodes:
    """the pod decides which refusals end readiness from ``FINAL_REFUSAL_CODES``; each must be a
    code this registry actually sends, and the transient one must not be among them."""

    def test_every_final_code_is_a_refusal_code(self) -> None:
        """a misspelled final code would never match, and its refusal would be waited out forever.

        :return: none
        :rtype: None
        """
        codes = {code.value for code in RefusalCode}
        assert FINAL_REFUSAL_CODES
        assert codes
        assert FINAL_REFUSAL_CODES <= codes

    def test_the_graph_being_unreadable_is_not_final(self) -> None:
        """the registry's own reason says the next heartbeat retries.

        :return: none
        :rtype: None
        """
        assert RefusalCode.OWNERSHIP_GRAPH_UNAVAILABLE.value not in FINAL_REFUSAL_CODES

    def test_a_store_the_host_could_not_read_is_not_final(self) -> None:
        """the registry refuses for want of its own store; the next heartbeat is a real retry.

        :return: none
        :rtype: None
        """
        assert RefusalCode.PUBLISHER_VERIFICATION_UNAVAILABLE.value not in FINAL_REFUSAL_CODES
        assert RefusalCode.CATALOG_UNAVAILABLE.value not in FINAL_REFUSAL_CODES
