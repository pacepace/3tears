"""unit -- ``threetears`` is co-hosted, so no ``tool_provider`` node may claim it.

**The shape of the framework namespace, which the ownership rule cannot express.**
A provider node admits exactly ONE owner: the most specific node containing an
offered name decides, and only that node's owner may register under it. That is
right for a provider -- ``pentest`` is served by the pentest pod and by nothing
else. It is wrong for ``threetears``, because two different kinds of process serve
tools beneath that stem by design:

* the shared built-in tool pod (``threetears.agent.tools.serve``) hosts the half
  that needs infrastructure or credentials -- ``web_search`` wants a SearXNG
  endpoint, ``web_fetch`` an HTML extractor, ``parse_document`` an OCR stack;
* every AGENT process hosts the half that cannot leave it.
  :data:`~threetears.agent.tools.builtin.STANDARD_BUILTIN_FACTORIES` is
  materialised on the agent's own in-process ``ToolServer``, and
  ``threetears.context_recall`` in particular resolves the LIVE per-conversation
  context manager, which a separate pod does not have. The 19-tool
  :data:`~threetears.agent.tools.aliases.WORKSPACE_TOOLS` bundle is in-process for
  the same reason -- it holds a lease on the agent's own ``bind_root``.

So a host that declares ``threetears`` as one pod's provider node refuses every
one of those in-process tools, on every agent, and ``context_recall`` and the
workspace bundle are then served by NOBODY. The rule is not at fault and is not
softened here: what this module pins is that the rule's existing answer for
unclaimed territory -- claimable only by a pod that owns no node -- is the one that
serves a co-hosted namespace, and that claiming the stem is what breaks it.

Every refusal below is paired with an admitted twin, per the sibling module.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from threetears.agent.tools.aliases import WORKSPACE_TOOLS
from threetears.agent.tools.builtin import STANDARD_BUILTIN_FACTORIES
from threetears.agent.tools.server import RegistrationManifest, ToolManifestEntry
from threetears.core.namespaces import build_tool_provider_node_name
from threetears.nats import IncomingMessage, set_default_namespace
from threetears.registry.auth import ToolPodAuth
from threetears.registry.catalog import ToolCatalog
from threetears.registry.ownership import tool_is_registrable
from threetears.registry.registration import RegistrationHandler

__all__: list[str] = []

#: every tool an agent registers on its OWN in-process ``ToolServer``, taken from
#: the modules that define them rather than restated, so a builtin added upstream
#: is covered the day it lands.
_IN_PROCESS: tuple[str, ...] = (*sorted(STANDARD_BUILTIN_FACTORIES), *sorted(WORKSPACE_TOOLS))

#: a graph in which real providers are owned and the framework stem is not.
_UNCLAIMED_GRAPH: tuple[str, ...] = ("tools.pentest", "tools.aibots.admin")

#: the same graph with the framework stem claimed by somebody. This is the state
#: that broke every agent, and it is kept here as the A/B rather than described.
_CLAIMED_GRAPH: tuple[str, ...] = (*_UNCLAIMED_GRAPH, "tools.threetears")


class TestAnAgentHostsHalfTheFrameworkNamespace:
    """the in-process half, against a graph that leaves the stem alone."""

    @pytest.mark.parametrize("tool_name", _IN_PROCESS)
    def test_an_agent_owning_no_node_may_register_its_in_process_tools(self, tool_name: str) -> None:
        """the agent-owned pod is tokenless and owns nothing, and must still register."""
        assert tool_is_registrable(
            tool_name=tool_name,
            owned_nodes=(),
            provider_nodes=_UNCLAIMED_GRAPH,
        )

    @pytest.mark.parametrize("tool_name", ["threetears.web_search", "threetears.calculator"])
    def test_the_shared_builtin_pod_owning_no_node_may_register_its_half(self, tool_name: str) -> None:
        """the shared pod is bound to no provider either, and is admitted the same way.

        Both halves of the namespace reach the catalog through the SAME clause, which
        is what makes co-hosting expressible at all.
        """
        assert tool_is_registrable(
            tool_name=tool_name,
            owned_nodes=(),
            provider_nodes=_UNCLAIMED_GRAPH,
        )


class TestClaimingTheStemIsWhatBreaksIt:
    """the A/B: the same names, the same pod, one extra node in the graph."""

    @pytest.mark.parametrize("tool_name", ["threetears.context_recall", "threetears.workspace.fs_read"])
    def test_a_claimed_stem_refuses_every_in_process_tool(self, tool_name: str) -> None:
        """the cause, pinned rather than described.

        ``context_recall`` and the workspace bundle are hosted by no other process,
        so a refusal here is not a fallback to the shared pod -- the tool simply never
        enters the catalog. And the agent does not stop: a refused registration is a
        warning on the pod, and the readiness barrier derives what it expects FROM the
        catalog, so a name nobody published is a name nobody waits for. The agent comes
        up healthy, missing a capability, saying nothing.
        """
        assert not tool_is_registrable(
            tool_name=tool_name,
            owned_nodes=(),
            provider_nodes=_CLAIMED_GRAPH,
        )
        assert tool_is_registrable(
            tool_name=tool_name,
            owned_nodes=(),
            provider_nodes=_UNCLAIMED_GRAPH,
        )

    def test_only_the_claimants_own_pod_survives_a_claimed_stem(self) -> None:
        """and it is a genuine ownership answer, not a uniform refusal."""
        assert tool_is_registrable(
            tool_name="threetears.calculator",
            owned_nodes=("tools.threetears",),
            provider_nodes=_CLAIMED_GRAPH,
        )


class TestUnclaimedIsNotUnguarded:
    """leaving the stem unowned must not read as leaving it open to everyone."""

    @pytest.mark.parametrize("tool_name", ["threetears.calculator", "threetears.workspace.fs_read"])
    def test_a_pod_bound_to_a_provider_still_may_not_claim_a_framework_name(self, tool_name: str) -> None:
        """a bound pod stays inside its own provider, exactly as for any other name."""
        assert not tool_is_registrable(
            tool_name=tool_name,
            owned_nodes=("tools.pentest",),
            provider_nodes=_UNCLAIMED_GRAPH,
        )
        assert tool_is_registrable(
            tool_name="pentest.sqlmap",
            owned_nodes=("tools.pentest",),
            provider_nodes=_UNCLAIMED_GRAPH,
        )

    @pytest.mark.parametrize("tool_name", ["pentest.sqlmap", "aibots.admin.list_pods"])
    def test_an_unbound_pod_still_may_not_reach_into_an_owned_provider(self, tool_name: str) -> None:
        """the agent that gained its framework tools gained nothing else with them."""
        assert not tool_is_registrable(
            tool_name=tool_name,
            owned_nodes=(),
            provider_nodes=_UNCLAIMED_GRAPH,
        )


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    """bind a deterministic subject namespace for the reachability probes."""
    set_default_namespace("test")


class _Directory:
    """a ``ToolPodAuthenticator`` double reporting one ownership graph.

    ``verify_pod`` refuses every token: the pod under test is the AGENT-OWNED
    in-process server, which presents none. What the handler asks this for is the
    inventory, and the inventory is the whole variable.
    """

    def __init__(self, *nodes: str) -> None:
        """record the provider stems this graph holds.

        :param nodes: provider stems, rooted by the one builder
        :ptype nodes: str
        :return: nothing
        :rtype: None
        """
        self._nodes = tuple(build_tool_provider_node_name(stem) for stem in nodes)

    async def verify_pod(self, token: str) -> ToolPodAuth | None:
        """refuse every token; an agent-owned pod carries none.

        :param token: the presented token
        :ptype token: str
        :return: always ``None``
        :rtype: ToolPodAuth | None
        """
        del token
        return None

    async def provider_nodes(self) -> tuple[str, ...]:
        """the whole inventory this graph holds.

        :return: canonical provider node names
        :rtype: tuple[str, ...]
        """
        return self._nodes


def _probing_nc() -> AsyncMock:
    """a NATS wrapper that answers every reachability probe.

    :return: the configured mock
    :rtype: AsyncMock
    """

    async def _reply(*, subject: Any, message: Any, response_type: Any, timeout: Any) -> Any:
        del message, timeout
        return response_type(pod_id=subject.path.rsplit(".", 1)[-1], ready=True)

    nc = AsyncMock()
    nc.request = AsyncMock(side_effect=_reply)
    return nc


def _in_process_manifest() -> bytes:
    """the manifest an agent's in-process ToolServer publishes, tokenless.

    Every name comes from the upstream factory maps, so this is the real offer
    rather than a sample of it.

    :return: the serialized manifest
    :rtype: bytes
    """
    entries = [
        ToolManifestEntry(
            name=name,
            version="1.0.0",
            description=f"{name} tool",
            input_schema={"type": "object", "properties": {}},
        )
        for name in _IN_PROCESS
    ]
    manifest = RegistrationManifest(pod_id="agent-pod-001", tools=entries, bootstrap_token=None)
    return manifest.model_dump_json().encode("utf-8")


async def _register(directory: _Directory) -> tuple[Any, ToolCatalog]:
    """drive one whole registration of the in-process manifest.

    :param directory: the ownership graph the handler reads
    :ptype directory: _Directory
    :return: the reply the handler published, and the catalog it wrote
    :rtype: tuple[Any, ToolCatalog]
    """
    catalog = ToolCatalog()
    handler = RegistrationHandler(catalog, namespace="test", authenticator=directory)
    nc = _probing_nc()
    await handler.start(nc)
    await handler.handle_registration(
        IncomingMessage(data=_in_process_manifest(), reply_subject="reply.to", subject="test.tools.register")
    )
    return nc.publish_reply.await_args.kwargs["message"], catalog


class TestTheWholeInProcessManifestReachesTheCatalog:
    """the end-to-end half: a real manifest through the real handler.

    The rule is exercised above as a function. This drives the offer an agent
    actually publishes -- every standard builtin and the whole workspace bundle, in
    one tokenless manifest -- through manifest parsing, the ownership filter, the
    reachability probe and the catalog write, and asserts what a running agent
    would find waiting for its readiness gate.
    """

    @pytest.mark.asyncio
    async def test_every_name_is_registered_when_the_stem_is_unclaimed(self) -> None:
        """all 25 of them, and the catalog holds each one."""
        reply, catalog = await _register(_Directory("pentest", "aibots.admin"))

        assert reply.success is True
        assert sorted(reply.registered_tools) == sorted(f"{name}@1.0.0" for name in _IN_PROCESS)
        for name in _IN_PROCESS:
            assert catalog.get(f"{name}@1.0.0") is not None, name

    @pytest.mark.asyncio
    async def test_claiming_the_stem_empties_the_catalog(self) -> None:
        """the A/B, at the same level: one extra node and the agent registers nothing.

        The reply is a refusal rather than a partial success, because the filter
        rejects every name the manifest offered and a manifest with no surviving tool
        is refused outright.
        """
        reply, catalog = await _register(_Directory("pentest", "aibots.admin", "threetears"))

        assert reply.success is False
        for name in _IN_PROCESS:
            assert catalog.get(f"{name}@1.0.0") is None, name
