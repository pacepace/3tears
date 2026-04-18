"""tool-provisioning strategy protocol.

defines the contract an agent bootstrap delegates tool registration to.
concrete implementations live in the consuming package (e.g.
``aibots_agents.runtime.tool_strategies``). the protocol isolates
environment-specific choices -- "do we host tools in-process" vs
"tools register themselves from external pods" -- from the bootstrap
body so the bootstrap stays environment-agnostic.

a strategy owns zero or one :class:`ToolServer`. the agent bootstrap
NEVER constructs a ToolServer directly; it hands the strategy a
:class:`BootstrapContext` carrying the shared NATS connection and
other runtime handles, and the strategy decides whether to start a
ToolServer, which tools to register on it, and when the tool surface
is ready for traffic.

four lifecycle methods:

- :meth:`ToolProvisioningStrategy.provision` wires up tools. called
  exactly once during the ``TOOL_STRATEGY_PROVISIONED`` bootstrap
  phase, after the NATS connection and (if configured) workspace
  runtime are live.
- :meth:`ToolProvisioningStrategy.await_ready` blocks until the tools
  are discoverable through the Registry. in-process strategies that
  register synchronously return immediately; strategies that expect
  external pods poll the Registry for the tool manifest.
- :meth:`ToolProvisioningStrategy.reload_workspace_tools` swaps the
  workspace tool bundle while the agent keeps running. called when
  a developer edits ``workspace.yaml``; strategies that host
  workspace tools atomically deregister the old bundle and register
  the new one through the ToolServer's public helpers. strategies
  that do not host workspace tools (prod-external with no workspace
  block) return immediately.
- :meth:`ToolProvisioningStrategy.teardown` is called during shutdown
  in reverse-phase order. strategies that own a ToolServer stop and
  disconnect it here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    pass


__all__ = [
    "BootstrapContext",
    "ToolProvisioningStrategy",
]


@dataclass(frozen=True)
class BootstrapContext:
    """shared bootstrap handles passed to a tool-provisioning strategy.

    frozen so strategies cannot mutate the bootstrap's state. carries
    only the minimum runtime handles a strategy needs: the NATS
    client (so strategies can reuse the bootstrap's connection rather
    than opening a competing session), the agent identity, the
    namespace prefix for subject construction, and optional runtime
    handles a strategy MAY consume (workspace runtime, registry
    client) when present.

    strategies must NOT open their own NATS connections. the bootstrap
    owns the connection lifecycle; strategies publish and subscribe
    via the supplied ``nats_client``.

    :ivar nats_client: connected NATS client owned by the bootstrap;
        strategies publish / subscribe on this connection rather than
        opening their own
    :ivar agent_id: agent UUID this bootstrap is initializing
    :ivar namespace: NATS subject-namespace prefix
    :ivar nats_url: NATS server URL the bootstrap connected with.
        strategies that stand up their own :class:`ToolServer` (which
        opens a dedicated client internally for its own subscription
        lifecycle) need the URL so they connect to the same server.
        the bootstrap-owned :attr:`nats_client` is NOT reused because
        the ToolServer owns subscription ordering its own way
    :ivar bootstrap_token: optional bootstrap token for tool-registry
        authentication; strategies that stand up a ToolServer pass it
        through so the Registry accepts the registration
    :ivar workspace_runtime: optional workspace runtime handle carrying
        the L3 proxy + collection factories used by workspace tools;
        strategies that register workspace tools consume this. None
        when the agent config declares no workspace block
    :ivar registry_client: optional handle for Registry discovery
        polling; strategies that expect tools from external pods use
        this to poll the registry for readiness. None when the
        strategy registers tools in-process
    """

    nats_client: Any
    agent_id: UUID
    namespace: str
    nats_url: str = ""
    bootstrap_token: str | None = None
    workspace_runtime: Any = None
    registry_client: Any = None


@runtime_checkable
class ToolProvisioningStrategy(Protocol):
    """contract a tool-provisioning strategy must satisfy.

    runtime-checkable so bootstrap code can assert
    ``isinstance(obj, ToolProvisioningStrategy)`` at the call site
    before wiring it in. the three methods are called in fixed order
    during an agent lifecycle:

    1. :meth:`provision` during the ``TOOL_STRATEGY_PROVISIONED``
       phase, after ``BACKEND_CONNECTED`` and
       ``WORKSPACE_RUNTIME_READY`` have completed.
    2. :meth:`await_ready` immediately after provision returns,
       blocking until tools are discoverable in the Registry. the
       bootstrap forwards failures so startup aborts before
       handler subscription.
    3. :meth:`teardown` during shutdown, after the handler stops and
       before NATS drain. called in reverse phase order with the
       other teardown steps.

    strategies MAY hold references to TearsTool instances, a
    :class:`ToolServer` they own, or nothing at all (a prod-external
    strategy with no workspace block owns no server). they MUST NOT
    hold references to the NATS client beyond the context handed to
    :meth:`provision` -- reaching for the bootstrap's client without
    going through the context breaks the "one connection per agent"
    invariant.
    """

    async def provision(self, bootstrap_context: BootstrapContext) -> None:
        """wire up tools for the agent.

        called exactly once during the ``TOOL_STRATEGY_PROVISIONED``
        bootstrap phase. strategies that own a :class:`ToolServer`
        should construct it here (reusing the supplied NATS client
        rather than opening a new connection), register their tools,
        and start the server's serve loop. strategies that expect
        tools from external pods may use this phase to record the
        expected manifest or publish a "ready to receive" signal.

        :param bootstrap_context: shared bootstrap handles including
            the live NATS connection and agent identity
        :ptype bootstrap_context: BootstrapContext
        :return: nothing
        :rtype: None
        :raises Exception: any failure propagates to the phase runner
            which aborts bootstrap (fail-fast); handlers have not yet
            subscribed so no external traffic is dropped
        """
        ...

    async def await_ready(self, timeout: float) -> None:
        """block until tools are discoverable through the Registry.

        called immediately after :meth:`provision` returns, before
        the bootstrap enters the ``GRAPH_BUILT`` phase. in-process
        strategies that register tools synchronously may short-circuit
        to an immediate return; external-pod strategies poll the
        Registry's discovery catalog until the expected tools report
        ``status=available``.

        :param timeout: maximum seconds to wait before raising
        :ptype timeout: float
        :return: nothing
        :rtype: None
        :raises TimeoutError: when the timeout elapses without the
            expected tools appearing in the Registry; the message
            names the missing tools so the operator knows which
            external pod to investigate
        """
        ...

    async def reload_workspace_tools(
        self,
        workspace_runtime: Any,
        workspace_config: Any,
    ) -> None:
        """swap the workspace tool bundle without stopping the agent.

        called when a developer edits ``workspace.yaml`` (or the
        operator applies a hot workspace-config change). strategies
        that host workspace tools deregister the old bundle and
        register the new one through the :class:`ToolServer`
        public ``deregister_tool`` / ``register_tool`` helpers,
        which each publish an updated manifest atomically.
        strategies that do not host workspace tools return
        immediately.

        :param workspace_runtime: live workspace runtime handle
            (context factory, L3 proxy, collection factories) for
            the new bundle. caller is responsible for the runtime
            lifecycle
        :ptype workspace_runtime: Any
        :param workspace_config: new workspace configuration
            declaring the tool allow-list + bind roots for the
            reloaded bundle
        :ptype workspace_config: Any
        :return: nothing
        :rtype: None
        :raises RuntimeError: when a strategy that supports
            workspace tools is not in a state that permits reload
            (e.g. :meth:`provision` has not run, or teardown has
            already stopped the ToolServer)
        """
        ...

    async def teardown(self) -> None:
        """release any resources the strategy acquired.

        called during agent shutdown in reverse phase order. strategies
        that own a :class:`ToolServer` call ``shutdown()`` on it here.
        strategies with no owned resources return immediately.

        :return: nothing
        :rtype: None
        """
        ...
