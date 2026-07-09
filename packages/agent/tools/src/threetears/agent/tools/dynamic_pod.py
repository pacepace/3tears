"""generic dynamic tool-pod base.

two kinds of pod share one lifecycle: load a set of specs, build one or
more :class:`~threetears.agent.tools.base_tool.TearsTool` per spec,
register them on a :class:`~threetears.agent.tools.server.ToolServer`,
publish the registration manifest, serve, and -- at runtime -- add /
remove a spec's tools and re-publish. :class:`DynamicToolPod` owns that
lifecycle: the :class:`ToolServer` construction, the serve-task spawn,
the per-spec ``spec_key -> tool_keys`` bookkeeping, register / deregister
/ publish, and resource teardown. a subclass supplies only the domain-
specific parts: :meth:`DynamicToolPod.load_specs` (how to load specs) and
:meth:`DynamicToolPod.build_tools` (how to build a spec's tools plus an
optional closeable resource).

the base composes existing primitives -- it does NOT reimplement a serve
loop, a manifest publish, or a registry handshake. the serve loop,
manifest publish, and connection state all live on :class:`ToolServer`;
the background-task spawn is
:func:`threetears.observe.spawn_background`.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar
from uuid import uuid7

from threetears.agent.tools.base_tool import TearsTool
from threetears.agent.tools.server import ToolServer
from threetears.observe import get_logger, spawn_background, traced

__all__ = ["BuiltSpec", "DynamicToolPod"]

log = get_logger(__name__)

SpecT = TypeVar("SpecT")

_SERVE_TASK_NAME = "dynamic-tool-pod-serve"


@dataclass(frozen=True)
class BuiltSpec:
    """result of building one spec's tools.

    carries the spec's registration key, the
    :class:`~threetears.agent.tools.base_tool.TearsTool` instances built
    for it, and an optional closeable resource (e.g. a datasource driver)
    the pod tracks and closes when the spec is deregistered or the pod
    stops. ``resource`` is ``None`` when the spec owns no resource.

    :ivar key: per-spec bookkeeping key (the datasource name, the API
        capability-source name); unique within one pod
    :ivar tools: TearsTool instances registered on the ToolServer for
        this spec
    :ivar resource: optional closeable handle the pod tracks for this
        spec; closed via :meth:`DynamicToolPod.close_resource`. ``None``
        when the spec owns no resource
    """

    key: str
    tools: list[TearsTool]
    resource: Any | None = None


class DynamicToolPod(ABC, Generic[SpecT]):
    """base owning the dynamic tool-pod lifecycle over a ``ToolServer``.

    the base constructs and owns one
    :class:`~threetears.agent.tools.server.ToolServer`, spawns the serve
    loop, and tracks per-spec ``key -> tool_keys`` bookkeeping plus each
    spec's optional closeable resource. subclasses implement the two
    domain hooks -- :meth:`load_specs` (startup spec discovery) and
    :meth:`build_tools` (build one spec's tools + resource) -- and MAY
    override the non-abstract :meth:`on_started` (register deployment-wide
    singleton tools) and :meth:`close_resource` (custom teardown) hooks.

    the base never opens its own NATS connection: it attaches the
    injected ``nats_client`` to the ToolServer, mirroring the hub-side
    datasource tool pod (the enforcing bus rejects a credential-less
    self-connect).

    :param nats_url: NATS server URL threaded to the ToolServer
    :ptype nats_url: str
    :param nats_client: pre-connected, already-authenticated NATS client
        injected into the ToolServer as ``nats_client`` so its serve loop
        attaches to it rather than opening its own connection. the owner
        of this client owns its lifecycle; the ToolServer does NOT close
        it on shutdown
    :ptype nats_client: Any
    :param namespace: NATS subject-namespace prefix
    :ptype namespace: str
    :param namespace_collection: three-tier NamespaceCollection threaded
        into the ToolServer for tool-namespace materialization. ``None``
        suppresses namespace emission -- used by the hub-side datasource
        pod, where the admin path owns those rows; the API pod passes a
        real collection
    :ptype namespace_collection: Any
    :param pod_id: unique pod identifier; generated (uuid7) when omitted
    :ptype pod_id: str | None
    """

    def __init__(
        self,
        *,
        nats_url: str,
        nats_client: Any,
        namespace: str,
        namespace_collection: Any = None,
        pod_id: str | None = None,
    ) -> None:
        """initialize the dynamic tool pod.

        :param nats_url: NATS server URL threaded to the ToolServer
        :ptype nats_url: str
        :param nats_client: pre-connected NATS client injected into the
            ToolServer; its lifecycle belongs to the caller
        :ptype nats_client: Any
        :param namespace: NATS subject-namespace prefix
        :ptype namespace: str
        :param namespace_collection: NamespaceCollection for tool-namespace
            materialization, or ``None`` to suppress emission
        :ptype namespace_collection: Any
        :param pod_id: unique pod identifier; generated when omitted
        :ptype pod_id: str | None
        :return: nothing
        :rtype: None
        """
        self._nats_url = nats_url
        self._nats_client = nats_client
        self._namespace = namespace
        self._namespace_collection = namespace_collection
        self._pod_id = pod_id or str(uuid7())
        self._tool_server: ToolServer | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._resources: dict[str, Any] = {}
        self._tool_names: dict[str, list[str]] = {}

    @property
    def pod_id(self) -> str:
        """return this pod's identifier.

        :return: pod identifier string
        :rtype: str
        """
        return self._pod_id

    def build_tool_server(self) -> ToolServer:
        """construct the pod's ToolServer from the injected handles.

        the base owns ToolServer construction; this is the single seam
        subclasses / tests override to supply an alternative (a fake, a
        differently-configured server). production subclasses use the
        default, which attaches the injected ``nats_client`` and threads
        the ``namespace_collection`` through.

        :return: newly constructed ToolServer
        :rtype: ToolServer
        """
        return ToolServer(
            nats_url=self._nats_url,
            nats_client=self._nats_client,
            namespace=self._namespace,
            pod_id=self._pod_id,
            namespace_collection=self._namespace_collection,
        )

    @abstractmethod
    async def load_specs(self) -> list[SpecT]:
        """load the specs whose tools this pod serves at startup.

        called once from :meth:`start`. the subclass returns the set of
        domain specs (datasource entities, OpenAPI capability-source rows)
        to build tools for.

        :return: specs to build tools for
        :rtype: list[SpecT]
        """
        ...

    @abstractmethod
    async def build_tools(self, spec: SpecT) -> BuiltSpec:
        """build one spec's tools plus its optional closeable resource.

        called once per spec from :meth:`start` and from
        :meth:`register_spec`. the subclass builds the spec's
        :class:`TearsTool` instances and, when the spec owns a closeable
        handle (e.g. a datasource driver), returns it as
        :attr:`BuiltSpec.resource` so the base tracks and closes it.

        :param spec: spec to build tools for
        :ptype spec: SpecT
        :return: built spec (key, tools, optional resource)
        :rtype: BuiltSpec
        """
        ...

    async def on_started(self) -> None:
        """hook run after per-spec tools register, before the serve spawn.

        default no-op. subclasses override to register deployment-wide
        singleton tools (one instance for the whole pod, not per spec)
        onto ``self._tool_server`` before the serve loop is spawned, so
        those tools are counted in the ``tools_count > 0`` serve-spawn
        decision.

        :return: nothing
        :rtype: None
        """
        return None

    async def close_resource(self, resource: Any) -> None:
        """close one tracked resource.

        default: ``await resource.close()`` when ``resource`` is not
        ``None`` (a resourceless spec is a no-op). overridable: a
        subclass whose specs share one long-lived handle closed once at
        :meth:`stop` overrides this to a no-op and closes the shared
        handle itself.

        :param resource: resource handle to close, or ``None``
        :ptype resource: Any
        :return: nothing
        :rtype: None
        """
        if resource is not None:
            await resource.close()

    @traced
    async def start(self) -> None:
        """build the ToolServer, register every spec's tools, then serve.

        constructs the ToolServer via :meth:`build_tool_server`, loads
        specs via :meth:`load_specs`, builds + registers each spec's
        tools, runs :meth:`on_started`, then spawns the serve loop only
        when the server has at least one registered tool (a tool-less pod
        has nothing to serve).

        :return: nothing
        :rtype: None
        """
        server = self.build_tool_server()
        self._tool_server = server
        specs = await self.load_specs()
        for spec in specs:
            built = await self.build_tools(spec)
            self._register_built(built)
        await self.on_started()
        if server.tools_count > 0:
            self._serve_task = spawn_background(
                server.serve(),
                name=_SERVE_TASK_NAME,
                logger=log,
            )
            log.info(
                "dynamic tool pod started: pod_id=%s tools_count=%d",
                self._pod_id,
                server.tools_count,
            )
        else:
            log.info(
                "dynamic tool pod started with no tools: pod_id=%s",
                self._pod_id,
            )

    @traced
    async def stop(self) -> None:
        """shut the ToolServer, cancel the serve task, close resources.

        shuts down the ToolServer, cancels the serve task (swallowing only
        the ``CancelledError`` raised by its own ``.cancel()``), closes
        every tracked resource via :meth:`close_resource`, and clears the
        per-spec bookkeeping. a second call is a no-op.

        :return: nothing
        :rtype: None
        """
        server = self._tool_server
        if server is not None:
            await server.shutdown()
            self._tool_server = None

        if self._serve_task is not None:
            self._serve_task.cancel()
            try:
                await self._serve_task
            # NOSILENT: consuming CancelledError from our own .cancel() above; task-level outcome is logged by spawn_background done-callback
            except asyncio.CancelledError:
                pass
            self._serve_task = None

        for resource in self._resources.values():
            await self.close_resource(resource)
        self._resources.clear()
        self._tool_names.clear()

        log.info("dynamic tool pod stopped: pod_id=%s", self._pod_id)

    @traced
    async def register_spec(self, spec: SpecT) -> None:
        """build + register a spec's tools and re-publish when connected.

        builds the spec's tools via :meth:`build_tools`, registers them on
        the ToolServer, and -- only when the server is connected --
        publishes the updated manifest once (the startup serve publishes
        the full manifest, so a registration before connect must not
        publish). safe to call before :meth:`start` has built the server:
        the guard makes it a no-op.

        :param spec: spec to build + register tools for
        :ptype spec: SpecT
        :return: nothing
        :rtype: None
        """
        server = self._tool_server
        if server is None:
            return
        built = await self.build_tools(spec)
        self._register_built(built)
        if built.key in self._tool_names and server.is_connected:
            await server.publish_registration()
            log.info(
                "dynamic tool pod spec registered: key=%s pod_id=%s",
                built.key,
                self._pod_id,
            )

    @traced
    async def deregister_spec(self, key: str) -> bool:
        """unregister a spec's tools, close its resource, re-publish.

        pops the spec's tracked tool keys and resource, unregisters each
        tool family on the ToolServer by ``mcp_name``, closes the resource
        via :meth:`close_resource`, and -- when the server is connected --
        publishes the reduced manifest once. returns whether the spec was
        known so callers can distinguish a real deregister from a no-op.

        :param key: spec key to deregister
        :ptype key: str
        :return: true when the spec was known (tools or resource removed)
        :rtype: bool
        """
        tool_keys = self._tool_names.pop(key, None)
        resource = self._resources.pop(key, None)
        removed = tool_keys is not None or resource is not None

        server = self._tool_server
        if tool_keys is not None and server is not None:
            for tool_key in tool_keys:
                mcp_name = tool_key.split("@", 1)[0]
                server.unregister(mcp_name)

        if resource is not None:
            await self.close_resource(resource)

        if removed and server is not None and server.is_connected:
            await server.publish_registration()
            log.info(
                "dynamic tool pod spec deregistered: key=%s pod_id=%s",
                key,
                self._pod_id,
            )
        return removed

    def _register_built(self, built: BuiltSpec) -> None:
        """register a built spec's tools and record its bookkeeping.

        registers each tool via :meth:`ToolServer.register` (no publish --
        the caller publishes once when appropriate), records the spec's
        ``key -> tool_keys`` mapping, and tracks the spec's resource when
        non-``None``. a no-op when the ToolServer is not yet built.

        :param built: built spec to register
        :ptype built: BuiltSpec
        :return: nothing
        :rtype: None
        """
        server = self._tool_server
        if server is None:
            return
        for tool in built.tools:
            server.register(tool)
        self._tool_names[built.key] = [f"{tool.mcp_name()}@{tool.mcp_version()}" for tool in built.tools]
        if built.resource is not None:
            self._resources[built.key] = built.resource
