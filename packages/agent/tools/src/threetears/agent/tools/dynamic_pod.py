"""generic dynamic tool-pod base.

two kinds of pod share one lifecycle: load a set of specs, build one or
more :class:`~threetears.agent.tools.base_tool.TearsTool` per spec,
register them on a :class:`~threetears.agent.tools.server.ToolServer`,
publish the registration manifest, serve, and -- at runtime -- add /
remove a spec's tools and re-publish. :class:`DynamicToolPod` owns that
lifecycle: the :class:`ToolServer` construction, the serve-task spawn,
the per-spec ``spec_key -> tool_keys`` bookkeeping, register / deregister
/ publish, and resource teardown. a subclass supplies only the domain-
specific parts: :meth:`DynamicToolPod.load_specs` (how to load specs),
:meth:`DynamicToolPod.spec_key` (the key a spec registers under) and
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
    :param pod_id: unique pod identifier; generated (uuid7) when omitted
    :ptype pod_id: str | None
    """

    def __init__(
        self,
        *,
        nats_url: str,
        nats_client: Any,
        namespace: str,
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
        :param pod_id: unique pod identifier; generated when omitted
        :ptype pod_id: str | None
        :return: nothing
        :rtype: None
        """
        self._nats_url = nats_url
        self._nats_client = nats_client
        self._namespace = namespace
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
        default, which attaches the injected ``nats_client``. the pod
        writes no ``namespaces`` row of its own -- the hub reconciles
        those off the registration manifest.

        :return: newly constructed ToolServer
        :rtype: ToolServer
        """
        return ToolServer(
            nats_url=self._nats_url,
            nats_client=self._nats_client,
            namespace=self._namespace,
            pod_id=self._pod_id,
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
    def spec_key(self, spec: SpecT) -> str:
        """the bookkeeping key a spec registers under, known before its tools are built.

        :meth:`register_spec` needs it BEFORE the build: a spec already registered under
        the key is forgotten -- its tools dropped and its resource closed -- before the new
        one is built, so a rebuild never holds two resources at once. :meth:`build_tools`
        must return a :class:`BuiltSpec` carrying this same key.

        :param spec: the spec
        :ptype spec: SpecT
        :return: its key (the datasource name, the API source name, ...)
        :rtype: str
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
        if self._ensure_serving():
            log.info(
                "dynamic tool pod started: pod_id=%s tools_count=%d",
                self._pod_id,
                server.tools_count,
            )
        else:
            # Not an error, and NOT permanent: the pod begins serving as soon
            # as its first spec arrives. Worded explicitly because the previous
            # message ("started with no tools") read as benign while describing
            # a pod that was, at the time, unreachable for the rest of the
            # process's life.
            log.info(
                "dynamic tool pod started with no tools yet: pod_id=%s; "
                "it will begin serving when its first spec registers",
                self._pod_id,
            )

    def _ensure_serving(self) -> bool:
        """spawn the serve loop if there is something to serve and none is running.

        ``serve()`` is what subscribes to the pod's call AND probe subjects, so
        a pod that never spawns it is unreachable: the registry accepts its
        registration, fails the reachability probe with "no responders", and
        leaves every one of its tools PENDING -- while the pod's own log says
        it registered successfully.

        This used to be decided once, in :meth:`start`, against the tool count
        at that instant. A pod whose specs arrive LATER -- the Hub's dataset
        pod, which has no tools until an ``access_mode='build'`` datasource
        exists -- therefore stayed silent for the rest of the process's life.
        Adding a datasource to a running cluster is the ordinary steady-state
        operation, so the decision is re-taken whenever the pod gains tools.

        Idempotent by design: two serve loops on one subject would be a
        duplicate-delivery bug, so a live task is never replaced.

        :return: ``True`` when a serve loop is running after this call
        :rtype: bool
        """
        server = self._tool_server
        result = False
        if server is not None:
            if self._serve_task is not None and not self._serve_task.done():
                result = True
            elif server.tools_count > 0:
                self._serve_task = spawn_background(
                    server.serve(),
                    name=_SERVE_TASK_NAME,
                    logger=log,
                )
                result = True
        return result

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
        """register a spec's tools, replacing any it already had, and announce the result once.

        a spec already registered under :meth:`spec_key` is forgotten first, publishing
        nothing: its tools are unregistered, so a tool the rebuild no longer builds cannot
        stay dispatchable, and its resource is closed BEFORE the new one is built, so a
        rebuild never holds two -- a datasource driver's pool counts against the warehouse
        user's connection limit. a failed close is logged and does not abort the rebuild.
        the spec is then built and registered, the serve loop is made sure of (``serve()``
        is what subscribes the pod's call and probe subjects), and the manifest is announced
        ONCE by whichever party can do it safely:

        - a registration that changed nothing a manifest carries -- no tools before, none
          now -- publishes nothing;
        - when the serve loop has already bound its subjects
          (:attr:`~threetears.agent.tools.server.ToolServer.is_ready`) and the connection
          is up, this publishes the updated manifest;
        - otherwise the serve loop publishes it: it subscribes first and then publishes the
          manifest as it stands, these tools included. publishing here first would name the
          new endpoints before their probe subject exists. the registry's one probe would
          fail, and since it does not probe an endpoint it already holds, the loop's own
          publish would not probe again, leaving the tools pending until the next heartbeat.

        one publish, never a deregister's reduced manifest followed by the rebuilt one: for a
        pod whose only tools are this spec's, the reduced manifest is empty, and the registry
        refuses an empty manifest moments before the real one lands.

        safe to call before :meth:`start` has built the server: the guard makes it a no-op.

        :param spec: spec to build + register tools for
        :ptype spec: SpecT
        :return: nothing
        :rtype: None
        :raises ValueError: when :meth:`build_tools` returns a key other than :meth:`spec_key`'s
        """
        server = self._tool_server
        if server is None:
            return
        key = self.spec_key(spec)
        _known, had_tools = await self._forget(key)
        built = await self.build_tools(spec)
        if built.key != key:
            raise ValueError(
                f"{type(self).__name__}.build_tools returned key {built.key!r} for a spec whose spec_key is "
                f"{key!r}; the two must agree, or a rebuild forgets one registration and replaces another"
            )
        self._register_built(built)
        await self._announce(server, built, manifest_changed=bool(built.tools) or had_tools)

    async def _announce(self, server: ToolServer, built: BuiltSpec, *, manifest_changed: bool) -> None:
        """publish the manifest after a registration, or leave it to whoever can do it safely.

        :param server: the pod's tool server
        :ptype server: ToolServer
        :param built: the spec just registered
        :ptype built: BuiltSpec
        :param manifest_changed: whether the registration changed what a manifest carries
        :ptype manifest_changed: bool
        :return: nothing
        :rtype: None
        """
        serving = self._ensure_serving()
        if not manifest_changed:
            log.info(
                "dynamic tool pod spec built no tools; manifest unchanged: key=%s pod_id=%s",
                built.key,
                self._pod_id,
            )
        elif server.is_ready and server.is_connected:
            await server.publish_registration()
            log.info(
                "dynamic tool pod spec registered: key=%s pod_id=%s",
                built.key,
                self._pod_id,
            )
        else:
            log.info(
                "dynamic tool pod spec registered before its serve loop bound its subjects; the loop "
                "announces it once bound: key=%s pod_id=%s serving=%s",
                built.key,
                self._pod_id,
                serving,
            )

    @traced
    async def deregister_spec(self, key: str) -> bool:
        """unregister a spec's tools, close its resource, re-publish.

        pops the spec's tracked tool keys and resource, unregisters each
        tool family on the ToolServer by ``mcp_name``, closes the resource
        via :meth:`close_resource`, and -- when the server is connected --
        publishes the reduced manifest once. returns whether the spec was
        known so callers can distinguish a real deregister from a no-op.
        a spec being rebuilt goes through :meth:`register_spec` instead, which
        publishes once rather than the reduced manifest and then the rebuilt one.

        :param key: spec key to deregister
        :ptype key: str
        :return: true when the spec was known (tools or resource removed)
        :rtype: bool
        """
        removed, _had_tools = await self._forget(key)
        server = self._tool_server
        if removed and server is not None and server.is_connected:
            await server.publish_registration()
            log.info(
                "dynamic tool pod spec deregistered: key=%s pod_id=%s",
                key,
                self._pod_id,
            )
        return removed

    async def _forget(self, key: str) -> tuple[bool, bool]:
        """drop a spec's tools and close its resource, publishing nothing.

        a failed close is logged rather than raised: the spec is already gone from the
        bookkeeping, and aborting here would leave a rebuild with no tools at all.

        :param key: spec key to drop
        :ptype key: str
        :return: whether the spec was known, and whether it had tools
        :rtype: tuple[bool, bool]
        """
        tool_keys = self._tool_names.pop(key, None)
        resource = self._resources.pop(key, None)
        server = self._tool_server
        if tool_keys is not None and server is not None:
            for tool_key in tool_keys:
                mcp_name = tool_key.split("@", 1)[0]
                server.unregister(mcp_name)
        if resource is not None:
            try:
                await self.close_resource(resource)
            except Exception:  # prawduct:allow prawduct/broad-except -- a resource's close may raise anything; the spec is already forgotten and the rebuild must proceed
                log.exception(
                    "dynamic tool pod could not close a forgotten spec's resource; it may still hold "
                    "connections until the process ends: key=%s pod_id=%s",
                    key,
                    self._pod_id,
                )
        return tool_keys is not None or resource is not None, bool(tool_keys)

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
