"""``ToolServerBootstrap`` -- shared lifecycle for tool-pod entrypoints.

every tool-pod ``main`` function -- ``threetears.agent.tools.serve``, the
admin tool pod's ``serve.py``, the agent SDK's
``runtime/tool_server_bootstrap.py`` -- needs the same scaffolding:
configure logging, instantiate ``ToolServer``, register tools, install
SIGTERM/SIGINT handlers that schedule ``server.shutdown()`` via
``spawn_background``, await ``server.serve()``, log start/stop. copies of
it drift on small details: signal-handler implementation, log message
format, exception handling around ``serve()``.

this module owns the canonical lifecycle. host applications subclass
``ToolServerBootstrap`` to add lifecycle (e.g. opening / closing a hub
client around ``serve()``); the base class provides everything that is
not host-specific.

it also owns the *terminal* half of that lifecycle. a permanent config
fault and a transient crash both used to exit ``1``, and a supervisor
cannot tell them apart, so ``restart: unless-stopped`` restarted a pod
that could never start. :class:`ToolPodConfigError` + :data:`EX_CONFIG`
give the supervisor something to branch on, and give the observability
pipeline one record instead of a traceback per restart.

``coll-task-07c`` adds the pod's THREE-TIER half to the same lifecycle:
:func:`build_tool_pod_collection_stack` gives a tool pod a working L1 and
L2 tier scoped so no other principal can reach its data and it can reach
no one else's. it lives here rather than on ``ToolServer`` because this
module owns the canonical start/stop scaffolding and already wires the
health surface, while ``ToolServer`` is the MCP request handler and owns
no cache concern.

the stack now carries a payload rather than only an empty tier: the runtime's own
:class:`~threetears.agent.tools.object_resolution_collection.ObjectResolutionCollection`
and the proxy-assertion replay anchor are built here too and handed to the server, so
every tool-pod principal -- whether or not it declares collection tables of its own --
gets a resolution cache shared with its replicas instead of a per-process dict, and a
guard that can tell a first run from a lost bucket. they are wired by the lifecycle
owner rather than by the host pod on purpose -- they back a resolver and a guard the
host never constructs either, and a store the host has to remember to build is one the
host will forget to build.

a pod running INSIDE an agent process gets none of this. it rides the agent's
connection, authenticated as the agent, with an ``{agent_id}.{instance}`` pod id from
which no tool-pod key scope can be derived; declaring collection tables on such a pod
is refused as a :class:`ToolPodConfigError`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import MetaData
from threetears.core.backends import BrokerGenerationSource
from threetears.core.collections import bind_collections_bucket
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.coordination.replay_anchor import CollectionReplayAnchor
from threetears.epoch import EpochGenerationReader
from threetears.nats import Principal, kv_key_scope_for
from threetears.observe import (
    HealthCheck,
    HealthServer,
    HealthTier,
    configure_logging,
    get_logger,
    spawn_background,
)

from threetears.agent.tools.config import get_owner_poll_interval, get_shutdown_timeout
from threetears.agent.tools.l1_cache import TOOL_POD_L1_DB_NAME, create_tool_pod_l1_backend
from threetears.agent.tools.object_resolution_collection import ObjectResolutionCollection

if TYPE_CHECKING:
    from threetears.agent.tools.server import ToolServer
    from threetears.nats import NatsClient

__all__ = [
    "EX_CONFIG",
    "EX_SOFTWARE",
    "OWNER_PID_ENV",
    "ToolPodConfigError",
    "ToolPodShutdownError",
    "ToolServerBootstrap",
    "build_tool_pod_collection_stack",
    "resolve_owner_pid",
]

log = get_logger(__name__)

#: exit status for a permanent, operator-fixable configuration fault: ``EX_CONFIG``
#: from BSD ``sysexits.h``.
#:
#: a supervisor can branch on a STATUS and on nothing else. it never reads the
#: message, so a config fault that exits ``1`` is, to docker's
#: ``restart: unless-stopped`` or a k8s ``restartPolicy``, the same event as a
#: transient crash, and gets the same answer: restart. bluelabsio/14-eng-ai-bot#235
#: is what that costs -- a tool pod started with a renamed identity env var
#: restarted 9,580 times over 8 days, never once reaching ``running``, writing
#: 278,110 byte-identical traceback lines. nothing alerted: a healthcheck cannot
#: fire on a container that never starts, and nothing declared ``depends_on`` it,
#: so the only signal was a ``RestartCount`` no one reads.
#:
#: 78 is the conventional "the configuration is wrong, do not try again" status.
#: being conventional is the point: systemd's ``RestartPreventExitStatus``, a k8s
#: exit-code alert, and an operator reading ``docker inspect`` all classify it with
#: no agreement needed between them and this module.
EX_CONFIG = 78

#: exit status for a pod whose shutdown failed or overran its bound: ``EX_SOFTWARE`` from BSD
#: ``sysexits.h``. Distinct from :data:`EX_CONFIG` on purpose -- a supervisor SHOULD restart this
#: one -- and from ``0``, because the process left without the drain it owed, and an exit that
#: reads as clean would hide that.
EX_SOFTWARE = 70

#: the environment variable naming the process that OWNS this tool pod. Opt-in: when it is set, the
#: pod shuts itself down once that process no longer exists. Written by whatever spawns tool pods as
#: its own children (the aibots SDK's in-process launcher does), because a pod outliving its owner is
#: a pod nobody will ever stop. See :func:`resolve_owner_pid` for what is accepted.
OWNER_PID_ENV = "THREETEARS_TOOL_POD_OWNER_PID"


class ToolPodConfigError(ValueError):
    """a tool pod's configuration is permanently wrong; restarting cannot fix it.

    raise this from a ``build_server`` (or any startup-config validation) when a
    required environment variable is missing, renamed, or malformed.
    :meth:`ToolServerBootstrap.run` catches exactly this type, logs one structured
    ERROR naming ``variable``, and exits :data:`EX_CONFIG`.

    it subclasses :class:`ValueError` deliberately. the config-validation sites it
    replaces raised bare ``ValueError``, so every existing ``except ValueError``
    caller and test keeps working unchanged. the reason it is a distinct type
    rather than the bare ``ValueError`` the incident report proposed catching: a
    ``ValueError`` raised from deep inside a tool's own business logic during
    startup is a bug, not a config fault, and catching the base type would make it
    terminal -- turning a pod that a restart might have recovered into one that
    stays down. the same subclass-for-compatibility shape is already used by
    :class:`threetears.core.security.secret_refs.SecretResolutionError`.

    :param message: operator-facing description of what is wrong and how to fix it
    :ptype message: str
    :param variable: name of the environment variable, or of the bootstrap
        parameter when no variable is involved, at fault. required, because an
        operator reading one ERROR line needs the thing to go change, and a
        message that only says "config is invalid" sends them to the source
    :ptype variable: str
    """

    def __init__(self, message: str, *, variable: str) -> None:
        """record the offending variable alongside the message.

        :param message: operator-facing description of the fault
        :ptype message: str
        :param variable: environment variable (or bootstrap parameter) name at fault
        :ptype variable: str
        :return: None
        :rtype: None
        """
        super().__init__(message)
        self.variable = variable


class ToolPodShutdownError(RuntimeError):
    """a tool pod's shutdown failed or overran its bound; the pod left ``serve`` anyway.

    raised by :meth:`ToolServerBootstrap.run_async` once teardown is done, chaining what failed --
    the exception the server's shutdown raised, or the :class:`TimeoutError` of the bound. A caller
    that drives ``run_async`` itself decides what that means; :meth:`ToolServerBootstrap.run` exits
    :data:`EX_SOFTWARE`. The failure was already logged once at ERROR, where it happened.

    :param message: what failed, for a person
    :ptype message: str
    """


def resolve_owner_pid() -> int | None:
    """read and validate :data:`OWNER_PID_ENV`.

    Unset means no owner is watched. Set, it must name a process that can actually be watched: a
    decimal integer greater than 1 (``0`` and negatives name process GROUPS to ``kill``, and pid 1
    is init, which never exits and which an orphan is reparented to) and not this process itself,
    which exists for as long as anything could check. A blank value is refused rather than read as
    unset: a spawner that meant to set it and rendered nothing should hear about it at startup.

    :return: the owner's pid, or ``None`` when the variable is unset
    :rtype: int | None
    :raises ToolPodConfigError: when the variable is set to anything that cannot name an owner
    """
    raw = os.environ.get(OWNER_PID_ENV)
    result: int | None = None
    if raw is not None:
        value = raw.strip()
        if not value.isdecimal():
            raise ToolPodConfigError(
                f"{OWNER_PID_ENV}={raw!r} is not a process id; set it to the owner's pid (an integer "
                f"greater than 1), or unset it to run without an owner",
                variable=OWNER_PID_ENV,
            )
        pid = int(value)
        if pid <= 1:
            raise ToolPodConfigError(
                f"{OWNER_PID_ENV}={pid} cannot name an owner process: 0 names a process group and 1 is "
                f"init, which never exits",
                variable=OWNER_PID_ENV,
            )
        if pid == os.getpid():
            raise ToolPodConfigError(
                f"{OWNER_PID_ENV}={pid} is this tool pod's own pid; a pod cannot outlive itself, so "
                f"the watch would never fire. set it to the pid of the process that spawned the pod",
                variable=OWNER_PID_ENV,
            )
        result = pid
    return result


def _process_exists(pid: int) -> bool:
    """whether a process with ``pid`` exists, without signalling it.

    ``kill(pid, 0)`` performs the existence and permission checks and delivers nothing, on macOS
    and Linux alike. ``PermissionError`` means the process exists and belongs to someone else.
    An exited child that its parent has not yet reaped still exists; the owner watch covers that
    case separately, through the parent pid (see :meth:`ToolServerBootstrap.watch_owner`).

    :param pid: the process id
    :ptype pid: int
    :return: ``True`` while the process exists
    :rtype: bool
    """
    result = True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        result = False
    except PermissionError:
        result = True
    return result


def _is_tool_pod_id(pod_id: str) -> bool:
    """whether ``pod_id`` names a tool-pod principal rather than an in-process pod.

    A tool pod's id is its ``tool_pods.id``, a single uuid token, and the auth callout pins
    it as ``claims.sub``. An in-process pod's id is
    :meth:`~threetears.nats.Subjects.agent_inprocess_pod_id` -- ``{agent_id}.{instance}`` --
    and that pod connects as its agent. The test is the one
    :func:`~threetears.nats.kv_key_scope_for` applies to a tool-pod scope, so a pod this
    answers ``True`` for is exactly a pod that scope can be derived for.

    :param pod_id: the id the tool server was constructed with
    :ptype pod_id: str
    :return: ``True`` when ``pod_id`` is a uuid
    :rtype: bool
    """
    result = True
    try:
        UUID(pod_id)
    except ValueError:
        result = False
    return result


async def build_tool_pod_collection_stack(
    *,
    nats_client: "NatsClient",
    pod_id: str,
    l1_metadata: "MetaData",
    l1_db_name: str = TOOL_POD_L1_DB_NAME,
) -> CollectionRegistry:
    """build one tool pod's L1 + L2 collection tiers, scoped to its own ``tool_pods.id``.

    ONE function, so a pod gets its tiers without constructing a
    :class:`~threetears.core.cache.sqlite.SQLiteBackend` itself. That matters more here than
    elsewhere: the cache-primitive allowlist that sanctions L1 construction sites is PER REPOSITORY
    and cannot see a partner-operated tool pod at all, so the question is made moot rather than
    exempted.

    Four decisions are baked in, and each is the thing that makes the stack safe rather than merely
    present:

    - **the key scope is ``tool_pods.id``** -- the pod's registry primary key, which is already its
      authenticated ``claims.sub``. It is configured once per DEPLOYMENT, so every replica resolves
      to one scope and L2 stays a cross-pod cache, while two different pods can never collide. A
      namespace-derived scope was designed and rejected: ``tool_namespace_id`` mints one row PER
      TOOL, is a pure function of manifest values the pod itself sends, is deliberately
      collision-inducing across pods, and no such row exists at connect time. A non-uuid pod id is
      REFUSED, because a boundary derived from a display name is not provably collision-free.
    - **the shared bucket is opened EAGERLY, first.** ``BaseCollection._ensure_kv`` resolves it on
      the first read, so a bucket carrying a configuration this process refuses would otherwise
      raise inside a request path under load. See
      :func:`threetears.core.collections.bucket.bind_collections_bucket`.
    - **the bucket is BOUND, never declared.** A tool pod's minted grant carries
      ``$JS.API.STREAM.INFO`` on the collections stream and no ``CREATE``: on a SHARED stream
      ``STREAM.UPDATE`` is a read-all primitive, so ``declare`` belongs to the hub alone. A refused
      create is never answered, so declaring would cost a JetStream deadline at every startup and
      then bind anyway.
    - **the invalidation listener is started here.** Without it the pod's L1 keeps serving a value a
      peer replica has already replaced. The caller stops it -- :class:`ToolServerBootstrap` does
      that in its own teardown.

    :param nats_client: the pod's connected canonical NATS client, used as the L2 tier and as the
        invalidation listener's transport
    :ptype nats_client: NatsClient
    :param pod_id: the pod's ``tool_pods.id``, as a uuid string
    :ptype pod_id: str
    :param l1_metadata: the pod's declared Collection tables, mirrored into its L1 database
    :ptype l1_metadata: MetaData
    :param l1_db_name: name of this process's in-memory L1 database. The default is right for
        every deployment -- replicas are separate PROCESSES, and within one process every
        collection must share one L1 tier, which is why the name is fixed rather than random.
        Override it only where one process has to stand in for two replicas, which is a
        cross-pod test: left at the default, both "replicas" resolve to one L1 database and a
        read that was supposed to cross L2 is answered locally instead. The parameter one layer
        down (:func:`~threetears.agent.tools.l1_cache.create_tool_pod_l1_backend`) exists for
        the same reason; this passes it through rather than hiding it
    :ptype l1_db_name: str
    :return: the configured registry, with its invalidation listener running
    :rtype: CollectionRegistry
    :raises ValueError: if ``pod_id`` is not a uuid, so no collision-free scope can be derived
    :raises threetears.nats.errors.KvError: if the shared bucket cannot be bound within the attempt
        budget -- nothing has declared it, or this principal is not granted it
    :raises threetears.nats.errors.KvConfigMismatch: if the live bucket carries a configuration this
        process refuses
    """
    # scope FIRST, because it is the one failure that needs no network: a pod id that cannot
    # produce a collision-free scope is refused before a single JetStream request is issued.
    scope = kv_key_scope_for(Principal.TOOL_POD, pod_id=pod_id)
    # then the bucket, BEFORE the registry is configured: a refused or mismatched bucket must take
    # the process down at wiring, with nothing half-built behind it.
    # component=: several processes bind this same bucket, and the interesting failure is
    # an ORDERING one -- which reached it before the declaring identity. An unnamed bind log
    # cannot answer that.
    await bind_collections_bucket(nats_client, component="tool-pod")
    registry = CollectionRegistry()
    registry.configure(
        l1_backend=create_tool_pod_l1_backend(l1_metadata, db_name=l1_db_name),
        l2_client=nats_client,
        kv_key_scope=scope,
        l2_create_if_missing=False,
    )
    # epoch-task-06: the pod may not write the epoch bucket, so the hub's L3 broker advances each
    # switched-on table's write generation after committing the pod's write and names the token in
    # its reply; this source hands it to the collection. Its reader binds the epoch bucket the pod
    # is granted read of, so a collection caching absences can stamp them with a generation, and a
    # cache the pod derives from a table can follow it (``threetears.epoch.follow_generation_key``,
    # or ``threetears.agent.acl.generation_follow.AccessTableFollower`` for the access tables).
    registry.set_generation_source(BrokerGenerationSource(EpochGenerationReader(nats_client)))
    await registry.start_invalidation_listener(nats_client)
    log.info(
        "tool pod collection stack wired",
        extra={"extra_data": {"pod_id": pod_id, "kv_key_scope": scope, "tables": len(l1_metadata.tables)}},
    )
    return registry


class ToolServerBootstrap:
    """canonical tool-pod lifecycle: logging, signals, serve loop.

    subclasses customize the build-and-serve pipeline by overriding
    ``build_server``, ``register_tools``, and ``run_serve``, and add
    their own health checks by passing ``extra_health_checks`` rather
    than by reaching into how the health server is built. the
    base ``run`` is the one-line entrypoint every tool-pod ``main()``
    calls.

    typical use::

        class MyBootstrap(ToolServerBootstrap):
            def __init__(self, ...): ...
            async def build_server(self) -> ToolServer: ...
            async def register_tools(self, server: ToolServer) -> None: ...

        def main() -> None:
            MyBootstrap(...).run()

    :param service_name: short identifier used in log messages and
        signal-handler task names
    :ptype service_name: str
    :param log_level: stdlib logging level name passed to
        ``configure_logging``
    :ptype log_level: str
    """

    def __init__(
        self,
        service_name: str,
        *,
        log_level: str = "INFO",
        health_port: int | None = None,
        collection_tables: "MetaData | None" = None,
        version: str | None = None,
        extra_health_checks: Sequence[HealthCheck] = (),
    ) -> None:
        """initialize bootstrap with service identity and log level.

        :param service_name: short identifier for log messages
        :ptype service_name: str
        :param log_level: log level name (e.g. ``"INFO"``, ``"DEBUG"``)
        :ptype log_level: str
        :param collection_tables: the pod's OWN Collection tables, mirrored into its tiers
            alongside the runtime's. the stack (:func:`build_tool_pod_collection_stack`) is built
            for every tool-pod principal once its NATS connection exists and torn down with it,
            because the runtime holds collections on every such pod's behalf -- the
            proxy-assertion replay anchor among them. an in-process pod (one whose pod id is not
            a uuid) gets no stack, and passing tables for one is a :class:`ToolPodConfigError`.
            ``None`` -> the pod declares none of its own
        :ptype collection_tables: MetaData | None
        :param health_port: port the readiness HealthServer binds to;
            defaults to THREETEARS_TOOL_SERVER_HEALTH_PORT env var,
            falling back to 8000. each container in the platform's
            docker stack owns its own port namespace so 8000 is fine
            there; honcho-driven dev runs every Procfile entry in the
            host namespace and the hub uvicorn already binds host:8000,
            so honcho callers MUST set THREETEARS_TOOL_SERVER_HEALTH_PORT
            to a distinct port. bind failures are swallowed at startup
            (the tool pod's primary job is NATS, not health probing),
            so a collision degrades to "no /healthz" rather than aborting.
        :ptype health_port: int | None
        :param version: release version of the pod's own distribution, echoed on the health
            server's JSON body and the pod's ``starting`` log line. ``None`` (a subclass that
            passes none) leaves both carrying a null version
        :ptype version: str | None
        :param extra_health_checks: the pod's OWN checks, evaluated by the health server beside
            the bootstrap's (``nats`` LIVE, ``tools_registered`` and ``jwks_warmed`` READY). each
            carries its own :class:`~threetears.observe.HealthTier`, so a pod states the blast
            radius of its own failure: READY for state a restart cannot fix (storage still being
            wired, a cache warming), LIVE only when a restart is the remedy. a probe reads the
            pod's state when the health server asks, not when the check is built, so a pod may
            construct its checks before the state they read exists. names must be unique and
            must not repeat a bootstrap check's, because two components with one name make the
            ``?format=json`` body ambiguous; a collision is a :class:`ToolPodConfigError` raised
            before the pod serves. empty -> the bootstrap's checks alone
        :ptype extra_health_checks: Sequence[HealthCheck]
        """
        self._service_name = service_name
        self._version = version
        self._extra_health_checks: tuple[HealthCheck, ...] = tuple(extra_health_checks)
        self._log_level = log_level
        self._collection_tables = collection_tables
        self._collection_registry: CollectionRegistry | None = None
        self._object_resolutions: ObjectResolutionCollection | None = None
        # the serve loop, as a task, so a shutdown that fails can still end it. set by run_async.
        self._serve_task: asyncio.Task[None] | None = None
        # one shutdown per process: a second signal, or the owner going as well, joins the first.
        self._shutdown_requested = False
        # what ended this pod's shutdown badly, if anything; run_async raises it once torn down.
        self._shutdown_failure: BaseException | None = None
        if health_port is not None:
            self._health_port = health_port
        else:
            env_port = os.environ.get("THREETEARS_TOOL_SERVER_HEALTH_PORT")
            self._health_port = int(env_port) if env_port else 8000

    @property
    def version(self) -> str | None:
        """return the release version this pod reports on its health body and boot line.

        :return: the version passed at construction, or ``None`` when none was given
        :rtype: str | None
        """
        return self._version

    @property
    def collection_registry(self) -> CollectionRegistry | None:
        """the pod's three-tier registry, once its NATS connection has been established.

        ``None`` until then, and ``None`` forever for an in-process pod, which holds no tool-pod
        stack. Tools read it lazily rather than at construction time, because the connection
        the stack rides on does not exist until :meth:`ToolServer.serve` opens it -- which is
        still strictly before the pod subscribes its call subject, so no tool call can arrive
        at a tool-pod principal while this is unset.

        :return: the configured registry, or ``None`` before the connection exists or for an
            in-process pod
        :rtype: CollectionRegistry | None
        """
        return self._collection_registry

    @property
    def object_resolutions(self) -> ObjectResolutionCollection | None:
        """the runtime's two-tier object-resolution store, once NATS is up.

        ``None`` until the connection exists, and ``None`` forever for an in-process pod.
        Exposed for the same reason :attr:`collection_registry` is: a host that wants to
        read or drop a mapping of its own has one object to reach for rather than a second
        store of its own.

        :return: the collection, or ``None`` before the connection exists or for an
            in-process pod
        :rtype: ObjectResolutionCollection | None
        """
        return self._object_resolutions

    def install_collection_stack(self, server: "ToolServer") -> None:
        """arrange for the pod's collection stack to be built the moment NATS is up.

        Built for every tool-pod principal: one that declared no ``collection_tables`` still gets
        the runtime's own collections, the replay anchor among them. Registered as a CONNECTED
        callback rather than built inline, because :meth:`ToolServer.serve` is what opens the
        connection -- and the callback runs before the pod subscribes its call subject and
        publishes its registration manifest, so the pod is never discoverable with its own
        collections unwired.

        The pod id is taken from the ``ToolServer``, which is the same value the pod presents at
        connect and the auth callout pins as ``claims.sub``. A second piece of bootstrap
        configuration naming it could drift from the authenticated identity, and a key scope that
        drifts from the grant is a dead cache that logs nothing.

        An in-process pod -- one whose pod id is not a uuid, because it runs inside an agent
        process on the agent's connection -- is not a tool-pod principal. It gets no stack, as
        before this stack existed; declaring tables on one is refused here, before the pod
        serves, rather than failing inside the connected callback on every restart.

        :param server: the tool server whose connection the stack rides on
        :ptype server: ToolServer
        :return: nothing
        :rtype: None
        :raises ToolPodConfigError: if an in-process pod declares ``collection_tables``
        """
        # ONLY a tool-pod principal gets the stack. A pod running inside an agent process rides
        # the agent's injected connection, authenticated as the AGENT, with a pod id of
        # ``{agent_id}.{instance}``: no tool-pod key scope derives from that id, and the grant it
        # connected with carries the agent's scope rather than ``tool_pod-<hex>``. Building the
        # stack for it raised inside the connected callback on every start.
        if not _is_tool_pod_id(server.pod_id):
            if self._collection_tables is not None:
                raise ToolPodConfigError(
                    f"{self._service_name} declares collection tables but runs as in-process pod "
                    f"{server.pod_id!r} on an agent's connection. Collection tables are scoped to a "
                    f"tool pod's own identity (a tool_pods.id uuid); run this pod as its own tool "
                    f"pod, or declare no tables.",
                    variable="collection_tables",
                )
            log.info(
                "in-process pod rides its agent's connection; no tool-pod collection stack",
                extra={"extra_data": {"service": self._service_name, "pod_id": server.pod_id}},
            )
            return
        # EVERY tool-pod principal gets the stack, declared tables or not. The runtime holds
        # collections of its own on every pod's behalf -- the object-resolution cache and the
        # proxy-assertion replay anchor -- and the anchor is not optional: without it the guard
        # cannot tell a bucket it never had from one it lost, so every cold start refused the
        # pod's first proxied call. Gating the stack on host tables left every pod that declared
        # none with exactly that.
        tables = self._collection_tables if self._collection_tables is not None else MetaData()

        async def _on_connected(nats_client: "NatsClient") -> None:
            """build the pod's tiers on the freshly-established connection.

            :param nats_client: the pod's connected canonical NATS client
            :ptype nats_client: NatsClient
            :return: nothing
            :rtype: None
            """
            registry = await build_tool_pod_collection_stack(
                nats_client=nats_client,
                pod_id=server.pod_id,
                l1_metadata=tables,
            )
            self._collection_registry = registry
            # the runtime's OWN collection, wired here rather than by the host pod: it
            # backs a resolver the pod never constructs either, and a store a host had
            # to remember to build is one a host will forget to build.
            self._object_resolutions = ObjectResolutionCollection(
                registry,
                DefaultCoreConfig(),
                nats_client,
            )
            server.attach_object_resolution_cache(self._object_resolutions)
            # THE SAME ARGUMENT, for the proxy-assertion guard. Without an anchor that guard
            # cannot tell a bucket it never had from one it lost, so it applies its
            # creation-time watermark to both -- and `proxy_assertion_nonces` is memory-backed,
            # so it dies with the broker. Every cold start therefore refused its first proxied
            # call, naming `proxy assertion nonce replay`, which is the one thing that had not
            # happened.
            #
            # WIRED HERE BECAUSE NOWHERE ELSE CAN. An anchor reads through this registry, which
            # needs a connected NATS client -- and the thing that connects is the ToolServer the
            # pod has already finished constructing by then. So a pod cannot pass one at
            # construction without deferring the lookup by hand, and every pod that did not
            # think to got a silent refusal window after every restart. This callback runs
            # BEFORE `serve` builds the guard, so the ordering is guaranteed rather than hoped
            # for, and a pod that supplied its own anchor keeps it.
            server.attach_assertion_replay_anchor(CollectionReplayAnchor(registry))

        server.add_connected_callback(_on_connected)

    def run(self) -> None:
        """entrypoint: configure logging then drive the async serve loop.

        this is the only method tool-pod ``main()`` functions need to
        call. subclasses override async hooks to plug in host-specific
        lifecycle (build dependency, register tools, await serve).

        a :class:`ToolPodConfigError` terminates here: one structured ERROR
        record, then :data:`EX_CONFIG`. every other failure propagates
        untouched, exits ``1``, and stays retryable -- a NATS server that is
        not up yet is exactly the case a supervisor restart exists for.

        the catch lives here and NOT in :meth:`run_async` because this is the
        method that owns the process exit status. a caller that drives
        ``run_async`` on its own loop owns its own failure policy and must be
        free to handle the error rather than have the library exit under it.

        a :class:`ToolPodShutdownError` terminates here too, with
        :data:`EX_SOFTWARE`: the pod left ``serve`` without the shutdown it
        owed, which is neither a clean exit nor a configuration fault.

        :return: None
        :rtype: None
        :raises SystemExit: with :data:`EX_CONFIG` when startup config is
            permanently wrong, or :data:`EX_SOFTWARE` when shutdown failed
        """
        configure_logging(level=self._log_level)
        try:
            asyncio.run(self.run_async())
        except ToolPodConfigError as exc:
            # ONE record, through the repo logger, so this reaches the observability
            # pipeline as a queryable event rather than as container stderr nobody
            # tails. `from None` drops the traceback: 278,110 lines of it is what the
            # incident actually produced, and none of them said anything this one
            # record does not.
            log.error(
                f"{self._service_name} configuration is invalid; exiting without retry",
                extra={
                    "extra_data": {
                        "service": self._service_name,
                        "variable": exc.variable,
                        "error": str(exc),
                        "exit_code": EX_CONFIG,
                    }
                },
            )
            raise SystemExit(EX_CONFIG) from None
        except ToolPodShutdownError:
            # already logged once at ERROR where it happened, naming the cause. The status is the
            # one thing left to say, and a traceback here would only repeat the record.
            raise SystemExit(EX_SOFTWARE) from None

    async def run_async(self) -> None:
        """async driver: build server, register tools, install signals, serve.

        public async entrypoint suitable for tests and callers that
        already own the event loop. ``run`` is the sync convenience
        wrapper that drives this through ``asyncio.run``.

        also starts a canonical :class:`HealthServer` on port 8000
        so docker / k8s liveness probes can verify the pod is alive
        + connected to NATS without a custom per-pod health
        endpoint.

        **serving ends even when shutdown fails.** The serve loop runs as a task that
        :meth:`shutdown_server` cancels when the server's own shutdown raises or overruns
        :func:`~threetears.agent.tools.config.get_shutdown_timeout`. It used to be awaited
        directly, and ``ToolServer.shutdown`` raising before it released ``serve`` left two tool
        pods alive for two days after SIGTERM. The teardown after serving gets the same bound.

        when :data:`OWNER_PID_ENV` is set the pod also watches that process
        (:meth:`watch_owner`) and shuts down through the same path once it is gone.

        :return: None
        :rtype: None
        :raises ToolPodConfigError: when :data:`OWNER_PID_ENV` is set to something that names no owner
        :raises ToolPodShutdownError: once torn down, when the shutdown failed or overran its bound
        """
        owner_pid = resolve_owner_pid()
        server = await self.build_server()
        await self.register_tools(server)
        self.install_collection_stack(server)

        health_server = await self._start_health_server(server)

        log.info(
            f"{self._service_name} starting",
            extra={
                "extra_data": {
                    "service": self._service_name,
                    "version": self._version,
                    "tools_count": server.tools_count,
                    "owner_pid": owner_pid,
                }
            },
        )
        serve_task = asyncio.create_task(self.run_serve(server), name=f"{self._service_name}-serve")
        self._serve_task = serve_task
        # AFTER the serve task exists and before it first runs (nothing has awaited since it was
        # created), so no signal can arrive to a shutdown path that has no serve loop to end.
        self.install_signal_handlers(server)
        owner_watch: asyncio.Task[None] | None = None
        if owner_pid is not None:
            owner_watch = asyncio.create_task(
                self.watch_owner(server, owner_pid), name=f"{self._service_name}-owner-watch"
            )
        try:
            await self._await_serve(serve_task)
        finally:
            if owner_watch is not None:
                owner_watch.cancel()
                # NOSILENT: the only thing suppressed is the cancellation sent on the line above.
                with contextlib.suppress(asyncio.CancelledError):
                    await owner_watch
            await self._bounded_teardown(health_server)
            log.info(
                f"{self._service_name} stopped",
                extra={"extra_data": {"service": self._service_name}},
            )
        if self._shutdown_failure is not None:
            raise ToolPodShutdownError(
                f"{self._service_name} left serve without completing its shutdown"
            ) from self._shutdown_failure

    async def _await_serve(self, serve_task: "asyncio.Task[None]") -> None:
        """await the serve loop, treating ITS cancellation by a failed shutdown as the loop ending.

        :param serve_task: the running serve loop
        :ptype serve_task: asyncio.Task[None]
        :return: nothing
        :rtype: None
        :raises asyncio.CancelledError: when this coroutine itself is cancelled, or the serve loop
            was cancelled by anything other than a failed shutdown
        """
        try:
            await serve_task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            forced = self._shutdown_failure is not None and serve_task.cancelled()
            if not forced or (current is not None and current.cancelling() > 0):
                raise

    async def _bounded_teardown(self, health_server: "HealthServer | None") -> None:
        """release what serving held, within the shutdown bound.

        :param health_server: the health listener, or ``None`` when it never bound
        :ptype health_server: HealthServer | None
        :return: nothing
        :rtype: None
        """
        timeout = get_shutdown_timeout()
        try:
            await asyncio.wait_for(self._teardown(health_server), timeout=timeout)
        except TimeoutError as exc:
            log.error(
                f"{self._service_name} teardown overran its bound; exiting without it",
                extra={
                    "extra_data": {
                        "service": self._service_name,
                        "timeout_seconds": timeout,
                        "error_type": type(exc).__name__,
                    }
                },
            )
            if self._shutdown_failure is None:
                self._shutdown_failure = exc

    async def _teardown(self, health_server: "HealthServer | None") -> None:
        """stop the collection stack and the health listener.

        :param health_server: the health listener, or ``None`` when it never bound
        :ptype health_server: HealthServer | None
        :return: nothing
        :rtype: None
        """
        registry = self._collection_registry
        if registry is not None:
            # ``coll-task-01``'s teardown half. It does not raise on a draining connection
            # (``NatsClient.unsubscribe`` absorbs the transport failures a shutdown produces),
            # so it runs FIRST in this block: a listener left bound holds a subscription on a
            # client the process no longer owns.
            await registry.stop_invalidation_listener()
            # and whatever a collection itself started: a write-behind coordination
            # collection's flusher owes one last flush before the loop closes.
            await registry.close_collections()
        if health_server is not None:
            try:
                await health_server.stop()
            except Exception as exc:
                log.warning(
                    "health server stop failed",
                    extra={"extra_data": {"error": str(exc)}},
                )

    async def shutdown_server(self, server: "ToolServer", *, reason: str) -> None:
        """shut ``server`` down, and make sure serving ends whether or not that succeeds.

        The one shutdown path: both signal handlers and the owner watch call it. Only the first
        call acts; a later one (SIGTERM then SIGINT, or the owner going during a shutdown) is
        logged and returns.

        ``server.shutdown()`` is given :func:`~threetears.agent.tools.config.get_shutdown_timeout`
        seconds. If it raises or overruns, the failure is logged ONCE at ERROR with its cause and
        recorded for :meth:`run_async` to raise, and the serve loop is cancelled so the process
        leaves. If it succeeds but the serve loop does not return within the same bound, that is
        the same failure.

        :param server: the tool server to shut down
        :ptype server: ToolServer
        :param reason: what asked for the shutdown (``"sigterm"``, ``"sigint"``, ``"owner-gone"``)
        :ptype reason: str
        :return: nothing
        :rtype: None
        """
        if self._shutdown_requested:
            log.info(
                f"{self._service_name} shutdown already in progress; ignoring a second request",
                extra={"extra_data": {"service": self._service_name, "reason": reason}},
            )
            return
        self._shutdown_requested = True
        timeout = get_shutdown_timeout()
        failure: BaseException | None = None
        try:
            await asyncio.wait_for(server.shutdown(), timeout=timeout)
        except Exception as exc:  # noqa: BLE001 -- recorded and raised by run_async; logged once here
            failure = exc
        serve_task = self._serve_task
        if failure is None and serve_task is not None:
            done, _pending = await asyncio.wait({serve_task}, timeout=timeout)
            if not done:
                failure = TimeoutError(f"serve did not return within {timeout}s of a completed shutdown")
        if failure is not None:
            self._shutdown_failure = failure
            log.error(
                f"{self._service_name} shutdown failed; leaving serve and exiting non-zero",
                extra={
                    "extra_data": {
                        "service": self._service_name,
                        "reason": reason,
                        "timeout_seconds": timeout,
                        "error_type": type(failure).__name__,
                        "error": str(failure),
                    }
                },
            )
            if serve_task is not None and not serve_task.done():
                serve_task.cancel()

    async def watch_owner(self, server: "ToolServer", owner_pid: int) -> None:
        """shut the pod down once the process that owns it no longer exists.

        Polled every :func:`~threetears.agent.tools.config.get_owner_poll_interval` seconds with
        ``kill(pid, 0)``, which is portable across macOS and Linux and signals nothing. When the
        owner is this pod's PARENT, a change of parent pid counts as gone too: a parent that exited
        without being reaped still answers ``kill(pid, 0)``, but its children are reparented the
        moment it exits.

        :param server: the tool server to shut down
        :ptype server: ToolServer
        :param owner_pid: the owner's pid, from :func:`resolve_owner_pid`
        :ptype owner_pid: int
        :return: nothing
        :rtype: None
        """
        interval = get_owner_poll_interval()
        owner_is_parent = os.getppid() == owner_pid
        while _process_exists(owner_pid) and (not owner_is_parent or os.getppid() == owner_pid):
            await asyncio.sleep(interval)
        log.warning(
            "owner process %d of %s is gone; shutting the tool pod down",
            owner_pid,
            self._service_name,
            extra={"extra_data": {"service": self._service_name, "owner_pid": owner_pid}},
        )
        await self.shutdown_server(server, reason="owner-gone")

    async def _start_health_server(self, server: "ToolServer") -> "HealthServer | None":
        """start the canonical /healthz listener on port 8000.

        port 8000 matches the inherited 3tears-hub Dockerfile
        HEALTHCHECK so docker / k8s liveness probes work for every
        consumer of that base image without per-service compose
        overrides. failures here are logged + swallowed -- the tool
        pod's primary job is NATS request/reply, not health probing,
        so a port-bind collision (multiple pods on the same docker
        network competing for 8000) must not abort startup.

        the pod's own ``extra_health_checks`` are appended after the bootstrap's, in the order
        given, so a readiness probe that short-circuits names a bootstrap failure (a dead NATS
        data plane) before a pod-specific one it may be causing.

        :param server: tool server whose state the checks read from
        :ptype server: ToolServer
        :return: started :class:`HealthServer`, or ``None`` if the
            listener failed to bind
        :rtype: HealthServer | None
        :raises ToolPodConfigError: if a contributed check repeats a name already in the list
        """
        checks = [
            HealthCheck(
                # key liveness on REAL NATS health, not object-existence: a
                # terminal close (user-JWT expiry) or a persistent auth /
                # overflow wedge trips is_healthy so k8s recycles the pod. the
                # old is_connected probe reported healthy forever with a dead
                # connection (the silent-zombie bug). the in-process
                # heartbeat-loop supervisor is the no-k8s net (host / docker).
                name="nats",
                probe=lambda: server.is_healthy,
                tier=HealthTier.LIVE,
            ),
            # readiness, NOT liveness: a pod that has registered no tools cannot
            # serve a call, but restarting it does not conjure tools -- it just
            # replays the same startup. this check being tier-less is why the
            # tool pods shipped with no livenessProbe at all.
            HealthCheck(
                name="tools_registered",
                probe=lambda: server.tools_count > 0,
                tier=HealthTier.READY,
            ),
            # readiness gate: report NOT-READY until the pod's Hub-JWKS cache has had its first
            # successful fetch. before it warms, the pod verifies every inbound identity token
            # against an EMPTY keyset and rejects fail-closed, so a k8s readiness probe that
            # flipped ready too early would route calls the pod is guaranteed to fail. gating on
            # jwks_warmed keeps the pod out of rotation until it can actually verify a token.
            HealthCheck(
                name="jwks_warmed",
                probe=lambda: server.jwks_warmed,
                tier=HealthTier.READY,
            ),
        ]
        self._refuse_colliding_health_checks(checks)
        checks.extend(self._extra_health_checks)
        health_server = HealthServer(
            port=self._health_port,
            service_name=self._service_name,
            version=self._version,
            # serve the pod's in-flight-requests gauge on /metrics so KEDA's
            # prometheus scaler can autoscale the tool-pod Deployment on
            # aggregate in-flight call load through the one HTTP listener the
            # pod already runs for /healthz.
            metrics_provider=server.render_metrics,
            checks=checks,
        )
        try:
            await health_server.start()
        except Exception as exc:
            log.warning(
                "health server failed to start; tool pod will run without /healthz",
                extra={
                    "extra_data": {
                        "service": self._service_name,
                        "error": str(exc),
                    }
                },
            )
            return None
        return health_server

    def _refuse_colliding_health_checks(self, builtin: list[HealthCheck]) -> None:
        """refuse a contributed check whose name is already taken.

        :class:`HealthServer` keys nothing by name, so a duplicate would be evaluated twice and
        reported as two components called the same thing -- and an operator reading
        ``/healthz/ready?format=json`` to find which subsystem is down could not tell which
        ``nats`` is red.

        :param builtin: the checks the bootstrap itself registers
        :ptype builtin: list[HealthCheck]
        :return: nothing
        :rtype: None
        :raises ToolPodConfigError: naming the colliding check
        """
        seen = {check.name for check in builtin}
        for check in self._extra_health_checks:
            if check.name in seen:
                raise ToolPodConfigError(
                    f"{self._service_name} contributes a health check named {check.name!r}, which "
                    f"is already registered (bootstrap checks: "
                    f"{', '.join(c.name for c in builtin)}); give each check in "
                    f"extra_health_checks a unique name",
                    variable="extra_health_checks",
                )
            seen.add(check.name)

    async def build_server(self) -> "ToolServer":
        """build the ``ToolServer`` instance.

        subclasses MUST override to construct the server with
        host-specific configuration (NATS URL, namespace, pod id,
        bootstrap token, namespace collection, etc.).

        :return: configured but unstarted ToolServer
        :rtype: ToolServer
        :raises NotImplementedError: when subclass does not override
        """
        raise NotImplementedError("subclasses must override build_server")

    async def register_tools(self, server: "ToolServer") -> None:
        """register all tools on the server before serving.

        subclasses MUST override to call ``server.register(tool)`` for
        each tool the pod exposes.

        :param server: tool server returned by ``build_server``
        :ptype server: ToolServer
        :return: None
        :rtype: None
        :raises NotImplementedError: when subclass does not override
        """
        raise NotImplementedError("subclasses must override register_tools")

    async def run_serve(self, server: "ToolServer") -> None:
        """await ``server.serve()`` until shutdown completes.

        default implementation calls ``server.serve()`` directly.
        subclasses with additional lifecycle (e.g. holding a hub
        client open across the serve loop) override this hook.

        :param server: tool server with tools registered and signals installed
        :ptype server: ToolServer
        :return: None
        :rtype: None
        """
        await server.serve()

    def install_signal_handlers(self, server: "ToolServer") -> None:
        """install SIGTERM and SIGINT handlers that schedule shutdown.

        each handler spawns :meth:`shutdown_server` via ``spawn_background``
        so the coroutine outcome lands in the structured logger rather
        than the default loop's exception printer, and so a shutdown that
        fails still ends the serve loop.

        :param server: tool server whose ``shutdown`` coroutine is scheduled
        :ptype server: ToolServer
        :return: None
        :rtype: None
        """
        loop = asyncio.get_running_loop()
        for sig, sig_label in ((signal.SIGTERM, "sigterm"), (signal.SIGINT, "sigint")):
            handler = self.make_signal_handler(server, sig_label)
            loop.add_signal_handler(sig, handler)

    def make_signal_handler(
        self,
        server: "ToolServer",
        sig_label: str,
    ) -> Callable[[], Awaitable[None] | None]:
        """build the signal-handler closure for ``sig_label``.

        :param server: tool server whose ``shutdown`` will be scheduled
        :ptype server: ToolServer
        :param sig_label: short label used in the spawned task name
        :ptype sig_label: str
        :return: signal-handler callable suitable for ``loop.add_signal_handler``
        :rtype: Callable[[], Awaitable[None] | None]
        """
        task_name = f"{self._service_name}-shutdown-{sig_label}"

        def _handler() -> None:
            spawn_background(self.shutdown_server(server, reason=sig_label), name=task_name, logger=log)

        return _handler
