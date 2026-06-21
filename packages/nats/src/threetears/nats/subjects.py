"""typed NATS subject builders.

every NATS subject produced by 3tears applications passes through this
module. callers never construct subject strings by ``f"..."`` —
:class:`Subjects` owns the format, the namespace prefix, and the
sanitization rules. this is the single point where a typo or
namespace-prefix omission gets caught.

design properties
-----------------

- :class:`Subject` is an opaque ``@dataclass(frozen=True, slots=True)``,
  not a ``str`` subclass. consumers that want the raw subject string
  call ``str(subject)``; everything else takes ``Subject`` by type.
  this means a function expecting ``Subject`` cannot accidentally be
  passed a bare interpolated string.
- :class:`Subjects` factory has one classmethod per subject family in
  the canonical aibots topology
  (``docs/done/design-01-nats-topology.md``). adding a new family is
  one method; reformatting an existing family is one method.
- the namespace prefix (``aibots`` by default, env-driven by
  ``FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE``) is bound at
  :class:`NatsClient` connect time. :class:`Subjects` reads it from a
  ``ContextVar`` that the client populates; tests and library code can
  set it explicitly via :func:`set_default_namespace`.
- ``Subject.raw(...)`` is the diagnostics escape hatch — for one-off
  subjects in tests or migrations. production code that finds itself
  reaching for ``raw`` is missing a builder; add the builder.

separator sanitization
----------------------

raw segment values may contain ``.``; when interpolated they are
replaced with ``-`` so the dot separator is never overloaded.
mirror's :func:`threetears.core.namespaces.build_namespace_name`
rule.
"""

from __future__ import annotations

import hashlib
import os
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Final, Literal
from uuid import UUID

__all__ = [
    "DEFAULT_NAMESPACE",
    "Subject",
    "SubjectKind",
    "Subjects",
    "get_default_namespace",
    "set_default_namespace",
]


#: default subject namespace when neither
#: :data:`FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE` env var nor an explicit
#: :func:`set_default_namespace` call has set one. matches the platform
#: documented default.
DEFAULT_NAMESPACE: Final[str] = "aibots"


SubjectKind = Literal["point", "pattern", "reply"]
"""subject category.

- ``point``: concrete fully-qualified subject (e.g.
  ``aibots.tools.heartbeat.pod-abc``). usable for both publish and
  subscribe.
- ``pattern``: contains nats wildcards (``*`` or ``>``); subscribe-only.
- ``reply``: opaque inbox returned by nats-py for request/reply paths;
  treated as opaque by every caller.
"""


_namespace_var: ContextVar[str | None] = ContextVar(
    "threetears_nats_default_namespace",
    default=None,
)


def set_default_namespace(namespace: str) -> None:
    """set process-wide default namespace prefix used by :class:`Subjects`.

    :class:`NatsClient.connect` calls this with its configured
    ``nats_subject_namespace`` so every subject built afterwards picks
    up the correct prefix without callers having to thread it through.

    :param namespace: subject namespace prefix; must be non-empty
    :ptype namespace: str
    :return: nothing
    :rtype: None
    :raises ValueError: if namespace is empty
    """
    if not namespace:
        raise ValueError("namespace must be non-empty")
    _namespace_var.set(namespace)


def get_default_namespace() -> str:
    """resolve current default namespace prefix.

    resolution order:

    1. value previously set via :func:`set_default_namespace`
    2. env var ``FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE`` (read at call
       time, not import time, so changes during process lifetime are
       observed)
    3. :data:`DEFAULT_NAMESPACE`

    :return: resolved namespace prefix
    :rtype: str
    """
    explicit = _namespace_var.get()
    result: str
    if explicit is not None:
        result = explicit
    else:
        result = os.environ.get("FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE", DEFAULT_NAMESPACE)
    return result


def _sanitize(segment: str | UUID) -> str:
    """coerce a raw segment value to a subject-safe token.

    replaces ``.`` with ``-`` so the dot separator is never
    overloaded. UUID values render as their 36-char canonical string.

    :param segment: raw segment value
    :ptype segment: str | UUID
    :return: subject-safe token
    :rtype: str
    """
    raw = str(segment)
    return raw.replace(".", "-")


@dataclass(frozen=True, slots=True)
class Subject:
    """opaque NATS subject token.

    consumers should treat instances as opaque; cast to ``str`` at the
    nats-py boundary. wrapper code in :class:`threetears.nats.NatsClient`
    is the only legitimate ``str(subject)`` call site outside of
    tests / diagnostics.

    :param path: full dotted subject (with namespace prefix already applied)
    :ptype path: str
    :param kind: subject category — point, wildcard pattern, or reply inbox
    :ptype kind: SubjectKind
    """

    path: str
    kind: SubjectKind

    def __str__(self) -> str:
        """return raw subject string for nats-py boundary.

        :return: dotted subject string
        :rtype: str
        """
        return self.path

    @classmethod
    def raw(cls, full_subject: str, *, kind: SubjectKind = "point") -> Subject:
        """construct a :class:`Subject` from an already-formatted string.

        diagnostics / migration escape hatch. production code should
        prefer a builder on :class:`Subjects`; reaching for ``raw`` is a
        signal that a new builder is missing.

        :param full_subject: pre-formatted subject string
        :ptype full_subject: str
        :param kind: subject category
        :ptype kind: SubjectKind
        :return: opaque subject token
        :rtype: Subject
        :raises ValueError: if subject string is empty
        """
        if not full_subject:
            raise ValueError("full_subject must be non-empty")
        return cls(path=full_subject, kind=kind)


def _ns() -> str:
    """resolve namespace prefix at builder-call time.

    :return: namespace prefix
    :rtype: str
    """
    return get_default_namespace()


class Subjects:
    """factory for every canonical 3tears / aibots NATS subject family.

    classmethods are organized by area:

    - **agent**: hub <-> agent pod routing, registration, heartbeats
    - **tools**: tool registry, tool pod heartbeats, tool calls
    - **gateway**: AI model gateway request/reply + streaming
    - **hub**: handshake, secrets, user resolution, streaming bridge,
      usage tracking
    - **audit**: audit event publishing
    - **l3**: L3 broker query / batch / transactional
    - **acl**: ACL invalidation broadcasts
    - **namespace**: namespace catalog discovery
    - **datasource**: datasource query routing
    - **cache**: cross-pod cache invalidation
    - **config epochs**: per-domain generation-stamped reload signals
      (metallm capabilities, gateway catalog, MCP RBAC)
    - **deadletter**: catch-all for failed callbacks
    """

    # ------------------------------------------------------------------
    # agent (hub <-> agent pod)
    # ------------------------------------------------------------------

    @classmethod
    def agent_register(cls) -> Subject:
        """request/reply subject for agent pod self-registration.

        :return: subject ``{ns}.agents.register``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.agents.register", kind="point")

    @classmethod
    def agent_deregister(cls) -> Subject:
        """fire-and-forget subject an agent pod publishes on graceful shutdown.

        lets a cleanly-stopping pod remove its endpoint from the router catalog
        IMMEDIATELY rather than waiting out the heartbeat timeout (the slow
        backstop that only catches ungraceful kills). every router replica
        subscribes WITHOUT a queue group so each replica's catalog drops the pod.

        :return: subject ``{ns}.agents.deregister``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.agents.deregister", kind="point")

    @classmethod
    def agent_heartbeat(cls, pod_id: str | UUID) -> Subject:
        """publish subject for one agent pod's heartbeat.

        :param pod_id: agent pod identifier
        :ptype pod_id: str | UUID
        :return: subject ``{ns}.agents.heartbeat.{pod_id}``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.agents.heartbeat.{_sanitize(pod_id)}", kind="point")

    @classmethod
    def agent_heartbeat_wildcard(cls) -> Subject:
        """wildcard subscribe pattern for all agent-pod heartbeats.

        :return: subject ``{ns}.agents.heartbeat.>``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.agents.heartbeat.>", kind="pattern")

    @classmethod
    def agent_reregister_request(cls, pod_id: str | UUID) -> Subject:
        """router -> agent pod nudge to re-register after deregistration.

        Published by :class:`AgentHeartbeatMonitor.handle_heartbeat` when a
        heartbeat arrives from a pod whose endpoint is missing from the
        catalog (typically because the heartbeat-timeout monitor
        deregistered the pod during a long host pause such as a laptop
        sleep, and the pod is now back alive and emitting heartbeats
        again). The agent pod subscribes to its own ``pod_id``-tailed
        subject and runs the full registration handshake on receipt --
        restoring the catalog endpoint without forcing a process restart.

        :param pod_id: target agent pod identifier
        :ptype pod_id: str | UUID
        :return: subject ``{ns}.agents.reregister_request.{pod_id}``
        :rtype: Subject
        """
        return Subject(
            path=f"{_ns()}.agents.reregister_request.{_sanitize(pod_id)}",
            kind="point",
        )

    @classmethod
    def agent_route(cls, agent_id: str | UUID) -> Subject:
        """request/reply subject for hub -> agent inbound user message.

        :param agent_id: target agent identifier
        :ptype agent_id: str | UUID
        :return: subject ``{ns}.agents.route.{agent_id}``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.agents.route.{_sanitize(agent_id)}", kind="point")

    @classmethod
    def agent_route_wildcard(cls) -> Subject:
        """wildcard subscribe pattern for all agent inbound traffic.

        :return: subject ``{ns}.agents.route.>``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.agents.route.>", kind="pattern")

    @classmethod
    def agent_internal(cls, agent_id: str | UUID, pod_id: str | UUID) -> Subject:
        """direct agent-pod inbox subject for routed traffic.

        :param agent_id: target agent identifier
        :ptype agent_id: str | UUID
        :param pod_id: target pod identifier
        :ptype pod_id: str | UUID
        :return: subject ``{ns}.agents.internal.{agent_id}.{pod_id}``
        :rtype: Subject
        """
        return Subject(
            path=f"{_ns()}.agents.internal.{_sanitize(agent_id)}.{_sanitize(pod_id)}",
            kind="point",
        )

    # ------------------------------------------------------------------
    # tools (registry / tool pods)
    # ------------------------------------------------------------------

    @classmethod
    def tools_register(cls) -> Subject:
        """request/reply subject for tool-pod registration with registry.

        :return: subject ``{ns}.tools.register``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.tools.register", kind="point")

    @classmethod
    def tools_heartbeat(cls, pod_id: str | UUID) -> Subject:
        """publish subject for one tool-pod's heartbeat.

        :param pod_id: tool pod identifier
        :ptype pod_id: str | UUID
        :return: subject ``{ns}.tools.heartbeat.{pod_id}``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.tools.heartbeat.{_sanitize(pod_id)}", kind="point")

    @classmethod
    def tools_heartbeat_wildcard(cls) -> Subject:
        """wildcard subscribe pattern for all tool-pod heartbeats.

        :return: subject ``{ns}.tools.heartbeat.>``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.tools.heartbeat.>", kind="pattern")

    @classmethod
    def tools_discover(cls) -> Subject:
        """request/reply subject for agent -> registry tool catalog discovery.

        :return: subject ``{ns}.tools.discover``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.tools.discover", kind="point")

    @classmethod
    def tools_call(cls) -> Subject:
        """request/reply subject for agent -> registry tool invocation.

        :return: subject ``{ns}.tools.call``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.tools.call", kind="point")

    @classmethod
    def tools_internal(cls, pod_id: str | UUID) -> Subject:
        """request/reply subject for registry -> tool-pod proxied call.

        :param pod_id: target tool pod identifier
        :ptype pod_id: str | UUID
        :return: subject ``{ns}.tools.internal.{pod_id}``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.tools.internal.{_sanitize(pod_id)}", kind="point")

    @classmethod
    def tools_probe(cls, pod_id: str | UUID) -> Subject:
        """request/reply subject for liveness probe against one tool pod.

        :param pod_id: target tool pod identifier
        :ptype pod_id: str | UUID
        :return: subject ``{ns}.tools.probe.{pod_id}``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.tools.probe.{_sanitize(pod_id)}", kind="point")

    # ------------------------------------------------------------------
    # gateway (AI model gateway)
    # ------------------------------------------------------------------

    @classmethod
    def gateway_completion(cls) -> Subject:
        """request/reply subject for completion request to AI gateway.

        :return: subject ``{ns}.gateway.completion``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.gateway.completion", kind="point")

    @classmethod
    def gateway_embedding(cls) -> Subject:
        """request/reply subject for embedding request to AI gateway.

        :return: subject ``{ns}.gateway.embedding``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.gateway.embedding", kind="point")

    @classmethod
    def gateway_health(cls) -> Subject:
        """request/reply subject for gateway health check.

        :return: subject ``{ns}.gateway.health``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.gateway.health", kind="point")

    @classmethod
    def gateway_stream(cls, correlation_id: str | UUID) -> Subject:
        """publish subject for gateway -> agent token streaming.

        :param correlation_id: correlation identifier shared with originating request
        :ptype correlation_id: str | UUID
        :return: subject ``{ns}.gateway.stream.{correlation_id}``
        :rtype: Subject
        """
        return Subject(
            path=f"{_ns()}.gateway.stream.{_sanitize(correlation_id)}",
            kind="point",
        )

    # ------------------------------------------------------------------
    # hub (handshake, user resolve, secrets, streaming bridge, usage)
    # ------------------------------------------------------------------

    @classmethod
    def hub_handshake(cls) -> Subject:
        """request/reply subject for agent / tool-pod handshake with hub.

        :return: subject ``{ns}.hub.handshake``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.hub.handshake", kind="point")

    @classmethod
    def hub_secrets_request(cls) -> Subject:
        """request/reply subject for encrypted secret delivery from hub.

        :return: subject ``{ns}.hub.secrets.request``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.hub.secrets.request", kind="point")

    @classmethod
    def hub_user_resolve(cls) -> Subject:
        """request/reply subject for user identity lookup against hub.

        :return: subject ``{ns}.hub.user.resolve``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.hub.user.resolve", kind="point")

    @classmethod
    def hub_channel_installs(cls) -> Subject:
        """request/reply subject for a channel adapter to fetch its installs.

        the adapter is sandboxed (NATS-only, no DB credentials), so it
        asks the hub for the active bot installs of a channel type
        (bot token refs + the agent each routes to) instead of reading
        ``platform.channel_configs`` directly.

        :return: subject ``{ns}.hub.channel.installs``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.hub.channel.installs", kind="point")

    @classmethod
    def channels_deliver(cls, channel_type: str) -> Subject:
        """durable JetStream subject for an agent answer awaiting channel delivery.

        the agent publishes a finished answer here (with the channel routing
        lifted off the inbound message) on completion; the channel adapter is a
        durable consumer that posts it to the destination thread. durable so an
        answer that completes while the adapter is restarting is redelivered,
        never lost. backed by the ``{ns}_channels_deliver`` JetStream stream
        over ``{ns}.channels.deliver.*``.

        :param channel_type: channel family (e.g. ``slack``, ``discord``)
        :ptype channel_type: str
        :return: subject ``{ns}.channels.deliver.{channel_type}``
        :rtype: Subject
        """
        return Subject(
            path=f"{_ns()}.channels.deliver.{_sanitize(channel_type)}",
            kind="point",
        )

    @classmethod
    def channels_deliver_wildcard(cls) -> Subject:
        """wildcard subject covering every channel-delivery family.

        the JetStream ``{ns}_channels_deliver`` stream is declared over this
        pattern; durable consumers filter to one ``channel_type``.

        :return: subject ``{ns}.channels.deliver.*``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.channels.deliver.*", kind="pattern")

    @classmethod
    def hub_usage_track(cls) -> Subject:
        """publish subject for usage-tracking events posted to hub.

        :return: subject ``{ns}.hub.usage.track``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.hub.usage.track", kind="point")

    @classmethod
    def hub_stream(cls, correlation_id: str | UUID) -> Subject:
        """publish subject for agent -> hub token streaming.

        :param correlation_id: correlation identifier for client request
        :ptype correlation_id: str | UUID
        :return: subject ``{ns}.hub.stream.{correlation_id}``
        :rtype: Subject
        """
        return Subject(
            path=f"{_ns()}.hub.stream.{_sanitize(correlation_id)}",
            kind="point",
        )

    # ------------------------------------------------------------------
    # audit
    # ------------------------------------------------------------------

    @classmethod
    def audit_event(cls, event_type: str) -> Subject:
        """publish subject for one audit event.

        :param event_type: dotted event type (e.g. ``workspace.doc_set``)
        :ptype event_type: str
        :return: subject ``{ns}.audit.{event_type}`` (event_type passed through verbatim — its dots are part of the addressable subject hierarchy)
        :rtype: Subject
        :raises ValueError: if event_type is empty
        """
        if not event_type:
            raise ValueError("event_type must be non-empty")
        # NOTE: event_type intentionally NOT sanitized — its dots are
        # the namespace separators audit consumers subscribe against
        # (e.g. wildcard `aibots.audit.workspace.>` for workspace events).
        return Subject(path=f"{_ns()}.audit.{event_type}", kind="point")

    @classmethod
    def audit_wildcard(cls, *, area: str | None = None) -> Subject:
        """wildcard subscribe pattern for audit events.

        :param area: optional sub-area filter; when ``None`` matches every audit event
        :ptype area: str | None
        :return: subject ``{ns}.audit.>`` or ``{ns}.audit.{area}.>``
        :rtype: Subject
        """
        result: Subject
        if area is None:
            result = Subject(path=f"{_ns()}.audit.>", kind="pattern")
        else:
            result = Subject(path=f"{_ns()}.audit.{area}.>", kind="pattern")
        return result

    # ------------------------------------------------------------------
    # workspaces
    # ------------------------------------------------------------------

    @classmethod
    def workspaces_create(cls) -> Subject:
        """publish subject for workspace-create namespace emission.

        every successful agent-side ``threetears.workspace.create``
        publishes one event on this subject after the workspace + file
        rows commit in the per-agent schema. the hub-side
        :class:`aibots.hub.workspace.namespace_emitter
        .WorkspaceNamespaceEmitter` subscribes (no queue group, every
        replica observes) and upserts the paired ``platform.namespaces``
        row of type ``workspace``.

        decoupling the namespace upsert from the agent is the canonical
        platform pattern -- the agent-side L3 proxy routes writes to
        the agent's own ``agent_<hex>`` schema, which has no
        ``namespaces`` table; the hub owns direct DB access via
        :class:`HubNamespaceCollection` and is the SOLE writer of
        platform-scoped catalog rows. mirrors :meth:`tools_register`.

        :return: subject ``{ns}.workspaces.create``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.workspaces.create", kind="point")

    # ------------------------------------------------------------------
    # op-log (durable write-path WAL)
    # ------------------------------------------------------------------

    @classmethod
    def oplog(cls, repo: str, branch: str) -> Subject:
        """publish/replay subject for one ``(repo, branch)`` op-log.

        the durable write path keeps one JetStream stream per
        ``(repo, branch)``; this is the single subject that stream is
        bound to. ``repo`` and ``branch`` are sanitized (``.`` -> ``-``)
        so a dotted ref name never overloads the subject separator.
        consumed by :class:`threetears.nats.OpLog`.

        :param repo: repository identifier
        :ptype repo: str
        :param branch: branch / ref name
        :ptype branch: str
        :return: subject ``{ns}.oplog.{repo}.{branch}``
        :rtype: Subject
        :raises ValueError: if repo or branch is empty
        """
        if not repo:
            raise ValueError("repo must be non-empty")
        if not branch:
            raise ValueError("branch must be non-empty")
        return Subject(
            path=f"{_ns()}.oplog.{_sanitize(repo)}.{_sanitize(branch)}",
            kind="point",
        )

    # ------------------------------------------------------------------
    # channels (cross-pod room fanout)
    # ------------------------------------------------------------------

    @classmethod
    def room(cls, room_id: str) -> Subject:
        """publish/subscribe subject for one room's cross-pod message fanout.

        the live room fanout (channels cross-pod design D1 / shard B)
        publishes one frame to this subject; every pod holding a local
        member of the room subscribes and delivers to its own sockets.

        the room id is ``{customer}:{story}:{branch}:{file}`` — arbitrary
        app-supplied segments may carry ``.``, spaces, ``*``, ``>`` (all
        illegal or ambiguous in a NATS subject token), so the room id is
        NOT :func:`_sanitize`-mapped (a colon/space room id would still
        leave illegal characters or collide across distinct ids). instead
        the token is the **SHA-256 hex digest** of the room id: a
        subject-safe (``[0-9a-f]`` only), collision-resistant,
        deterministic token. the raw room id rides in the wire envelope
        (:class:`threetears.channels.presence.wire.RoomFrame`), so the
        one-way digest needs no reversibility — the same robustness move
        the presence KV key uses. consumed by
        :class:`threetears.channels.presence.fanout.RoomFanout`.

        :param room_id: ``{customer}:{story}:{branch}:{file}`` room key
        :ptype room_id: str
        :return: subject ``{ns}.channels.room.{sha256hex(room_id)}``
        :rtype: Subject
        :raises ValueError: if room_id is empty
        """
        if not room_id:
            raise ValueError("room_id must be non-empty")
        token = hashlib.sha256(room_id.encode("utf-8")).hexdigest()
        return Subject(path=f"{_ns()}.channels.room.{token}", kind="point")

    # ------------------------------------------------------------------
    # owner-routed forward (request -> whichever pod serves a key)
    # ------------------------------------------------------------------

    @classmethod
    def forward(cls, key: str) -> Subject:
        """request/reply subject for owner-routed forwarding of one key.

        the owner-routed forward primitive
        (:func:`threetears.nats.forward` / :func:`serve_owner`) sends a
        request to whichever pod currently *serves* ``key`` and returns
        its reply. this is the single subject that request/reply rides.

        the ``key`` is arbitrary, app-supplied, and may carry ``.``,
        spaces, ``*`` or ``>`` — all illegal or ambiguous in a NATS
        subject token. as with :meth:`room`, the key is therefore NOT
        :func:`_sanitize`-mapped (which only handles ``.``); instead the
        token is the **SHA-256 hex digest** of the key: subject-safe
        (``[0-9a-f]`` only), collision-resistant, and deterministic, so
        every pod derives the same subject for the same key. the digest
        is one-way and needs no reversibility — both ends start from the
        same ``key``. consumed by
        :func:`threetears.nats.serve_owner` (subscribe, queue-grouped on
        this subject's path) and :func:`threetears.nats.forward`
        (request).

        :param key: ownership key (arbitrary app-supplied string)
        :ptype key: str
        :return: subject ``{ns}.forward.{sha256hex(key)}``
        :rtype: Subject
        :raises ValueError: if key is empty
        """
        if not key:
            raise ValueError("key must be non-empty")
        token = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return Subject(path=f"{_ns()}.forward.{token}", kind="point")

    # knowledge (correction-harvest drafts)
    # ------------------------------------------------------------------

    @classmethod
    def knowledge_draft(cls) -> Subject:
        """publish subject for correction-harvest draft events + commands.

        every agent-side correction harvest (knowledge-task-06) publishes
        one :class:`threetears.knowledge.KnowledgeDraftEvent` (new draft)
        or :class:`threetears.knowledge.KnowledgeDraftCommand` (confirm /
        edit / discard of the author's own draft) on this subject. the
        hub-side ``KnowledgeDraftEmitter`` subscribes (no queue group,
        every replica observes) and materializes / mutates the
        ``status='draft'`` row through the hub's knowledge Collections.

        decoupling the knowledge write from the agent is the canonical
        platform pattern (mirrors :meth:`workspaces_create`): the
        agent-side L3 proxy is admitted only SELECT traffic against the
        ``platform.*`` knowledge tables (the broker's read-only
        carve-out), so the hub — sole writer of platform-scoped rows —
        owns the write. the deterministic ``draft_id`` keying makes the
        upsert idempotent under at-least-once delivery.

        :return: subject ``{ns}.knowledge.draft``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.knowledge.draft", kind="point")

    # ------------------------------------------------------------------
    # l3 broker
    # ------------------------------------------------------------------

    @classmethod
    def l3_query(cls) -> Subject:
        """request/reply subject for agent -> L3 broker single-statement query.

        :return: subject ``{ns}.l3.query``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.l3.query", kind="point")

    @classmethod
    def l3_batch(cls) -> Subject:
        """request/reply subject for agent -> L3 broker transactional batch.

        :return: subject ``{ns}.l3.batch``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.l3.batch", kind="point")

    @classmethod
    def l3_tx(
        cls,
        op: Literal["begin", "execute", "fetchrow", "fetch", "commit", "rollback"],
    ) -> Subject:
        """request/reply subject for L3 broker explicit-transaction operations.

        the platform broker exposes six per-op subjects (versus the
        original four-method shape) so DML ``execute``, single-row
        ``fetchrow``, and multi-row ``fetch`` are addressable
        independently from a generic ``exec``. the asyncpg session
        on the broker side dispatches each op to the matching
        connection method without pre-classifying the SQL verb.

        :param op: transaction operation phase
        :ptype op: Literal["begin", "execute", "fetchrow", "fetch", "commit", "rollback"]
        :return: subject ``{ns}.l3.tx.{op}``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.l3.tx.{op}", kind="point")

    # ------------------------------------------------------------------
    # acl
    # ------------------------------------------------------------------

    @classmethod
    def acl_invalidate(
        cls,
        kind: Literal["membership", "assignment", "role"],
    ) -> Subject:
        """publish subject for ACL cache invalidation broadcasts.

        :param kind: which ACL surface to invalidate
        :ptype kind: Literal["membership", "assignment", "role"]
        :return: subject ``{ns}.acl.{kind}.invalidate``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.acl.{kind}.invalidate", kind="point")

    # ------------------------------------------------------------------
    # namespace
    # ------------------------------------------------------------------

    @classmethod
    def namespace_discover(cls) -> Subject:
        """request/reply subject for agent -> hub namespace catalog discovery.

        :return: subject ``{ns}.namespace.discover``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.namespace.discover", kind="point")

    # ------------------------------------------------------------------
    # datasource
    # ------------------------------------------------------------------

    @classmethod
    def datasource_query(cls, name: str) -> Subject:
        """request/reply subject for datasource query routing.

        :param name: datasource name
        :ptype name: str
        :return: subject ``{ns}.datasource.{name}.query``
        :rtype: Subject
        :raises ValueError: if name is empty
        """
        if not name:
            raise ValueError("datasource name must be non-empty")
        return Subject(path=f"{_ns()}.datasource.{_sanitize(name)}.query", kind="point")

    # ------------------------------------------------------------------
    # cache invalidation
    # ------------------------------------------------------------------

    @classmethod
    def cache_invalidate(cls) -> Subject:
        """publish subject for cross-pod cache invalidation broadcasts.

        the cache invalidation subject is NOT namespace-prefixed: it is
        a cross-platform constant (``threetears.cache.invalidate``) so
        that every 3tears collection in every consumer process listens
        on the same subject regardless of the env-specific aibots
        prefix.

        :return: subject ``threetears.cache.invalidate``
        :rtype: Subject
        """
        return Subject(path="threetears.cache.invalidate", kind="point")

    # ------------------------------------------------------------------
    # config epochs (cross-pod cache reload coherence)
    # ------------------------------------------------------------------
    #
    # epoch subjects live in the publishing product's namespace --
    # subscribers are always sibling pods of the same product, never
    # cross-product, so the standard namespace-prefixed shape applies.
    # multi-env deployments on a shared NATS cluster get partitioned by
    # the prefix.
    #
    # the path shape is asymmetric between metallm and aibots-family
    # builders BY DESIGN: metallm uses its own namespace
    # (``metallm``) and is single-product per namespace, so its
    # subjects have nothing after the namespace to disambiguate
    # against (``metallm.capabilities.epoch``). the aibots namespace
    # multiplexes hub / gateway / agents / tools / channels / MCP
    # under one prefix, so aibots subjects always include a product
    # segment as the second token (``aibots.gateway.catalog.epoch``).
    # the asymmetry mirrors the underlying namespace shape; do not
    # "normalize" it. consumed by :class:`threetears.epoch.client.
    # EpochClient` and :class:`threetears.epoch.listener.EpochListener`.

    @classmethod
    def metallm_capabilities_epoch(cls) -> Subject:
        """publish + subscribe subject for metallm capabilities-registry epoch.

        bumped by metallm admin POST/PATCH on the ``models`` table
        after the in-process ``register_model_capabilities_bulk(...)``
        call. metallm sibling pods subscribe to reload their local
        :class:`threetears.models.tracking.ModelCapabilities` registry
        from the row in the next read. intended call site: metallm
        process bound to namespace ``metallm``.

        constraint (single-product-per-namespace): the path has no
        product segment after ``{ns}`` because the metallm namespace
        is single-product. if a second product ever joins the
        metallm namespace, this subject must be renamed to
        ``{ns}.metallm.capabilities.epoch`` and every subscriber
        must roll forward together — a coordinated wire-protocol
        break, not a hot-deploy change. the constraint is documented
        here at the call site (not just in the section header) so
        the implication is visible from the builder's own docstring.

        :return: subject ``{ns}.capabilities.epoch`` (metallm-bound:
            ``metallm.capabilities.epoch``)
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.capabilities.epoch", kind="point")

    @classmethod
    def gateway_catalog_epoch(cls) -> Subject:
        """publish + subscribe subject for aibots-gateway catalog epoch.

        bumped by hub admin endpoints that mutate ``gateway_models``,
        ``gateway_providers``, or ``gateway_credit_rates``. gateway
        sibling pods subscribe to re-run ``_load_catalog`` immediately
        rather than waiting for the next ``cache_ttl_seconds`` tick.
        intended call site: aibots-bound process (hub or gateway).

        :return: subject ``{ns}.gateway.catalog.epoch`` (aibots-bound:
            ``aibots.gateway.catalog.epoch``)
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.gateway.catalog.epoch", kind="point")

    @classmethod
    def mcp_rbac_epoch(cls) -> Subject:
        """publish + subscribe subject for MCP per-tool RBAC epoch.

        bumped by MCP admin endpoints that grant or revoke per-tool
        access. MCP-host sibling pods subscribe to reload their RBAC
        view from the row before the next tool-call authorization
        check, so a revoked grant cannot ride a stale cache forward.
        intended call site: aibots-bound MCP-host process.

        :return: subject ``{ns}.mcp.rbac.epoch`` (aibots-bound:
            ``aibots.mcp.rbac.epoch``)
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.mcp.rbac.epoch", kind="point")

    # ------------------------------------------------------------------
    # deadletter
    # ------------------------------------------------------------------

    @classmethod
    def deadletter(cls, original_path: str) -> Subject:
        """publish subject for deadlettering a failed callback's message.

        produced by :class:`NatsClient` when a subscribe callback raises
        and the subscription has ``deadletter_on_failure=True`` (default).

        :param original_path: dotted path of subject the failed message arrived on
        :ptype original_path: str
        :return: subject ``{ns}.deadletter.{original_path}``
        :rtype: Subject
        """
        return Subject(path=f"{_ns()}.deadletter.{original_path}", kind="point")
