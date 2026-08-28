"""registration handler for tool pod manifests.

subscribes to NATS registration subject, validates incoming
manifests, authenticates pods, and registers tools with
additive endpoint merging for multi-pod horizontal scaling.
freshly registered endpoints are parked in the 'pending'
state until an end-to-end reachability probe round-trips;
only then are they promoted to 'available' and exposed to
routing. this eliminates the window where a pod is in the
catalog but its NATS subscription has not yet propagated.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import BaseModel

from threetears.agent.tools.server import RegistrationManifest, RegistrationResponse
from threetears.core.namespaces import (
    build_agent_namespace_name,
    build_tool_provider_node_name,
)
from threetears.nats import IncomingMessage, Subjects
from threetears.observe import get_logger
from threetears.registry.auth import ToolPodAuth, ToolPodAuthenticator
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.ownership import tool_is_registrable

if TYPE_CHECKING:
    from threetears.nats import NatsClient, Subscription

__all__ = [
    "ProbeRequest",
    "ProbeResponse",
    "RegistrationHandler",
    # RE-EXPORTED, not defined here. It lives beside ``RegistrationManifest`` in
    # ``threetears.agent.tools.server`` because the POD is what parses it, and this
    # package depends on that one rather than the other way round. Kept in ``__all__``
    # so ``from threetears.registry import RegistrationResponse`` still resolves.
    "RegistrationResponse",
]

# NOTE: ``RegistrationHandler.handle_registration`` is a public method on the
# class; classes exported through ``__all__`` publish their public methods
# automatically. the rename from ``_handle_registration`` to ``handle_registration``
# codifies the existing stability contract: tests drive this handler directly,
# subclass authors may override it, so the leading underscore was wrong.

log = get_logger(__name__)


class ProbeRequest(BaseModel):
    """reachability probe sent from registry to pod after registration.

    :param pod_id: identifier of pod being probed
    :ptype pod_id: str
    """

    pod_id: str


class ProbeResponse(BaseModel):
    """reachability probe acknowledgment returned by pod.

    :param pod_id: identifier of pod that answered the probe
    :ptype pod_id: str
    :param ready: whether pod reports itself ready to serve calls
    :ptype ready: bool
    """

    pod_id: str
    ready: bool = True


def _provider_node_names(nodes: "Iterable[str]", pod_id: str) -> tuple[str, ...]:
    """canonical names of the provider nodes one verified pod owns.

    A host may hold ownership as a bare NODE (``pentest``, ``aibots.admin``) or as
    the canonical namespace ROW it was materialized as (``tools.pentest``).
    :func:`build_tool_provider_node_name` accepts either and returns the canonical
    form -- it is the one builder, shared with the subject layer that mints the
    pod's grants, so the name the pod is told and the family its grant was keyed on
    cannot drift.

    **A value that cannot compose a name is DROPPED rather than raised on.** The
    ownership record is written elsewhere and this process does not own its
    validation, so one malformed entry must not refuse a registration whose other
    nodes are good. The pod simply learns nothing about that entry, and -- because
    the same tuple is what its tools are filtered against -- reaches nothing under
    it either. Logged, because a node nobody can name is a node whose sessions will
    never arrive.

    :param nodes: the ownership record's entries, in row order
    :ptype nodes: Iterable[str]
    :param pod_id: the registering pod's id, for the diagnostic
    :ptype pod_id: str
    :return: canonical provider-node names, in row order, malformed entries dropped
    :rtype: tuple[str, ...]
    """
    names: list[str] = []
    for node in nodes:
        try:
            names.append(build_tool_provider_node_name(node))
        except ValueError as exc:
            log.warning(
                "tool pod ownership entry names no provider node; the pod is told "
                "nothing about it, reaches nothing under it, and any human-in-the-loop "
                "session under it will never arrive",
                extra={"extra_data": {"pod_id": pod_id, "entry": node, "error": str(exc)}},
            )
    return tuple(names)


@dataclass(frozen=True, slots=True)
class _AuthOutcome:
    """what authentication decided, and the self-identity it resolved on the way.

    Two values rather than one because they are answered by the SAME lookup and must not be
    answered by two: ``verify_pod`` is a network round trip against the pod's row, and
    calling it twice to learn a second thing about one registration is both slower and a
    place for the two answers to disagree.

    :param error: the refusal to send back, or ``None`` when the pod is admitted
    :ptype error: str | None
    :param owned_namespaces: canonical names of the namespaces this pod owns, empty when
        none could be resolved
    :ptype owned_namespaces: tuple[str, ...]
    """

    error: str | None
    owned_namespaces: tuple[str, ...]


class RegistrationHandler:
    """handles tool registration requests from tool pods.

    subscribes to registration subject, validates manifests,
    and registers tools in catalog. multiple pods can register
    the same tool -- endpoints are merged additively by the
    catalog for horizontal scaling.
    """

    def __init__(
        self,
        catalog: ToolCatalog,
        namespace: str = "3tears",
        authenticator: ToolPodAuthenticator | None = None,
        probe_timeout: float | None = None,
    ) -> None:
        """initialize registration handler.

        :param catalog: tool catalog to register tools into
        :ptype catalog: ToolCatalog
        :param namespace: NATS subject namespace prefix
        :ptype namespace: str
        :param authenticator: optional tool pod authenticator for token verification
        :ptype authenticator: ToolPodAuthenticator | None
        :param probe_timeout: seconds to wait for reachability probe reply before
            leaving endpoint pending. sourced from THREETEARS_REGISTRY_PROBE_TIMEOUT
            env var if not provided.
        :ptype probe_timeout: float | None
        """
        from threetears.registry.config import get_probe_timeout

        self._catalog = catalog
        self._namespace = namespace
        self._authenticator = authenticator
        self._probe_timeout = probe_timeout if probe_timeout is not None else get_probe_timeout()
        self._nc: "NatsClient | None" = None
        self._sub: "Subscription | None" = None

    @property
    def subscription_active(self) -> bool:
        """whether the registration subscription is currently bound.

        readiness signal: a registry whose registration intake is not
        subscribed cannot learn about tool pods, so it must leave rotation --
        but a restart would not help if the cause is a NATS outage, which is
        why this is readiness and not liveness.

        ``True`` between a successful :meth:`start` and :meth:`stop`.

        :return: true while the registration subject is subscribed
        :rtype: bool
        """
        return self._sub is not None

    async def start(self, nc: "NatsClient") -> None:
        """start listening for registration requests.

        DQ-B7 queue-group note: registration is intentionally NOT in a
        queue group -- every registry instance must observe every
        tool-pod manifest so the catalog stays consistent across
        replicas. de-duplication happens inside :class:`ToolCatalog`.

        :param nc: connected canonical NATS wrapper client
        :ptype nc: NatsClient
        :return: nothing
        :rtype: None
        """
        self._nc = nc
        subject = Subjects.tools_register()
        self._sub = await nc.subscribe(subject=subject, cb=self.handle_registration)
        log.info(
            "registration handler started",
            extra={"extra_data": {"subject": subject.path}},
        )

    async def stop(self) -> None:
        """stop listening for registration requests."""
        if self._sub is not None and self._nc is not None:
            await self._nc.unsubscribe(self._sub)
            self._sub = None
        log.info("registration handler stopped")

    async def handle_registration(self, msg: IncomingMessage) -> None:
        """public NATS-subject handler for incoming registration manifest.

        bound by :meth:`start` as the ``cb`` callback on
        ``{namespace}.tools.register`` so every registering tool pod's
        manifest arrives here. tests exercise this surface directly by
        synthesizing a wrapper :class:`IncomingMessage` and awaiting the
        handler; keeping the entry point public is a stability contract
        -- subclasses and test doubles may rely on the name, the single
        ``msg`` parameter, and the absence of return value.

        validates manifest, authenticates pod, and registers
        tools with additive endpoint merging. replies with
        success or error response via :meth:`NatsClient.publish_reply`.

        :param msg: incoming wrapper envelope containing registration manifest
        :ptype msg: IncomingMessage
        :raises RuntimeError: when invoked before ``start`` connects NATS
        """
        if self._nc is None:
            raise RuntimeError("handle_registration invoked before NATS connected")
        try:
            manifest = RegistrationManifest.model_validate_json(msg.data)
        except Exception as exc:
            log.error(
                "registration rejected: malformed manifest",
                extra={"extra_data": {"error": str(exc)}},
            )
            response = RegistrationResponse(
                success=False,
                pod_id="unknown",
                error=f"malformed manifest: {exc}",
            )
            if msg.reply_subject is not None:
                await self._nc.publish_reply(
                    reply_subject=msg.reply_subject,
                    message=response,
                )
            return

        validation_error = self._validate_manifest(manifest)
        if validation_error is not None:
            log.warning(
                "registration rejected: validation failed",
                extra={"extra_data": {"pod_id": manifest.pod_id, "error": validation_error}},
            )
            response = RegistrationResponse(
                success=False,
                pod_id=manifest.pod_id,
                error=validation_error,
            )
            if msg.reply_subject is not None:
                await self._nc.publish_reply(
                    reply_subject=msg.reply_subject,
                    message=response,
                )
            return

        outcome = await self._authenticate_and_filter(manifest)
        if outcome.error is not None:
            log.warning(
                "registration rejected: auth failed",
                extra={"extra_data": {"pod_id": manifest.pod_id, "error": outcome.error}},
            )
            # a refusal names NOTHING. handing a rejected pod its self-identity as a
            # consolation would leak which nodes exist to a caller that just failed to
            # prove it holds any of them.
            response = RegistrationResponse(
                success=False,
                pod_id=manifest.pod_id,
                error=outcome.error,
            )
            if msg.reply_subject is not None:
                await self._nc.publish_reply(
                    reply_subject=msg.reply_subject,
                    message=response,
                )
            return

        registered = await self._register_tools(manifest)

        response = RegistrationResponse(
            success=True,
            pod_id=manifest.pod_id,
            registered_tools=registered,
            # SELF-IDENTITY. The pod's subject grants were minted at connect from the row
            # this handler just read, and the pod never sees that row -- so without this it
            # holds only tool LEAVES and derives its human-in-the-loop family from a value
            # no grant was keyed on. Derived from the verified auth context, never from the
            # manifest, so a pod cannot name a namespace it does not own.
            owned_namespaces=list(outcome.owned_namespaces),
        )
        if msg.reply_subject is not None:
            await self._nc.publish_reply(
                reply_subject=msg.reply_subject,
                message=response,
            )
        log.info(
            "registration completed",
            extra={
                "extra_data": {
                    "pod_id": manifest.pod_id,
                    "tools_count": len(registered),
                }
            },
        )

    async def _authenticate_and_filter(self, manifest: RegistrationManifest) -> _AuthOutcome:
        """authenticate the pod, filter its tools by OWNERSHIP, and resolve its self-identity.

        **One rule, on every path.** This used to answer three different questions.
        A pod carrying a token had its tools prefix-filtered against a text column
        naming what it was permitted to register; a TOKENLESS pod -- every agent's
        in-process ``ToolServer``, not a rare in-hub case -- returned here before
        any filtering ran; and a handler built with no authenticator returned
        before that. All three now go through
        :func:`~threetears.registry.ownership.tool_is_registrable`, which asks the
        namespace GRAPH who owns the most specific provider node containing the
        offered name. See that module for the rule and for why an empty graph
        enforces nothing.

        Authentication itself is unchanged and still gated on token PRESENCE:

        * a manifest carrying a token is a PLATFORM tool pod under per-key identity.
          The RAW token goes to :meth:`ToolPodAuthenticator.verify_pod`, which
          verifies it against the pod's stored key; failure REJECTS.
        * a TOKENLESS manifest is the AGENT-OWNED in-process pod. It registers over
          the agent's own NATS connection, which the auth-callout already
          authenticated per-key as an AGENT, so its identity is enforced at the
          transport; it holds no row in the host's tool-pod store and could never
          present a token. It is still ADMITTED as a principal -- what changed is
          that it is no longer admitted as an owner of everything.

        **Two different tuples, deliberately not merged.** ``owned_nodes`` is what
        the filter compares against: the PROVIDER nodes this pod owns, empty for
        an agent-owned pod because an agent owns none. ``self_identity`` is what
        travels back on the reply so the pod can key its human-in-the-loop
        subscriptions on the same node its grants were minted from -- for an
        agent-owned pod that is ``agents.<uuid>``, which is an ownership edge
        rather than a name-containment one and must never be handed to the filter.

        :param manifest: registration manifest to authenticate and filter
        :ptype manifest: RegistrationManifest
        :return: the refusal, if any, and the namespaces this pod owns
        :rtype: _AuthOutcome
        """
        owned_nodes: tuple[str, ...] = ()
        self_identity: tuple[str, ...] = ()
        pod_name = manifest.pod_id

        if self._authenticator is not None and manifest.bootstrap_token is not None:
            pod_auth: ToolPodAuth | None = await self._authenticator.verify_pod(manifest.bootstrap_token)
            if pod_auth is None:
                log.warning(
                    "tool pod registration rejected: invalid token",
                    extra={"extra_data": {"pod_id": manifest.pod_id}},
                )
                return _AuthOutcome(error="invalid bootstrap token", owned_namespaces=())
            owned_nodes = _provider_node_names(pod_auth.owned_namespaces, manifest.pod_id)
            self_identity = owned_nodes
            pod_name = pod_auth.name
        else:
            # agent-owned / tokenless registration, or a handler with no authenticator
            # at all. Authenticated at the NATS layer in the first case and not
            # authenticated at all in the second; either way it owns no provider node
            # and is filtered against what other pods own.
            self_identity = self._agent_owned_namespaces(manifest)

        directory = await self._provider_node_directory(manifest.pod_id)
        if directory is None:
            return _AuthOutcome(
                error=(
                    "ownership graph unavailable; registration refused rather than admitted "
                    "unfiltered. this is retried on the pod's next heartbeat"
                ),
                owned_namespaces=(),
            )

        allowed_tools = []
        rejected_tools = []
        for tool in manifest.tools:
            if tool_is_registrable(tool_name=tool.name, owned_nodes=owned_nodes, provider_nodes=directory):
                allowed_tools.append(tool)
            else:
                rejected_tools.append(tool.name)

        if rejected_tools:
            log.warning(
                "tool pod tools rejected (a provider node this pod does not own contains the name)",
                extra={
                    "extra_data": {
                        "pod_id": manifest.pod_id,
                        "pod_name": pod_name,
                        "rejected": rejected_tools,
                        "owned_nodes": list(owned_nodes),
                    }
                },
            )

        if not allowed_tools:
            # NAME both sides of the comparison that failed. A bare "no tools
            # authorized" is true and unactionable: it arrives AFTER the pod
            # authenticated, so it reads as a missing RBAC grant when the usual
            # causes are an ownership entry that can never match anything -- a node
            # written with a trailing separator (`evd.`) or as a glob (`evd.*`) --
            # or a name that lands inside a provider node somebody else owns.
            owns = sorted(owned_nodes) if owned_nodes else "no provider namespace"
            return _AuthOutcome(
                error=(
                    f"no tools authorized: offered {sorted(rejected_tools)}, "
                    f"this pod owns {owns}. a tool name is placed under the MOST SPECIFIC "
                    "`tools.` provider node that contains it, and only that node's owner may "
                    "register it; a name under no provider node at all may be registered only "
                    "by a pod that owns none. a node is compared on a segment boundary and is "
                    "written WITHOUT a trailing separator and WITHOUT a glob "
                    "(`evd`, never `evd.` or `evd.*`)"
                ),
                owned_namespaces=(),
            )

        manifest.tools = allowed_tools

        log.info(
            "tool pod registration authorized",
            extra={
                "extra_data": {
                    "pod_id": manifest.pod_id,
                    "pod_name": pod_name,
                    "tools_accepted": len(allowed_tools),
                    "tools_rejected": len(rejected_tools),
                }
            },
        )
        return _AuthOutcome(error=None, owned_namespaces=self_identity)

    async def _provider_node_directory(self, pod_id: str) -> tuple[str, ...] | None:
        """every provider node the host's graph holds, or ``None`` when it cannot be read.

        A handler with no authenticator has no host to ask, and answers with an
        EMPTY inventory rather than a failure: there is no ownership data in that
        deployment, so no node contains anything and an unbound pod is admitted --
        which is exactly what open mode did before this filter existed.

        A read that FAILS is a different thing and is not allowed to look like the
        first one. An empty inventory silently widens every unbound pod to
        everything, so a host that raises gets the registration REFUSED. That is
        the recoverable direction: a pod re-announces on its heartbeat, so a
        transient failure costs one interval, while admitting the manifest would
        write catalog entries nobody can take back.

        :param pod_id: the registering pod, for the diagnostic
        :ptype pod_id: str
        :return: the inventory, or ``None`` when the host could not answer
        :rtype: tuple[str, ...] | None
        """
        result: tuple[str, ...] | None = ()
        if self._authenticator is not None:
            try:
                result = tuple(await self._authenticator.provider_nodes())
            except Exception as exc:  # noqa: BLE001 -- any host failure refuses, never admits
                log.error(
                    "tool pod registration refused: the ownership graph could not be read, and a "
                    "manifest must not be admitted unfiltered. the pod retries on its next heartbeat",
                    extra={"extra_data": {"pod_id": pod_id, "error": str(exc)}},
                )
                result = None
        return result

    @staticmethod
    def _agent_owned_namespaces(manifest: RegistrationManifest) -> tuple[str, ...]:
        """the namespace an AGENT-OWNED in-process pod owns, or nothing.

        A pod that presents no token is the agent's own in-process tool server: it is not a
        row in ``tool_pods``, so it owns no provider node. What it owns is ``agents.<uuid>``
        -- the same name :class:`~threetears.agent.tools.server.ToolServer` already stamps
        as ``owner_namespace`` on every namespace row it emits.

        ``owner_agent_id`` is taken off the manifest here, which is a claim rather than a
        verified fact -- and that is acceptable for exactly this value and no other. The
        pod is on the agent's OWN authenticated NATS connection, so the identity was
        settled at the transport, and the name composed from it is the one the pod is
        about to write onto its own rows regardless. It confers no authority: nothing in
        this reply is a credential.

        :param manifest: the registering pod's manifest
        :ptype manifest: RegistrationManifest
        :return: the agent's namespace name, or an empty tuple when the pod names no agent
        :rtype: tuple[str, ...]
        """
        if manifest.owner_agent_id is None:
            return ()
        return (build_agent_namespace_name(manifest.owner_agent_id),)

    def _validate_manifest(self, manifest: RegistrationManifest) -> str | None:
        """validate registration manifest fields.

        :param manifest: manifest to validate
        :ptype manifest: RegistrationManifest
        :return: error message if validation fails, None if valid
        :rtype: str | None
        """
        if not manifest.pod_id:
            return "pod_id is required"
        if not manifest.tools:
            return "tools list is required and must not be empty"
        result = None
        return result

    async def _register_tools(
        self,
        manifest: RegistrationManifest,
    ) -> list[str]:
        """register all tools from manifest with pending endpoint for this pod.

        creates catalog entry for each tool with a single endpoint
        for the registering pod, parked in the 'pending' state.
        catalog.register() handles merging with existing entries
        for multi-pod support. after all tools are written, issues
        a reachability probe to the pod; on successful round-trip
        promotes every pending endpoint for the pod to 'available'
        via ``catalog.mark_ready``. on probe failure, endpoints
        remain pending so routing refuses to forward until the
        next heartbeat can retry promotion.

        :param manifest: validated manifest containing tools to register
        :ptype manifest: RegistrationManifest
        :return: list of full_name values registered
        :rtype: list[str]
        """
        registered: list[str] = []
        needs_probe = False
        now = datetime.now(UTC)
        for tool in manifest.tools:
            full_name = f"{tool.name}@{tool.version}"
            existing_entry = self._catalog.get(full_name)
            existing_endpoint = existing_entry.get_endpoint(manifest.pod_id) if existing_entry is not None else None
            # Preserve status for endpoints the pod has previously registered
            # so heartbeat-driven re-publication does not regress an already
            # 'available' endpoint back to 'pending' (which would trigger a
            # needless re-probe on every heartbeat). A brand-new endpoint
            # enters 'pending' and drives exactly one probe round-trip.
            if existing_endpoint is None:
                endpoint_status = "pending"
                needs_probe = True
            else:
                endpoint_status = existing_endpoint.status
            endpoint = ToolEndpoint(
                pod_id=manifest.pod_id,
                status=endpoint_status,
                in_flight=existing_endpoint.in_flight if existing_endpoint else 0,
                date_last_heartbeat=now,
            )
            entry = CatalogEntry(
                tool_name=tool.name,
                tool_version=tool.version,
                full_name=full_name,
                description=tool.description,
                input_schema=tool.input_schema,
                timeout_seconds=tool.timeout_seconds,
                requires_confirmation=tool.requires_confirmation,
                endpoints=[endpoint],
                date_registered=now,
            )
            await self._catalog.register(entry)
            registered.append(full_name)

        if needs_probe:
            await self._probe_and_promote(manifest.pod_id)

        result = registered
        return result

    async def _probe_and_promote(self, pod_id: str) -> None:
        """issue reachability probe and promote pending endpoints on success.

        sends a request-reply probe to the pod's probe subject and,
        on a successful reply within ``probe_timeout`` that parses as
        a :class:`ProbeResponse` with ``ready=True``, transitions all
        pending endpoints for the pod to 'available'. on timeout, a
        malformed reply, or ``ready=False``, leaves endpoints pending
        so subsequent registrations can retry promotion. logs the
        registered -> ready transition with per-pod latency so
        cold-start slowness surfaces in observability data.

        :param pod_id: identifier of pod whose pending endpoints to confirm
        :ptype pod_id: str
        """
        if self._nc is None:
            return
        subject = Subjects.tools_probe(pod_id)
        request = ProbeRequest(pod_id=pod_id)
        start = datetime.now(UTC)
        try:
            ack = await self._nc.request(
                subject=subject,
                message=request,
                response_type=ProbeResponse,
                timeout=timedelta(seconds=self._probe_timeout),
            )
        except Exception as exc:
            log.warning(
                "tool pod reachability probe failed or reply was malformed; endpoints remain pending",
                extra={
                    "extra_data": {
                        "pod_id": pod_id,
                        "probe_subject": subject.path,
                        "probe_timeout": self._probe_timeout,
                        "error": str(exc),
                    }
                },
            )
            return
        if not ack.ready:
            log.warning(
                "tool pod probe reply reported not-ready; endpoints remain pending",
                extra={
                    "extra_data": {
                        "pod_id": pod_id,
                        "probe_subject": subject.path,
                    }
                },
            )
            return
        promoted = await self._catalog.mark_ready(pod_id)
        ms_to_ready = (datetime.now(UTC) - start).total_seconds() * 1000.0
        for tool_key in promoted:
            log.info(
                "tool endpoint transitioned registered -> ready",
                extra={
                    "extra_data": {
                        "pod_id": pod_id,
                        "tool_key": tool_key,
                        "ms_to_ready": ms_to_ready,
                    }
                },
            )
