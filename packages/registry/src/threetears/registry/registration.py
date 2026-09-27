"""registration handler for tool pod manifests.

subscribes to NATS registration subject, validates incoming
manifests, verifies WHO published each one from its credential --
never from the pod id or anything else the manifest claims --
admits each offered tool copy by copy
(:func:`~threetears.registry.ownership.admit_copy`), and registers
the admitted copies, each keeping its own definition. every refusal
is named in the reply. multiple pods may serve one tool for
horizontal scaling.
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

from threetears.agent.tools.server import (
    RefusedTool,
    RegistrationManifest,
    RegistrationResponse,
    ToolManifestEntry,
)
from threetears.core.namespaces import (
    build_agent_namespace_name,
    build_tool_provider_node_name,
)
from threetears.nats import IncomingMessage, Subjects
from threetears.observe import get_logger
from threetears.registry.auth import ToolPodAuth, ToolPodAuthenticator
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolDefinition, ToolEndpoint
from threetears.registry.ownership import (
    PublisherStanding,
    RefusalCode,
    admit_copy,
    audience_of,
)

if TYPE_CHECKING:
    from uuid import UUID

    from threetears.nats import NatsClient, Subscription

__all__ = [
    "ProbeRequest",
    "ProbeResponse",
    # RE-EXPORTED beside ``RegistrationResponse``, for the same reason.
    "RefusedTool",
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


#: prose for each refusal code, completed with the tool name where one applies. ``code`` is
#: what a pod branches on; this is for the person reading the pod's log.
_REFUSAL_REASONS: dict[RefusalCode, str] = {
    RefusalCode.OWNED_ELSEWHERE: (
        "a provider node this publisher does not own contains the name; only that node's owner may register it"
    ),
    RefusalCode.NOT_PLATFORM_SHARED: (
        "no provider node contains the name, and a copy serving every caller under no provider node may be "
        "published only by the platform (the shared built-in pod, or a pod the host runs itself)"
    ),
    RefusalCode.UNVERIFIED_PUBLISHER: (
        "the publisher's identity was not verified, and an unverified publisher may serve no caller but its own agent"
    ),
    RefusalCode.POD_ID_MISMATCH: "the verified publisher registered under a pod id that is not its own",
    RefusalCode.INVALID_TOOL_NAME: (
        "the name composes no namespace node; offer the bare mcp name, never one rooted at `tools.`"
    ),
    RefusalCode.OWNERSHIP_GRAPH_UNAVAILABLE: (
        "the ownership graph could not be read; nothing is admitted unfiltered, and the next heartbeat retries"
    ),
    RefusalCode.PUBLISHER_VERIFICATION_UNAVAILABLE: (
        "the registry could not read the store it verifies publishers against, so nothing was decided about "
        "this publisher; the next heartbeat retries"
    ),
    RefusalCode.CATALOG_UNAVAILABLE: (
        "the registry admitted the tool but could not record it in its catalog; the next heartbeat retries"
    ),
}


#: the per-tool ADMISSION verdicts, which say this publisher may not hold this copy. only these
#: withdraw a verified publisher's prior copy; a refusal for a transient reason -- the ownership
#: graph could not be read -- says nothing about the copy, which keeps serving until the retry.
_WITHDRAWING_CODES = frozenset(
    {
        RefusalCode.OWNED_ELSEWHERE.value,
        RefusalCode.NOT_PLATFORM_SHARED.value,
        RefusalCode.INVALID_TOOL_NAME.value,
    }
)


def _refused(tools: "Iterable[ToolManifestEntry]", code: RefusalCode, reason: str | None = None) -> list[RefusedTool]:
    """one :class:`RefusedTool` per offered tool, all for the same reason.

    :param tools: the tools refused
    :ptype tools: Iterable[ToolManifestEntry]
    :param code: why
    :ptype code: RefusalCode
    :param reason: prose overriding the code's standard reason, when the refusal has more to say
    :ptype reason: str | None
    :return: the refusals, in manifest order
    :rtype: list[RefusedTool]
    """
    text = reason if reason is not None else _REFUSAL_REASONS[code]
    return [RefusedTool(name=tool.name, version=tool.version, code=code.value, reason=text) for tool in tools]


@dataclass(frozen=True, slots=True)
class _Publisher:
    """who published a manifest, as far as the registry could verify, or why it could not.

    :param standing: what was verified, for :func:`~threetears.registry.ownership.admit_copy`
    :ptype standing: PublisherStanding
    :param self_identity: the namespaces the pod owns, returned on the reply
    :ptype self_identity: tuple[str, ...]
    :param name: the publisher's display name, for the log
    :ptype name: str
    :param error: the whole-manifest refusal, or ``None`` when the publisher may proceed
    :ptype error: str | None
    :param error_code: the refusal's code, ``None`` when the publisher may proceed
    :ptype error_code: RefusalCode | None
    """

    standing: PublisherStanding
    self_identity: tuple[str, ...]
    name: str
    error: str | None = None
    error_code: RefusalCode | None = None


@dataclass(frozen=True, slots=True)
class _Verdict:
    """what one manifest is allowed to register.

    :param standing: the publisher's verified standing
    :ptype standing: PublisherStanding
    :param admitted: the tools admitted
    :ptype admitted: list[ToolManifestEntry]
    :param refused: every tool refused, with its code
    :ptype refused: list[RefusedTool]
    :param owned_namespaces: the self-identity returned on a successful reply
    :ptype owned_namespaces: tuple[str, ...]
    :param error: the whole-manifest refusal, or ``None`` when at least one tool was admitted
    :ptype error: str | None
    :param error_code: the refusal's code
    :ptype error_code: RefusalCode | None
    """

    standing: PublisherStanding
    admitted: list[ToolManifestEntry]
    refused: list[RefusedTool]
    owned_namespaces: tuple[str, ...]
    error: str | None = None
    error_code: RefusalCode | None = None


class RegistrationHandler:
    """handles tool registration requests from tool pods.

    subscribes to registration subject, validates manifests, verifies
    WHO published each one, admits each tool copy by copy, and registers
    the admitted ones in the catalog. multiple pods can register the same
    tool -- each pod's copy keeps its own definition in the catalog.
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
        :param authenticator: the host's publisher verifier; ``None`` is OPEN MODE, in which no
            identity or ownership is enforced and the handler says so once at :meth:`start`
        :ptype authenticator: ToolPodAuthenticator | None
        :param probe_timeout: seconds to wait for reachability probe reply before
            leaving endpoint pending. sourced from THREETEARS_REGISTRY_PROBE_TIMEOUT
            env var if not provided.
        :ptype probe_timeout: float | None
        :raises TypeError: when ``authenticator`` does not implement the whole
            :class:`~threetears.registry.auth.ToolPodAuthenticator` protocol -- caught where it is
            wired, rather than on the first registration that needs the missing method
        """
        from threetears.registry.config import get_probe_timeout

        missing = [
            name
            for name in ("verify_pod", "verify_agent", "provider_nodes")
            if authenticator is not None and not callable(getattr(authenticator, name, None))
        ]
        if missing:
            raise TypeError(
                f"{type(authenticator).__name__} does not implement ToolPodAuthenticator; missing {missing}. "
                "an authenticator verifies Tool Pods (verify_pod), agents' in-process servers (verify_agent) "
                "and reads the ownership graph (provider_nodes)"
            )
        self._catalog = catalog
        self._namespace = namespace
        self._authenticator = authenticator
        self._probe_timeout = probe_timeout if probe_timeout is not None else get_probe_timeout()
        self._nc: "NatsClient | None" = None
        self._sub: "Subscription | None" = None
        # agent pod ids already warned about for registering unsigned, so the rollout's progress
        # reads as one line per not-yet-rebuilt agent process rather than one per heartbeat.
        self._unsigned_agent_pods_warned: set[str] = set()

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
        if self._authenticator is None:
            log.warning(
                "registration handler running in open mode: no authenticator is wired, so no publisher "
                "identity and no tool ownership is enforced; any pod may register any tool for every caller",
                extra={"extra_data": {"subject": subject.path}},
            )
        log.info(
            "registration handler started",
            extra={"extra_data": {"subject": subject.path, "open_mode": self._authenticator is None}},
        )

    async def stop(self) -> None:
        """stop listening for registration requests."""
        if self._sub is not None and self._nc is not None:
            await self._nc.unsubscribe(self._sub)
            self._sub = None
        log.info("registration handler stopped")

    async def _reply(self, msg: IncomingMessage, response: RegistrationResponse) -> None:
        """publish ``response`` to the manifest's reply subject, when it has one.

        :param msg: the incoming manifest message
        :ptype msg: IncomingMessage
        :param response: the reply
        :ptype response: RegistrationResponse
        :return: nothing
        :rtype: None
        """
        if msg.reply_subject is not None and self._nc is not None:
            await self._nc.publish_reply(reply_subject=msg.reply_subject, message=response)

    async def handle_registration(self, msg: IncomingMessage) -> None:
        """public NATS-subject handler for incoming registration manifest.

        bound by :meth:`start` as the ``cb`` callback on
        ``{namespace}.tools.register`` so every registering tool pod's
        manifest arrives here. tests exercise this surface directly by
        synthesizing a wrapper :class:`IncomingMessage` and awaiting the
        handler; keeping the entry point public is a stability contract
        -- subclasses and test doubles may rely on the name, the single
        ``msg`` parameter, and the absence of return value.

        validates the manifest, verifies its publisher, admits each tool
        copy by copy, registers the admitted ones, withdraws a verified
        publisher's prior copy of any tool it was refused, and replies --
        naming every refused tool whether or not the registration as a
        whole succeeded.

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
                extra={"extra_data": {"error": str(exc), "error_code": RefusalCode.INVALID_MANIFEST.value}},
            )
            await self._reply(
                msg,
                RegistrationResponse(
                    success=False,
                    pod_id="unknown",
                    error=f"malformed manifest: {exc}",
                    error_code=RefusalCode.INVALID_MANIFEST.value,
                ),
            )
            return

        validation_error = self._validate_manifest(manifest)
        if validation_error is not None:
            log.warning(
                "registration rejected: validation failed",
                extra={
                    "extra_data": {
                        "pod_id": manifest.pod_id,
                        "error": validation_error,
                        "error_code": RefusalCode.INVALID_MANIFEST.value,
                    }
                },
            )
            await self._reply(
                msg,
                RegistrationResponse(
                    success=False,
                    pod_id=manifest.pod_id,
                    error=validation_error,
                    error_code=RefusalCode.INVALID_MANIFEST.value,
                ),
            )
            return

        verdict = await self._judge(manifest)
        try:
            await self._withdraw_refused_copies(manifest.pod_id, verdict)
        except Exception as exc:  # noqa: BLE001 -- a catalog failure is answered, never left as a dropped reply
            await self._answer_catalog_unavailable(msg, manifest, verdict, exc)
            return
        if verdict.error is not None:
            error_code = verdict.error_code.value if verdict.error_code is not None else None
            log.warning(
                "registration rejected: auth failed",
                extra={
                    "extra_data": {
                        "pod_id": manifest.pod_id,
                        "error": verdict.error,
                        "error_code": error_code,
                        "refused_tools": [f"{tool.name}@{tool.version} {tool.code}" for tool in verdict.refused],
                    }
                },
            )
            # a refusal names no namespace. handing a rejected pod its self-identity as a
            # consolation would leak which nodes exist to a caller that just failed to prove it
            # holds any of them. it DOES name the pod's own refused tools, which it offered.
            await self._reply(
                msg,
                RegistrationResponse(
                    success=False,
                    pod_id=manifest.pod_id,
                    refused_tools=verdict.refused,
                    error=verdict.error,
                    error_code=error_code,
                ),
            )
            return

        try:
            registered = await self._register_tools(manifest.pod_id, verdict.admitted, verdict.standing)
        except Exception as exc:  # noqa: BLE001 -- a catalog failure is answered, never left as a dropped reply
            await self._answer_catalog_unavailable(msg, manifest, verdict, exc)
            return

        await self._reply(
            msg,
            RegistrationResponse(
                success=True,
                pod_id=manifest.pod_id,
                registered_tools=registered,
                # SELF-IDENTITY. The pod's subject grants were minted at connect from the row
                # this handler just read, and the pod never sees that row -- so without this it
                # holds only tool LEAVES and derives its human-in-the-loop family from a value
                # no grant was keyed on. Derived from the verified auth context, never from the
                # manifest, so a pod cannot name a namespace it does not own.
                owned_namespaces=list(verdict.owned_namespaces),
                refused_tools=verdict.refused,
            ),
        )
        log.info(
            "registration completed",
            extra={
                "extra_data": {
                    "pod_id": manifest.pod_id,
                    "tools_count": len(registered),
                    "tools_refused": len(verdict.refused),
                    "publisher_verified": verdict.standing.verified,
                }
            },
        )

    async def _judge(self, manifest: RegistrationManifest) -> _Verdict:
        """verify the publisher, then admit or refuse each tool copy by copy.

        :param manifest: the validated manifest
        :ptype manifest: RegistrationManifest
        :return: what the manifest may register
        :rtype: _Verdict
        """
        publisher = await self._verify_publisher(manifest)
        if publisher.error is not None:
            code = publisher.error_code if publisher.error_code is not None else RefusalCode.UNVERIFIED_PUBLISHER
            return _Verdict(
                standing=publisher.standing,
                admitted=[],
                refused=_refused(manifest.tools, code, reason=publisher.error),
                owned_namespaces=(),
                error=publisher.error,
                error_code=code,
            )

        directory = await self._provider_node_directory(manifest.pod_id)
        if directory is None:
            return _Verdict(
                standing=publisher.standing,
                admitted=[],
                refused=_refused(manifest.tools, RefusalCode.OWNERSHIP_GRAPH_UNAVAILABLE),
                owned_namespaces=(),
                error=(
                    "ownership graph unavailable; registration refused rather than admitted "
                    "unfiltered. this is retried on the pod's next heartbeat"
                ),
                error_code=RefusalCode.OWNERSHIP_GRAPH_UNAVAILABLE,
            )

        audience = audience_of(manifest.pod_id)
        admitted: list[ToolManifestEntry] = []
        refused: list[RefusedTool] = []
        for tool in manifest.tools:
            verdict = admit_copy(
                tool_name=tool.name,
                audience=audience,
                standing=publisher.standing,
                provider_nodes=directory,
            )
            if verdict is None:
                admitted.append(tool)
            else:
                refused.extend(_refused([tool], verdict))

        owned_nodes = list(publisher.standing.owned_nodes)
        if refused:
            log.warning(
                "tool pod tools rejected (a provider node this pod does not own contains the name)",
                extra={
                    "extra_data": {
                        "pod_id": manifest.pod_id,
                        "pod_name": publisher.name,
                        "rejected": [tool.name for tool in refused],
                        "owned_nodes": owned_nodes,
                        "refusal_codes": {f"{tool.name}@{tool.version}": tool.code for tool in refused},
                        "audience": audience.value,
                        "publisher_verified": publisher.standing.verified,
                        "platform_shared": publisher.standing.platform_shared,
                    }
                },
            )

        if not admitted:
            # NAME both sides of the comparison that failed. A bare "no tools authorized" is
            # true and unactionable: it arrives AFTER the pod authenticated, so it reads as a
            # missing RBAC grant when the usual causes are an ownership entry that can never
            # match anything -- a node written with a trailing separator (`evd.`) or as a glob
            # (`evd.*`) -- or a name that lands inside a provider node somebody else owns.
            owns = sorted(owned_nodes) if owned_nodes else "no provider namespace"
            return _Verdict(
                standing=publisher.standing,
                admitted=[],
                refused=refused,
                owned_namespaces=(),
                error=(
                    f"no tools authorized: offered {sorted(tool.name for tool in refused)}, "
                    f"this pod owns {owns}. a tool name is placed under the MOST SPECIFIC "
                    "`tools.` provider node that contains it, and only that node's owner may "
                    "register it; a name under no provider node at all may be served to every caller "
                    "only by the platform, and by an agent's in-process server only to that agent. a "
                    "node is compared on a segment boundary and is written WITHOUT a trailing separator "
                    "and WITHOUT a glob (`evd`, never `evd.` or `evd.*`). refusal codes: "
                    f"{sorted({tool.code for tool in refused})}"
                ),
                error_code=RefusalCode.NO_TOOLS_ADMITTED,
            )

        log.info(
            "tool pod registration authorized",
            extra={
                "extra_data": {
                    "pod_id": manifest.pod_id,
                    "pod_name": publisher.name,
                    "tools_accepted": len(admitted),
                    "tools_rejected": len(refused),
                    "publisher_verified": publisher.standing.verified,
                    "platform_shared": publisher.standing.platform_shared,
                }
            },
        )
        return _Verdict(
            standing=publisher.standing,
            admitted=admitted,
            refused=refused,
            owned_namespaces=publisher.self_identity,
        )

    async def _verify_publisher(self, manifest: RegistrationManifest) -> _Publisher:
        """decide who published ``manifest``, from verified identity and never from its body.

        The pod id says which KIND of publisher is claimed, and so which verifier is asked; the
        credential then has to prove it:

        * **open mode** (no authenticator): nothing is verified and nothing enforced;
        * **a dotted pod id** is an agent's in-process server. A token must verify as the agent
          the pod id names (:meth:`ToolPodAuthenticator.verify_agent`), or the whole manifest is
          refused -- a failed signature is never downgraded to an unsigned one. With no token the
          manifest is UNSIGNED: agents built on an older SDK register this way, and in this
          release they keep their own agent-scoped copies (see
          :func:`~threetears.registry.ownership.admit_copy`), warned once per pod id, and 0.56.0
          refuses them -- ``test_unsigned_agent_concession_expires.py`` fails until it does. A pod id
          that has ever registered verified is refused unsigned, so the concession cannot be used
          to rewrite a signed agent's copy;
        * **a single-token pod id** is a Tool Pod's, whose copies serve every caller. It must
          carry a token, the token must verify (:meth:`ToolPodAuthenticator.verify_pod`), and the
          verified pod must BE the pod the manifest names.

        :param manifest: the validated manifest
        :ptype manifest: RegistrationManifest
        :return: the publisher, or why it is refused
        :rtype: _Publisher
        """
        pod_id = manifest.pod_id
        owner = Subjects.agent_inprocess_owner_id(pod_id)
        agent_identity: tuple[str, ...] = (build_agent_namespace_name(owner),) if owner is not None else ()
        unverified = PublisherStanding(verified=False, platform_shared=False, owned_nodes=())
        token = manifest.bootstrap_token
        result: _Publisher
        if self._authenticator is None:
            result = _Publisher(standing=PublisherStanding.unenforced(), self_identity=agent_identity, name=pod_id)
        elif owner is not None and token is not None:
            result = await self._verified_agent(pod_id, owner, token, agent_identity)
        elif owner is not None:
            result = self._unsigned_agent_publisher(pod_id, agent_identity)
        elif token is None:
            result = _Publisher(
                standing=unverified,
                self_identity=(),
                name=pod_id,
                error=(
                    "tool pod manifest carries no identity token; a copy that serves every caller must "
                    "come from a verified publisher"
                ),
                error_code=RefusalCode.UNVERIFIED_PUBLISHER,
            )
        else:
            result = await self._verified_tool_pod(pod_id, token, unverified)
        return result

    async def _verified_agent(
        self, pod_id: str, owner: UUID, token: str, agent_identity: tuple[str, ...]
    ) -> _Publisher:
        """verify an agent's in-process manifest by the agent's own token, naming the agent its pod id names.

        :param pod_id: the manifest's pod id
        :ptype pod_id: str
        :param owner: the agent the pod id names
        :ptype owner: UUID
        :param token: the agent's raw identity token
        :ptype token: str
        :param agent_identity: the agent's namespace, returned on the reply
        :ptype agent_identity: tuple[str, ...]
        :return: the verified publisher, or why it is refused
        :rtype: _Publisher
        """
        assert self._authenticator is not None  # guarded by the caller
        unverified = PublisherStanding(verified=False, platform_shared=False, owned_nodes=())
        result: _Publisher
        try:
            verified_agent = await self._authenticator.verify_agent(token)
        except Exception as exc:  # noqa: BLE001 -- the host's store failing is a refusal, never a dropped reply
            result = self._verification_unavailable(pod_id, "agent", exc)
        else:
            if verified_agent is None:
                result = _Publisher(
                    standing=unverified,
                    self_identity=(),
                    name=pod_id,
                    error="agent identity token did not verify",
                    error_code=RefusalCode.UNVERIFIED_PUBLISHER,
                )
            elif verified_agent != owner:
                result = _Publisher(
                    standing=unverified,
                    self_identity=(),
                    name=pod_id,
                    error=(
                        f"agent {verified_agent} registered under pod id {pod_id!r}, which belongs to agent "
                        f"{owner}; an agent's in-process server registers under its own agent id"
                    ),
                    error_code=RefusalCode.POD_ID_MISMATCH,
                )
            else:
                result = _Publisher(
                    standing=PublisherStanding(verified=True, platform_shared=False, owned_nodes=()),
                    self_identity=agent_identity,
                    name=f"agent {owner}",
                )
        return result

    def _unsigned_agent_publisher(self, pod_id: str, agent_identity: tuple[str, ...]) -> _Publisher:
        """the standing of an agent's UNSIGNED manifest in this release.

        :param pod_id: the agent in-process pod id
        :ptype pod_id: str
        :param agent_identity: the agent's namespace, returned on the reply
        :ptype agent_identity: tuple[str, ...]
        :return: an unverified publisher, or a refusal when the pod id has registered verified
        :rtype: _Publisher
        """
        unverified = PublisherStanding(verified=False, platform_shared=False, owned_nodes=())
        result: _Publisher
        if self._catalog.pod_has_verified_copy(pod_id):
            result = _Publisher(
                standing=unverified,
                self_identity=(),
                name=pod_id,
                error=(
                    f"pod id {pod_id!r} has registered with a verified agent identity; an unsigned manifest "
                    "under it cannot be told apart from another agent impersonating it, and is refused"
                ),
                error_code=RefusalCode.UNVERIFIED_PUBLISHER,
            )
        else:
            if pod_id not in self._unsigned_agent_pods_warned:
                self._unsigned_agent_pods_warned.add(pod_id)
                log.warning(
                    "agent in-process server registered unsigned; its copies are admitted for its own agent "
                    "only. an agent SDK that signs its registration with the agent's own identity removes "
                    "this line; 0.56.0 refuses unsigned registrations",
                    extra={"extra_data": {"pod_id": pod_id}},
                )
            result = _Publisher(standing=unverified, self_identity=agent_identity, name=pod_id)
        return result

    async def _verified_tool_pod(self, pod_id: str, token: str, unverified: PublisherStanding) -> _Publisher:
        """verify a Tool Pod's token and that it names the pod the manifest names.

        :param pod_id: the manifest's pod id
        :ptype pod_id: str
        :param token: the pod's raw token
        :ptype token: str
        :param unverified: the standing reported on a refusal
        :ptype unverified: PublisherStanding
        :return: the verified publisher, or why it is refused
        :rtype: _Publisher
        """
        assert self._authenticator is not None  # guarded by the caller
        result: _Publisher
        pod_auth: ToolPodAuth | None = None
        failure: Exception | None = None
        try:
            pod_auth = await self._authenticator.verify_pod(token)
        except Exception as exc:  # noqa: BLE001 -- the host's store failing is a refusal, never a dropped reply
            failure = exc
        if failure is not None:
            result = self._verification_unavailable(pod_id, "tool pod", failure)
        elif pod_auth is None:
            log.warning(
                "tool pod registration rejected: invalid token",
                extra={"extra_data": {"pod_id": pod_id}},
            )
            result = _Publisher(
                standing=unverified,
                self_identity=(),
                name=pod_id,
                error="invalid bootstrap token",
                error_code=RefusalCode.UNVERIFIED_PUBLISHER,
            )
        elif pod_auth.pod_entity_id != pod_id:
            result = _Publisher(
                standing=unverified,
                self_identity=(),
                name=pod_auth.name,
                error=(
                    f"tool pod {pod_auth.pod_entity_id} ({pod_auth.name}) registered under pod id {pod_id!r}; "
                    "a verified pod registers under its own id"
                ),
                error_code=RefusalCode.POD_ID_MISMATCH,
            )
        else:
            owned_nodes = _provider_node_names(pod_auth.owned_namespaces, pod_id)
            result = _Publisher(
                standing=PublisherStanding(
                    verified=True,
                    platform_shared=pod_auth.platform_shared,
                    owned_nodes=owned_nodes,
                ),
                self_identity=owned_nodes,
                name=pod_auth.name,
            )
        return result

    @staticmethod
    def _verification_unavailable(pod_id: str, kind: str, exc: Exception) -> _Publisher:
        """the refusal for a publisher the host could not verify because its store failed.

        The authenticator's contract is to answer ``None`` for every verification FAILURE; raising
        means the check never ran -- a store read the host's broker refused, a database outage.
        Left to escape, the exception killed the subscription callback before any reply, and the
        publisher saw only its own request time out. This is the one ERROR line naming the cause.

        :param pod_id: the manifest's pod id
        :ptype pod_id: str
        :param kind: which verifier failed (``"agent"`` or ``"tool pod"``), for the log
        :ptype kind: str
        :param exc: what the authenticator raised
        :ptype exc: Exception
        :return: an unverified publisher refused with ``PUBLISHER_VERIFICATION_UNAVAILABLE``
        :rtype: _Publisher
        """
        log.error(
            "tool registration refused: the host could not verify the publisher, because reading the store "
            "it verifies against failed. the pod is told the refusal is temporary and retries on its heartbeat",
            extra={
                "extra_data": {
                    "pod_id": pod_id,
                    "publisher_kind": kind,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "error_code": RefusalCode.PUBLISHER_VERIFICATION_UNAVAILABLE.value,
                }
            },
        )
        return _Publisher(
            standing=PublisherStanding(verified=False, platform_shared=False, owned_nodes=()),
            self_identity=(),
            name=pod_id,
            error=(f"{_REFUSAL_REASONS[RefusalCode.PUBLISHER_VERIFICATION_UNAVAILABLE]} ({type(exc).__name__}: {exc})"),
            error_code=RefusalCode.PUBLISHER_VERIFICATION_UNAVAILABLE,
        )

    async def _answer_catalog_unavailable(
        self,
        msg: IncomingMessage,
        manifest: RegistrationManifest,
        verdict: _Verdict,
        exc: Exception,
    ) -> None:
        """answer a manifest whose verdict the catalog could not record.

        Every tool the verdict admitted is refused with ``CATALOG_UNAVAILABLE``; every tool it
        refused keeps its own code, so a final refusal stays final to the pod. Registration is
        idempotent, so the pod's next heartbeat re-offering the manifest completes whatever part of
        the write did not land.

        :param msg: the incoming manifest message
        :ptype msg: IncomingMessage
        :param manifest: the manifest
        :ptype manifest: RegistrationManifest
        :param verdict: what the manifest was judged to be allowed
        :ptype verdict: _Verdict
        :param exc: what the catalog raised
        :ptype exc: Exception
        :return: nothing
        :rtype: None
        """
        code = RefusalCode.CATALOG_UNAVAILABLE
        log.error(
            "tool registration could not be recorded: the catalog write failed. the pod is told the refusal "
            "is temporary and retries on its heartbeat",
            extra={
                "extra_data": {
                    "pod_id": manifest.pod_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "error_code": code.value,
                }
            },
        )
        await self._reply(
            msg,
            RegistrationResponse(
                success=False,
                pod_id=manifest.pod_id,
                refused_tools=[*verdict.refused, *_refused(verdict.admitted, code)],
                error=f"{_REFUSAL_REASONS[code]} ({type(exc).__name__}: {exc})",
                error_code=code.value,
            ),
        )

    async def _withdraw_refused_copies(self, pod_id: str, verdict: _Verdict) -> None:
        """remove a VERIFIED publisher's prior copy of every tool it was just refused.

        Re-registration goes through admission every time, so a pod that no longer owns a node,
        or is no longer the platform, is refused tools it held a copy of. That copy goes; every
        other pod's copy of the same tool stays. An unverified manifest withdraws nothing: it has
        not proved it IS the pod whose copy it would take. Nor does a refusal for a transient
        reason (:data:`_WITHDRAWING_CODES` lists the ones that withdraw).

        :param pod_id: the publishing pod
        :ptype pod_id: str
        :param verdict: the manifest's verdict
        :ptype verdict: _Verdict
        :return: nothing
        :rtype: None
        """
        if not verdict.standing.verified:
            return
        for tool in verdict.refused:
            if tool.code in _WITHDRAWING_CODES:
                await self._catalog.remove_copy(f"{tool.name}@{tool.version}", pod_id)

    async def _provider_node_directory(self, pod_id: str) -> tuple[str, ...] | None:
        """every provider node the host's graph holds, or ``None`` when it cannot be read.

        A handler with no authenticator has no host to ask, and answers with an
        EMPTY inventory rather than a failure: open mode enforces nothing.

        A read that FAILS is a different thing and is not allowed to look like the
        first one. An empty inventory silently widens every pod, so a host that
        raises gets the registration REFUSED. That is the recoverable direction: a
        pod re-announces on its heartbeat, so a transient failure costs one
        interval, while admitting the manifest would write catalog entries nobody
        can take back.

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

    def _validate_manifest(self, manifest: RegistrationManifest) -> str | None:
        """validate registration manifest fields.

        :param manifest: manifest to validate
        :ptype manifest: RegistrationManifest
        :return: error message if validation fails, None if valid
        :rtype: str | None
        """
        if not manifest.pod_id:
            return "pod_id is required"
        try:
            # routing reads an endpoint's owner from its pod-id, so an id that names no owner could
            # never be routed. refused here, loudly, rather than admitted as an endpoint that sits
            # in the catalog routable by no caller.
            Subjects.agent_inprocess_owner_id(manifest.pod_id)
        except ValueError as exc:
            return (
                f"pod_id cannot be routed: {exc}. a Tool Pod's id is one token; an agent's in-process "
                "server's id is Subjects.agent_inprocess_pod_id(agent_id, instance)"
            )
        if not manifest.tools:
            return "tools list is required and must not be empty"
        result = None
        return result

    async def _register_tools(
        self,
        pod_id: str,
        tools: list[ToolManifestEntry],
        standing: PublisherStanding,
    ) -> list[str]:
        """register each admitted tool as this pod's copy, carrying the definition it announced.

        creates catalog entry for each tool with a single endpoint
        for the registering pod. catalog.register() folds it into
        that pod's copy and no other. a brand-new copy is parked in
        the 'pending' state; after all tools are written, issues
        a reachability probe to the pod; on successful round-trip
        promotes every pending endpoint for the pod to 'available'
        via ``catalog.mark_ready``. on probe failure, endpoints
        remain pending so routing refuses to forward until the
        next heartbeat can retry promotion.

        :param pod_id: the registering pod
        :ptype pod_id: str
        :param tools: the admitted tools
        :ptype tools: list[ToolManifestEntry]
        :param standing: the publisher's verified standing, recorded on each copy
        :ptype standing: PublisherStanding
        :return: list of full_name values registered
        :rtype: list[str]
        """
        registered: list[str] = []
        needs_probe = False
        now = datetime.now(UTC)
        for tool in tools:
            full_name = f"{tool.name}@{tool.version}"
            existing_entry = self._catalog.get(full_name)
            existing_endpoint = existing_entry.get_endpoint(pod_id) if existing_entry is not None else None
            # Preserve status for endpoints the pod has previously registered
            # so heartbeat-driven re-publication does not regress an already
            # 'available' endpoint back to 'pending' (which would trigger a
            # needless re-probe on every heartbeat). A brand-new endpoint
            # enters 'pending' and drives exactly one probe round-trip. The
            # re-registration itself passed admission like any other.
            if existing_endpoint is None:
                endpoint_status = "pending"
                needs_probe = True
            else:
                endpoint_status = existing_endpoint.status
            endpoint = ToolEndpoint(
                pod_id=pod_id,
                status=endpoint_status,
                date_last_heartbeat=now,
                verified_publisher=standing.verified,
            )
            # output_schema is not carried on the manifest today, so every copy announces none.
            endpoint.announce(
                ToolDefinition(
                    description=tool.description,
                    input_schema=tool.input_schema,
                    output_schema=None,
                    timeout_seconds=tool.timeout_seconds,
                    requires_confirmation=tool.requires_confirmation,
                ),
                now,
            )
            entry = CatalogEntry(
                tool_name=tool.name,
                tool_version=tool.version,
                full_name=full_name,
                endpoints=[endpoint],
                date_registered=now,
            )
            await self._catalog.register(entry)
            registered.append(full_name)

        if needs_probe:
            await self._probe_and_promote(pod_id)

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
