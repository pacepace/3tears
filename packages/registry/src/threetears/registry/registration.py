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

import hashlib
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from threetears.agent.tools.server import RegistrationManifest
from threetears.observe import get_logger
from threetears.registry.auth import ToolPodAuth, ToolPodAuthenticator
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint

__all__ = [
    "ProbeRequest",
    "ProbeResponse",
    "RegistrationHandler",
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


class RegistrationResponse(BaseModel):
    """response sent back to registering pod.

    :param success: whether registration succeeded
    :ptype success: bool
    :param pod_id: identifier of pod that attempted registration
    :ptype pod_id: str
    :param registered_tools: list of full_name values successfully registered
    :ptype registered_tools: list[str]
    :param error: error message if registration failed
    :ptype error: str | None
    """

    success: bool
    pod_id: str
    registered_tools: list[str] = []
    error: str | None = None


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
        namespace: str = "aibots",
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
        self._nc: Any | None = None
        self._sub: Any | None = None

    async def start(self, nc: Any) -> None:
        """start listening for registration requests.

        :param nc: connected NATS client
        :ptype nc: Any
        """
        self._nc = nc
        subject = f"{self._namespace}.tools.register"
        self._sub = await nc.subscribe(subject, cb=self.handle_registration)
        log.info(
            "registration handler started",
            extra={"extra_data": {"subject": subject}},
        )

    async def stop(self) -> None:
        """stop listening for registration requests."""
        if self._sub is not None:
            await self._sub.unsubscribe()
            self._sub = None
        log.info("registration handler stopped")

    async def handle_registration(self, msg: Any) -> None:
        """public NATS-subject handler for incoming registration manifest.

        bound by :meth:`start` as the ``cb`` callback on
        ``{namespace}.tools.register`` so every registering tool pod's
        manifest arrives here. tests exercise this surface directly by
        synthesizing a NATS message and awaiting the handler; keeping
        the entry point public is a stability contract -- subclasses and
        test doubles may rely on the name, the single ``msg`` parameter,
        and the absence of return value.

        validates manifest, authenticates pod, and registers
        tools with additive endpoint merging. replies with
        success or error response.

        :param msg: incoming NATS message containing registration manifest
        :ptype msg: Any
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
            if msg.reply:
                await self._nc.publish(
                    msg.reply,
                    response.model_dump_json().encode("utf-8"),
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
            if msg.reply:
                await self._nc.publish(
                    msg.reply,
                    response.model_dump_json().encode("utf-8"),
                )
            return

        auth_error = await self._authenticate_and_filter(manifest)
        if auth_error is not None:
            log.warning(
                "registration rejected: auth failed",
                extra={"extra_data": {"pod_id": manifest.pod_id, "error": auth_error}},
            )
            response = RegistrationResponse(
                success=False,
                pod_id=manifest.pod_id,
                error=auth_error,
            )
            if msg.reply:
                await self._nc.publish(
                    msg.reply,
                    response.model_dump_json().encode("utf-8"),
                )
            return

        registered = await self._register_tools(manifest)

        response = RegistrationResponse(
            success=True,
            pod_id=manifest.pod_id,
            registered_tools=registered,
        )
        if msg.reply:
            await self._nc.publish(
                msg.reply,
                response.model_dump_json().encode("utf-8"),
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

    async def _authenticate_and_filter(self, manifest: RegistrationManifest) -> str | None:
        """authenticate tool pod and filter tools to allowed namespaces.

        if no authenticator is configured, all tools are allowed (open mode).
        if authenticator is configured, verifies bootstrap token and filters
        tools to only those within the pod's allowed_namespaces.

        :param manifest: registration manifest to authenticate and filter
        :ptype manifest: RegistrationManifest
        :return: error message if authentication fails, None if successful
        :rtype: str | None
        """
        if self._authenticator is None:
            return None

        if manifest.bootstrap_token is None:
            return "bootstrap_token required for registration"

        token_hash = hashlib.sha256(manifest.bootstrap_token.encode("utf-8")).hexdigest()
        pod_auth: ToolPodAuth | None = await self._authenticator.verify_pod(token_hash)

        if pod_auth is None:
            log.warning(
                "tool pod registration rejected: invalid token",
                extra={"extra_data": {"pod_id": manifest.pod_id}},
            )
            return "invalid bootstrap token"

        allowed_tools = []
        rejected_tools = []
        for tool in manifest.tools:
            authorized = False
            for ns in pod_auth.allowed_namespaces:
                if tool.name.startswith(ns):
                    authorized = True
                    break
            if authorized:
                allowed_tools.append(tool)
            else:
                rejected_tools.append(tool.name)

        if rejected_tools:
            log.warning(
                "tool pod tools rejected (outside allowed namespaces)",
                extra={
                    "extra_data": {
                        "pod_id": manifest.pod_id,
                        "pod_name": pod_auth.name,
                        "rejected": rejected_tools,
                        "allowed_namespaces": pod_auth.allowed_namespaces,
                    }
                },
            )

        if not allowed_tools:
            return "no tools authorized for this pod's namespaces"

        manifest.tools = allowed_tools

        log.info(
            "tool pod authenticated",
            extra={
                "extra_data": {
                    "pod_id": manifest.pod_id,
                    "pod_name": pod_auth.name,
                    "tools_accepted": len(allowed_tools),
                    "tools_rejected": len(rejected_tools),
                }
            },
        )
        result: str | None = None
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
        subject = f"{self._namespace}.tools.probe.{pod_id}"
        payload = ProbeRequest(pod_id=pod_id).model_dump_json().encode("utf-8")
        start = datetime.now(UTC)
        try:
            reply = await self._nc.request(
                subject,
                payload,
                timeout=self._probe_timeout,
            )
        except Exception as exc:
            log.warning(
                "tool pod reachability probe failed; endpoints remain pending",
                extra={
                    "extra_data": {
                        "pod_id": pod_id,
                        "probe_subject": subject,
                        "probe_timeout": self._probe_timeout,
                        "error": str(exc),
                    }
                },
            )
            return
        try:
            ack = ProbeResponse.model_validate_json(reply.data)
        except Exception as exc:
            log.warning(
                "tool pod probe reply was malformed; endpoints remain pending",
                extra={
                    "extra_data": {
                        "pod_id": pod_id,
                        "probe_subject": subject,
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
                        "probe_subject": subject,
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
