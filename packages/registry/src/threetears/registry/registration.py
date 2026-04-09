"""registration handler for tool pod manifests.

subscribes to NATS registration subject, validates incoming
manifests, authenticates pods, and registers tools with
additive endpoint merging for multi-pod horizontal scaling.
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

log = get_logger(__name__)


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
    ) -> None:
        """initialize registration handler.

        :param catalog: tool catalog to register tools into
        :ptype catalog: ToolCatalog
        :param namespace: NATS subject namespace prefix
        :ptype namespace: str
        :param authenticator: optional tool pod authenticator for token verification
        :ptype authenticator: ToolPodAuthenticator | None
        """
        self._catalog = catalog
        self._namespace = namespace
        self._authenticator = authenticator
        self._nc: Any | None = None
        self._sub: Any | None = None

    async def start(self, nc: Any) -> None:
        """start listening for registration requests.

        :param nc: connected NATS client
        :ptype nc: Any
        """
        self._nc = nc
        subject = f"{self._namespace}.tools.register"
        self._sub = await nc.subscribe(subject, cb=self._handle_registration)
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

    async def _handle_registration(self, msg: Any) -> None:
        """handle incoming registration manifest.

        validates manifest, authenticates pod, and registers
        tools with additive endpoint merging. replies with
        success or error response.

        :param msg: incoming NATS message containing registration manifest
        :ptype msg: Any
        """
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
            extra={"extra_data": {
                "pod_id": manifest.pod_id,
                "tools_count": len(registered),
            }},
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
                extra={"extra_data": {
                    "pod_id": manifest.pod_id,
                    "pod_name": pod_auth.name,
                    "rejected": rejected_tools,
                    "allowed_namespaces": pod_auth.allowed_namespaces,
                }},
            )

        if not allowed_tools:
            return "no tools authorized for this pod's namespaces"

        manifest.tools = allowed_tools

        log.info(
            "tool pod authenticated",
            extra={"extra_data": {
                "pod_id": manifest.pod_id,
                "pod_name": pod_auth.name,
                "tools_accepted": len(allowed_tools),
                "tools_rejected": len(rejected_tools),
            }},
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
        """register all tools from manifest with endpoint for this pod.

        creates catalog entry for each tool with a single endpoint
        for the registering pod. catalog.register() handles merging
        with existing entries for multi-pod support.

        :param manifest: validated manifest containing tools to register
        :ptype manifest: RegistrationManifest
        :return: list of full_name values registered
        :rtype: list[str]
        """
        registered: list[str] = []
        now = datetime.now(UTC)
        for tool in manifest.tools:
            full_name = f"{tool.name}@{tool.version}"
            endpoint = ToolEndpoint(
                pod_id=manifest.pod_id,
                status="available",
                in_flight=0,
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
        result = registered
        return result
