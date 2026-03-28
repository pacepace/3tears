"""registration handler for tool pod manifests.

subscribes to NATS registration subject, validates incoming
manifests, checks for conflicts, and registers tools atomically
in catalog.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from threetears.agent.tools.server import RegistrationManifest
from threetears.core.logging import get_logger
from threetears.registry.catalog import CatalogEntry, ToolCatalog

_logger = get_logger(__name__)


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
    detects conflicts, and registers tools in catalog. supports
    idempotent re-registration from same pod.
    """

    def __init__(self, catalog: ToolCatalog, namespace: str = "aibots") -> None:
        """initialize registration handler.

        :param catalog: tool catalog to register tools into
        :ptype catalog: ToolCatalog
        :param namespace: NATS subject namespace prefix
        :ptype namespace: str
        """
        self._catalog = catalog
        self._namespace = namespace
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
        _logger.info(
            "registration handler started",
            extra={"extra_data": {"subject": subject}},
        )

    async def stop(self) -> None:
        """stop listening for registration requests."""
        if self._sub is not None:
            await self._sub.unsubscribe()
            self._sub = None
        _logger.info("registration handler stopped")

    async def _handle_registration(self, msg: Any) -> None:
        """handle incoming registration manifest.

        validates manifest, checks for conflicts, and registers
        tools atomically. replies with success or error response.

        :param msg: incoming NATS message containing registration manifest
        :ptype msg: Any
        """
        try:
            manifest = RegistrationManifest.model_validate_json(msg.data)
        except Exception as exc:
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

        conflict_error = self._check_conflicts(manifest)
        if conflict_error is not None:
            response = RegistrationResponse(
                success=False,
                pod_id=manifest.pod_id,
                error=conflict_error,
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
        _logger.info(
            "registration completed",
            extra={"extra_data": {
                "pod_id": manifest.pod_id,
                "tools_count": len(registered),
            }},
        )

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

    def _check_conflicts(self, manifest: RegistrationManifest) -> str | None:
        """check for tool name@version conflicts with different pods.

        same name@version from same pod is allowed (re-registration).
        same name@version from different pod is conflict.

        :param manifest: manifest to check for conflicts
        :ptype manifest: RegistrationManifest
        :return: error message if conflict found, None if no conflicts
        :rtype: str | None
        """
        result: str | None = None
        for tool in manifest.tools:
            full_name = f"{tool.name}@{tool.version}"
            existing = self._catalog.get(full_name)
            if existing is not None and existing.pod_id != manifest.pod_id:
                result = (
                    f"conflict: {full_name} already registered by pod "
                    f"{existing.pod_id}"
                )
                break
        return result

    async def _register_tools(
        self,
        manifest: RegistrationManifest,
    ) -> list[str]:
        """register all tools from manifest atomically.

        :param manifest: validated manifest containing tools to register
        :ptype manifest: RegistrationManifest
        :return: list of full_name values registered
        :rtype: list[str]
        """
        registered: list[str] = []
        now = datetime.now(UTC)
        for tool in manifest.tools:
            full_name = f"{tool.name}@{tool.version}"
            entry = CatalogEntry(
                tool_name=tool.name,
                tool_version=tool.version,
                full_name=full_name,
                pod_id=manifest.pod_id,
                description=tool.description,
                input_schema=tool.input_schema,
                status="available",
                date_registered=now,
                date_last_heartbeat=now,
            )
            await self._catalog.register(entry)
            registered.append(full_name)
        result = registered
        return result
