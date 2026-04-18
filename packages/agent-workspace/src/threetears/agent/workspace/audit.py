"""agent-side audit publish helper for workspace mutations.

exposes :class:`WorkspaceAuditEnvelope` (the canonical pydantic
wire-format model) and :func:`publish_workspace_event`, a fire-and-forget
helper that builds the envelope and publishes it on
``{namespace}.audit.workspace.{action}``. failures log at WARN but never
raise -- tool-call success must never depend on audit infrastructure
availability. the Hub-side consumer
(``aibots.hub.workspace.audit_consumer.WorkspaceAuditConsumer``) owns
L3 persistence; agents never touch ``platform_audit.audit_events``
directly.

envelope schema (enforced by :class:`WorkspaceAuditEnvelope`):

- ``event_type`` -- fully-qualified event type (``workspace.fs_write`` etc.)
- ``actor_type`` -- always ``"agent"`` for agent-initiated events
- ``actor_id`` -- the originating actor's UUID (serialized as string)
- ``agent_id`` -- owning agent's UUID (serialized as string)
- ``resource_type`` -- ``"workspace_file"`` or ``"workspace"``
- ``resource_id`` -- path string (``{workspace_id}/{relative_path}`` for
  file ops, ``{workspace_id}`` for lifecycle ops)
- ``action`` -- short verb (``write`` / ``edit`` / ``set`` / ``merge`` /
  ``create`` / ``reset`` / ``delete`` / ``rollback_to`` / ``bind``)
- ``details`` -- operation-specific mapping carrying sha/version/bytes/etc.
- ``correlation_id`` -- tool-call correlation UUID (serialized as string)
- ``timestamp`` -- timezone-aware UTC datetime (serialized ISO-8601)

using pydantic at the wire boundary keeps the agent and Hub-side
consumer in a single typed contract: both sides import this model so
drift is a compile-time problem rather than a runtime one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from threetears.observe import get_logger

__all__ = [
    "WorkspaceAuditEnvelope",
    "publish_workspace_event",
]

log = get_logger(__name__)


class WorkspaceAuditEnvelope(BaseModel):
    """typed wire-format envelope for workspace audit events.

    built on the agent side by :func:`publish_workspace_event`, decoded
    on the Hub side by
    :class:`aibots.hub.workspace.audit_consumer.WorkspaceAuditConsumer`.
    pydantic handles UUID -> string and datetime -> ISO-8601 coercion at
    :meth:`model_dump_json` automatically, and validates types on
    :meth:`model_validate_json` so both sides catch drift early.

    ``model_config`` uses ``extra='ignore'`` so future envelope
    extensions do not break older Hub consumers -- forward-compat.

    :param event_type: fully-qualified event type string
    :ptype event_type: str
    :param actor_type: short taxonomy tag, ``"agent"`` for agent events
    :ptype actor_type: str
    :param actor_id: identifier of actor performing the action
    :ptype actor_id: UUID
    :param agent_id: identifier of owning agent
    :ptype agent_id: UUID
    :param resource_type: short taxonomy tag, ``"workspace_file"`` or
        ``"workspace"``
    :ptype resource_type: str
    :param resource_id: stable path-like identifier for the resource
    :ptype resource_id: str
    :param action: short verb; also the last segment of the NATS subject
    :ptype action: str
    :param details: operation-specific extras (sha256 before/after,
        version, bytes, jsonpath, template_name, files_changed, etc.)
    :ptype details: dict[str, Any]
    :param correlation_id: originating tool-call correlation UUID
    :ptype correlation_id: UUID
    :param timestamp: timezone-aware UTC instant when envelope was built
    :ptype timestamp: datetime
    """

    model_config = ConfigDict(extra="ignore")

    event_type: str
    actor_type: str = "agent"
    actor_id: UUID
    agent_id: UUID
    resource_type: str
    resource_id: str
    action: str
    details: dict[str, Any] = Field(default_factory=dict)
    correlation_id: UUID
    timestamp: datetime


async def publish_workspace_event(
    *,
    nats_client: Any,
    namespace: str,
    event_type: str,
    actor_id: UUID,
    agent_id: UUID,
    resource_type: str,
    resource_id: str,
    action: str,
    details: dict[str, Any],
    correlation_id: UUID,
) -> None:
    """
    publish workspace audit envelope on ``{namespace}.audit.workspace.{action}``.

    fire-and-forget: the helper builds a typed
    :class:`WorkspaceAuditEnvelope`, serializes it via
    :meth:`BaseModel.model_dump_json`, and awaits one
    ``nats_client.publish``. any exception during publish is caught and
    logged at WARN; no exception propagates to the caller. this
    invariant is load-bearing: tool-call success must not depend on
    audit infrastructure being healthy.

    :param nats_client: connected NATS client exposing ``publish(subject,
        payload)``; when ``None``, the call is a no-op (useful in tests
        and bootstrap windows before wiring is complete)
    :ptype nats_client: Any
    :param namespace: NATS subject namespace (environment-scoped prefix)
    :ptype namespace: str
    :param event_type: fully-qualified event type string, e.g.
        ``"workspace.fs_write"``
    :ptype event_type: str
    :param actor_id: identifier of actor performing the action; for
        agent-initiated events equals ``agent_id``
    :ptype actor_id: UUID
    :param agent_id: identifier of owning agent
    :ptype agent_id: UUID
    :param resource_type: short taxonomy tag, ``"workspace_file"`` or
        ``"workspace"``
    :ptype resource_type: str
    :param resource_id: stable path-like identifier for the resource
    :ptype resource_id: str
    :param action: short verb; also the last segment of the NATS subject
    :ptype action: str
    :param details: operation-specific extras (sha256 before/after,
        version, bytes, jsonpath, template_name, files_changed, etc.)
    :ptype details: dict[str, Any]
    :param correlation_id: originating tool-call correlation UUID
    :ptype correlation_id: UUID
    :return: nothing
    :rtype: None
    """
    if nats_client is None:
        # bootstrap / test scenario; explicit no-op
        return
    envelope = WorkspaceAuditEnvelope(
        event_type=event_type,
        actor_type="agent",
        actor_id=actor_id,
        agent_id=agent_id,
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
        details=details,
        correlation_id=correlation_id,
        timestamp=datetime.now(UTC),
    )
    payload = envelope.model_dump_json().encode("utf-8")
    subject = f"{namespace}.audit.workspace.{action}"
    try:
        await nats_client.publish(subject, payload)
    # NOSILENT: audit infrastructure failures are logged at WARN; tool
    # success must not depend on audit path - callers continue normally.
    except Exception as exc:
        log.warning(
            "workspace audit publish failed",
            extra={
                "extra_data": {
                    "subject": subject,
                    "event_type": event_type,
                    "error": str(exc),
                },
            },
        )
