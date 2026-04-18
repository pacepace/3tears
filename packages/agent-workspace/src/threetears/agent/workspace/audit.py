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
- ``actor_user_id`` -- the invoking user's UUID (serialized as string);
  WS-ACL-10 renamed this from ``actor_id``; every call site was updated
  in the same PR and any stray ``actor_id=`` kwarg raises a pydantic
  validation error naming ``actor_user_id`` so the mistake is loud
- ``agent_id`` -- the agent envelope carrier for backwards contract
  shape; duplicates ``calling_agent_id`` when the call is not cross-
  agent. retained so existing indexes on ``audit_events.agent_id``
  continue to resolve
- ``calling_agent_id`` -- agent whose process ran the tool (the
  ``scope.context.agent_id``). identical to ``owner_agent_id`` for
  owner-path calls; differs under cross-agent sharing
- ``owner_agent_id`` -- owning agent from the workspace's paired
  ``platform.namespaces`` row; the answer to "whose data is this?"
- ``customer_id`` -- owning customer, sourced from the scope and
  verified against the workspace/namespace row
- ``namespace_id`` -- the workspace id (shared PK with the namespace
  post-WS-ACL-03)
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

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

    ``model_config`` uses ``extra='forbid'`` -- any stray field on the
    wire fails the parse loudly. Combined with the ``model_validator``
    that rewrites the legacy ``actor_id`` key to a clear error naming
    ``actor_user_id``, a miswired publisher surfaces at the construction
    site rather than silently dropping the field.

    :param event_type: fully-qualified event type string
    :ptype event_type: str
    :param actor_type: short taxonomy tag, ``"agent"`` for agent events
    :ptype actor_type: str
    :param actor_user_id: identifier of the user whose action ran the
        tool; the load-bearing "who did this?" field for compliance
    :ptype actor_user_id: UUID
    :param agent_id: identifier of agent attached to the envelope;
        duplicates :attr:`calling_agent_id` for owner-path calls and is
        retained so :attr:`platform_audit.audit_events.agent_id`
        indexes continue to resolve against agent-scoped queries
    :ptype agent_id: UUID
    :param calling_agent_id: agent whose process ran the tool (the
        ``scope.context.agent_id``); matches :attr:`owner_agent_id`
        unless the workspace was accessed under a cross-agent grant
    :ptype calling_agent_id: UUID
    :param owner_agent_id: owning agent for the workspace's paired
        namespace row; the authoritative "whose data is this?" answer
    :ptype owner_agent_id: UUID
    :param customer_id: owning-customer UUID, sourced from the call
        scope and verified equal to the workspace's customer
    :ptype customer_id: UUID
    :param namespace_id: workspace id / paired-namespace id (they share
        a primary key post-WS-ACL-03)
    :ptype namespace_id: UUID
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

    model_config = ConfigDict(extra="forbid")

    event_type: str
    actor_type: str = "agent"
    actor_user_id: UUID
    agent_id: UUID
    calling_agent_id: UUID
    owner_agent_id: UUID
    customer_id: UUID
    namespace_id: UUID
    resource_type: str
    resource_id: str
    action: str
    details: dict[str, Any] = Field(default_factory=dict)
    correlation_id: UUID
    timestamp: datetime

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_actor_id(cls, data: Any) -> Any:
        """reject the legacy ``actor_id`` kwarg with a clear rename message.

        WS-ACL-10 renamed ``actor_id`` to ``actor_user_id``. we ship no
        back-compat shim (NO BACKWARDS-COMPATIBILITY SHIMS, per
        ``CLAUDE.md``): every construction site updated in the same
        commit. this validator exists so any stray caller fails LOUDLY
        with an error naming the new field, rather than silently
        dropping the argument under ``extra='ignore'`` or being
        accepted then unused.

        :param data: raw kwargs or dict passed to the model
        :ptype data: Any
        :return: the same data unchanged when no legacy key is present
        :rtype: Any
        :raises ValueError: when ``actor_id`` is present; message names
            ``actor_user_id`` so the caller knows exactly what to
            rename
        """
        if isinstance(data, dict) and "actor_id" in data:
            raise ValueError(
                "WorkspaceAuditEnvelope: 'actor_id' was renamed to "
                "'actor_user_id' (WS-ACL-10). update the call site to "
                "pass actor_user_id=<UUID>; no back-compat shim ships."
            )
        return data


async def publish_workspace_event(
    *,
    nats_client: Any,
    namespace: str,
    event_type: str,
    actor_user_id: UUID,
    agent_id: UUID,
    calling_agent_id: UUID,
    owner_agent_id: UUID,
    customer_id: UUID,
    namespace_id: UUID,
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
    :param actor_user_id: identifier of the invoking user; the
        compliance-visible "who did this?" answer
    :ptype actor_user_id: UUID
    :param agent_id: identifier of agent attached to the envelope for
        index-resolution compatibility; typically equal to
        ``calling_agent_id``
    :ptype agent_id: UUID
    :param calling_agent_id: agent whose process ran the tool (the
        scope's ``agent_id``); differs from ``owner_agent_id`` under
        cross-agent sharing
    :ptype calling_agent_id: UUID
    :param owner_agent_id: owner agent on the workspace's paired
        namespace row
    :ptype owner_agent_id: UUID
    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :param namespace_id: workspace id (shared PK with namespace)
    :ptype namespace_id: UUID
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
        actor_user_id=actor_user_id,
        agent_id=agent_id,
        calling_agent_id=calling_agent_id,
        owner_agent_id=owner_agent_id,
        customer_id=customer_id,
        namespace_id=namespace_id,
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
