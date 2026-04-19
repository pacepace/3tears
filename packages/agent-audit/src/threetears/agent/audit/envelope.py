"""unified audit envelope for every domain on the 3tears platform.

the pre-audit-task-01 world had per-domain envelope types
(:class:`WorkspaceAuditEnvelope`, :class:`RbacAuditEnvelope`, etc.)
each on a different NATS subject with a slightly different wire shape.
cross-domain audit queries paid for this with UNIONs across lightly-
different schemas and the admin query API had to special-case each
domain. audit-task-01 collapses all of it into one envelope, one
publish helper, one subject tree, one consumer, one table.

the envelope below is the single wire contract. every domain
(:mod:`threetears.agent.workspace.tools.*`, hub rbac endpoints, agent
memory, user tools) builds an :class:`AuditEvent` and hands it to
:func:`publish_audit`. the hub-side ``unified_audit_consumer``
subscribes to ``{namespace}.audit.>`` and persists to
``platform_audit.audit_events``.

anti-drift guarantees:

- ``model_config`` uses ``extra='forbid'`` -- any stray field on the
  wire fails parse loudly. a miswired publisher surfaces at construct
  time rather than silently dropping the field.
- ``timestamp`` must be timezone-aware UTC; a validator rejects naive
  datetimes with a clear message.
- ``event_type`` uses the dotted ``{domain}.{verb}[.{sub_verb}]``
  convention (e.g. ``workspace.fs_write``, ``rbac.assignment.create``,
  ``rbac.introspect.explain``). it is an open ``str`` rather than a
  closed ``Literal``: the envelope intentionally does not gate new
  event types at the wire layer because new domains (memory, custom
  tools) must be able to publish without editing this module. per-
  domain closed enums belong at the producer site, not on the shared
  envelope.
- ``details: dict[str, Any]`` is the escape hatch for event-type-
  specific extras. commonly-queried fields (actor, resource, action,
  outcome) are typed columns so admin queries don't JSONB-parse on
  every row.

identity axes on the envelope:

- ``actor_user_id`` -- the human (or service account) whose action
  triggered the event; the load-bearing "who did this?" field.
- ``calling_agent_id`` -- the agent whose pod ran the tool. equal to
  ``owner_agent_id`` for owner-path calls; differs under cross-agent
  sharing.
- ``owner_agent_id`` -- the agent that owns the resource being
  touched. sourced from the resource's namespace row.
- ``customer_id`` -- the owning customer. ``None`` only for
  platform-scoped events (e.g. platform role CRUD).
- ``resource_namespace_id`` -- the resource's namespace row id. every
  resource on the platform is a namespace post-workspace-task-19, so
  one column answers "whose data?" across every domain.
- ``resource_namespace_type`` -- short taxonomy tag (``workspace`` /
  ``group`` / ``role`` / ``assignment`` / ``introspection`` / ...).
  matches the ``namespaces.namespace_type`` taxonomy.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["AuditEvent"]


class AuditEvent(BaseModel):
    """typed wire-format envelope for every platform audit event.

    single canonical envelope shared by every producing domain and
    the hub-side ``unified_audit_consumer``. built on the producer
    side (workspace tool, rbac endpoint, memory subsystem, custom
    tool) and handed to :func:`publish_audit`. decoded on the hub
    side from ``{namespace}.audit.>`` by the unified consumer.

    :param id: envelope identity (uuid7 time-ordered); doubles as the
        audit row primary key when the consumer persists it
    :ptype id: UUID
    :param timestamp: timezone-aware UTC instant the event occurred
    :ptype timestamp: datetime
    :param event_type: dotted event family, e.g. ``workspace.fs_write``,
        ``rbac.assignment.create``, ``rbac.introspect.explain``. used
        as the last segment of the NATS subject and as the primary
        ``event_type`` column on ``audit_events``
    :ptype event_type: str
    :param actor_user_id: human / service-account actor; ``None`` for
        system-triggered events (e.g. scheduled retention)
    :ptype actor_user_id: UUID | None
    :param calling_agent_id: agent whose pod ran the tool; ``None`` for
        events emitted from the hub directly (e.g. rbac admin endpoints)
    :ptype calling_agent_id: UUID | None
    :param owner_agent_id: agent that owns the resource being touched;
        ``None`` for platform-scoped events
    :ptype owner_agent_id: UUID | None
    :param customer_id: owning customer; ``None`` only for platform-
        scoped events (platform role CRUD, platform-scoped groups)
    :ptype customer_id: UUID | None
    :param resource_namespace_id: namespace row id of the resource;
        ``None`` for events whose subject is not a single namespace
        (introspection queries, cross-resource dry-runs)
    :ptype resource_namespace_id: UUID | None
    :param resource_namespace_type: short taxonomy tag matching
        ``namespaces.namespace_type`` (``workspace`` / ``group`` /
        ``role`` / ``assignment`` / ``introspection`` / ...). ``None``
        when ``resource_namespace_id`` is ``None``
    :ptype resource_namespace_type: str | None
    :param action: short verb (``read`` / ``write`` / ``create`` /
        ``update`` / ``delete`` / ``add_member`` / ``remove_member`` /
        ``explain`` / ...). kept open-set so new domains don't edit
        the envelope module
    :ptype action: str
    :param outcome: one of ``success``, ``failure``, ``error``. tool-
        dispatch auto-emission stamps this based on the handler return
        path; explicit emissions pass ``success`` by convention unless
        they know otherwise
    :ptype outcome: str
    :param correlation_id: request / tool-call correlation UUID; used
        with ``event_type`` as the idempotency key for at-least-once
        NATS redelivery
    :ptype correlation_id: UUID
    :param details: event-type-specific structured payload (sha/bytes/
        version for fs_write; added_member_id for group.member.add;
        etc.). admin queries must not depend on JSONB fields for hot
        paths
    :ptype details: dict[str, Any]
    """

    model_config = ConfigDict(extra="forbid")

    id: UUID
    timestamp: datetime
    event_type: str
    actor_user_id: UUID | None = None
    calling_agent_id: UUID | None = None
    owner_agent_id: UUID | None = None
    customer_id: UUID | None = None
    resource_namespace_id: UUID | None = None
    resource_namespace_type: str | None = None
    action: str
    outcome: str = "success"
    correlation_id: UUID
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def _require_timezone_aware(cls, value: datetime) -> datetime:
        """reject naive datetimes -- every audit envelope is UTC-aware.

        :param value: parsed datetime
        :ptype value: datetime
        :return: the same datetime when it carries a tzinfo
        :rtype: datetime
        :raises ValueError: when the input is naive
        """
        if value.tzinfo is None:
            raise ValueError(
                "AuditEvent.timestamp must be timezone-aware (UTC); "
                "received a naive datetime",
            )
        return value

    @field_validator("outcome")
    @classmethod
    def _require_known_outcome(cls, value: str) -> str:
        """restrict ``outcome`` to the three recognised values.

        the closed set keeps the admin query API's outcome filter
        well-defined without forcing a ``Literal[...]`` on the whole
        envelope (which would break dict-based construction from YAML
        test fixtures).

        :param value: candidate outcome string
        :ptype value: str
        :return: the input value unchanged when valid
        :rtype: str
        :raises ValueError: when ``value`` is not success/failure/error
        """
        allowed = {"success", "failure", "error"}
        if value not in allowed:
            raise ValueError(
                f"AuditEvent.outcome must be one of {sorted(allowed)}; "
                f"received {value!r}",
            )
        return value

    @field_validator("event_type")
    @classmethod
    def _require_dotted_event_type(cls, value: str) -> str:
        """require the ``{domain}.{verb}`` dotted convention.

        the envelope does not gate the set of event types (new domains
        must be able to publish without editing this module) but it
        does require the dotted shape so consumers can parse the
        leading domain segment reliably.

        :param value: candidate event_type
        :ptype value: str
        :return: the input value unchanged when it matches the shape
        :rtype: str
        :raises ValueError: when the value is empty or lacks a dot
        """
        if not value or "." not in value:
            raise ValueError(
                "AuditEvent.event_type must be dotted "
                "'{domain}.{verb}[.{sub_verb}]'; "
                f"received {value!r}",
            )
        return value
