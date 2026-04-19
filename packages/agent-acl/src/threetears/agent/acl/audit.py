"""shared rbac audit envelope (phase 7 of rbac-task-01).

this module lives in the ``agent-acl`` package so any future agent-side
consumer of rbac audit events imports the same pydantic model that the
hub publisher + hub consumer use. carrying the envelope in a shared
package keeps wire-format drift impossible: the publisher can only
serialize what the model knows about, and the consumer can only parse
what the model exposes.

every rbac mutation (group, role, assignment lifecycle; group
membership changes) and every rbac introspection query publishes one
:class:`RbacAuditEnvelope` on ``{namespace}.audit.rbac``. the hub-side
consumer (``aibots.hub.audit.rbac_consumer.RbacAuditConsumer``)
subscribes under a queue group and persists to
``platform_audit.audit_events``.

envelope anti-drift guarantees:

- ``model_config`` uses ``extra='forbid'`` -- any stray field on the
  wire fails parse loudly. a miswired publisher surfaces at construct
  time rather than silently dropping the field.
- ``event_type`` is a closed :data:`RbacEventType` ``Literal[...]``; a
  publisher adding a new event type without extending this enum fails
  validation before the payload ever hits NATS.
- ``resource_type`` is a closed ``Literal[...]`` over the four rbac
  resource shapes (``group`` / ``role`` / ``assignment`` /
  ``introspection``); same guarantee.
- ``action`` is a closed ``Literal[...]`` over the set of verbs these
  events carry (``create`` / ``update`` / ``delete`` /
  ``add_member`` / ``remove_member`` / ``explain`` / ``effective`` /
  ``namespace_access`` / ``dry_run``).

the ``details`` mapping is intentionally permissive (``dict[str, Any]``)
because each event_type has its own per-event-type details shape;
documenting those shapes in the phase-7 shard + enforcing them in the
publisher-side tests is a stricter contract than a union of pydantic
submodels would give, without ballooning the surface area here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "RBAC_AUDIT_ACTIONS",
    "RBAC_AUDIT_EVENT_TYPES",
    "RBAC_AUDIT_RESOURCE_TYPES",
    "RbacAuditAction",
    "RbacAuditEnvelope",
    "RbacAuditResourceType",
    "RbacEventType",
]


#: closed set of rbac audit event types published by the hub.
#: kept as a tuple + ``Literal`` for both runtime iteration (e.g. test
#: parameterization) and static type-checking (pydantic refuses any
#: value outside the set).
RBAC_AUDIT_EVENT_TYPES: tuple[str, ...] = (
    "rbac.group.create",
    "rbac.group.update",
    "rbac.group.delete",
    "rbac.group.member.add",
    "rbac.group.member.remove",
    "rbac.role.create",
    "rbac.role.update",
    "rbac.role.delete",
    "rbac.assignment.create",
    "rbac.assignment.delete",
    "rbac.introspect.explain",
    "rbac.introspect.effective",
    "rbac.introspect.namespace_access",
    "rbac.introspect.dry_run",
)


RbacEventType = Literal[
    "rbac.group.create",
    "rbac.group.update",
    "rbac.group.delete",
    "rbac.group.member.add",
    "rbac.group.member.remove",
    "rbac.role.create",
    "rbac.role.update",
    "rbac.role.delete",
    "rbac.assignment.create",
    "rbac.assignment.delete",
    "rbac.introspect.explain",
    "rbac.introspect.effective",
    "rbac.introspect.namespace_access",
    "rbac.introspect.dry_run",
]


RBAC_AUDIT_RESOURCE_TYPES: tuple[str, ...] = (
    "group",
    "role",
    "assignment",
    "introspection",
)


RbacAuditResourceType = Literal["group", "role", "assignment", "introspection"]


RBAC_AUDIT_ACTIONS: tuple[str, ...] = (
    "create",
    "update",
    "delete",
    "add_member",
    "remove_member",
    "explain",
    "effective",
    "namespace_access",
    "dry_run",
)


RbacAuditAction = Literal[
    "create",
    "update",
    "delete",
    "add_member",
    "remove_member",
    "explain",
    "effective",
    "namespace_access",
    "dry_run",
]


class RbacAuditEnvelope(BaseModel):
    """typed wire-format envelope for rbac audit events.

    built on the hub side by :mod:`aibots.hub.rbac.audit_publish`
    helpers (mutations) and directly inside
    :mod:`aibots.hub.rbac.introspect` (introspection queries). decoded
    on the hub side by
    :class:`aibots.hub.audit.rbac_consumer.RbacAuditConsumer`. the wire
    contract is identical on both sides because both import this
    model from the shared package; drift is a static-type problem
    rather than a runtime one.

    per rbac-task-01 phase 7, every mutation-scope envelope carries
    before/after effective-access snapshots keyed by affected group id
    (see :attr:`permissions_before` / :attr:`permissions_after`).
    snapshots are truncated at :data:`~SNAPSHOT_MAX_ACTORS` entries to
    bound audit-row size; the consumer sees a ``"_truncated"`` marker
    and a count when a truncation fires.

    :param event_type: fully-qualified event type string from
        :data:`RBAC_AUDIT_EVENT_TYPES`
    :ptype event_type: RbacEventType
    :param actor_user_id: admin user who issued the request (load-
        bearing "who did this?" field for compliance)
    :ptype actor_user_id: UUID
    :param customer_id: customer scope of the mutation; ``None`` when
        the mutation is platform-scoped (``scope='all'`` assignment,
        platform-scoped group, role CRUD)
    :ptype customer_id: UUID | None
    :param correlation_id: request correlation UUID
    :ptype correlation_id: UUID
    :param date_occurred: timezone-aware UTC instant the mutation ran
    :ptype date_occurred: datetime
    :param resource_type: closed taxonomy tag from
        :data:`RBAC_AUDIT_RESOURCE_TYPES`
    :ptype resource_type: RbacAuditResourceType
    :param resource_id: id of the affected resource (group UUID, role
        UUID, assignment UUID). ``None`` for introspection queries
        whose subject is not a single mutated row.
    :ptype resource_id: UUID | None
    :param action: closed verb from :data:`RBAC_AUDIT_ACTIONS`
    :ptype action: RbacAuditAction
    :param details: event-type-specific structured payload (per phase
        7 shape table). never carries raw request / response bodies.
    :ptype details: dict[str, Any]
    :param permissions_before: per-affected-group effective permissions
        snapshot prior to the mutation, keyed by ``str(group_id)``.
        populated on mutation events, ``None`` on introspection
        events. each value is a ``{resource_type: [action, ...]}``
        mapping; truncated snapshots include a synthetic
        ``"_truncated"`` key.
    :ptype permissions_before: dict[str, list[str]] | None
    :param permissions_after: per-affected-group effective permissions
        snapshot after the mutation committed, same shape as
        :attr:`permissions_before`.
    :ptype permissions_after: dict[str, list[str]] | None
    """

    model_config = ConfigDict(extra="forbid")

    event_type: RbacEventType
    actor_user_id: UUID
    customer_id: UUID | None = None
    correlation_id: UUID
    date_occurred: datetime
    resource_type: RbacAuditResourceType
    resource_id: UUID | None = None
    action: RbacAuditAction
    details: dict[str, Any] = Field(default_factory=dict)
    permissions_before: dict[str, list[str]] | None = None
    permissions_after: dict[str, list[str]] | None = None

    @field_validator("date_occurred")
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
                "RbacAuditEnvelope.date_occurred must be timezone-aware "
                "(UTC); received a naive datetime",
            )
        return value
