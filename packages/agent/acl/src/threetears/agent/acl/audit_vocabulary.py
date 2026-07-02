"""closed-set vocabulary for rbac audit events.

audit-task-01 Phase 4 retired the per-domain :class:`RbacAuditEnvelope`
in favour of the single :class:`threetears.agent.audit.AuditEvent`
envelope shared by every producing domain. the typed wire envelope is
gone; what remains here is the *closed vocabulary* of rbac event_type,
action, and resource-type strings that the hub publisher validates
against before handing the unified envelope to
:func:`threetears.agent.audit.publish_audit`.

this module is **not** a back-compat shim. the legacy envelope code
path no longer exists -- the publisher cannot dual-emit, the hub no
longer runs a domain-specific consumer, and there is no parallel
serialization model. the tuples + ``Literal`` aliases below exist so
the publisher (``3tears.hub.rbac.audit_publish.publish_rbac_audit``)
keeps the closed-set guarantee on its three identity strings even
though the unified envelope intentionally accepts open ``str``
event_type / action / resource_namespace_type values (new domains must
be able to publish without editing the shared envelope module).

callers that want to enumerate the rbac event vocabulary -- test
parameterization, admin UI dropdowns, documentation generation -- import
:data:`RBAC_AUDIT_EVENT_TYPES` here. callers that want to assert a
candidate string is in the closed set membership-test against the same
tuple. callers that want a static-typing fence on a literal-typed
parameter use :data:`RbacEventType` (and the action / resource-type
analogues).
"""

from __future__ import annotations

from typing import Literal

__all__ = [
    "RBAC_AUDIT_ACTIONS",
    "RBAC_AUDIT_EVENT_TYPES",
    "RBAC_AUDIT_RESOURCE_TYPES",
    "RbacAuditAction",
    "RbacAuditResourceType",
    "RbacEventType",
]


#: closed set of rbac audit event types published by the hub. used by
#: the publisher's validate-before-emit guard and by test
#: parameterization. the unified envelope's ``event_type`` column is
#: open ``str``; the closed set lives at the producer site.
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


#: closed set of rbac audit resource_namespace_type values (the short
#: taxonomy tag that rides on the unified envelope's
#: ``resource_namespace_type`` column for rbac events).
RBAC_AUDIT_RESOURCE_TYPES: tuple[str, ...] = (
    "group",
    "role",
    "assignment",
    "introspection",
)


RbacAuditResourceType = Literal["group", "role", "assignment", "introspection"]


#: closed set of rbac audit action verbs the publisher emits.
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
