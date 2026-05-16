"""unified audit envelope + publish helper for the 3tears platform.

single :class:`AuditEvent` + :func:`publish_audit` shared by every
producing domain (workspace tools, hub rbac endpoints, agent memory,
custom tools). the hub-side ``unified_audit_consumer`` subscribes to
``{namespace}.audit.>`` and persists every event to
``platform_audit.audit_events`` without a domain-specific consumer.

produced by ``audit-task-01``; supersedes
:class:`threetears.agent.workspace.audit.WorkspaceAuditEnvelope` and
:class:`threetears.agent.acl.audit.RbacAuditEnvelope` (both deleted in
the same shard).
"""

from __future__ import annotations

from threetears.agent.audit.envelope import AuditEvent
from threetears.agent.audit.publish import publish_audit

__version__ = "0.7.0"

__all__ = [
    "AuditEvent",
    "publish_audit",
]
