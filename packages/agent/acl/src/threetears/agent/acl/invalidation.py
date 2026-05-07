"""typed NATS payload models for the rbac invalidation subjects.

every 3tears app that participates in cross-process rbac caching
publishes / subscribes to the same three subjects (subjects built by
:meth:`threetears.nats.Subjects.acl_invalidate`). the wire format
must match byte-for-byte across publishers (admin endpoints, agent
self-mutations) and subscribers (broker-side cache, every agent
pod's :class:`AclCache`); promoting these payload models from
deploying-app modules to the canonical package guarantees a single
source of truth.

all three are :class:`pydantic.BaseModel` subclasses so callers
serialize via ``.model_dump_json()`` and deserialize via
``.model_validate_json()`` at the NATS-message border. fields are
typed with :class:`uuid.UUID` so deserialized values are real UUID
instances ready for direct comparison / cache-key construction
without a manual ``UUID(...)`` round-trip on every callsite.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel

__all__ = [
    "AssignmentInvalidatePayload",
    "MembershipInvalidatePayload",
    "RoleInvalidatePayload",
]


class MembershipInvalidatePayload(BaseModel):
    """typed payload for ``{ns}.acl.membership.invalidate``.

    fired when an actor's group membership set changes (admin adds /
    removes a group_member row, or an agent's auto-membership
    translator updates its grants). subscribers drop that actor's
    membership-layer cache entry.

    :ivar actor_type: ``"user"`` or ``"agent"``
    :ivar actor_id: actor UUID (deserialized at validation time)
    """

    actor_type: Literal["user", "agent"]
    actor_id: UUID


class AssignmentInvalidatePayload(BaseModel):
    """typed payload for ``{ns}.acl.assignment.invalidate``.

    fired when a group's role assignments change (admin grants /
    revokes a role on a scope, or an agent's auto-assignment
    translator does the same). subscribers drop every
    assignment-layer cache entry naming the group.

    :ivar group_id: group UUID whose assignment layers should be
        dropped
    """

    group_id: UUID


class RoleInvalidatePayload(BaseModel):
    """typed payload for ``{ns}.acl.role.invalidate``.

    fired on a role-definition edit (the role's ``permissions`` JSONB
    changed). the payload is intentionally empty; the subscription
    simply clears every assignment-layer cache entry because
    permissions may have shifted under any group holding an
    assignment to the role. declared as a model (rather than skipped)
    so every invalidation subject has a typed schema at the border
    and the wire-format enforcement sees uniform shape across all
    three handlers.
    """
