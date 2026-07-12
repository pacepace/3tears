"""identity access authorization -- owner-only RBAC (v0.15.0).

Identity reads and writes are agent-internal: the ``identity_propose``
tool runs as the owning agent, and the consumer's consent / rollback path
acts as the agent owner behind the host's own auth. So authz is exactly
the evaluator's owner short-circuit (``caller_agent_id == owner_agent_id``):
the agent owns its own identity namespace by construction, so every action
is allowed grant-free with no acl migration, no seed row, and no
``platform.namespaces`` write. Mirrors ``agent/intention``'s authorizer.

**User isolation is NOT RBAC.** Every metallm user shares one ``agent_id``,
so the owner short-circuit sees every user's blocks -- isolation is the
``user_id`` WHERE clause on the collection's reads, enforced separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import NAMESPACE_DNS, UUID, uuid5

from threetears.agent.acl import (
    AccessDenied,
    AclCache,
    authorize_on_entity,
)
from threetears.observe import get_logger

__all__ = [
    "ACTION_IDENTITY_READ",
    "ACTION_IDENTITY_WRITE",
    "IDENTITY_NAMESPACE_TYPE",
    "IdentityAccessDenied",
    "IdentityAuthorizerDependencies",
    "authorize_identity_access",
    "identity_namespace_name",
]

log = get_logger(__name__)


#: namespace_type discriminator for identity rows. A free string, never
#: persisted (the owner descriptor is built in-process), so it needs no
#: acl schema change.
IDENTITY_NAMESPACE_TYPE = "identity"


#: canonical action string for identity reads (resolve / list versions).
ACTION_IDENTITY_READ: Literal["identity.read"] = "identity.read"


#: canonical action string for identity writes (propose / consent / rollback).
ACTION_IDENTITY_WRITE: Literal["identity.write"] = "identity.write"


class IdentityAccessDenied(AccessDenied):
    """raised when the evaluator denies an identity access."""


def identity_namespace_name(agent_id: UUID, customer_id: UUID) -> str:
    """build the canonical identity namespace name for an (agent, customer) pair.

    Shape ``identity.<agent_hex[:8]>.<customer_hex[:8]>`` -- a display / log
    handle only; uniqueness is carried by the deterministic id.

    :param agent_id: owning agent UUID
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :return: canonical namespace name
    :rtype: str
    """
    return f"identity.{agent_id.hex[:8]}.{customer_id.hex[:8]}"


def _identity_namespace_id(agent_id: UUID, customer_id: UUID) -> UUID:
    """deterministic namespace id for the (agent, customer) identity pair.

    :param agent_id: owning agent UUID
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :return: deterministic identity namespace UUID
    :rtype: UUID
    """
    return uuid5(
        NAMESPACE_DNS,
        f"threetears.namespaces.identity.{agent_id.hex}.{customer_id.hex}",
    )


@dataclass(frozen=True)
class _OwnerIdentityNamespace:
    """minimal namespace descriptor for the agent-owner identity path.

    Exposes exactly the four fields :func:`authorize_on_entity` reads
    (``id``, ``customer_id``, ``namespace_type``, ``owner_agent_id``),
    built deterministically in-process WITHOUT reading or creating a
    ``platform.namespaces`` row. Mirrors intention's descriptor.
    """

    id: UUID
    customer_id: UUID
    owner_agent_id: UUID
    namespace_type: str = IDENTITY_NAMESPACE_TYPE


@dataclass(frozen=True)
class IdentityAuthorizerDependencies:
    """dependency bundle the identity authorizer needs at call time.

    v1 authz is the owner short-circuit only, which needs just the shared
    :class:`AclCache`. Kept as a bundle so a later user-grant slice can
    widen it additively.

    :ivar acl_cache: shared :class:`AclCache` carrying loaders + ttl layers
    """

    acl_cache: AclCache


async def authorize_identity_access(
    *,
    action: Literal["identity.read", "identity.write"],
    agent_id: UUID,
    customer_id: UUID,
    caller_agent_id: UUID,
    deps: IdentityAuthorizerDependencies,
) -> None:
    """evaluate an action for the owning agent against its identity namespace.

    Agent-only evaluation: when ``caller_agent_id == agent_id`` (always true
    on the agent-internal path) the evaluator's owner short-circuit allows
    the action grant-free. No ``user_id`` is passed -- user isolation is the
    collection's ``user_id`` WHERE clause, not RBAC.

    :param action: canonical identity action string
    :ptype action: Literal["identity.read", "identity.write"]
    :param agent_id: owning agent UUID (the identity namespace owner)
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID (namespace scope grain)
    :ptype customer_id: UUID
    :param caller_agent_id: invoking agent UUID; the owner short-circuit
        fires when this equals ``agent_id``
    :ptype caller_agent_id: UUID
    :param deps: authorizer dependency bundle (the acl cache)
    :ptype deps: IdentityAuthorizerDependencies
    :return: nothing
    :rtype: None
    :raises IdentityAccessDenied: when the evaluator denies the action
    """
    ns_entity = _OwnerIdentityNamespace(
        id=_identity_namespace_id(agent_id, customer_id),
        customer_id=customer_id,
        owner_agent_id=agent_id,
    )
    try:
        await authorize_on_entity(
            ns_entity=ns_entity,
            action=action,
            user_id=None,
            agent_id=caller_agent_id,
            cache=deps.acl_cache,
            namespace_name=identity_namespace_name(agent_id, customer_id),
        )
    except AccessDenied as exc:
        raise IdentityAccessDenied(
            f"evaluator denied {action} on identity namespace {ns_entity.id}",
        ) from exc
