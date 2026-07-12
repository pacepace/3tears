"""intention access authorization -- owner-only RBAC (v0.15.0).

Intention reads and writes are agent-internal: the deliberation wake and
the private tools run as the owning agent, and the presence API reads the
agent's wants as the agent owner behind metallm's own JWT auth. So authz
is exactly the evaluator's owner short-circuit
(``caller_agent_id == owner_agent_id``, ``evaluator.py:194``): the agent
owns its own intention namespace by construction, so every action is
allowed grant-free with no acl migration, no seed row, and no
``platform.namespaces`` write.

``namespace_type`` is a free string in 3tears (``acl/types.py``), NOT a
constrained enum, and the owner descriptor is built deterministically
in-process, so ``"intention"`` needs no acl schema change. The descriptor
mirrors memory's :class:`_OwnerMemoryNamespace`; the thin wrapper reuses
the generic :func:`~threetears.agent.acl.authorize_on_entity` primitive.

**User isolation is NOT RBAC.** Every metallm user shares one
``agent_id``, so the owner short-circuit sees every user's wants --
isolation is the ``user_id`` WHERE clause on the collection's read
methods (see :meth:`IntentionsCollection.find_by_user`), enforced
separately from this authorizer. The user-read 3tears grant path (a
hub-side role seed) is deferred until direct user->intention 3tears
access is needed (design §6.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from uuid import NAMESPACE_DNS, UUID, uuid5

from threetears.agent.acl import (
    AccessDenied,
    AclCache,
    authorize_on_entity,
)
from threetears.observe import get_logger

__all__ = [
    "ACTION_INTENTION_READ",
    "ACTION_INTENTION_WRITE",
    "INTENTION_NAMESPACE_TYPE",
    "IntentionAccessDenied",
    "IntentionAuthorizerDependencies",
    "authorize_intention_access",
    "intention_namespace_name",
]

log = get_logger(__name__)


#: namespace_type discriminator for intention rows. A free string, never
#: persisted (the owner descriptor is built in-process), so it needs no
#: acl CHECK-constraint admission the way ``memory`` did.
INTENTION_NAMESPACE_TYPE = "intention"


#: canonical action string for intention reads (list / dedup lookup).
ACTION_INTENTION_READ: Literal["intention.read"] = "intention.read"


#: canonical action string for intention writes (log / mark-surfaced).
ACTION_INTENTION_WRITE: Literal["intention.write"] = "intention.write"


class IntentionAccessDenied(AccessDenied):
    """raised when the evaluator denies an intention access.

    Carries through as the single denial type the intention tools
    catch-or-propagate, matching memory's :class:`MemoryAccessDenied`
    shape so callers surface a precise denial instead of a silent
    empty-set fallback.
    """


def intention_namespace_name(agent_id: UUID, customer_id: UUID) -> str:
    """build the canonical intention namespace name for an (agent, customer) pair.

    Shape ``intentions.<agent_hex[:8]>.<customer_hex[:8]>`` -- a
    human-readable display / log handle only. Uniqueness is carried by
    the ``(namespace_type, owner_agent_id, customer_id)`` triple baked
    into the deterministic id, so the short prefix is not a uniqueness
    key. Used solely for the evaluator's denial / log messages.

    :param agent_id: owning agent UUID
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :return: canonical namespace name
    :rtype: str
    """
    return f"intentions.{agent_id.hex[:8]}.{customer_id.hex[:8]}"


def _intention_namespace_id(agent_id: UUID, customer_id: UUID) -> UUID:
    """deterministic namespace id for the (agent, customer) intention pair.

    Same triple -> same :func:`uuid5`, so the owner descriptor is stable
    across calls (and would converge with any future persisted row). The
    evaluator only reads the four descriptor fields; the id is a stable
    identity for logs, never a DB lookup key on the owner path.

    :param agent_id: owning agent UUID
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :return: deterministic intention namespace UUID
    :rtype: UUID
    """
    return uuid5(
        NAMESPACE_DNS,
        f"threetears.namespaces.intention.{agent_id.hex}.{customer_id.hex}",
    )


@dataclass(frozen=True)
class _OwnerIntentionNamespace:
    """minimal namespace descriptor for the agent-owner intention path.

    Exposes exactly the four fields :func:`authorize_on_entity` reads
    (``id``, ``customer_id``, ``namespace_type``, ``owner_agent_id``),
    built deterministically in-process WITHOUT reading or creating a
    ``platform.namespaces`` row. Mirrors memory's
    :class:`_OwnerMemoryNamespace`: the agent's sandboxed L3 search_path
    is its own schema, which has no ``namespaces`` table, so the
    evaluator's owner short-circuit must resolve from the descriptor's
    fields alone -- no row need exist.
    """

    id: UUID
    customer_id: UUID
    owner_agent_id: UUID
    namespace_type: str = INTENTION_NAMESPACE_TYPE


@dataclass(frozen=True)
class IntentionAuthorizerDependencies:
    """dependency bundle the intention authorizer needs at call time.

    v1 authz is the owner short-circuit only, which needs just the shared
    :class:`AclCache` to run the evaluator -- no namespace / group / role
    collections (the user-grant path that would need them is deferred,
    design §6.5). Kept as a bundle (not a bare cache param) so the tool
    signatures mirror memory's :class:`MemoryAuthorizerDependencies` and
    a later user-grant slice can widen it additively.

    :ivar acl_cache: shared :class:`AclCache` carrying loaders + ttl layers
    """

    acl_cache: AclCache


async def authorize_intention_access(
    *,
    action: Literal["intention.read", "intention.write"],
    agent_id: UUID,
    customer_id: UUID,
    caller_agent_id: UUID,
    deps: IntentionAuthorizerDependencies,
) -> None:
    """evaluate an action for the owning agent against its intention namespace.

    Agent-only evaluation: the descriptor's ``owner_agent_id`` is
    ``agent_id`` and the caller is ``caller_agent_id``, so when they match
    (always true on the agent-internal path) the evaluator's owner
    short-circuit allows the action grant-free. No ``user_id`` is passed
    to the evaluator -- user isolation is the collection's ``user_id``
    WHERE clause, not RBAC (see the module docstring).

    :param action: canonical intention action string
    :ptype action: Literal["intention.read", "intention.write"]
    :param agent_id: owning agent UUID (the intention namespace owner)
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID (namespace scope grain)
    :ptype customer_id: UUID
    :param caller_agent_id: invoking agent UUID; the owner short-circuit
        fires when this equals ``agent_id``
    :ptype caller_agent_id: UUID
    :param deps: authorizer dependency bundle (the acl cache)
    :ptype deps: IntentionAuthorizerDependencies
    :return: nothing
    :rtype: None
    :raises IntentionAccessDenied: when the evaluator denies the action
    """
    ns_entity = _OwnerIntentionNamespace(
        id=_intention_namespace_id(agent_id, customer_id),
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
            namespace_name=intention_namespace_name(agent_id, customer_id),
        )
    except AccessDenied as exc:
        raise IntentionAccessDenied(
            f"evaluator denied {action} on intention namespace {ns_entity.id}",
        ) from exc
