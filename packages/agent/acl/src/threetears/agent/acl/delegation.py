"""the ceiling on what a delegated admin may hand out.

a customer admin who can author a role, or bind an existing role to a
group, can hand out permissions. nothing in the grant tables stops them
handing out MORE than they themselves hold: an assignment row names a
role and a scope, and neither field is compared against the author. so
an admin whose own grant is ``customer.admin`` on their customer could
bind the ``PlatformSuperAdmin`` role to a group they belong to, at a
scope covering their own namespaces, and come back holding
``datasource.admin`` and ``workspace write`` they were never given.

this module is the missing comparison. one rule, stated once:

    **an admin may write a grant only for actions they already hold at
    the scope the grant will reach.**

"already hold" is answered by the evaluator
(:func:`~threetears.agent.acl.evaluator.evaluate_with_trail`) and by
nothing else. there is deliberately no second notion of "what does this
caller have" here — a parallel answer would drift from the one the
authorization path actually uses, and the drift would be invisible
until it was a bypass. what this module adds on top of the evaluator is
the QUANTIFIER: over which namespaces the question is asked.

**the quantifier is the whole design.** a role is not a grant. it is a
bundle that can later be bound at any scope the author is allowed to
bind at, so checking it against one convenient namespace proves
nothing. :class:`ScopeType.TYPE_CUSTOMER` binds a role to EVERY
namespace of a type within a customer, so authoring is checked against
every one of them:

- a role's ``permissions[resource_type]`` bucket is checked against the
  actions the author holds on EVERY namespace of that type in the
  customer — the intersection, not the union. holding ``write`` on one
  workspace does not license authoring a role that grants ``write`` on
  all of them.
- a customer with zero namespaces of that type yields the EMPTY set,
  not "everything". an intersection over nothing is conventionally
  universal; here that would be a hole you could drive a role through
  by authoring against a resource type the customer does not use yet,
  so :func:`held_actions_on` returns empty for an empty namespace list
  by an explicit branch rather than by the identity of ``&``.

**known conservatism, deliberate: pre-provisioning is refused.** an
admin whose customer has no namespaces of a type yet cannot author or
assign anything on that type, because there is nothing to demonstrate
their own grant against. the obvious fix -- probe a SYNTHETIC namespace
carrying only ``(customer_id, namespace_type)``, which
:meth:`RoleAssignment.covers` resolves correctly for
:attr:`ScopeType.TYPE_CUSTOMER` and :attr:`ScopeType.ALL` -- was
considered and NOT taken, because
:class:`~threetears.agent.acl.cache.GroupNamespaceKey` keys the
per-namespace cache layer on ``(group_id, namespace_id)`` alone. a
synthetic id reused across probes would serve one probe's answer to a
different resource type, and a per-call random id would never hit the
cache while filling it with garbage. the safe version of that idea
needs a cache key that carries the type, which is a change to the cache
primitive rather than to this module. until then the refusal is
fail-closed and says why.

**the wildcard bucket is refused for customer-authored roles**, by
:func:`enforce_within_held_permissions` when ``held`` carries no
:data:`~threetears.agent.acl.types.WILDCARD_RESOURCE_TYPE` entry.
``{"*": ["read"]}`` grants an action on every resource type there is
AND every resource type there will ever be, so its meaning changes when
the platform adds a namespace type — a grant no customer admin can
reason about, widening without anybody writing anything.
:mod:`threetears.agent.acl.catalog` cannot check it either (it skips
that bucket by an explicit branch, because ``"*"`` names no resource
type and so has no catalog entry), so a wildcard bucket would slip the
catalog gate as well. platform-owned roles keep it — the shipped
``Reader`` / ``Auditor`` shapes are built out of it — and a caller
resolving held permissions for a platform admin simply supplies the
bucket.

refusals carry the :class:`~threetears.agent.acl.types.Trail` set the
evaluator produced, because "you may not grant this" is only actionable
next to "here is what you do hold, and via which group". that is the
same evidence the introspection API returns, not a second rendering of
it.

nothing here participates in evaluation. a role that evaluated one way
before this module existed evaluates the same way after; this is a
write-path gate, in the same way :mod:`threetears.agent.acl.catalog` is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from threetears.agent.acl.cache import AclCache
from threetears.agent.acl.evaluator import evaluate_with_trail
from threetears.agent.acl.types import (
    WILDCARD_RESOURCE_TYPE,
    EvaluationContext,
    Namespace,
    Trail,
)
from threetears.observe import get_logger, traced

__all__ = [
    "EscalationViolation",
    "HeldPermissions",
    "PermissionEscalation",
    "enforce_within_held_permissions",
    "escalating_permissions",
    "held_actions_on",
    "resolve_held_permissions",
]

log = get_logger(__name__)


#: sentinel passed as ``EvaluationContext.action`` when the question is
#: "what does this caller hold here?" rather than "may they do X?".
#: the evaluator's per-side action sets do not depend on the action
#: asked about, so any stem works and the decision bool is discarded;
#: :func:`~threetears.agent.acl.evaluator.evaluate_file_access` already
#: reads the evaluator this way. the value is deliberately not a real
#: action string so a stray log line naming it cannot be mistaken for a
#: permission somebody actually checked.
_HELD_ACTIONS_PROBE = "acl.delegation.probe"


class EscalationViolation(BaseModel):
    """one permission a caller tried to grant without holding it.

    returned rather than raised by :func:`escalating_permissions` so a
    role builder can render every refusal at once instead of one per
    round trip — the shape :mod:`threetears.agent.acl.catalog` uses for
    the same reason.

    :param resource_type: bucket key the role named
    :ptype resource_type: str
    :param action: action the caller does not hold, or ``None`` when
        the whole bucket is refused and no single action is at fault
    :ptype action: str | None
    :param message: operator-facing sentence naming what was refused
    :ptype message: str
    """

    model_config = ConfigDict(frozen=True)

    resource_type: str
    action: str | None
    message: str


class PermissionEscalation(PermissionError):
    """raised when a write would grant more than the caller holds.

    carries every violation rather than only the first, and the
    evaluation trails that are the evidence for what the caller DOES
    hold, so a refusal can cite its own reasoning.

    a :class:`PermissionError` rather than a :class:`ValueError`: the
    input is well-formed, the caller is simply not entitled to it, and
    an HTTP layer mapping this to a status code wants 403 rather than
    422.

    :param violations: every refused pair, in the deterministic order
        :func:`escalating_permissions` produces
    :ptype violations: tuple[EscalationViolation, ...]
    :param trails: grant paths establishing what the caller holds
    :ptype trails: tuple[Trail, ...]
    """

    def __init__(
        self,
        violations: tuple[EscalationViolation, ...],
        trails: tuple[Trail, ...] = (),
    ) -> None:
        """build with rendered detail and keep the structured form.

        :param violations: every refused pair
        :ptype violations: tuple[EscalationViolation, ...]
        :param trails: grant paths establishing what the caller holds
        :ptype trails: tuple[Trail, ...]
        """
        self.violations = violations
        self.trails = trails
        super().__init__("; ".join(violation.message for violation in violations))


@dataclass(frozen=True)
class HeldPermissions:
    """what a caller demonstrably holds, per resource type, at one scope.

    the answer :func:`resolve_held_permissions` computes and
    :func:`escalating_permissions` checks against. frozen for the same
    reason every type in :mod:`threetears.agent.acl.types` is: it goes
    straight into a refusal that gets logged and audited.

    a resource type ABSENT from :attr:`actions_by_resource_type` is not
    the same as one present with an empty set only in what it says to a
    reader — both refuse every action under it. the wildcard is the one
    key where absence carries its own meaning: absent means "this
    caller may not author wildcard buckets at all" (see the module
    docstring), which is the state for every customer-scoped caller.

    :ivar actions_by_resource_type: ``{resource_type: frozenset(action)}``
        the caller holds across the whole scope
    :ivar trails: every grant path that contributed, in the order the
        evaluator produced them; the evidence a refusal cites
    :ivar namespace_count: how many namespaces the intersection ran
        over. zero is why an empty answer is empty, and a refusal that
        does not say so reads as a bug
    """

    actions_by_resource_type: Mapping[str, frozenset[str]]
    trails: tuple[Trail, ...]
    namespace_count: int


@traced
async def held_actions_on(
    *,
    namespaces: Sequence[Namespace],
    user_id: UUID,
    agent_id: UUID | None,
    cache: AclCache,
) -> tuple[frozenset[str], tuple[Trail, ...]]:
    """actions the caller holds on EVERY namespace in ``namespaces``.

    the intersection, not the union: a grant written against this
    answer may later reach any of these namespaces, so an action held
    on only some of them is not held for this purpose.

    an empty ``namespaces`` returns the empty set by an explicit
    branch. the identity element of intersection is the universal set,
    and returning that here would let a caller author against a
    resource type their customer has no namespaces of and receive
    every action for free.

    ``user_id`` is required and ``agent_id`` is optional because
    delegated administration is a human act. the restriction is not
    cosmetic: on an agent-ONLY evaluation where the agent owns the
    namespace, the evaluator answers with ``{ctx.action}`` — the
    action asked about, echoed back — which for the probe stem here
    would report the caller holds the probe and nothing else. that
    branch cannot be reached while a user is on the call
    (:func:`~threetears.agent.acl.evaluator._assemble_result` takes the
    intersection path instead, where the ownership sentinel correctly
    reduces to the user's own set).

    :param namespaces: every namespace the prospective grant could
        reach; empty yields an empty answer
    :ptype namespaces: Sequence[Namespace]
    :param user_id: the acting admin's user UUID
    :ptype user_id: UUID
    :param agent_id: acting agent UUID when an agent is on the call,
        else ``None``; capped against the user side as usual
    :ptype agent_id: UUID | None
    :param cache: shared :class:`AclCache` carrying loaders + layers
    :ptype cache: AclCache
    :return: ``(intersected_action_set, trails)`` pair
    :rtype: tuple[frozenset[str], tuple[Trail, ...]]
    """
    if not namespaces:
        # explicit: an intersection over nothing is universal, and
        # universal here is a hole. see the module docstring.
        return frozenset(), ()

    held: frozenset[str] | None = None
    trails: list[Trail] = []
    for namespace in namespaces:
        ctx = EvaluationContext(
            namespace=namespace,
            action=_HELD_ACTIONS_PROBE,
            user_id=user_id,
            agent_id=agent_id,
        )
        result = await evaluate_with_trail(ctx, cache=cache)
        # ``effective_actions`` is what a grant here would be capped to:
        # the user side alone for a user-only call, and the
        # user-capped intersection when an agent is also on the call.
        # the decision bool is discarded -- the probe stem is not a
        # real action and was never expected to be allowed.
        actions = result.effective_actions
        held = actions if held is None else (held & actions)
        trails.extend(result.trails)
        trails.extend(result.user_trails)
        trails.extend(result.agent_trails)
        if not held:
            # nothing survives the intersection; further namespaces
            # cannot re-add an action. the trails already collected
            # stay, because they are still the honest answer to "what
            # does this caller hold" for the refusal message.
            break
    resolved = held if held is not None else frozenset()
    return resolved, tuple(trails)


@traced
async def resolve_held_permissions(
    *,
    namespace_collection: Any,
    customer_id: UUID,
    resource_types: Iterable[str],
    user_id: UUID,
    agent_id: UUID | None,
    cache: AclCache,
) -> HeldPermissions:
    """resolve what a customer admin holds across their whole customer.

    for each resource type, enumerates the customer's namespaces OF
    that type and intersects the caller's effective actions across
    them via :func:`held_actions_on`. that is the ceiling for authoring:
    a role bound at :attr:`ScopeType.TYPE_CUSTOMER` reaches exactly
    that namespace set.

    the result never carries a
    :data:`~threetears.agent.acl.types.WILDCARD_RESOURCE_TYPE` key, so
    :func:`enforce_within_held_permissions` refuses any wildcard bucket
    checked against it. that is the intended outcome for a
    customer-scoped caller; the module docstring gives the reasoning.

    :param namespace_collection: a Collection exposing ``async def
        find_by_type_and_customer(*, namespace_type: str, customer_id:
        UUID) -> list[entity]``; typed ``Any`` for the same reason
        :func:`~threetears.agent.acl.authorize.authorize` types its
        own -- the concrete class lives in the consuming app
    :ptype namespace_collection: Any
    :param customer_id: the tenant whose namespaces bound the answer
    :ptype customer_id: UUID
    :param resource_types: bucket keys to resolve; typically the keys
        of the permissions map being authored
    :ptype resource_types: Iterable[str]
    :param user_id: the acting admin's user UUID
    :ptype user_id: UUID
    :param agent_id: acting agent UUID, or ``None``
    :ptype agent_id: UUID | None
    :param cache: shared :class:`AclCache`
    :ptype cache: AclCache
    :return: the caller's ceiling, with the trails behind it
    :rtype: HeldPermissions
    """
    actions_by_resource_type: dict[str, frozenset[str]] = {}
    trails: list[Trail] = []
    namespace_count = 0
    for resource_type in sorted(set(resource_types)):
        if resource_type == WILDCARD_RESOURCE_TYPE:
            # never resolved: leaving the key absent is what makes the
            # enforcement below refuse the bucket.
            continue
        entities = await namespace_collection.find_by_type_and_customer(
            namespace_type=resource_type,
            customer_id=customer_id,
        )
        namespaces = [
            Namespace(
                id=entity.id,
                customer_id=entity.customer_id,
                namespace_type=entity.namespace_type,
                owner_agent_id=entity.owner_agent_id,
                owner_namespace=getattr(entity, "owner_namespace", None),
                name=getattr(entity, "name", None),
            )
            for entity in entities
        ]
        namespace_count += len(namespaces)
        actions, resolved_trails = await held_actions_on(
            namespaces=namespaces,
            user_id=user_id,
            agent_id=agent_id,
            cache=cache,
        )
        actions_by_resource_type[resource_type] = actions
        trails.extend(resolved_trails)
    return HeldPermissions(
        actions_by_resource_type=actions_by_resource_type,
        trails=tuple(trails),
        namespace_count=namespace_count,
    )


def escalating_permissions(
    permissions: Mapping[str, Iterable[str]],
    held: HeldPermissions,
) -> tuple[EscalationViolation, ...]:
    """report every pair in ``permissions`` the caller does not hold.

    accepts the shape a role write path holds
    (``dict[str, list[str]]``) and the shape a loaded
    :class:`~threetears.agent.acl.types.Role` holds
    (``Mapping[str, frozenset[str]]``) alike, matching
    :func:`~threetears.agent.acl.catalog.validate_permissions`.

    a :data:`~threetears.agent.acl.types.WILDCARD_RESOURCE_TYPE` bucket
    is refused whole (one violation, ``action=None``) unless ``held``
    carries an explicit wildcard entry — see the module docstring for
    why a customer never gets one.

    violations come back in a deterministic order — resource types
    sorted, then actions sorted within each — so the message an
    operator reads does not depend on set iteration order.

    :param permissions: ``{resource_type: [action, ...]}`` map to check
    :ptype permissions: Mapping[str, Iterable[str]]
    :param held: the caller's resolved ceiling
    :ptype held: HeldPermissions
    :return: every refused pair, empty when the map is within the
        ceiling
    :rtype: tuple[EscalationViolation, ...]
    """
    violations: list[EscalationViolation] = []
    for resource_type in sorted(permissions):
        held_actions = held.actions_by_resource_type.get(resource_type)
        if resource_type == WILDCARD_RESOURCE_TYPE and held_actions is None:
            violations.append(
                EscalationViolation(
                    resource_type=resource_type,
                    action=None,
                    message=(
                        f"resource type {resource_type!r} is the type-agnostic wildcard; "
                        "it grants every action on every resource type that exists now or "
                        "is added later, so it cannot be authored from a delegated scope"
                    ),
                ),
            )
            continue
        if held_actions is None:
            held_actions = frozenset()
        for action in sorted(permissions[resource_type]):
            if action not in held_actions:
                violations.append(
                    EscalationViolation(
                        resource_type=resource_type,
                        action=action,
                        message=(
                            f"action {action!r} on resource type {resource_type!r} "
                            "is not held by the caller across the scope this grant "
                            f"would reach ({held.namespace_count} namespace(s) checked)"
                        ),
                    ),
                )
    return tuple(violations)


def enforce_within_held_permissions(
    permissions: Mapping[str, Iterable[str]],
    held: HeldPermissions,
) -> None:
    """raise unless every pair in ``permissions`` is within the ceiling.

    the raising form of :func:`escalating_permissions`, for a write
    path that refuses rather than renders. mirrors
    :func:`~threetears.agent.acl.catalog.enforce_declared_permissions`.

    :param permissions: ``{resource_type: [action, ...]}`` map to check
    :ptype permissions: Mapping[str, Iterable[str]]
    :param held: the caller's resolved ceiling
    :ptype held: HeldPermissions
    :return: nothing; the refusal is the exception
    :rtype: None
    :raises PermissionEscalation: when any pair exceeds the ceiling,
        carrying every violation and the trails behind the ceiling
    """
    violations = escalating_permissions(permissions, held)
    if violations:
        raise PermissionEscalation(violations, held.trails)
