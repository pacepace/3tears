"""rbac evaluator: pure-functional resolver for "can actor do action on namespace".

two entry points share one resolution algorithm:

- :func:`evaluate_decision` — the production hot path. returns a bool.
  callers wire it behind the cache so the answer is served from
  process memory most of the time.
- :func:`evaluate_with_trail` — the introspection / audit path.
  returns the full :class:`EvaluationResult` with every contributing
  ``(group, assignment, role)`` chain. trail-mode is uncached by
  design; freshness beats throughput when a human is asking
  "why does alice have write here?"

resolution rules (in order):

1. **agent ownership short-circuit** — if the namespace's
   ``owner_agent_id`` matches the calling agent, the agent side
   short-circuits to "every action allowed" without ever touching
   the membership / grant tables. ownership is a property of the
   namespace row, not a grant.

2. **side resolution** — for each side that the caller supplied
   (user side iff ``user_id`` set; agent side iff ``agent_id`` set):

     a. ask :class:`MembershipLoader` for the actor's group ids.
     b. ask :class:`GrantLoader` for assignments those groups hold
        that could cover the namespace.
     c. for each assignment that actually covers the namespace
        (re-check via :meth:`RoleAssignment.covers`), resolve the
        role and union its actions for the namespace's type.

   the side's action set is the union of every contributing role's
   action set. an empty side is a deny on that side.

3. **intersection** — when both sides ran, the effective action set
   is ``user_actions ∩ agent_actions``. either side empty produces
   a deny. when only one side ran, its action set is the effective
   set as-is.

4. **decision** — allow iff ``ctx.action`` is in the effective set.

cross-customer wall: every resolver that produces a non-empty action
set requires the contributing group to be platform-scoped or to share
the namespace's customer. the membership loader is responsible for
returning the correct customer_id on each membership row; the
evaluator double-checks at the side-resolution boundary so a buggy
loader cannot leak grants across customer lines.

mixed-membership groups (a group with both user and agent members)
are handled correctly by the side split: the user side only counts
groups in which the actor is a user member, the agent side only
counts groups in which the actor is an agent member. one group can
contribute to both sides simultaneously when both members happen to
be calling, but never to the same side via the wrong member type.
"""

from __future__ import annotations

from typing import Iterable
from uuid import UUID

from threetears.agent.acl.loader import GrantLoader, MembershipLoader
from threetears.agent.acl.types import (
    EvaluationContext,
    EvaluationResult,
    Group,
    GroupMembership,
    LimitingSide,
    MemberType,
    Namespace,
    Role,
    RoleAssignment,
    Trail,
)
from threetears.observe import get_logger

__all__ = [
    "evaluate_decision",
    "evaluate_with_trail",
]

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------


async def evaluate_decision(
    ctx: EvaluationContext,
    *,
    membership_loader: MembershipLoader,
    grant_loader: GrantLoader,
) -> bool:
    """resolve "is ``ctx.action`` allowed?" and return the bool.

    thin wrapper around :func:`evaluate_with_trail` that throws away
    the trail. cache-friendly because the loaders may serve cached
    rows; the evaluator itself does no caching.

    :param ctx: evaluation context (namespace, action, optional
        ``user_id`` and / or ``agent_id``)
    :ptype ctx: EvaluationContext
    :param membership_loader: actor -> groups resolver
    :ptype membership_loader: MembershipLoader
    :param grant_loader: groups -> assignments + roles resolver
    :ptype grant_loader: GrantLoader
    :return: True iff the action is allowed
    :rtype: bool
    """
    result = await evaluate_with_trail(
        ctx,
        membership_loader=membership_loader,
        grant_loader=grant_loader,
    )
    return result.decision


async def evaluate_with_trail(
    ctx: EvaluationContext,
    *,
    membership_loader: MembershipLoader,
    grant_loader: GrantLoader,
) -> EvaluationResult:
    """resolve the question and return the full :class:`EvaluationResult`.

    walks the membership + grant graph deterministically and packages
    every contributing chain as a :class:`Trail`. when both sides are
    supplied, populates :attr:`EvaluationResult.user_actions`,
    :attr:`agent_actions`, :attr:`limiting_side`, and the per-side
    trails. when only one side is supplied, populates
    :attr:`EvaluationResult.trails` and leaves the intersection
    fields empty.

    :param ctx: evaluation context
    :ptype ctx: EvaluationContext
    :param membership_loader: actor -> groups resolver
    :ptype membership_loader: MembershipLoader
    :param grant_loader: groups -> assignments + roles resolver
    :ptype grant_loader: GrantLoader
    :return: full evaluation result with trails
    :rtype: EvaluationResult
    :raises ValueError: when ``ctx`` carries neither ``user_id`` nor
        ``agent_id`` (an evaluation needs at least one actor)
    """
    if ctx.user_id is None and ctx.agent_id is None:
        raise ValueError(
            "evaluate_with_trail requires at least one of user_id or agent_id",
        )

    user_actions: frozenset[str] = frozenset()
    agent_actions: frozenset[str] = frozenset()
    user_trails: tuple[Trail, ...] = ()
    agent_trails: tuple[Trail, ...] = ()
    agent_owner_short_circuited = False

    if ctx.user_id is not None:
        user_actions, user_trails = await _resolve_side(
            actor_id=ctx.user_id,
            member_type=MemberType.USER,
            namespace=ctx.namespace,
            membership_loader=membership_loader,
            grant_loader=grant_loader,
        )

    if ctx.agent_id is not None:
        if ctx.namespace.owner_agent_id == ctx.agent_id:
            # ownership short-circuit: agent side is implicitly full
            # access. record a sentinel "every action" set and skip
            # the loader trip. the trail surface stays empty because
            # there is no grant to point at.
            agent_actions = _OWNER_ACTIONS
            agent_owner_short_circuited = True
        else:
            agent_actions, agent_trails = await _resolve_side(
                actor_id=ctx.agent_id,
                member_type=MemberType.AGENT,
                namespace=ctx.namespace,
                membership_loader=membership_loader,
                grant_loader=grant_loader,
            )

    result = _assemble_result(
        ctx=ctx,
        user_actions=user_actions,
        agent_actions=agent_actions,
        user_trails=user_trails,
        agent_trails=agent_trails,
        agent_owner_short_circuited=agent_owner_short_circuited,
    )
    return result


# ---------------------------------------------------------------------------
# owner sentinel
# ---------------------------------------------------------------------------


class _AllActionsSet(frozenset[str]):
    """sentinel set that contains every action.

    the agent ownership short-circuit needs to express "the agent has
    every action" without enumerating an open-ended action vocabulary.
    a real frozenset would have to list every possible action up
    front; this subclass overrides ``__contains__`` and the
    intersection operator so any string is considered a member and
    intersection with another set is just the other set.

    instances stay empty as a frozenset for ``len()`` / iteration
    purposes (those are not exercised on the hot path); equality and
    hashing inherit from frozenset so callers may still use the
    sentinel as a dict key. only :meth:`__contains__` and
    :meth:`__and__` / :meth:`__rand__` are overridden because those
    are the only operations the evaluator runs against the action
    set on the agent side.
    """

    def __contains__(self, item: object) -> bool:
        """every action string is a member; non-strings are not.

        :param item: candidate value to test for membership
        :ptype item: object
        :return: True iff item is a string
        :rtype: bool
        """
        return isinstance(item, str)

    def __and__(self, other: Iterable[str]) -> frozenset[str]:
        """intersection with any iterable returns that iterable as a frozenset.

        because the sentinel "contains" every action, the intersection
        ``ALL ∩ X`` is just ``X``. used to cap the user side by the
        agent's owner-shortcircuit set: the user can do whatever the
        user-side grants allow, no further reduction.

        :param other: action set to intersect with
        :ptype other: Iterable[str]
        :return: frozenset of the other side
        :rtype: frozenset[str]
        """
        return frozenset(other)

    def __rand__(self, other: Iterable[str]) -> frozenset[str]:
        """reflected intersection — see :meth:`__and__`.

        :param other: action set on the left
        :ptype other: Iterable[str]
        :return: frozenset of the other side
        :rtype: frozenset[str]
        """
        return frozenset(other)


_OWNER_ACTIONS: _AllActionsSet = _AllActionsSet()


# ---------------------------------------------------------------------------
# side resolution
# ---------------------------------------------------------------------------


async def _resolve_side(
    *,
    actor_id: UUID,
    member_type: MemberType,
    namespace: Namespace,
    membership_loader: MembershipLoader,
    grant_loader: GrantLoader,
) -> tuple[frozenset[str], tuple[Trail, ...]]:
    """compute one side's action set + trails for ``actor_id``.

    runs the four-step pipeline:

    1. load every membership for the actor from the appropriate
       loader method.
    2. filter memberships by member_type (a user-side resolution
       only counts memberships where the actor is a user member,
       and vice versa) and by customer (cross-customer wall).
    3. ask :class:`GrantLoader` for the assignments those groups
       hold that could cover the namespace, plus the roles those
       assignments reference plus the group rows themselves.
    4. for every assignment that covers the namespace, resolve its
       role and union the role's contribution; record one
       :class:`Trail` per ``(group, assignment, role)`` chain.

    :param actor_id: user UUID or agent UUID under evaluation
    :ptype actor_id: UUID
    :param member_type: user or agent — drives both the loader call
        and the membership filter
    :ptype member_type: MemberType
    :param namespace: namespace under evaluation
    :ptype namespace: Namespace
    :param membership_loader: actor -> memberships resolver
    :ptype membership_loader: MembershipLoader
    :param grant_loader: groups -> assignments + roles + groups resolver
    :ptype grant_loader: GrantLoader
    :return: ``(action_set, trails)`` pair for this side
    :rtype: tuple[frozenset[str], tuple[Trail, ...]]
    """
    if member_type == MemberType.USER:
        memberships = await membership_loader.load_for_user(actor_id)
    else:
        memberships = await membership_loader.load_for_agent(actor_id)

    eligible = _filter_memberships(
        memberships=memberships,
        actor_id=actor_id,
        member_type=member_type,
        namespace=namespace,
    )

    group_ids = tuple(_unique_ordered(m.group_id for m in eligible))
    if not group_ids:
        return frozenset(), ()

    assignments = await grant_loader.load_assignments_for_groups(
        group_ids=group_ids,
        namespace=namespace,
    )
    role_ids = tuple(_unique_ordered(a.role_id for a in assignments))
    roles = await grant_loader.load_roles(role_ids) if role_ids else {}
    groups_raw = await grant_loader.load_groups(group_ids) if group_ids else {}
    groups = _coerce_groups(groups_raw)

    actions, trails = _walk_assignments(
        assignments=assignments,
        roles=roles,
        groups=groups,
        eligible_group_ids=frozenset(group_ids),
        namespace=namespace,
    )
    return actions, trails


def _filter_memberships(
    *,
    memberships: tuple[GroupMembership, ...],
    actor_id: UUID,
    member_type: MemberType,
    namespace: Namespace,
) -> tuple[GroupMembership, ...]:
    """drop memberships that do not name ``actor_id`` as the requested type.

    enforces:

    - the membership's :attr:`GroupMembership.member_id` equals
      ``actor_id`` (defensive — the loader was already given the
      actor id but a buggy loader could return rows for the wrong
      actor).
    - the membership's :attr:`GroupMembership.member_type` equals
      ``member_type`` (the user side never counts agent memberships
      and vice versa; mixed groups split correctly).
    - the membership's :attr:`GroupMembership.customer_id` either
      matches the namespace's customer (same-customer grant) or is
      ``None`` (the membership row is platform-scoped — admin user
      or platform-managed agent — and may reach across customers
      via a platform-scoped group).

    the cross-customer cut is the second line of defense behind the
    loader's own filtering. if the loader is wrong, the filter still
    refuses to count a customer-scoped membership against another
    customer's namespace.

    :param memberships: raw memberships from the loader
    :ptype memberships: tuple[GroupMembership, ...]
    :param actor_id: caller's UUID
    :ptype actor_id: UUID
    :param member_type: side under evaluation
    :ptype member_type: MemberType
    :param namespace: namespace under evaluation
    :ptype namespace: Namespace
    :return: memberships that survived the filter
    :rtype: tuple[GroupMembership, ...]
    """
    surviving: list[GroupMembership] = []
    for membership in memberships:
        if membership.member_id != actor_id:
            continue
        if membership.member_type != member_type:
            continue
        if (
            membership.customer_id is not None
            and membership.customer_id != namespace.customer_id
        ):
            continue
        surviving.append(membership)
    return tuple(surviving)


def _walk_assignments(
    *,
    assignments: tuple[RoleAssignment, ...],
    roles: dict[UUID, Role],
    groups: dict[UUID, Group],
    eligible_group_ids: frozenset[UUID],
    namespace: Namespace,
) -> tuple[frozenset[str], tuple[Trail, ...]]:
    """assemble the action set and trail tuple for one side.

    iteration is deterministic in (group_id, assignment_id) order so
    trails come out repeatable across runs — operators reading two
    explanations side-by-side see the same row order both times.
    only assignments whose ``group_id`` is in ``eligible_group_ids``
    are considered (defensive: the grant loader may over-return).

    cross-customer + platform-scope check on the group: a
    customer-scoped group must match the namespace's customer for
    its assignments to count; a platform-scoped group counts
    universally.

    :param assignments: role assignments held by the eligible groups
    :ptype assignments: tuple[RoleAssignment, ...]
    :param roles: role_id -> Role mapping (loader may omit ids
        without a row; assignments whose role is absent are skipped
        with a debug log)
    :ptype roles: dict[UUID, Role]
    :param groups: group_id -> Group mapping
    :ptype groups: dict[UUID, Group]
    :param eligible_group_ids: groups whose assignments may
        contribute (set-membership lookup)
    :ptype eligible_group_ids: frozenset[UUID]
    :param namespace: namespace under evaluation; supplied to
        :meth:`RoleAssignment.covers` and to
        :meth:`Role.actions_for`
    :ptype namespace: Namespace
    :return: ``(action_set, trails)`` pair
    :rtype: tuple[frozenset[str], tuple[Trail, ...]]
    """
    actions: set[str] = set()
    trails: list[Trail] = []
    ordered = sorted(assignments, key=lambda a: (str(a.group_id), str(a.id)))
    for assignment in ordered:
        if assignment.group_id not in eligible_group_ids:
            continue
        if not assignment.covers(namespace):
            continue
        group = groups.get(assignment.group_id)
        if group is None:
            log.debug(
                "skipping assignment with unresolved group",
                extra={"extra_data": {
                    "assignment_id": str(assignment.id),
                    "group_id": str(assignment.group_id),
                }},
            )
            continue
        if (
            group.customer_id is not None
            and group.customer_id != namespace.customer_id
        ):
            # the group is customer-scoped to a different customer; an
            # assignment it holds cannot reach this namespace.
            continue
        role = roles.get(assignment.role_id)
        if role is None:
            log.debug(
                "skipping assignment with unresolved role",
                extra={"extra_data": {
                    "assignment_id": str(assignment.id),
                    "role_id": str(assignment.role_id),
                }},
            )
            continue
        contributed = role.actions_for(namespace.namespace_type)
        actions.update(contributed)
        trails.append(
            Trail(
                group=group,
                assignment=assignment,
                role=role,
                contributed_actions=contributed,
            ),
        )
    return frozenset(actions), tuple(trails)


def _coerce_groups(raw: dict[UUID, object]) -> dict[UUID, Group]:
    """coerce a loader-provided ``dict[UUID, object]`` to ``dict[UUID, Group]``.

    accepts either real :class:`Group` instances or duck-typed objects
    with the same three attributes. the loader Protocol returns
    ``object`` to keep callers free to pass their own ORM rows; this
    coercion happens once per evaluation so the trail builder sees a
    uniform type.

    objects that do not expose the three attributes (``id``, ``name``,
    ``customer_id``) are dropped from the result; the trail walker
    treats their absence as "skip this assignment."

    :param raw: loader-provided mapping
    :ptype raw: dict[UUID, object]
    :return: normalized mapping keyed by the same group ids
    :rtype: dict[UUID, Group]
    """
    result: dict[UUID, Group] = {}
    for group_id, value in raw.items():
        if isinstance(value, Group):
            result[group_id] = value
        elif (
            hasattr(value, "id")
            and hasattr(value, "name")
            and hasattr(value, "customer_id")
        ):
            result[group_id] = Group(
                id=value.id,
                name=value.name,
                customer_id=value.customer_id,
            )
    return result


def _unique_ordered(values: Iterable[UUID]) -> list[UUID]:
    """dedupe ``values`` preserving first-seen order.

    used to build deterministic ``group_ids`` / ``role_ids`` lists
    before handing them to a loader. a plain ``set()`` would scramble
    the order, which makes test fixtures harder to reason about and
    hurts the deterministic trail assembly downstream.

    :param values: iterable of UUIDs that may contain duplicates
    :ptype values: Iterable[UUID]
    :return: deduplicated list in first-seen order
    :rtype: list[UUID]
    """
    seen: set[UUID] = set()
    result: list[UUID] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


# ---------------------------------------------------------------------------
# result assembly
# ---------------------------------------------------------------------------


def _assemble_result(
    *,
    ctx: EvaluationContext,
    user_actions: frozenset[str],
    agent_actions: frozenset[str],
    user_trails: tuple[Trail, ...],
    agent_trails: tuple[Trail, ...],
    agent_owner_short_circuited: bool,
) -> EvaluationResult:
    """wrap the per-side action sets into an :class:`EvaluationResult`.

    branches by which sides ran:

    - both sides ran (intersection mode): effective set is
      ``user_actions ∩ agent_actions``; populates
      :attr:`EvaluationResult.user_trails`,
      :attr:`agent_trails`, :attr:`limiting_side`. either side empty
      forces a deny and :attr:`LimitingSide.NEITHER`.
    - only user side ran: effective set is ``user_actions``; populates
      :attr:`EvaluationResult.trails`.
    - only agent side ran: effective set is ``agent_actions`` (with
      the owner short-circuit translated to "every action" via the
      sentinel set's ``__contains__``); populates :attr:`trails`.

    decision = ``ctx.action in effective_actions``.

    :param ctx: original evaluation context
    :ptype ctx: EvaluationContext
    :param user_actions: action set the user side resolved to
    :ptype user_actions: frozenset[str]
    :param agent_actions: action set the agent side resolved to
    :ptype agent_actions: frozenset[str]
    :param user_trails: user-side grant chains
    :ptype user_trails: tuple[Trail, ...]
    :param agent_trails: agent-side grant chains
    :ptype agent_trails: tuple[Trail, ...]
    :param agent_owner_short_circuited: True iff agent side was
        resolved via the namespace ownership shortcut
    :ptype agent_owner_short_circuited: bool
    :return: full evaluation result
    :rtype: EvaluationResult
    """
    has_user = ctx.user_id is not None
    has_agent = ctx.agent_id is not None
    effective: frozenset[str]
    limiting: LimitingSide
    trails: tuple[Trail, ...]
    user_trails_out: tuple[Trail, ...]
    agent_trails_out: tuple[Trail, ...]

    if has_user and has_agent:
        if not user_actions or (
            not agent_actions and not agent_owner_short_circuited
        ):
            effective = frozenset()
            limiting = LimitingSide.NEITHER
        else:
            # owner shortcut: agent_actions is the open-ended sentinel;
            # ``user_actions & sentinel`` returns ``user_actions``.
            effective = frozenset(user_actions & agent_actions)
            limiting = _classify_limiting_side(
                user_actions=user_actions,
                agent_actions=agent_actions,
                agent_owner_short_circuited=agent_owner_short_circuited,
            )
        trails = ()
        user_trails_out = user_trails
        agent_trails_out = agent_trails
    elif has_user:
        effective = frozenset(user_actions)
        limiting = LimitingSide.NEITHER
        trails = user_trails
        user_trails_out = ()
        agent_trails_out = ()
    else:
        # agent-only evaluation. with the owner shortcut, every action
        # is allowed; we surface "the requested action" as the
        # effective set so the bool decision works without the
        # sentinel leaking into caller-visible state.
        if agent_owner_short_circuited:
            effective = frozenset({ctx.action})
        else:
            effective = frozenset(agent_actions)
        limiting = LimitingSide.NEITHER
        trails = agent_trails
        user_trails_out = ()
        agent_trails_out = ()

    decision = ctx.action in effective

    return EvaluationResult(
        decision=decision,
        effective_actions=effective,
        trails=trails,
        user_actions=user_actions if has_user else frozenset(),
        agent_actions=_materialize_agent_actions(
            agent_actions=agent_actions,
            short_circuited=agent_owner_short_circuited,
        ) if has_agent else frozenset(),
        limiting_side=limiting,
        user_trails=user_trails_out,
        agent_trails=agent_trails_out,
        agent_owner_short_circuited=agent_owner_short_circuited,
    )


def _classify_limiting_side(
    *,
    user_actions: frozenset[str],
    agent_actions: frozenset[str],
    agent_owner_short_circuited: bool,
) -> LimitingSide:
    """name which side is capping the intersection.

    rules:

    - agent owner shortcut active -> :attr:`LimitingSide.USER`. the
      agent has every action by ownership; the user side is
      necessarily the cap (its set is finite, agent's is open-ended).
    - either side empty -> :attr:`LimitingSide.NEITHER` (caller
      already forced the deny upstream; this branch is unreachable
      from :func:`_assemble_result`'s call site, kept for clarity).
    - sets equal -> :attr:`LimitingSide.EQUAL`.
    - user_actions strictly subset of agent_actions ->
      :attr:`LimitingSide.USER`.
    - agent_actions strictly subset of user_actions ->
      :attr:`LimitingSide.AGENT`.
    - sets overlap but neither subset of the other -> the side with
      the smaller size wins; ties break toward
      :attr:`LimitingSide.AGENT` (agent caps are the more common
      "i'm an admin but the agent is read-only" answer the operator
      wants surfaced first).

    :param user_actions: user-side contribution
    :ptype user_actions: frozenset[str]
    :param agent_actions: agent-side contribution (may be the open-
        ended sentinel when ``agent_owner_short_circuited``)
    :ptype agent_actions: frozenset[str]
    :param agent_owner_short_circuited: True iff the agent side
        resolved via namespace ownership
    :ptype agent_owner_short_circuited: bool
    :return: classification
    :rtype: LimitingSide
    """
    result: LimitingSide
    if agent_owner_short_circuited:
        result = LimitingSide.USER
    elif not user_actions or not agent_actions:
        result = LimitingSide.NEITHER
    elif user_actions == agent_actions:
        result = LimitingSide.EQUAL
    elif user_actions < agent_actions:
        result = LimitingSide.USER
    elif agent_actions < user_actions:
        result = LimitingSide.AGENT
    elif len(agent_actions) <= len(user_actions):
        result = LimitingSide.AGENT
    else:
        result = LimitingSide.USER
    return result


def _materialize_agent_actions(
    agent_actions: frozenset[str],
    short_circuited: bool,
) -> frozenset[str]:
    """flatten the owner-shortcircuit sentinel for caller-visible reporting.

    the owner shortcut uses the open-ended :class:`_AllActionsSet`
    sentinel internally so intersection works correctly without an
    enumerated action vocabulary. the public-facing
    :class:`EvaluationResult` should not surface the sentinel
    instance, so we coerce it to a regular frozenset of the actions
    the result actually exposed. when the agent side did not
    short-circuit, the input is a plain frozenset and is returned
    as-is.

    :param agent_actions: agent-side action set (may be the sentinel)
    :ptype agent_actions: frozenset[str]
    :param short_circuited: whether the input is the sentinel
    :ptype short_circuited: bool
    :return: plain frozenset suitable for serialization
    :rtype: frozenset[str]
    """
    result: frozenset[str]
    if short_circuited:
        # the open-ended set has no enumerable contents; downstream
        # consumers should read agent_owner_short_circuited and treat
        # this as "every action."
        result = frozenset()
    else:
        result = agent_actions
    return result
