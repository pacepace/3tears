"""rbac evaluator: pure-functional resolver for "can actor do action on namespace".

two entry points share one resolution algorithm:

- :func:`evaluate_decision` — production hot path. returns a bool.
  consults the :class:`AclCache` membership and per-namespace layers
  before falling back to the loaders the cache carries; cached entries
  let the bulk of authorize hits stay in process memory.
- :func:`evaluate_with_trail` — introspection / audit path. returns
  the full :class:`EvaluationResult` with every contributing
  ``(group, assignment, role)`` chain. trail-mode reads the same
  cache layers and stores the produced trails alongside the action
  set so subsequent decision-mode lookups serve from cache.

resolution rules (in order):

1. **agent ownership short-circuit** — if namespace's
   ``owner_agent_id`` matches calling agent, agent side
   short-circuits to "every action allowed" without ever touching
   membership / grant tables. ownership is a property of the
   namespace row, not a grant.

2. **side resolution** — for each side caller supplied
   (user side iff ``user_id`` set; agent side iff ``agent_id`` set):

     a. consult :meth:`AclCache.get_membership`; on miss call the
        cache's :class:`MembershipLoader` and write back via
        :meth:`AclCache.put_membership`.
     b. for each eligible group, consult
        :meth:`AclCache.get_group_namespace`; on miss walk the cache's
        :class:`GrantLoader` for the group + namespace and write back
        via :meth:`AclCache.put_group_namespace`. trails are stored
        alongside actions so trail-mode and decision-mode share one
        cached value.
     c. union the per-group action sets and trails into the side's
        contribution.

3. **intersection** — when both sides ran, effective action set
   is ``user_actions ∩ agent_actions``. either side empty produces
   a deny. when only one side ran, its action set is effective
   set as-is.

4. **decision** — allow iff ``ctx.action`` is in effective set.

cross-customer wall: every resolver that produces a non-empty action
set requires contributing group to be platform-scoped or to share
namespace's customer. membership loader is responsible for
returning correct customer_id on each membership row; evaluator
double-checks at side-resolution boundary so a buggy loader cannot
leak grants across customer lines.

mixed-membership groups (a group with both user and agent members)
are handled correctly by side split: user side only counts
groups in which actor is a user member, agent side only
counts groups in which actor is an agent member. one group can
contribute to both sides simultaneously when both members happen to
be calling, but never to same side via wrong member type.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Iterable, Literal
from uuid import UUID

from threetears.agent.acl.cache import (
    AclCache,
    ActorMembershipKey,
    GroupNamespaceKey,
)
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
from threetears.observe import get_logger, traced

__all__ = [
    "READ_FILE_MATCHING_PREFIX",
    "WRITE_FILE_MATCHING_PREFIX",
    "evaluate_decision",
    "evaluate_file_access",
    "evaluate_with_trail",
]


#: action-string prefix declaring "caller may read files whose path
#: matches the suffix glob." namespace-task-01 phase 7 path-level gate.
#: the suffix is a :class:`pathlib.PurePosixPath.full_match` glob with
#: ``**`` recursion support, stored verbatim on the role's ``workspace``
#: permissions bucket: ``"workspace": ["read_file_matching:**/*.yaml"]``.
READ_FILE_MATCHING_PREFIX = "read_file_matching:"


#: action-string prefix declaring "caller may write files whose path
#: matches the suffix glob." symmetric to
#: :data:`READ_FILE_MATCHING_PREFIX`; suffix uses same glob syntax.
WRITE_FILE_MATCHING_PREFIX = "write_file_matching:"

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------


@traced
async def evaluate_decision(
    ctx: EvaluationContext,
    *,
    cache: AclCache,
) -> bool:
    """resolve "is ``ctx.action`` allowed?" and return bool.

    thin wrapper around :func:`evaluate_with_trail` that throws away
    trail. cache-friendly because the underlying side resolver
    serves cached membership and per-namespace contributions from
    the supplied :class:`AclCache`; misses fall through to the
    cache's :attr:`AclCache.membership_loader` /
    :attr:`AclCache.grant_loader` and are written back.

    :param ctx: evaluation context (namespace, action, optional
        ``user_id`` and / or ``agent_id``)
    :ptype ctx: EvaluationContext
    :param cache: shared :class:`AclCache` carrying loaders + ttl
        layers; required, no silent-bypass path
    :ptype cache: AclCache
    :return: True iff action is allowed
    :rtype: bool
    """
    result = await evaluate_with_trail(ctx, cache=cache)
    return result.decision


@traced
async def evaluate_with_trail(
    ctx: EvaluationContext,
    *,
    cache: AclCache,
) -> EvaluationResult:
    """resolve question and return full :class:`EvaluationResult`.

    walks membership + grant graph deterministically and packages
    every contributing chain as a :class:`Trail`. when both sides are
    supplied, populates :attr:`EvaluationResult.user_actions`,
    :attr:`agent_actions`, :attr:`limiting_side`, and per-side
    trails. when only one side is supplied, populates
    :attr:`EvaluationResult.trails` and leaves intersection
    fields empty.

    consults :class:`AclCache` for membership and per-namespace grant
    layers; on miss falls back to cache's loaders and writes results
    back. trail rows are cached alongside action sets so successive
    decision-mode and trail-mode calls for same actor + namespace
    serve from cache.

    :param ctx: evaluation context
    :ptype ctx: EvaluationContext
    :param cache: shared :class:`AclCache`
    :ptype cache: AclCache
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
            cache=cache,
        )

    if ctx.agent_id is not None:
        if ctx.namespace.owner_agent_id == ctx.agent_id:
            # ownership short-circuit: agent side is implicitly full
            # access. record a sentinel "every action" set and skip
            # loader trip. trail surface stays empty because there
            # is no grant to point at.
            agent_actions = _OWNER_ACTIONS
            agent_owner_short_circuited = True
        else:
            agent_actions, agent_trails = await _resolve_side(
                actor_id=ctx.agent_id,
                member_type=MemberType.AGENT,
                namespace=ctx.namespace,
                cache=cache,
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


@traced
async def evaluate_file_access(
    *,
    namespace: Namespace,
    user_id: UUID | None,
    agent_id: UUID | None,
    path: str,
    direction: Literal["read", "write"],
    cache: AclCache,
) -> bool:
    """resolve "may actor read/write ``path`` in ``namespace``?" (path-glob gate).

    namespace-task-01 phase 7 path-level rbac gate for workspace
    sandbox. workspace-as-namespace model encodes path-level grants
    via custom action types
    :data:`READ_FILE_MATCHING_PREFIX`\\ ``<glob>`` and
    :data:`WRITE_FILE_MATCHING_PREFIX`\\ ``<glob>``. standard
    :func:`evaluate_with_trail` resolves granted-action set for
    caller; this helper inspects that set, filters to
    direction-appropriate prefix, and
    :meth:`pathlib.PurePosixPath.full_match`es each suffix glob against
    ``path``. allow iff any glob matches.

    resolution details:

    1. :func:`evaluate_with_trail` is called with
       ``action=READ_FILE_MATCHING_PREFIX`` (or write equivalent)
       so evaluator reduces its contribution walk to actor's
       ``workspace``-bucket action set without an exact-action match
       short-circuit mid-walk. ``ctx.action`` passed is a prefix
       stem; decision bool coming out of evaluator is ignored
       because action set is what we want.

    2. on agent owner short-circuit agent side is "every
       action" — we treat that as "every path" and return True without
       running glob match.

    3. effective action set is computed from whichever sides
       caller supplied (user-only, agent-only, or intersection). for
       each action string starting with direction prefix, suffix
       is tested via :meth:`PurePosixPath.full_match` against
       ``path``. first match returns True; exhausting set
       returns False.

    :param namespace: workspace-type namespace file belongs to
    :ptype namespace: Namespace
    :param user_id: invoking user UUID, or ``None`` for agent-only
        evaluation
    :ptype user_id: UUID | None
    :param agent_id: invoking agent UUID, or ``None`` for user-only
        evaluation
    :ptype agent_id: UUID | None
    :param path: workspace-relative file path to test
    :ptype path: str
    :param direction: ``"read"`` or ``"write"``
    :ptype direction: Literal["read", "write"]
    :param cache: shared :class:`AclCache`
    :ptype cache: AclCache
    :return: True iff actor's grants include a glob matching ``path``
    :rtype: bool
    :raises ValueError: if ``direction`` is not ``"read"``/``"write"``
        or if neither ``user_id`` nor ``agent_id`` is supplied
    """
    if direction not in ("read", "write"):
        raise ValueError(
            f"evaluate_file_access direction must be 'read' or 'write', got {direction!r}",
        )
    if user_id is None and agent_id is None:
        raise ValueError(
            "evaluate_file_access requires at least one of user_id or agent_id",
        )

    prefix = READ_FILE_MATCHING_PREFIX if direction == "read" else WRITE_FILE_MATCHING_PREFIX
    # ``ctx.action`` is a sentinel stem — decision bool from
    # :func:`evaluate_with_trail` is ignored. what matters is
    # action-set surfaces: :attr:`EvaluationResult.user_actions`,
    # :attr:`agent_actions`, and
    # :attr:`agent_owner_short_circuited` flag for ownership.
    ctx = EvaluationContext(
        namespace=namespace,
        action=prefix,
        user_id=user_id,
        agent_id=agent_id,
    )
    trail_result = await evaluate_with_trail(ctx, cache=cache)

    # owner short-circuit: agent owns namespace -> every action
    # + every path on agent side. when only agent side ran this
    # is an immediate allow; on intersection user side still
    # caps (intersection narrows owner-implicit wildcard to
    # user's actual action set, so we fall through to glob walk
    # over user-side contribution).
    has_user = user_id is not None
    has_agent = agent_id is not None
    if trail_result.agent_owner_short_circuited and not has_user:
        return True

    # collect every action string actor could contribute. when
    # intersecting, only actions present on user side matter (
    # agent short-circuit means agent side is open-ended); when
    # user-only, single-side ``trails`` carry contributions
    # and flat ``effective_actions`` is correct.
    action_strings: set[str] = set()
    if trail_result.user_trails:
        for trail in trail_result.user_trails:
            action_strings.update(trail.contributed_actions)
    if trail_result.trails:
        for trail in trail_result.trails:
            action_strings.update(trail.contributed_actions)
    if trail_result.agent_trails and not has_user:
        # agent-only non-owner evaluation: agent-side trails carry
        # contributions. (intersection path caps by user so
        # agent-side trails are redundant there.)
        for trail in trail_result.agent_trails:
            action_strings.update(trail.contributed_actions)

    posix_path = PurePosixPath(path)
    decision = False
    for action in action_strings:
        if not action.startswith(prefix):
            continue
        glob = action[len(prefix) :]
        if not glob:
            continue
        if posix_path.full_match(glob):
            decision = True
            break
    return decision


# ---------------------------------------------------------------------------
# owner sentinel
# ---------------------------------------------------------------------------


class _AllActionsSet(frozenset[str]):
    """sentinel set that contains every action.

    agent ownership short-circuit needs to express "agent has
    every action" without enumerating an open-ended action vocabulary.
    a real frozenset would have to list every possible action up
    front; this subclass overrides ``__contains__`` and
    intersection operator so any string is considered a member and
    intersection with another set is just other set.

    instances stay empty as a frozenset for ``len()`` / iteration
    purposes (those are not exercised on hot path); equality and
    hashing inherit from frozenset so callers may still use
    sentinel as a dict key. only :meth:`__contains__` and
    :meth:`__and__` / :meth:`__rand__` are overridden because those
    are only operations evaluator runs against action
    set on agent side.
    """

    def __contains__(self, item: object) -> bool:
        """every action string is a member; non-strings are not.

        :param item: candidate value to test for membership
        :ptype item: object
        :return: True iff item is a string
        :rtype: bool
        """
        return isinstance(item, str)

    # the ALL sentinel intentionally narrows frozenset.__and__ (other:
    # Iterable[str] vs the builtin's AbstractSet[object]) so ALL ∩ X == X
    # keeps the str element type that AbstractSet[object] would erase; the
    # LSP override warning is expected for this deliberate sentinel.
    def __and__(self, other: Iterable[str]) -> frozenset[str]:  # type: ignore[override]
        """intersection with any iterable returns that iterable as a frozenset.

        because sentinel "contains" every action, intersection
        ``ALL ∩ X`` is just ``X``. used to cap user side by
        agent's owner-shortcircuit set: user can do whatever
        user-side grants allow, no further reduction.

        :param other: action set to intersect with
        :ptype other: Iterable[str]
        :return: frozenset of other side
        :rtype: frozenset[str]
        """
        return frozenset(other)

    def __rand__(self, other: Iterable[str]) -> frozenset[str]:
        """reflected intersection — see :meth:`__and__`.

        :param other: action set on left
        :ptype other: Iterable[str]
        :return: frozenset of other side
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
    cache: AclCache,
) -> tuple[frozenset[str], tuple[Trail, ...]]:
    """compute one side's action set + trails for ``actor_id``, cache-aware.

    runs four-step pipeline against :class:`AclCache`:

    1. consult membership layer for ``(actor_kind, actor_id)``; on
       miss, call cache's :class:`MembershipLoader` and write back.
    2. filter memberships by member_type (a user-side resolution
       only counts memberships where actor is a user member,
       and vice versa) and by customer (cross-customer wall).
    3. for each eligible group, consult per-namespace layer for
       ``(group_id, namespace.id)``; on miss, ask cache's
       :class:`GrantLoader` for assignments + roles + groups
       restricted to ``group_id``, walk them, and write back the
       resulting ``(actions, trails)`` pair.
    4. union per-group contributions into side's action set + trail
       tuple.

    :param actor_id: user UUID or agent UUID under evaluation
    :ptype actor_id: UUID
    :param member_type: user or agent — drives both loader call
        and membership filter
    :ptype member_type: MemberType
    :param namespace: namespace under evaluation
    :ptype namespace: Namespace
    :param cache: shared :class:`AclCache` carrying loaders + layers
    :ptype cache: AclCache
    :return: ``(action_set, trails)`` pair for this side
    :rtype: tuple[frozenset[str], tuple[Trail, ...]]
    """
    actor_kind = "user" if member_type == MemberType.USER else "agent"
    membership_key = ActorMembershipKey(
        actor_kind=actor_kind,
        actor_id=actor_id,
    )
    membership_entry = cache.get_membership(membership_key)
    if membership_entry is not None:
        memberships = membership_entry.memberships
    else:
        if member_type == MemberType.USER:
            memberships = await cache.membership_loader.load_for_user(actor_id)
        else:
            memberships = await cache.membership_loader.load_for_agent(actor_id)
        cache.put_membership(membership_key, memberships)

    eligible = _filter_memberships(
        memberships=memberships,
        actor_id=actor_id,
        member_type=member_type,
        namespace=namespace,
    )
    eligible_group_ids = tuple(_unique_ordered(m.group_id for m in eligible))
    if not eligible_group_ids:
        return frozenset(), ()

    # per-group resolution: cache the (group, namespace) contribution
    # so two evaluations against the same namespace re-using the same
    # group serve the second from cache. the layer is keyed on the
    # group id, so different actors who happen to share a group also
    # benefit.
    actions_acc: set[str] = set()
    trails_acc: list[Trail] = []
    for group_id in eligible_group_ids:
        ns_key = GroupNamespaceKey(
            group_id=group_id,
            namespace_id=namespace.id,
        )
        ns_entry = cache.get_group_namespace(ns_key)
        if ns_entry is not None:
            actions_acc.update(ns_entry.actions)
            trails_acc.extend(ns_entry.trails)
            continue

        group_actions, group_trails = await _resolve_group_for_namespace(
            group_id=group_id,
            namespace=namespace,
            cache=cache,
        )
        cache.put_group_namespace(ns_key, group_actions, group_trails)
        actions_acc.update(group_actions)
        trails_acc.extend(group_trails)

    # deterministic trail ordering across groups so two runs with the
    # same dataset produce the same trail order.
    trails_acc.sort(
        key=lambda t: (
            str(t.group.id),
            str(t.assignment.id),
        ),
    )
    return frozenset(actions_acc), tuple(trails_acc)


async def _resolve_group_for_namespace(
    *,
    group_id: UUID,
    namespace: Namespace,
    cache: AclCache,
) -> tuple[frozenset[str], tuple[Trail, ...]]:
    """resolve one group's contribution against ``namespace`` via loaders.

    cache-miss path for the per-namespace layer. asks the cache's
    :class:`GrantLoader` for assignments held by ``group_id`` covering
    ``namespace``, plus the roles and group rows the trail builder
    needs, then walks them.

    :param group_id: group to resolve
    :ptype group_id: UUID
    :param namespace: namespace under evaluation
    :ptype namespace: Namespace
    :param cache: shared :class:`AclCache`
    :ptype cache: AclCache
    :return: ``(action_set, trails)`` pair for this group
    :rtype: tuple[frozenset[str], tuple[Trail, ...]]
    """
    group_ids = (group_id,)
    assignments = await cache.grant_loader.load_assignments_for_groups(
        group_ids=group_ids,
        namespace=namespace,
    )
    role_ids = tuple(_unique_ordered(a.role_id for a in assignments))
    roles = await cache.grant_loader.load_roles(role_ids) if role_ids else {}
    groups_raw = await cache.grant_loader.load_groups(group_ids)
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

    - membership's :attr:`GroupMembership.member_id` equals
      ``actor_id`` (defensive — loader was already given
      actor id but a buggy loader could return rows for wrong
      actor).
    - membership's :attr:`GroupMembership.member_type` equals
      ``member_type`` (user side never counts agent memberships
      and vice versa; mixed groups split correctly).
    - membership's :attr:`GroupMembership.customer_id` either
      matches namespace's customer (same-customer grant) or is
      ``None`` (membership row is platform-scoped — admin user
      or platform-managed agent — and may reach across customers
      via a platform-scoped group).

    cross-customer cut is second line of defense behind
    loader's own filtering. if loader is wrong, filter still
    refuses to count a customer-scoped membership against another
    customer's namespace.

    :param memberships: raw memberships from loader
    :ptype memberships: tuple[GroupMembership, ...]
    :param actor_id: caller's UUID
    :ptype actor_id: UUID
    :param member_type: side under evaluation
    :ptype member_type: MemberType
    :param namespace: namespace under evaluation
    :ptype namespace: Namespace
    :return: memberships that survived filter
    :rtype: tuple[GroupMembership, ...]
    """
    surviving: list[GroupMembership] = []
    for membership in memberships:
        if membership.member_id != actor_id:
            continue
        if membership.member_type != member_type:
            continue
        if membership.customer_id is not None and membership.customer_id != namespace.customer_id:
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
    """assemble action set and trail tuple for one side.

    iteration is deterministic in (group_id, assignment_id) order so
    trails come out repeatable across runs — operators reading two
    explanations side-by-side see same row order both times.
    only assignments whose ``group_id`` is in ``eligible_group_ids``
    are considered (defensive: grant loader may over-return).

    cross-customer + platform-scope check on group: a
    customer-scoped group must match namespace's customer for
    its assignments to count; a platform-scoped group counts
    universally.

    :param assignments: role assignments held by eligible groups
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
                extra={
                    "extra_data": {
                        "assignment_id": str(assignment.id),
                        "group_id": str(assignment.group_id),
                    }
                },
            )
            continue
        if group.customer_id is not None and group.customer_id != namespace.customer_id:
            # group is customer-scoped to a different customer; an
            # assignment it holds cannot reach this namespace.
            continue
        role = roles.get(assignment.role_id)
        if role is None:
            log.debug(
                "skipping assignment with unresolved role",
                extra={
                    "extra_data": {
                        "assignment_id": str(assignment.id),
                        "role_id": str(assignment.role_id),
                    }
                },
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
    with same three attributes. loader Protocol returns
    ``object`` to keep callers free to pass their own ORM rows; this
    coercion happens once per evaluation so trail builder sees a
    uniform type.

    objects that do not expose three attributes (``id``, ``name``,
    ``customer_id``) are dropped from result; trail walker
    treats their absence as "skip this assignment."

    :param raw: loader-provided mapping
    :ptype raw: dict[UUID, object]
    :return: normalized mapping keyed by same group ids
    :rtype: dict[UUID, Group]
    """
    result: dict[UUID, Group] = {}
    for group_id, value in raw.items():
        if isinstance(value, Group):
            result[group_id] = value
        elif hasattr(value, "id") and hasattr(value, "name") and hasattr(value, "customer_id"):
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
    order, which makes test fixtures harder to reason about and
    hurts deterministic trail assembly downstream.

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
    """wrap per-side action sets into an :class:`EvaluationResult`.

    branches by which sides ran:

    - both sides ran (intersection mode): effective set is
      ``user_actions ∩ agent_actions``; populates
      :attr:`EvaluationResult.user_trails`,
      :attr:`agent_trails`, :attr:`limiting_side`. either side empty
      forces a deny and :attr:`LimitingSide.NEITHER`.
    - only user side ran: effective set is ``user_actions``; populates
      :attr:`EvaluationResult.trails`.
    - only agent side ran: effective set is ``agent_actions`` (with
      owner short-circuit translated to "every action" via
      sentinel set's ``__contains__``); populates :attr:`trails`.

    decision = ``ctx.action in effective_actions``.

    :param ctx: original evaluation context
    :ptype ctx: EvaluationContext
    :param user_actions: action set user side resolved to
    :ptype user_actions: frozenset[str]
    :param agent_actions: action set agent side resolved to
    :ptype agent_actions: frozenset[str]
    :param user_trails: user-side grant chains
    :ptype user_trails: tuple[Trail, ...]
    :param agent_trails: agent-side grant chains
    :ptype agent_trails: tuple[Trail, ...]
    :param agent_owner_short_circuited: True iff agent side was
        resolved via namespace ownership shortcut
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
        if not user_actions or (not agent_actions and not agent_owner_short_circuited):
            effective = frozenset()
            limiting = LimitingSide.NEITHER
        else:
            # owner shortcut: agent_actions is open-ended sentinel;
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
        # agent-only evaluation. with owner shortcut, every action
        # is allowed; we surface "the requested action" as
        # effective set so bool decision works without
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
        )
        if has_agent
        else frozenset(),
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
    """name which side is capping intersection.

    rules:

    - agent owner shortcut active -> :attr:`LimitingSide.USER`.
      agent has every action by ownership; user side is
      necessarily cap (its set is finite, agent's is open-ended).
    - either side empty -> :attr:`LimitingSide.NEITHER` (caller
      already forced deny upstream; this branch is unreachable
      from :func:`_assemble_result`'s call site, kept for clarity).
    - sets equal -> :attr:`LimitingSide.EQUAL`.
    - user_actions strictly subset of agent_actions ->
      :attr:`LimitingSide.USER`.
    - agent_actions strictly subset of user_actions ->
      :attr:`LimitingSide.AGENT`.
    - sets overlap but neither subset of other -> side with
      smaller size wins; ties break toward
      :attr:`LimitingSide.AGENT` (agent caps are more common
      "i'm an admin but agent is read-only" answer operator
      wants surfaced first).

    :param user_actions: user-side contribution
    :ptype user_actions: frozenset[str]
    :param agent_actions: agent-side contribution (may be open-
        ended sentinel when ``agent_owner_short_circuited``)
    :ptype agent_actions: frozenset[str]
    :param agent_owner_short_circuited: True iff agent side
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
    """flatten owner-shortcircuit sentinel for caller-visible reporting.

    owner shortcut uses open-ended :class:`_AllActionsSet`
    sentinel internally so intersection works correctly without an
    enumerated action vocabulary. public-facing
    :class:`EvaluationResult` should not surface sentinel
    instance, so we coerce it to a regular frozenset of actions
    result actually exposed. when agent side did not
    short-circuit, input is a plain frozenset and is returned
    as-is.

    :param agent_actions: agent-side action set (may be sentinel)
    :ptype agent_actions: frozenset[str]
    :param short_circuited: whether input is sentinel
    :ptype short_circuited: bool
    :return: plain frozenset suitable for serialization
    :rtype: frozenset[str]
    """
    result: frozenset[str]
    if short_circuited:
        # open-ended set has no enumerable contents; downstream
        # consumers should read agent_owner_short_circuited and treat
        # this as "every action."
        result = frozenset()
    else:
        result = agent_actions
    return result
