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
- :func:`evaluate_many_with_trail` -- many trail-mode questions at once
  (an audit snapshot across every namespace an actor reaches). the
  answers are :func:`evaluate_with_trail`'s, which is its one-question
  case; the reads are shared across the questions (step 2b).

resolution rules (in order):

1. **ownership short-circuit** — if the namespace's
   ``owner_namespace`` is the namespace the calling agent IS, agent
   side short-circuits to "every action allowed" without ever
   touching membership / grant tables. ownership is a property of
   the namespace row, not a grant.

   the owner is a NAMESPACE rather than an agent because a tool pod,
   a capability source and an in-hub component own namespaces too and
   none of them is an agent. the caller's own namespace name is
   DERIVED from its agent id in process, so the short-circuit still
   costs no i/o -- which is a requirement rather than an optimisation:
   an agent pod's sandboxed L3 search_path is its own schema, which
   carries no ``namespaces`` table, and the owner path is the one path
   that must work there.

   ``owner_namespace`` of ``None`` matches NOTHING, and the guard is
   explicit rather than implied. under the previous ``owner_agent_id``
   comparison an unowned namespace could not match because the
   calling agent id was non-null by the enclosing branch; that
   accident does not carry over, because the caller's resolved
   namespace id can itself be ``None``, and ``None == None`` would
   hand every unowned namespace to any agent whose own namespace could
   not be resolved.

2. **side resolution** — for each side caller supplied
   (user side iff ``user_id`` set; agent side iff ``agent_id`` set; a
   group-member side iff ``group_id`` set, which runs alone and starts
   from that group instead of loading memberships):

     a. consult :meth:`AclCache.get_membership`; on miss call the
        cache's :class:`MembershipLoader` and write back via
        :meth:`AclCache.put_membership`.
     b. for each eligible group, consult
        :meth:`AclCache.get_group_namespace`; the misses are read from
        the cache's :class:`GrantLoader` together -- every missing
        group's assignments in one read (one per namespace when the
        loader is not a :class:`HeldAssignmentsLoader`), their roles and
        groups in one read each -- walked per group + namespace, and
        written back via :meth:`AclCache.put_group_namespace`. trails
        are stored alongside actions so trail-mode and decision-mode
        share one cached value.
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

second, independent wall — role ownership: a customer-authored
:class:`~threetears.agent.acl.types.Role` (``customer_id`` set)
contributes nothing against a namespace belonging to a different
customer, checked in :func:`_walk_assignments`. the group wall does
not cover this case: a PLATFORM-scoped group has ``customer_id is
None`` and passes the group check against every customer, so an
assignment binding such a group to a customer-authored role would
otherwise carry that customer's role everywhere. built-in and
operator-authored platform roles carry ``customer_id is None`` and are
unaffected — their evaluation is identical before and after this wall
existed.

mixed-membership groups (a group with both user and agent members)
are handled correctly by side split: user side only counts
groups in which actor is a user member, agent side only
counts groups in which actor is an agent member. one group can
contribute to both sides simultaneously when both members happen to
be calling, but never to same side via wrong member type.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, Literal
from uuid import UUID

from threetears.agent.acl.cache import (
    AclCache,
    ActorMembershipKey,
    GroupNamespaceKey,
)
from threetears.agent.acl.loader import HeldAssignmentsLoader
from threetears.agent.acl.types import (
    EvaluationContext,
    EvaluationResult,
    Group,
    GroupMembership,
    LimitingSide,
    MAX_GROUP_MEMBERSHIP_DEPTH,
    MemberType,
    Namespace,
    Role,
    RoleAssignment,
    Trail,
)
from threetears.core.namespaces import build_agent_namespace_name
from threetears.observe import get_logger, traced

__all__ = [
    "READ_FILE_MATCHING_PREFIX",
    "WRITE_FILE_MATCHING_PREFIX",
    "evaluate_decision",
    "evaluate_file_access",
    "evaluate_many_with_trail",
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

    :param ctx: evaluation context; :class:`EvaluationContext` says
        which actors it may carry and in which combinations
    :ptype ctx: EvaluationContext
    :param cache: shared :class:`AclCache` carrying loaders + its layers
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
    :attr:`EvaluationResult.trails` and that side's action set, and
    leaves the other side and :attr:`limiting_side` at their defaults.

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
    :raises ValueError: when ``ctx`` carries none of ``user_id``,
        ``agent_id`` or ``group_id`` (an evaluation needs an actor), or
        carries ``group_id`` alongside a user or an agent (a group
        member is answered alone)
    """
    results = await evaluate_many_with_trail((ctx,), cache=cache)
    return results[0]


@traced
async def evaluate_many_with_trail(
    contexts: Sequence[EvaluationContext],
    *,
    cache: AclCache,
) -> list[EvaluationResult]:
    """answer many questions together; each answer is :func:`evaluate_with_trail`'s.

    the questions share their reads. each side's group set is planned once per actor and
    customer (the customer wall is the only thing a namespace changes about it), and every
    (group, namespace) contribution the cache does not hold is computed from one read of the
    assignments those groups hold, one of their roles and one of the groups -- not one read
    per group per namespace. a reader of hundreds of namespaces through dozens of groups is
    a handful of round trips rather than tens of thousands.

    what is computed is stored in the cache's per-namespace layer under the read fence, as a
    single evaluation stores it, and used for this call whether or not the cache keeps it.

    :param contexts: the questions, each validated as :func:`evaluate_with_trail` validates it
    :ptype contexts: Sequence[EvaluationContext]
    :param cache: shared :class:`AclCache` carrying loaders + layers
    :ptype cache: AclCache
    :return: one result per context, in order
    :rtype: list[EvaluationResult]
    :raises ValueError: when a context names no actor, or a group alongside a person
    """
    for ctx in contexts:
        _check_context(ctx)
    planner = _GroupSetPlanner(cache)
    plans: list[_SidePlan] = []
    for ctx in contexts:
        plans.append(await planner.plan(ctx))
    wanted: dict[GroupNamespaceKey, Namespace] = {}
    for ctx, plan in zip(contexts, plans, strict=True):
        for group_id in (*plan.user_groups, *plan.agent_groups):
            wanted[GroupNamespaceKey(group_id=group_id, namespace_id=ctx.namespace.id)] = ctx.namespace
    contributions = await _group_namespace_contributions(wanted, cache)
    results: list[EvaluationResult] = []
    for ctx, plan in zip(contexts, plans, strict=True):
        user_actions, user_trails = _accumulate_groups(plan.user_groups, ctx.namespace, contributions)
        agent_actions: frozenset[str] = _OWNER_ACTIONS if plan.agent_owner_short_circuited else frozenset()
        agent_trails: tuple[Trail, ...] = ()
        if plan.agent_groups:
            agent_actions, agent_trails = _accumulate_groups(plan.agent_groups, ctx.namespace, contributions)
        results.append(
            _assemble_result(
                ctx=ctx,
                user_actions=user_actions,
                agent_actions=agent_actions,
                user_trails=user_trails,
                agent_trails=agent_trails,
                agent_owner_short_circuited=plan.agent_owner_short_circuited,
            )
        )
    return results


def _check_context(ctx: EvaluationContext) -> None:
    """refuse a question that names no actor, or a group alongside a person.

    :param ctx: evaluation context
    :ptype ctx: EvaluationContext
    :return: nothing
    :rtype: None
    :raises ValueError: when the context's actors are not a question the evaluator answers
    """
    if ctx.group_id is not None and (ctx.user_id is not None or ctx.agent_id is not None):
        raise ValueError(
            "evaluate_with_trail answers for a group member alone; group_id cannot be combined with user_id or agent_id",
        )
    if ctx.user_id is None and ctx.agent_id is None and ctx.group_id is None:
        raise ValueError(
            "evaluate_with_trail requires at least one of user_id, agent_id or group_id",
        )


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
       so evaluator surfaces actor's ``workspace``-bucket action
       set on each side without an exact-action match short-circuit
       mid-walk. ``ctx.action`` passed is a prefix stem; decision
       bool coming out of evaluator is ignored because per-side
       action sets are what we glob against.

    2. on agent owner short-circuit agent side is "every
       action" — we treat that as "every path" so agent side
       matches unconditionally.

    3. sides are AND-ed, mirroring the action-set intersection
       :func:`evaluate_with_trail` performs for exact actions. when
       both a user and an agent are on the call, ``path`` must be
       matched by a glob on user side AND by a glob on agent
       side: a broad user grant never widens access past agent's
       own grant (agent's workspace sandbox), and a
       missing/narrower agent grant constrains result. when only
       one side is supplied that side alone decides. each candidate
       action string starting with direction prefix has its suffix
       tested via :meth:`PurePosixPath.full_match` against ``path``.

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

    has_user = user_id is not None
    has_agent = agent_id is not None
    posix_path = PurePosixPath(path)

    # owner short-circuit: agent owns namespace -> every action + every
    # path on agent side, so the agent side matches unconditionally.
    agent_side_open = has_agent and trail_result.agent_owner_short_circuited

    if has_user and has_agent:
        # intersection: the path must be granted on BOTH sides, mirroring
        # the ``user_actions ∩ agent_actions`` cut :func:`evaluate_with_trail`
        # applies to exact actions. the agent side is a genuine constraint
        # here — a broad user grant must not widen access past what the
        # agent is itself granted (the agent's workspace sandbox), and a
        # missing/narrower agent grant denies. owner short-circuit is the
        # only case where the agent side matches every path.
        user_ok = _path_matches_trails(
            trails=trail_result.user_trails,
            prefix=prefix,
            posix_path=posix_path,
        )
        agent_ok = agent_side_open or _path_matches_trails(
            trails=trail_result.agent_trails,
            prefix=prefix,
            posix_path=posix_path,
        )
        decision = user_ok and agent_ok
    elif has_agent:
        # agent-only evaluation. owner short-circuit permits every path;
        # otherwise the agent's single-side trails carry contributions.
        decision = agent_side_open or _path_matches_trails(
            trails=trail_result.trails,
            prefix=prefix,
            posix_path=posix_path,
        )
    else:
        # user-only evaluation; single-side trails carry contributions.
        decision = _path_matches_trails(
            trails=trail_result.trails,
            prefix=prefix,
            posix_path=posix_path,
        )
    return decision


def _path_matches_trails(
    *,
    trails: tuple[Trail, ...],
    prefix: str,
    posix_path: PurePosixPath,
) -> bool:
    """test whether any direction-prefixed glob in ``trails`` matches ``posix_path``.

    walks every contributed action across ``trails``, keeps those
    beginning with ``prefix`` (the direction-appropriate
    ``read_file_matching:`` / ``write_file_matching:`` stem),
    strips the stem, and :meth:`pathlib.PurePosixPath.full_match`es
    the remaining glob suffix against ``posix_path``. empty suffixes
    are skipped. first match wins.

    :param trails: grant chains contributing one side's action set
    :ptype trails: tuple[Trail, ...]
    :param prefix: direction-appropriate action-string prefix
    :ptype prefix: str
    :param posix_path: workspace-relative path under evaluation
    :ptype posix_path: PurePosixPath
    :return: True iff any suffix glob matches ``posix_path``
    :rtype: bool
    """
    matched = False
    for trail in trails:
        for action in trail.contributed_actions:
            if not action.startswith(prefix):
                continue
            glob = action[len(prefix) :]
            if not glob:
                continue
            if posix_path.full_match(glob):
                matched = True
                break
        if matched:
            break
    return matched


# ---------------------------------------------------------------------------
# ownership
# ---------------------------------------------------------------------------


def _agent_owns_namespace(namespace: Namespace, agent_id: UUID) -> bool:
    """resolve whether the calling agent's own namespace OWNS ``namespace``.

    ``namespace.owner_namespace`` is compared against the name of the
    namespace the caller IS, derived in process by
    :func:`threetears.core.namespaces.build_agent_namespace_name`. no
    lookup happens: the derivation is the same rule the hub writes the
    owner column with, and the owner path has to answer inside an agent
    pod whose L3 cannot see the hub's ``namespaces`` at all.

    the ``is not None`` guard is EXPLICIT and load-bearing. the previous
    comparison was ``owner_agent_id == agent_id``, where a null owner
    could not match because the enclosing branch guaranteed a non-null
    agent id. that accident does not survive the move: two ``None``
    values compare EQUAL in python, so an unowned namespace would be
    handed to any caller whose own name came out null. the guard is
    what keeps "nothing owns this" meaning "nobody owns this".

    :param namespace: the namespace under evaluation
    :ptype namespace: Namespace
    :param agent_id: the calling agent
    :ptype agent_id: UUID
    :return: whether the caller owns the namespace
    :rtype: bool
    """
    owner_namespace = namespace.owner_namespace
    return owner_namespace is not None and owner_namespace == build_agent_namespace_name(agent_id)


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


@dataclass(frozen=True)
class _SidePlan:
    """the groups each side of one question reaches on its namespace, walls applied.

    :param user_groups: the user side's groups (or a group member's), depth-expanded
    :param agent_groups: the agent side's groups, empty when the agent owns the namespace
    :param agent_owner_short_circuited: whether the agent side is its owner's full set
    """

    user_groups: tuple[UUID, ...]
    agent_groups: tuple[UUID, ...]
    agent_owner_short_circuited: bool


class _GroupSetPlanner:
    """plan each question's group sets, once per actor and customer within one call.

    a side's group set depends on its namespace only through the cross-customer wall
    (:func:`_filter_memberships` reads ``namespace.customer_id`` and nothing else), so one
    plan serves every namespace of a customer. the plan is this call's working set, not a
    cache: the next call plans again, through the :class:`AclCache` membership layer as a
    single evaluation does.
    """

    def __init__(self, cache: AclCache) -> None:
        """
        :param cache: shared :class:`AclCache` carrying loaders + layers
        :ptype cache: AclCache
        :return: nothing
        :rtype: None
        """
        self._cache = cache
        self._planned: dict[tuple[str, UUID, UUID | None], tuple[UUID, ...]] = {}

    async def plan(self, ctx: EvaluationContext) -> _SidePlan:
        """the groups each side of ``ctx`` reaches.

        :param ctx: a validated evaluation context
        :ptype ctx: EvaluationContext
        :return: the plan
        :rtype: _SidePlan
        """
        user_groups: tuple[UUID, ...] = ()
        agent_groups: tuple[UUID, ...] = ()
        owner = False
        if ctx.group_id is not None:
            user_groups = await self._groups("group", ctx.group_id, ctx.namespace)
        if ctx.user_id is not None:
            user_groups = await self._groups("user", ctx.user_id, ctx.namespace)
        if ctx.agent_id is not None:
            owner = _agent_owns_namespace(ctx.namespace, ctx.agent_id)
            if not owner:
                agent_groups = await self._groups("agent", ctx.agent_id, ctx.namespace)
        return _SidePlan(user_groups=user_groups, agent_groups=agent_groups, agent_owner_short_circuited=owner)

    async def _groups(self, kind: str, actor_id: UUID, namespace: Namespace) -> tuple[UUID, ...]:
        """one side's depth-expanded groups on ``namespace``, planned once per customer.

        :param kind: ``user``, ``agent``, or ``group`` (a direct member of the group)
        :ptype kind: str
        :param actor_id: the actor, or the group whose direct member is evaluated
        :ptype actor_id: UUID
        :param namespace: namespace under evaluation
        :ptype namespace: Namespace
        :return: the side's groups, deterministic order
        :rtype: tuple[UUID, ...]
        """
        key = (kind, actor_id, namespace.customer_id)
        planned = self._planned.get(key)
        if planned is None:
            if kind == "group":
                direct: tuple[UUID, ...] = (actor_id,)
            else:
                direct = await _direct_groups(
                    actor_id=actor_id,
                    member_type=MemberType.USER if kind == "user" else MemberType.AGENT,
                    namespace=namespace,
                    cache=self._cache,
                )
            planned = await _expand_group_parents(direct_group_ids=direct, namespace=namespace, cache=self._cache)
            self._planned[key] = planned
        return planned


async def _direct_groups(
    *,
    actor_id: UUID,
    member_type: MemberType,
    namespace: Namespace,
    cache: AclCache,
) -> tuple[UUID, ...]:
    """the groups ``actor_id`` is a direct member of that may count on ``namespace``.

    consults the membership layer for ``(actor_kind, actor_id)``; on a miss, calls the
    cache's :class:`MembershipLoader` and writes back. memberships are then filtered by
    member type and by customer (the cross-customer wall).

    :param actor_id: user UUID or agent UUID under evaluation
    :ptype actor_id: UUID
    :param member_type: user or agent -- drives both loader call and membership filter
    :ptype member_type: MemberType
    :param namespace: namespace under evaluation
    :ptype namespace: Namespace
    :param cache: shared :class:`AclCache` carrying loaders + layers
    :ptype cache: AclCache
    :return: depth-1 group ids, first-seen order
    :rtype: tuple[UUID, ...]
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
        fence = cache.read_fence()
        if member_type == MemberType.USER:
            memberships = await cache.membership_loader.load_for_user(actor_id)
        else:
            memberships = await cache.membership_loader.load_for_agent(actor_id)
        cache.put_membership(membership_key, memberships, fence=fence)

    eligible = _filter_memberships(
        memberships=memberships,
        actor_id=actor_id,
        member_type=member_type,
        namespace=namespace,
    )
    return tuple(_unique_ordered(m.group_id for m in eligible))


def _accumulate_groups(
    group_ids: tuple[UUID, ...],
    namespace: Namespace,
    contributions: dict[GroupNamespaceKey, tuple[frozenset[str], tuple[Trail, ...]]],
) -> tuple[frozenset[str], tuple[Trail, ...]]:
    """union what each reachable group contributes on ``namespace``.

    the one place a side's reachable groups become an action set, shared by every kind of
    side so their answers cannot drift.

    :param group_ids: every group the side reaches, depth-expanded
    :ptype group_ids: tuple[UUID, ...]
    :param namespace: namespace under evaluation
    :ptype namespace: Namespace
    :param contributions: each (group, namespace) pair's contribution
    :ptype contributions: dict[GroupNamespaceKey, tuple[frozenset[str], tuple[Trail, ...]]]
    :return: ``(action_set, trails)`` pair for the side
    :rtype: tuple[frozenset[str], tuple[Trail, ...]]
    """
    actions_acc: set[str] = set()
    trails_acc: list[Trail] = []
    for group_id in group_ids:
        actions, trails = contributions[GroupNamespaceKey(group_id=group_id, namespace_id=namespace.id)]
        actions_acc.update(actions)
        trails_acc.extend(trails)

    # deterministic trail ordering across groups so two runs with the
    # same dataset produce the same trail order.
    trails_acc.sort(
        key=lambda t: (
            str(t.group.id),
            str(t.assignment.id),
        ),
    )
    return frozenset(actions_acc), tuple(trails_acc)


async def _group_namespace_contributions(
    wanted: dict[GroupNamespaceKey, Namespace],
    cache: AclCache,
) -> dict[GroupNamespaceKey, tuple[frozenset[str], tuple[Trail, ...]]]:
    """what each (group, namespace) pair contributes, from the cache or read in bulk.

    a pair the per-namespace layer holds is served from it. the rest are computed together:
    the groups' assignments in one read when the grant loader is a
    :class:`HeldAssignmentsLoader` (else one read per namespace, for every missing group at
    once), then their roles and the groups in one read each, and each pair walked by
    :func:`_walk_assignments` exactly as a single evaluation walks it. every computed pair is
    stored under the fence taken before the reads, so an eviction that lands meanwhile keeps
    a stale answer out of the cache.

    :param wanted: each pair the evaluation needs, with its namespace
    :ptype wanted: dict[GroupNamespaceKey, Namespace]
    :param cache: shared :class:`AclCache` carrying loaders + layers
    :ptype cache: AclCache
    :return: ``(actions, trails)`` for every wanted pair
    :rtype: dict[GroupNamespaceKey, tuple[frozenset[str], tuple[Trail, ...]]]
    """
    result: dict[GroupNamespaceKey, tuple[frozenset[str], tuple[Trail, ...]]] = {}
    missing: dict[GroupNamespaceKey, Namespace] = {}
    for key, namespace in wanted.items():
        entry = cache.get_group_namespace(key)
        if entry is not None:
            result[key] = (entry.actions, entry.trails)
        else:
            missing[key] = namespace
    if not missing:
        return result

    fence = cache.read_fence()
    loader = cache.grant_loader
    missing_groups = tuple(_unique_ordered(key.group_id for key in missing))
    held: dict[tuple[UUID, UUID], tuple[RoleAssignment, ...]] = {}
    if isinstance(loader, HeldAssignmentsLoader):
        by_group: dict[UUID, list[RoleAssignment]] = {}
        for assignment in await loader.load_assignments_held_by_groups(missing_groups):
            by_group.setdefault(assignment.group_id, []).append(assignment)
        for key, namespace in missing.items():
            held[(key.group_id, key.namespace_id)] = tuple(
                a for a in by_group.get(key.group_id, ()) if a.covers(namespace)
            )
    else:
        by_namespace: dict[UUID, tuple[Namespace, list[UUID]]] = {}
        for key, namespace in missing.items():
            by_namespace.setdefault(key.namespace_id, (namespace, []))[1].append(key.group_id)
        for namespace_id, (namespace, group_ids) in by_namespace.items():
            loaded = await loader.load_assignments_for_groups(group_ids=tuple(group_ids), namespace=namespace)
            for group_id in group_ids:
                held[(group_id, namespace_id)] = tuple(a for a in loaded if a.group_id == group_id)
    role_ids = tuple(_unique_ordered(a.role_id for assignments in held.values() for a in assignments))
    roles = await loader.load_roles(role_ids) if role_ids else {}
    groups = _coerce_groups(await loader.load_groups(missing_groups))
    for key, namespace in missing.items():
        assignments = held[(key.group_id, key.namespace_id)]
        actions, trails = _walk_assignments(
            assignments=assignments,
            roles=roles,
            groups=groups,
            eligible_group_ids=frozenset((key.group_id,)),
            namespace=namespace,
        )
        cache.put_group_namespace(key, actions, trails, role_ids=frozenset(a.role_id for a in assignments), fence=fence)
        result[key] = (actions, trails)
    return result


async def _expand_group_parents(
    *,
    direct_group_ids: tuple[UUID, ...],
    namespace: Namespace,
    cache: AclCache,
) -> tuple[UUID, ...]:
    """widen a side's group set through group-in-group edges, at read.

    an actor's DIRECT groups are depth 1; each further round adds the
    PARENT groups those groups are members of, up to
    :data:`MAX_GROUP_MEMBERSHIP_DEPTH` levels total. the walk reads
    through the membership cache under each child group's OWN key
    (``("group", group_id)``), never expanding into any actor's entry --
    changing one group's parents invalidates exactly one key and every
    actor entry beneath it stays valid, which is the invalidation
    property the read-time walk exists to keep.

    every walked edge passes the same eligibility filter a direct
    membership does (the row must name the child as a GROUP member and
    must clear the cross-customer wall), so nesting cannot widen what a
    direct membership could not.

    :param direct_group_ids: the side's depth-1 group ids, order
        preserved
    :ptype direct_group_ids: tuple[UUID, ...]
    :param namespace: namespace under evaluation, for the customer wall
    :ptype namespace: Namespace
    :param cache: shared :class:`AclCache` carrying loaders + layers
    :ptype cache: AclCache
    :return: direct ids plus reachable parent ids, deterministic order
    :rtype: tuple[UUID, ...]
    """
    seen: dict[UUID, None] = dict.fromkeys(direct_group_ids)
    frontier: tuple[UUID, ...] = direct_group_ids
    for _ in range(MAX_GROUP_MEMBERSHIP_DEPTH - 1):
        if not frontier:
            break
        next_frontier: list[UUID] = []
        for group_id in frontier:
            membership_key = ActorMembershipKey(actor_kind="group", actor_id=group_id)
            entry = cache.get_membership(membership_key)
            if entry is not None:
                parent_rows = entry.memberships
            else:
                fence = cache.read_fence()
                parent_rows = await cache.membership_loader.load_for_group(group_id)
                cache.put_membership(membership_key, parent_rows, fence=fence)
            eligible_parents = _filter_memberships(
                memberships=tuple(parent_rows),
                actor_id=group_id,
                member_type=MemberType.GROUP,
                namespace=namespace,
            )
            for row in eligible_parents:
                if row.group_id not in seen:
                    seen[row.group_id] = None
                    next_frontier.append(row.group_id)
        frontier = tuple(next_frontier)
    return tuple(seen)


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
        if role is not None and role.customer_id is not None and role.customer_id != namespace.customer_id:
            # role is customer-authored and owned by a different
            # customer. this is a SEPARATE wall from the group check
            # above: a platform-scoped group (customer_id IS NULL)
            # passes that check universally, so without this line an
            # assignment binding a platform group to a customer-authored
            # role would carry that role's actions into every customer.
            # ownership travels with the role, not with the grant.
            log.debug(
                "skipping assignment whose role belongs to another customer",
                extra={
                    "extra_data": {
                        "assignment_id": str(assignment.id),
                        "role_id": str(assignment.role_id),
                        "role_customer_id": str(role.customer_id),
                    }
                },
            )
            continue
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
    # a group-member evaluation runs alone on the member side, so it is
    # assembled exactly as a user-only one.
    has_user = ctx.user_id is not None or ctx.group_id is not None
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
