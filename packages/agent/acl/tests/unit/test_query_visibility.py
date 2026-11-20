"""unit tests for the caller-visibility SQL clause generator.

:func:`caller_visible_customer_clause` is the SINGLE shared copy of the
rbac caller-visibility ``EXISTS`` fragment — the hub broker and every
agent pod import it from here so the security SQL cannot drift across
the repo boundary. these tests pin the exact fragment shape, the
two-bind contract (caller user_id + the listed entity's namespace type),
and the ``param_offset`` composition knob so an accidental edit to the
SQL (a forgotten JOIN, a flipped scope branch, a moved placeholder, a
dropped namespace-type predicate) fails loud rather than silently
widening or narrowing what rows a caller can see.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Final
from uuid import UUID, uuid4

import pytest

from threetears.agent.acl import (
    MAX_GROUP_MEMBERSHIP_DEPTH,
    caller_visible_customer_clause,
    VisibleCustomers,
    customer_scope_visibility_clause,
    three_scope_visibility_clause,
    visible_customers,
)
from threetears.agent.acl.types import (
    MAX_CONFIGURABLE_GROUP_NESTING_LEVELS,
    MAX_GROUP_NESTING_LEVELS_ENV,
    MIN_CONFIGURABLE_GROUP_NESTING_LEVELS,
)


def test_returns_fragment_and_two_binds() -> None:
    """clause returns the EXISTS fragment plus two binds (user_id, namespace type).

    the contract callers rely on: the fragment allocates two ``$N``
    placeholders and the returned param list carries the caller user_id
    followed by the listed entity's namespace type, so the caller can
    splice both binds into its own param sequence at the named offset.
    """
    user_id = uuid4()
    fragment, params = caller_visible_customer_clause(
        user_id=user_id,
        customer_id_column="g.customer_id",
        scope_namespace_type="api_key",
    )
    assert params == [user_id, "api_key"]
    assert fragment.strip().startswith("EXISTS (")
    assert "FROM role_assignments ra" in fragment
    assert "JOIN group_members gm ON gm.group_id = ra.group_id" in fragment
    assert "LEFT JOIN namespaces ns ON ns.namespace_id = ra.scope_namespace_id" in fragment


def test_default_param_offset_is_one() -> None:
    """the caller user_id bind defaults to ``$1``, namespace type to ``$2``.

    callers building a single-clause WHERE start at offset 1; the
    fragment must reference ``$1`` for the membership match and ``$2``
    for the namespace-type predicate so the two binds line up with the
    first two params.
    """
    fragment, _ = caller_visible_customer_clause(
        user_id=uuid4(),
        customer_id_column="g.customer_id",
        scope_namespace_type="api_key",
    )
    assert "gm.member_id = $1" in fragment
    assert "ra.scope_namespace_type = $2" in fragment


def test_param_offset_shifts_placeholder() -> None:
    """``param_offset`` names the FIRST ``$N`` the fragment allocates.

    a caller that has already allocated three binds passes
    ``param_offset=4`` so the membership match references ``$4``, the
    namespace-type predicate ``$5``, and the fragment composes additively
    with the caller's existing predicates.
    """
    fragment, params = caller_visible_customer_clause(
        user_id=uuid4(),
        customer_id_column="pe.customer_id",
        scope_namespace_type="knowledge",
        param_offset=4,
    )
    assert "gm.member_id = $4" in fragment
    assert "ra.scope_namespace_type = $5" in fragment
    assert len(params) == 2


def test_customer_id_column_embedded_verbatim() -> None:
    """the trusted ``customer_id_column`` identifier is embedded as given.

    the column reference is interpolated verbatim into both scope
    branches (``type_customer`` and ``namespace``); the caller passes a
    trusted identifier (never user input).
    """
    fragment, _ = caller_visible_customer_clause(
        user_id=uuid4(),
        customer_id_column="api_keys.customer_id",
        scope_namespace_type="api_key",
    )
    assert "ra.scope_customer_id = api_keys.customer_id" in fragment
    assert "ns.customer_id = api_keys.customer_id" in fragment


def test_type_customer_arm_constrains_on_namespace_type() -> None:
    """the ``type_customer`` arm ANDs ``scope_namespace_type`` onto the customer match.

    a ``type_customer`` grant is scoped to ONE namespace type within a
    customer (:meth:`RoleAssignment.covers` requires the type to match).
    without the type predicate a grant on one type (e.g. ``datasource``)
    would admit every OTHER type's rows in the same customer — a
    cross-type data leak. the type predicate must sit inside the
    ``type_customer`` arm, bound (not embedded), alongside the customer
    match.
    """
    fragment, params = caller_visible_customer_clause(
        user_id=uuid4(),
        customer_id_column="api_keys.customer_id",
        scope_namespace_type="api_key",
    )
    # the namespace-type predicate must sit INSIDE the type_customer arm:
    # after the type_customer marker + its customer match, and before the
    # separate namespace arm. index ordering pins the placement without
    # coupling the test to exact whitespace.
    type_customer_at = fragment.index("ra.scope_type = 'type_customer'")
    customer_match_at = fragment.index("ra.scope_customer_id = api_keys.customer_id")
    type_pred_at = fragment.index("ra.scope_namespace_type = $2")
    namespace_arm_at = fragment.index("ra.scope_type = 'namespace'")
    assert type_customer_at < customer_match_at < type_pred_at < namespace_arm_at
    # the namespace-type value is bound, never embedded verbatim.
    assert "= 'api_key'" not in fragment
    assert params[1] == "api_key"


def test_three_scope_branches_present() -> None:
    """all three admit branches (all / type_customer / namespace) are emitted.

    dropping any branch silently changes the visibility decision: losing
    ``scope_type='all'`` blinds platform admins; losing either scoped
    branch blinds customer admins. the test pins all three.
    """
    fragment, _ = caller_visible_customer_clause(
        user_id=uuid4(),
        customer_id_column="g.customer_id",
        scope_namespace_type="customer",
    )
    assert "ra.scope_type = 'all'" in fragment
    assert "ra.scope_type = 'type_customer'" in fragment
    assert "ra.scope_type = 'namespace'" in fragment
    assert "gm.member_type = 'user'" in fragment


# ---------------------------------------------------------------------------
# customer_scope_visibility_clause: the platform-OR-customer read wrap (D10)
# ---------------------------------------------------------------------------


def test_customer_scope_wraps_platform_or_caller_visible() -> None:
    """clause admits platform rows (customer_id NULL) OR caller-visible ones.

    the customer-scope read rule: a row is visible iff it is platform-scope
    (``customer_id IS NULL``) OR its customer is RBAC-admitted by the
    delegated EXISTS subquery. the wrap composes the existing clause; it
    does NOT re-implement the EXISTS.
    """
    user_id = uuid4()
    fragment, params = customer_scope_visibility_clause(
        user_id=user_id,
        customer_id_column="pe.customer_id",
        scope_namespace_type="knowledge",
    )
    assert params == [user_id, "knowledge"]
    assert fragment.startswith("(pe.customer_id IS NULL OR (")
    assert "EXISTS (" in fragment
    assert "gm.member_id = $1" in fragment
    assert "ra.scope_namespace_type = $2" in fragment


def test_customer_scope_param_offset_shifts() -> None:
    """``param_offset`` flows through to the delegated EXISTS membership match.

    a caller that has already allocated two binds passes ``param_offset=3``;
    the user_id bind references ``$3`` and the namespace-type bind ``$4``.
    """
    fragment, params = customer_scope_visibility_clause(
        user_id=uuid4(),
        customer_id_column="co.customer_id",
        scope_namespace_type="knowledge",
        param_offset=3,
    )
    assert "gm.member_id = $3" in fragment
    assert "ra.scope_namespace_type = $4" in fragment
    assert len(params) == 2


# ---------------------------------------------------------------------------
# three_scope_visibility_clause: the full entity-list read rule (D10)
# ---------------------------------------------------------------------------


def test_three_scope_adds_user_privacy_predicate() -> None:
    """clause ANDs the user-privacy predicate onto the customer-scope wrap.

    the full three-scope read rule: customer-scope visibility AND the row
    is platform/customer-scope (``user_id IS NULL``) OR the caller's own
    (``user_id = caller``). a user-scope row is admitted only to its owner.
    """
    user_id = uuid4()
    fragment, params = three_scope_visibility_clause(
        user_id=user_id,
        customer_id_column="pe.customer_id",
        user_id_column="pe.user_id",
        scope_namespace_type="knowledge",
    )
    assert params == [user_id, "knowledge"]
    assert "(pe.customer_id IS NULL OR (" in fragment
    assert "(pe.user_id IS NULL OR pe.user_id = $1)" in fragment
    assert " AND " in fragment


def test_three_scope_reuses_user_bind_and_adds_namespace_type() -> None:
    """the user predicate REUSES the caller-user bind; the type is a SECOND bind.

    a user-scope row's owner IS the caller the customer EXISTS subquery
    already binds at ``$param_offset``, so both halves reference the SAME
    placeholder for the user_id. the namespace type is the only additional
    bind, so the clause returns exactly two binds (user_id, namespace type).
    """
    user_id = uuid4()
    _, params = three_scope_visibility_clause(
        user_id=user_id,
        customer_id_column="pe.customer_id",
        user_id_column="pe.user_id",
        scope_namespace_type="knowledge",
    )
    assert params == [user_id, "knowledge"]
    assert len(params) == 2


def test_three_scope_user_predicate_honors_param_offset() -> None:
    """the user predicate references ``$param_offset``, never a hard-coded ``$1``.

    this is the regression guard against the hand-rolled sites that assumed
    ``param_offset=1`` and hard-coded ``$1``: at offset 5 BOTH the EXISTS
    membership match and the user predicate must reference ``$5``, and the
    namespace-type predicate references ``$6``.
    """
    fragment, params = three_scope_visibility_clause(
        user_id=uuid4(),
        customer_id_column="co.customer_id",
        user_id_column="co.user_id",
        scope_namespace_type="knowledge",
        param_offset=5,
    )
    assert "gm.member_id = $5" in fragment
    assert "(co.user_id IS NULL OR co.user_id = $5)" in fragment
    assert "ra.scope_namespace_type = $6" in fragment
    assert len(params) == 2


def test_three_scope_columns_embedded_verbatim() -> None:
    """both trusted column identifiers are embedded as given.

    callers pass trusted identifiers (never user input); the user column
    appears verbatim in the privacy predicate and the customer column in
    the delegated EXISTS branches.
    """
    fragment, _ = three_scope_visibility_clause(
        user_id=uuid4(),
        customer_id_column="ec.customer_id",
        user_id_column="ec.user_id",
        scope_namespace_type="knowledge",
    )
    assert "ra.scope_customer_id = ec.customer_id" in fragment
    assert "(ec.user_id IS NULL OR ec.user_id = $1)" in fragment


def test_the_clause_admits_only_a_role_that_grants_where_its_scope_needs() -> None:
    """the role predicates ride the clause: the listed type's, and the bound namespace's own.

    a fleet-wide or per-type grant admits through a role granting on the listed type; a namespace
    grant through a role granting on the namespace it is bound to. an assignment failing its arm's
    predicate admits no row: the semantics are proved against a real database in
    ``integration/test_query_visibility_roles.py`` (beside :func:`visible_customers`, which reads
    the same roles); this pins that the generator does not drop the predicate.
    """
    fragment, _ = caller_visible_customer_clause(
        user_id=uuid4(), customer_id_column="t.customer_id", scope_namespace_type="thing", param_offset=4
    )
    assert "JOIN roles vis_role ON vis_role.role_id = ra.role_id" in fragment
    assert "vis_bucket.key IN ($5, '*')" in fragment
    assert "vis_bucket.key IN (ns.namespace_type, '*')" in fragment
    # which actions each arm counts (only an action of the type's own) is pinned by behaviour on a real database, where
    # the clause and visible_customers admit the same customers for every scope and role
    # (integration/test_query_visibility_roles.py::test_the_customer_set_a_listing_pages_within_is_the_set_the_clause_admits).


def test_caller_clause_reaches_through_one_nesting_level() -> None:
    """the fragment carries the group-hop arm, and it binds the same caller.

    group-in-group membership resolves at authorization time, so the
    visibility fragment must name the nested caller too. pinned here so an
    edit that drops the hop fails loud; the semantics are proved against a
    real database in ``integration/test_query_visibility_nesting.py``.
    """
    fragment, _ = caller_visible_customer_clause(
        user_id=uuid4(),
        customer_id_column="t.customer_id",
        scope_namespace_type="thing",
    )
    assert "gm.member_type = 'group'" in fragment
    assert "gm2.group_id = gm.member_id" in fragment
    assert "gm2.member_type = 'user'" in fragment
    assert fragment.count("$1") >= 2


def test_the_clause_nests_one_hop_per_edge_up_to_the_cap() -> None:
    """``MAX_GROUP_MEMBERSHIP_DEPTH - 1`` nested hops, each keyed on the next alias, and none past it."""
    fragment, _ = caller_visible_customer_clause(
        user_id=uuid4(),
        customer_id_column="t.customer_id",
        scope_namespace_type="thing",
    )
    for level in range(2, MAX_GROUP_MEMBERSHIP_DEPTH + 1):
        assert f"gm{level}.group_id = gm{level - 1 if level > 2 else ''}.member_id" in fragment
    assert f"gm{MAX_GROUP_MEMBERSHIP_DEPTH + 1}" not in fragment


# ---------------------------------------------------------------------------
# visible_customers: the concrete visible-customer set, read by key
# ---------------------------------------------------------------------------


class _Tables:
    """a connection double holding group_members, role_assignments and namespaces rows, answering the keyed reads
    :func:`visible_customers` sends (``read_by``: pins as ``= $n`` or ``= ANY($n)``, in the order the statement names
    them)."""

    def __init__(
        self,
        members: list[tuple[str, UUID, UUID]],
        grants: list[dict[str, Any]],
        namespaces: dict[UUID, UUID | None] | None = None,
    ) -> None:
        self.members = members  # (member_type, member_id, group_id)
        self.grants = grants
        self.namespaces = namespaces or {}
        self.reads: list[str] = []

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """answer the one capped walk :func:`visible_customers` sends: every membership edge within the depth (each a
        platform row: no customer), each beside its group's grants and a namespace grant's namespace's customer (an
        edge whose group holds none kept, as LEFT joins keep it); every group and role a platform one."""
        self.reads.append(sql)
        assert sql.startswith("WITH RECURSIVE reached") and '"group_members"' in sql, sql
        # the discriminators are the code's own constants, written as literals (Const) so the generic plan caches
        assert "t.\"member_type\" = 'user'" in sql and "t.\"member_type\" = 'group'" in sql, sql
        start_kind, link_kind = "user", "group"
        user, depth, *_scopes = params
        level = {(m, g) for t, m, g in self.members if t == start_kind and m == user}
        edges = set(level)
        for _ in range(depth - 1):
            level = {(m, g) for t, m, g in self.members if t == link_kind and m in {above for _, above in level}}
            edges |= level
        rows: list[dict[str, Any]] = []
        for member, group in edges:
            grants = [grant for grant in self.grants if grant["group_id"] == group] or [{}]
            for grant in grants:
                namespace = grant.get("scope_namespace_id")
                rows.append(
                    {
                        "member_id": member,
                        "customer_id": None,
                        "group_customer_id": None,
                        "role_customer_id": None,
                        "group_id": group,
                        **{
                            key: grant.get(key)
                            for key in ("scope_type", "scope_customer_id", "scope_namespace_type", "scope_namespace_id")
                        },
                        "namespace_customer_id": self.namespaces.get(namespace) if namespace is not None else None,
                        # the bound namespace's own type, and the role each grant binds (a reader on every type unless
                        # the grant names its own): the set counts a grant only through a role acting where it needs
                        "namespace_type": "customer" if namespace in self.namespaces else None,
                        "role_permissions": grant.get("role_permissions", _READER) if grant else None,
                    }
                )
        return rows


#: the role a double's grant binds unless it names its own: a reader on every type
_READER: Final = {"*": ["read"]}


def _grant(group: UUID, scope_type: str, **values: Any) -> dict[str, Any]:
    return {"group_id": group, "scope_type": scope_type, **values}


async def test_visible_customers_reads_the_three_scope_arms() -> None:
    """Flips if the set stops reading any of the three scope arms the per-row clause admits by (the RLS GUC and the
    clause must agree): ``all`` sets is_all, ``type_customer`` adds its customer, ``namespace`` its namespace's
    customer; a ``subtree`` grant adds none."""
    user, group, namespace = uuid4(), uuid4(), uuid4()
    first, second = UUID(int=1), UUID(int=2)
    tables = _Tables(
        [("user", user, group)],
        [
            _grant(group, "type_customer", scope_customer_id=first, scope_namespace_type="tool"),
            _grant(group, "namespace", scope_namespace_id=namespace, scope_namespace_type="tool"),
            _grant(group, "subtree", scope_namespace_type="tool"),
        ],
        {namespace: second},
    )
    assert await visible_customers(tables, user_id=user, namespace_type=None) == VisibleCustomers(
        is_all=False, customer_ids=(first, second)
    )
    tables.grants.append(_grant(group, "all"))
    assert (await visible_customers(tables, user_id=user, namespace_type=None)).is_all


async def test_visible_customers_is_fail_closed_on_no_grants() -> None:
    """Flips if a caller with no group, or groups with no grant, gets anything but no customer and not every one:
    the set admits only platform rows then, never silently widening."""
    user, group = uuid4(), uuid4()
    nobody = VisibleCustomers(is_all=False, customer_ids=())
    assert await visible_customers(_Tables([], []), user_id=user, namespace_type="tool") == nobody
    assert await visible_customers(_Tables([("user", user, group)], []), user_id=user, namespace_type="tool") == nobody


async def test_the_typed_set_differs_only_in_its_type_customer_arm() -> None:
    """Flips if naming a namespace type changes anything but which ``type_customer`` grants add their customer: a
    grant on another type adds none, a namespace grant and ``all`` are untouched."""
    user, group, namespace = uuid4(), uuid4(), uuid4()
    on_type, on_other, by_namespace = UUID(int=1), UUID(int=2), UUID(int=3)
    tables = _Tables(
        [("user", user, group)],
        [
            _grant(group, "type_customer", scope_customer_id=on_type, scope_namespace_type="tool"),
            _grant(group, "type_customer", scope_customer_id=on_other, scope_namespace_type="knowledge"),
            _grant(group, "namespace", scope_namespace_id=namespace, scope_namespace_type="knowledge"),
        ],
        {namespace: by_namespace},
    )
    untyped = await visible_customers(tables, user_id=user, namespace_type=None)
    typed = await visible_customers(tables, user_id=user, namespace_type="tool")
    assert untyped.customer_ids == (on_type, on_other, by_namespace)
    assert typed.customer_ids == (on_type, by_namespace)


async def test_visible_customers_reaches_through_nesting_to_the_configured_depth_and_no_further() -> None:
    """Flips if the set misses a grant a chain of up to MAX_GROUP_MEMBERSHIP_DEPTH groups reaches (the caller's
    group nested group in group), or takes one a chain longer than that reaches; a cycle ends the walk."""
    user = uuid4()
    chain = [uuid4() for _ in range(MAX_GROUP_MEMBERSHIP_DEPTH + 1)]
    members = [("user", user, chain[0])]
    members += [("group", below, above) for below, above in zip(chain, chain[1:])]
    members.append(("group", chain[1], chain[0]))  # a cycle back down the chain
    grants = [
        _grant(group, "type_customer", scope_customer_id=UUID(int=level + 1), scope_namespace_type="tool")
        for level, group in enumerate(chain)
    ]
    visible = await visible_customers(_Tables(members, grants), user_id=user, namespace_type="tool")
    assert visible.customer_ids == tuple(UUID(int=level + 1) for level in range(MAX_GROUP_MEMBERSHIP_DEPTH))


async def test_visible_customers_is_one_statement() -> None:
    """Flips if the set costs more than one round trip, for a caller in groups or in none: the capped walk and its
    lookups are one statement."""
    user, group = uuid4(), uuid4()
    tables = _Tables([("user", user, group)], [_grant(group, "all")])
    await visible_customers(tables, user_id=user, namespace_type="tool")
    alone = _Tables([], [])
    await visible_customers(alone, user_id=user, namespace_type="tool")
    assert (len(tables.reads), len(alone.reads)) == (1, 1)


_DEPTH_AT_A_NESTING = """
from threetears.agent.acl import visible_customers
from threetears.agent.acl.types import MAX_GROUP_MEMBERSHIP_DEPTH
print(MAX_GROUP_MEMBERSHIP_DEPTH)
"""


def _queries_configured_for(levels: int) -> subprocess.CompletedProcess[str]:
    """a process whose deployment configures ``levels`` nesting levels (its front door: the environment, read once
    at import), importing the visible-customer set."""
    environ = {**os.environ, MAX_GROUP_NESTING_LEVELS_ENV: str(levels)}
    return subprocess.run(
        [sys.executable, "-c", _DEPTH_AT_A_NESTING], env=environ, capture_output=True, text=True, check=False
    )


#: every chain a deployment may configure, in groups (levels less the person)
_CONFIGURABLE_DEPTHS = range(MIN_CONFIGURABLE_GROUP_NESTING_LEVELS - 1, MAX_CONFIGURABLE_GROUP_NESTING_LEVELS)


@pytest.mark.parametrize("depth", _CONFIGURABLE_DEPTHS)
def test_every_configurable_nesting_is_walked(depth: int) -> None:
    """Flips if a deployment configured for a nesting it may configure walks any other depth."""
    answered = _queries_configured_for(depth + 1)
    assert answered.returncode == 0, answered.stderr
    assert int(answered.stdout.strip()) == depth


@pytest.mark.parametrize(
    "levels", [MIN_CONFIGURABLE_GROUP_NESTING_LEVELS - 1, MAX_CONFIGURABLE_GROUP_NESTING_LEVELS + 1]
)
def test_a_nesting_past_the_written_statement_refuses(levels: int) -> None:
    """Flips if a deployment configured for a nesting the statement is not written out for is cut silently to
    another walk: the process refuses the setting before any statement is built."""
    answered = _queries_configured_for(levels)
    assert answered.returncode != 0
    assert MAX_GROUP_NESTING_LEVELS_ENV in answered.stderr and "outside" in answered.stderr, answered.stderr
