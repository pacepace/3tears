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

from uuid import uuid4

from threetears.agent.acl import (
    caller_visible_customer_clause,
    caller_visible_customers_query,
    customer_scope_visibility_clause,
    three_scope_visibility_clause,
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


# ---------------------------------------------------------------------------
# caller_visible_customers_query: the concrete visible-customer set for the RLS GUC
# ---------------------------------------------------------------------------


def test_visible_customers_query_returns_sql_and_single_user_id_bind() -> None:
    user_id = uuid4()
    sql, params = caller_visible_customers_query(user_id=user_id)
    assert params == [user_id]
    assert "$1" in sql


def test_visible_customers_query_mirrors_the_three_scope_arms() -> None:
    # the SET resolver must read the SAME three scope arms as the per-row EXISTS clause, or the
    # RLS GUC would admit a different set of rows than the app-layer visibility rule -- a leak or
    # an over-restriction. pins each arm so an edit to one generator without the other fails loud.
    sql, _ = caller_visible_customers_query(user_id=uuid4())
    assert "FROM role_assignments ra" in sql
    assert "JOIN group_members gm ON gm.group_id = ra.group_id" in sql
    assert "LEFT JOIN namespaces ns ON ns.namespace_id = ra.scope_namespace_id" in sql
    assert "gm.member_type = 'user'" in sql
    # scope=all -> the is_all flag (becomes the GUC '*' wildcard)
    assert "bool_or(ra.scope_type = 'all')" in sql
    assert "AS is_all" in sql
    # type_customer + namespace arms feed the concrete customer_ids array
    assert "ra.scope_type = 'type_customer' THEN ra.scope_customer_id" in sql
    assert "ra.scope_type = 'namespace' THEN ns.customer_id" in sql
    assert "AS customer_ids" in sql


def test_visible_customers_query_is_fail_closed_on_no_grants() -> None:
    # a caller with no grants must yield a non-NULL (empty) array, not NULL -- so the GUC admits
    # only platform (customer_id IS NULL) rows, never silently widening.
    sql, _ = caller_visible_customers_query(user_id=uuid4())
    assert "COALESCE(bool_or(ra.scope_type = 'all'), false)" in sql
    assert "ARRAY[]::uuid[]" in sql
    assert "array_remove(" in sql


def test_visible_customers_query_respects_param_offset() -> None:
    sql, params = caller_visible_customers_query(user_id=uuid4(), param_offset=3)
    assert "gm.member_id = $3" in sql
    assert len(params) == 1
