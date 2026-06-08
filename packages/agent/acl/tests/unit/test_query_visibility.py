"""unit tests for the caller-visibility SQL clause generator.

:func:`caller_visible_customer_clause` is the SINGLE shared copy of the
rbac caller-visibility ``EXISTS`` fragment — the hub broker and every
agent pod import it from here so the security SQL cannot drift across
the repo boundary. these tests pin the exact fragment shape, the
single-bind contract, and the ``param_offset`` composition knob so an
accidental edit to the SQL (a forgotten JOIN, a flipped scope branch,
a moved placeholder) fails loud rather than silently widening or
narrowing what rows a caller can see.
"""

from __future__ import annotations

from uuid import uuid4

from threetears.agent.acl import caller_visible_customer_clause


def test_returns_fragment_and_single_user_id_bind() -> None:
    """clause returns the EXISTS fragment plus exactly one bind (user_id).

    the contract callers rely on: the fragment allocates exactly one
    ``$N`` placeholder and the returned param list carries the caller
    user_id as its sole element, so the caller can splice the bind into
    its own param sequence at the named offset.
    """
    user_id = uuid4()
    fragment, params = caller_visible_customer_clause(
        user_id=user_id,
        customer_id_column="g.customer_id",
    )
    assert params == [user_id]
    assert fragment.strip().startswith("EXISTS (")
    assert "FROM role_assignments ra" in fragment
    assert "JOIN group_members gm ON gm.group_id = ra.group_id" in fragment
    assert (
        "LEFT JOIN namespaces ns ON ns.namespace_id = ra.scope_namespace_id"
        in fragment
    )


def test_default_param_offset_is_one() -> None:
    """the caller user_id bind defaults to ``$1``.

    callers building a single-clause WHERE start at offset 1; the
    fragment must reference ``$1`` for the membership match so the lone
    bind lines up with the first param.
    """
    fragment, _ = caller_visible_customer_clause(
        user_id=uuid4(),
        customer_id_column="g.customer_id",
    )
    assert "gm.member_id = $1" in fragment


def test_param_offset_shifts_placeholder() -> None:
    """``param_offset`` names the FIRST ``$N`` the fragment allocates.

    a caller that has already allocated three binds passes
    ``param_offset=4`` so the membership match references ``$4`` and
    composes additively with the caller's existing predicates.
    """
    fragment, params = caller_visible_customer_clause(
        user_id=uuid4(),
        customer_id_column="pe.customer_id",
        param_offset=4,
    )
    assert "gm.member_id = $4" in fragment
    assert len(params) == 1


def test_customer_id_column_embedded_verbatim() -> None:
    """the trusted ``customer_id_column`` identifier is embedded as given.

    the column reference is interpolated verbatim into both scope
    branches (``type_customer`` and ``namespace``); the caller passes a
    trusted identifier (never user input).
    """
    fragment, _ = caller_visible_customer_clause(
        user_id=uuid4(),
        customer_id_column="api_keys.customer_id",
    )
    assert "ra.scope_customer_id = api_keys.customer_id" in fragment
    assert "ns.customer_id = api_keys.customer_id" in fragment


def test_three_scope_branches_present() -> None:
    """all three admit branches (all / type_customer / namespace) are emitted.

    dropping any branch silently changes the visibility decision: losing
    ``scope_type='all'`` blinds platform admins; losing either scoped
    branch blinds customer admins. the test pins all three.
    """
    fragment, _ = caller_visible_customer_clause(
        user_id=uuid4(),
        customer_id_column="g.customer_id",
    )
    assert "ra.scope_type = 'all'" in fragment
    assert "ra.scope_type = 'type_customer'" in fragment
    assert "ra.scope_type = 'namespace'" in fragment
    assert "gm.member_type = 'user'" in fragment
