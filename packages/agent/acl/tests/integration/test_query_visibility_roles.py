"""a list shows a row only through a grant whose role acts where its scope needs it to, on a real database.

the visibility clause and the customer set a listing pages within (once the RLS customer-set query, now
:func:`visible_customers`) once ignored the role an assignment binds:
an assignment of ANY role widened a caller's lists. a narrow role bound at ``scope_type='all'``
(one action on one system namespace) showed its holder every customer's rows of every type, and
a role bound to a namespace where it grants nothing opened that namespace's customer. these
tests prove both now admit a fleet-wide or per-type grant only through a role acting
on the listed type, and a namespace grant only through a role acting on the namespace it is bound
to; and that the SQL reads a role exactly as the evaluator does: through
the production :class:`RoleCollection`, :meth:`Role.actions_for` decides the same shapes the SQL
decides, the wildcard bucket, an empty bucket, a bucket that is not a list and a double-encoded
``permissions`` value included.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.agent.acl import (
    RoleCollection,
    caller_visible_customer_clause,
    visible_customers,
)
from threetears.core.collections import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeNatsClient
from threetears.nats import set_default_namespace

from .access_tables_ddl import ACCESS_TABLE_NAMES, ACCESS_TABLES_DDL

pytestmark = pytest.mark.integration

#: the listed entity's namespace type.
_LISTED = "thing"

_LISTED_DDL = "CREATE TABLE listed_things (thing_id uuid PRIMARY KEY, customer_id uuid)"


@pytest.fixture
async def pg_pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    """per-test pool over the access tables and one listed table.

    :param db_container: the test database
    :ptype db_container: str
    :return: the pool
    :rtype: AsyncIterator[asyncpg.Pool]
    """
    pool: asyncpg.Pool = await asyncpg.create_pool(db_container, min_size=1, max_size=4, init=init_connection)
    try:
        async with pool.acquire() as conn:
            for table in ("listed_things", *ACCESS_TABLE_NAMES):
                await conn.execute(f"DROP TABLE IF EXISTS {table}")
            for ddl in ACCESS_TABLES_DDL:
                await conn.execute(ddl)
            await conn.execute(_LISTED_DDL)
        yield pool
    finally:
        await pool.close()


async def _role(conn: asyncpg.Connection, permissions: Any, *, double_encoded: bool = False) -> uuid.UUID:
    """seed one role carrying ``permissions``.

    :param conn: open connection
    :ptype conn: asyncpg.Connection
    :param permissions: the permissions value, stored as given
    :ptype permissions: Any
    :param double_encoded: store the value as a JSON string holding the JSON, as a double-encoded write does
    :ptype double_encoded: bool
    :return: the role id
    :rtype: uuid.UUID
    """
    role_id = uuid.uuid4()
    encoded = json.dumps(json.dumps(permissions) if double_encoded else permissions)
    await conn.execute(
        "INSERT INTO roles (role_id, name, permissions) VALUES ($1, $2, $3::text::jsonb)",
        role_id,
        f"role-{role_id.hex}",
        encoded,
    )
    return role_id


async def _holder(conn: asyncpg.Connection, role_id: uuid.UUID, **scope: Any) -> uuid.UUID:
    """seed a user in a group holding ``role_id`` on ``scope``.

    :param conn: open connection
    :ptype conn: asyncpg.Connection
    :param role_id: the role the group holds
    :ptype role_id: uuid.UUID
    :param scope: ``scope_type`` and the scope columns it needs
    :ptype scope: Any
    :return: the user
    :rtype: uuid.UUID
    """
    user_id, group_id = uuid.uuid4(), uuid.uuid4()
    await conn.execute(
        "INSERT INTO group_members (id, group_id, member_type, member_id) VALUES ($1, $2, 'user', $3)",
        uuid.uuid4(),
        group_id,
        user_id,
    )
    await conn.execute(
        "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type, scope_namespace_id,"
        " scope_namespace_type, scope_customer_id) VALUES ('platform', $1, $2, $3, $4, $5, $6, $7)",
        uuid.uuid4(),
        role_id,
        group_id,
        scope["scope_type"],
        scope.get("scope_namespace_id"),
        scope.get("scope_namespace_type"),
        scope.get("scope_customer_id"),
    )
    return user_id


async def _namespace(conn: asyncpg.Connection, customer_id: uuid.UUID, namespace_type: str) -> uuid.UUID:
    """seed one namespace of ``customer_id``.

    :param conn: open connection
    :ptype conn: asyncpg.Connection
    :param customer_id: the namespace's customer
    :ptype customer_id: uuid.UUID
    :param namespace_type: its type
    :ptype namespace_type: str
    :return: the namespace id
    :rtype: uuid.UUID
    """
    namespace_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, customer_id, date_created,"
        " date_updated) VALUES ('customer', $1, $2, $3, $4, now(), now())",
        namespace_id,
        f"ns-{namespace_id.hex}",
        namespace_type,
        customer_id,
    )
    return namespace_id


async def _things(conn: asyncpg.Connection, *customer_ids: uuid.UUID) -> None:
    """seed one listed row per customer.

    :param conn: open connection
    :ptype conn: asyncpg.Connection
    :param customer_ids: the customers
    :ptype customer_ids: uuid.UUID
    :return: nothing
    :rtype: None
    """
    for customer_id in customer_ids:
        await conn.execute(
            "INSERT INTO listed_things (thing_id, customer_id) VALUES ($1, $2)", uuid.uuid4(), customer_id
        )


async def _visible(conn: asyncpg.Connection, user_id: uuid.UUID) -> set[uuid.UUID]:
    """the customers whose listed rows the per-row clause shows ``user_id``.

    :param conn: open connection
    :ptype conn: asyncpg.Connection
    :param user_id: the caller
    :ptype user_id: uuid.UUID
    :return: the customers
    :rtype: set[uuid.UUID]
    """
    fragment, params = caller_visible_customer_clause(
        user_id=user_id, customer_id_column="listed_things.customer_id", scope_namespace_type=_LISTED
    )
    rows = await conn.fetch(f"SELECT customer_id FROM listed_things WHERE {fragment}", *params)
    return {row["customer_id"] for row in rows}


async def _customer_set(conn: asyncpg.Connection, user_id: uuid.UUID) -> tuple[bool, set[uuid.UUID]]:
    """the customers a listing of the listed type pages within for ``user_id`` (:func:`visible_customers`).

    :param conn: open connection
    :ptype conn: asyncpg.Connection
    :param user_id: the caller
    :ptype user_id: uuid.UUID
    :return: ``(is_all, customer_ids)``
    :rtype: tuple[bool, set[uuid.UUID]]
    """
    seen = await visible_customers(conn, user_id=user_id, namespace_type=_LISTED)
    return seen.is_all, set(seen.customer_ids)


async def test_a_narrow_role_bound_fleet_wide_shows_nothing_of_another_type(pg_pool: asyncpg.Pool) -> None:
    """the hole: a one-action system role at scope ``all`` once listed every customer's rows of every type."""
    customer_a, customer_b = uuid.uuid4(), uuid.uuid4()
    async with pg_pool.acquire() as conn:
        await _things(conn, customer_a, customer_b)
        narrow = await _role(conn, {"system": ["group.read"]})
        holder = await _holder(conn, narrow, scope_type="all")
        visible = await _visible(conn, holder)
        is_all, customers = await _customer_set(conn, holder)

    assert visible == set()
    assert (is_all, customers) == (False, set())


async def test_a_role_granting_the_type_fleet_wide_shows_every_customer(pg_pool: asyncpg.Pool) -> None:
    """positive control for the test above: the same scope, a role that grants on the listed type."""
    customer_a, customer_b = uuid.uuid4(), uuid.uuid4()
    async with pg_pool.acquire() as conn:
        await _things(conn, customer_a, customer_b)
        reader = await _role(conn, {_LISTED: ["read"]})
        holder = await _holder(conn, reader, scope_type="all")
        visible = await _visible(conn, holder)
        is_all, _ = await _customer_set(conn, holder)

    assert visible == {customer_a, customer_b}
    assert is_all is True


async def test_a_namespace_grant_opens_its_customer_only_through_a_role_acting_on_that_namespace(
    pg_pool: asyncpg.Pool,
) -> None:
    """a namespace grant is membership in its customer, when its role acts where it is bound.

    a customer admin's grant on the customer namespace opens every list of that customer -- the
    admin model the hub's routes gate on -- although the role carries only a ``customer`` bucket.
    a role bound where it grants nothing is no grant, and opens nothing.
    """
    customer_a = uuid.uuid4()
    async with pg_pool.acquire() as conn:
        await _things(conn, customer_a)
        customer_namespace = await _namespace(conn, customer_a, "customer")
        admin_role = await _role(conn, {"customer": ["customer.read", "customer.admin"]})
        inert_role = await _role(conn, {"system": ["group.read"]})
        admin = await _holder(conn, admin_role, scope_type="namespace", scope_namespace_id=customer_namespace)
        inert = await _holder(conn, inert_role, scope_type="namespace", scope_namespace_id=customer_namespace)
        admin_sees, admin_set = await _visible(conn, admin), await _customer_set(conn, admin)
        inert_sees, inert_set = await _visible(conn, inert), await _customer_set(conn, inert)

    assert (admin_sees, admin_set) == ({customer_a}, (False, {customer_a}))
    assert (inert_sees, inert_set) == (set(), (False, set()))


async def test_a_per_type_customer_grant_of_another_type_joins_no_customer_set(pg_pool: asyncpg.Pool) -> None:
    """the RLS set honours the ``type_customer`` arm's type, as the per-row clause always has."""
    customer_a = uuid.uuid4()
    async with pg_pool.acquire() as conn:
        await _things(conn, customer_a)
        broad = await _role(conn, {"*": ["read"]})
        other_type = await _holder(
            conn, broad, scope_type="type_customer", scope_customer_id=customer_a, scope_namespace_type="datasource"
        )
        same_type = await _holder(
            conn, broad, scope_type="type_customer", scope_customer_id=customer_a, scope_namespace_type=_LISTED
        )
        other_sees, other_set = await _visible(conn, other_type), await _customer_set(conn, other_type)
        same_sees, same_set = await _visible(conn, same_type), await _customer_set(conn, same_type)

    assert (other_sees, other_set) == (set(), (False, set()))
    assert (same_sees, same_set) == ({customer_a}, (False, {customer_a}))


#: every way a role's permissions can be stored: ``(value, stored double-encoded)``. whether each
#: grants anything on the listed type is the evaluator's verdict, read through the real collection
#: below, never asserted by hand.
_PERMISSION_SHAPES: dict[str, tuple[Any, bool]] = {
    "the listed type's bucket": ({_LISTED: ["read"]}, False),
    "the wildcard bucket": ({"*": ["read"]}, False),
    "another type's bucket only": ({"system": ["group.read"]}, False),
    "an empty bucket": ({_LISTED: []}, False),
    "a bucket that is not a list": ({_LISTED: "read"}, False),
    "no buckets": ({}, False),
    "the listed type's bucket, double-encoded": ({_LISTED: ["read"]}, True),
    "another type's bucket, double-encoded": ({"system": ["group.read"]}, True),
}


@pytest.mark.parametrize("shape", list(_PERMISSION_SHAPES))
async def test_the_sql_reads_a_role_as_the_evaluator_does(pg_pool: asyncpg.Pool, shape: str) -> None:
    """for every permissions shape, the clause shows rows exactly when :meth:`Role.actions_for` is non-empty.

    the evaluator's role comes through the production :class:`RoleCollection`, so its decoding of
    the stored value -- a double-encoded string, a bucket that is not a list -- is the one the SQL is
    held to.
    """
    permissions, double_encoded = _PERMISSION_SHAPES[shape]
    customer_a = uuid.uuid4()
    async with pg_pool.acquire() as conn:
        await _things(conn, customer_a)
        role_id = await _role(conn, permissions, double_encoded=double_encoded)
        holder = await _holder(conn, role_id, scope_type="all")
        visible = await _visible(conn, holder)
        is_all, _ = await _customer_set(conn, holder)

    set_default_namespace("visroles")
    bus = FakeNatsClient()
    registry = CollectionRegistry()
    registry.configure(l3_pool=pg_pool, l2_client=bus, kv_key_scope="visroles")
    roles = RoleCollection(registry, DefaultCoreConfig(), nats_client=bus)
    [role] = await roles.get_many((role_id,))
    granted = bool(role.actions_for(_LISTED))

    assert (visible == {customer_a}) is granted, (shape, visible, granted)
    assert is_all is granted, (shape, is_all, granted)


#: aliases callers give their listed table today (``dataset_runs r``, ``groups g``, ``playbook_entries pe``, ...).
_CALLER_ALIASES = ("r", "g", "t", "d", "pe", "co", "ec")


@pytest.mark.parametrize("alias", _CALLER_ALIASES)
async def test_a_caller_aliasing_its_listed_table_still_sees_its_customer(pg_pool: asyncpg.Pool, alias: str) -> None:
    """a caller's listed-table alias is not captured by the clause's own joins.

    callers alias their listed table and pass ``<alias>.customer_id``; when the clause joined
    ``roles r``, a caller's ``r.customer_id`` resolved to the role's NULL customer inside the
    subquery and the caller saw nothing (the hub's dataset-run list, caught by its live-database
    test). the clause's own aliases are named so no caller's short alias collides.
    """
    customer_a = uuid.uuid4()
    async with pg_pool.acquire() as conn:
        await _things(conn, customer_a)
        reader = await _role(conn, {_LISTED: ["read"]})
        holder = await _holder(
            conn, reader, scope_type="type_customer", scope_customer_id=customer_a, scope_namespace_type=_LISTED
        )
        fragment, params = caller_visible_customer_clause(
            user_id=holder, customer_id_column=f"{alias}.customer_id", scope_namespace_type=_LISTED
        )
        rows = await conn.fetch(f"SELECT {alias}.customer_id FROM listed_things {alias} WHERE {fragment}", *params)

    assert {row["customer_id"] for row in rows} == {customer_a}


async def test_a_grant_on_any_namespace_of_a_customer_is_membership_in_it(pg_pool: asyncpg.Pool) -> None:
    """a role acting on the namespace it is bound to opens that namespace's customer for every listed type.

    deliberate, and relied on: an agent granted a datasource reader on one datasource namespace is
    answered from that customer's knowledge. what the role check removes is only the grant that
    acts nowhere at its namespace; the per-customer reach of a real namespace grant is unchanged.
    """
    customer_a, customer_b = uuid.uuid4(), uuid.uuid4()
    async with pg_pool.acquire() as conn:
        await _things(conn, customer_a, customer_b)
        datasource_namespace = await _namespace(conn, customer_a, "datasource")
        datasource_reader = await _role(conn, {"datasource": ["datasource.read"]})
        holder = await _holder(conn, datasource_reader, scope_type="namespace", scope_namespace_id=datasource_namespace)
        sees, customer_set = await _visible(conn, holder), await _customer_set(conn, holder)

    assert sees == {customer_a}
    assert customer_set == (False, {customer_a})


#: every scope a grant takes here, as ``_holder``'s keywords given the customer and its namespaces: each is read the
#: same way by the per-row clause and by :func:`visible_customers`
_SCOPES: dict[str, Any] = {
    "fleet-wide": lambda customer, namespaces: {"scope_type": "all"},
    "per-type, the listed type": lambda customer, namespaces: {
        "scope_type": "type_customer",
        "scope_customer_id": customer,
        "scope_namespace_type": _LISTED,
    },
    "per-type, another type": lambda customer, namespaces: {
        "scope_type": "type_customer",
        "scope_customer_id": customer,
        "scope_namespace_type": "datasource",
    },
    "the customer namespace": lambda customer, namespaces: {
        "scope_type": "namespace",
        "scope_namespace_id": namespaces["customer"],
    },
    "a datasource namespace": lambda customer, namespaces: {
        "scope_type": "namespace",
        "scope_namespace_id": namespaces["datasource"],
    },
}

#: the roles a grant binds, as permissions: what each grants is the evaluator's to say, never asserted by hand
_ROLES: dict[str, tuple[Any, bool]] = {
    **_PERMISSION_SHAPES,
    "a customer admin": ({"customer": ["customer.read", "customer.admin"]}, False),
    "a datasource reader": ({"datasource": ["datasource.read"]}, False),
    "a narrow system role": ({"system": ["group.read"]}, False),
}


@pytest.mark.parametrize("scope", list(_SCOPES))
@pytest.mark.parametrize("role", list(_ROLES))
async def test_the_customer_set_a_listing_pages_within_is_the_set_the_clause_admits(
    pg_pool: asyncpg.Pool, scope: str, role: str
) -> None:
    """Flips if :func:`visible_customers` and :func:`caller_visible_customer_clause` admit different customers for
    one grant: every scope a grant takes, through every role shape. Both read the role a grant binds; they are two
    spellings (Python over the grant rows, SQL per row) of :meth:`Role.actions_for`'s one rule, held equal here."""
    permissions, double_encoded = _ROLES[role]
    customer_a, customer_b = uuid.uuid4(), uuid.uuid4()
    async with pg_pool.acquire() as conn:
        await _things(conn, customer_a, customer_b)
        namespaces = {kind: await _namespace(conn, customer_a, kind) for kind in ("customer", "datasource")}
        role_id = await _role(conn, permissions, double_encoded=double_encoded)
        holder = await _holder(conn, role_id, **_SCOPES[scope](customer_a, namespaces))
        clause_sees = await _visible(conn, holder)
        is_all, customers = await _customer_set(conn, holder)

    set_sees = {customer_a, customer_b} if is_all else customers & {customer_a, customer_b}
    assert set_sees == clause_sees, (scope, role, is_all, customers, clause_sees)


async def _any_type_set(conn: asyncpg.Connection, user_id: uuid.UUID) -> tuple[bool, set[uuid.UUID]]:
    seen = await visible_customers(conn, user_id=user_id, namespace_type=None)
    return seen.is_all, set(seen.customer_ids)


async def test_a_listing_of_no_one_type_counts_a_grant_only_through_a_role_acting_somewhere(
    pg_pool: asyncpg.Pool,
) -> None:
    """``namespace_type=None``: a fleet-wide grant through a role with no action at all (``{}``, or only empty
    buckets) is no grant; through a role acting on any type it is every customer. A per-type grant counts through a
    role acting on the grant's own type alone."""
    customer_a = uuid.uuid4()
    async with pg_pool.acquire() as conn:
        empty = await _holder(conn, await _role(conn, {}), scope_type="all")
        hollow = await _holder(conn, await _role(conn, {_LISTED: [], "*": []}), scope_type="all")
        narrow = await _holder(conn, await _role(conn, {"system": ["group.read"]}), scope_type="all")
        reader = await _role(conn, {"datasource": ["datasource.read"]})
        typed_here = await _holder(
            conn, reader, scope_type="type_customer", scope_customer_id=customer_a, scope_namespace_type="datasource"
        )
        typed_elsewhere = await _holder(
            conn, reader, scope_type="type_customer", scope_customer_id=customer_a, scope_namespace_type=_LISTED
        )
        sets = [await _any_type_set(conn, user) for user in (empty, hollow, narrow, typed_here, typed_elsewhere)]

    assert sets == [(False, set()), (False, set()), (True, set()), (False, {customer_a}), (False, set())]
