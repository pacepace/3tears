"""a list shows a row only through a grant whose role acts where its scope needs it to, on a real database.

the visibility clause and the customer set a listing pages within (once the RLS customer-set query, now
:func:`visible_customers`) once ignored the role an assignment binds:
an assignment of ANY role widened a caller's lists. a narrow role bound at ``scope_type='all'``
(one action on one system namespace) showed its holder every customer's rows of every type, and
a role bound to a namespace where it grants nothing opened that namespace's customer. these
tests prove both now admit a fleet-wide or per-type grant only through a role acting
on the listed type, and a namespace grant only through a role acting on the namespace it is bound
to; and that the SQL reads a role by :func:`permissions_act_on`'s rule, narrower than the evaluator's
(an action counts only on its own type): through the production :class:`RoleCollection`, the
evaluator's decoding with that rule over it decides the same shapes the SQL decides, the wildcard
bucket, an empty bucket, a bucket that is not a list, a JSON ``null`` action and a double-encoded
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
from threetears.agent.acl.types import action_acts_on_sql, decode_role_permissions, permissions_act_on
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


#: every listing type a consumer asks the customer set or the clause for, None (a listing of no one type) included
_LISTING_TYPES: tuple[str | None, ...] = (
    None,
    "agent",
    "api_key",
    "channel",
    "customer",
    "datasource",
    "dataset",
    "group",
    "knowledge",
    "memory",
    "tool",
    "workspace",
)


async def test_a_narrow_action_in_the_customer_bucket_opens_no_list_of_that_customer(pg_pool: asyncpg.Pool) -> None:
    """the hole: a role whose ``customer`` bucket holds one narrow action (``group.read``), bound on the customer
    namespace, once handed its holder that customer for every listing type, because the role "acted" on the
    namespace's type by having a non-empty bucket there, whichever action the bucket held.

    a principal of the same customer holding no grant sees nothing; the narrow holder must see the same.
    """
    customer_a = uuid.uuid4()
    async with pg_pool.acquire() as conn:
        await _things(conn, customer_a)
        customer_namespace = await _namespace(conn, customer_a, "customer")
        narrow = await _role(conn, {"customer": ["group.read"]})
        holder = await _holder(conn, narrow, scope_type="namespace", scope_namespace_id=customer_namespace)
        sets: dict[str | None, tuple[bool, set[uuid.UUID]]] = {}
        rows: dict[str, set[uuid.UUID]] = {}
        for listing in _LISTING_TYPES:
            seen = await visible_customers(conn, user_id=holder, namespace_type=listing)
            sets[listing] = (seen.is_all, set(seen.customer_ids))
            if listing is not None:
                fragment, params = caller_visible_customer_clause(
                    user_id=holder, customer_id_column="listed_things.customer_id", scope_namespace_type=listing
                )
                fetched = await conn.fetch(f"SELECT customer_id FROM listed_things WHERE {fragment}", *params)
                rows[listing] = {row["customer_id"] for row in fetched}

    assert sets == {listing: (False, set()) for listing in _LISTING_TYPES}
    assert rows == {listing: set() for listing in _LISTING_TYPES if listing is not None}


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
#: grants an action of the listed type's own is :func:`permissions_act_on`'s verdict over the role the
#: real collection decodes below, never asserted by hand.
_PERMISSION_SHAPES: dict[str, tuple[Any, bool]] = {
    "the listed type's bucket": ({_LISTED: ["read"]}, False),
    "the wildcard bucket": ({"*": ["read"]}, False),
    "another type's bucket only": ({"system": ["group.read"]}, False),
    "the listed type's bucket, another type's action": ({_LISTED: ["group.read"]}, False),
    "the listed type's bucket, its own action": ({_LISTED: [f"{_LISTED}.read"]}, False),
    "the wildcard bucket, another type's action": ({"*": ["group.read"]}, False),
    "the listed type's bucket, a JSON null action": ({_LISTED: [None]}, False),
    "the listed type's bucket, a JSON null beside another type's action": ({_LISTED: [None, "group.read"]}, False),
    "the wildcard bucket, a JSON null action": ({"*": [None]}, False),
    "an empty bucket": ({_LISTED: []}, False),
    "a bucket that is not a list": ({_LISTED: "read"}, False),
    "no buckets": ({}, False),
    "the listed type's bucket, double-encoded": ({_LISTED: ["read"]}, True),
    "another type's bucket, double-encoded": ({"system": ["group.read"]}, True),
}


@pytest.mark.parametrize("shape", list(_PERMISSION_SHAPES))
async def test_the_sql_reads_a_role_as_the_evaluator_does(pg_pool: asyncpg.Pool, shape: str) -> None:
    """for every permissions shape, the clause shows rows exactly when :meth:`Role.actions_for` holds an action of
    the listed type's own (:func:`permissions_act_on`).

    the evaluator's role comes through the production :class:`RoleCollection`, so its decoding of
    the stored value -- a double-encoded string, a bucket that is not a list -- is the one the SQL is
    held to; which of the decoded actions count is the visibility rule's, the same one on both sides.
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
    granted = permissions_act_on(role.permissions, _LISTED)

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

#: the roles a grant binds, as permissions: what each opens is :func:`permissions_act_on`'s to say, never asserted by
#: hand
_ROLES: dict[str, tuple[Any, bool]] = {
    **_PERMISSION_SHAPES,
    "a customer admin": ({"customer": ["customer.read", "customer.admin"]}, False),
    "a datasource reader": ({"datasource": ["datasource.read"]}, False),
    "a narrow system role": ({"system": ["group.read"]}, False),
    "a narrow customer role": ({"customer": ["group.read"]}, False),
    "a customer admin with a narrow action beside": ({"customer": ["group.read", "customer.read"]}, False),
    "a parameterized workspace verb": ({"workspace": ["read_file_matching:**/*.yaml"]}, False),
}


@pytest.mark.parametrize("scope", list(_SCOPES))
@pytest.mark.parametrize("role", list(_ROLES))
async def test_the_customer_set_a_listing_pages_within_is_the_set_the_clause_admits(
    pg_pool: asyncpg.Pool, scope: str, role: str
) -> None:
    """Flips if :func:`visible_customers` and :func:`caller_visible_customer_clause` admit different customers for
    one grant: every scope a grant takes, through every role shape. Both read the role a grant binds; they are two
    spellings (Python over the grant rows, SQL per row) of :func:`permissions_act_on`'s rule -- the evaluator's
    decoding of :meth:`Role.actions_for`, narrowed to the actions of the type's own -- held equal here."""
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

    # ``narrow`` opens nothing: ``group.read`` is an action on groups, not on the ``system`` namespace it is filed
    # under, so the role holds no action of its own type anywhere (Pace's ruling 2026-11-20, option A). the hub's
    # GroupReader carries exactly this map, and its docstring names a scope-``all`` binding of it as a hazard.
    assert sets == [(False, set()), (False, set()), (False, set()), (False, {customer_a}), (False, set())]


#: the action shapes the SQL spelling is held to, each against a type it names and one it does not
_ACTIONS_AND_TYPES: tuple[tuple[str, str], ...] = tuple(
    (action, namespace_type)
    for action in (
        "read",
        "call",
        "",
        "read_file_matching:**/*",
        "read_file_matching:**/*.yaml",
        "read_file_matching:",
        "customer.read",
        "group.read",
        "control_plane.read",
        "datasource.relation_read",
        "shared_agent.invoke",
        "a.b.c",
        "datasource.read_matching:x.y",
        ".read",
    )
    for namespace_type in ("customer", "datasource", "group", "a", "")
)


async def test_the_sql_and_python_action_rules_agree(pg_pool: asyncpg.Pool) -> None:
    """Flips if :func:`action_acts_on_sql` and the Python rule (:func:`permissions_act_on` over a role holding that one
    action in the type's own bucket) disagree on any action shape: the per-row clause and :func:`visible_customers`
    would then admit different customers for one grant."""
    async with pg_pool.acquire() as conn:
        answers = {
            (action, namespace_type): await conn.fetchval(
                f"SELECT {action_acts_on_sql('$1::text', '$2::text')}", action, namespace_type
            )
            for action, namespace_type in _ACTIONS_AND_TYPES
        }

    expected = {
        (action, namespace_type): permissions_act_on(
            decode_role_permissions({namespace_type: [action]}), namespace_type
        )
        for action, namespace_type in _ACTIONS_AND_TYPES
    }
    assert answers == expected


#: the platform's seeded roles, as ``(permissions, the namespace type each is bound on)``: the hub's v001 seeds and
#: later migrations (hub ``src/aibots/hub/migrations``), 3tears' own PlatformBuiltinToolUser, and the narrow role the
#: defect was proven with. ``sees`` is whether a grant of it on a namespace of that type, naming a customer, opens the
#: customer: every seeded role but the narrow one keeps exactly the visibility it had before option A.
_SEEDED_ROLES: dict[str, tuple[dict[str, list[str]], str, bool]] = {
    "CustomerAdmin": (
        {"customer": ["customer.read", "customer.admin", "customer.impersonate"]},
        "customer",
        True,
    ),
    "CustomerMember": ({"customer": ["customer.read"]}, "customer", True),
    "Support": (
        {
            "audit": ["audit.read"],
            "customer": ["customer.read"],
            "workspace": ["read"],
            "conversation": ["conversation.read"],
        },
        "customer",
        True,
    ),
    "DatasourceReader": ({"datasource": ["datasource.read", "datasource.relation_read"]}, "datasource", True),
    "DatasourceBuilder": ({"datasource": ["datasource.build"]}, "datasource", True),
    "ChannelSender": ({"channel": ["channel.read", "channel.send"]}, "channel", True),
    "ChannelDmSender": ({"channel": ["channel.dm"]}, "channel", True),
    "ApiKeyAdmin": ({"api_key": ["api_key.admin", "api_key.read"]}, "api_key", True),
    "ToolCaller": ({"tool": ["tool.call"]}, "tool", True),
    "ToolApprover": ({"tool": ["tool.approve"]}, "tool", True),
    "PlatformBuiltinToolUser": ({"tool": ["call"]}, "tool", True),
    "ModelUser": ({"model": ["model.invoke"]}, "model", True),
    "MemoryWriter": ({"memory": ["memory.write"]}, "memory", True),
    "SharedAgentInvoker": ({"shared_agent": ["shared_agent.invoke"]}, "shared_agent", True),
    "WorkspaceEditor": (
        {
            "workspace": ["read", "read_file_matching:**/*", "write", "write_file_matching:**/*"],
            "workspace_file": ["read", "write"],
        },
        "workspace",
        True,
    ),
    "AgentDataReader": ({"agent": ["read"]}, "agent", True),
    "ToolProviderStorage": ({"tool_provider": ["read", "write"]}, "tool_provider", True),
    "a narrow customer role": ({"customer": ["group.read"]}, "customer", False),
}


@pytest.mark.parametrize("role", list(_SEEDED_ROLES))
async def test_a_seeded_role_keeps_the_customer_its_own_action_opens(pg_pool: asyncpg.Pool, role: str) -> None:
    """Flips if option A takes visibility from a role that legitimately had it, or leaves it with the narrow role: a
    grant on a namespace naming customer A opens A for every listing type, and through the clause, exactly when the
    role holds an action of that namespace's own type -- and the set and the clause agree on it."""
    permissions, namespace_type, sees = _SEEDED_ROLES[role]
    customer_a, customer_b = uuid.uuid4(), uuid.uuid4()
    async with pg_pool.acquire() as conn:
        await _things(conn, customer_a, customer_b)
        namespace = await _namespace(conn, customer_a, namespace_type)
        holder = await _holder(
            conn, await _role(conn, permissions), scope_type="namespace", scope_namespace_id=namespace
        )
        clause_sees = await _visible(conn, holder)
        sets = {
            listing: await visible_customers(conn, user_id=holder, namespace_type=listing) for listing in _LISTING_TYPES
        }

    expected = {customer_a} if sees else set()
    assert clause_sees == expected
    assert {listing: (seen.is_all, set(seen.customer_ids)) for listing, seen in sets.items()} == {
        listing: (False, expected) for listing in _LISTING_TYPES
    }


#: a grant chain's customers, by the letter of the customer each row names (None for a platform row): the caller's
#: membership row, the nesting edge above it (``"absent"``: the caller's group holds the grant itself), the granting
#: group and the role. the evaluator's cross-customer wall (``evaluator._filter_memberships`` on every membership edge,
#: ``_walk_assignments`` on the group and the role) drops each row naming a customer other than the namespace's.
_CHAINS: dict[str, dict[str, str | None]] = {
    "every row platform": {"member": None, "nested": "absent", "group": None, "role": None},
    "nested, every row platform": {"member": None, "nested": None, "group": None, "role": None},
    "the membership A's": {"member": "A", "nested": "absent", "group": None, "role": None},
    "the membership B's": {"member": "B", "nested": "absent", "group": None, "role": None},
    "the nesting edge A's": {"member": None, "nested": "A", "group": None, "role": None},
    "the nesting edge B's": {"member": None, "nested": "B", "group": None, "role": None},
    "the group A's": {"member": None, "nested": "absent", "group": "A", "role": None},
    "the group B's": {"member": None, "nested": "absent", "group": "B", "role": None},
    "the role A's": {"member": None, "nested": "absent", "group": None, "role": "A"},
    "the role B's": {"member": None, "nested": "absent", "group": None, "role": "B"},
    "membership and group B's": {"member": "B", "nested": "absent", "group": "B", "role": None},
    "membership A's, group B's": {"member": "A", "nested": "absent", "group": "B", "role": None},
}

#: the scopes a chain's grant takes, each covering namespaces of customer A, or every namespace
_WALL_SCOPES: tuple[str, ...] = ("namespace", "type_customer", "all")


def _walled(chain: dict[str, str | None], scope: str) -> tuple[bool, set[str]]:
    """the customers the evaluator lets ``chain``'s grant reach, by letter: ``(every customer, these)``.

    a namespace of customer X is reached when no row of the chain names a customer other than X; a ``namespace`` or
    ``type_customer`` grant covers customer A's namespaces only, a fleet-wide grant every customer's.

    :param chain: the chain's customers
    :ptype chain: dict[str, str | None]
    :param scope: the grant's scope
    :ptype scope: str
    :return: ``(is_all, customers)``
    :rtype: tuple[bool, set[str]]
    """
    named = {value for value in chain.values() if value not in (None, "absent")}
    covered = {"A"} if scope != "all" else {"A", "B"}
    reached = {customer for customer in covered if named <= {customer}}
    is_all = scope == "all" and not named
    return is_all, set() if is_all else reached


async def _seed_chain(
    conn: asyncpg.Connection, chain: dict[str, str | None], scope: str, customers: dict[str, uuid.UUID]
) -> uuid.UUID:
    """seed a caller whose grant reaches it through ``chain``'s rows, on ``scope``.

    :param conn: open connection
    :ptype conn: asyncpg.Connection
    :param chain: the chain's customers, by letter
    :ptype chain: dict[str, str | None]
    :param scope: the grant's scope
    :ptype scope: str
    :param customers: each letter's customer
    :ptype customers: dict[str, uuid.UUID]
    :return: the caller
    :rtype: uuid.UUID
    """

    def customer(letter: str | None) -> uuid.UUID | None:
        """the customer a letter names, or None for a platform row.

        :param letter: the letter, or None
        :ptype letter: str | None
        :return: the customer, or None
        :rtype: uuid.UUID | None
        """
        return None if letter is None else customers[letter]

    async def group(letter: str | None) -> uuid.UUID:
        """seed one group of the customer ``letter`` names (a platform group for None).

        :param letter: the group's customer, by letter
        :ptype letter: str | None
        :return: the group
        :rtype: uuid.UUID
        """
        group_id = uuid.uuid4()
        await conn.execute(
            "INSERT INTO groups (row_scope, group_id, customer_id, name) VALUES ($1, $2, $3, $4)",
            "platform" if letter is None else "customer",
            group_id,
            customer(letter),
            f"group-{group_id.hex}",
        )
        return group_id

    async def member(group_id: uuid.UUID, member_type: str, member_id: uuid.UUID, letter: str | None) -> None:
        """seed one membership row naming the customer ``letter`` names.

        :param group_id: the group joined
        :ptype group_id: uuid.UUID
        :param member_type: ``user`` or ``group``
        :ptype member_type: str
        :param member_id: the member
        :ptype member_id: uuid.UUID
        :param letter: the row's customer, by letter
        :ptype letter: str | None
        :return: nothing
        :rtype: None
        """
        await conn.execute(
            "INSERT INTO group_members (id, group_id, member_type, member_id, customer_id) VALUES ($1, $2, $3, $4, $5)",
            uuid.uuid4(),
            group_id,
            member_type,
            member_id,
            customer(letter),
        )

    user_id = uuid.uuid4()
    nested = chain["nested"] != "absent"
    granting = await group(chain["group"])
    bottom = await group(None) if nested else granting
    await member(bottom, "user", user_id, chain["member"])
    if nested:
        await member(granting, "group", bottom, chain["nested"])
    role_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO roles (role_id, name, permissions, customer_id) VALUES ($1, $2, $3::text::jsonb, $4)",
        role_id,
        f"role-{role_id.hex}",
        json.dumps({"*": ["read"]}),
        customer(chain["role"]),
    )
    namespace_id = await _namespace(conn, customers["A"], "shared_agent") if scope == "namespace" else None
    await conn.execute(
        "INSERT INTO role_assignments (row_scope, assignment_id, role_id, group_id, scope_type, scope_namespace_id,"
        " scope_namespace_type, scope_customer_id) VALUES ('platform', $1, $2, $3, $4, $5, $6, $7)",
        uuid.uuid4(),
        role_id,
        granting,
        scope,
        namespace_id,
        _LISTED if scope == "type_customer" else None,
        customers["A"] if scope == "type_customer" else None,
    )
    return user_id


@pytest.mark.parametrize("scope", _WALL_SCOPES)
@pytest.mark.parametrize("chain", list(_CHAINS))
async def test_a_grant_opens_only_the_customers_the_evaluators_wall_lets_it_reach(
    pg_pool: asyncpg.Pool, chain: str, scope: str
) -> None:
    """Flips if either arm admits a customer the evaluator's cross-customer wall refuses the grant on, or refuses one
    it lets through: a membership edge, the granting group or the role naming a customer other than the namespace's
    counts for nothing there, and a platform row (no customer) passes everywhere. :func:`visible_customers` (for the
    listed type and for no one type) and :func:`caller_visible_customer_clause` must both say exactly that."""
    customers = {"A": uuid.uuid4(), "B": uuid.uuid4()}
    async with pg_pool.acquire() as conn:
        await _things(conn, *customers.values())
        holder = await _seed_chain(conn, _CHAINS[chain], scope, customers)
        clause_sees = await _visible(conn, holder)
        typed = await visible_customers(conn, user_id=holder, namespace_type=_LISTED)
        untyped = await visible_customers(conn, user_id=holder, namespace_type=None)

    is_all, letters = _walled(_CHAINS[chain], scope)
    expected = (is_all, {customers[letter] for letter in letters})
    every = set(customers.values())
    assert clause_sees == (every if is_all else expected[1]), (chain, scope)
    assert (typed.is_all, set(typed.customer_ids)) == expected, (chain, scope)
    assert (untyped.is_all, set(untyped.customer_ids)) == expected, (chain, scope)


async def test_another_customers_group_holding_an_invoker_grant_opens_no_list_of_the_owner(
    pg_pool: asyncpg.Pool,
) -> None:
    """a customer-B group holding SharedAgentInvoker on customer A's ``shared_agent`` namespace opens no list of A.

    The evaluator drops a membership or a group naming a customer against a namespace of another customer, so the grant
    authorizes nothing there; visibility counts it for nothing either.
    """
    customers = {"A": uuid.uuid4(), "B": uuid.uuid4()}
    async with pg_pool.acquire() as conn:
        await _things(conn, *customers.values())
        holder = await _seed_chain(conn, _CHAINS["membership and group B's"], "namespace", customers)
        clause_sees = await _visible(conn, holder)
        set_sees = {
            listing: set((await visible_customers(conn, user_id=holder, namespace_type=listing)).customer_ids)
            for listing in _LISTING_TYPES
        }

    assert clause_sees == set()
    assert set_sees == {listing: set() for listing in _LISTING_TYPES}


async def test_a_platform_group_holding_an_invoker_grant_opens_the_owners_lists(pg_pool: asyncpg.Pool) -> None:
    """the shape the hub grants a shared agent across customers with (a platform group and platform membership rows,
    the only one ``SHARED_AGENT_REQUIRES_PLATFORM_GROUP`` admits): the evaluator grants it on customer A's namespace, so
    visibility opens A, for every listing type and through the clause."""
    customers = {"A": uuid.uuid4(), "B": uuid.uuid4()}
    async with pg_pool.acquire() as conn:
        await _things(conn, *customers.values())
        holder = await _seed_chain(conn, _CHAINS["every row platform"], "namespace", customers)
        clause_sees = await _visible(conn, holder)
        set_sees = {
            listing: set((await visible_customers(conn, user_id=holder, namespace_type=listing)).customer_ids)
            for listing in _LISTING_TYPES
        }

    assert clause_sees == {customers["A"]}
    assert set_sees == {listing: {customers["A"]} for listing in _LISTING_TYPES}
