"""``ensure_conversation_owner_assignment`` against arity-enforcing collections.

Nothing exercised this function before these tests. ``test_authorize.py``
constructs its bundle with ``group_collection=object()``, which is honest for
what it asserts -- it never reaches the ensure path -- and is why the ensure
path had no coverage at all. ``groups`` is keyed on the composite
``(row_scope, group_id)`` and ``group_members`` on ``(group_id, id)``; both
reads passed the bare id, so the function raised ``primary key arity
mismatch`` at its first statement in every process that reached it, and
neither the membership write nor the role assignment below had ever executed.

So these tests use the REAL :class:`GroupCollection` /
:class:`GroupMemberCollection` / :class:`RoleAssignmentCollection`, wired with
no L1 and no L2 over a mocked asyncpg pool. The pk declaration under test is
then the production one, ``normalize_pk`` is the production one, and the SQL
the pool is handed carries the values the caller actually addressed the row
with.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import NAMESPACE_DNS, UUID, uuid5, uuid7

import pytest
from threetears.agent.acl import (
    GroupCollection,
    GroupMemberCollection,
    Role,
    RoleAssignmentCollection,
)
from threetears.conversations.authorize import (
    CONVERSATION_OWNER_GROUP_PREFIX,
    CONVERSATION_OWNER_ROLE_NAME,
    ConversationAuthorizerDependencies,
    ensure_conversation_owner_assignment,
)


class _StubNamespace:
    """duck-typed namespace entity carrying the two fields the ensure reads.

    :ivar id: namespace UUID the owner grant is scoped to
    :ivar customer_id: owning customer UUID
    """

    __slots__ = ("id", "customer_id")

    def __init__(self, *, id: UUID, customer_id: UUID) -> None:
        """store the namespace id and its owning customer.

        :param id: namespace UUID
        :ptype id: UUID
        :param customer_id: owning customer UUID
        :ptype customer_id: UUID
        :return: nothing
        :rtype: None
        """
        self.id = id
        self.customer_id = customer_id


# not a Fake<Name>: the parity walker does not inspect this, and a marker
# would claim a surface comparison that is not run.
class _BuiltinRoleCollection:
    """role collection stand-in serving one builtin role.

    only :meth:`list_builtin` participates in the ensure path; the rest of
    :class:`~threetears.agent.acl.collections.RoleCollection`'s surface is
    inherited from :class:`~threetears.core.collections.base.BaseCollection`
    and is not reached here.
    """

    def __init__(self, role: Role) -> None:
        """store the single builtin role this stand-in reports.

        :param role: builtin role the ensure path looks for by name
        :ptype role: Role
        :return: nothing
        :rtype: None
        """
        self._role = role

    async def list_builtin(self) -> tuple[Role, ...]:
        """return the one builtin role.

        :return: single-element tuple carrying the builtin role
        :rtype: tuple[Role, ...]
        """
        return (self._role,)


def _pool() -> AsyncMock:
    """build a mocked asyncpg pool that answers every read as a miss.

    :return: pool whose ``fetchrow`` misses and whose ``execute`` reports one
        affected row
    :rtype: AsyncMock
    """
    pool = AsyncMock()
    pool.fetchrow = AsyncMock(return_value=None)
    pool.fetch = AsyncMock(return_value=[])
    pool.execute = AsyncMock(return_value="INSERT 0 1")
    return pool


def _collection(cls: type, pool: AsyncMock) -> Any:
    """build a real collection over a mocked pool, with no L1 and no L2.

    :param cls: collection class to construct
    :ptype cls: type
    :param pool: mocked asyncpg pool serving as the L3 backend
    :ptype pool: AsyncMock
    :return: constructed collection instance
    :rtype: Any
    """
    registry = MagicMock()
    registry.get_l1_backend.return_value = None
    registry.get_l3_pool.return_value = pool
    registry.register.return_value = None
    registry.publish_invalidation = AsyncMock()

    config = MagicMock()
    config.collection_flush = "ALWAYS"
    config.collection_flush_tables = ""

    return cls(registry=registry, config=config, nats_client=None)


@pytest.fixture
def wiring() -> tuple[ConversationAuthorizerDependencies, dict[str, AsyncMock], UUID, UUID, UUID]:
    """build the dependency bundle over real rbac collections.

    :return: the bundle, the per-table pools, and the user / customer /
        namespace UUIDs the assertions address rows by
    :rtype: tuple[ConversationAuthorizerDependencies, dict[str, AsyncMock], UUID, UUID, UUID]
    """
    pools = {"groups": _pool(), "group_members": _pool(), "role_assignments": _pool()}
    deps = ConversationAuthorizerDependencies(
        acl_cache=MagicMock(),
        namespace_collection=MagicMock(),
        group_collection=_collection(GroupCollection, pools["groups"]),
        group_member_collection=_collection(GroupMemberCollection, pools["group_members"]),
        role_collection=_BuiltinRoleCollection(
            Role(id=uuid7(), name=CONVERSATION_OWNER_ROLE_NAME, permissions={}, is_built_in=True),
        ),
        role_assignment_collection=_collection(RoleAssignmentCollection, pools["role_assignments"]),
    )
    return deps, pools, uuid7(), uuid7(), uuid7()


@pytest.mark.asyncio
async def test_ensure_addresses_groups_by_its_composite_key(
    wiring: tuple[ConversationAuthorizerDependencies, dict[str, AsyncMock], UUID, UUID, UUID],
) -> None:
    """the ``groups`` read binds ``row_scope`` alongside ``group_id``.

    :param wiring: bundle / pools / ids fixture
    :ptype wiring: tuple[ConversationAuthorizerDependencies, dict[str, AsyncMock], UUID, UUID, UUID]
    :return: nothing
    :rtype: None
    """
    deps, pools, user_id, customer_id, namespace_id = wiring
    await ensure_conversation_owner_assignment(
        user_id=user_id,
        namespace=_StubNamespace(id=namespace_id, customer_id=customer_id),
        deps=deps,
    )

    expected_group_id = uuid5(
        NAMESPACE_DNS,
        f"threetears.groups.{CONVERSATION_OWNER_GROUP_PREFIX}.{customer_id.hex}.{user_id.hex}",
    )
    args = pools["groups"].fetchrow.await_args.args
    assert args[1:] == ("customer", expected_group_id)


@pytest.mark.asyncio
async def test_ensure_addresses_group_members_by_its_composite_key(
    wiring: tuple[ConversationAuthorizerDependencies, dict[str, AsyncMock], UUID, UUID, UUID],
) -> None:
    """the ``group_members`` read binds ``group_id`` alongside the row id.

    :param wiring: bundle / pools / ids fixture
    :ptype wiring: tuple[ConversationAuthorizerDependencies, dict[str, AsyncMock], UUID, UUID, UUID]
    :return: nothing
    :rtype: None
    """
    deps, pools, user_id, customer_id, namespace_id = wiring
    await ensure_conversation_owner_assignment(
        user_id=user_id,
        namespace=_StubNamespace(id=namespace_id, customer_id=customer_id),
        deps=deps,
    )

    expected_group_id = uuid5(
        NAMESPACE_DNS,
        f"threetears.groups.{CONVERSATION_OWNER_GROUP_PREFIX}.{customer_id.hex}.{user_id.hex}",
    )
    expected_membership_id = uuid5(
        NAMESPACE_DNS,
        f"threetears.group_members.{expected_group_id.hex}.{user_id.hex}",
    )
    args = pools["group_members"].fetchrow.await_args.args
    assert args[1:] == (expected_group_id, expected_membership_id)


@pytest.mark.asyncio
async def test_ensure_writes_the_group_the_membership_and_the_assignment(
    wiring: tuple[ConversationAuthorizerDependencies, dict[str, AsyncMock], UUID, UUID, UUID],
) -> None:
    """all three rows are written, so the function runs past its first statement.

    the point of this one is coverage of the whole body: before the composite
    keys were supplied, the first ``get`` raised and neither the membership nor
    the role assignment was ever reached by any caller.

    :param wiring: bundle / pools / ids fixture
    :ptype wiring: tuple[ConversationAuthorizerDependencies, dict[str, AsyncMock], UUID, UUID, UUID]
    :return: nothing
    :rtype: None
    """
    deps, pools, user_id, customer_id, namespace_id = wiring
    await ensure_conversation_owner_assignment(
        user_id=user_id,
        namespace=_StubNamespace(id=namespace_id, customer_id=customer_id),
        deps=deps,
    )

    assert pools["groups"].execute.await_count == 1
    assert pools["group_members"].execute.await_count == 1
    assert pools["role_assignments"].execute.await_count == 1
    assert namespace_id in pools["role_assignments"].execute.await_args.args


@pytest.mark.asyncio
async def test_ensure_skips_every_write_when_the_owner_role_is_unseeded(
    wiring: tuple[ConversationAuthorizerDependencies, dict[str, AsyncMock], UUID, UUID, UUID],
) -> None:
    """the admitted twin's refusal case: no builtin role, no rows written.

    :param wiring: bundle / pools / ids fixture
    :ptype wiring: tuple[ConversationAuthorizerDependencies, dict[str, AsyncMock], UUID, UUID, UUID]
    :return: nothing
    :rtype: None
    """
    deps, pools, user_id, customer_id, namespace_id = wiring
    deps.role_collection = _BuiltinRoleCollection(
        Role(id=uuid7(), name="SomeOtherRole", permissions={}, is_built_in=True),
    )

    await ensure_conversation_owner_assignment(
        user_id=user_id,
        namespace=_StubNamespace(id=namespace_id, customer_id=customer_id),
        deps=deps,
    )

    assert pools["groups"].fetchrow.await_count == 0
    assert pools["groups"].execute.await_count == 0
    assert pools["group_members"].execute.await_count == 0
    assert pools["role_assignments"].execute.await_count == 0
