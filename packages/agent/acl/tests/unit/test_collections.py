"""unit tests for the canonical rbac three-tier Collections.

these tests exercise the Collection methods every rbac-consuming
3tears app shares (canonical CRUD + evaluator-loader query shapes
+ universal lookup helpers). hub-specific subclasses live in the
deploying repo and have their own tests there.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid7

import pytest
from threetears.agent.acl import (
    GroupCollection,
    GroupMemberCollection,
    GroupMembership,
    MemberType,
    NamespaceCollection,
    Role,
    RoleAssignment,
    RoleAssignmentCollection,
    RoleCollection,
    ScopeType,
)


def _make_collection(
    cls: type,
    *,
    l3_pool: AsyncMock | None = None,
) -> Any:
    """build a Collection instance with mocked registry + config.

    :param cls: Collection class to instantiate
    :ptype cls: type
    :param l3_pool: optional mocked pool
    :ptype l3_pool: AsyncMock | None
    :return: Collection instance with mocks wired in
    :rtype: Any
    """
    mock_registry = MagicMock()
    mock_registry.get_l1_backend.return_value = None
    mock_registry.get_l3_pool.return_value = l3_pool
    mock_registry.register.return_value = None

    mock_config = MagicMock()
    mock_config.collection_flush = "ALWAYS"
    mock_config.collection_flush_tables = ""

    return cls(registry=mock_registry, config=mock_config)


def _group_row(
    *,
    customer_id: UUID | None = None,
    name: str = "Engineering",
) -> dict[str, Any]:
    """build a fake ``groups`` row.

    :param customer_id: owning customer UUID
    :ptype customer_id: UUID | None
    :param name: group name
    :ptype name: str
    :return: row dict
    :rtype: dict[str, Any]
    """
    now = datetime.now(UTC)
    cid = customer_id if customer_id is not None else uuid7()
    return {
        "row_scope": "customer" if cid is not None else "platform",
        "group_id": uuid7(),
        "customer_id": cid,
        "name": name,
        "description": f"{name} description",
        "date_created": now,
        "date_updated": now,
    }


def _role_row(
    *,
    name: str = "Reader",
    is_builtin: bool = True,
) -> dict[str, Any]:
    """build a fake ``roles`` row.

    :param name: role name
    :ptype name: str
    :param is_builtin: builtin flag
    :ptype is_builtin: bool
    :return: row dict
    :rtype: dict[str, Any]
    """
    now = datetime.now(UTC)
    return {
        "role_id": uuid7(),
        "name": name,
        "description": f"{name} role",
        "permissions": {"*": ["read"]},
        "is_builtin": is_builtin,
        "date_created": now,
        "date_updated": now,
    }


def _assignment_row(
    *,
    group_id: UUID,
    role_id: UUID,
    scope_type: str = "namespace",
    scope_namespace_id: UUID | None = None,
    scope_namespace_type: str | None = None,
    scope_customer_id: UUID | None = None,
) -> dict[str, Any]:
    """build a fake ``role_assignments`` row.

    :param group_id: owning group UUID
    :ptype group_id: UUID
    :param role_id: role UUID
    :ptype role_id: UUID
    :param scope_type: scope discriminator
    :ptype scope_type: str
    :param scope_namespace_id: target namespace UUID for namespace
        scope
    :ptype scope_namespace_id: UUID | None
    :param scope_namespace_type: namespace type for type_customer
        scope
    :ptype scope_namespace_type: str | None
    :param scope_customer_id: customer UUID for type_customer scope
    :ptype scope_customer_id: UUID | None
    :return: row dict
    :rtype: dict[str, Any]
    """
    now = datetime.now(UTC)
    if scope_type == "all":
        row_scope = "platform"
    elif scope_type == "type_customer" and scope_customer_id is None:
        row_scope = "platform"
    else:
        row_scope = "customer"
    return {
        "row_scope": row_scope,
        "assignment_id": uuid7(),
        "role_id": role_id,
        "group_id": group_id,
        "scope_type": scope_type,
        "scope_namespace_id": scope_namespace_id,
        "scope_namespace_type": scope_namespace_type,
        "scope_customer_id": scope_customer_id,
        "granted_by": None,
        "date_granted": now,
        "managed_by": "manual",
    }


# ---------------------------------------------------------------------------
# GroupCollection
# ---------------------------------------------------------------------------


class TestGroupCollectionListByCustomer:
    """``GroupCollection.list_by_customer`` filter shape."""

    @pytest.mark.asyncio
    async def test_returns_entities_for_customer(self) -> None:
        """rows fanned out as ``GroupEntity`` instances."""
        customer_id = uuid7()
        rows = [
            _group_row(customer_id=customer_id, name="Group A"),
            _group_row(customer_id=customer_id, name="Group B"),
        ]
        pool = AsyncMock()
        pool.fetch.return_value = rows
        coll = _make_collection(GroupCollection, l3_pool=pool)

        result = await coll.list_by_customer(customer_id)

        assert len(result) == 2
        assert {e.name for e in result} == {"Group A", "Group B"}
        pool.fetch.assert_awaited_once()
        sql = pool.fetch.await_args.args[0]
        assert "row_scope = 'customer'" in sql
        assert pool.fetch.await_args.args[1] == customer_id

    @pytest.mark.asyncio
    async def test_empty_when_no_rows(self) -> None:
        """empty fetch returns empty list (not None)."""
        pool = AsyncMock()
        pool.fetch.return_value = []
        coll = _make_collection(GroupCollection, l3_pool=pool)
        result = await coll.list_by_customer(uuid7())
        assert result == []


class TestGroupCollectionGetMany:
    """``GroupCollection.get_many`` bulk lookup."""

    @pytest.mark.asyncio
    async def test_empty_input_no_round_trip(self) -> None:
        """empty input short-circuits without a SQL call."""
        pool = AsyncMock()
        coll = _make_collection(GroupCollection, l3_pool=pool)
        result = await coll.get_many([])
        assert result == []
        pool.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_entities_for_ids(self) -> None:
        """fetched rows surface as ``GroupEntity`` instances."""
        gid_a = uuid7()
        gid_b = uuid7()
        row_a = _group_row(name="A")
        row_b = _group_row(name="B")
        row_a["group_id"] = gid_a
        row_b["group_id"] = gid_b
        pool = AsyncMock()
        pool.fetch.return_value = [row_a, row_b]
        coll = _make_collection(GroupCollection, l3_pool=pool)
        result = await coll.get_many([gid_a, gid_b])
        assert {e.id for e in result} == {gid_a, gid_b}

    @pytest.mark.asyncio
    async def test_coerces_string_uuid_from_proxy_pool(self) -> None:
        """string UUIDs from the NATS proxy pool round-trip to UUID."""
        gid = uuid7()
        cid = uuid7()
        row = _group_row(customer_id=cid)
        row["group_id"] = str(gid)
        row["customer_id"] = str(cid)
        pool = AsyncMock()
        pool.fetch.return_value = [row]
        coll = _make_collection(GroupCollection, l3_pool=pool)
        result = await coll.get_many([gid])
        assert len(result) == 1
        assert isinstance(result[0].id, UUID)


class TestGroupMemberLoadForUser:
    """``GroupMemberCollection.load_for_user`` returns dataclasses."""

    @pytest.mark.asyncio
    async def test_returns_membership_dataclass(self) -> None:
        """rows surface as :class:`GroupMembership`, not entity."""
        user_id = uuid7()
        group_id = uuid7()
        cid = uuid7()
        rows = [
            {
                "group_id": group_id,
                "member_type": "user",
                "member_id": user_id,
                "customer_id": cid,
            },
        ]
        pool = AsyncMock()
        pool.fetch.return_value = rows
        coll = _make_collection(GroupMemberCollection, l3_pool=pool)
        result = await coll.load_for_user(user_id)
        assert len(result) == 1
        assert isinstance(result[0], GroupMembership)
        assert result[0].member_type == MemberType.USER
        assert result[0].group_id == group_id
        assert result[0].customer_id == cid

    @pytest.mark.asyncio
    async def test_filters_by_user_member_type(self) -> None:
        """SQL filters on ``member_type='user'`` exactly."""
        pool = AsyncMock()
        pool.fetch.return_value = []
        coll = _make_collection(GroupMemberCollection, l3_pool=pool)
        await coll.load_for_user(uuid7())
        sql = pool.fetch.await_args.args[0]
        assert "member_type = 'user'" in sql


class TestGroupMemberLoadForAgent:
    """``GroupMemberCollection.load_for_agent`` symmetric to user side."""

    @pytest.mark.asyncio
    async def test_returns_membership_dataclass(self) -> None:
        agent_id = uuid7()
        group_id = uuid7()
        rows = [
            {
                "group_id": group_id,
                "member_type": "agent",
                "member_id": agent_id,
                "customer_id": uuid7(),
            },
        ]
        pool = AsyncMock()
        pool.fetch.return_value = rows
        coll = _make_collection(GroupMemberCollection, l3_pool=pool)
        result = await coll.load_for_agent(agent_id)
        assert len(result) == 1
        assert result[0].member_type == MemberType.AGENT


class TestRoleListBuiltin:
    """``RoleCollection.list_builtin`` filters on ``is_builtin=TRUE``."""

    @pytest.mark.asyncio
    async def test_filters_builtin(self) -> None:
        rows = [_role_row(name="Reader"), _role_row(name="Writer")]
        pool = AsyncMock()
        pool.fetch.return_value = rows
        coll = _make_collection(RoleCollection, l3_pool=pool)
        result = await coll.list_builtin()
        assert len(result) == 2
        sql = pool.fetch.await_args.args[0]
        assert "is_builtin = TRUE" in sql


class TestRoleGetMany:
    """``RoleCollection.get_many`` returns ACL :class:`Role` dataclass."""

    @pytest.mark.asyncio
    async def test_empty_input_no_round_trip(self) -> None:
        pool = AsyncMock()
        coll = _make_collection(RoleCollection, l3_pool=pool)
        result = await coll.get_many([])
        assert result == []
        pool.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_role_dataclass_with_permissions(self) -> None:
        """permissions JSONB coerces to ``dict[str, frozenset[str]]``."""
        role_id = uuid7()
        rows = [
            {
                "role_id": role_id,
                "name": "Reader",
                "permissions": {"workspace": ["read"], "*": ["read"]},
                "is_builtin": True,
            },
        ]
        pool = AsyncMock()
        pool.fetch.return_value = rows
        coll = _make_collection(RoleCollection, l3_pool=pool)
        result = await coll.get_many([role_id])
        assert len(result) == 1
        role = result[0]
        assert isinstance(role, Role)
        assert role.permissions["workspace"] == frozenset({"read"})
        assert role.permissions["*"] == frozenset({"read"})


class TestRoleAssignmentLoadForGroups:
    """``RoleAssignmentCollection.load_for_groups`` returns dataclasses."""

    @pytest.mark.asyncio
    async def test_empty_input_no_round_trip(self) -> None:
        pool = AsyncMock()
        coll = _make_collection(RoleAssignmentCollection, l3_pool=pool)
        result = await coll.load_for_groups([])
        assert result == []
        pool.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_assignment_dataclass(self) -> None:
        group_id = uuid7()
        role_id = uuid7()
        ns_id = uuid7()
        rows = [
            {
                "assignment_id": uuid7(),
                "role_id": role_id,
                "group_id": group_id,
                "scope_type": "namespace",
                "scope_namespace_id": ns_id,
                "scope_namespace_type": None,
                "scope_customer_id": None,
            },
        ]
        pool = AsyncMock()
        pool.fetch.return_value = rows
        coll = _make_collection(RoleAssignmentCollection, l3_pool=pool)
        result = await coll.load_for_groups([group_id])
        assert len(result) == 1
        ra = result[0]
        assert isinstance(ra, RoleAssignment)
        assert ra.scope_type == ScopeType.NAMESPACE
        assert ra.scope_namespace_id == ns_id


class TestEnsureGroupRoleAssignment:
    """``RoleAssignmentCollection.ensure_group_role_assignment`` idempotent."""

    @pytest.mark.asyncio
    async def test_returns_existing_id(self) -> None:
        """existing matching row wins; no INSERT."""
        existing_id = uuid7()
        pool = AsyncMock()
        pool.fetchrow.return_value = {"assignment_id": existing_id}
        coll = _make_collection(RoleAssignmentCollection, l3_pool=pool)
        result = await coll.ensure_group_role_assignment(
            group_id=uuid7(),
            role_id=uuid7(),
            scope_type="namespace",
            scope_id=uuid7(),
        )
        assert result == existing_id
        pool.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_inserts_new_when_absent(self) -> None:
        """missing row triggers INSERT and returns fresh uuid7."""
        pool = AsyncMock()
        pool.fetchrow.return_value = None
        pool.execute.return_value = "INSERT 0 1"
        coll = _make_collection(RoleAssignmentCollection, l3_pool=pool)
        result = await coll.ensure_group_role_assignment(
            group_id=uuid7(),
            role_id=uuid7(),
            scope_type="namespace",
            scope_id=uuid7(),
        )
        assert isinstance(result, UUID)
        pool.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rejects_unsupported_scope_type(self) -> None:
        coll = _make_collection(
            RoleAssignmentCollection,
            l3_pool=AsyncMock(),
        )
        with pytest.raises(ValueError):
            await coll.ensure_group_role_assignment(
                group_id=uuid7(),
                role_id=uuid7(),
                scope_type="type_customer",
                scope_id=None,
            )

    @pytest.mark.asyncio
    async def test_rejects_namespace_scope_without_id(self) -> None:
        coll = _make_collection(
            RoleAssignmentCollection,
            l3_pool=AsyncMock(),
        )
        with pytest.raises(ValueError):
            await coll.ensure_group_role_assignment(
                group_id=uuid7(),
                role_id=uuid7(),
                scope_type="namespace",
                scope_id=None,
            )

    @pytest.mark.asyncio
    async def test_rejects_all_scope_with_id(self) -> None:
        coll = _make_collection(
            RoleAssignmentCollection,
            l3_pool=AsyncMock(),
        )
        with pytest.raises(ValueError):
            await coll.ensure_group_role_assignment(
                group_id=uuid7(),
                role_id=uuid7(),
                scope_type="all",
                scope_id=uuid7(),
            )


class TestDeleteByGroupAndScope:
    """``RoleAssignmentCollection.delete_by_group_and_scope``."""

    @pytest.mark.asyncio
    async def test_returns_zero_on_empty_delete(self) -> None:
        pool = AsyncMock()
        pool.execute.return_value = "DELETE 0"
        coll = _make_collection(RoleAssignmentCollection, l3_pool=pool)
        n = await coll.delete_by_group_and_scope(
            group_id=uuid7(),
            scope_type="namespace",
            scope_id=uuid7(),
        )
        assert n == 0

    @pytest.mark.asyncio
    async def test_returns_count_on_match(self) -> None:
        pool = AsyncMock()
        pool.execute.return_value = "DELETE 3"
        coll = _make_collection(RoleAssignmentCollection, l3_pool=pool)
        n = await coll.delete_by_group_and_scope(
            group_id=uuid7(),
            scope_type="namespace",
            scope_id=uuid7(),
        )
        assert n == 3

    @pytest.mark.asyncio
    async def test_managed_by_filter_appended(self) -> None:
        """``managed_by`` argument adds a fifth predicate."""
        pool = AsyncMock()
        pool.execute.return_value = "DELETE 1"
        coll = _make_collection(RoleAssignmentCollection, l3_pool=pool)
        await coll.delete_by_group_and_scope(
            group_id=uuid7(),
            scope_type="namespace",
            scope_id=uuid7(),
            managed_by="auto:agent-yaml",
        )
        sql = pool.execute.await_args.args[0]
        assert "managed_by = $5" in sql


# ---------------------------------------------------------------------------
# NamespaceCollection
# ---------------------------------------------------------------------------


def _namespace_row(
    *,
    customer_id: UUID | None = None,
    namespace_type: str = "workspace",
    name: str = "ws.acme.alpha",
    owner_agent_id: UUID | None = None,
) -> dict[str, Any]:
    """build a fake ``namespaces`` row."""
    now = datetime.now(UTC)
    return {
        "row_scope": "platform" if customer_id is None else "customer",
        "namespace_id": uuid7(),
        "name": name,
        "namespace_type": namespace_type,
        "owner_agent_id": owner_agent_id,
        "customer_id": customer_id,
        "schema_name": None,
        "metadata": None,
        "date_created": now,
        "date_updated": now,
    }


class TestNamespaceCollectionFindById:
    """``NamespaceCollection.find_by_id`` resolves via UNIQUE (id)."""

    @pytest.mark.asyncio
    async def test_returns_entity(self) -> None:
        cid = uuid7()
        row = _namespace_row(customer_id=cid)
        pool = AsyncMock()
        pool.fetchrow.return_value = row
        coll = _make_collection(NamespaceCollection, l3_pool=pool)
        result = await coll.find_by_id(row["namespace_id"])
        assert result is not None
        assert result.id == row["namespace_id"]

    @pytest.mark.asyncio
    async def test_returns_none_when_absent(self) -> None:
        pool = AsyncMock()
        pool.fetchrow.return_value = None
        coll = _make_collection(NamespaceCollection, l3_pool=pool)
        result = await coll.find_by_id(uuid7())
        assert result is None


class TestNamespaceGetByName:
    """``NamespaceCollection.get_by_name`` resolves a unique name."""

    @pytest.mark.asyncio
    async def test_returns_entity(self) -> None:
        cid = uuid7()
        row = _namespace_row(customer_id=cid, name="ws.acme")
        pool = AsyncMock()
        pool.fetchrow.return_value = row
        coll = _make_collection(NamespaceCollection, l3_pool=pool)
        result = await coll.get_by_name("ws.acme")
        assert result is not None
        assert result.name == "ws.acme"

    @pytest.mark.asyncio
    async def test_returns_none_when_absent(self) -> None:
        pool = AsyncMock()
        pool.fetchrow.return_value = None
        coll = _make_collection(NamespaceCollection, l3_pool=pool)
        result = await coll.get_by_name("missing")
        assert result is None


class TestNamespaceGetByOwnerAndCustomer:
    """``NamespaceCollection.get_by_owner_and_customer`` triple lookup."""

    @pytest.mark.asyncio
    async def test_platform_scope_when_customer_none(self) -> None:
        """``customer_id=None`` queries the platform partition."""
        pool = AsyncMock()
        pool.fetchrow.return_value = None
        coll = _make_collection(NamespaceCollection, l3_pool=pool)
        await coll.get_by_owner_and_customer(
            namespace_type="shared",
            owner_agent_id=None,
            customer_id=None,
        )
        # row_scope is the first parameter
        assert pool.fetchrow.await_args.args[1] == "platform"

    @pytest.mark.asyncio
    async def test_customer_scope_when_customer_set(self) -> None:
        cid = uuid7()
        pool = AsyncMock()
        pool.fetchrow.return_value = None
        coll = _make_collection(NamespaceCollection, l3_pool=pool)
        await coll.get_by_owner_and_customer(
            namespace_type="memory",
            owner_agent_id=uuid7(),
            customer_id=cid,
        )
        assert pool.fetchrow.await_args.args[1] == "customer"


class TestNamespaceListIdsByCustomerAndType:
    """``list_ids_by_customer_and_type`` returns ids only."""

    @pytest.mark.asyncio
    async def test_returns_list_of_uuids(self) -> None:
        ids = [uuid7(), uuid7()]
        pool = AsyncMock()
        pool.fetch.return_value = [{"namespace_id": i} for i in ids]
        coll = _make_collection(NamespaceCollection, l3_pool=pool)
        result = await coll.list_ids_by_customer_and_type(uuid7(), "workspace")
        assert result == ids


class TestNamespaceListAllIds:
    """``list_all_ids`` returns every namespace id."""

    @pytest.mark.asyncio
    async def test_returns_all(self) -> None:
        ids = [uuid7(), uuid7(), uuid7()]
        pool = AsyncMock()
        pool.fetch.return_value = [{"namespace_id": i} for i in ids]
        coll = _make_collection(NamespaceCollection, l3_pool=pool)
        result = await coll.list_all_ids()
        assert result == ids
