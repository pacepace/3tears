"""a grant records its customer (acl v005): every writer derives it, ``create()`` and the ensure.

A ``namespace`` grant's customer is its namespace's (None for a platform namespace), a ``type_customer`` grant's is its
``scope_customer_id``, ``all`` and ``subtree`` have none. A customer admin's listing pins the column, so a writer that
left it unset would hide a customer's grant from its admin, and one that set it on a platform grant would show the
platform grant in a customer's page.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest
from threetears.agent.acl import RoleAssignmentCollection

from .collection_support import make_collection


def _created(data: dict[str, Any]) -> dict[str, Any]:
    """the row ``create()`` builds for a grant of ``data``'s scope."""
    collection = make_collection(RoleAssignmentCollection)
    entity = collection.create({"assignment_id": uuid7(), "role_id": uuid7(), "group_id": uuid7(), **data})
    row: dict[str, Any] = entity.to_dict()
    return row


def test_a_customerless_scope_records_no_customer() -> None:
    """Flips if an 'all' or 'subtree' grant records a customer (a platform grant in a customer's page)."""
    assert _created({"scope_type": "all"})["customer_id"] is None
    assert _created({"scope_type": "subtree", "scope_namespace_name": "tools.web"})["customer_id"] is None


def test_a_type_grant_records_its_scope_customer() -> None:
    """Flips if a type_customer grant's customer is not the customer its scope names."""
    customer = uuid7()
    row = _created({"scope_type": "type_customer", "scope_namespace_type": "agent", "scope_customer_id": customer})
    assert row["customer_id"] == customer


def test_a_namespace_grant_records_the_customer_its_writer_names() -> None:
    """Flips if a namespace grant's customer is not the one its writer read off the namespace, None included."""
    customer = uuid7()
    assert (
        _created({"scope_type": "namespace", "scope_namespace_id": uuid7(), "customer_id": customer})["customer_id"]
        == customer
    )
    assert (
        _created({"scope_type": "namespace", "scope_namespace_id": uuid7(), "customer_id": None})["customer_id"] is None
    )


def test_a_namespace_grant_without_its_customer_is_refused() -> None:
    """Flips if a namespace grant can be written without its namespace's customer (it would read as a platform
    grant)."""
    with pytest.raises(ValueError, match="pass customer_id"):
        _created({"scope_type": "namespace", "scope_namespace_id": uuid7()})


@pytest.mark.parametrize(
    "data",
    [
        {"scope_type": "all", "customer_id": uuid7()},
        {"scope_type": "subtree", "scope_namespace_name": "tools.web", "customer_id": uuid7()},
        {
            "scope_type": "type_customer",
            "scope_namespace_type": "agent",
            "scope_customer_id": uuid7(),
            "customer_id": uuid7(),
        },
    ],
)
def test_a_customer_contradicting_the_scope_is_refused(data: dict[str, Any]) -> None:
    """Flips if a writer may record a customer its grant's scope does not name."""
    with pytest.raises(ValueError, match="contradicts"):
        _created(data)


def test_create_fills_the_customer_from_the_scope() -> None:
    """Flips if create() leaves a type_customer grant's customer unset."""
    customer = uuid7()
    collection = make_collection(RoleAssignmentCollection)
    entity = collection.create(
        {
            "assignment_id": uuid7(),
            "role_id": uuid7(),
            "group_id": uuid7(),
            "scope_type": "type_customer",
            "scope_namespace_type": "agent",
            "scope_customer_id": customer,
        }
    )
    assert entity.to_dict()["customer_id"] == customer
    assert entity.to_dict()["row_scope"] == "customer"


def test_create_refuses_a_namespace_grant_without_its_customer() -> None:
    """Flips if create() accepts a namespace grant naming no customer."""
    collection = make_collection(RoleAssignmentCollection)
    with pytest.raises(ValueError, match="pass customer_id"):
        collection.create(
            {"assignment_id": uuid7(), "role_id": uuid7(), "group_id": uuid7(), "scope_type": "namespace"}
        )


async def test_an_ensured_namespace_grant_records_its_namespaces_customer() -> None:
    """Flips if the ensure inserts a namespace grant without the customer read off its namespace by key."""
    customer, namespace_id = uuid7(), uuid7()
    pool = AsyncMock()
    pool.fetchrow.return_value = None

    async def fetch(sql: str, *args: Any) -> list[dict[str, Any]]:
        # the grant's natural-key read finds none held; the namespace read answers its customer
        return [] if '"role_assignments"' in sql else [{"customer_id": customer}]

    pool.fetch.side_effect = fetch
    pool.fetchval.side_effect = lambda sql, *args: args[1]
    collection = make_collection(RoleAssignmentCollection, l3_pool=pool)
    collection.invalidate_cache_many = AsyncMock()
    await collection.ensure_group_role_assignment(
        group_id=uuid7(), role_id=uuid7(), scope_type="namespace", scope_id=namespace_id
    )
    read = pool.fetch.await_args
    assert read is not None
    assert '"namespaces"' in read.args[0] and namespace_id in read.args[1:]
    insert = pool.fetchval.await_args
    assert insert is not None
    assert "customer_id" in insert.args[0]
    assert insert.args[-1] == customer


async def test_an_ensured_all_grant_records_no_customer_and_reads_no_namespace() -> None:
    """Flips if an 'all' grant records a customer, or the ensure reads a namespace it does not name."""
    pool = AsyncMock()
    pool.fetchrow.return_value = None
    pool.fetchval.side_effect = lambda sql, *args: args[1]
    collection = make_collection(RoleAssignmentCollection, l3_pool=pool)
    collection.invalidate_cache_many = AsyncMock()
    pool.fetch.return_value = []  # the grant's natural-key read finds none held
    await collection.ensure_group_role_assignment(group_id=uuid7(), role_id=uuid7(), scope_type="all", scope_id=None)
    assert not [call for call in pool.fetch.await_args_list if '"namespaces"' in call.args[0]]
    insert = pool.fetchval.await_args
    assert insert is not None
    assert insert.args[-1] is None
