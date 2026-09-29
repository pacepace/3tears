"""``LocalMemoryNamespaceProvisioner``: materialize a memory namespace where there is no hub to ask.

The authorizer asks its ``namespace_provisioner`` for a memory namespace it cannot find. Hub deployments
answer through ``HubMemoryNamespaceProvisioner`` (NATS request/reply to the hub); a deployment that owns
its own control plane had nothing to plug in, so it wrote the row with raw SQL and left
``owner_namespace`` NULL (no agent owned it). This provisioner writes the SAME row the hub writes --
every field from the same public helpers -- through ``NamespaceCollection.ensure_namespace``, and checks
the row it resolved is the pair it was asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.memory import (
    MEMORY_NAMESPACE_TYPE,
    LocalMemoryNamespaceProvisioner,
    MemoryNamespaceUnavailableError,
    memory_namespace_id,
    memory_namespace_name,
    memory_namespace_schema_name,
)
from threetears.core.namespaces import build_agent_namespace_name


@dataclass
class _Row:
    id: UUID
    customer_id: UUID
    owner_agent_id: UUID
    namespace_type: str
    owner_namespace: str | None
    name: str = "memories.unnamed"
    schema_name: str | None = None


# parity-exempt: a stand-in for the two NamespaceCollection reads/writes the local provisioner makes
@dataclass
class _FakeNamespaces:
    rows: dict[tuple[str, UUID, UUID], _Row] = field(default_factory=dict)
    ensured: list[dict[str, Any]] = field(default_factory=list)
    lose_the_write: bool = False
    fail_with: Exception | None = None

    async def get_by_owner_and_customer(
        self, *, namespace_type: str, owner_agent_id: UUID | None, customer_id: UUID | None
    ) -> _Row | None:
        assert owner_agent_id is not None and customer_id is not None
        return self.rows.get((namespace_type, owner_agent_id, customer_id))

    async def ensure_namespace(self, **fields: Any) -> _Row:
        if self.fail_with is not None:
            raise self.fail_with
        self.ensured.append(fields)
        # the hub's table is UNIQUE on name and on schema_name, and ensure_namespace refuses a
        # write one of them absorbed; a fake that let two rows share either would hide exactly
        # the collision a name rule can cause.
        for taken in self.rows.values():
            if taken.id != fields["namespace_id"] and (
                taken.name == fields["name"] or taken.schema_name == fields["schema_name"]
            ):
                raise ValueError(f"namespace {fields['namespace_id']} was not written: a unique value is taken")
        row = _Row(
            id=fields["namespace_id"],
            customer_id=fields["customer_id"],
            owner_agent_id=fields["owner_agent_id"],
            namespace_type=fields["namespace_type"],
            owner_namespace=fields["owner_namespace"],
            name=fields["name"],
            schema_name=fields["schema_name"],
        )
        if not self.lose_the_write:
            self.rows[(row.namespace_type, row.owner_agent_id, row.customer_id)] = row
        return row


def _provisioner(rows: _FakeNamespaces) -> LocalMemoryNamespaceProvisioner:
    return LocalMemoryNamespaceProvisioner(rows)  # type: ignore[arg-type]  # a structural stand-in, deliberately partial


async def test_it_writes_the_row_the_hub_writes() -> None:
    agent, customer = uuid4(), uuid4()
    rows = _FakeNamespaces()
    ref = await _provisioner(rows).ensure(agent_id=agent, customer_id=customer)
    [written] = rows.ensured
    assert written == {
        "namespace_id": memory_namespace_id(agent, customer),
        "name": memory_namespace_name(agent, customer),
        "namespace_type": MEMORY_NAMESPACE_TYPE,
        "owner_agent_id": agent,
        "customer_id": customer,
        "owner_namespace": build_agent_namespace_name(agent),
        "schema_name": memory_namespace_schema_name(agent, customer),
        "metadata": {},
    }
    assert (ref.id, ref.owner_agent_id, ref.customer_id) == (memory_namespace_id(agent, customer), agent, customer)
    assert ref.owner_namespace == build_agent_namespace_name(agent), "an agent owns the row it provisioned"


async def test_an_existing_row_for_the_pair_is_found_not_duplicated() -> None:
    """Like the hub: resolved by (type, owner, customer) first, so a row under another id is reused."""
    agent, customer = uuid4(), uuid4()
    existing = _Row(uuid4(), customer, agent, MEMORY_NAMESPACE_TYPE, build_agent_namespace_name(agent))
    rows = _FakeNamespaces(rows={(MEMORY_NAMESPACE_TYPE, agent, customer): existing})
    ref = await _provisioner(rows).ensure(agent_id=agent, customer_id=customer)
    assert ref.id == existing.id
    assert rows.ensured == []


async def test_a_write_it_cannot_read_back_is_refused() -> None:
    rows = _FakeNamespaces(lose_the_write=True)
    with pytest.raises(MemoryNamespaceUnavailableError):
        await _provisioner(rows).ensure(agent_id=uuid4(), customer_id=uuid4())


async def test_a_failed_write_is_unavailable_not_a_crash() -> None:
    rows = _FakeNamespaces(fail_with=ValueError("already exists as something else"))
    with pytest.raises(MemoryNamespaceUnavailableError, match="already exists"):
        await _provisioner(rows).ensure(agent_id=uuid4(), customer_id=uuid4())


async def test_a_resolved_row_for_another_pair_is_refused() -> None:
    agent, customer = uuid4(), uuid4()
    wrong = _Row(uuid4(), uuid4(), agent, MEMORY_NAMESPACE_TYPE, None)
    rows = _FakeNamespaces(rows={(MEMORY_NAMESPACE_TYPE, agent, customer): wrong})
    with pytest.raises(MemoryNamespaceUnavailableError, match="asked"):
        await _provisioner(rows).ensure(agent_id=agent, customer_id=customer)


#: two agents a uuid7 generator could mint in one millisecond for one customer: the leading
#: 48 bits (the timestamp) are identical and only the random tail differs.
_AGENT_A = UUID("019470a8-b5c3-7def-8123-456789abcdef")
_AGENT_B = UUID("019470a8-b5c3-7a01-9fed-cba987654321")
_CUSTOMER = UUID("019470a8-b5c4-7000-8000-000000000001")


async def test_two_agents_minted_in_one_millisecond_both_get_a_memory_namespace() -> None:
    """A cluster apply creates several agents at once; each must get its own row.

    A name cut from the head of a uuid7 is its timestamp, so both agents asked for one name and
    the second ensure was refused by the unique index.
    """
    assert _AGENT_A.int >> 80 == _AGENT_B.int >> 80, "the fixture must share the uuid7 timestamp"
    rows = _FakeNamespaces()
    provisioner = _provisioner(rows)

    first = await provisioner.ensure(agent_id=_AGENT_A, customer_id=_CUSTOMER)
    second = await provisioner.ensure(agent_id=_AGENT_B, customer_id=_CUSTOMER)

    assert (first.owner_agent_id, second.owner_agent_id) == (_AGENT_A, _AGENT_B)
    assert len({row.name for row in rows.rows.values()}) == 2


async def test_a_row_named_by_the_earlier_rule_is_found_not_refused() -> None:
    """Rows written before the name rule changed keep their name and are resolved by their pair.

    The provisioner reaches ``ensure_namespace`` -- which refuses any row that disagrees with the
    fields it is handed -- only for a pair that has no row, so an existing row is never compared
    against the new name at all.
    """
    legacy_name = "memories.019470a8.019470a8"
    existing = _Row(
        memory_namespace_id(_AGENT_A, _CUSTOMER),
        _CUSTOMER,
        _AGENT_A,
        MEMORY_NAMESPACE_TYPE,
        build_agent_namespace_name(_AGENT_A),
        name=legacy_name,
        schema_name="memory__019470a8__019470a8",
    )
    rows = _FakeNamespaces(rows={(MEMORY_NAMESPACE_TYPE, _AGENT_A, _CUSTOMER): existing})

    ref = await _provisioner(rows).ensure(agent_id=_AGENT_A, customer_id=_CUSTOMER)

    assert ref.id == existing.id
    assert ref.name == legacy_name, "the row is judged by the name it stores, not a recomputed one"
    assert rows.ensured == []
