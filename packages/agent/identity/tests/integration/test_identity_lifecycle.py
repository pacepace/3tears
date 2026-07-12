"""Live integration: the migration executes + the full self-evolution lifecycle.

Against a real Postgres container: applies migration v001 (validating the
raw DDL, enums, and the partial-UNIQUE index actually execute), then drives
propose → consent → propose → consent → rollback → tier-2 auto-apply,
asserting the one-active invariant, the linear chain, and the reads over
real rows -- the coverage the DSL/mock unit tests cannot give.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from threetears.agent.acl import AclCache

from threetears.agent.identity import lifecycle
from threetears.agent.identity.authorize import IdentityAuthorizerDependencies

from .conftest import (
    InMemoryNatsBus,
    apply_migrations,
    build_collection,
    make_pool,
)

pytestmark = pytest.mark.integration

_AGENT = uuid.UUID("00000000-0000-0000-0000-000000000001")
_USER = uuid.UUID("00000000-0000-0000-0000-00000000000b")
_CUST = _USER
_BLOCK = "personality"


def _authorizer() -> IdentityAuthorizerDependencies:
    """owner-only authorizer with empty ACL loaders (short-circuit path)."""

    class _EmptyMembershipLoader:
        async def load_for_user(self, user_id: uuid.UUID) -> tuple[Any, ...]:
            _ = user_id
            return ()

        async def load_for_agent(self, agent_id: uuid.UUID) -> tuple[Any, ...]:
            _ = agent_id
            return ()

    class _EmptyGrantLoader:
        async def load_assignments_for_groups(self, group_ids: Any, namespace: Any) -> tuple[Any, ...]:
            _ = group_ids, namespace
            return ()

        async def load_roles(self, role_ids: Any) -> dict[Any, Any]:
            _ = role_ids
            return {}

        async def load_groups(self, group_ids: Any) -> dict[Any, Any]:
            _ = group_ids
            return {}

    return IdentityAuthorizerDependencies(
        acl_cache=AclCache(
            membership_loader=_EmptyMembershipLoader(),
            grant_loader=_EmptyGrantLoader(),
        ),
    )


async def _stack(pg_schema: tuple[str, str]) -> tuple[Any, Any]:
    url, schema = pg_schema
    await apply_migrations(url, schema)  # validates the raw v001 DDL executes
    pool = await make_pool(url, schema)
    coll, _l1 = build_collection(pool, InMemoryNatsBus())
    return coll, pool


def _kw(**extra: Any) -> dict[str, Any]:
    base = dict(agent_id=_AGENT, customer_id=_CUST, user_id=_USER, caller_agent_id=_AGENT)
    base.update(extra)
    return base


async def _count_active(pool: Any, block: str) -> int:
    return await pool.fetchval(
        "SELECT count(*) FROM identity_versions WHERE block_key = $1 AND status = 'active'",
        block,
    )


async def test_full_lifecycle(pg_schema: tuple[str, str]) -> None:
    coll, pool = await _stack(pg_schema)
    authz = _authorizer()
    try:
        # tier-1 propose -> proposed; pending queue shows it; no active yet
        v1 = await lifecycle.propose(
            coll, authz, block_key=_BLOCK, content="v1", rationale="seed",
            proposer_agent_id=_AGENT, **_kw(),
        )
        assert v1 is not None and v1.status == "proposed"
        pending = await coll.find_pending(agent_id=_AGENT, user_id=_USER)
        assert [p.version_id for p in pending] == [v1.version_id]
        assert await coll.resolve_active(
            agent_id=_AGENT, customer_id=_CUST, user_id=_USER, block_key=_BLOCK
        ) is None

        # consent -> active
        applied = await lifecycle.consent(
            coll, authz, version_id=v1.version_id, consenter_user_id=_USER, **_kw()
        )
        assert applied is not None and applied.status == "active"
        active = await coll.resolve_active(
            agent_id=_AGENT, customer_id=_CUST, user_id=_USER, block_key=_BLOCK
        )
        assert active is not None and active.version_id == v1.version_id and active.content == "v1"
        assert await _count_active(pool, _BLOCK) == 1

        # propose v2 -> consent -> v1 superseded, v2 active, still exactly one active
        v2 = await lifecycle.propose(
            coll, authz, block_key=_BLOCK, content="v2", rationale="sharper",
            proposer_agent_id=_AGENT, **_kw(),
        )
        assert v2 is not None and v2.parent_version_id == v1.version_id  # linear chain
        await lifecycle.consent(coll, authz, version_id=v2.version_id, consenter_user_id=_USER, **_kw())
        active2 = await coll.resolve_active(
            agent_id=_AGENT, customer_id=_CUST, user_id=_USER, block_key=_BLOCK
        )
        assert active2 is not None and active2.version_id == v2.version_id
        assert await _count_active(pool, _BLOCK) == 1

        # rollback to v1 -> a NEW clone becomes active with v1's content
        clone = await lifecycle.rollback(
            coll, authz, target_version_id=v1.version_id, consenter_user_id=_USER, **_kw()
        )
        assert clone is not None and clone.content == "v1" and clone.status == "active"
        assert clone.version_id not in (v1.version_id, v2.version_id)
        active3 = await coll.resolve_active(
            agent_id=_AGENT, customer_id=_CUST, user_id=_USER, block_key=_BLOCK
        )
        assert active3 is not None and active3.version_id == clone.version_id
        assert await _count_active(pool, _BLOCK) == 1

        # history: all four versions for the block are retained
        history = await coll.find_versions_for_block(
            agent_id=_AGENT, customer_id=_CUST, user_id=_USER, block_key=_BLOCK
        )
        assert len(history) == 3  # v1, v2, clone (rollback of v1 content)

        # tier-2 block auto-applies with no consent step
        t2 = await lifecycle.propose(
            coll, authz, block_key="self_improvement", content="note", rationale="r",
            proposer_agent_id=_AGENT, **_kw(),
        )
        assert t2 is not None and t2.status == "active"
        assert await _count_active(pool, "self_improvement") == 1
    finally:
        await pool.close()


async def test_reject_leaves_active_untouched(pg_schema: tuple[str, str]) -> None:
    coll, pool = await _stack(pg_schema)
    authz = _authorizer()
    try:
        proposed = await lifecycle.propose(
            coll, authz, block_key=_BLOCK, content="maybe", rationale="r",
            proposer_agent_id=_AGENT, **_kw(),
        )
        assert proposed is not None
        rejected = await lifecycle.reject(coll, authz, version_id=proposed.version_id, **_kw())
        assert rejected is not None and rejected.status == "rejected"
        assert await coll.resolve_active(
            agent_id=_AGENT, customer_id=_CUST, user_id=_USER, block_key=_BLOCK
        ) is None
        assert await coll.find_pending(agent_id=_AGENT, user_id=_USER) == []
    finally:
        await pool.close()
