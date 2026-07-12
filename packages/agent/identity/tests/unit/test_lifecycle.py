"""Unit coverage for the identity lifecycle control flow (mocked collection).

The real DB behaviour (the one-active partial-unique index, the migration,
resolve_active over real rows) is the integration suite's job. This file
pins the branching logic with a mocked collection + a patched authorizer:
tier-1 vs tier-2 propose, dedup no-op, consent apply, ownership gates,
reject, rollback-clone, and which FrameworkEvent each op emits.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from threetears.agent.identity import lifecycle
from threetears.agent.identity.entities import IdentityVersionEntity
from threetears.agent.identity.events import (
    IdentityAppliedEvent,
    IdentityConsentedEvent,
    IdentityProposedEvent,
    IdentityRolledBackEvent,
)
from threetears.agent.identity.lifecycle import content_hash

pytestmark = pytest.mark.asyncio

_AGENT = UUID("00000000-0000-0000-0000-000000000001")
_USER = UUID("00000000-0000-0000-0000-00000000000b")
_OTHER = UUID("00000000-0000-0000-0000-00000000000c")


def _version(**overrides: object) -> IdentityVersionEntity:
    data: dict[str, object] = {
        "agent_id": _AGENT,
        "version_id": uuid4(),
        "customer_id": _USER,
        "user_id": _USER,
        "block_key": "personality",
        "content": "You are Saoirse.",
        "rationale": "seed",
        "content_hash": content_hash("You are Saoirse."),
        "parent_version_id": None,
        "status": "active",
        "proposer_agent_id": _AGENT,
        "consenter_user_id": None,
        "date_created": None,
        "date_updated": None,
    }
    data.update(overrides)
    # keep the hash consistent with the (possibly overridden) content unless
    # a test pins it explicitly
    if "content_hash" not in overrides:
        data["content_hash"] = content_hash(str(data["content"]))
    return IdentityVersionEntity(data, is_new=False)


def _collection(*, active=None, get_result=None) -> MagicMock:
    coll = MagicMock()
    coll.resolve_active = AsyncMock(return_value=active)
    coll.get = AsyncMock(return_value=get_result)
    coll.save_entity = AsyncMock()
    coll.create = MagicMock(side_effect=lambda data: IdentityVersionEntity(data, is_new=True))
    return coll


def _authorizer() -> MagicMock:
    return MagicMock()


def _kwargs(**extra):
    base = dict(
        agent_id=_AGENT,
        customer_id=_USER,
        user_id=_USER,
        caller_agent_id=_AGENT,
    )
    base.update(extra)
    return base


# authorize + event dispatch are patched at the lifecycle module boundary
_AUTHZ = "threetears.agent.identity.lifecycle.authorize_identity_access"
_DISPATCH = "threetears.agent.identity.lifecycle.dispatch_event"


class TestPropose:
    async def test_tier1_creates_proposed_awaiting_consent(self) -> None:
        coll = _collection(active=None)
        with patch(_AUTHZ, new_callable=AsyncMock), patch(_DISPATCH, new_callable=AsyncMock) as ev:
            out = await lifecycle.propose(
                coll, _authorizer(), block_key="personality",
                content="v2", rationale="sharper", proposer_agent_id=_AGENT, **_kwargs(),
            )
        assert out is not None and out.status == "proposed"
        coll.save_entity.assert_awaited_once()
        assert isinstance(ev.await_args.args[0], IdentityProposedEvent)

    async def test_tier2_auto_applies(self) -> None:
        coll = _collection(active=None)
        with patch(_AUTHZ, new_callable=AsyncMock), patch(_DISPATCH, new_callable=AsyncMock) as ev:
            out = await lifecycle.propose(
                coll, _authorizer(), block_key="self_improvement",
                content="note", rationale="r", proposer_agent_id=_AGENT, **_kwargs(),
            )
        assert out is not None and out.status == "active"
        applied = ev.await_args.args[0]
        assert isinstance(applied, IdentityAppliedEvent) and applied.auto_applied is True

    async def test_tier2_supersedes_prior_active(self) -> None:
        prior = _version(block_key="self_improvement", content="old")
        coll = _collection(active=prior)
        with patch(_AUTHZ, new_callable=AsyncMock), patch(_DISPATCH, new_callable=AsyncMock):
            await lifecycle.propose(
                coll, _authorizer(), block_key="self_improvement",
                content="new", rationale="r", proposer_agent_id=_AGENT, **_kwargs(),
            )
        # supersede prior + save new = two saves; prior flipped to superseded
        assert coll.save_entity.await_count == 2
        assert prior.status == "superseded"

    async def test_dedup_identical_content_is_noop(self) -> None:
        active = _version(content="same")
        coll = _collection(active=active)
        with patch(_AUTHZ, new_callable=AsyncMock), patch(_DISPATCH, new_callable=AsyncMock) as ev:
            out = await lifecycle.propose(
                coll, _authorizer(), block_key="personality",
                content="same", rationale="r", proposer_agent_id=_AGENT, **_kwargs(),
            )
        assert out is active
        coll.save_entity.assert_not_awaited()
        ev.assert_not_awaited()


class TestConsent:
    async def test_consent_activates_and_supersedes(self) -> None:
        proposed = _version(status="proposed")
        prior = _version(content="old")
        coll = _collection(active=prior, get_result=proposed)
        with patch(_AUTHZ, new_callable=AsyncMock), patch(_DISPATCH, new_callable=AsyncMock) as ev:
            out = await lifecycle.consent(
                coll, _authorizer(), version_id=proposed.version_id,
                consenter_user_id=_USER, **_kwargs(),
            )
        assert out is proposed and proposed.status == "active"
        assert proposed.consenter_user_id == _USER
        assert prior.status == "superseded"
        kinds = [c.args[0].__class__ for c in ev.await_args_list]
        assert IdentityConsentedEvent in kinds and IdentityAppliedEvent in kinds

    async def test_consent_foreign_version_returns_none(self) -> None:
        foreign = _version(status="proposed", user_id=_OTHER)
        coll = _collection(get_result=foreign)
        with patch(_AUTHZ, new_callable=AsyncMock), patch(_DISPATCH, new_callable=AsyncMock):
            out = await lifecycle.consent(
                coll, _authorizer(), version_id=foreign.version_id,
                consenter_user_id=_USER, **_kwargs(),
            )
        assert out is None
        coll.save_entity.assert_not_awaited()

    async def test_consent_non_proposed_returns_none(self) -> None:
        already = _version(status="active")
        coll = _collection(get_result=already)
        with patch(_AUTHZ, new_callable=AsyncMock), patch(_DISPATCH, new_callable=AsyncMock):
            out = await lifecycle.consent(
                coll, _authorizer(), version_id=already.version_id,
                consenter_user_id=_USER, **_kwargs(),
            )
        assert out is None


class TestReject:
    async def test_reject_marks_rejected(self) -> None:
        proposed = _version(status="proposed")
        coll = _collection(get_result=proposed)
        with patch(_AUTHZ, new_callable=AsyncMock), patch(_DISPATCH, new_callable=AsyncMock):
            out = await lifecycle.reject(
                coll, _authorizer(), version_id=proposed.version_id, **_kwargs(),
            )
        assert out is proposed and proposed.status == "rejected"
        coll.save_entity.assert_awaited_once()


class TestRollback:
    async def test_rollback_clones_target_content_as_new_active(self) -> None:
        target = _version(status="superseded", content="the good version")
        prior = _version(content="current")
        coll = _collection(active=prior, get_result=target)
        with patch(_AUTHZ, new_callable=AsyncMock), patch(_DISPATCH, new_callable=AsyncMock) as ev:
            out = await lifecycle.rollback(
                coll, _authorizer(), target_version_id=target.version_id,
                consenter_user_id=_USER, **_kwargs(),
            )
        assert out is not None
        assert out.content == "the good version" and out.status == "active"
        assert out.version_id != target.version_id  # a NEW version (clone)
        assert prior.status == "superseded"
        rolled = ev.await_args.args[0]
        assert isinstance(rolled, IdentityRolledBackEvent)
        assert rolled.target_version_id == str(target.version_id)

    async def test_rollback_foreign_target_returns_none(self) -> None:
        foreign = _version(user_id=_OTHER)
        coll = _collection(get_result=foreign)
        with patch(_AUTHZ, new_callable=AsyncMock), patch(_DISPATCH, new_callable=AsyncMock):
            out = await lifecycle.rollback(
                coll, _authorizer(), target_version_id=foreign.version_id,
                consenter_user_id=_USER, **_kwargs(),
            )
        assert out is None
