"""the serving pod answers a forwarded identity that does not verify ``IDENTITY_REFUSED``.

The registry answers this condition with :data:`~threetears.core.security.IDENTITY_REFUSED` and
one undiscriminating message, as every hub door does. The pod re-verifies the same identity at its
own door, and until this contract it answered with no code and a message naming which check had
failed: an unnamed refusal that the hub's HTTP face could only render as its 502 fallback, and a
message telling a caller which check to work around.

So every refusal of the pod's identity gate -- an absent, expired, unknown-key or malformed
handshake token, a user assertion that does not verify or bind, an assertion on a tool pod's
token -- answers the one code and the one message, and the specific reason stays in the pod's own
WARNING log, beside the tool name. The proxy-assertion gate that follows answers a DIFFERENT
condition (the call's provenance through the registry, not the identity it forwards) and is
pinned here as not answering this code.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

import pytest

from threetears.agent.tools.server import CallResponse, ToolServer
from threetears.core.security import IDENTITY_REFUSED, IDENTITY_REFUSED_MESSAGE, PLATFORM_CUSTOMER_SENTINEL
from threetears.core.testing.replay_guard import FakeReplayGuard

from packages.agent.tools.tests.unit.tools._pod_auth import (
    RecordingNatsClient,
    ScopeRecordingTool,
    deliver_call,
    jwks_provider,
    mint_user_assertion,
    recording_tool_server,
    signed_call_payload,
)

_POD_ID = "refusal-pod"
_LOGGER = "threetears.agent.tools.server"


def _server(*, jwks: Any = jwks_provider) -> tuple[ToolServer, ScopeRecordingTool, RecordingNatsClient]:
    """a pod serving one recording tool, answering on a recording client.

    :param jwks: the JWKS provider the pod verifies against
    :ptype jwks: Any
    :return: the server, its tool and the client it answers on
    :rtype: tuple[ToolServer, ScopeRecordingTool, RecordingNatsClient]
    """
    server, rec = recording_tool_server(pod_id=_POD_ID, jwks_provider=jwks, assertion_replay_guard=FakeReplayGuard())
    tool = ScopeRecordingTool()
    server.register(tool)
    return server, tool, rec


def _without_identity_token() -> dict[str, Any]:
    payload = signed_call_payload(pod_id=_POD_ID)
    del payload["context"]["identity_token"]
    return payload


def _with_expired_identity_token() -> dict[str, Any]:
    payload = signed_call_payload(pod_id=_POD_ID)
    payload["context"]["identity_token"] = mint_user_assertion(
        sub=uuid4(), customer_id=uuid4(), user_id=None, exp_delta=-3600
    )
    return payload


def _with_a_customer_claim_that_is_no_uuid() -> dict[str, Any]:
    payload = signed_call_payload(pod_id=_POD_ID)
    payload["context"]["identity_token"] = mint_user_assertion(sub=uuid4(), customer_id="not-a-uuid", user_id=None)
    return payload


def _with_a_user_assertion_for_another_agent() -> dict[str, Any]:
    agent_id, customer_id, conversation_id = uuid4(), uuid4(), uuid4()
    return signed_call_payload(
        pod_id=_POD_ID,
        agent_id=agent_id,
        customer_id=customer_id,
        conversation_id=conversation_id,
        user_assertion=mint_user_assertion(
            sub=uuid4(), customer_id=customer_id, user_id=uuid4(), conversation_id=conversation_id
        ),
    )


def _with_a_user_assertion_for_another_conversation() -> dict[str, Any]:
    agent_id, customer_id = uuid4(), uuid4()
    return signed_call_payload(
        pod_id=_POD_ID,
        agent_id=agent_id,
        customer_id=customer_id,
        conversation_id=uuid4(),
        user_assertion=mint_user_assertion(
            sub=agent_id, customer_id=customer_id, user_id=uuid4(), conversation_id=uuid4()
        ),
    )


def _with_a_user_assertion_on_a_tool_pod_token() -> dict[str, Any]:
    return signed_call_payload(
        pod_id=_POD_ID,
        agent_id=uuid4(),
        customer_id=PLATFORM_CUSTOMER_SENTINEL,
        conversation_id=uuid4(),
        user_id=uuid4(),
    )


#: every way the pod's identity gate refuses, and the server-side log line that names it.
_REFUSALS: dict[str, tuple[Any, str]] = {
    "handshake token absent": (
        _without_identity_token,
        "pod identity verification failed; rejecting call",
    ),
    "handshake token expired": (
        _with_expired_identity_token,
        "pod identity verification failed; rejecting call",
    ),
    "handshake customer claim neither a uuid nor the sentinel": (
        _with_a_customer_claim_that_is_no_uuid,
        "pod identity verification failed; rejecting call",
    ),
    "user assertion bound to another agent": (
        _with_a_user_assertion_for_another_agent,
        "pod user-assertion verification failed; rejecting call",
    ),
    "user assertion minted for another conversation": (
        _with_a_user_assertion_for_another_conversation,
        "pod user-assertion verification failed; rejecting call",
    ),
    "user assertion on a tool pod token": (
        _with_a_user_assertion_on_a_tool_pod_token,
        "pod user-assertion presented on a tool pod token; rejecting call",
    ),
}


def _only_reply(rec: RecordingNatsClient) -> CallResponse:
    """the single answer the pod gave.

    :param rec: the client the pod answered on
    :ptype rec: RecordingNatsClient
    :return: the answer
    :rtype: CallResponse
    """
    assert len(rec.replies) == 1, f"the pod answered {len(rec.replies)} times; a refusal is answered once"
    reply = rec.replies[0][1]
    assert isinstance(reply, CallResponse)
    return reply


class TestEveryIdentityRefusalAnswersTheOneCode:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("shape", sorted(_REFUSALS))
    async def test_the_refusal_is_identity_refused_with_the_one_message(self, shape: str) -> None:
        build, _log_line = _REFUSALS[shape]
        server, tool, rec = _server()

        await deliver_call(server, build(), pod_id=_POD_ID)

        reply = _only_reply(rec)
        assert reply.success is False
        assert reply.error_code == IDENTITY_REFUSED
        assert reply.error == IDENTITY_REFUSED_MESSAGE
        assert tool.scopes == []  # refused before the tool ran

    @pytest.mark.asyncio
    async def test_a_key_the_pod_does_not_hold_is_refused_the_same_way(self) -> None:
        """the stale-JWKS shape (no refresh wired), which logs differently, answers identically."""
        hub_less = {"keys": [k for k in jwks_provider()["keys"] if k.get("kid") != "kid-1"]}
        server, tool, rec = _server(jwks=lambda: hub_less)

        await deliver_call(server, signed_call_payload(pod_id=_POD_ID), pod_id=_POD_ID)

        reply = _only_reply(rec)
        assert (reply.error_code, reply.error) == (IDENTITY_REFUSED, IDENTITY_REFUSED_MESSAGE)
        assert tool.scopes == []


class TestTheReasonStaysInThePodsLog:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("shape", sorted(_REFUSALS))
    async def test_the_log_names_the_check_and_the_reply_does_not(
        self, shape: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        build, log_line = _REFUSALS[shape]
        server, _tool, rec = _server()

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            await deliver_call(server, build(), pod_id=_POD_ID)

        records = [r for r in caplog.records if r.getMessage() == log_line]
        assert len(records) == 1, f"expected one {log_line!r} line, got {[r.getMessage() for r in caplog.records]}"
        extra = getattr(records[0], "extra_data", None)
        assert extra is not None
        detail = extra["detail"]
        assert detail, "the server-side line must say which check refused"
        assert extra["tool_name"] == "test.stub"
        reply = _only_reply(rec)
        assert reply.error is not None
        assert detail not in reply.error
        assert extra["reason"] not in reply.error


class TestTheProxyAssertionGateIsADifferentCondition:
    @pytest.mark.asyncio
    async def test_a_verified_identity_without_the_registrys_assertion_is_not_identity_refused(self) -> None:
        """the identity verified; what failed is the call's proof that it came through the registry."""
        server, tool, rec = _server()
        payload = signed_call_payload(pod_id=_POD_ID)
        del payload["proxy_assertion"]

        await deliver_call(server, payload, pod_id=_POD_ID)

        reply = _only_reply(rec)
        assert reply.success is False
        assert reply.error_code != IDENTITY_REFUSED
        assert reply.error != IDENTITY_REFUSED_MESSAGE
        assert tool.scopes == []
