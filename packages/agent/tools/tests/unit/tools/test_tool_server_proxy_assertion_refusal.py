"""the serving pod answers a call that did not come through the registry ``TOOL_PROXY_ASSERTION_UNVERIFIED``.

The registry signs every call it forwards with an assertion binding the verified caller, the call
body, a single-use nonce and the target pod. The pod checks it after the forwarded identity has
verified, so a refusal here is not an identity refusal: the identity is good, and the call could
not show it came through the registry for this body and this pod. Until this contract the pod
answered it with no code and a message naming the exception, which the hub's faces could only
render as their unnamed-failure fallback.

Every refusal of that gate -- no assertion, a spliced body, a replayed nonce, an assertion for
another pod or under a key the pod does not hold, a pod with no replay guard -- answers the one
code and the one message, and the specific reason stays in the pod's WARNING log.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any
from uuid import uuid4

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import SecretStr

from threetears.agent.tools.server import CallResponse, ToolServer
from threetears.core.security import (
    IDENTITY_REFUSED,
    TOOL_PROXY_ASSERTION_UNVERIFIED,
    TOOL_PROXY_ASSERTION_UNVERIFIED_MESSAGE,
    ProxyAssertionSigner,
    canonical_call_hash,
)
from threetears.core.testing.replay_guard import FakeReplayGuard

from packages.agent.tools.tests.unit.tools.pod_auth import (
    RecordingNatsClient,
    ScopeRecordingTool,
    deliver_call,
    jwks_provider,
    recording_tool_server,
    signed_call_payload,
)

_POD_ID = "assertion-pod"
_LOGGER = "threetears.agent.tools.server"
_LOG_LINE = "pod proxy-assertion verification failed; rejecting"


def _server(*, with_guard: bool = True) -> tuple[ToolServer, ScopeRecordingTool, RecordingNatsClient]:
    """a pod serving one recording tool, answering on a recording client.

    :param with_guard: whether the pod holds an assertion replay guard
    :ptype with_guard: bool
    :return: the server, its tool and the client it answers on
    :rtype: tuple[ToolServer, ScopeRecordingTool, RecordingNatsClient]
    """
    kwargs: dict[str, Any] = {"pod_id": _POD_ID, "jwks_provider": jwks_provider}
    if with_guard:
        kwargs["assertion_replay_guard"] = FakeReplayGuard()
    server, rec = recording_tool_server(**kwargs)
    tool = ScopeRecordingTool()
    server.register(tool)
    return server, tool, rec


def _without_assertion() -> dict[str, Any]:
    payload = signed_call_payload(pod_id=_POD_ID)
    del payload["proxy_assertion"]
    return payload


def _with_a_spliced_body() -> dict[str, Any]:
    payload = signed_call_payload(pod_id=_POD_ID, arguments={"text": "signed"})
    payload["arguments"] = {"text": "spliced"}
    return payload


def _for_another_pod() -> dict[str, Any]:
    return signed_call_payload(pod_id="some-other-pod")


def _under_a_key_the_pod_does_not_hold() -> dict[str, Any]:
    payload = signed_call_payload(pod_id=_POD_ID)
    seed = base64.urlsafe_b64encode(Ed25519PrivateKey.generate().private_bytes_raw()).decode("ascii")
    stranger = ProxyAssertionSigner.from_secret(SecretStr(seed))
    context = payload["context"]
    payload["proxy_assertion"] = stranger.mint(
        pod_id=_POD_ID,
        agent_id=str(uuid4()),
        customer_id=str(uuid4()),
        body_hash=canonical_call_hash(payload["tool_name"], payload["arguments"], context["correlation_id"]),
        nonce=str(uuid4()),
        now=int(time.time()),
    )
    return payload


#: every single-delivery way the gate refuses.
_REFUSALS: dict[str, Any] = {
    "no assertion": _without_assertion,
    "spliced body": _with_a_spliced_body,
    "assertion for another pod": _for_another_pod,
    "assertion under a key the pod does not hold": _under_a_key_the_pod_does_not_hold,
}


def _replies(rec: RecordingNatsClient) -> list[CallResponse]:
    """every answer the pod gave, typed.

    :param rec: the client the pod answered on
    :ptype rec: RecordingNatsClient
    :return: the answers in order
    :rtype: list[CallResponse]
    """
    replies = [reply for _subject, reply in rec.replies]
    assert all(isinstance(reply, CallResponse) for reply in replies)
    return replies


def _assert_the_refusal(reply: CallResponse) -> None:
    """the one code and the one message, and nothing else.

    :param reply: the pod's answer
    :ptype reply: CallResponse
    :return: nothing
    :rtype: None
    """
    assert reply.success is False
    assert reply.error_code == TOOL_PROXY_ASSERTION_UNVERIFIED
    assert reply.error == TOOL_PROXY_ASSERTION_UNVERIFIED_MESSAGE


class TestEveryAssertionRefusalAnswersTheOneCode:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("shape", sorted(_REFUSALS))
    async def test_the_refusal_is_the_one_code_and_message(self, shape: str) -> None:
        server, tool, rec = _server()

        await deliver_call(server, _REFUSALS[shape](), pod_id=_POD_ID)

        replies = _replies(rec)
        assert len(replies) == 1  # answered once
        _assert_the_refusal(replies[0])
        assert tool.scopes == []  # refused before the tool ran

    @pytest.mark.asyncio
    async def test_a_replayed_assertion_is_refused_the_same_way(self) -> None:
        server, tool, rec = _server()
        payload = signed_call_payload(pod_id=_POD_ID)

        await deliver_call(server, payload, pod_id=_POD_ID)
        await deliver_call(server, payload, pod_id=_POD_ID)

        first, second = _replies(rec)
        assert first.success is True
        _assert_the_refusal(second)
        assert len(tool.scopes) == 1  # the replay never ran

    @pytest.mark.asyncio
    async def test_a_pod_with_no_replay_guard_refuses_the_same_way(self) -> None:
        server, tool, rec = _server(with_guard=False)

        await deliver_call(server, signed_call_payload(pod_id=_POD_ID), pod_id=_POD_ID)

        (reply,) = _replies(rec)
        _assert_the_refusal(reply)
        assert tool.scopes == []

    @pytest.mark.asyncio
    async def test_it_is_not_an_identity_refusal(self) -> None:
        """the identity verified; the code says what failed after it, not that the identity did."""
        server, _tool, rec = _server()

        await deliver_call(server, _without_assertion(), pod_id=_POD_ID)

        (reply,) = _replies(rec)
        assert reply.error_code != IDENTITY_REFUSED


class TestTheReasonStaysInThePodsLog:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("shape", sorted(_REFUSALS))
    async def test_the_log_names_the_check_and_the_reply_does_not(
        self, shape: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        server, _tool, rec = _server()

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            await deliver_call(server, _REFUSALS[shape](), pod_id=_POD_ID)

        records = [r for r in caplog.records if r.getMessage() == _LOG_LINE]
        assert len(records) == 1, f"expected one {_LOG_LINE!r} line, got {[r.getMessage() for r in caplog.records]}"
        extra = getattr(records[0], "extra_data", None)
        assert extra is not None
        assert extra["detail"], "the server-side line must say which check refused"
        assert extra["tool_name"] == "test.stub"
        (reply,) = _replies(rec)
        assert reply.error is not None
        assert extra["detail"] not in reply.error
        assert extra["reason"] not in reply.error
