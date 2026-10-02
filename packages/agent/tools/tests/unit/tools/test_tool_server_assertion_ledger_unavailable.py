"""a tool pod whose proxy-assertion replay ledger cannot be reached answers, and does not time out.

The pod records every proxy assertion's nonce in a shared KV ledger before it runs the tool. The
ledger fails closed: when its bucket cannot be reached it raises rather than answering "fresh". That
exception used to escape the dispatch with no reply published, so the registry waited its whole
budget and the caller was told ``TOOL_TIMEOUT`` -- about a pod that had refused in milliseconds.

The pod now answers the same outage the registry answers for its own ledger,
``TOOL_POP_LEDGER_UNAVAILABLE``, with the one shared message. The ledger's error, and the fix,
go to the pod's ERROR log. A gate a pod reads its JWKS through is the same class of bug: a provider
that raises during the assertion check is a refusal with a reply, never a silent dispatch death.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from threetears.agent.tools.server import CallResponse, ToolServer
from threetears.core.security import (
    TOOL_POP_LEDGER_UNAVAILABLE,
    TOOL_POP_LEDGER_UNAVAILABLE_MESSAGE,
    TOOL_PROXY_ASSERTION_UNVERIFIED,
    TOOL_PROXY_ASSERTION_UNVERIFIED_MESSAGE,
)
from threetears.core.testing.replay_guard import FakeReplayGuard
from threetears.nats import KvError

from packages.agent.tools.tests.unit.tools.pod_auth import (
    RecordingNatsClient,
    ScopeRecordingTool,
    deliver_call,
    jwks_provider,
    recording_tool_server,
    signed_call_payload,
)

_POD_ID = "ledger-pod"
_LOGGER = "threetears.agent.tools.server"
_LEDGER_ERROR = "KV create failed: bucket=aibots_proxy_assertion_nonces key=k: nats: timeout"


def _server(guard: FakeReplayGuard, **kwargs: Any) -> tuple[ToolServer, ScopeRecordingTool, RecordingNatsClient]:
    """a pod serving one recording tool over ``guard``, answering on a recording client.

    :param guard: the pod's assertion replay guard
    :ptype guard: FakeReplayGuard
    :param kwargs: further :class:`ToolServer` arguments; ``jwks_provider`` defaults to the shared one
    :ptype kwargs: Any
    :return: the server, its tool and the client it answers on
    :rtype: tuple[ToolServer, ScopeRecordingTool, RecordingNatsClient]
    """
    kwargs.setdefault("jwks_provider", jwks_provider)
    server, rec = recording_tool_server(pod_id=_POD_ID, assertion_replay_guard=guard, **kwargs)
    tool = ScopeRecordingTool()
    server.register(tool)
    return server, tool, rec


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


class TestAnUnreachableLedgerIsAnswered:
    @pytest.mark.asyncio
    async def test_exactly_one_reply_with_the_ledger_code_and_the_tool_never_runs(self) -> None:
        guard = FakeReplayGuard(record_error=KvError(_LEDGER_ERROR))
        server, tool, rec = _server(guard)

        await deliver_call(server, signed_call_payload(pod_id=_POD_ID), pod_id=_POD_ID)

        replies = _replies(rec)
        assert len(replies) == 1  # answered, once -- not left to the caller's timeout
        (reply,) = replies
        assert reply.success is False
        assert reply.error_code == TOOL_POP_LEDGER_UNAVAILABLE
        assert reply.error == TOOL_POP_LEDGER_UNAVAILABLE_MESSAGE
        assert len(guard.seen) == 1  # the ledger was asked, and failed
        assert tool.scopes == []  # refused before the tool ran

    @pytest.mark.asyncio
    async def test_the_pods_error_log_names_the_ledger_failure_and_the_reply_does_not(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        guard = FakeReplayGuard(record_error=KvError(_LEDGER_ERROR), bucket_name="proxy_assertion_nonces")
        server, _tool, rec = _server(guard)

        with caplog.at_level(logging.ERROR, logger=_LOGGER):
            await deliver_call(server, signed_call_payload(pod_id=_POD_ID), pod_id=_POD_ID)

        errors = [r for r in caplog.records if r.levelno == logging.ERROR and r.name == _LOGGER]
        assert len(errors) == 1, [r.getMessage() for r in caplog.records]
        extra = getattr(errors[0], "extra_data", None)
        assert extra is not None
        assert extra["error_code"] == TOOL_POP_LEDGER_UNAVAILABLE
        assert extra["bucket"] == "proxy_assertion_nonces"
        assert extra["error_type"] == "KvError"
        assert extra["error"] == _LEDGER_ERROR
        assert extra["tool_name"] == "test.stub"
        # the line says what to do about it, not only what happened
        assert "NATS" in errors[0].getMessage()
        (reply,) = _replies(rec)
        assert reply.error is not None
        assert _LEDGER_ERROR not in reply.error
        assert "KvError" not in reply.error


class TestAJwksProviderFailingAtTheAssertionGateIsAnswered:
    @pytest.mark.asyncio
    async def test_a_provider_that_fails_after_the_identity_check_refuses_with_one_reply(self) -> None:
        """the identity check reads the JWKS first; a provider that fails on the next read -- a cache
        whose backing store dropped between the two -- must still produce exactly one answer."""
        reads: list[int] = []

        def flaky_provider() -> dict[str, Any]:
            reads.append(1)
            if len(reads) > 1:
                raise ConnectionError("jwks cache backend went away")
            return jwks_provider()

        guard = FakeReplayGuard()
        server, tool, rec = _server(guard, jwks_provider=flaky_provider)

        await deliver_call(server, signed_call_payload(pod_id=_POD_ID), pod_id=_POD_ID)

        replies = _replies(rec)
        assert len(replies) == 1
        (reply,) = replies
        assert reply.error_code == TOOL_PROXY_ASSERTION_UNVERIFIED
        assert reply.error == TOOL_PROXY_ASSERTION_UNVERIFIED_MESSAGE
        assert guard.seen == []  # never reached the ledger
        assert tool.scopes == []
