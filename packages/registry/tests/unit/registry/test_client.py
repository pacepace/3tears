"""unit tests for :class:`ToolCallClient`: the pod's half of the tool-call wire.

the contract these pin, against a NATS stand-in that records what crossed:

- an empty tool name or version is refused before any publish;
- an empty identity token is refused before any publish, with the code that
  names the wiring gap;
- the request that crosses is the registry's own ``ProxyCallRequest``, carrying
  the principal on the context, the token, and a proof of possession the
  registry's own verifier accepts for THIS token and THIS body;
- a refused reply raises with the registry's code; a tool's own failure raises
  with the tool's code; a transport fault raises as ``REQUEST_FAILED``;
- a successful reply is returned whole;
- the client's deadline sits above the registry's forward budget.
"""

from __future__ import annotations

import time
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid7

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from threetears.agent.tools.context_envelope import CallContext
from threetears.core.security import (
    access_token_hash,
    canonical_call_hash,
    jwk_thumbprint,
    make_pop_proof,
    verify_pop_proof,
)
from threetears.nats import set_default_namespace
from threetears.nats.errors import RequestError, RequestTimeoutError

from threetears.registry.client import (
    CALL_TIMEOUT_MARGIN_SECONDS,
    DEFAULT_CALL_TIMEOUT_SECONDS,
    ToolCallClient,
    ToolCallError,
)
from threetears.registry.config import get_call_timeout
from threetears.registry.proxy import ProxyCallRequest, ProxyCallResponse

_POD_ID = UUID("01948a00-aaaa-7000-8000-00000000f00d")
_TOKEN = "header.payload.signature"
_HOLDER_KEY = Ed25519PrivateKey.generate()


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    set_default_namespace("test")


class _Signer:
    """the SDK's ``PopSigner`` shape over a real holder key, so the proof is verifiable."""

    def __init__(self, holder_key: Ed25519PrivateKey) -> None:
        self._holder_key = holder_key
        self.calls: list[dict[str, Any]] = []

    def sign(
        self,
        *,
        access_token: str,
        tool_name: str,
        arguments: dict[str, Any],
        correlation_id: str | None,
    ) -> str:
        self.calls.append(
            {
                "access_token": access_token,
                "tool_name": tool_name,
                "arguments": arguments,
                "correlation_id": correlation_id,
            }
        )
        return make_pop_proof(
            holder_key=self._holder_key,
            access_token_hash=access_token_hash(access_token),
            body_hash=canonical_call_hash(tool_name, arguments, correlation_id),
            nonce=str(uuid7()),
            iat=int(time.time()),
        )


# parity-exempt: subset stand-in for NatsClient exposing only the typed request the client publishes through
class _RecordingNats:
    """records the typed request the client sends and answers with a canned reply."""

    def __init__(self, reply: ProxyCallResponse | Exception) -> None:
        self._reply = reply
        self.requests: list[dict[str, Any]] = []

    async def request(
        self,
        *,
        subject: Any,
        message: Any,
        response_type: Any,
        timeout: timedelta,
    ) -> Any:
        self.requests.append(
            {"subject": subject, "message": message, "response_type": response_type, "timeout": timeout}
        )
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply


def _ok_reply() -> ProxyCallResponse:
    return ProxyCallResponse(
        success=True,
        content="42",
        metadata={"rows": 1},
        context=CallContext(agent_id=_POD_ID, correlation_id=uuid7()),
    )


def _client(nats: Any, *, token: str | None = _TOKEN, signer: _Signer | None = None) -> ToolCallClient:
    return ToolCallClient(
        nats,
        principal_id=_POD_ID,
        identity_token=lambda: token,
        pop_signer=signer if signer is not None else _Signer(_HOLDER_KEY),
    )


class TestRefusedBeforeTheBus:
    @pytest.mark.asyncio
    async def test_an_empty_tool_name_is_refused_before_any_publish(self) -> None:
        nats = _RecordingNats(_ok_reply())
        with pytest.raises(ToolCallError) as excinfo:
            await _client(nats).call("", "1.0", {})
        assert excinfo.value.error_code == "INVALID_TOOL_NAME"
        assert nats.requests == []

    @pytest.mark.asyncio
    async def test_an_empty_tool_version_is_refused_before_any_publish(self) -> None:
        nats = _RecordingNats(_ok_reply())
        with pytest.raises(ToolCallError) as excinfo:
            await _client(nats).call("probe.echo", "", {})
        assert excinfo.value.error_code == "INVALID_TOOL_VERSION"
        assert nats.requests == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("token", [None, ""])
    async def test_a_missing_identity_token_is_refused_before_any_publish(self, token: str | None) -> None:
        nats = _RecordingNats(_ok_reply())
        signer = _Signer(_HOLDER_KEY)
        with pytest.raises(ToolCallError) as excinfo:
            await _client(nats, token=token, signer=signer).call("probe.echo", "1.0", {})
        assert excinfo.value.error_code == "NO_IDENTITY_TOKEN"
        assert nats.requests == []
        assert signer.calls == []  # nothing was signed for a call that was never sent


class TestTheRequestThatCrosses:
    @pytest.mark.asyncio
    async def test_the_wire_shape_is_the_registry_request_on_the_call_subject(self) -> None:
        nats = _RecordingNats(_ok_reply())
        correlation_id = uuid7()

        await _client(nats).call("probe.echo", "1.0", {"text": "hi"}, correlation_id=correlation_id)

        assert len(nats.requests) == 1
        sent = nats.requests[0]
        assert sent["subject"].path == "test.tools.call"
        assert sent["response_type"] is ProxyCallResponse
        request = sent["message"]
        assert isinstance(request, ProxyCallRequest)
        assert request.tool_name == "probe.echo"
        assert request.tool_version == "1.0"
        assert request.arguments == {"text": "hi"}
        assert request.result_subject is None  # synchronous: no durable reply is requested
        assert request.context is not None
        assert request.context.agent_id == _POD_ID
        assert request.context.correlation_id == correlation_id
        assert request.context.identity_token == _TOKEN
        assert request.context.user_id is None
        assert request.context.user_identity_token is None

    @pytest.mark.asyncio
    async def test_the_pop_binds_this_token_to_this_body_as_the_registry_recomputes_it(self) -> None:
        nats = _RecordingNats(_ok_reply())
        correlation_id = uuid7()

        await _client(nats).call("probe.echo", "1.0", {"text": "hi"}, correlation_id=correlation_id)

        request = nats.requests[0]["message"]
        assert request.pop is not None
        # the registry's own verifier, over the hash it recomputes from the deserialized request:
        # tool name, arguments as sent, correlation id as the string form of the context's id.
        verify_pop_proof(
            request.pop,
            expected_jkt=jwk_thumbprint(_HOLDER_KEY.public_key()),
            access_token_hash=access_token_hash(_TOKEN),
            body_hash=canonical_call_hash("probe.echo", {"text": "hi"}, str(correlation_id)),
        )

    @pytest.mark.asyncio
    async def test_the_token_is_read_from_the_provider_on_every_call(self) -> None:
        # a refresh loop rewrites the token in place; the second call must carry the new one.
        nats = _RecordingNats(_ok_reply())
        tokens = iter(["first.token.a", "second.token.b"])
        client = ToolCallClient(
            nats,
            principal_id=_POD_ID,
            identity_token=lambda: next(tokens),
            pop_signer=_Signer(_HOLDER_KEY),
        )

        await client.call("probe.echo", "1.0", {})
        await client.call("probe.echo", "1.0", {})

        assert [r["message"].context.identity_token for r in nats.requests] == ["first.token.a", "second.token.b"]

    @pytest.mark.asyncio
    async def test_a_correlation_id_is_minted_when_omitted(self) -> None:
        nats = _RecordingNats(_ok_reply())

        await _client(nats).call("probe.echo", "1.0")

        assert nats.requests[0]["message"].context.correlation_id is not None
        assert nats.requests[0]["message"].arguments == {}


class TestTheReply:
    @pytest.mark.asyncio
    async def test_a_successful_reply_is_returned_whole(self) -> None:
        reply = _ok_reply()
        response = await _client(_RecordingNats(reply)).call("probe.echo", "1.0", {})
        assert response is reply
        assert response.content == "42"
        assert response.metadata == {"rows": 1}

    @pytest.mark.asyncio
    async def test_a_registry_refusal_raises_with_the_registry_code(self) -> None:
        refused = ProxyCallResponse(
            success=False,
            content="",
            error="principal not authorized for tool probe.echo",
            error_code="TOOL_NOT_AUTHORIZED",
            context=CallContext(agent_id=_POD_ID),
        )
        with pytest.raises(ToolCallError) as excinfo:
            await _client(_RecordingNats(refused)).call("probe.echo", "1.0", {})
        assert excinfo.value.error_code == "TOOL_NOT_AUTHORIZED"
        assert "not authorized" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_a_tool_failure_raises_with_the_tool_code(self) -> None:
        # the tool answered, and said no: the same exception, the tool's code, so one ``except``
        # covers the registry's refusals and the tool's own.
        failed = ProxyCallResponse(
            success=False,
            content="",
            error="run not found",
            error_code="not_found",
            context=CallContext(agent_id=_POD_ID),
        )
        with pytest.raises(ToolCallError) as excinfo:
            await _client(_RecordingNats(failed)).call("ripple.audience_run_status", "1.0", {"run_id": "x"})
        assert excinfo.value.error_code == "not_found"

    @pytest.mark.asyncio
    async def test_a_failure_with_no_code_raises_unknown(self) -> None:
        failed = ProxyCallResponse(success=False, content="", error="something", context=CallContext(agent_id=_POD_ID))
        with pytest.raises(ToolCallError) as excinfo:
            await _client(_RecordingNats(failed)).call("probe.echo", "1.0", {})
        assert excinfo.value.error_code == "UNKNOWN"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "fault",
        [RequestTimeoutError("request timed out"), RequestError("no responders for subject")],
    )
    async def test_a_transport_fault_raises_request_failed(self, fault: Exception) -> None:
        with pytest.raises(ToolCallError) as excinfo:
            await _client(_RecordingNats(fault)).call("probe.echo", "1.0", {})
        assert excinfo.value.error_code == "REQUEST_FAILED"
        assert excinfo.value.__cause__ is fault


class TestTheDeadline:
    def test_the_client_waits_longer_than_the_registry_forwards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # a client deadline below the registry's budget turns every slow tool into the client's own
        # REQUEST_FAILED, which cannot tell a slow tool from a dead bus. the default is pinned above
        # the registry's platform default by the margin.
        monkeypatch.delenv("THREETEARS_REGISTRY_CALL_TIMEOUT", raising=False)
        assert DEFAULT_CALL_TIMEOUT_SECONDS == get_call_timeout() + CALL_TIMEOUT_MARGIN_SECONDS
        assert CALL_TIMEOUT_MARGIN_SECONDS > 0

    @pytest.mark.asyncio
    async def test_the_deadline_reaches_the_transport(self) -> None:
        nats = _RecordingNats(_ok_reply())
        client = ToolCallClient(
            nats,
            principal_id=_POD_ID,
            identity_token=lambda: _TOKEN,
            pop_signer=_Signer(_HOLDER_KEY),
            timeout=7.5,
        )
        await client.call("probe.echo", "1.0", {})
        assert nats.requests[0]["timeout"] == timedelta(seconds=7.5)
