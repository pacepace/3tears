"""A large answer crosses the bus compressed for a caller that reads it so, and one too large is refused aloud.

The hub's REST face serves a tool's answer on as it is, with ``Content-Encoding: gzip``, so it asks for
gzip (``CallContext.accept_encoding``) and the pod compresses the success. An agent never asks, and its
model reads the text: it gets plain text, even when the tool compressed the answer itself. An answer the
broker's ``max_payload`` cannot carry is refused ``TOOL_RESULT_TOO_LARGE`` naming its size, not dropped
by the broker after the tool ran while the caller waits out its timeout.

Every assertion reads the reply as the bus carries it: the published message serialized to JSON first.
"""

from __future__ import annotations

import gzip
import json
from typing import Any
from uuid import uuid4

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.content_encoding import (
    CONTENT_ENCODING_METADATA_KEY,
    GZIP,
    GZIP_MIN_BYTES,
    ContentEncodingError,
    DecodedTooLargeError,
    body_for_client,
    encode_for_caller,
    gzip_bytes,
    gzipped_content,
    plain_content,
)
from threetears.agent.tools.context_envelope import CallContext
from threetears.agent.tools.server import TOOL_RESULT_TOO_LARGE
from threetears.nats import IncomingMessage

from threetears.core.testing.replay_guard import FakeReplayGuard
from packages.agent.tools.tests.unit.tools.pod_auth import jwks_provider as _pod_jwks_provider
from packages.agent.tools.tests.unit.tools.pod_auth import recording_tool_server
from packages.agent.tools.tests.unit.tools.pod_auth import signed_call_payload as _signed_call_payload

_NS = "3tears"
_POD = "rows-pod-1"

#: an answer shaped like a report's rows: large, and compressing as JSON does
_ANSWER = json.dumps({"rows": [{"county": f"County {i}", "votes": i * 7, "share": 0.5} for i in range(4000)]})


class _AnswerTool(TearsTool):
    """answers with fixed content and metadata."""

    def __init__(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        super().__init__()
        self._content = content
        self._metadata = metadata

    async def execute(self, **kwargs: Any) -> ToolResult:
        del kwargs
        return ToolResult(success=True, content=self._content, metadata=self._metadata)

    def mcp_schema(self) -> MCPToolDefinition:
        return MCPToolDefinition(
            name="test.stub", version="1.0", description="stub", input_schema={"type": "object", "properties": {}}
        )

    def mcp_name(self) -> str:
        return "test.stub"

    def mcp_version(self) -> str:
        return "1.0"


def _call(*, accept_encoding: str | None) -> IncomingMessage:
    payload = _signed_call_payload(pod_id=_POD, conversation_id=uuid4(), user_id=uuid4())
    if accept_encoding is not None:
        # the context is not signed: the proof binds the tool, its arguments and the correlation id
        payload["context"] = {**(payload.get("context") or {}), "accept_encoding": accept_encoding}
    return IncomingMessage(
        data=json.dumps(payload).encode("utf-8"),
        reply_subject="_INBOX_registry_reg-1.abc",
        subject=f"{_NS}.tools.internal.{_POD}",
    )


async def _reply(tool: TearsTool, *, accept_encoding: str | None, max_payload: int | None = None) -> dict[str, Any]:
    server, rec = recording_tool_server(
        namespace=_NS, pod_id=_POD, jwks_provider=_pod_jwks_provider, assertion_replay_guard=FakeReplayGuard()
    )
    rec.max_payload = max_payload
    server.register(tool)
    await server.handle_call(_call(accept_encoding=accept_encoding))
    _, message = rec.last_reply
    return json.loads(message.model_dump_json())


class TestTheHelpers:
    def test_a_gzip_caller_gets_large_content_compressed_and_it_comes_back_whole(self) -> None:
        content, metadata = encode_for_caller(_ANSWER, None, CallContext(accept_encoding=GZIP))
        assert metadata == {CONTENT_ENCODING_METADATA_KEY: GZIP}
        compressed = gzip_bytes(content, metadata)
        assert compressed is not None and len(compressed) < len(_ANSWER) / 5
        assert gzip.decompress(compressed).decode("utf-8") == _ANSWER
        assert plain_content(content, metadata) == (_ANSWER, None)

    def test_small_content_stays_plain(self) -> None:
        small = "x" * (GZIP_MIN_BYTES - 1)
        assert encode_for_caller(small, None, CallContext(accept_encoding=GZIP)) == (small, None)

    def test_a_caller_that_did_not_ask_gets_a_tools_own_gzip_as_text(self) -> None:
        own = gzipped_content(gzip.compress(_ANSWER.encode("utf-8")))
        metadata = {CONTENT_ENCODING_METADATA_KEY: GZIP, "kept": 1}
        assert encode_for_caller(own, metadata, CallContext()) == (_ANSWER, {"kept": 1})
        assert encode_for_caller(own, metadata, None) == (_ANSWER, {"kept": 1})

    def test_a_tools_own_gzip_passes_to_a_gzip_caller_as_it_is(self) -> None:
        own = gzipped_content(gzip.compress(_ANSWER.encode("utf-8")))
        metadata = {CONTENT_ENCODING_METADATA_KEY: GZIP}
        assert encode_for_caller(own, metadata, CallContext(accept_encoding=GZIP)) == (own, metadata)

    def test_a_passthrough_body_is_left_alone(self) -> None:
        metadata = {"http": {"status": 200, "headers": {}, "content_type": "text/csv"}}
        assert encode_for_caller(_ANSWER, metadata, CallContext(accept_encoding=GZIP)) == (_ANSWER, metadata)


class TestTheWire:
    @pytest.mark.asyncio
    async def test_a_gzip_caller_is_answered_compressed(self) -> None:
        reply = await _reply(_AnswerTool(_ANSWER), accept_encoding=GZIP)
        assert reply["success"] is True
        assert reply["metadata"] == {CONTENT_ENCODING_METADATA_KEY: GZIP}
        assert len(reply["content"]) < len(_ANSWER) / 4
        assert gzip.decompress(gzip_bytes(reply["content"], reply["metadata"]) or b"").decode("utf-8") == _ANSWER

    @pytest.mark.asyncio
    async def test_an_agent_is_answered_in_text(self) -> None:
        reply = await _reply(_AnswerTool(_ANSWER), accept_encoding=None)
        assert (reply["content"], reply["metadata"]) == (_ANSWER, None)

    @pytest.mark.asyncio
    async def test_an_agent_reads_text_even_when_the_tool_compressed(self) -> None:
        own = gzipped_content(gzip.compress(_ANSWER.encode("utf-8")))
        reply = await _reply(_AnswerTool(own, {CONTENT_ENCODING_METADATA_KEY: GZIP}), accept_encoding=None)
        assert (reply["content"], reply["metadata"]) == (_ANSWER, None)

    @pytest.mark.asyncio
    async def test_an_answer_too_large_for_the_bus_is_refused_with_its_size(self) -> None:
        reply = await _reply(_AnswerTool(_ANSWER), accept_encoding=None, max_payload=128 * 1024)
        assert (reply["success"], reply["error_code"], reply["content"]) == (False, TOOL_RESULT_TOO_LARGE, "")
        assert "max_payload 131072" in reply["error"]

    @pytest.mark.asyncio
    async def test_the_same_answer_fits_compressed(self) -> None:
        reply = await _reply(_AnswerTool(_ANSWER), accept_encoding=GZIP, max_payload=128 * 1024)
        assert (reply["success"], reply["metadata"]) == (True, {CONTENT_ENCODING_METADATA_KEY: GZIP})


class _FailingTool(_AnswerTool):
    """fails, carrying its own gzip content."""

    async def execute(self, **kwargs: Any) -> ToolResult:
        del kwargs
        return ToolResult(success=False, content=self._content, metadata=self._metadata, error="refused")


class TestAFailure:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("accept_encoding", [GZIP, None])
    async def test_a_failure_carrying_gzip_reaches_every_caller_as_text(self, accept_encoding: str | None) -> None:
        own = gzipped_content(gzip.compress(b"why it failed"))
        reply = await _reply(_FailingTool(own, {CONTENT_ENCODING_METADATA_KEY: GZIP}), accept_encoding=accept_encoding)
        assert (reply["success"], reply["content"], reply["metadata"]) == (False, "why it failed", None)


class TestEveryDecodeIsGuardedAndBounded:
    def test_content_that_decodes_past_the_bound_is_refused_before_it_is_held(self) -> None:
        compressed = gzipped_content(gzip.compress(b"0" * 10_000))
        with pytest.raises(DecodedTooLargeError):
            plain_content(compressed, {CONTENT_ENCODING_METADATA_KEY: GZIP}, max_bytes=9_999)
        assert plain_content(compressed, {CONTENT_ENCODING_METADATA_KEY: GZIP}, max_bytes=10_000)[0] == "0" * 10_000

    @pytest.mark.parametrize(
        "content",
        [
            "not base64 at all!",
            gzipped_content(b"plain bytes, not gzip"),
            gzipped_content(gzip.compress(b"truncated answer")[:-8]),
            gzipped_content(gzip.compress(b"\xff\xfe not utf-8")),
        ],
        ids=["bad base64", "not gzip", "truncated", "not utf-8"],
    )
    def test_content_not_as_declared_raises_one_error(self, content: str) -> None:
        with pytest.raises(ContentEncodingError):
            plain_content(content, {CONTENT_ENCODING_METADATA_KEY: GZIP})

    def test_a_client_reading_gzip_gets_the_bytes_and_any_other_the_text(self) -> None:
        compressed = gzip.compress(_ANSWER.encode())
        content, metadata = gzipped_content(compressed), {CONTENT_ENCODING_METADATA_KEY: GZIP}
        assert body_for_client(content, metadata, client_accepts_gzip=True) == (compressed, True)
        assert body_for_client(content, metadata, client_accepts_gzip=False) == (_ANSWER.encode(), False)
        assert body_for_client("plain", None, client_accepts_gzip=True) == (b"plain", False)
