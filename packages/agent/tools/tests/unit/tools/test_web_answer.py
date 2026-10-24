"""``threetears.web_answer``: Brave's cited answer as a tool (answers-requirements-design.md §10.5, AD13).

Driven through the tool's front door over a fake streaming transport that
replays the live Brave capture the search package ships as package data, so
the prose and the projection checked here are what a model and a program would
really receive. The ``answers`` alias is pinned beside it: a paid tool is
granted only by its own name, never by ``web``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from importlib.resources import files
from types import TracebackType
from typing import Self

import pytest
from pydantic import JsonValue

from threetears.agent.tools.aliases import GROUP_ALIASES, WEB_TOOLS, expand_selectors
from threetears.agent.tools.builtin import register_builtins
from threetears.agent.tools.builtin.web_answer import WebAnswerTool, create_web_answer_tool
from threetears.agent.tools.registry import ToolRegistry
from threetears.search.contracts import (
    ANSWER_METADATA_KEY,
    EGRESS_DIRECT,
    AnswerMetadata,
    StreamedResponse,
    StreamingTransport,
)


def _capture(name: str) -> bytes:
    """One of the live Brave captures the search package ships.

    :param name: the capture's file name
    :ptype name: str
    :return: the recorded bytes
    :rtype: bytes
    """
    return files("threetears.search.testing").joinpath("recordings", name).read_bytes()


class _FakeStreamedResponse(StreamedResponse):  # parity-with: threetears.search.contracts.StreamedResponse
    """One recorded body, replayed one SSE event per chunk."""

    def __init__(self, status_code: int, body: bytes, url: str) -> None:
        """Hold the recording.

        :param status_code: the recorded status
        :ptype status_code: int
        :param body: the recorded body
        :ptype body: bytes
        :param url: the URL asked for
        :ptype url: str
        """
        self._status_code = status_code
        self._chunks = [event + b"\n\n" for event in body.split(b"\n\n") if event]
        self._url = url
        self.closed = False

    @property
    def status_code(self) -> int:
        """The recorded status.

        :return: the status
        :rtype: int
        """
        return self._status_code

    @property
    def headers(self) -> Mapping[str, str]:
        """No headers beyond the stream's type.

        :return: the headers
        :rtype: Mapping[str, str]
        """
        return {"content-type": "text/event-stream"}

    @property
    def final_url(self) -> str:
        """The URL asked for.

        :return: the URL
        :rtype: str
        """
        return self._url

    @property
    def egress(self) -> str:
        """The default exit.

        :return: ``direct``
        :rtype: str
        """
        return EGRESS_DIRECT

    @property
    def attempts(self) -> int:
        """One attempt.

        :return: 1
        :rtype: int
        """
        return 1

    @property
    def elapsed_seconds(self) -> float:
        """A token figure.

        :return: elapsed seconds
        :rtype: float
        """
        return 0.01

    def __aiter__(self) -> AsyncIterator[bytes]:
        """Replay the body.

        :return: the chunks
        :rtype: AsyncIterator[bytes]
        """
        return self._replay()

    async def _replay(self) -> AsyncIterator[bytes]:
        """Yield each recorded chunk.

        :return: the chunks
        :rtype: AsyncIterator[bytes]
        """
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        """Release the stream."""
        self.closed = True

    async def __aenter__(self) -> Self:
        """Enter the owning block.

        :return: this stream
        :rtype: Self
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release on leaving the block.

        :param exc_type: exception type, if any
        :ptype exc_type: type[BaseException] | None
        :param exc: the exception, if any
        :ptype exc: BaseException | None
        :param traceback: the traceback, if any
        :ptype traceback: TracebackType | None
        """
        await self.aclose()


class _FakeStreamingTransport(StreamingTransport):  # parity-with: threetears.search.contracts.StreamingTransport
    """Answers every stream with one recording, counting what it was asked."""

    def __init__(self, body: bytes, *, status_code: int = 200) -> None:
        """Hold the recording every stream replays.

        :param body: the recorded body
        :ptype body: bytes
        :param status_code: the recorded status
        :ptype status_code: int
        """
        self._body = body
        self._status_code = status_code
        self.bodies: list[Mapping[str, JsonValue] | None] = []

    @property
    def egress_name(self) -> str:
        """The default exit.

        :return: ``direct``
        :rtype: str
        """
        return EGRESS_DIRECT

    async def stream(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Mapping[str, JsonValue] | None = None,
        timeout_seconds: float | None = None,
        max_bytes: int,
    ) -> StreamedResponse:
        """Open the recording.

        :param method: HTTP method, ignored
        :ptype method: str
        :param url: the URL asked for
        :ptype url: str
        :param headers: ignored
        :ptype headers: Mapping[str, str] | None
        :param json_body: the request body, recorded
        :ptype json_body: Mapping[str, JsonValue] | None
        :param timeout_seconds: ignored
        :ptype timeout_seconds: float | None
        :param max_bytes: ignored
        :ptype max_bytes: int
        :return: the open recording
        :rtype: StreamedResponse
        """
        self.bodies.append(json_body)
        return _FakeStreamedResponse(self._status_code, self._body, url)


async def test_the_tool_answers_with_the_provider_named_first_and_the_answer_under_its_key() -> None:
    """Content is the rendered answer; metadata carries the typed answer under ANSWER_METADATA_KEY."""
    transport = _FakeStreamingTransport(_capture("brave_answers_single.sse"))
    tool = WebAnswerTool(api_key="k", transport=transport)

    result = await tool.execute(question="  what is the largest rodent?  ")

    assert result.success and result.error is None
    assert result.content.splitlines()[0] == "Answer from brave, 12 sources:"
    payload = AnswerMetadata.from_metadata(result.metadata[ANSWER_METADATA_KEY])
    assert payload.question == "what is the largest rodent?"
    assert payload.answer is not None and len(payload.answer.sources) == 12
    assert payload.spend.calls == 1 and payload.spend.money > 0
    (sent,) = transport.bodies
    assert sent is not None and sent["messages"] == [{"role": "user", "content": "what is the largest rodent?"}]


async def test_a_refusal_is_a_failed_result_carrying_its_failure_record() -> None:
    transport = _FakeStreamingTransport(_capture("brave_answers_invalid_token.json"), status_code=422)

    result = await WebAnswerTool(api_key="bad", transport=transport).execute(question="anything")

    assert not result.success
    assert result.error == result.content
    assert result.content.startswith("answer failed (auth-failed):")
    payload = AnswerMetadata.from_metadata(result.metadata[ANSWER_METADATA_KEY])
    assert payload.answer is None
    assert payload.failure is not None and payload.failure.failure_class == "auth-failed"


async def test_an_empty_question_is_refused_before_anything_is_sent() -> None:
    transport = _FakeStreamingTransport(_capture("brave_answers_single.sse"))

    result = await WebAnswerTool(api_key="k", transport=transport).execute(question="   ")

    assert not result.success
    assert "needs a question" in result.content
    assert transport.bodies == []


def test_the_tool_is_named_and_takes_one_question() -> None:
    tool = WebAnswerTool(api_key="k", transport=_FakeStreamingTransport(b""))
    schema = tool.mcp_schema()

    assert tool.mcp_name() == schema.name == "threetears.web_answer"
    assert schema.input_schema["required"] == ["question"]


def test_the_factory_needs_a_host_supplied_key() -> None:
    with pytest.raises(ValueError, match="api_key"):
        create_web_answer_tool({}, "answer questions")
    built = create_web_answer_tool(
        {"api_key": "k", "transport": _FakeStreamingTransport(b""), "timeout_seconds": 40.0}, "answer questions"
    )
    assert built.name == "threetears.web_answer"


def test_with_no_transport_the_tool_reads_through_the_standalone_one() -> None:
    """The host-side traced client buffers, so the default is the leaf's streaming transport."""
    assert WebAnswerTool(api_key="k").mcp_name() == "threetears.web_answer"


def test_the_builtin_registry_offers_web_answer() -> None:
    registry = ToolRegistry()
    register_builtins(registry)
    assert "web_answer" in registry.list_types()


def test_answers_is_its_own_alias_and_web_does_not_grant_it() -> None:
    """AD13: a persona granted ``web`` for free search does not acquire a paid tool."""
    assert expand_selectors(["answers"]) == ["threetears.web_answer"]
    assert "threetears.web_answer" not in WEB_TOOLS
    assert "threetears.web_answer" not in expand_selectors(["web"])
    assert GROUP_ALIASES["web"] == frozenset({"threetears.web_search", "threetears.web_fetch"})
