"""The provider's end of the HTTP connection, for the provider-wrapper suites.

The wrapper suites used to replace ``ChatOpenAI._astream`` / ``ChatAnthropic._agenerate`` and
their siblings on the class, which reached into LangChain's private surface and skipped the
provider code that parses a real answer. These scripted wires sit where the provider's API sits
instead: each model is built through its public factory with a public client or ``base_url``
setting, the SDK serialises the request it would really send, and the answer the wire returns is
parsed by the SDK and LangChain exactly as a live one would be.

- :class:`ChatCompletionsWire` speaks the chat-completions protocol, which OpenAI and OpenRouter
  share. It is an ``httpx`` handler: OpenAI takes it through ``http_async_client`` /
  ``http_client``, OpenRouter through the ``openrouter`` SDK's ``client`` / ``async_client``.
- :class:`AnthropicMessagesWire` speaks the Anthropic Messages protocol. ``ChatAnthropic`` builds
  its own HTTP client and accepts no transport, so this one is served on a loopback socket and
  reached through the public ``base_url``.
- :func:`serve_http_handler` serves any ``httpx`` handler the same way, for a suite that scripts
  the answers itself -- a server error, then a success -- rather than describing one answer.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.server
import json
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

__all__ = [
    "AnthropicMessagesWire",
    "ChatCompletionsWire",
    "TextBlock",
    "ToolUseBlock",
    "anthropic_model",
    "openai_model",
    "openrouter_model",
    "serve_http_handler",
    "text_deltas",
    "tool_call_delta",
]

_CREATED = 1790000000


def text_deltas(*parts: str) -> list[dict[str, Any]]:
    """the chat-completions deltas that stream ``parts`` as assistant text, one delta per part.

    :param parts: the text pieces, in order
    :ptype parts: str
    :return: the deltas
    :rtype: list[dict[str, Any]]
    """
    return [{"role": "assistant", "content": part} if i == 0 else {"content": part} for i, part in enumerate(parts)]


def tool_call_delta(*calls: dict[str, Any]) -> dict[str, Any]:
    """one chat-completions delta carrying ``calls`` side by side.

    Each call is ``{"index": int, "id": str | None, "name": str | None, "arguments": str}``; a
    ``None`` id and name is a continuation fragment, as a provider sends every delta after a
    call's first.

    :param calls: the tool-call fragments this delta carries
    :ptype calls: dict[str, Any]
    :return: the delta
    :rtype: dict[str, Any]
    """
    fragments: list[dict[str, Any]] = []
    for call in calls:
        function: dict[str, Any] = {"arguments": call["arguments"]}
        fragment: dict[str, Any] = {"index": call["index"], "function": function}
        if call.get("name") is not None:
            function["name"] = call["name"]
        if call.get("id") is not None:
            fragment["id"] = call["id"]
            fragment["type"] = "function"
        fragments.append(fragment)
    return {"role": "assistant", "content": "", "tool_calls": fragments}


@dataclass
class ChatCompletionsWire:
    """the chat-completions API: records each request body and answers it, streamed or not.

    :ivar deltas: what a streamed answer sends, one server-sent event per delta
    :ivar message: what a non-streamed answer returns as its single choice's message
    :ivar finish: the finish reason the answer ends with
    :ivar silence_after: when set, a streamed answer sends this many deltas and then sends
        nothing for ``silence_s`` seconds before the rest
    :ivar silence_s: the length of that silence
    :ivar gap_s: the pause before every delta of a streamed answer
    :ivar answer_delay_s: the pause before a non-streamed answer's body arrives
    :ivar bodies: every request body received, in order
    """

    deltas: list[dict[str, Any]] = field(default_factory=lambda: text_deltas("ok"))
    message: dict[str, Any] = field(default_factory=lambda: {"role": "assistant", "content": "ok"})
    finish: str = "stop"
    silence_after: int | None = None
    silence_s: float = 0.0
    gap_s: float = 0.0
    answer_delay_s: float = 0.0
    bodies: list[dict[str, Any]] = field(default_factory=list)

    def _base(self, model: str) -> dict[str, Any]:
        return {"id": "gen-1", "provider": "Scripted", "model": model, "created": _CREATED}

    def _event(self, model: str, delta: dict[str, Any], finish: str | None) -> bytes:
        chunk = {
            **self._base(model),
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}],
        }
        return f"data: {json.dumps(chunk)}\n\n".encode()

    async def _events(self, model: str) -> AsyncIterator[bytes]:
        last = len(self.deltas) - 1
        for i, delta in enumerate(self.deltas):
            if self.silence_after is not None and i == self.silence_after:
                await asyncio.sleep(self.silence_s)
            if self.gap_s:
                await asyncio.sleep(self.gap_s)
            yield self._event(model, delta, self.finish if i == last else None)
        usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        tail = {**self._base(model), "object": "chat.completion.chunk", "choices": [], "usage": usage}
        yield f"data: {json.dumps(tail)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    def handle(self, request: httpx.Request) -> httpx.Response:
        """answer one chat request as the API would.

        :param request: the request the SDK sent
        :ptype request: httpx.Request
        :return: the response
        :rtype: httpx.Response
        """
        body = json.loads(request.content)
        self.bodies.append(body)
        model = str(body.get("model", "scripted"))
        if body.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=self._events(model))
        answer = {
            **self._base(model),
            "object": "chat.completion",
            "system_fingerprint": None,
            "choices": [{"index": 0, "finish_reason": self.finish, "logprobs": None, "message": self.message}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        if not self.answer_delay_s:
            return httpx.Response(200, json=answer)
        return httpx.Response(
            200, headers={"content-type": "application/json"}, content=self._delayed(json.dumps(answer).encode())
        )

    async def _delayed(self, body: bytes) -> AsyncIterator[bytes]:
        await asyncio.sleep(self.answer_delay_s)
        yield body

    def sent_tool_call_names(self) -> list[str]:
        """the tool-call names on the assistant messages of the last request sent.

        :return: the names, in message order
        :rtype: list[str]
        """
        return [
            call["function"]["name"]
            for message in self.bodies[-1]["messages"]
            if message.get("role") == "assistant"
            for call in message.get("tool_calls") or []
        ]


def openai_model(wire: ChatCompletionsWire, model_name: str = "gpt-4o", **settings: Any) -> Any:
    """an OpenAI chat model from the public factory, whose SDK talks to ``wire``.

    :param wire: the scripted API
    :ptype wire: ChatCompletionsWire
    :param model_name: the model id
    :ptype model_name: str
    :param settings: further factory keyword arguments
    :ptype settings: Any
    :return: the chat model
    :rtype: Any
    """
    from threetears.models.providers.openai import create_openai_chat

    transport = httpx.MockTransport(wire.handle)
    return create_openai_chat(
        model_name,
        "sk-test",
        max_retries=0,
        http_client=httpx.Client(transport=transport),
        http_async_client=httpx.AsyncClient(transport=transport),
        **settings,
    )


def openrouter_model(
    wire: ChatCompletionsWire, model_name: str = "deepseek/deepseek-chat-v3-0324", **settings: Any
) -> Any:
    """an OpenRouter chat model from the public factory, whose ``openrouter`` SDK talks to ``wire``.

    :param wire: the scripted API
    :ptype wire: ChatCompletionsWire
    :param model_name: the model id
    :ptype model_name: str
    :param settings: further factory keyword arguments
    :ptype settings: Any
    :return: the chat model
    :rtype: Any
    """
    import openrouter

    from threetears.models.providers.openrouter import create_openrouter_chat

    transport = httpx.MockTransport(wire.handle)
    sdk = openrouter.OpenRouter(
        api_key="sk-or-test",
        client=httpx.Client(transport=transport),
        async_client=httpx.AsyncClient(transport=transport),
    )
    return create_openrouter_chat(model_name, "sk-or-test", max_retries=0, client=sdk, **settings)


@dataclass
class TextBlock:
    """an Anthropic text content block, streamed one delta per part.

    :ivar parts: the text pieces, in order
    """

    parts: list[str]


@dataclass
class ToolUseBlock:
    """an Anthropic ``tool_use`` content block, its input streamed as raw JSON fragments.

    :ivar tool_use_id: the block id
    :ivar name: the tool name the model called
    :ivar json_parts: the ``input_json_delta`` fragments, in order
    """

    tool_use_id: str
    name: str
    json_parts: list[str]


@dataclass
class AnthropicMessagesWire:
    """the Anthropic Messages API: records each request body and answers it, streamed or not.

    :ivar blocks: the answer's content blocks
    :ivar stop_reason: the reason the answer ends with
    :ivar bodies: every request body received, in order
    """

    blocks: list[TextBlock | ToolUseBlock] = field(default_factory=lambda: [TextBlock(["ok"])])
    stop_reason: str = "end_turn"
    bodies: list[dict[str, Any]] = field(default_factory=list)

    def _message(self, model: str, content: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": None if not content else self.stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }

    def events(self, model: str) -> bytes:
        """the answer as the server-sent events a streamed request receives.

        :param model: the model id the request named
        :ptype model: str
        :return: the event-stream body
        :rtype: bytes
        """
        events: list[dict[str, Any]] = [{"type": "message_start", "message": self._message(model, [])}]
        for index, block in enumerate(self.blocks):
            if isinstance(block, TextBlock):
                start: dict[str, Any] = {"type": "text", "text": ""}
                deltas = [{"type": "text_delta", "text": part} for part in block.parts]
            else:
                start = {"type": "tool_use", "id": block.tool_use_id, "name": block.name, "input": {}}
                deltas = [{"type": "input_json_delta", "partial_json": part} for part in block.json_parts]
            events.append({"type": "content_block_start", "index": index, "content_block": start})
            events.extend({"type": "content_block_delta", "index": index, "delta": delta} for delta in deltas)
            events.append({"type": "content_block_stop", "index": index})
        events.append(
            {
                "type": "message_delta",
                "delta": {"stop_reason": self.stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 2},
            }
        )
        events.append({"type": "message_stop"})
        return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()

    def answer(self, model: str) -> bytes:
        """the answer as the JSON body a non-streamed request receives.

        :param model: the model id the request named
        :ptype model: str
        :return: the response body
        :rtype: bytes
        """
        content: list[dict[str, Any]] = []
        for block in self.blocks:
            if isinstance(block, TextBlock):
                content.append({"type": "text", "text": "".join(block.parts)})
            else:
                raw = "".join(block.json_parts)
                content.append(
                    {"type": "tool_use", "id": block.tool_use_id, "name": block.name, "input": json.loads(raw or "{}")}
                )
        return json.dumps(self._message(model, content)).encode()

    def sent_tool_use_names(self) -> list[str]:
        """the ``tool_use`` block names on the assistant messages of the last request sent.

        :return: the names, in message order
        :rtype: list[str]
        """
        return [
            block["name"]
            for message in self.bodies[-1]["messages"]
            if message.get("role") == "assistant" and isinstance(message.get("content"), list)
            for block in message["content"]
            if block.get("type") == "tool_use"
        ]

    @contextlib.contextmanager
    def serve(self) -> Iterator[str]:
        """serve this wire on a loopback port for as long as the block runs.

        :return: the base URL to hand ``create_anthropic_chat``
        :rtype: Iterator[str]
        """
        wire = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 -- the stdlib's dispatch name
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                wire.bodies.append(body)
                model = str(body.get("model", "scripted"))
                streamed = bool(body.get("stream"))
                payload = wire.events(model) if streamed else wire.answer(model)
                self.send_response(200)
                self.send_header("content-type", "text/event-stream" if streamed else "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                del format, args

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


def anthropic_model(base_url: str, model_name: str | None = None, **settings: Any) -> Any:
    """an Anthropic chat model from the public factory, pointed at a served wire.

    :param base_url: the URL :meth:`AnthropicMessagesWire.serve` yielded
    :ptype base_url: str
    :param model_name: the model id; the package default chat model when omitted
    :ptype model_name: str | None
    :param settings: further factory keyword arguments
    :ptype settings: Any
    :return: the chat model
    :rtype: Any
    """
    from threetears.models import DEFAULT_CHAT_MODEL
    from threetears.models.providers.anthropic import create_anthropic_chat

    return create_anthropic_chat(
        model_name or DEFAULT_CHAT_MODEL, "sk-ant-api03-test", base_url=base_url, max_retries=0, **settings
    )


@contextlib.contextmanager
def serve_http_handler(handler: Callable[[httpx.Request], httpx.Response]) -> Iterator[str]:
    """serve an ``httpx`` request handler on a loopback port for as long as the block runs.

    For a provider that accepts no transport (``ChatAnthropic``): the model is pointed at the
    yielded URL through its public ``base_url``, the SDK sends the request it really would, and
    each POST is handed to ``handler`` as an :class:`httpx.Request` whose answer is written back
    verbatim -- status, content type and body.

    :param handler: answers one request
    :ptype handler: Callable[[httpx.Request], httpx.Response]
    :return: the base URL to hand the provider factory
    :rtype: Iterator[str]
    """

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 -- the stdlib's dispatch name
            content = self.rfile.read(int(self.headers["content-length"]))
            request = httpx.Request(
                "POST", f"http://127.0.0.1{self.path}", headers=dict(self.headers.items()), content=content
            )
            response = handler(request)
            payload = response.read()
            self.send_response(response.status_code)
            self.send_header("content-type", response.headers.get("content-type", "application/json"))
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
