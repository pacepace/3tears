"""ComfyUIClient against a scripted ComfyUI (httpx.MockTransport): what it sends, what it retries."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from threetears.models.providers.image.comfyui import ComfyUIClient, ComfyUIError
from threetears.models.providers.image.http_retry import RetryPolicy

BASE = "http://comfy:8188"
GRAPH = {"1": {"class_type": "SaveImage", "inputs": {}}}
DONE = {
    "p1": {
        "outputs": {"9": {"images": [{"filename": "a.png", "subfolder": "x", "type": "output"}]}},
        "status": {"status_str": "success", "completed": True},
    }
}


async def _no_sleep(_seconds: float) -> None:
    return None


def _client(handler: Callable[[httpx.Request], httpx.Response], attempts: int = 3, **kw: Any) -> ComfyUIClient:
    return ComfyUIClient(
        BASE,
        transport=httpx.MockTransport(handler),
        sleep=_no_sleep,
        poll_interval=0.0,
        retry=RetryPolicy(attempts=attempts, sleep=_no_sleep),
        **kw,
    )


class _Server:
    """a scripted ComfyUI: answers per path, in order, and records every request."""

    def __init__(self, answers: dict[str, list[Any]]) -> None:
        self.answers = answers
        self.seen: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        key = f"{request.method} {request.url.path}"
        answer = self.answers[key].pop(0) if len(self.answers[key]) > 1 else self.answers[key][0]
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, httpx.Response):
            return answer
        return httpx.Response(200, json=answer)

    def count(self, key: str) -> int:
        return sum(1 for r in self.seen if f"{r.method} {r.url.path}" == key)


async def test_run_queues_waits_and_returns_the_saved_pictures() -> None:
    server = _Server({"POST /prompt": [{"prompt_id": "p1"}], "GET /history/p1": [{}, DONE]})
    async with _client(server, client_id="samhla") as comfy:
        images = await comfy.run(GRAPH)
    assert images == [{"filename": "a.png", "subfolder": "x", "type": "output"}]
    sent = json.loads(server.seen[0].content)
    assert sent == {"prompt": GRAPH, "client_id": "samhla"}
    assert server.count("GET /history/p1") == 2


async def test_no_client_id_sends_the_graph_alone() -> None:
    server = _Server({"POST /prompt": [{"prompt_id": "p1"}]})
    async with _client(server) as comfy:
        await comfy.queue(GRAPH)
    assert json.loads(server.seen[0].content) == {"prompt": GRAPH}


async def test_view_reads_one_picture_by_its_name_folder_and_type() -> None:
    server = _Server({"GET /view": [httpx.Response(200, content=b"png")]})
    async with _client(server) as comfy:
        data = await comfy.view({"filename": "a.png", "subfolder": "x", "type": "output"})
    assert data == b"png"
    assert dict(server.seen[0].url.params) == {"filename": "a.png", "subfolder": "x", "type": "output"}


async def test_a_read_is_retried_through_a_5xx_and_an_unreachable_server() -> None:
    server = _Server(
        {
            "GET /queue": [
                httpx.Response(503),
                httpx.ConnectError("refused"),
                {"queue_running": [1], "queue_pending": [2, 3]},
            ]
        }
    )
    async with _client(server) as comfy:
        assert await comfy.queue_depth() == 3
    assert server.count("GET /queue") == 3


async def test_a_read_gives_up_after_its_attempts() -> None:
    server = _Server({"GET /queue": [httpx.Response(502, text="bad gateway")]})
    async with _client(server, attempts=2) as comfy:
        with pytest.raises(ComfyUIError) as err:
            await comfy.queue_depth()
    assert err.value.status_code == 502 and server.count("GET /queue") == 2


async def test_queueing_is_not_repeated_once_the_request_left() -> None:
    # a 5xx or a read timeout may mean ComfyUI queued it: sending it again could draw it twice
    for failure in (httpx.Response(500, text="boom"), httpx.ReadTimeout("slow")):
        server = _Server({"POST /prompt": [failure, {"prompt_id": "p2"}]})
        async with _client(server) as comfy:
            with pytest.raises((ComfyUIError, httpx.ReadTimeout)):
                await comfy.queue(GRAPH)
        assert server.count("POST /prompt") == 1


async def test_queueing_is_retried_when_the_request_never_left() -> None:
    server = _Server({"POST /prompt": [httpx.ConnectError("refused"), {"prompt_id": "p3"}]})
    async with _client(server) as comfy:
        assert await comfy.queue(GRAPH) == "p3"
    assert server.count("POST /prompt") == 2


async def test_a_rejected_graph_names_comfyuis_errors() -> None:
    server = _Server(
        {"POST /prompt": [httpx.Response(400, json={"error": {"message": "Prompt outputs failed validation"}})]}
    )
    async with _client(server) as comfy:
        with pytest.raises(ComfyUIError, match="failed validation") as err:
            await comfy.queue(GRAPH)
    assert err.value.status_code == 400 and server.count("POST /prompt") == 1


async def test_node_errors_in_an_accepted_answer_are_a_rejection() -> None:
    server = _Server({"POST /prompt": [{"prompt_id": "p4", "node_errors": {"3": {"errors": ["no model"]}}}]})
    async with _client(server) as comfy:
        with pytest.raises(ComfyUIError, match="no model"):
            await comfy.queue(GRAPH)


async def test_a_failed_graph_raises_with_its_messages() -> None:
    failed = {"p5": {"outputs": {}, "status": {"status_str": "error", "messages": [["execution_error", {"x": 1}]]}}}
    server = _Server({"GET /history/p5": [failed]})
    async with _client(server) as comfy:
        with pytest.raises(ComfyUIError, match="execution_error"):
            await comfy.wait("p5")


async def test_waiting_stops_at_its_bound() -> None:
    now = [0.0]

    def clock() -> float:
        now[0] += 10.0
        return now[0]

    server = _Server({"GET /history/p6": [{}]})
    comfy = ComfyUIClient(BASE, transport=httpx.MockTransport(server), sleep=_no_sleep, poll_interval=0.0, clock=clock)
    async with comfy:
        with pytest.raises(TimeoutError):
            await comfy.wait("p6", max_wait_seconds=25.0)
    assert server.count("GET /history/p6") == 3  # looks at 20s and 30s are inside 35s; at 40s it stops


async def test_upload_puts_a_picture_in_the_input_folder() -> None:
    server = _Server({"POST /upload/image": [{"name": "ref (1).png", "subfolder": "", "type": "input"}]})
    async with _client(server) as comfy:
        assert await comfy.upload_image(b"png-bytes", "ref.png") == "ref (1).png"
    body = server.seen[0].content
    assert b'name="image"; filename="ref.png"' in body and b"png-bytes" in body


async def test_used_outside_its_block_it_says_so() -> None:
    with pytest.raises(RuntimeError, match="async with"):
        await ComfyUIClient(BASE).queue_depth()


def test_attempts_below_one_are_refused() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(attempts=0)
