"""SwarmUIClient against a scripted SwarmUI (httpx.MockTransport): sessions, refusals in the body,
the rescan and backend waits, and what is never sent twice."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from threetears.models.providers.image.http_retry import RetryPolicy
from threetears.models.providers.image.swarmui import SwarmUIClient, SwarmUIError

BASE = "http://swarm:7801"


async def _no_sleep(_seconds: float) -> None:
    return None


class _Swarm:
    """a scripted SwarmUI: answers per API call, in order (the last one repeats), recording each body."""

    def __init__(self, answers: dict[str, list[Any]]) -> None:
        self.answers = {"GetNewSession": [{"session_id": "s1"}, {"session_id": "s2"}], **answers}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        name = request.url.path.removeprefix("/API/")
        self.calls.append((name, json.loads(request.content or b"{}")))
        queue = self.answers[name]
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, httpx.Response):
            return answer
        return httpx.Response(200, json=answer)

    def names(self) -> list[str]:
        return [n for n, _ in self.calls]


def _client(swarm: _Swarm, **kw: Any) -> SwarmUIClient:
    return SwarmUIClient(
        BASE, transport=httpx.MockTransport(swarm), retry=RetryPolicy(sleep=_no_sleep), sleep=_no_sleep, **kw
    )


async def test_a_picture_is_drawn_in_a_session_with_its_parameters() -> None:
    swarm = _Swarm({"GenerateText2Image": [{"images": ["View/local/raw/a.png"]}]})
    async with _client(swarm) as s:
        out = await s.generate({"prompt": "a lake", "model": "m", "seed": 7, "exactbackendid": "2"})
    assert out == ["View/local/raw/a.png"]
    name, body = swarm.calls[-1]
    assert name == "GenerateText2Image"
    assert body == {"images": 1, "prompt": "a lake", "model": "m", "seed": 7, "exactbackendid": "2", "session_id": "s1"}


async def test_a_refusal_in_the_body_raises_with_its_words() -> None:
    swarm = _Swarm({"GenerateText2Image": [{"error": "Prompt is empty"}]})
    async with _client(swarm) as s:
        with pytest.raises(SwarmUIError, match="Prompt is empty"):
            await s.generate({"prompt": ""})
    assert swarm.names().count("GenerateText2Image") == 1


async def test_an_expired_session_is_taken_again_once() -> None:
    gone = {"error": "Invalid session ID.", "error_id": "invalid_session_id"}
    swarm = _Swarm({"ListBackends": [gone, {"0": {"status": "running"}}]})
    async with _client(swarm) as s:
        assert await s.list_backends() == {"0": {"status": "running"}}
    assert [b.get("session_id") for n, b in swarm.calls if n == "ListBackends"] == ["s1", "s2"]


async def test_a_model_added_since_the_last_scan_gets_one_rescan() -> None:
    swarm = _Swarm(
        {
            "GenerateText2Image": [{"error": "Invalid model value"}, {"images": ["View/b.png"]}],
            "TriggerRefresh": [{}],
        }
    )
    async with _client(swarm) as s:
        assert await s.generate({"model": "new"}) == ["View/b.png"]
    assert swarm.names()[1:] == ["GenerateText2Image", "TriggerRefresh", "GenerateText2Image"]


async def test_a_second_unknown_model_after_the_rescan_is_raised() -> None:
    swarm = _Swarm({"GenerateText2Image": [{"error": "option does not exist"}], "TriggerRefresh": [{}]})
    async with _client(swarm) as s:
        with pytest.raises(SwarmUIError, match="option does not exist"):
            await s.generate({"loras": "x"})
    assert swarm.names().count("TriggerRefresh") == 1


async def test_a_connecting_backend_is_waited_for_a_bounded_number_of_times() -> None:
    swarm = _Swarm({"GenerateText2Image": [{"error": "No backends match the request"}]})
    async with _client(swarm, backend_waits=3) as s:
        with pytest.raises(SwarmUIError, match="No backends match"):
            await s.generate({"exactbackendid": "5"})
    assert swarm.names().count("GenerateText2Image") == 4


async def test_a_picture_is_not_asked_for_twice_once_the_request_left() -> None:
    swarm = _Swarm({"GenerateText2Image": [httpx.ReadTimeout("slow"), {"images": ["View/c.png"]}]})
    async with _client(swarm) as s:
        with pytest.raises(httpx.ReadTimeout):
            await s.generate({"prompt": "x"})
    assert swarm.names().count("GenerateText2Image") == 1


async def test_reads_are_sent_again_through_a_dropped_server() -> None:
    swarm = _Swarm({"ListBackends": [httpx.ConnectError("refused"), httpx.Response(502), {"1": {"status": "idle"}}]})
    async with _client(swarm) as s:
        assert await s.list_backends() == {"1": {"status": "idle"}}


async def test_a_non_2xx_status_is_an_error_with_its_status() -> None:
    swarm = _Swarm({"ListBackends": [httpx.Response(404, text="nope")]})
    async with _client(swarm) as s:
        with pytest.raises(SwarmUIError) as err:
            await s.list_backends()
    assert err.value.status_code == 404


async def test_reconnecting_a_backend_switches_it_off_then_on() -> None:
    swarm = _Swarm({"ToggleBackend": [{}]})
    async with _client(swarm) as s:
        await s.reconnect_backend(3, timeout=120)
    toggles = [b for n, b in swarm.calls if n == "ToggleBackend"]
    assert [(b["backend_id"], b["enabled"]) for b in toggles] == [(3, False), (3, True)]


def test_its_comfyui_is_the_proxy() -> None:
    assert SwarmUIClient(BASE + "/").comfy().base_url == f"{BASE}/ComfyBackendDirect"


async def test_used_outside_its_block_it_says_so() -> None:
    with pytest.raises(RuntimeError, match="async with"):
        await SwarmUIClient(BASE).list_backends()
