"""A large object is hashed on a worker thread, never the event loop; a small one inline.

SHA-256 runs at about 1-2 GB/s, so the digest of a 100 MB snapshot checked on the loop held every other
request on the pod for 50-100 ms (cobalt prod reports pods, 2026-11-06, the startup snapshot fetch).
``hashlib`` lets go of the interpreter lock while it hashes a large piece, so a thread is real relief.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import threading
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from nats.js.errors import NotFoundError

from threetears.nats.object_store import NatsObjectStore

#: past the size from which an object is hashed off the loop (1 MiB)
_LARGE = 2 << 20


class _JetStream:
    """JetStream as a put and a get meet it: every publish is kept, and the last metadata message is answered."""

    def __init__(self) -> None:
        self.published: list[tuple[str, bytes]] = []

    async def get_last_msg(self, stream: str, subject: str, *, direct: bool = False) -> Any:
        for published_subject, payload in reversed(self.published):
            if published_subject == subject:
                return SimpleNamespace(data=payload)
        raise NotFoundError

    async def stream_info(self, name: str, subjects_filter: str | None = None) -> Any:
        return SimpleNamespace(state=SimpleNamespace(bytes=0), config=SimpleNamespace(max_bytes=-1))

    async def publish(self, subject: str, payload: bytes, *, headers: Any = None) -> None:
        self.published.append((subject, payload))


class _Client:
    def __init__(self) -> None:
        self.js = _JetStream()

    def jetstream_context(self) -> _JetStream:
        return self.js


class _Store(NatsObjectStore):
    """the store, its chunk reads answered with the chunks its put published to the fake."""

    fake_js: _JetStream

    async def _read_subject(self, subject: str, *, count: int, **_: Any) -> list[bytes]:  # type: ignore[override]
        return [payload for at, payload in self.fake_js.published if at == subject][:count]


def _store(chunk_bytes: int = 1 << 20) -> tuple[NatsObjectStore, _JetStream]:
    """a store over the fake JetStream, and the fake."""
    client = _Client()
    store = _Store(client=client, full_name="OBJ_pod-objects", chunk_bytes=chunk_bytes)  # type: ignore[arg-type]
    store.fake_js = client.js
    return store, client.js


@pytest.fixture
def hashing_threads(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """the thread each SHA-256 of this test was made on."""
    threads: list[int] = []
    original: Callable[..., Any] = hashlib.sha256

    def recording(*args: Any, **kwargs: Any) -> Any:
        threads.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(hashlib, "sha256", recording)
    return threads


class TestALargeObjectIsHashedOffTheLoop:
    async def test_a_large_put_and_get_hash_on_a_worker_thread(self, hashing_threads: list[int]) -> None:
        store, _ = _store()
        data = bytes(range(256)) * (_LARGE // 256)
        await store.put("enr/TX/1/big", data)
        assert await store.get("enr/TX/1/big") == data
        assert len(hashing_threads) >= 2, "the put and the get did not both hash"
        assert threading.get_ident() not in hashing_threads, "a large object was hashed on the event loop"

    async def test_a_small_object_hashes_inline(self, hashing_threads: list[int]) -> None:
        store, _ = _store()
        await store.put("enr/TX/1/small", b"small")
        assert await store.get("enr/TX/1/small") == b"small"
        assert hashing_threads and set(hashing_threads) == {threading.get_ident()}

    async def test_a_large_object_whose_bytes_differ_is_still_refused(self) -> None:
        store, js = _store()
        data = b"x" * (_LARGE)
        await store.put("enr/TX/1/bad", data)
        subject, payload = js.published[0]
        js.published[0] = (subject, b"y" + payload[1:])
        with pytest.raises(Exception, match="did not arrive as recorded"):
            await store.get("enr/TX/1/bad")

    async def test_the_recorded_digest_is_the_nats_digest_of_the_bytes(self) -> None:
        store, js = _store()
        data = b"z" * (_LARGE * 2)
        await store.put("enr/TX/1/digest", data)
        recorded = json.loads(js.published[-1][1])
        assert recorded["digest"].startswith("SHA-256=")
        assert recorded["digest"] == "SHA-256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode()


async def _ticker(stop: asyncio.Event, late: list[float], every: float = 0.002) -> None:
    """sleep ``every`` over and over, keeping how much later than asked each wake came."""
    while not stop.is_set():
        asked = time.perf_counter()
        await asyncio.sleep(every)
        late.append(time.perf_counter() - asked - every)


@pytest.mark.loop_timing
class TestTheLoopRunsWhileALargeObjectIsHashed:
    async def test_a_ticker_is_never_held_while_a_256_mb_object_is_put_and_got(self) -> None:
        store, _ = _store(chunk_bytes=8 << 20)
        data = b"\x5a" * (256 << 20)
        # how long one hash of it takes here, so the bound below is in proportion to this machine
        started = time.perf_counter()
        hashlib.sha256(data).digest()
        one_hash = time.perf_counter() - started
        stop = asyncio.Event()
        late: list[float] = []
        ticking = asyncio.create_task(_ticker(stop, late))
        await asyncio.sleep(0.01)
        await store.put("enr/TX/1/huge", data)
        got = await store.get("enr/TX/1/huge")
        stop.set()
        await ticking
        assert len(got) == len(data)
        worst = max(late)
        assert worst < max(0.05, one_hash / 3), (
            f"the loop was held {worst * 1000:.0f} ms while a 256 MB object was hashed "
            f"(one hash takes {one_hash * 1000:.0f} ms)"
        )
