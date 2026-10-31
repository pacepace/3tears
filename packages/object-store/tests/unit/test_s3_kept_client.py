"""S3ObjectStore keeps one aiobotocore client a loop, rather than opening one a request.

Each request once opened a client of its own: a botocore client built on the loop each time, and about a thousand
objects left in reference cycles. A process holding its cycle collector off while it works (a refresh sweeping only
the young generation) kept every request's until the hold ended, ~0.2 MiB a request: a full LSF pull's export reads.
"""

from __future__ import annotations

import asyncio
import gc
import threading
from collections.abc import AsyncIterator
from typing import Any

import pytest
from aiobotocore.session import AioSession  # type: ignore[import-untyped]
from aiohttp import web
from botocore.loaders import JSONFileLoader  # type: ignore[import-untyped]

from threetears.object_store.s3 import S3ObjectStore

_BODY = b"x" * (256 << 10)


async def _handle(request: web.Request) -> web.Response:
    if request.method == "DELETE":
        return web.Response(status=204)
    return web.Response(body=_BODY, headers={"ETag": '"e"', "Content-Type": "application/octet-stream"})


@pytest.fixture
async def endpoint() -> AsyncIterator[str]:
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", _handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    yield f"http://127.0.0.1:{port}"
    await runner.cleanup()


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    counted: list[int] = []
    create = AioSession.create_client

    def creating(self: Any, *args: Any, **kwargs: Any) -> Any:
        counted.append(1)
        return create(self, *args, **kwargs)

    monkeypatch.setattr(AioSession, "create_client", creating)
    return counted


def _store(endpoint: str) -> S3ObjectStore:
    return S3ObjectStore(endpoint_url=endpoint, access_key="AKIATEST", secret_key="secret", bucket="exports")


async def _read(store: S3ObjectStore, key: str) -> int:
    return sum([len(chunk) async for chunk in store.open_read(key)])


async def test_every_request_shares_one_client_until_it_is_closed(endpoint: str, opened: list[int]) -> None:
    store = _store(endpoint)
    sizes = await asyncio.gather(*(_read(store, f"part-{n}.parquet") for n in range(12)))
    await store.delete("part-0.parquet")

    assert sizes == [len(_BODY)] * 12
    assert len(opened) == 1

    await store.aclose()
    assert await _read(store, "part-1.parquet") == len(_BODY)
    assert len(opened) == 2
    await store.aclose()


async def test_requests_leave_no_cycles_for_a_held_collector(endpoint: str) -> None:
    """with the collector off and only the young generation swept, as a refresh holds it, what is left for a full
    collection afterwards does not grow with the requests made."""
    store = _store(endpoint)
    await _read(store, "warm.parquet")
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        for n in range(100):
            await _read(store, f"part-{n}.parquet")
            gc.collect(0)
        left = gc.collect()
    finally:
        if enabled:
            gc.enable()
        await store.aclose()
    # a client a request left ~1,100 objects each (110k here)
    assert left < 5_000, left


async def test_a_store_given_a_session_opens_a_client_a_request(endpoint: str, opened: list[int]) -> None:
    store = S3ObjectStore(
        endpoint_url=endpoint, access_key="AKIATEST", secret_key="secret", bucket="exports", session=AioSession()
    )
    for n in range(3):
        await _read(store, f"part-{n}.parquet")
    assert len(opened) == 3


async def test_the_first_client_never_reads_its_service_models_on_the_loop(
    endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """the s3 and sts models and endpoint rules a client is built from are read on a worker thread before the store's
    first client opens: read on the loop, the open held it ~50 ms (cobalt prod, 2026-10-31, a load's first export
    read)."""
    on_loop: list[str] = []
    loop_thread = threading.get_ident()
    load_file = JSONFileLoader.load_file

    def loading(self: Any, file_path: str, *args: Any, **kwargs: Any) -> Any:
        if threading.get_ident() == loop_thread:
            on_loop.append(file_path)
        return load_file(self, file_path, *args, **kwargs)

    monkeypatch.setattr(JSONFileLoader, "load_file", loading)
    store = _store(endpoint)
    assert await _read(store, "part-0.parquet") == len(_BODY)
    await store.aclose()
    assert on_loop == [], f"read on the event loop: {on_loop}"
