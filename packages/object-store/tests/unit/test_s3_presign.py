"""S3ObjectStore.presigned_get_url signs off the event loop, with one client built once and kept.

Opening an aiobotocore client for each presign built a whole botocore client class on the loop each time: ~100 ms a
URL on cobalt dev's shared CPU, a stall per file of a load's partitions. A store that opened its own session signs
on a worker thread with one synchronous client, and its URLs are the ones the aiobotocore client signs.
"""

from __future__ import annotations

import threading
from typing import Any
from urllib.parse import parse_qs, urlsplit

import botocore.session  # type: ignore[import-untyped]
import pytest
from botocore.config import Config as BotoConfig  # type: ignore[import-untyped]
from aiobotocore.session import get_session  # type: ignore[import-untyped]

from threetears.object_store.s3 import S3ObjectStore

_ENDPOINT = "http://minio:9000"


def _store() -> S3ObjectStore:
    return S3ObjectStore(endpoint_url=_ENDPOINT, access_key="AKIATEST", secret_key="secret", bucket="exports")


def _without_time(url: str) -> tuple[str, str, dict[str, list[str]]]:
    """the URL's host, path and query, less what the signing time moves."""
    parts = urlsplit(url)
    query = {k: v for k, v in parse_qs(parts.query).items() if k not in ("X-Amz-Date", "X-Amz-Signature")}
    return parts.netloc, parts.path, query


async def test_presigns_on_a_worker_thread_with_one_client_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[int] = []
    signed_on: list[int] = []
    create = botocore.session.Session.create_client

    def creating(self: Any, *args: Any, **kwargs: Any) -> Any:
        built.append(threading.get_ident())
        client = create(self, *args, **kwargs)
        sign = client.generate_presigned_url

        def signing(*a: Any, **k: Any) -> str:
            signed_on.append(threading.get_ident())
            url: str = sign(*a, **k)
            return url

        client.generate_presigned_url = signing
        return client

    monkeypatch.setattr(botocore.session.Session, "create_client", creating)
    store = _store()
    urls = [await store.presigned_get_url(f"part-{n}.parquet", expires_in=60) for n in range(20)]

    loop_thread = threading.get_ident()
    assert len(built) == 1
    assert loop_thread not in built
    assert len(signed_on) == 20
    assert loop_thread not in signed_on
    assert len(set(urls)) == 20


async def test_its_urls_are_the_ones_the_async_client_signs() -> None:
    store = _store()
    url = await store.presigned_get_url("lsf/part-0.parquet", expires_in=60)
    async with get_session().create_client(
        "s3",
        endpoint_url=_ENDPOINT,
        aws_access_key_id="AKIATEST",
        aws_secret_access_key="secret",
        region_name="us-east-1",
        config=BotoConfig(signature_version="s3v4", read_timeout=900.0),
    ) as client:
        expected = await client.generate_presigned_url(
            "get_object", Params={"Bucket": "exports", "Key": "lsf/part-0.parquet"}, ExpiresIn=60
        )

    assert _without_time(url) == _without_time(expected)
    host, path, query = _without_time(url)
    assert (host, path) == ("minio:9000", "/exports/lsf/part-0.parquet")
    assert query["X-Amz-Expires"] == ["60"]
    assert "X-Amz-Signature" in parse_qs(urlsplit(url).query)
