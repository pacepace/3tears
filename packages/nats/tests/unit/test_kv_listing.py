"""``live_kv_keys``: a bucket's live keys by name, for a caller holding only the client, with no consumer.

The same listing ``NatsKvBucket.list_keys`` makes -- ``STREAM.INFO`` paged past the server's cap, a subject-carried
``DIRECT.GET`` per key, a removal marker left out -- and its two failures that must never read as an empty bucket: a
refused ``STREAM.INFO``, and a key's read failing for a reason other than the key being gone.
"""

from __future__ import annotations

from typing import Any

import pytest
from nats.js.errors import APIError, NotFoundError

from threetears.nats import KvError, live_kv_keys
from threetears.nats.errors import KvBucketNotFoundError
from threetears.nats.raw_errors import JS_ERR_STREAM_NOT_FOUND

from .listed_stream import SERVER_SUBJECTS_CAP, ListedClient, ListedStream

_BUCKET = "hub-collections"
_STREAM = f"KV_{_BUCKET}"


def _key(name: str) -> str:
    return f"$KV.{_BUCKET}.{name}"


async def test_a_bucket_lists_its_live_keys_with_no_consumer_and_no_marker() -> None:
    latest = [
        (_key("a"), b"1", None),
        (_key("gone-del"), b"", {"KV-Operation": "DEL"}),
        (_key("b"), b"2", None),
        (_key("gone-purge"), b"", {"KV-Operation": "PURGE"}),
    ]
    stream = ListedStream(_STREAM, latest)

    keys = await live_kv_keys(ListedClient(stream), _BUCKET)  # type: ignore[arg-type]

    assert keys == ["a", "b"]
    assert stream.infos == [(f"$KV.{_BUCKET}.>", 0)]
    assert sorted(stream.direct_reads) == sorted(subject for subject, _, _ in latest)


async def test_keys_past_the_servers_subject_cap_are_read_from_the_next_page() -> None:
    names = [f"k{n}" for n in range(SERVER_SUBJECTS_CAP + 3)]
    stream = ListedStream(_STREAM, [(_key(name), b"v", None) for name in names])

    keys = await live_kv_keys(ListedClient(stream), _BUCKET)  # type: ignore[arg-type]

    assert keys == names
    assert stream.infos == [(f"$KV.{_BUCKET}.>", 0), (f"$KV.{_BUCKET}.>", SERVER_SUBJECTS_CAP)]


# parity-exempt: a ListedStream whose STREAM.INFO the server answers with an error instead of the stream's state
class _Refused(ListedStream):
    def __init__(self, error: APIError) -> None:
        super().__init__(_STREAM, [(_key("a"), b"1", None)])
        self._error = error

    async def stream_info(self, name: str, subjects_filter: str | None = None) -> Any:
        raise self._error


async def test_a_refused_stream_info_raises_never_an_empty_listing() -> None:
    refused = _Refused(APIError(code=403, description="permissions violation", err_code=10100))

    with pytest.raises(KvError, match="refused") as raised:
        await live_kv_keys(ListedClient(refused), _BUCKET)  # type: ignore[arg-type]

    assert not isinstance(raised.value, KvBucketNotFoundError)
    assert refused.direct_reads == [], "a key was read after the listing was refused"


async def test_a_gone_bucket_raises_its_own_type() -> None:
    gone = _Refused(NotFoundError(code=404, description="stream not found", err_code=JS_ERR_STREAM_NOT_FOUND))

    with pytest.raises(KvBucketNotFoundError) as raised:
        await live_kv_keys(ListedClient(gone), _BUCKET)  # type: ignore[arg-type]

    assert raised.value.bucket == _BUCKET


# parity-exempt: a ListedStream one of whose keys' reads fails for a reason other than the key being gone
class _ReadFails(ListedStream):
    async def get_last_msg(self, stream: str, subject: str, *, direct: bool = False) -> Any:
        if subject == _key("b"):
            raise TimeoutError("the read was not answered")
        return await super().get_last_msg(stream, subject, direct=direct)


async def test_a_read_failing_for_another_reason_than_absence_raises_and_drops_no_key() -> None:
    stream = _ReadFails(_STREAM, [(_key("a"), b"1", None), (_key("b"), b"2", None)])

    with pytest.raises(TimeoutError):
        await live_kv_keys(ListedClient(stream), _BUCKET)  # type: ignore[arg-type]
