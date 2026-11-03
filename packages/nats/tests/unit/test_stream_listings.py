"""a listing of a growing stream -- a bucket's keys, an object store's objects -- replays nothing through a consumer.

On cobalt dev on 2026-10-30 every one of six scoped-snapshot retirements failed: the orphan sweep listed the object
store's metadata by replaying all 2345 messages through a named push consumer, and under load the push stalled part
way (``stalled after N of M``). Every listing now reads the subjects from ``STREAM.INFO`` and each subject's latest
message by ``DIRECT.GET``: request and reply, however large the stream. The fakes fail any consumer create, so a
listing that replays would fail here.
"""

from __future__ import annotations

import base64
import json
from datetime import timedelta

from threetears.nats import NatsKvBucket, NatsObjectStore
from threetears.nats.kv import KvTimings

from .listed_stream import SERVER_SUBJECTS_CAP, ListedClient, ListedKv, ListedStream

_BUCKET = "3tears-big"
_KV_STREAM = f"KV_{_BUCKET}"
_STORE = "pod-objects"
_OBJ_STREAM = f"OBJ_{_STORE}"

#: the objects the dev store held when the replay stalled
_DEV_OBJECTS = 2345


def _bucket(stream: ListedStream) -> NatsKvBucket:
    return NatsKvBucket(
        client=ListedClient(stream),  # type: ignore[arg-type]
        full_name=_BUCKET,
        kv=ListedKv(stream, _BUCKET),  # type: ignore[arg-type]
        ttl=None,
        timings=KvTimings(key_listing_timeout_seconds=30.0),
    )


def _meta_subject(name: str) -> str:
    return f"$O.{_STORE}.M.{base64.urlsafe_b64encode(name.encode()).decode()}"


def _record(name: str, nuid: str, *, deleted: bool = False) -> bytes:
    body: dict[str, object] = {"name": name, "bucket": _STORE, "nuid": nuid, "size": 1, "chunks": 1, "digest": "d"}
    if deleted:
        body["deleted"] = True
    return json.dumps(body).encode()


class TestAKeyListing:
    async def test_a_large_bucket_lists_its_live_keys_with_no_consumer_and_no_marker(self) -> None:
        live = [(f"$KV.{_BUCKET}.tiles.{n}", b"v", None) for n in range(_DEV_OBJECTS)]
        removed = [
            (f"$KV.{_BUCKET}.tiles.gone-del", b"", {"KV-Operation": "DEL"}),
            (f"$KV.{_BUCKET}.tiles.gone-purge", b"", {"KV-Operation": "PURGE"}),
        ]
        stream = ListedStream(_KV_STREAM, [*live[:10], *removed, *live[10:]])

        keys = await _bucket(stream).list_keys(prefix="tiles.")

        assert keys == [f"tiles.{n}" for n in range(_DEV_OBJECTS)], "a removed key listed, or the order lost"
        assert stream.infos == [(f"$KV.{_BUCKET}.tiles.>", 0)]

    async def test_the_owners_sweep_finds_the_markers_by_direct_read(self) -> None:
        stream = ListedStream(
            _KV_STREAM,
            [
                (f"$KV.{_BUCKET}.a", b"v", None),
                (f"$KV.{_BUCKET}.b", b"", {"KV-Operation": "DEL"}),
            ],
        )

        assert await _bucket(stream).purge_removal_markers() == 1
        assert stream.purged == [(f"$KV.{_BUCKET}.b", 3)]


class TestAnObjectListing:
    async def test_a_large_store_lists_its_objects_by_direct_read_with_no_consumer(self) -> None:
        latest = [(_meta_subject(f"o/{n}"), _record(f"o/{n}", f"n{n}"), None) for n in range(_DEV_OBJECTS)]
        latest.append((_meta_subject("o/gone"), _record("o/gone", "ngone", deleted=True), None))
        stream = ListedStream(_OBJ_STREAM, latest)
        store = NatsObjectStore(client=ListedClient(stream), full_name=_STORE)  # type: ignore[arg-type]

        listed = await store.list_objects(prefix="o/")

        assert [info.name for info in listed] == [f"o/{n}" for n in range(_DEV_OBJECTS)]

    async def test_names_past_the_servers_subject_cap_are_read_from_the_next_page(self) -> None:
        names = [f"a/{n}" for n in range(SERVER_SUBJECTS_CAP + 3)]
        stream = ListedStream(_OBJ_STREAM, [(_meta_subject(name), b"{}", None) for name in names])
        store = NatsObjectStore(client=ListedClient(stream), full_name=_STORE)  # type: ignore[arg-type]

        assert await store.list_names(prefix="a/") == sorted(names)
        assert stream.infos == [(f"$O.{_STORE}.M.>", 0), (f"$O.{_STORE}.M.>", SERVER_SUBJECTS_CAP)]
        assert stream.direct_reads == [], "a name listing reads the names from the subjects alone"

    async def test_the_orphan_sweep_reads_the_metadata_by_direct_read_and_purges_only_the_orphan(self) -> None:
        latest = [(f"$O.{_STORE}.C.n{n}", b"chunk", None) for n in range(_DEV_OBJECTS)]
        latest.append((f"$O.{_STORE}.C.orphan", b"chunk", None))
        latest.extend((_meta_subject(f"o/{n}"), _record(f"o/{n}", f"n{n}"), None) for n in range(_DEV_OBJECTS))
        stream = ListedStream(_OBJ_STREAM, latest)
        store = NatsObjectStore(  # type: ignore[arg-type]
            client=ListedClient(stream), full_name=_STORE, op_timeout=timedelta(seconds=5)
        )

        class _NoClaims:
            async def list_keys(self) -> list[str]:
                return []

        assert await store.purge_orphan_chunks(pointers=_NoClaims()) == 1  # type: ignore[arg-type]
        assert stream.purged == [(f"$O.{_STORE}.C.orphan", None)]
