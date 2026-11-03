"""A write JetStream refuses for want of room is an ObjectStoreFullError; every other refusal is not.

The scoped snapshot sweeps its store and tries once more only on this error, so the mapping from
JetStream's codes is what its recovery rests on: a code mapped wrongly either never sweeps a full
store or sweeps one that is failing for another reason.
"""

from __future__ import annotations

from typing import Any

import pytest
from nats.js.errors import APIError, NotFoundError

from threetears.nats import ObjectStoreError, ObjectStoreFullError
from threetears.nats.object_store import NatsObjectStore


class _JetStream:
    """JetStream as the store's put meets it: the name is free, and every publish is refused."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    async def get_last_msg(self, stream: str, subject: str, *, direct: bool = False) -> Any:
        raise NotFoundError

    async def publish(self, subject: str, payload: bytes, *, headers: Any = None) -> None:
        raise self._error


class _Client:
    def __init__(self, error: BaseException) -> None:
        self._js = _JetStream(error)

    def jetstream_context(self) -> _JetStream:
        return self._js


async def _publish(error: BaseException) -> None:
    """a put whose first publish JetStream refuses with ``error``."""
    store = NatsObjectStore(client=_Client(error), full_name="OBJ_pod-objects")  # type: ignore[arg-type]
    await store.put("enr/TX/1/r", b"data")


async def test_a_write_past_max_bytes_is_a_full_store_naming_the_bucket_and_object() -> None:
    error = APIError(code=503, err_code=10077, description="maximum bytes exceeded")
    with pytest.raises(ObjectStoreFullError) as raised:
        await _publish(error)
    assert (raised.value.bucket, raised.value.name) == ("OBJ_pod-objects", "enr/TX/1/r")
    assert "maximum bytes exceeded" in str(raised.value)


async def test_insufficient_server_resources_is_a_full_store_said_so() -> None:
    with pytest.raises(ObjectStoreFullError, match="out of the storage"):
        await _publish(APIError(code=503, err_code=10047, description="insufficient resources"))


@pytest.mark.parametrize(
    ("err_code", "description"),
    [(10077, "maximum messages exceeded"), (10077, "store closed"), (10059, "stream not found")],
)
async def test_any_other_store_failure_is_a_plain_store_error(err_code: int, description: str) -> None:
    with pytest.raises(ObjectStoreError) as raised:
        await _publish(APIError(code=503, err_code=err_code, description=description))
    assert not isinstance(raised.value, ObjectStoreFullError)
    assert description in str(raised.value)


def test_a_write_claim_key_is_recognised_and_nothing_else() -> None:
    from threetears.nats.object_store_requests import is_write_claim_key

    assert is_write_claim_key("enr.w.17.06ac5bf5")
    assert not any(
        is_write_claim_key(key) for key in ("enr.s.TX", "enr.index", "enr.rebuild", "enr.w.x.a", "enr.w.17", "w.1.a.b")
    )


# parity-exempt: a JetStream stand-in answering only the metadata read that NatsObjectStore.info makes; the rest of the surface is not exercised
class _MetaJetStream:
    """JetStream answering one metadata read with a recorded ObjectInfo whose write time is ``mtime``."""

    def __init__(self, mtime: str) -> None:
        self._mtime = mtime

    async def get_last_msg(self, stream: str, subject: str, *, direct: bool = False) -> Any:
        import json
        from types import SimpleNamespace

        body = {"name": "a/b", "bucket": "pod-objects", "nuid": "n", "size": 1, "chunks": 1, "digest": "d"}
        body["mtime"] = self._mtime
        return SimpleNamespace(data=json.dumps(body).encode())


async def test_an_unparseable_recorded_mtime_is_warned_with_the_object_named(caplog: pytest.LogCaptureFixture) -> None:
    """failure injection: the writer recorded a time that is not ISO-8601; None must not read as 'never recorded'."""

    class _Client:
        def jetstream_context(self) -> _MetaJetStream:
            return _MetaJetStream("yesterday-ish")

    store = NatsObjectStore(client=_Client(), full_name="OBJ_pod-objects")  # type: ignore[arg-type]
    with caplog.at_level("WARNING"):
        info = await store.info("a/b")
    assert info is not None and info.mtime is None
    assert any("a/b" in r.getMessage() and "does not parse" in r.getMessage() for r in caplog.records)


# parity-exempt: a JetStream stand-in answering only the stream info a name listing reads; a consumer create fails the test
class _SubjectsJetStream:
    """JetStream holding metadata subjects for some names; counts STREAM.INFO and refuses consumers."""

    def __init__(self, bucket: str, names: list[str]) -> None:
        import base64

        self.infos: list[str | None] = []
        self._subjects = {f"$O.{bucket}.M.{base64.urlsafe_b64encode(n.encode()).decode()}": 1 for n in names}

    async def stream_info(self, name: str, subjects_filter: str | None = None) -> Any:
        from types import SimpleNamespace

        self.infos.append(subjects_filter)
        return SimpleNamespace(state=SimpleNamespace(subjects=dict(self._subjects)))

    async def add_consumer(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a name listing created a consumer: it reads one message per object")


async def test_listing_names_reads_one_stream_info_and_creates_no_consumer() -> None:
    """listing names is one STREAM.INFO, however many objects the bucket holds: the sweeper lists per publish,
    and a consumer listing delivers one message per object to every caller."""
    names = ["enr/TX/1/results", "enr/TX/2/results", "enr/NV/1/results", "enrp/TX/1/results"]
    js = _SubjectsJetStream("pod-objects", names)

    class _Client:
        def jetstream_context(self) -> _SubjectsJetStream:
            return js

    store = NatsObjectStore(client=_Client(), full_name="pod-objects")  # type: ignore[arg-type]
    assert await store.list_names() == sorted(names)
    assert await store.list_names(prefix="enr/TX/") == ["enr/TX/1/results", "enr/TX/2/results"]
    assert js.infos == ["$O.pod-objects.M.>", "$O.pod-objects.M.>"]


class _NoClaims:
    async def list_keys(self) -> list[str]:
        return []


# parity-exempt: a JetStream stand-in answering only the stream infos the orphan sweep reads
class _ChunksJetStream(_SubjectsJetStream):
    """JetStream holding named objects and some chunk subjects; a consumer create fails the test."""

    def __init__(self, bucket: str, names: list[str], chunks: list[str]) -> None:
        super().__init__(bucket, names)
        self._chunks = {f"$O.{bucket}.C.{nuid}": 1 for nuid in chunks}

    async def stream_info(self, name: str, subjects_filter: str | None = None) -> Any:
        from types import SimpleNamespace

        self.infos.append(subjects_filter)
        held = self._chunks if subjects_filter and ".C." in subjects_filter else self._subjects
        return SimpleNamespace(state=SimpleNamespace(subjects=dict(held)))


async def test_the_orphan_sweep_reads_no_metadata_while_no_more_chunk_subjects_than_objects_stand() -> None:
    """the declarer sweeps on every retire it answers; while every chunk subject can belong to an object
    (no more of them than objects), there is no orphan to find and the metadata is not read."""
    js = _ChunksJetStream("pod-objects", ["a", "b"], ["n1", "n2"])

    class _Client:
        def jetstream_context(self) -> _ChunksJetStream:
            return js

    store = NatsObjectStore(client=_Client(), full_name="pod-objects")  # type: ignore[arg-type]
    assert await store.purge_orphan_chunks(pointers=_NoClaims()) == 0  # type: ignore[arg-type]
    assert js.infos == ["$O.pod-objects.C.>", "$O.pod-objects.M.>"]
