"""A KV removal's marker can lapse, and a bucket's owner can purge the markers it already holds.

A delete publishes a marker, and on a bucket with no expiry the marker is the key's one message forever. A
delete given ``marker_ttl`` publishes a purge marker the server removes after that lifetime, still fenced on
the caller's revision; a stream that refuses one gets the plain delete. The owner's sweep
(:meth:`threetears.nats.NatsKvBucket.purge_removal_markers`) purges each marker by its exact subject and only
below its own sequence plus one, so a write that lands after the sweep read the marker is kept.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from nats.js.errors import APIError, KeyNotFoundError

from threetears.nats import NatsKvBucket
from threetears.nats import kv as kv_module

_BUCKET = "ns-leases"


# parity-exempt: the two JetStream verbs a lapsing removal and the owner's sweep send, recorded; full JetStream parity would be over-mocking
class _Js:
    def __init__(self, refusal: APIError | None = None) -> None:
        self.refusal = refusal
        self.published: list[dict[str, Any]] = []
        self.purged: list[tuple[str, str, int]] = []

    async def publish(
        self, subject: str, payload: bytes = b"", *, headers: dict[Any, str] | None = None, msg_ttl: float
    ) -> object:
        self.published.append(
            {
                "subject": subject,
                "payload": payload,
                "headers": {f"{getattr(name, 'value', name)}": value for name, value in (headers or {}).items()},
                "msg_ttl": msg_ttl,
            }
        )
        if self.refusal is not None:
            raise self.refusal
        return MagicMock(seq=9)

    async def purge_stream(self, name: str, *, subject: str, seq: int) -> bool:
        self.purged.append((name, subject, seq))
        return True


def _bucket(js: _Js, kv: Any | None = None) -> NatsKvBucket:
    client = MagicMock()
    client.jetstream_context = MagicMock(return_value=js)
    return NatsKvBucket(client=client, full_name=_BUCKET, kv=kv or MagicMock(), ttl=None)


class TestARemovalWithAMarkerLifetime:
    async def test_it_publishes_a_fenced_purge_marker_that_lapses(self) -> None:
        js = _Js()
        kv = MagicMock()
        kv.delete = AsyncMock()

        assert await _bucket(js, kv).delete(key="enr.answer-1", revision=7, marker_ttl=timedelta(seconds=5)) is True

        assert js.published == [
            {
                "subject": f"$KV.{_BUCKET}.enr.answer-1",
                "payload": b"",
                "headers": {
                    "KV-Operation": "PURGE",
                    "Nats-Rollup": "sub",
                    "Nats-Expected-Last-Subject-Sequence": "7",
                },
                "msg_ttl": 5.0,
            }
        ]
        kv.delete.assert_not_called()

    async def test_a_revision_the_key_moved_past_is_refused(self) -> None:
        js = _Js(APIError(code=400, err_code=10071, description="wrong last sequence"))
        kv = MagicMock()
        kv.delete = AsyncMock()

        assert await _bucket(js, kv).delete(key="k", revision=7, marker_ttl=timedelta(seconds=5)) is False
        kv.delete.assert_not_called()

    async def test_a_stream_that_refuses_a_marker_lifetime_gets_the_plain_fenced_delete(self) -> None:
        js = _Js(APIError(code=400, err_code=10166, description="per-message TTL is disabled"))
        kv = MagicMock()
        kv.delete = AsyncMock(return_value=True)

        assert await _bucket(js, kv).delete(key="k", revision=7, marker_ttl=timedelta(seconds=5)) is True
        kv.delete.assert_awaited_once_with("k", last=7)

    async def test_a_revision_of_zero_still_fences(self) -> None:
        js = _Js()

        await _bucket(js).delete(key="k", revision=0, marker_ttl=timedelta(seconds=5))

        assert js.published[0]["headers"]["Nats-Expected-Last-Subject-Sequence"] == "0"

    async def test_the_fallback_names_the_streams_own_refusal(self, caplog: pytest.LogCaptureFixture) -> None:
        js = _Js(APIError(code=400, err_code=10115, description="rollup not permitted"))
        kv = MagicMock()
        kv.delete = AsyncMock(return_value=True)

        with caplog.at_level(logging.WARNING):
            await _bucket(js, kv).delete(key="k", revision=7, marker_ttl=timedelta(seconds=5))

        refusal = next(
            record.getMessage() for record in caplog.records if "lapsing purge marker" in record.getMessage()
        )
        assert "10115" in refusal and "rollup not permitted" in refusal
        assert "allow_msg_ttl" not in refusal

    async def test_a_lifetime_under_a_second_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one second"):
            await _bucket(_Js()).delete(key="k", marker_ttl=timedelta(milliseconds=500))


# parity-exempt: the one read a key listing makes of each key (nats-py KeyValue.get); full KeyValue parity would be over-mocking
class _Kv:
    """each key's latest message, as nats-py's KeyValue.get answers it: a marker raises with its entry."""

    def __init__(self, latest: dict[str, tuple[int, str | None]]) -> None:
        self._latest = latest

    async def get(self, key: str, revision: int | None = None, validate_keys: bool = True) -> Any:
        sequence, operation = self._latest[key]
        entry = MagicMock(key=key, revision=sequence)
        if operation is not None:
            raise KeyNotFoundError(entry, operation)
        return entry


_LATEST = {"refresh.enr": (3, None), "enr_answers.a": (5, "DEL"), "enr_answers.b": (6, "PURGE")}


class TestTheOwnersSweep:
    @pytest.fixture(autouse=True)
    def _listing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        subjects = {f"$KV.{_BUCKET}.{key}": 1 for key in _LATEST}
        monkeypatch.setattr(kv_module, "stream_subjects", AsyncMock(return_value=subjects))

    async def test_it_purges_each_marker_below_its_own_sequence_and_no_live_key(self) -> None:
        js = _Js()

        assert await _bucket(js, _Kv(_LATEST)).purge_removal_markers() == 2

        assert js.purged == [
            (f"KV_{_BUCKET}", f"$KV.{_BUCKET}.enr_answers.a", 6),
            (f"KV_{_BUCKET}", f"$KV.{_BUCKET}.enr_answers.b", 7),
        ]

    async def test_a_dry_run_counts_and_purges_nothing(self) -> None:
        js = _Js()

        assert await _bucket(js, _Kv(_LATEST)).purge_removal_markers(apply=False) == 2
        assert js.purged == []
