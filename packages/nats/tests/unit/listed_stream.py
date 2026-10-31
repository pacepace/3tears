"""a stand-in stream answering only what a consumer-free listing asks: ``STREAM.INFO`` pages and ``DIRECT.GET``.

A consumer create fails the test: a listing that replays a growing stream through a consumer stalls part way under
load (``threetears.nats._stream_listing``), so a listing that would create one is the defect these fakes catch.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from nats.js.errors import KeyNotFoundError, NotFoundError

#: the server's cap on the subjects one ``STREAM.INFO`` answer carries (``JSMaxSubjectDetails``)
SERVER_SUBJECTS_CAP = 100_000


def _matches(subject: str, pattern: str) -> bool:
    """whether ``subject`` matches the NATS subject ``pattern`` (``*`` one token, ``>`` the rest).

    :param subject: a literal subject
    :ptype subject: str
    :param pattern: a subject filter
    :ptype pattern: str
    :return: whether it matches
    :rtype: bool
    """
    tokens, wanted = subject.split("."), pattern.split(".")
    for index, token in enumerate(wanted):
        if token == ">":
            return len(tokens) > index
        if index >= len(tokens) or token not in ("*", tokens[index]):
            return False
    return len(tokens) == len(wanted)


# parity-exempt: a JetStream stand-in answering only STREAM.INFO and subject-carried DIRECT.GET; a consumer create fails the test
class ListedStream:
    """one stream's latest message per subject, in stream order, and every request made of it.

    :ivar infos: each ``STREAM.INFO`` asked, as (filter, offset)
    :ivar direct_reads: each subject read by ``DIRECT.GET``
    :ivar purged: each purge, as (subject, below sequence)
    """

    def __init__(self, stream: str, latest: list[tuple[str, bytes, dict[str, str] | None]], *, held: int = 1) -> None:
        self.stream = stream
        self._latest = {
            subject: (sequence + 1, data, headers) for sequence, (subject, data, headers) in enumerate(latest)
        }
        self._held = held
        self.infos: list[tuple[str | None, int]] = []
        self.direct_reads: list[str] = []
        self.purged: list[tuple[str, int | None]] = []

    def _subjects(self, subjects_filter: str | None, offset: int) -> dict[str, int]:
        matching = [
            subject for subject in self._latest if subjects_filter is None or _matches(subject, subjects_filter)
        ]
        return dict.fromkeys(matching[offset : offset + SERVER_SUBJECTS_CAP], self._held)

    async def stream_info(self, name: str, subjects_filter: str | None = None) -> Any:
        assert name == self.stream
        self.infos.append((subjects_filter, 0))
        return SimpleNamespace(state=SimpleNamespace(subjects=self._subjects(subjects_filter, 0)))

    async def get_last_msg(self, stream: str, subject: str, *, direct: bool = False) -> Any:
        assert stream == self.stream and direct, "a listing reads by subject-carried DIRECT.GET"
        self.direct_reads.append(subject)
        if subject not in self._latest:
            raise NotFoundError
        sequence, data, headers = self._latest[subject]
        return SimpleNamespace(subject=subject, seq=sequence, data=data, headers=headers)

    async def purge_stream(self, name: str, *, subject: str | None = None, seq: int | None = None) -> None:
        assert name == self.stream
        self.purged.append((subject or "", seq))

    async def add_consumer(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a listing created a consumer: a replay of a growing stream stalls part way under load")

    async def request_raw(self, *, subject: Any, payload: bytes, timeout: Any) -> bytes:
        """a ``STREAM.INFO`` page past the first, asked by offset."""
        assert str(subject.path) == f"$JS.API.STREAM.INFO.{self.stream}", subject
        asked = json.loads(payload)
        self.infos.append((asked["subjects_filter"], int(asked["offset"])))
        matching = [s for s in self._latest if _matches(s, asked["subjects_filter"])]
        page = dict.fromkeys(matching[int(asked["offset"]) : int(asked["offset"]) + SERVER_SUBJECTS_CAP], self._held)
        return json.dumps(
            {
                "state": {"subjects": page},
                "total": len(matching),
                "offset": asked["offset"],
                "limit": SERVER_SUBJECTS_CAP,
            }
        ).encode()


# parity-exempt: stands in for NatsClient; a listing calls jetstream_context and request_raw, a bucket handle raw
class ListedClient:
    """the wrapper client over one :class:`ListedStream`.

    :param stream: the stream
    :ptype stream: ListedStream
    """

    def __init__(self, stream: ListedStream) -> None:
        self._stream = stream
        self.raw = SimpleNamespace(is_closed=False)

    def jetstream_context(self) -> ListedStream:
        return self._stream

    async def request_raw(self, *, subject: Any, payload: bytes, timeout: Any) -> bytes:
        return await self._stream.request_raw(subject=subject, payload=payload, timeout=timeout)


# parity-exempt: the one read a key listing makes of each key (nats-py KeyValue.get), answered from a ListedStream
class ListedKv:
    """a bucket's nats-py handle over a :class:`ListedStream`: a marker raises as nats-py raises it, with its entry.

    :param stream: the bucket's stream
    :ptype stream: ListedStream
    :param bucket: the bucket's full name
    :ptype bucket: str
    """

    def __init__(self, stream: ListedStream, bucket: str) -> None:
        self._stream = stream
        self._prefix = f"$KV.{bucket}."

    async def get(self, key: str, revision: int | None = None, validate_keys: bool = True) -> Any:
        try:
            message = await self._stream.get_last_msg(self._stream.stream, self._prefix + key, direct=True)
        except NotFoundError:
            raise KeyNotFoundError from None
        entry = SimpleNamespace(key=key, revision=message.seq, value=message.data)
        operation = (message.headers or {}).get("KV-Operation")
        if operation in ("DEL", "PURGE"):
            raise KeyNotFoundError(entry, operation)
        return entry
