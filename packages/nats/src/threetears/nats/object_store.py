"""JetStream Object Store primitive: immutable named objects, read through NAMED consumers only.

:class:`NatsObjectStore` is the wrapper around one Object Store bucket. Callers never call
``js.object_store`` / ``js.create_object_store`` directly; they go through
:meth:`threetears.nats.NatsClient.object_store` (bind) or
:meth:`threetears.nats.NatsClient.ensure_object_store` (declare, the bucket's owner only).

design notes
------------

- **the wire shape is the NATS Object Store's.** A bucket ``<b>`` is the stream ``OBJ_<b>`` over
  ``$O.<b>.C.>`` (chunks) and ``$O.<b>.M.>`` (metadata, one message per object -- names are written once, so no rollup -- its subject
  the base64url of the name). Metadata is nats-py's ``ObjectInfo`` JSON, so the ``nats`` CLI and
  nats-py's own Object Store read what this wrapper writes.
- **reads use named consumers only.** nats-py's ``ObjectStore.get`` reads chunks through an
  ORDERED consumer, which is created unnamed (``$JS.API.CONSUMER.CREATE.{stream}``): its filter
  rides only in the request body, so a pod's grant -- which admits only the named form, whose filter
  rides in the SUBJECT where the server checks it -- refuses it, and a refused JetStream request is
  never answered, so it arrives as a deadline. :meth:`NatsObjectStore.get` and
  :meth:`NatsObjectStore.list_objects` create a freshly named push consumer instead
  (``$JS.API.CONSUMER.CREATE.{stream}.{name}.$O.<b>.…``), and metadata is read by subject through
  the direct get (``$JS.API.DIRECT.GET.{stream}.$O.<b>.M.…``), which the declarer turns on.
- **a name is written once.** :meth:`NatsObjectStore.put` publishes its metadata with
  ``Nats-Expected-Last-Subject-Sequence: 0``, so the server itself refuses a second object of one
  name, and a racing second writer cannot overwrite the first. Replacing an object in place would
  need the stream purge a pod never holds (nats-py's ``put`` purges the old chunks); a changed
  object is a new name instead -- a new epoch.
- **deletion is the declarer's.** :meth:`NatsObjectStore.delete` and
  :meth:`NatsObjectStore.purge_orphan_chunks` purge stream subjects, a management verb the hub holds
  and no pod does; a pod asks the hub to retire what it no longer serves.
- **every operation runs under a deadline** the operation cannot swallow
  (:func:`threetears.nats._publish.run_bounded`), and on the client's CURRENT connection, so a
  handle held across a credential renewal follows it without a rebind.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

from nats.js.api import DeliverPolicy
from nats.js.api import ObjectInfo as _NatsObjectInfo
from nats.js.api import ObjectMetaOptions
from nats.errors import NoRespondersError as _NatsNoRespondersError
from nats.errors import TimeoutError as _NatsTimeoutError
from nats.js.errors import APIError, NotFoundError
from threetears.observe import get_logger

from threetears.nats._named_read import NamedRead, read_through_named_consumer
from threetears.nats._publish import run_bounded
from threetears.nats.object_store_requests import OBJECT_NAME_PATTERN, ORPHAN_CHUNK_MIN_AGE
from threetears.nats.errors import (
    ObjectExistsError,
    ObjectNotFoundError,
    ObjectStoreError,
    ObjectStoreFullError,
    ObjectStoreNotFoundError,
    PublishTimeoutError,
)

if TYPE_CHECKING:
    from threetears.nats.client import NatsClient

__all__ = [
    "DEFAULT_OBJECT_CHUNK_BYTES",
    "DEFAULT_OBJECT_OP_TIMEOUT",
    "OBJECT_NAME_GRAMMAR",
    "OBJECT_STORE_STREAM_PREFIX",
    "NatsObjectStore",
    "ObjectInfo",
    "object_store_stream_name",
]

log = get_logger(__name__)

#: the stream backing an Object Store bucket ``<b>`` is ``OBJ_<b>``, as the NATS Object Store names it.
OBJECT_STORE_STREAM_PREFIX: Final[str] = "OBJ_"

#: what an object name may contain: the NATS Object Store's own key grammar.
OBJECT_NAME_GRAMMAR: Final[re.Pattern[str]] = re.compile(OBJECT_NAME_PATTERN)

#: bytes per chunk message. Under the default 1 MiB ``max_payload`` with room for headers, and large
#: enough that a few-megabyte object is a handful of messages rather than dozens.
DEFAULT_OBJECT_CHUNK_BYTES: Final[int] = 512 * 1024

#: the ceiling on one operation's wait: a chunk, a metadata read, a consumer create.
DEFAULT_OBJECT_OP_TIMEOUT: Final[timedelta] = timedelta(seconds=10)


#: consumer names a read mints carry these prefixes, so an operator listing a stream's consumers can
#: tell an object read (``og_``) and a listing (``ol_``) from anything else.
_GET_CONSUMER_PREFIX: Final[str] = "og_"
_LIST_CONSUMER_PREFIX: Final[str] = "ol_"

#: JetStream's refusals of a publish whose expected last subject sequence did not match.
_JS_ERR_WRONG_LAST_SEQUENCE: Final[frozenset[int]] = frozenset({10071, 10164})

#: JetStream's "stream name already in use with a different configuration".
_JS_ERR_STREAM_NAME_IN_USE: Final[int] = 10058

#: JetStream's refusal of a publish that would take the stream past its ``max_bytes``.
_JS_ERR_STORE_FULL: Final[frozenset[int]] = frozenset({10047, 10077})

#: how a JetStream request the server never answered arrives: our own deadline, nats-py's request
#: timeout, or no responder. A request this principal is not granted is dropped unanswered, so each
#: of these may name a missing grant as much as an unreachable broker.
_UNANSWERED: Final[tuple[type[BaseException], ...]] = (
    PublishTimeoutError,
    _NatsTimeoutError,
    _NatsNoRespondersError,
)

#: an answered refusal or an unanswered request: every way a JetStream call can fail to do its work
_FAILED: Final[tuple[type[BaseException], ...]] = (APIError, *_UNANSWERED)

_EXPECTED_LAST_SUBJECT_SEQUENCE: Final[str] = "Nats-Expected-Last-Subject-Sequence"

_DIGEST_PREFIX: Final[str] = "SHA-256="


def object_store_stream_name(bucket: str) -> str:
    """the JetStream stream backing an Object Store bucket.

    :param bucket: the fully-qualified bucket name
    :ptype bucket: str
    :return: ``OBJ_{bucket}``
    :rtype: str
    """
    return f"{OBJECT_STORE_STREAM_PREFIX}{bucket}"


@dataclass(frozen=True, slots=True)
class ObjectInfo:
    """what the bucket records about one object.

    :ivar name: the object's name
    :ivar size: its length in bytes
    :ivar chunks: how many chunk messages hold it
    :ivar digest: ``SHA-256=`` and the base64url of its SHA-256
    :ivar nuid: the token its chunks' subject carries
    :ivar mtime: when it was written, as the writer recorded it; ``None`` when not recorded
    """

    name: str
    size: int
    chunks: int
    digest: str
    nuid: str
    mtime: datetime | None


def _meta_subject(bucket: str, name: str) -> str:
    """the metadata subject of one object, as the NATS Object Store composes it.

    :param bucket: the fully-qualified bucket name
    :ptype bucket: str
    :param name: the object's name
    :ptype name: str
    :return: ``$O.{bucket}.M.{base64url(name)}``
    :rtype: str
    """
    return f"$O.{bucket}.M.{base64.urlsafe_b64encode(name.encode('utf-8')).decode('ascii')}"


def _chunk_subject(bucket: str, nuid: str) -> str:
    """the subject every chunk of one object is published on.

    :param bucket: the fully-qualified bucket name
    :ptype bucket: str
    :param nuid: the object's chunk token
    :ptype nuid: str
    :return: ``$O.{bucket}.C.{nuid}``
    :rtype: str
    """
    return f"$O.{bucket}.C.{nuid}"


def _digest(data: Iterable[bytes]) -> str:
    """the NATS Object Store digest of some bytes, fed in pieces.

    :param data: the bytes, in order
    :ptype data: Iterable[bytes]
    :return: ``SHA-256=`` and the base64url of the SHA-256
    :rtype: str
    """
    hasher = hashlib.sha256()
    for piece in data:
        hasher.update(piece)
    return f"{_DIGEST_PREFIX}{base64.urlsafe_b64encode(hasher.digest()).decode('ascii')}"


def _parse_mtime(value: str | None) -> datetime | None:
    """a recorded write time as an aware UTC datetime, or ``None`` when it does not parse.

    :param value: the recorded ISO time
    :ptype value: str | None
    :return: the time
    :rtype: datetime | None
    """
    parsed: datetime | None = None
    if value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
    if parsed is not None and parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _public_info(raw: _NatsObjectInfo) -> ObjectInfo:
    """the wrapper's record of an object, from nats-py's metadata model.

    :param raw: the decoded metadata
    :ptype raw: nats.js.api.ObjectInfo
    :return: the record
    :rtype: ObjectInfo
    """
    return ObjectInfo(
        name=raw.name,
        size=int(raw.size or 0),
        chunks=int(raw.chunks or 0),
        digest=raw.digest or "",
        nuid=raw.nuid or "",
        mtime=_parse_mtime(raw.mtime),
    )


def _check_name(name: str) -> None:
    """refuse an object name outside the Object Store's key grammar.

    :param name: the name
    :ptype name: str
    :return: nothing
    :rtype: None
    :raises ValueError: when the name is empty or outside :data:`OBJECT_NAME_GRAMMAR`
    """
    if not OBJECT_NAME_GRAMMAR.match(name):
        raise ValueError(f"object name {name!r} does not match {OBJECT_NAME_GRAMMAR.pattern}")


class NatsObjectStore:
    """one JetStream Object Store bucket.

    produced by :meth:`NatsClient.object_store` and :meth:`NatsClient.ensure_object_store`; the bare
    constructor is internal. Holds no nats-py handle: every operation takes the client's current
    JetStream context, so a credential renewal never strands it.

    :param client: the connected wrapper client
    :ptype client: NatsClient
    :param full_name: the fully-qualified bucket name
    :ptype full_name: str
    :param chunk_bytes: bytes per chunk message a put writes
    :ptype chunk_bytes: int
    :param op_timeout: the ceiling on one operation's wait
    :ptype op_timeout: timedelta
    """

    __slots__ = ("_chunk_bytes", "_client", "_full_name", "_op_timeout")

    def __init__(
        self,
        *,
        client: NatsClient,
        full_name: str,
        chunk_bytes: int = DEFAULT_OBJECT_CHUNK_BYTES,
        op_timeout: timedelta = DEFAULT_OBJECT_OP_TIMEOUT,
    ) -> None:
        if chunk_bytes <= 0:
            raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes!r}")
        self._client = client
        self._full_name = full_name
        self._chunk_bytes = chunk_bytes
        self._op_timeout = op_timeout

    @property
    def name(self) -> str:
        """the fully-qualified bucket name.

        :return: the bucket name
        :rtype: str
        """
        return self._full_name

    @property
    def stream(self) -> str:
        """the JetStream stream backing the bucket.

        :return: ``OBJ_{name}``
        :rtype: str
        """
        return object_store_stream_name(self._full_name)

    async def _bounded(self, op: Any, *, what: str) -> Any:
        """run one JetStream call under the store's deadline.

        :param op: builds the coroutine to run
        :ptype op: Any
        :param what: what the call is, for the timeout's message
        :ptype what: str
        :return: what the call returned
        :rtype: Any
        :raises PublishTimeoutError: when the call blew its deadline
        """
        return await run_bounded(op, timeout=self._op_timeout.total_seconds(), what=f"{what} on {self._full_name}")

    async def bind(self) -> None:
        """confirm the bucket exists (one ``STREAM.INFO``).

        :return: nothing
        :rtype: None
        :raises ObjectStoreNotFoundError: when the server answered that the bucket is absent
        :raises ObjectStoreError: when the bind failed otherwise -- an ungranted one is never
            answered and arrives here as a timeout naming the grant
        """
        js = self._client.jetstream_context()
        try:
            await self._bounded(lambda: js.stream_info(self.stream), what="object store bind")
        except NotFoundError as exc:
            raise ObjectStoreNotFoundError(
                f"object store {self._full_name} does not exist: its declarer has not declared it, or a NATS "
                f"restart wiped it",
                bucket=self._full_name,
            ) from exc
        except _UNANSWERED as exc:
            raise ObjectStoreError(
                f"binding object store {self._full_name} was not answered; an ungranted request is never "
                f"answered -- check this principal's grant on $JS.API.STREAM.INFO.{self.stream}"
            ) from exc
        except APIError as exc:
            raise ObjectStoreError(f"binding object store {self._full_name} failed: {exc}") from exc

    async def bytes_held(self) -> int:
        """how many bytes the bucket's stream holds, chunks and metadata together.

        :return: the stream's byte count
        :rtype: int
        :raises ObjectStoreError: when the stream cannot be read
        """
        js = self._client.jetstream_context()
        try:
            info = await self._bounded(lambda: js.stream_info(self.stream), what="object store size")
        except _FAILED as exc:
            raise ObjectStoreError(f"reading the size of object store {self._full_name} failed: {exc}") from exc
        return int(info.state.bytes or 0)

    async def _raw_info(self, name: str) -> _NatsObjectInfo | None:
        """an object's metadata as recorded, or ``None`` when it has none or was deleted.

        :param name: the object's name
        :ptype name: str
        :return: the decoded metadata
        :rtype: nats.js.api.ObjectInfo | None
        :raises ObjectStoreError: when the read fails or the metadata does not decode
        """
        js = self._client.jetstream_context()
        subject = _meta_subject(self._full_name, name)
        raw: _NatsObjectInfo | None = None
        try:
            msg = await self._bounded(
                lambda: js.get_last_msg(self.stream, subject, direct=True), what="object metadata read"
            )
        except NotFoundError:
            msg = None
        except _FAILED as exc:
            raise ObjectStoreError(
                f"reading metadata of {name!r} in {self._full_name} failed: {exc}. an ungranted read is never "
                f"answered -- check this principal's grant on $JS.API.DIRECT.GET.{self.stream}.$O.{self._full_name}.>"
            ) from exc
        if msg is not None and msg.data:
            try:
                raw = _NatsObjectInfo.from_response(json.loads(msg.data))
            except (ValueError, TypeError, KeyError) as exc:
                raise ObjectStoreError(f"metadata of {name!r} in {self._full_name} does not decode: {exc}") from exc
            if raw.deleted:
                raw = None
        return raw

    async def info(self, name: str) -> ObjectInfo | None:
        """what the bucket records about one object.

        :param name: the object's name
        :ptype name: str
        :return: the record, or ``None`` when the bucket holds no such object
        :rtype: ObjectInfo | None
        :raises ValueError: when the name is outside :data:`OBJECT_NAME_GRAMMAR`
        :raises ObjectStoreError: when the read fails
        """
        _check_name(name)
        raw = await self._raw_info(name)
        return None if raw is None else _public_info(raw)

    async def put(self, name: str, data: bytes) -> ObjectInfo:
        """write one object under a name the bucket does not hold yet.

        Chunks first, then the metadata, which the server accepts only while the name has none
        (``Nats-Expected-Last-Subject-Sequence: 0``): a reader that finds the metadata finds every
        chunk, and a second object of the name is refused however the writers race. A put that
        fails part way leaves chunks no object names; the declarer's
        :meth:`purge_orphan_chunks` removes them.

        :param name: the object's name; never reused
        :ptype name: str
        :param data: its bytes
        :ptype data: bytes
        :return: the record of what was written
        :rtype: ObjectInfo
        :raises ValueError: when the name is outside :data:`OBJECT_NAME_GRAMMAR`
        :raises ObjectExistsError: when the bucket already holds an object of that name
        :raises ObjectStoreError: when a chunk or the metadata could not be written, the bucket's
            ``max_bytes`` among the reasons
        """
        _check_name(name)
        if await self._raw_info(name) is not None:
            raise ObjectExistsError(
                f"object {name!r} already exists in {self._full_name}; objects are written once",
                bucket=self._full_name,
                name=name,
            )
        js = self._client.jetstream_context()
        nuid = uuid.uuid7().hex
        chunk_subject = _chunk_subject(self._full_name, nuid)
        pieces = [data[start : start + self._chunk_bytes] for start in range(0, len(data), self._chunk_bytes)]
        for piece in pieces:
            await self._publish(js, chunk_subject, piece, headers=None, name=name)
        record = _NatsObjectInfo(
            name=name,
            bucket=self._full_name,
            nuid=nuid,
            size=len(data),
            chunks=len(pieces),
            digest=_digest(pieces),
            mtime=datetime.now(UTC).isoformat(),
            options=ObjectMetaOptions(max_chunk_size=self._chunk_bytes),
        )
        await self._publish(
            js,
            _meta_subject(self._full_name, name),
            json.dumps(record.as_dict()).encode("utf-8"),
            headers={_EXPECTED_LAST_SUBJECT_SEQUENCE: "0"},
            name=name,
        )
        log.debug(
            "object written",
            extra={
                "extra_data": {"bucket": self._full_name, "object": name, "bytes": len(data), "chunks": len(pieces)}
            },
        )
        return _public_info(record)

    async def _publish(
        self, js: Any, subject: str, payload: bytes, *, headers: dict[str, str] | None, name: str
    ) -> None:
        """publish one chunk or metadata message and wait for the stream's acknowledgement.

        :param js: the JetStream context
        :ptype js: Any
        :param subject: the message's subject
        :ptype subject: str
        :param payload: its bytes
        :ptype payload: bytes
        :param headers: its headers, or ``None``
        :ptype headers: dict[str, str] | None
        :param name: the object it belongs to, for the error
        :ptype name: str
        :return: nothing
        :rtype: None
        :raises ObjectExistsError: when the metadata's expected sequence was refused
        :raises ObjectStoreError: on any other refusal or failure
        """
        try:
            await self._bounded(lambda: js.publish(subject, payload, headers=headers), what="object write")
        except APIError as exc:
            code = getattr(exc, "err_code", None)
            if code in _JS_ERR_WRONG_LAST_SEQUENCE:
                raise ObjectExistsError(
                    f"object {name!r} already exists in {self._full_name}; another writer named it first",
                    bucket=self._full_name,
                    name=name,
                ) from exc
            if code in _JS_ERR_STORE_FULL:
                raise ObjectStoreFullError(
                    f"object store {self._full_name} is full: writing {name!r} would pass its max_bytes. retire "
                    f"objects no longer served, or raise the bucket's bound"
                ) from exc
            raise ObjectStoreError(f"writing object {name!r} to {self._full_name} failed: {exc}") from exc
        except _UNANSWERED as exc:
            raise ObjectStoreError(
                f"writing object {name!r} to {self._full_name} was not acknowledged: {exc}. an ungranted publish "
                f"is never acknowledged -- check this principal's grant on $O.{self._full_name}.>"
            ) from exc

    async def get(self, name: str) -> bytes:
        """read one object whole, its size and digest checked.

        :param name: the object's name
        :ptype name: str
        :return: its bytes
        :rtype: bytes
        :raises ValueError: when the name is outside :data:`OBJECT_NAME_GRAMMAR`
        :raises ObjectNotFoundError: when the bucket holds no such object
        :raises ObjectStoreError: when the read fails, or what arrived is not what was recorded
        """
        _check_name(name)
        raw = await self._raw_info(name)
        if raw is None:
            raise ObjectNotFoundError(f"object {name!r} is not in {self._full_name}", bucket=self._full_name, name=name)
        chunks: list[bytes] = []
        if int(raw.chunks or 0) > 0:
            chunks = await self._read_subject(
                _chunk_subject(self._full_name, raw.nuid or ""),
                count=int(raw.chunks or 0),
                deliver_policy=DeliverPolicy.ALL,
                prefix=_GET_CONSUMER_PREFIX,
            )
        data = b"".join(chunks)
        if len(data) != int(raw.size or 0) or _digest(chunks) != (raw.digest or _digest(())):
            raise ObjectStoreError(
                f"object {name!r} in {self._full_name} did not arrive as recorded: {len(data)} bytes against "
                f"{raw.size} recorded, or its digest differs"
            )
        return data

    async def list_objects(self, *, prefix: str = "") -> list[ObjectInfo]:
        """every live object whose name starts with ``prefix``, in stream order.

        :param prefix: keep names starting with this; ``""`` lists every object
        :ptype prefix: str
        :return: the records
        :rtype: list[ObjectInfo]
        :raises ObjectStoreError: when the listing fails
        """
        result: list[ObjectInfo] = []
        for raw in await self._list_raw():
            if not raw.deleted and raw.name.startswith(prefix):
                result.append(_public_info(raw))
        return result

    async def _list_raw(self) -> list[_NatsObjectInfo]:
        """every object's latest metadata, deleted ones included.

        :return: the decoded metadata, in stream order
        :rtype: list[nats.js.api.ObjectInfo]
        :raises ObjectStoreError: when the listing fails or a record does not decode
        """
        messages = await self._read_subject(
            f"$O.{self._full_name}.M.>",
            count=None,
            deliver_policy=DeliverPolicy.LAST_PER_SUBJECT,
            prefix=_LIST_CONSUMER_PREFIX,
        )
        records: list[_NatsObjectInfo] = []
        for payload in messages:
            try:
                records.append(_NatsObjectInfo.from_response(json.loads(payload)))
            except (ValueError, TypeError, KeyError) as exc:
                raise ObjectStoreError(f"a metadata record in {self._full_name} does not decode: {exc}") from exc
        return records

    async def _read_subject(
        self, filter_subject: str, *, count: int | None, deliver_policy: DeliverPolicy, prefix: str
    ) -> list[bytes]:
        """read messages through one freshly NAMED push consumer filtered in its create subject.

        The inbox is subscribed first, so nothing the consumer delivers arrives before anything
        listens. With ``count`` given, exactly that many messages are read; without it, as many as
        the consumer reported pending when it was created.

        :param filter_subject: the consumer's filter
        :ptype filter_subject: str
        :param count: how many messages to read, or ``None`` for every pending one
        :ptype count: int | None
        :param deliver_policy: where the consumer starts
        :ptype deliver_policy: DeliverPolicy
        :param prefix: the consumer name's prefix
        :ptype prefix: str
        :return: the messages' payloads, in stream order
        :rtype: list[bytes]
        :raises ObjectStoreError: when the consumer cannot be created or a message does not arrive
        """
        messages = await read_through_named_consumer(
            self._client,
            NamedRead(
                stream=self.stream,
                filter_subject=filter_subject,
                deliver_policy=deliver_policy,
                name_prefix=prefix,
                count=count,
                timeout_seconds=self._op_timeout.total_seconds(),
            ),
            failure=lambda message, _cause: ObjectStoreError(f"{message} ({self._full_name})"),
        )
        return [bytes(msg.data) for msg in messages]

    async def delete(self, name: str) -> bool:
        """remove one object, its chunks and its metadata: the DECLARER's operation.

        Purges the object's subjects, a stream-management verb no pod holds; a pod asks its
        declarer (the hub) to retire what it no longer serves.

        :param name: the object's name
        :ptype name: str
        :return: whether the bucket held the object
        :rtype: bool
        :raises ValueError: when the name is outside :data:`OBJECT_NAME_GRAMMAR`
        :raises ObjectStoreError: when the purge fails
        """
        _check_name(name)
        raw = await self._raw_info(name)
        if raw is None:
            return False
        js = self._client.jetstream_context()
        try:
            if raw.nuid:
                await self._bounded(
                    lambda: js.purge_stream(self.stream, subject=_chunk_subject(self._full_name, raw.nuid or "")),
                    what="object chunk purge",
                )
            await self._bounded(
                lambda: js.purge_stream(self.stream, subject=_meta_subject(self._full_name, name)),
                what="object metadata purge",
            )
        except _FAILED as exc:
            raise ObjectStoreError(f"deleting object {name!r} from {self._full_name} failed: {exc}") from exc
        return True

    async def purge_orphan_chunks(self, *, older_than: timedelta = ORPHAN_CHUNK_MIN_AGE) -> int:
        """remove chunks no object names, once they are old enough not to be a put in progress.

        A put writes its chunks before its metadata, so chunks with no metadata are either a put
        still running or one that failed part way. Only those whose last chunk is older than
        ``older_than`` are purged. The DECLARER's operation, like :meth:`delete`.

        :param older_than: how old an unnamed chunk subject's last message must be
        :ptype older_than: timedelta
        :return: how many chunk subjects were purged
        :rtype: int
        :raises ObjectStoreError: when the stream cannot be read or a purge fails
        """
        js = self._client.jetstream_context()
        named = {raw.nuid for raw in await self._list_raw() if raw.nuid and not raw.deleted}
        try:
            info = await self._bounded(
                lambda: js.stream_info(self.stream, subjects_filter=f"$O.{self._full_name}.C.>"),
                what="object chunk listing",
            )
            subjects = dict(info.state.subjects or {})
            cutoff = datetime.now(UTC) - older_than
            purged = 0
            for subject in sorted(subjects):
                if subject.rsplit(".", 1)[-1] in named:
                    continue
                last = await self._bounded(lambda s=subject: js.get_last_msg(self.stream, s), what="chunk age read")
                stamp = last.time if last.time is None or last.time.tzinfo else last.time.replace(tzinfo=UTC)
                if stamp is not None and stamp > cutoff:
                    continue
                await self._bounded(lambda s=subject: js.purge_stream(self.stream, subject=s), what="chunk purge")
                purged += 1
        except _FAILED as exc:
            raise ObjectStoreError(f"sweeping unnamed chunks from {self._full_name} failed: {exc}") from exc
        if purged:
            log.info(
                "unnamed object chunks purged", extra={"extra_data": {"bucket": self._full_name, "subjects": purged}}
            )
        return purged
