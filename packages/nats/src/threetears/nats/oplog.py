"""JetStream op-log (WAL) primitive — the durable write-path log.

:class:`OpLog` is the canonical wrapper around one JetStream stream used
as an append-only operation log for the durable multi-pod write path
(scriob ``docs/arch/write-path.md`` §2). It is the same *class* of
primitive as :class:`threetears.nats.NatsKvBucket`: a thin, typed
wrapper over a JetStream-backed object, opened via
:meth:`OpLog.open` through the connected client's
:meth:`NatsClient.jetstream_context`.

design notes
------------

- **One stream per ``(repo, branch)``**, ``num_replicas=3`` (R3),
  **in-memory** storage: durability rides replication, not disk
  (write-path.md). A **generous dedup window** (5 minutes) makes a
  retried append an at-most-once no-op; the expected-last-sequence CAS
  is the unbounded backstop.
- **Two fences in one append.** ``Nats-Expected-Last-Sequence`` is the
  optimistic CAS (fence #1): a stale appender is rejected **in-band**
  with :class:`OpLogSequenceConflict`. ``Nats-Msg-Id`` is the dedup key:
  an op retried within the window returns the *original* sequence with
  ``deduplicated=True`` and writes no second message.
- **Ordering subtlety (live-probed, see
  ``docs/oplog-jetstream-api-notes.md``).** JetStream evaluates the CAS
  *before* dedup. So a duplicate ``op_id`` resent with a now-stale
  ``expected_last_seq`` surfaces as a wrong-last-sequence error, *not* a
  ``PubAck(duplicate=True)``. :meth:`append` discriminates: on a
  wrong-last-sequence error it scans the log for the ``op_id``; if found
  the append was a retry (no-op, ``deduplicated=True``), otherwise it is
  a genuine fence and raises.
- **Replay terminates.** :meth:`replay` reads the stream's current
  ``last_seq`` once and stops when caught up, so consuming it eagerly
  (``[r async for r in oplog.replay(...)]``) returns rather than hanging
  on a live subscription. A ``from_seq`` past the end yields nothing.
- Stream names cannot contain dots; the ``(repo, branch)`` segments are
  sanitized (``.`` -> ``-``) and namespace-prefixed, mirroring
  :func:`threetears.nats.subjects._sanitize`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from nats.js.api import (
    AckPolicy,
    ConsumerConfig,
    DeliverPolicy,
    Header,
    StorageType,
    StreamConfig,
)
from nats.js.errors import APIError
from threetears.observe import get_logger

from threetears.nats.errors import OpLogError, OpLogSequenceConflict
from threetears.nats.subjects import Subjects

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from threetears.nats.client import NatsClient

__all__ = ["AppendResult", "OpLog", "OpRecord"]


log = get_logger(__name__)


#: JetStream ``err_code`` for "wrong last sequence" — the in-band signal
#: that an ``Nats-Expected-Last-Sequence`` CAS failed. The *only* safe
#: discriminator for fence #1; every other ``APIError`` must surface as a
#: generic :class:`OpLogError` rather than be mistaken for a CAS conflict.
_WRONG_LAST_SEQUENCE_ERR_CODE: Final[int] = 10071

#: JetStream ``err_code`` for "replicas > 1 not supported in non-clustered
#: mode" — raised when an R3 stream is created against a single-node broker
#: (a dev box / single-node testcontainer). The production intent is R3, but
#: a non-clustered server physically cannot host >1 replica, so creation falls
#: back to a single replica on exactly this error (and only this one).
_REPLICAS_UNSUPPORTED_ERR_CODE: Final[int] = 10074

#: Stream replication factor (R3): durability rides replication, not disk.
#: Falls back to 1 against a non-clustered broker (see
#: :data:`_REPLICAS_UNSUPPORTED_ERR_CODE`).
_NUM_REPLICAS: Final[int] = 3

#: Dedup window in seconds — generous (>= minutes) per write-path.md
#: Constants; the expected-last-sequence CAS is the unbounded backstop.
_DEDUP_WINDOW_SECONDS: Final[float] = 300.0

#: Replay fetch batch size and per-fetch timeout (seconds).
_REPLAY_BATCH: Final[int] = 256
_REPLAY_FETCH_TIMEOUT: Final[float] = 2.0


@dataclass(frozen=True, slots=True)
class AppendResult:
    """outcome of an :meth:`OpLog.append`.

    :param seq: stream sequence the op occupies (the co-edit version);
        for a deduplicated retry this is the *original* op's sequence
    :ptype seq: int
    :param deduplicated: ``True`` when the append was an at-most-once
        no-op (the ``op_id`` was already in the dedup window); ``False``
        for a fresh append
    :ptype deduplicated: bool
    """

    seq: int
    deduplicated: bool


@dataclass(frozen=True, slots=True)
class OpRecord:
    """one logged operation, as yielded by :meth:`OpLog.replay`.

    :param seq: stream sequence of the op (ordering / version key)
    :ptype seq: int
    :param payload: the op's opaque payload bytes
    :ptype payload: bytes
    :param op_id: the client-assigned dedup key the op was logged with
    :ptype op_id: str
    """

    seq: int
    payload: bytes
    op_id: str


class OpLog:
    """one JetStream-backed append-only op-log for a ``(repo, branch)``.

    instances are produced by :meth:`OpLog.open`; the bare constructor
    is internal. an instance is bound to one stream/subject for its
    client's lifetime; do not cache across client recreations.

    :param client: connected wrapper client owning this op-log
    :ptype client: NatsClient
    :param stream: fully-qualified JetStream stream name (dot-free)
    :ptype stream: str
    :param subject: the single subject the stream is bound to
    :ptype subject: str
    """

    __slots__ = ("_client", "_stream", "_subject")

    def __init__(self, *, client: NatsClient, stream: str, subject: str) -> None:
        self._client = client
        self._stream = stream
        self._subject = subject

    @property
    def stream(self) -> str:
        """fully-qualified JetStream stream name backing this op-log.

        :return: dot-free, namespace-prefixed stream name
        :rtype: str
        """
        return self._stream

    @property
    def subject(self) -> str:
        """the single subject this op-log's stream is bound to.

        :return: dotted subject string
        :rtype: str
        """
        return self._subject

    # ------------------------------------------------------------------
    # opener
    # ------------------------------------------------------------------

    @classmethod
    async def open(cls, *, client: NatsClient, repo: str, branch: str) -> OpLog:
        """open or create the op-log stream for one ``(repo, branch)``.

        idempotent create-or-bind, mirroring
        :meth:`NatsKvBucket.open`: a fresh ``(repo, branch)`` gets a new
        in-memory, R3, single-subject stream with a generous dedup
        window; an existing one is bound. The stream name is the subject
        with dots sanitized to dashes (JetStream stream names forbid
        dots), namespace-prefixed via :meth:`Subjects.oplog`.

        :param client: connected wrapper client
        :ptype client: NatsClient
        :param repo: repository identifier; must be non-empty
        :ptype repo: str
        :param branch: branch / ref name; must be non-empty
        :ptype branch: str
        :return: ready op-log handle
        :rtype: OpLog
        :raises ValueError: if repo or branch is empty
        :raises OpLogError: if stream creation or binding fails
        """
        if not repo:
            raise ValueError("repo must be non-empty")
        if not branch:
            raise ValueError("branch must be non-empty")

        subject_obj = Subjects.oplog(repo, branch)
        subject = str(subject_obj)
        # Stream names cannot contain dots; reuse the subject's already
        # namespace-prefixed, segment-sanitized form and flatten the
        # remaining dotted separators.
        stream = subject.replace(".", "-")

        js = client.jetstream_context()

        def _config(num_replicas: int) -> StreamConfig:
            return StreamConfig(
                name=stream,
                subjects=[subject],
                storage=StorageType.MEMORY,
                num_replicas=num_replicas,
                duplicate_window=_DEDUP_WINDOW_SECONDS,
            )

        try:
            replicas = await cls._create_stream(js, stream=stream, subject=subject, config_for=_config)
            log.info(
                "JetStream op-log stream created",
                extra={
                    "extra_data": {
                        "stream": stream,
                        "subject": subject,
                        "num_replicas": replicas,
                        "dedup_window_seconds": _DEDUP_WINDOW_SECONDS,
                    }
                },
            )
        except OpLogError:
            # Creation genuinely failed; the stream may already exist -- bind to confirm.
            try:
                await js.stream_info(stream)
            except Exception as bind_exc:
                raise OpLogError(f"open op-log failed: stream={stream}: bind={bind_exc!r}") from bind_exc
            log.debug(
                "JetStream op-log stream bound (already existed)",
                extra={"extra_data": {"stream": stream, "subject": subject}},
            )

        return cls(client=client, stream=stream, subject=subject)

    @staticmethod
    async def _create_stream(
        js: Any,
        *,
        stream: str,
        subject: str,
        config_for: Callable[[int], StreamConfig],
    ) -> int:
        """create the stream at R3, falling back to 1 replica on a non-clustered broker.

        The production intent is R3 (durability rides replication). A single-node
        broker (dev box / single-node testcontainer) physically cannot host >1
        replica and rejects it with ``err_code`` 10074; on *exactly* that error we
        retry with a single replica so the primitive works on non-clustered infra.
        Any other failure surfaces as :class:`OpLogError` (the caller then tries a
        bind, in case the stream already exists).

        :param js: nats-py JetStream context
        :ptype js: Any
        :param stream: stream name (for diagnostics)
        :ptype stream: str
        :param subject: subject (for diagnostics)
        :ptype subject: str
        :param config_for: builds a :class:`StreamConfig` for a given replica count
        :ptype config_for: Callable[[int], StreamConfig]
        :return: the replica count the stream was actually created with
        :rtype: int
        :raises OpLogError: on any creation failure other than the non-clustered fallback succeeding
        """
        try:
            await js.add_stream(config=config_for(_NUM_REPLICAS))
        except APIError as exc:
            if exc.err_code != _REPLICAS_UNSUPPORTED_ERR_CODE:
                raise OpLogError(f"create op-log stream failed: stream={stream}: {exc!r}") from exc
            log.warning(
                "JetStream broker is non-clustered; op-log stream downgraded R3 -> 1 replica",
                extra={"extra_data": {"stream": stream, "subject": subject}},
            )
            try:
                await js.add_stream(config=config_for(1))
            except Exception as retry_exc:
                raise OpLogError(
                    f"create op-log stream failed (single-replica fallback): stream={stream}: {retry_exc!r}"
                ) from retry_exc
            return 1
        except Exception as exc:
            raise OpLogError(f"create op-log stream failed: stream={stream}: {exc!r}") from exc
        return _NUM_REPLICAS

    # ------------------------------------------------------------------
    # append (CAS fence #1 + op-id dedup)
    # ------------------------------------------------------------------

    async def append(self, *, payload: bytes, op_id: str, expected_last_seq: int) -> AppendResult:
        """append one op under expected-last-sequence CAS + op-id dedup.

        Publishes with ``Nats-Expected-Last-Sequence`` (fence #1) and
        ``Nats-Msg-Id`` (dedup). ``expected_last_seq=0`` means "expect an
        empty stream".

        - **fresh append** -> ``AppendResult(seq=<new>, deduplicated=False)``.
        - **duplicate op_id within the window** -> at-most-once no-op:
          ``AppendResult(seq=<original>, deduplicated=True)``, no second
          message written.
        - **stale ``expected_last_seq``** -> :class:`OpLogSequenceConflict`
          (fence #1), unless the op_id is already logged (a retry whose
          CAS went stale because the original landed), in which case it
          is treated as the dedup no-op above.

        :param payload: opaque op bytes to log
        :ptype payload: bytes
        :param op_id: client-assigned dedup key, stable across retries; must be non-empty.
            **Caller invariant:** an ``op_id`` identifies exactly one op — distinct ops must
            carry distinct ``op_id`` s. The CAS-conflict path trusts this: a wrong-last-sequence
            append whose ``op_id`` is already logged is treated as a retry no-op (its payload is
            dropped, the original sequence returned), so a client that reused an ``op_id`` for a
            *different* op would lose that write silently.
        :ptype op_id: str
        :param expected_last_seq: stream sequence the appender believes is current (0 == empty); must be >= 0
        :ptype expected_last_seq: int
        :return: append outcome (sequence + dedup flag)
        :rtype: AppendResult
        :raises ValueError: if op_id is empty or expected_last_seq is negative
        :raises OpLogSequenceConflict: if the CAS fails and the op is not a known retry
        :raises OpLogError: on transport / publish failure
        """
        if not op_id:
            raise ValueError("op_id must be non-empty")
        if expected_last_seq < 0:
            raise ValueError(f"expected_last_seq must be >= 0, got {expected_last_seq}")

        js = self._client.jetstream_context()
        headers = {
            Header.EXPECTED_LAST_SEQUENCE: str(expected_last_seq),
            Header.MSG_ID: op_id,
        }
        try:
            ack = await js.publish(self._subject, payload, headers=headers, stream=self._stream)
        except APIError as exc:
            if exc.err_code != _WRONG_LAST_SEQUENCE_ERR_CODE:
                # Some other JetStream 4xx/5xx (bad stream, bad subject, ...): not a CAS
                # conflict -- never swallow it as one.
                raise OpLogError(f"op-log append failed: stream={self._stream} op_id={op_id}: {exc!r}") from exc
            # Wrong-last-sequence: either a stale fence (genuine conflict) or a retry whose
            # CAS went stale because the original already landed. Discriminate on the dedup
            # key: a logged op_id means it was a retry (no-op); otherwise it is fence #1.
            existing_seq = await self._find_seq_for_op_id(op_id)
            if existing_seq is not None:
                return AppendResult(seq=existing_seq, deduplicated=True)
            raise OpLogSequenceConflict(
                f"op-log append rejected by expected-last-sequence CAS: stream={self._stream} "
                f"op_id={op_id} expected_last_seq={expected_last_seq}: {exc.description}"
            ) from exc
        except Exception as exc:
            raise OpLogError(f"op-log append failed: stream={self._stream} op_id={op_id}: {exc!r}") from exc

        return AppendResult(seq=int(ack.seq), deduplicated=bool(ack.duplicate))

    async def _find_seq_for_op_id(self, op_id: str) -> int | None:
        """scan the log for the stored message carrying ``op_id`` and return its seq.

        Used only on the wrong-last-sequence path to tell a duplicate retry apart from a
        genuine fence. Returns the sequence of the first stored message whose
        ``Nats-Msg-Id`` header matches, or ``None`` if the op_id is not logged.

        :param op_id: dedup key to locate
        :ptype op_id: str
        :return: stream sequence of the logged op, or ``None`` if absent
        :rtype: int | None
        :raises OpLogError: on transport failure during the scan
        """
        async for record in self.replay(from_seq=1):
            if record.op_id == op_id:
                return record.seq
        return None

    # ------------------------------------------------------------------
    # replay (ordered, terminating)
    # ------------------------------------------------------------------

    async def replay(self, from_seq: int) -> AsyncIterator[OpRecord]:
        """replay logged ops in sequence order, starting at ``from_seq``.

        Ordered by stream sequence (``DeliverPolicy.BY_START_SEQUENCE``).
        **Terminates** when caught up to the stream's current last
        sequence (read once up front), so eager consumption returns
        rather than hanging. A ``from_seq`` past the end yields nothing
        and returns cleanly. Read-only (``AckPolicy.NONE``); the
        ephemeral consumer is torn down on exit.

        :param from_seq: first stream sequence to deliver (1-based); a value past the current end yields nothing
        :ptype from_seq: int
        :return: async iterator over logged ops in sequence order
        :rtype: AsyncIterator[OpRecord]
        :raises ValueError: if from_seq < 1
        :raises OpLogError: on transport / consumer failure
        """
        if from_seq < 1:
            raise ValueError(f"from_seq must be >= 1, got {from_seq}")

        js = self._client.jetstream_context()
        try:
            info = await js.stream_info(self._stream)
        except Exception as exc:
            raise OpLogError(f"op-log replay failed (stream_info): stream={self._stream}: {exc!r}") from exc
        last_seq = int(info.state.last_seq)
        if from_seq > last_seq:
            # Past the end (includes an empty stream where last_seq == 0): nothing to yield.
            return

        config = ConsumerConfig(
            deliver_policy=DeliverPolicy.BY_START_SEQUENCE,
            opt_start_seq=from_seq,
            ack_policy=AckPolicy.NONE,
        )
        try:
            psub = await js.pull_subscribe(self._subject, stream=self._stream, config=config)
        except Exception as exc:
            raise OpLogError(f"op-log replay failed (subscribe): stream={self._stream}: {exc!r}") from exc

        try:
            done = False
            while not done:
                try:
                    msgs = await psub.fetch(_REPLAY_BATCH, timeout=_REPLAY_FETCH_TIMEOUT)
                except TimeoutError:
                    # Drained tail: fetch could not fill a batch within the timeout. Together
                    # with the `seq >= last_seq` exit below this is the normal terminator when
                    # the final batch is short. nats-py's FetchTimeoutError subclasses builtins
                    # TimeoutError, so this catches it. A real transport failure is NOT a
                    # timeout and must NOT be mistaken for end-of-stream: a silently-truncated
                    # replay would corrupt materialise / failover (the durability invariant).
                    break
                except APIError as exc:
                    raise OpLogError(f"op-log replay failed (fetch): stream={self._stream}: {exc!r}") from exc
                except Exception as exc:  # noqa: BLE001 - any non-timeout fetch failure surfaces loudly, never as a clean tail
                    raise OpLogError(f"op-log replay failed (fetch): stream={self._stream}: {exc!r}") from exc
                for msg in msgs:
                    seq = int(msg.metadata.sequence.stream)
                    headers = msg.headers or {}
                    op_id = headers.get(Header.MSG_ID, "")
                    yield OpRecord(seq=seq, payload=bytes(msg.data), op_id=op_id)
                    if seq >= last_seq:
                        done = True
                        break
        finally:
            try:
                await psub.unsubscribe()
            except Exception:  # noqa: BLE001 - best-effort teardown of the ephemeral replay consumer
                log.debug(
                    "op-log replay consumer teardown failed (ignored)",
                    extra={"extra_data": {"stream": self._stream}},
                )
