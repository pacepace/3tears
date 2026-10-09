"""the tool pod -> hub contract for a pod's OWN Object Store: declare it, and retire what it no longer serves.

A tool pod is granted an Object Store bucket and a pointer KV bucket of its own, composed under its
own scope (:func:`threetears.nats.subject_permissions.tool_pod_object_store_name`,
:func:`~threetears.nats.subject_permissions.tool_pod_pointers_bucket_name`), but it holds no
stream-management verb, so it can neither create them nor delete an object (a delete purges
subjects). It asks the hub, which owns every bucket's management:

- **declare** (:func:`declare_pod_object_store`): the hub declares both buckets -- the Object Store
  bounded and on memory storage, the pointer bucket in the uniform pod-bucket shape -- and remembers
  them, so a NATS restart that wipes them brings them back, empty. A pod asks at start and again
  whenever it finds the buckets gone; asking is idempotent.
- **retire** (:func:`retire_pod_objects`): the hub deletes the named objects from the pod's bucket
  and sweeps chunks a failed write left behind. A name the bucket does not hold is counted, not
  refused, so a retry after a lost reply is safe.

**Identity comes from the token.** Neither request names a bucket: the hub composes it from the
VERIFIED forwarded token, so a pod reaches its own bucket and no other.

The request and reply models, the subjects and the pod's client live here; the hub imports the models.

**Hub responder obligations:**

1. Subscribe :meth:`threetears.nats.Subjects.hub_object_store_declare` and
   :meth:`~threetears.nats.Subjects.hub_object_store_retire` in a queue group; a body that does not
   decode or breaks its bounds is answered ``INVALID_REQUEST``.
2. Verify ``identity_token`` as every forwarded-token subject does; a token that does not verify, or
   names anything but a tool pod, is answered ``IDENTITY_REFUSED``.
3. Declare or retire in the VERIFIED pod's own buckets only; a failure after verification is
   answered ``DECLARE_FAILED`` / ``RETIRE_FAILED``, and a retry is safe.
4. The sweep of chunks a failed write left behind takes chunk subjects no object names only while
   no write is in flight, judged by state, never by age: a writer claims what it writes in the
   pod's pointer bucket (a key :func:`is_write_claim_key` recognises, held from before its first
   chunk until a pointer names the object, renewed while it lives), and an object's chunks are
   stored before its metadata, so while any claim stands an unnamed chunk may be a put in flight
   and sweeping it would tear that put. A claim whose writer died lapses unrenewed, and the next
   sweep takes what that writer left.
   :meth:`threetears.nats.object_store.NatsObjectStore.purge_orphan_chunks` judges it, given the
   pointer bucket.

``error_code`` vocabulary: :data:`OBJECT_STORE_REQUEST_ERROR_CODES`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Annotated, Final
from uuid import UUID, uuid7

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_serializer
from threetears.observe import get_logger

from threetears.nats.errors import RequestError
from threetears.nats.subjects import Subject, Subjects

if TYPE_CHECKING:
    from threetears.nats.client import NatsClient
    from threetears.nats.kv import NatsKvBucket
    from threetears.nats.object_store import NatsObjectStore

__all__ = [
    "DEFAULT_OBJECT_STORE_REQUEST_TIMEOUT_SECONDS",
    "MAX_RETIRED_OBJECTS",
    "OBJECT_NAME_PATTERN",
    "OBJECT_STORE_REQUEST_ERROR_CODES",
    "WRITE_CLAIM_SEGMENT",
    "is_write_claim_key",
    "DeclaredObjectStore",
    "ObjectStoreDeclareReply",
    "ObjectStoreDeclareRequest",
    "ObjectStoreNotDeclaredError",
    "ObjectStoreRequestError",
    "ObjectStoreRequestRefusedError",
    "ObjectStoreRequestUnavailableError",
    "ObjectStoreRetireReply",
    "ObjectStoreRetireRequest",
    "PodObjectStore",
    "RetiredObjects",
    "bind_pod_object_store",
    "declare_pod_object_store",
    "retire_pod_objects",
]

log = get_logger(__name__)

#: objects one retire request may name
MAX_RETIRED_OBJECTS: Final[int] = 1000

#: the segment that marks a write claim's key in a pod's pointer bucket: ``{name}.w.{epoch}.{writer}``
WRITE_CLAIM_SEGMENT: Final = "w"


def is_write_claim_key(key: str) -> bool:
    """whether a pointer-bucket key is a writer's claim on what it writes: ``{name}.w.{epoch}.{writer}``.

    :param key: the key
    :ptype key: str
    :return: whether it is a write claim
    :rtype: bool
    """
    parts = key.split(".")
    return len(parts) == 4 and parts[1] == WRITE_CLAIM_SEGMENT and parts[2].isdigit() and bool(parts[0] and parts[3])


#: seconds a pod waits for the hub's answer
DEFAULT_OBJECT_STORE_REQUEST_TIMEOUT_SECONDS: Final[float] = 30.0

#: every ``error_code`` a responder answers with.
#:
#: - ``INVALID_REQUEST`` -- the body did not decode, or broke its bounds
#: - ``IDENTITY_REFUSED`` -- the forwarded token did not verify, or names no tool pod
#: - ``OBJECT_STORE_NOT_GRANTED`` -- the pod has not been given an Object Store (its registry row)
#: - ``OBJECT_STORE_BUDGET_EXHAUSTED`` -- declaring it would pass the platform's Object Store budget
#: - ``OBJECT_STORE_NOT_DECLARED`` -- a retire found no bucket: NATS lost it; declare it again
#: - ``DECLARE_FAILED`` -- the buckets could not be declared after verification; safe to retry
#: - ``RETIRE_FAILED`` -- the objects could not all be deleted after verification; safe to retry
OBJECT_STORE_REQUEST_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        "INVALID_REQUEST",
        "IDENTITY_REFUSED",
        "OBJECT_STORE_NOT_GRANTED",
        "OBJECT_STORE_BUDGET_EXHAUSTED",
        "OBJECT_STORE_NOT_DECLARED",
        "DECLARE_FAILED",
        "RETIRE_FAILED",
    }
)

#: the code a retire answers when the pod's bucket is gone, which :class:`ObjectStoreNotDeclaredError` carries
_NOT_DECLARED: Final[str] = "OBJECT_STORE_NOT_DECLARED"

#: the codes a retry can get past; every other code, one a newer hub added included, is a refusal
_RETRYABLE_ERROR_CODES: Final[frozenset[str]] = frozenset({"DECLARE_FAILED", "RETIRE_FAILED"})

#: an object name: the NATS Object Store's own key grammar. The ONE spelling of it: the request models
#: validate against it and :data:`threetears.nats.object_store.OBJECT_NAME_GRAMMAR` compiles it. Kept
#: here, in a module that imports no NATS client, so a pod that only asks can import it.
OBJECT_NAME_PATTERN: Final[str] = r"^[-/_=.a-zA-Z0-9]+$"


class ObjectStoreRequestError(Exception):
    """base of every way an Object Store request to the hub does not complete."""


class ObjectStoreRequestRefusedError(ObjectStoreRequestError):
    """the hub answered and refused; the same request will be refused again.

    :ivar error_code: the hub's code
    :ivar error_message: the hub's description, for an operator
    """

    def __init__(self, error_code: str, error_message: str) -> None:
        """
        :param error_code: the hub's refusal code
        :ptype error_code: str
        :param error_message: the hub's description
        :ptype error_message: str
        """
        self.error_code = error_code
        self.error_message = error_message
        super().__init__(f"object store request refused: {error_code}: {error_message}")


class ObjectStoreNotDeclaredError(ObjectStoreRequestRefusedError):
    """the pod's bucket does not exist (NATS lost it): declare it again, then retry."""


class ObjectStoreRequestUnavailableError(ObjectStoreRequestError):
    """no usable answer, or a hub failure after it verified the pod; safe to retry."""


class _TokenRequest(BaseModel):
    """the fields every request carries.

    :param identity_token: the pod's CURRENT hub-minted identity token; a ``SecretStr`` so a log line
        or traceback showing the request shows no token
    :ptype identity_token: SecretStr
    :param correlation_id: echoed on the reply
    :ptype correlation_id: UUID
    """

    model_config = ConfigDict(extra="forbid")

    identity_token: SecretStr
    correlation_id: UUID

    @field_serializer("identity_token", when_used="json")
    def _emit_token_on_the_wire(self, value: SecretStr) -> str:
        """the token in clear on the wire, where the hub verifies it; redacted everywhere else.

        :param value: the token
        :ptype value: SecretStr
        :return: the token's value
        :rtype: str
        """
        return value.get_secret_value()


class ObjectStoreDeclareRequest(_TokenRequest):
    """a tool pod's ask that the hub declare its own Object Store and pointer bucket. names no bucket."""


class ObjectStoreRetireRequest(_TokenRequest):
    """a tool pod's ask that the hub delete objects from its own bucket.

    :param names: the objects to delete
    :ptype names: list[str]
    """

    names: Annotated[
        list[Annotated[str, Field(pattern=OBJECT_NAME_PATTERN)]],
        Field(min_length=1, max_length=MAX_RETIRED_OBJECTS),
    ]


class ObjectStoreDeclareReply(BaseModel):
    """the hub's answer to a declare.

    :param success: whether both buckets are declared
    :ptype success: bool
    :param correlation_id: the request's correlation id
    :ptype correlation_id: UUID | None
    :param bucket: the Object Store's full name (on success)
    :ptype bucket: str | None
    :param pointers_bucket: the pointer bucket's full name (on success)
    :ptype pointers_bucket: str | None
    :param max_bytes: the Object Store's bound (on success)
    :ptype max_bytes: int | None
    :param error_code: one of :data:`OBJECT_STORE_REQUEST_ERROR_CODES` (on refusal)
    :ptype error_code: str | None
    :param error_message: a description for an operator (on refusal)
    :ptype error_message: str | None
    """

    success: bool
    correlation_id: UUID | None = None
    bucket: str | None = None
    pointers_bucket: str | None = None
    max_bytes: int | None = None
    error_code: str | None = None
    error_message: str | None = None


class ObjectStoreRetireReply(BaseModel):
    """the hub's answer to a retire.

    :param success: whether every named object is gone
    :ptype success: bool
    :param correlation_id: the request's correlation id
    :ptype correlation_id: UUID | None
    :param retired: how many named objects were deleted (on success)
    :ptype retired: int | None
    :param absent: how many named objects the bucket did not hold (on success)
    :ptype absent: int | None
    :param orphan_chunks: how many chunk subjects no object named were swept (on success)
    :ptype orphan_chunks: int | None
    :param error_code: one of :data:`OBJECT_STORE_REQUEST_ERROR_CODES` (on refusal)
    :ptype error_code: str | None
    :param error_message: a description for an operator (on refusal)
    :ptype error_message: str | None
    """

    success: bool
    correlation_id: UUID | None = None
    retired: int | None = None
    absent: int | None = None
    orphan_chunks: int | None = None
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class DeclaredObjectStore:
    """the buckets the hub declared for the pod.

    :ivar bucket: the Object Store's full name
    :ivar pointers_bucket: the pointer bucket's full name
    :ivar max_bytes: the Object Store's bound
    """

    bucket: str
    pointers_bucket: str
    max_bytes: int


@dataclass(frozen=True, slots=True)
class RetiredObjects:
    """what a retire did.

    :ivar retired: named objects deleted
    :ivar absent: named objects the bucket did not hold
    :ivar orphan_chunks: chunk subjects no object named, swept
    """

    retired: int
    absent: int
    orphan_chunks: int


async def _ask(
    nats_client: NatsClient,
    *,
    subject: Subject,
    request: _TokenRequest,
    what: str,
    timeout_seconds: float,
) -> bytes:
    """send one request and return the raw reply.

    :param nats_client: the pod's connected NATS client
    :ptype nats_client: NatsClient
    :param subject: the request subject
    :ptype subject: Subject
    :param request: the request
    :ptype request: _TokenRequest
    :param what: what is asked, for the error
    :ptype what: str
    :param timeout_seconds: seconds to wait
    :ptype timeout_seconds: float
    :return: the reply's bytes
    :rtype: bytes
    :raises ObjectStoreRequestUnavailableError: on a transport failure or timeout
    """
    try:
        raw: bytes = await nats_client.request_raw(
            subject=subject,
            payload=request.model_dump_json().encode("utf-8"),
            timeout=timedelta(seconds=timeout_seconds),
        )
    except RequestError as exc:
        raise ObjectStoreRequestUnavailableError(
            f"{what} failed (correlation_id={request.correlation_id}): {exc}"
        ) from exc
    return raw


def _check_reply(reply: ObjectStoreDeclareReply | ObjectStoreRetireReply, *, correlation_id: UUID, what: str) -> None:
    """refuse a reply that answers another request, or that refuses.

    :param reply: the decoded reply
    :ptype reply: ObjectStoreDeclareReply | ObjectStoreRetireReply
    :param correlation_id: the request's correlation id
    :ptype correlation_id: UUID
    :param what: what was asked, for the error
    :ptype what: str
    :return: nothing
    :rtype: None
    :raises ObjectStoreRequestUnavailableError: on a reply to another request, or a retryable code
    :raises ObjectStoreRequestRefusedError: on any other refusal
    """
    # a refusal with no correlation id is this request's: a body the hub could not decode had none to echo
    if reply.correlation_id != correlation_id and (reply.success or reply.correlation_id is not None):
        raise ObjectStoreRequestUnavailableError(
            f"{what} reply carried correlation_id={reply.correlation_id}, not {correlation_id}"
        )
    if not reply.success and reply.error_code in _RETRYABLE_ERROR_CODES:
        raise ObjectStoreRequestUnavailableError(
            f"{what} failed hub-side (correlation_id={correlation_id}): "
            f"{reply.error_code}: {reply.error_message or 'no details'}"
        )
    if not reply.success and reply.error_code == _NOT_DECLARED:
        raise ObjectStoreNotDeclaredError(_NOT_DECLARED, reply.error_message or "no details")
    if not reply.success:
        raise ObjectStoreRequestRefusedError(reply.error_code or "UNKNOWN", reply.error_message or "no details")


async def declare_pod_object_store(
    nats_client: NatsClient,
    *,
    identity_token: str,
    timeout_seconds: float = DEFAULT_OBJECT_STORE_REQUEST_TIMEOUT_SECONDS,
) -> DeclaredObjectStore:
    """ask the hub to declare this pod's own Object Store and pointer bucket; idempotent.

    :param nats_client: this pod's connected NATS client
    :ptype nats_client: NatsClient
    :param identity_token: this pod's CURRENT hub identity token
    :ptype identity_token: str
    :param timeout_seconds: seconds to wait for the answer
    :ptype timeout_seconds: float
    :return: the declared buckets' names and the store's bound
    :rtype: DeclaredObjectStore
    :raises ObjectStoreRequestRefusedError: when the hub refuses with a non-retryable code
    :raises ObjectStoreRequestUnavailableError: on no token, a transport failure or timeout, a reply
        that does not decode or answers another request or leaves out the buckets, or the hub's
        ``DECLARE_FAILED``
    """
    if not identity_token:
        raise ObjectStoreRequestUnavailableError("an object store declare has no identity token to present")
    request = ObjectStoreDeclareRequest(identity_token=SecretStr(identity_token), correlation_id=uuid7())
    raw = await _ask(
        nats_client,
        subject=Subjects.hub_object_store_declare(),
        request=request,
        what="object store declare",
        timeout_seconds=timeout_seconds,
    )
    try:
        reply = ObjectStoreDeclareReply.model_validate_json(raw)
    except ValidationError as exc:
        raise ObjectStoreRequestUnavailableError(
            f"object store declare reply did not decode (correlation_id={request.correlation_id}): {exc}"
        ) from exc
    _check_reply(reply, correlation_id=request.correlation_id, what="object store declare")
    if reply.bucket is None or reply.pointers_bucket is None or reply.max_bytes is None:
        raise ObjectStoreRequestUnavailableError(
            f"object store declare reply does not name the buckets (correlation_id={request.correlation_id})"
        )
    declared = DeclaredObjectStore(
        bucket=reply.bucket, pointers_bucket=reply.pointers_bucket, max_bytes=reply.max_bytes
    )
    log.info(
        "pod object store declared",
        extra={"extra_data": {"bucket": declared.bucket, "pointers": declared.pointers_bucket}},
    )
    return declared


async def retire_pod_objects(
    nats_client: NatsClient,
    *,
    identity_token: str,
    names: Sequence[str],
    timeout_seconds: float = DEFAULT_OBJECT_STORE_REQUEST_TIMEOUT_SECONDS,
) -> RetiredObjects:
    """ask the hub to delete objects this pod no longer serves from its own bucket.

    :param nats_client: this pod's connected NATS client
    :ptype nats_client: NatsClient
    :param identity_token: this pod's CURRENT hub identity token
    :ptype identity_token: str
    :param names: the objects to delete; at most :data:`MAX_RETIRED_OBJECTS`
    :ptype names: Sequence[str]
    :param timeout_seconds: seconds to wait for the answer
    :ptype timeout_seconds: float
    :return: what the hub did
    :rtype: RetiredObjects
    :raises ObjectStoreNotDeclaredError: when the pod's bucket is gone; declare it, then retry
    :raises ObjectStoreRequestRefusedError: when the hub refuses with a non-retryable code, or
        ``INVALID_REQUEST`` without asking when ``names`` breaks the request's bounds
    :raises ObjectStoreRequestUnavailableError: on no token, a transport failure or timeout, a reply
        that does not decode or answers another request, or the hub's ``RETIRE_FAILED``
    """
    if not identity_token:
        raise ObjectStoreRequestUnavailableError("an object retire has no identity token to present")
    try:
        request = ObjectStoreRetireRequest(
            identity_token=SecretStr(identity_token), correlation_id=uuid7(), names=list(names)
        )
    except ValidationError as exc:
        # the hub would refuse the same body, so it is refused here with the hub's code
        raise ObjectStoreRequestRefusedError("INVALID_REQUEST", f"object retire is not valid: {exc}") from exc
    raw = await _ask(
        nats_client,
        subject=Subjects.hub_object_store_retire(),
        request=request,
        what="object retire",
        timeout_seconds=timeout_seconds,
    )
    try:
        reply = ObjectStoreRetireReply.model_validate_json(raw)
    except ValidationError as exc:
        raise ObjectStoreRequestUnavailableError(
            f"object retire reply did not decode (correlation_id={request.correlation_id}): {exc}"
        ) from exc
    _check_reply(reply, correlation_id=request.correlation_id, what="object retire")
    retired = RetiredObjects(
        retired=reply.retired or 0, absent=reply.absent or 0, orphan_chunks=reply.orphan_chunks or 0
    )
    log.info(
        "pod objects retired",
        extra={
            "extra_data": {"retired": retired.retired, "absent": retired.absent, "orphan_chunks": retired.orphan_chunks}
        },
    )
    return retired


class PodObjectStore:
    """a tool pod's own Object Store and pointer bucket, bound, with the two asks the pod makes of the hub.

    What a :class:`~threetears.core.collections.scoped_snapshot.ScopedSnapshot` takes as its
    ``store``, ``pointers``, ``ensure_buckets`` and ``retire``. Built by :func:`bind_pod_object_store`.

    :param nats_client: the pod's connected NATS client
    :ptype nats_client: NatsClient
    :param identity_token: the pod's CURRENT identity token, read at each ask (a token rotates)
    :ptype identity_token: Callable[[], str]
    :param declared: what the hub declared
    :ptype declared: DeclaredObjectStore
    :param store: the bound Object Store
    :ptype store: NatsObjectStore
    :param pointers: the bound pointer bucket
    :ptype pointers: NatsKvBucket
    """

    def __init__(
        self,
        nats_client: NatsClient,
        identity_token: Callable[[], str],
        declared: DeclaredObjectStore,
        store: NatsObjectStore,
        pointers: NatsKvBucket,
    ) -> None:
        self._client = nats_client
        self._token = identity_token
        self.declared = declared
        self.store = store
        self.pointers = pointers

    async def declare(self) -> None:
        """ask the hub to declare both buckets again, after NATS lost them; idempotent.

        :return: nothing
        :rtype: None
        :raises ObjectStoreRequestError: when the hub refuses or does not answer
        """
        self.declared = await declare_pod_object_store(self._client, identity_token=self._token())

    async def retire(self, names: list[str]) -> RetiredObjects:
        """ask the hub to delete objects the pod no longer serves; declares the bucket again first
        when the hub says NATS lost it.

        :param names: the objects, at most :data:`MAX_RETIRED_OBJECTS`
        :ptype names: list[str]
        :return: what the hub did
        :rtype: RetiredObjects
        :raises ObjectStoreRequestError: when the hub refuses or does not answer
        """
        try:
            retired = await retire_pod_objects(self._client, identity_token=self._token(), names=names)
        except ObjectStoreNotDeclaredError:
            await self.declare()
            retired = await retire_pod_objects(self._client, identity_token=self._token(), names=names)
        return retired


async def bind_pod_object_store(nats_client: NatsClient, *, identity_token: Callable[[], str]) -> PodObjectStore:
    """ask the hub to declare this pod's own Object Store and pointer bucket, then bind both.

    The pod's registry row must opt in (``aibots tool-pod set-object-store POD --on``), or the hub
    refuses with ``OBJECT_STORE_NOT_GRANTED``.

    :param nats_client: this pod's connected NATS client
    :ptype nats_client: NatsClient
    :param identity_token: this pod's CURRENT hub identity token, read at each ask
    :ptype identity_token: Callable[[], str]
    :return: the bound buckets and the pod's asks
    :rtype: PodObjectStore
    :raises ObjectStoreRequestError: when the hub refuses or does not answer
    :raises ObjectStoreError: when the Object Store cannot be bound
    :raises KvError: when the pointer bucket cannot be bound
    """
    declared = await declare_pod_object_store(nats_client, identity_token=identity_token())
    store = await nats_client.object_store(name=declared.bucket, prefix_namespace=False)
    # the KV bind layers the namespace itself; the hub answers the full name
    prefix = f"{nats_client.namespace}-"
    if not declared.pointers_bucket.startswith(prefix):
        raise ObjectStoreRequestUnavailableError(
            f"the hub declared pointer bucket {declared.pointers_bucket!r} outside this pod's namespace {prefix!r}"
        )
    pointers = await nats_client.kv_bucket(name=declared.pointers_bucket.removeprefix(prefix), create_if_missing=False)
    return PodObjectStore(nats_client, identity_token, declared, store, pointers)
