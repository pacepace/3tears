"""a KV bucket's live keys by name, with no consumer and no bucket handle: for a caller that holds only the client.

:meth:`NatsKvBucket.list_keys <threetears.nats.kv.NatsKvBucket.list_keys>` lists a bucket its caller has opened. A
caller sweeping buckets it never opened (a restore flushing every bucket, a migration copying keys) holds only the
client, and nats-py's own ``KeyValue.keys`` replays the bucket through a consumer, which stalls part way under load
(:mod:`threetears.nats._stream_listing`). :func:`live_kv_keys` is the same listing ``list_keys`` makes, by bucket name:
the keys are the stream's subjects from ``STREAM.INFO`` (paged past the server's cap on one answer), and each key's
latest message is read by a subject-carried ``DIRECT.GET``; a key whose latest message is a delete or purge marker is
left out. Both verbs are a bucket grant's own (bind, and a read by subject), so the listing needs no consumer grant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from threetears.nats._stream_listing import latest_messages, stream_subjects
from threetears.nats.errors import KvBucketNotFoundError, KvError
from threetears.nats.raw_errors import is_bucket_not_found
from threetears.nats.subject_permissions import kv_stream_name

if TYPE_CHECKING:
    from threetears.nats.client import NatsClient

__all__ = ["live_kv_keys"]

#: the ceiling on each request a listing makes (a ``STREAM.INFO`` page, a key's read)
_DEFAULT_TIMEOUT_SECONDS: Final[float] = 10.0

#: the ``KV-Operation`` header values that mark a removed key: its latest message is the marker, not a value
_REMOVAL_MARKERS: Final[frozenset[str]] = frozenset({"DEL", "PURGE"})


async def live_kv_keys(
    client: NatsClient, bucket: str, *, timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
) -> list[str]:
    """every live key of ``bucket``, in stream order, with no consumer.

    :param client: the connected client
    :ptype client: NatsClient
    :param bucket: the bucket's full name, as the server names it (``KV_<bucket>`` is its stream)
    :ptype bucket: str
    :param timeout_seconds: the ceiling on each request
    :ptype timeout_seconds: float
    :return: the keys whose latest message is a value
    :rtype: list[str]
    :raises KvBucketNotFoundError: when the server answered that the bucket's stream does not exist
    :raises KvError: when the server refused the ``STREAM.INFO`` request or did not answer it, or a page did not
        decode
    :raises Exception: whatever a key's read raised other than the key having no message (an unanswered read, as an
        ungranted one is, a closed connection): a read that fails is never taken for an absent key
    """
    stream = kv_stream_name(bucket)
    prefix = f"$KV.{bucket}."

    def failure(message: str, cause: BaseException | None) -> Exception:
        return KvError(message)

    try:
        subjects = await stream_subjects(
            client, stream=stream, subjects_filter=f"{prefix}>", timeout_seconds=timeout_seconds, failure=failure
        )
    except KvError:
        raise
    except (
        Exception
    ) as exc:  # prawduct:allow prawduct/broad-except -- re-raised as the listing's typed error, cause chained
        if is_bucket_not_found(exc):
            raise KvBucketNotFoundError(
                f"KV bucket {bucket} does not exist: its stream {stream} is gone", bucket=bucket
            ) from exc
        # what the server answered instead of the stream's state (a refused request), or an unanswered request
        raise KvError(
            f"listing the keys of {bucket} was refused or not answered: STREAM.INFO of {stream}: {exc}"
        ) from exc
    messages = await latest_messages(client, stream=stream, subjects=subjects, timeout_seconds=timeout_seconds)
    return [
        str(message.subject)[len(prefix) :]
        for message in messages
        if (message.headers or {}).get("KV-Operation") not in _REMOVAL_MARKERS
    ]
