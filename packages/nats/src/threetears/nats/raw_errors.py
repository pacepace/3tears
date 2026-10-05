"""classify the exceptions a RAW nats-py handle raises, without the caller importing nats-py.

Most consumers never see a nats-py exception: every method on
:class:`threetears.nats.NatsClient` and :class:`threetears.nats.NatsKvBucket` translates them into
the typed hierarchy in :mod:`threetears.nats.errors`. A few hold a raw nats-py handle on purpose --
a ``KeyValue`` reached through :meth:`threetears.nats.NatsClient.jetstream_context`, the wrapper's
documented escape hatch -- and their failures arrive untranslated. Those consumers keep ``nats.*``
imports out of their own production code by enforcement, and were left matching nats-py's
exception class NAMES as strings, which survives a nats-py rename only by luck.

These predicates are that match done properly: by type, inside 3tears, which owns the nats-py
dependency. None of them reads message text.

Resolved lazily from :mod:`threetears.nats` like every other nats-py-backed name there, so a
process that only uses the L1 tier never loads the client to import the package.
"""

from __future__ import annotations

from typing import Final

from nats.errors import Error as _NatsError
from nats.js.errors import (
    BucketNotFoundError as _NatsBucketNotFoundError,
    KeyDeletedError as _NatsKeyDeletedError,
    KeyNotFoundError as _NatsKeyNotFoundError,
    NoStreamResponseError as _NatsNoStreamResponseError,
    NotFoundError as _NatsNotFoundError,
)

from threetears.nats.errors import KvBucketNotFoundError

__all__ = [
    "JS_ERR_STREAM_NOT_FOUND",
    "is_bucket_not_found",
    "is_key_not_found",
    "is_nats_error",
]

#: JetStream API error code for "stream not found" (``JSStreamNotFoundErr``). A KV bucket IS a
#: stream, so a JetStream API call naming the bucket's stream that the server answers with this code
#: says the bucket does not exist. nats-py raises it as a bare :class:`nats.js.errors.NotFoundError`
#: carrying the code, which is how it is told apart from the other 404s that share the class -- a
#: missing KEY is also a ``NotFoundError``, and carries no code.
JS_ERR_STREAM_NOT_FOUND: Final[int] = 10059


def is_nats_error(exc: BaseException) -> bool:
    """whether ``exc`` is one of nats-py's own exceptions: an instance of ``nats.errors.Error``.

    Every failure a raw nats-py handle reports derives from it -- a stream the broker lost, a
    request deadline, a closed connection, a JetStream API refusal. A consumer that treats "the bus
    failed" one way and "my own code failed" another asks this rather than catching ``Exception``.
    A 3tears wrapper error (:class:`threetears.nats.NatsClientError`) is NOT a nats-py error and
    answers ``False``: the wrapper already translated it.

    :param exc: the exception to classify
    :ptype exc: BaseException
    :return: ``True`` for a nats-py exception, ``False`` for anything else
    :rtype: bool
    """
    return isinstance(exc, _NatsError)


def is_bucket_not_found(exc: BaseException) -> bool:
    """whether ``exc`` says a KV bucket does not exist -- the server ANSWERED that its stream is absent.

    ``True`` for:

    - :class:`threetears.nats.KvBucketNotFoundError`, the wrapper's own typed error, so one
      predicate serves a consumer that holds both kinds of handle;
    - nats-py's ``BucketNotFoundError``, which ``JetStreamContext.key_value`` raises when binding a
      bucket whose stream does not exist;
    - nats-py's ``NoStreamResponseError``, which a JetStream publish -- every KV ``put``,
      ``create``, ``update`` and ``delete`` -- raises when no stream captures the subject, which
      for ``$KV.{bucket}.{key}`` means the bucket's stream is gone;
    - nats-py's ``NotFoundError`` carrying :data:`JS_ERR_STREAM_NOT_FOUND`, which any JetStream API
      call naming the stream (``STREAM.INFO``, ``CONSUMER.CREATE``, ``STREAM.UPDATE``) raises for an
      absent one.

    ``False`` for everything else, deliberately including:

    - a missing or deleted KEY (``KeyNotFoundError``), which is a ``NotFoundError`` too but says
      the bucket exists -- see :func:`is_key_not_found`;
    - a deadline. A request this principal is not granted is never answered, so a refused bucket
      and an unreachable broker both arrive as a timeout, and neither says the bucket is absent;
    - a core ``NoRespondersError``. A direct get against a vanished bucket arrives as one, but so
      does every request to any subject nobody serves, so the error alone does not say a bucket
      is missing. The wrapper's own handle resolves that case by binding the bucket again, which
      answers the question; a raw handle has to do the same.

    :param exc: the exception to classify
    :ptype exc: BaseException
    :return: ``True`` when the exception says the bucket does not exist
    :rtype: bool
    """
    absent_stream = isinstance(exc, _NatsNotFoundError) and getattr(exc, "err_code", None) == JS_ERR_STREAM_NOT_FOUND
    return absent_stream or isinstance(
        exc, (KvBucketNotFoundError, _NatsBucketNotFoundError, _NatsNoStreamResponseError)
    )


def is_key_not_found(exc: BaseException) -> bool:
    """whether ``exc`` says a KV KEY is absent from a bucket that exists.

    ``True`` for nats-py's ``KeyNotFoundError`` -- what ``KeyValue.get`` raises for a key that was
    never written and, since it folds the one into the other, for a key whose latest message is a
    delete or purge marker -- and for ``KeyDeletedError``, which the lower-level reads raise for that
    second case. A raw-handle reader treats both as "no value", which is what the wrapper's own
    :meth:`threetears.nats.NatsKvBucket.get` reports as ``None``.

    ``False`` for an absent BUCKET (:func:`is_bucket_not_found`), even though nats-py raises both
    through the same ``NotFoundError`` base: a missing key is a normal answer, a missing bucket is a
    failure, and a reader that conflated them would report a wiped bucket as an empty one.

    :param exc: the exception to classify
    :ptype exc: BaseException
    :return: ``True`` when the exception says the key is absent
    :rtype: bool
    """
    return isinstance(exc, (_NatsKeyNotFoundError, _NatsKeyDeletedError))
