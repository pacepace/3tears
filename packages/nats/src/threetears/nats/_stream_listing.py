"""list a stream's subjects, and read each one's latest message, with no consumer: the listings KV and the Object Store share.

A listing that replays a growing stream through a consumer (:mod:`threetears.nats._named_read`) reads one pushed
message per key or object, and under load the push stalls part way: on cobalt dev on 2026-10-30 every one of six
scoped-snapshot retirements failed so, replaying 2345 object metadata messages (``stalled after N of M``). Here a
listing is request and reply only:

- :func:`stream_subjects` -- the stream's subjects matching a filter, each with its message count, from
  ``STREAM.INFO`` (one answer carries at most the server's cap of subjects; past it the next page is asked for by
  ``offset``, which nats-py's ``stream_info`` cannot ask, so that page is a request of its own);
- :func:`latest_messages` -- each subject's latest message by a subject-carried ``DIRECT.GET``, a bounded number
  in flight at once.

Both are verbs every bucket grant already carries (bind, and a read by subject), so a listing needs no consumer
grant at all.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final

from nats.js.errors import APIError, NotFoundError

from threetears.nats._publish import run_bounded
from threetears.nats.subjects import Subject

if TYPE_CHECKING:
    from nats.js.api import RawStreamMsg

    from threetears.nats.client import NatsClient

__all__ = ["latest_messages", "stream_subjects"]

#: the JetStream API prefix every listing request goes to
_JS_API: Final[str] = "$JS.API"

#: the most subjects one ``STREAM.INFO`` answer carries (the server's ``JSMaxSubjectDetails``); an answer this full
#: may have been cut short, and the next page is asked for
STREAM_INFO_SUBJECTS_CAP: Final[int] = 100_000

#: how many ``DIRECT.GET`` requests one listing keeps in flight at once
_READS_IN_FLIGHT: Final[int] = 32


async def stream_subjects(
    client: NatsClient,
    *,
    stream: str,
    subjects_filter: str,
    timeout_seconds: float,
    failure: Callable[[str, BaseException | None], Exception],
) -> dict[str, int]:
    """every subject of ``stream`` matching ``subjects_filter``, each with the messages held on it.

    One ``STREAM.INFO``, as nats-py asks it. An answer as full as the server's cap on subjects
    (``JSMaxSubjectDetails``) may have been cut short, so the next page is asked for by ``offset`` until one comes
    back short.

    :param client: the connected wrapper client
    :ptype client: NatsClient
    :param stream: the stream
    :ptype stream: str
    :param subjects_filter: the subjects to list
    :ptype subjects_filter: str
    :param timeout_seconds: the ceiling on each page's request
    :ptype timeout_seconds: float
    :param failure: builds the caller's error from a message and its cause
    :ptype failure: Callable[[str, BaseException | None], Exception]
    :return: subject -> messages held on it
    :rtype: dict[str, int]
    :raises nats.js.errors.APIError: what the server answered instead of the stream's state (a ``NotFoundError``
        carrying ``JS_ERR_STREAM_NOT_FOUND`` for a stream that is gone)
    :raises Exception: ``failure(...)`` when a page does not decode; whatever an unanswered request raised
    """
    js = client.jetstream_context()
    info = await run_bounded(
        lambda: js.stream_info(stream, subjects_filter=subjects_filter),
        timeout=timeout_seconds,
        what=f"STREAM.INFO of {stream}",
    )
    page = {str(subject): int(count) for subject, count in (info.state.subjects or {}).items()}
    found = dict(page)
    offset = len(page)
    while len(page) >= STREAM_INFO_SUBJECTS_CAP:
        answer = await _stream_info_page(
            client, stream=stream, subjects_filter=subjects_filter, offset=offset, timeout_seconds=timeout_seconds
        )
        page = _page_subjects(answer, stream=stream, failure=failure)
        found.update(page)
        offset += len(page)
    return found


async def _stream_info_page(
    client: NatsClient, *, stream: str, subjects_filter: str, offset: int, timeout_seconds: float
) -> dict[str, Any]:
    """one ``STREAM.INFO`` answer, its subjects from ``offset``.

    :param client: the connected wrapper client
    :ptype client: NatsClient
    :param stream: the stream
    :ptype stream: str
    :param subjects_filter: the subjects to list
    :ptype subjects_filter: str
    :param offset: the first subject of the page
    :ptype offset: int
    :param timeout_seconds: the ceiling on the request
    :ptype timeout_seconds: float
    :return: the decoded answer
    :rtype: dict[str, Any]
    :raises nats.js.errors.APIError: what the server answered instead of the stream's state
    :raises Exception: whatever the request raised (unanswered, a closed connection)
    """
    request = json.dumps({"subjects_filter": subjects_filter, "offset": offset}).encode("utf-8")
    raw = await client.request_raw(
        subject=Subject.raw(f"{_JS_API}.STREAM.INFO.{stream}"),
        payload=request,
        timeout=timedelta(seconds=timeout_seconds),
    )
    page: dict[str, Any] = json.loads(raw)
    error = page.get("error")
    if error is not None:
        APIError.from_error(error)
    return page


def _page_subjects(
    page: dict[str, Any], *, stream: str, failure: Callable[[str, BaseException | None], Exception]
) -> dict[str, int]:
    """the subjects one ``STREAM.INFO`` page carries.

    :param page: the decoded answer
    :ptype page: dict[str, Any]
    :param stream: the stream, for the error
    :ptype stream: str
    :param failure: builds the caller's error from a message and its cause
    :ptype failure: Callable[[str, BaseException | None], Exception]
    :return: subject -> messages held on it
    :rtype: dict[str, int]
    :raises Exception: ``failure(...)`` when the page's subjects do not decode
    """
    subjects = (page.get("state") or {}).get("subjects") or {}
    if not isinstance(subjects, dict):
        raise failure(f"STREAM.INFO of {stream} answered subjects that are not a mapping", None)
    return {str(subject): int(count) for subject, count in subjects.items()}


async def latest_messages(
    client: NatsClient,
    *,
    stream: str,
    subjects: Iterable[str],
    timeout_seconds: float,
) -> list[RawStreamMsg]:
    """each subject's latest message, by a subject-carried ``DIRECT.GET``, in stream order.

    A subject whose messages were all removed since it was listed has none, and is left out: the listing then
    meets its absence the way it meets any other race.

    :param client: the connected wrapper client
    :ptype client: NatsClient
    :param stream: the stream (it must run ``allow_direct``, as every bucket the platform declares does)
    :ptype stream: str
    :param subjects: the subjects
    :ptype subjects: Iterable[str]
    :param timeout_seconds: the ceiling on each read
    :ptype timeout_seconds: float
    :return: the messages, by stream sequence
    :rtype: list[nats.js.api.RawStreamMsg]
    :raises Exception: whatever a read raised other than the subject having no message (unanswered: an
        ungranted read is never answered)
    """
    js = client.jetstream_context()
    limit = asyncio.Semaphore(_READS_IN_FLIGHT)

    async def _read(subject: str) -> RawStreamMsg | None:
        async with limit:
            try:
                message: RawStreamMsg = await run_bounded(
                    lambda: js.get_last_msg(stream, subject, direct=True),
                    timeout=timeout_seconds,
                    what=f"direct read of {subject}",
                )
            except NotFoundError:
                # NOSILENT: the subject's messages went after it was listed; a listing leaves it out
                return None
            return message

    # every read runs to its own end (each is bounded), so a failure leaves none of them running behind the caller
    read = await asyncio.gather(*(_read(subject) for subject in subjects), return_exceptions=True)
    failed = next((outcome for outcome in read if isinstance(outcome, BaseException)), None)
    if failed is not None:
        raise failed
    return sorted(
        (message for message in read if message is not None and not isinstance(message, BaseException)),
        key=lambda message: int(message.seq or 0),
    )
