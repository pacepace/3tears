"""one ask of the hub on a forwarded-token subject: sent, decoded, matched to its request, classified.

Every pod -> hub request that presents an ``identity_token`` speaks the same protocol: the request
carries a ``correlation_id``; the reply is a model with ``success``, ``correlation_id``,
``error_code`` and ``error_message``; a hub-side failure is retryable, any other refusal is final.
:func:`ask_hub` is that protocol once, so each request module keeps only its models, its subject and
what its own success must carry.

**The correlation rule.** A reply answers this request when it carries this request's id. A refusal
carrying NO id is this request's too: a hub that could not decode the body had none to echo, and
taking its ``INVALID_REQUEST`` for a stray would retry a request the hub refuses every time. A
success with no id, or any reply under another id (a late answer to an earlier request, or a hub
bug), is not an answer to this request.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from datetime import timedelta
from typing import TYPE_CHECKING, Protocol, TypeVar, cast
from uuid import UUID

from pydantic import BaseModel, ValidationError

from threetears.nats.errors import RequestError
from threetears.nats.subjects import Subject

if TYPE_CHECKING:
    from threetears.nats.client import NatsClient

__all__ = ["HubReply", "HubRequest", "ask_hub"]


class HubRequest(Protocol):
    """a request to the hub: a model that serialises itself and names its correlation id."""

    @property
    def correlation_id(self) -> UUID:
        """the id the reply echoes."""
        ...

    def model_dump_json(self) -> str:
        """the request's wire body."""
        ...


class HubReply(Protocol):
    """what every hub reply carries, besides what its own success carries."""

    @property
    def success(self) -> bool:
        """whether the hub did what was asked."""
        ...

    @property
    def correlation_id(self) -> UUID | None:
        """the request's id, echoed."""
        ...

    @property
    def error_code(self) -> str | None:
        """the hub's refusal code."""
        ...

    @property
    def error_message(self) -> str | None:
        """the hub's description of a refusal."""
        ...


_ReplyT = TypeVar("_ReplyT", bound=BaseModel)


def _answers(reply: HubReply, correlation_id: UUID) -> bool:
    """whether ``reply`` answers the request with ``correlation_id`` (the module's correlation rule).

    :param reply: the decoded reply
    :ptype reply: HubReply
    :param correlation_id: the request's id
    :ptype correlation_id: UUID
    :return: True when it carries the id, or is a refusal carrying none
    :rtype: bool
    """
    return reply.correlation_id == correlation_id or (not reply.success and reply.correlation_id is None)


async def ask_hub(
    nats_client: NatsClient,
    *,
    subject: Subject,
    request: HubRequest,
    reply_type: type[_ReplyT],
    what: str,
    timeout_seconds: float,
    unavailable: Callable[[str], Exception],
    refused: Callable[[_ReplyT], Exception],
    retryable: Collection[str] = (),
) -> _ReplyT:
    """send ``request``, and return the hub's reply once it is a success that answers it.

    :param nats_client: the pod's connected NATS client
    :ptype nats_client: NatsClient
    :param subject: the request subject
    :ptype subject: Subject
    :param request: the request
    :ptype request: HubRequest
    :param reply_type: the reply model
    :ptype reply_type: type[_ReplyT]
    :param what: what is asked, for the errors
    :ptype what: str
    :param timeout_seconds: seconds to wait for the answer
    :ptype timeout_seconds: float
    :param unavailable: builds the error raised when there is no usable answer
    :ptype unavailable: Callable[[str], Exception]
    :param refused: builds the error raised for a final refusal, from the reply
    :ptype refused: Callable[[_ReplyT], Exception]
    :param retryable: the refusal codes that are hub-side failures, raised as ``unavailable``
    :ptype retryable: Collection[str]
    :return: the reply, a success answering this request
    :rtype: _ReplyT
    :raises Exception: ``unavailable(...)`` on a transport failure or timeout, a reply that does not
        decode or does not answer this request, or a retryable code; ``refused(reply)`` on any other
        refusal
    """
    correlation_id = request.correlation_id
    try:
        raw: bytes = await nats_client.request_raw(
            subject=subject,
            payload=request.model_dump_json().encode("utf-8"),
            timeout=timedelta(seconds=timeout_seconds),
        )
    except RequestError as exc:
        raise unavailable(f"{what} failed (correlation_id={correlation_id}): {exc}") from exc
    try:
        reply = reply_type.model_validate_json(raw)
    except ValidationError as exc:
        raise unavailable(f"{what} reply did not decode (correlation_id={correlation_id}): {exc}") from exc
    # every hub reply model carries the protocol's fields
    checked = cast("HubReply", reply)
    if not _answers(checked, correlation_id):
        raise unavailable(f"{what} reply carried correlation_id={checked.correlation_id}, not {correlation_id}")
    if not checked.success and checked.error_code in retryable:
        raise unavailable(
            f"{what} failed hub-side (correlation_id={correlation_id}): "
            f"{checked.error_code}: {checked.error_message or 'no details'}"
        )
    if not checked.success:
        raise refused(reply)
    return reply
