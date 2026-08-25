"""One outbound email, the transport that delivers it, and the one error it raises.

Promoted from `identity_core/email/mailer.py`, whose own module docstring set the
trigger: *"Not a new 3tears primitive -- revisit only if a second 3tears product needs
the same capability."* Survey fielding is that second product, so the implementation
moves here rather than being copied.

**Why not `ChannelDeliveryMessage`.** The channels protocol addresses a participant in a
conversation: it requires a `conversation_id` and an `agent_id` and routes to a thread.
A survey respondent has no platform presence at all -- no account, no conversation, no
agent -- so every one of those fields would be invented to satisfy the type. This is a
sibling shape in the same package, not a reuse of that one.

**What was added on promotion**, and why each is not scope creep:

- an optional HTML alternative, because a survey invitation is not a reset link;
- `List-Unsubscribe` / `List-Unsubscribe-Post`, which bulk survey mail legally needs and
  transactional mail never bothered with;
- caller-supplied headers, because a provider correlates its bounce callback to a send
  by a header the sender put there.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = [
    "EmailMessage",
    "EmailSendError",
    "EmailTransport",
    "Mailer",
]


@dataclass(frozen=True, slots=True)
class EmailMessage:
    """One outbound email.

    `body_text` is never optional, even when `body_html` is supplied: a recipient
    reading in plain text and every spam filter scoring the message both see the text
    part, so an HTML-only send is a deliverability decision nobody makes deliberately.

    :ivar to: single recipient address; one message addresses one person, so a batch is
        many messages rather than one message with many recipients (a shared `To`
        discloses every recipient to every other)
    :ivar subject: subject line, already substituted
    :ivar body_text: plain-text body, always present
    :ivar body_html: optional HTML alternative; when set, the wire form becomes
        `multipart/alternative` with the text part first
    :ivar list_unsubscribe_url: per-recipient unsubscribe URL, carried as
        `List-Unsubscribe`; an `https://` value also advertises RFC 8058 one-click
    :ivar list_unsubscribe_mailto: per-recipient unsubscribe address, carried in the
        same header alongside any URL
    :ivar headers: additional headers to set verbatim; envelope headers are refused
        by the transport rather than silently overwritten
    """

    to: str
    subject: str
    body_text: str
    body_html: str | None = None
    list_unsubscribe_url: str | None = None
    list_unsubscribe_mailto: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)


class EmailSendError(Exception):
    """Raised by an `EmailTransport` when a send genuinely fails.

    Promoted verbatim in intent from identity-core, including the discipline that makes
    it useful: EVERY failure of a transport's `send` leaves as this type. A caller then
    handles one exception rather than the union of smtplib, ssl, OS and stdlib-header
    errors, and the distinguishing detail lives in the log rather than in the type.

    It also carries a non-enumeration property the original depended on: a caller-facing
    request must never be able to tell "the mailbox does not exist" from "the mailer is
    down", so a caller catching this reports the same outcome for both.
    """


@runtime_checkable
class EmailTransport(Protocol):
    """The structural surface a `Mailer` needs.

    A `Protocol` rather than a base class so a unit test can pass a lightweight
    in-memory double, and so a future provider-API transport is a peer of the SMTP one
    rather than a subclass of it.
    """

    async def send(self, message: EmailMessage) -> None:
        """Deliver one message.

        :param message: the message to deliver
        :ptype message: EmailMessage
        :return: nothing
        :rtype: None
        :raises EmailSendError: the message was not delivered
        """
        ...


class Mailer:
    """The entry point a product sends through, decoupled from the concrete transport.

    Kept keyword-only and additive: identity-core's existing call sites pass exactly
    `to`, `subject` and `body_text`, and every field added on promotion defaults to
    `None`, so those calls compile and behave unchanged against this class.
    """

    def __init__(self, transport: EmailTransport) -> None:
        """
        :param transport: the transport every send is handed to
        :ptype transport: EmailTransport
        :return: nothing
        :rtype: None
        """
        self._transport = transport

    async def send(
        self,
        *,
        to: str,
        subject: str,
        body_text: str,
        body_html: str | None = None,
        list_unsubscribe_url: str | None = None,
        list_unsubscribe_mailto: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        """Send one message through the configured transport.

        :param to: recipient address
        :ptype to: str
        :param subject: subject line
        :ptype subject: str
        :param body_text: plain-text body
        :ptype body_text: str
        :param body_html: optional HTML alternative
        :ptype body_html: str | None
        :param list_unsubscribe_url: per-recipient unsubscribe URL
        :ptype list_unsubscribe_url: str | None
        :param list_unsubscribe_mailto: per-recipient unsubscribe address
        :ptype list_unsubscribe_mailto: str | None
        :param headers: additional headers to set verbatim
        :ptype headers: Mapping[str, str] | None
        :return: nothing
        :rtype: None
        :raises EmailSendError: the message was not delivered
        """
        await self._transport.send(
            EmailMessage(
                to=to,
                subject=subject,
                body_text=body_text,
                body_html=body_html,
                list_unsubscribe_url=list_unsubscribe_url,
                list_unsubscribe_mailto=list_unsubscribe_mailto,
                headers={} if headers is None else dict(headers),
            )
        )
