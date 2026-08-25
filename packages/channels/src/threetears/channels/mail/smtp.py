"""Delivering mail over SMTP, configured from wherever the product keeps its settings.

Promoted from `identity_core/email/smtp.py`. Two of its decisions are worth restating
because they are load-bearing rather than incidental:

**stdlib `smtplib`, moved off the event loop, rather than an async SMTP dependency.**
`smtplib` blocks, and blocking here would stall every other request the process is
serving, so each send runs in a worker thread. Adding `aiosmtplib` was the alternative
and the trade was dependency surface: `asyncio.to_thread` is a complete answer at the
volumes this reaches. See :mod:`threetears.channels.mail.batch` for what happens when it
is not, and for the ceiling.

**No provider SDK.** A relay that speaks SMTP is SendGrid, SES, a self-hosted relay, and
whatever replaces them; changing provider becomes editing settings rather than shipping
a release. A vendor SDK buys templates, campaigns and analytics that this path does not
want, in exchange for a rewrite per provider.

**Every failure leaves as :class:`EmailSendError`.** Narrowing the catch to the SMTP,
OS and TLS families reads as more precise and was measured wrong in the original: a
header the stdlib refuses to fold raises `HeaderWriteError`, and a settings read is real
IO that can fail any way IO fails. Both would escape a caller that handles exactly one
type.

**What replaced identity-core's audit call.** The original published a NATS lifecycle
event on every failed send, because a send that fails is the event an operator most
needs and is least likely to notice -- nobody reports an email that never arrived until
long afterwards. That call named identity's own bus and event vocabulary, so the
promoted version takes an `on_failure` callback instead and identity passes its existing
publisher into it. The property survives; the coupling does not.
"""

from __future__ import annotations

import asyncio
import smtplib
import ssl
from collections.abc import Awaitable, Callable, Mapping
from email.message import EmailMessage as MimeMessage
from typing import Final

from threetears.observe import get_logger

from threetears.channels.mail.message import EmailMessage, EmailSendError
from threetears.channels.mail.settings import EmailSettingsResolver, ResolvedEmailSettings

__all__ = [
    "FailureRecorder",
    "SmtpAuthWithoutTlsError",
    "SmtpEmailTransport",
]

log = get_logger(__name__)

#: Applied to connect, TLS negotiation and each command. Without it a hung relay holds a
#: worker thread open indefinitely, and threads are the scarce resource on this path.
_SMTP_TIMEOUT_SECONDS: Final[float] = 15.0

#: Headers the transport decides. A caller that could set these could redirect a message
#: the transport believes it addressed elsewhere, or strip an unsubscribe affordance the
#: message is legally required to carry.
_RESERVED_HEADERS: Final[frozenset[str]] = frozenset(
    {"from", "to", "subject", "list-unsubscribe", "list-unsubscribe-post", "content-type"}
)

FailureRecorder = Callable[[Mapping[str, object]], Awaitable[None]]
"""Called with an already-redacted description of a send that failed.

Exists so a product can put failed sends somewhere durable -- identity-core publishes a
lifecycle event, another product might write a row -- without this package naming a bus.
It is called before the :class:`EmailSendError` is raised and can never replace it.
"""


class SmtpAuthWithoutTlsError(Exception):
    """A send was asked to authenticate over an unencrypted connection.

    The settings surface refuses this configuration, so this is the backstop for one
    that arrived another way -- a direct write, a restore from an older dump. Raised
    rather than logged-and-continued: the whole point is that the password must not
    leave the process.
    """


def _unsubscribe_header(message: EmailMessage) -> str | None:
    """Build the `List-Unsubscribe` value for `message`, or ``None``.

    RFC 2369 wraps each form in angle brackets and separates them with a comma; a
    mailbox provider that finds one turns its own unsubscribe button into a header
    rather than into a spam report, which is the whole reason bulk mail carries it.

    :param message: the message being sent
    :ptype message: EmailMessage
    :return: header value, or ``None`` when the message offers no unsubscribe route
    :rtype: str | None
    """
    forms: list[str] = []
    if message.list_unsubscribe_url is not None:
        forms.append(f"<{message.list_unsubscribe_url}>")
    if message.list_unsubscribe_mailto is not None:
        forms.append(f"<mailto:{message.list_unsubscribe_mailto}>")
    return ", ".join(forms) if forms else None


def _build_mime(message: EmailMessage, settings: ResolvedEmailSettings) -> MimeMessage:
    """The wire form of one message.

    Built with the stdlib's `EmailMessage` rather than string concatenation because
    header construction is where injection lives: a display name or subject carrying a
    newline would otherwise let a caller append headers of its own. `set_content` and
    header assignment handle the encoding and folding that prevents it.

    :param message: the message to render
    :ptype message: EmailMessage
    :param settings: the relay configuration this send resolved
    :ptype settings: ResolvedEmailSettings
    :return: the MIME message to hand to the relay
    :rtype: MimeMessage
    :raises ValueError: a caller-supplied header would overwrite one the transport owns
    """
    mime = MimeMessage()
    mime["From"] = (
        settings.from_address if settings.from_name is None else f"{settings.from_name} <{settings.from_address}>"
    )
    mime["To"] = message.to
    mime["Subject"] = message.subject
    mime.set_content(message.body_text)
    if message.body_html is not None:
        # The text part is added first and stays first, which is what the ordering rule
        # requires: a client renders the LAST part it understands, so text-then-html
        # gives a plain-text reader the text and everyone else the HTML. Reversed, no
        # HTML-capable client would ever show the HTML.
        mime.add_alternative(message.body_html, subtype="html")
    unsubscribe = _unsubscribe_header(message)
    if unsubscribe is not None:
        mime["List-Unsubscribe"] = unsubscribe
    if message.list_unsubscribe_url is not None and message.list_unsubscribe_url.startswith("https://"):
        # RFC 8058 one-click, which Gmail and Yahoo require of bulk senders. Advertised
        # only for an HTTPS URL, because the mechanism IS an HTTPS POST -- claiming it
        # alongside a `mailto:` alone promises something no client can perform.
        mime["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
    for name, value in message.headers.items():
        if name.lower() in _RESERVED_HEADERS:
            raise ValueError(
                f"header {name!r} is set by the transport and cannot be supplied by a caller: it "
                "decides where the message goes and how it is read."
            )
        mime[name] = value
    return mime


def _send_blocking(mime: MimeMessage, settings: ResolvedEmailSettings) -> None:
    """One synchronous SMTP conversation. Runs in a worker thread, never the event loop.

    STARTTLS is upgraded on an already-open plaintext connection, which is what port 587
    expects. A relay on 465 speaking implicit TLS is NOT supported and would need its
    own branch rather than a silent fallback to plaintext -- the same limitation the
    original carried, restated here rather than quietly inherited.

    :param mime: the message to hand over
    :ptype mime: MimeMessage
    :param settings: the relay configuration this send resolved
    :ptype settings: ResolvedEmailSettings
    :return: nothing
    :rtype: None
    :raises SmtpAuthWithoutTlsError: credentials were configured on a plaintext relay
    """
    context = ssl.create_default_context()
    with smtplib.SMTP(settings.host, settings.port, timeout=_SMTP_TIMEOUT_SECONDS) as client:
        if settings.use_starttls:
            client.starttls(context=context)
        if settings.username is not None and settings.password is not None:
            # Fail CLOSED rather than authenticate in clear. Sending the password anyway
            # would disclose this platform's own mail credential to anyone on the path,
            # and whoever holds it can send mail as us. Not sending is the lesser
            # failure, and it is a loud one.
            if not settings.use_starttls:
                raise SmtpAuthWithoutTlsError(
                    "refusing to authenticate to an SMTP relay over an unencrypted connection"
                )
            client.login(settings.username, settings.password)
        client.send_message(mime)


def _redact(text: str, *addresses: str | None) -> str:
    """`text` with every known address replaced by a placeholder.

    Exception text on this path routinely carries addresses: `HeaderWriteError` names
    the offending header VALUE and the header in question is `To`, and
    `SMTPRecipientsRefused` / `SMTPSenderRefused` carry addresses in `args`. An address
    is personal data, so the diagnostic value of the message is kept while the part
    identifying a person is not.

    Redacting the addresses this send already KNOWS is deliberate scoping: it cannot
    catch an address the relay volunteers about somebody else, so the exception TYPE is
    logged alongside rather than relying on this alone.

    :param text: the exception text to sanitise
    :ptype text: str
    :param addresses: addresses known to this send
    :ptype addresses: str | None
    :return: the text with each known address masked
    :rtype: str
    """
    redacted = text
    for address in addresses:
        if address:
            redacted = redacted.replace(address, "[address redacted]")
    return redacted


class SmtpEmailTransport:
    """An :class:`~threetears.channels.mail.message.EmailTransport` backed by SMTP."""

    def __init__(self, settings: EmailSettingsResolver, *, on_failure: FailureRecorder | None = None) -> None:
        """
        :param settings: resolves the relay configuration per send
        :ptype settings: EmailSettingsResolver
        :param on_failure: called with a redacted description of each failed send;
            ``None`` records nothing beyond the WARNING log line
        :ptype on_failure: FailureRecorder | None
        :return: nothing
        :rtype: None
        """
        self._settings = settings
        self._on_failure = on_failure

    async def _record_failure(self, details: Mapping[str, object]) -> None:
        """Record a failed send, and NEVER become the reason one is reported differently.

        This call sits between the failure and the `raise EmailSendError` its callers
        depend on, so anything escaping here would REPLACE that error -- pre-empting the
        very contract the broad catches exist to guarantee, and doing it only when
        something else is already unwell.

        Swallowed to a log rather than to silence: the send failure itself has already
        been logged at WARNING by the caller, so nothing goes unrecorded.

        :param details: the failure description, with addresses already redacted
        :ptype details: Mapping[str, object]
        :return: nothing
        :rtype: None
        """
        if self._on_failure is None:
            return
        try:
            await self._on_failure(details)
        except Exception:
            log.exception("could not record the outbound-email failure")

    async def send(self, message: EmailMessage) -> None:
        """Deliver `message`, or raise :class:`EmailSendError`.

        Settings are resolved HERE, on every call, rather than held from construction --
        see this module's docstring for why that is the point rather than an oversight.

        :param message: the message to deliver
        :ptype message: EmailMessage
        :return: nothing
        :rtype: None
        :raises EmailSendError: the settings are missing, unusable, or the relay refused
        """
        try:
            settings = await self._settings.resolve_for_send()
        except Exception as exc:
            # EVERY failure of this method is an `EmailSendError`. Narrowing this to the
            # settings-layer exception types was measured wrong in the original: a
            # wrong-but-present seal key raises the crypto layer's own type, and the
            # per-send read is real IO. Both would leave a caller with an unhandled
            # exception rather than a failed send.
            reason = _redact(str(exc), message.to)
            log.warning(
                "outbound email not sent -- settings unusable",
                extra={"extra_data": {"error_type": type(exc).__name__, "reason": reason}},
            )
            await self._record_failure({"error_type": type(exc).__name__, "reason": reason})
            raise EmailSendError(reason) from exc

        try:
            mime = _build_mime(message, settings)
            await asyncio.to_thread(_send_blocking, mime, settings)
        except Exception as exc:
            reason = _redact(str(exc), message.to, settings.from_address)
            log.warning(
                "outbound email send failed",
                extra={
                    "extra_data": {
                        "smtp_host": settings.host,
                        "smtp_port": settings.port,
                        # The TYPE as well as the reason: redaction can only mask the
                        # addresses this send knows, so the class name is what stays
                        # diagnostic if the relay volunteers one about somebody else.
                        "error_type": type(exc).__name__,
                        "error": reason,
                    }
                },
            )
            await self._record_failure(
                {
                    "smtp_host": settings.host,
                    "smtp_port": settings.port,
                    "error_type": type(exc).__name__,
                    "reason": reason,
                }
            )
            raise EmailSendError(f"SMTP send to {settings.host}:{settings.port} failed: {reason}") from exc
        log.info(
            "outbound email sent",
            extra={"extra_data": {"smtp_host": settings.host, "subject": message.subject}},
        )
