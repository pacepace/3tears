"""Learning that a message did not arrive.

An SMTP send is fire-and-forget. The relay accepts the message, the conversation ends,
and nothing on the outbound path can ever report a bounce -- so a product that models a
`bounced` outcome has no way to populate it from :mod:`threetears.channels.mail.smtp`
alone. The provider tells us afterwards, over HTTP, and this module is the platform side
of that callback: verify the signature, turn the body into
:class:`DeliveryEvent` values, hand each to the product.

**Why not :class:`~threetears.channels.webhook.WebhookReceiver` end to end.** That
receiver resolves a `webhook_subscriptions` row, decrypts a per-subscription secret and
dispatches a `WakeTrigger` into a conversation for an agent to act on. A delivery report
has no conversation, no agent and no user -- exactly the thread-shaped mismatch that
rules `ChannelDeliveryMessage` out for an email recipient in the first place. What this
DOES reuse is that receiver's verification contract in full: the same :data:`Verifier`
callable shape, the same canonical :func:`verify_generic_hmac_sha256`, the same default
signature header and the same body-size cap. A second HMAC implementation was already
found and removed once in this package; there is not going to be a third.

**Requires the ``webhook`` extra**, for the same two reasons that receiver does: fastapi
for the mount, and `3tears-agent-wake` for the canonical verifier.

**The secret is a reference, resolved per callback.** Same discipline as the send path:
a rotated shared secret takes effect on the next callback rather than the next deploy,
and nothing holds the value between requests.

**A verifier for a specific provider is NOT shipped here, and that is deliberate rather
than missing.** The default scheme is the platform's own `sha256=<hex>` HMAC, which is
what a product's own relay or gateway signs with. Real providers do not use it: SendGrid
signs its event webhook with ECDSA, and SES arrives as an SNS notification signed with
RSA against a certificate fetched from a URL that itself has to be validated. Both are
:meth:`BounceReceiver.register_verifier` implementations with a real signature-scheme
decision behind them, and neither is guessed at here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from threetears.core.security.secret_refs import resolve_secret
from threetears.observe import get_logger

from threetears.channels.webhook import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    DEFAULT_SIGNATURE_HEADER,
    Verifier,
    verify_generic_hmac_sha256,
)

__all__ = [
    "DEFAULT_MAX_PAYLOAD_BYTES",
    "DEFAULT_SIGNATURE_HEADER",
    "BounceIngestResult",
    "BounceReceiver",
    "DeliveryEvent",
    "DeliveryEventHandler",
    "DeliveryEventParser",
    "DeliveryEventType",
]

log = get_logger(__name__)


class DeliveryEventType(StrEnum):
    """What a provider is reporting about one message to one recipient.

    Hard and soft bounces are separate values because the policies they drive are
    opposite: a hard bounce means the address does not exist and suppresses immediately,
    while a soft one is a full mailbox or a temporary refusal that is retried and only
    suppresses if it persists. A single `bounced` value cannot express either.

    `complained` is a spam report, which is a suppression event and NOT a bounce -- the
    address works perfectly and the person does not want the mail. Collapsing it into a
    bounce both loses an opt-out and corrupts the bounce rate a sender's reputation is
    measured on.
    """

    DELIVERED = "delivered"
    BOUNCED_HARD = "bounced_hard"
    BOUNCED_SOFT = "bounced_soft"
    COMPLAINED = "complained"
    DEFERRED = "deferred"
    UNSUBSCRIBED = "unsubscribed"


@dataclass(frozen=True, slots=True)
class DeliveryEvent:
    """One thing that happened to one message after it left this platform.

    Deliberately does NOT carry the provider's raw payload. That payload routinely
    quotes the message headers and sometimes the body, so retaining it turns a delivery
    report into a copy of the mail; a product that needs a field this type lacks should
    have its parser lift that field, not the whole envelope.

    :ivar event_type: what the provider is reporting
    :ivar recipient: the address the event concerns
    :ivar date_occurred: when the provider says it happened, not when we received it
    :ivar provider_message_id: the provider's id for the message, when it supplies one;
        this is what correlates the event back to a send
    :ivar provider_event_id: the provider's id for this event, for deduplication across
        redeliveries
    :ivar diagnostic_code: the relay's own explanation, when there is one
    """

    event_type: DeliveryEventType
    recipient: str
    date_occurred: datetime
    provider_message_id: str | None = None
    provider_event_id: str | None = None
    diagnostic_code: str | None = None


@runtime_checkable
class DeliveryEventParser(Protocol):
    """Turns one provider's callback body into platform delivery events.

    Provider-specific by nature -- every provider names its event types differently and
    batches them differently -- so this is a seam rather than an implementation. It is
    synchronous because it is pure parsing; anything needing IO belongs in the handler.
    """

    def parse(self, payload: bytes) -> Sequence[DeliveryEvent]:
        """Parse a verified callback body.

        :param payload: the raw, already-verified request body
        :ptype payload: bytes
        :return: the events the body describes, possibly none
        :rtype: Sequence[DeliveryEvent]
        :raises Exception: the body was not something this parser understands
        """
        ...


@runtime_checkable
class DeliveryEventHandler(Protocol):
    """What the product does with one delivery event.

    MUST be idempotent. A provider redelivers whenever it does not get a 2xx, and this
    receiver returns a non-2xx if the handler raises partway through a batch, so an
    event already handled will arrive again.
    """

    async def __call__(self, event: DeliveryEvent) -> None:
        """Act on one event.

        :param event: the event to act on
        :ptype event: DeliveryEvent
        :return: nothing
        :rtype: None
        """
        ...


@dataclass(frozen=True, slots=True)
class BounceIngestResult:
    """The outcome of one callback, in the terms the provider understands.

    The status code is the whole contract with a provider: a 2xx tells it the events are
    durably ours and it may stop, anything else tells it to resend. That is why a
    handler failure is a 500 rather than a logged warning and a 200.

    :ivar status_code: HTTP status to answer the callback with
    :ivar accepted: events handed to the handler successfully
    :ivar message: short diagnostic, carrying no address and no payload
    """

    status_code: int
    accepted: int
    message: str


class BounceReceiver:
    """HMAC-verified ingest for a provider's delivery-report callback.

    Every dependency is a constructor argument and there is no global state, matching
    :class:`~threetears.channels.webhook.WebhookReceiver`, so one process can mount
    several receivers -- a deployment sending through two relays has two secrets and two
    parsers.
    """

    def __init__(
        self,
        *,
        secret_ref: str,
        parser: DeliveryEventParser,
        handler: DeliveryEventHandler,
        verification_scheme: str = "generic_hmac_sha256",
        signature_header: str = DEFAULT_SIGNATURE_HEADER,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        """
        :param secret_ref: `scheme://locator` reference to the shared signing secret,
            resolved on every callback rather than held
        :ptype secret_ref: str
        :param parser: turns a verified body into events
        :ptype parser: DeliveryEventParser
        :param handler: the product's per-event action
        :ptype handler: DeliveryEventHandler
        :param verification_scheme: which registered verifier to use
        :ptype verification_scheme: str
        :param signature_header: header carrying the signature; defaults to
            :class:`~threetears.channels.webhook.WebhookReceiver`'s own, so a deployment
            running both does not have two conventions
        :ptype signature_header: str
        :param max_payload_bytes: bodies larger than this are refused before the HMAC,
            since the whole body must sit in memory for a constant-time compare
        :ptype max_payload_bytes: int
        :return: nothing
        :rtype: None
        """
        self.secret_ref = secret_ref
        self.parser = parser
        self.handler = handler
        self.verification_scheme = verification_scheme
        self.signature_header = signature_header
        self.max_payload_bytes = max_payload_bytes
        self._verifiers: dict[str, Verifier] = {"generic_hmac_sha256": verify_generic_hmac_sha256}

    def register_verifier(self, scheme: str, verifier: Verifier) -> None:
        """Register or replace a signature-verification scheme.

        A provider's own scheme -- SendGrid's ECDSA, SES-over-SNS's RSA -- plugs in here
        without modifying this module, exactly as vendor schemes do on
        :meth:`~threetears.channels.webhook.WebhookReceiver.register_verifier`. An
        implementation MUST compare in constant time.

        :param scheme: scheme name, matched against `verification_scheme`
        :ptype scheme: str
        :param verifier: callable with the :data:`~threetears.channels.webhook.Verifier`
            signature
        :ptype verifier: Verifier
        :return: nothing
        :rtype: None
        """
        self._verifiers[scheme] = verifier

    async def handle_payload(self, body: bytes, signature: str | None) -> BounceIngestResult:
        """Verify one callback body and dispatch the events it carries.

        Refusals happen in a deliberate order: size before HMAC (the compare needs the
        whole body in memory), signature before parse (an unverified body is not read at
        all), and parse before dispatch (a body we cannot read is the provider's problem
        and a 400 stops it retrying the same bytes forever).

        :param body: the raw request body
        :ptype body: bytes
        :param signature: the value of the configured signature header, if present
        :ptype signature: str | None
        :return: the status to answer with and how many events were accepted
        :rtype: BounceIngestResult
        """
        if len(body) > self.max_payload_bytes:
            return BounceIngestResult(status_code=413, accepted=0, message="payload too large")
        if not signature:
            return BounceIngestResult(status_code=401, accepted=0, message="missing signature header")
        verifier = self._verifiers.get(self.verification_scheme)
        if verifier is None:
            log.warning(
                "mail bounce receiver: unknown verification scheme",
                extra={"extra_data": {"scheme": self.verification_scheme}},
            )
            return BounceIngestResult(
                status_code=400, accepted=0, message=f"unknown verification scheme: {self.verification_scheme}"
            )
        try:
            secret = resolve_secret(self.secret_ref).get_secret_value()
        except Exception as exc:
            # Fail CLOSED. A receiver that cannot verify must not accept, because
            # "accepted" is precisely what tells the provider to stop resending -- so
            # admitting here would silently discard every bounce for as long as the
            # secret backend is unwell.
            log.warning(
                "mail bounce receiver: could not resolve the signing secret",
                extra={"extra_data": {"error_type": type(exc).__name__, "secret_ref": self.secret_ref}},
            )
            return BounceIngestResult(status_code=500, accepted=0, message="signing secret unavailable")
        if not verifier(secret.encode("utf-8"), body, signature):
            log.info(
                "mail bounce receiver: signature verification failed",
                extra={"extra_data": {"scheme": self.verification_scheme}},
            )
            return BounceIngestResult(status_code=403, accepted=0, message="invalid signature")
        try:
            events = self.parser.parse(body)
        except Exception as exc:
            log.warning(
                "mail bounce receiver: could not parse a verified callback",
                extra={"extra_data": {"error_type": type(exc).__name__}},
            )
            return BounceIngestResult(status_code=400, accepted=0, message="unparseable callback body")
        accepted = 0
        for event in events:
            try:
                await self.handler(event)
            except Exception as exc:
                # Never the exception text: a handler failing to write a suppression row
                # routinely names the address it was writing, and an address is personal
                # data. The type is what stays diagnostic.
                log.warning(
                    "mail bounce receiver: handler failed, asking the provider to resend",
                    extra={"extra_data": {"error_type": type(exc).__name__, "accepted_before_failure": accepted}},
                )
                return BounceIngestResult(
                    status_code=500,
                    accepted=accepted,
                    message=f"handler failed ({type(exc).__name__}); redelivery expected",
                )
            accepted += 1
        return BounceIngestResult(status_code=202, accepted=accepted, message="accepted")

    async def _handle(self, request: Request) -> Response:
        """FastAPI route handler -- the HTTP boundary around :meth:`handle_payload`.

        :param request: the inbound request
        :ptype request: Request
        :return: JSON response carrying the accepted count and a diagnostic message
        :rtype: Response
        """
        body = await request.body()
        result = await self.handle_payload(body, request.headers.get(self.signature_header))
        return JSONResponse(
            status_code=result.status_code,
            content={"accepted": result.accepted, "message": result.message},
        )

    def register(self, app: FastAPI, *, path: str = "/webhooks/mail/bounce") -> None:
        """Mount this receiver as a ``POST`` route on a FastAPI app.

        The path carries no subscription id, unlike
        :meth:`~threetears.channels.webhook.WebhookReceiver.register`: there is one mail
        provider per deployment and no per-subscription row to look up, so a mount point
        per receiver is the whole addressing scheme.

        :param app: FastAPI app to mount on
        :ptype app: FastAPI
        :param path: URL the provider is configured to POST to
        :ptype path: str
        :return: nothing
        :rtype: None
        """
        app.add_api_route(path, self._handle, methods=["POST"], tags=["webhooks"])
