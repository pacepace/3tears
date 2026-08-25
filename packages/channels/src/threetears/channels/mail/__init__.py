"""Outbound email, and the delivery reports that come back.

Promoted from `14-eng-ai-bot-identity`'s `identity_core/email/`, whose own module
docstring named the condition for doing so: *"Not a new 3tears primitive -- revisit only
if a second 3tears product needs the same capability."* Survey fielding is the second
product, so the implementation moves here and identity-core consumes it rather than
keeping a copy.

The pieces, and where each is explained in full:

- :mod:`~threetears.channels.mail.message` -- the message, the transport protocol, and
  the single error every failure leaves as.
- :mod:`~threetears.channels.mail.settings` -- the resolver protocol, and why settings
  are read per send rather than held from startup.
- :mod:`~threetears.channels.mail.smtp` -- the SMTP transport, promoted whole.
- :mod:`~threetears.channels.mail.templating` -- per-recipient substitution, and why it
  is deliberately not a template language.
- :mod:`~threetears.channels.mail.batch` -- many recipients with per-recipient failure
  isolation, and the ceiling past which this is the wrong tool.
- :mod:`~threetears.channels.mail.bounce` -- the inbound delivery-report contract.
  Requires the ``webhook`` extra, so it is imported defensively below and is simply
  absent without it.

This is a sibling of the channel adapters rather than one of them: a recipient here has
no conversation, no agent and usually no account, so nothing in
:mod:`threetears.channels.protocol` fits and none of it is reused.
"""

from __future__ import annotations

from threetears.channels.mail.batch import (
    BatchSendResult,
    RecipientFailure,
    SendPacer,
    SendRateLimited,
    TokenBucketPacer,
    send_batch,
)
from threetears.channels.mail.message import (
    EmailMessage,
    EmailSendError,
    EmailTransport,
    Mailer,
)
from threetears.channels.mail.settings import (
    EmailAuthWithoutTlsError,
    EmailCredentialsIncompleteError,
    EmailSettingsNotConfiguredError,
    EmailSettingsResolver,
    ResolvedEmailSettings,
    StaticEmailSettingsResolver,
)
from threetears.channels.mail.smtp import (
    FailureRecorder,
    SmtpAuthWithoutTlsError,
    SmtpEmailTransport,
)
from threetears.channels.mail.templating import (
    EmailTemplate,
    TemplateRenderError,
)

__all__ = [
    "BatchSendResult",
    "EmailAuthWithoutTlsError",
    "EmailCredentialsIncompleteError",
    "EmailMessage",
    "EmailSendError",
    "EmailSettingsNotConfiguredError",
    "EmailSettingsResolver",
    "EmailTemplate",
    "EmailTransport",
    "FailureRecorder",
    "Mailer",
    "RecipientFailure",
    "ResolvedEmailSettings",
    "SendPacer",
    "SendRateLimited",
    "SmtpAuthWithoutTlsError",
    "SmtpEmailTransport",
    "StaticEmailSettingsResolver",
    "TemplateRenderError",
    "TokenBucketPacer",
    "send_batch",
]

try:
    from threetears.channels.mail.bounce import (  # noqa: F401
        BounceIngestResult,
        BounceReceiver,
        DeliveryEvent,
        DeliveryEventHandler,
        DeliveryEventParser,
        DeliveryEventType,
    )

    __all__.extend(
        [
            "BounceIngestResult",
            "BounceReceiver",
            "DeliveryEvent",
            "DeliveryEventHandler",
            "DeliveryEventParser",
            "DeliveryEventType",
        ]
    )
except ImportError as _exc:
    # Same guard the slack / discord / webhook adapters use one level up: a missing
    # extra and a genuinely broken module raise the same ImportError, and both end with
    # these names simply absent. Without the underlying error the two are
    # indistinguishable, and the symptom surfaces much later as an AttributeError at a
    # call site pointing nowhere near the cause.
    from threetears.observe import get_logger as _get_logger

    _get_logger(__name__).debug(
        "mail bounce ingest not loaded",
        extra={"extra_data": {"adapter": "BounceReceiver", "packaging_extra": "webhook", "error": str(_exc)}},
    )
