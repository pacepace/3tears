"""a credential renewal the auth-callout refused on purpose, told to the connection it refused.

When the auth-callout denies a connection, nats-server tells the client nothing but
``-ERR 'Authorization Violation'``: ``client.authViolation`` sends that fixed text "regardless of the
authErr override", and the reason the callout gave (``AuthorizationResponse.error``) reaches only the
server's log. A pod renewing its credential therefore cannot tell a DELIBERATE refusal -- its
identity was superseded by a newer runner of the same pod-session, and it must stop serving now --
from a callout that was merely unreachable or slow, after which it should keep its still-valid
connection and try again.

So a deliberate refusal is also SENT, as a typed :class:`CredentialRefusal`, over the connection the
pod still holds: the responder publishes it to the refused principal's own inbox subtree
(:meth:`threetears.nats.Subjects.credential_refusal`), which that principal is always granted, and
the pod closes its connections when a refusal names it
(:meth:`threetears.nats.NatsClient.abandon_on_refusal`). Anything else -- a timeout, an unreachable
callout, an error -- produces no refusal, and the renewal keeps retrying on the old connection until
its own credential expires.

A resolver asks for this by returning a :class:`RefusedPrincipal` instead of ``None``
(:class:`threetears.nats.PrincipalResolver`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

__all__ = [
    "CREDENTIAL_REFUSAL_SUBJECT_TOKEN",
    "CredentialRefusal",
    "CredentialRefusalReason",
    "RefusedPrincipal",
]

#: the token under a principal's inbox prefix a refusal is published on.
CREDENTIAL_REFUSAL_SUBJECT_TOKEN = "credential-refused"


class CredentialRefusalReason(StrEnum):
    """why the auth-callout refused a connection on purpose. the wire value is stable."""

    #: a newer runner of the same pod-session handshook, so this runner's identity generation is
    #: stale: it is a zombie and must stop serving.
    SUPERSEDED = "superseded"


class CredentialRefusal(BaseModel):
    """a deliberate refusal, as published to the refused principal.

    :param reason: why the connection was refused
    :ptype reason: CredentialRefusalReason
    :param pod_id: the pod-session whose connection was refused; the principal's inbox may be shared
        by several pods (every pod of one agent), so each checks this is its own
    :ptype pod_id: str
    :param identity_generation: the generation the refused connection presented; a runner holding a
        different generation is not the one refused
    :ptype identity_generation: str
    """

    reason: CredentialRefusalReason
    pod_id: str
    identity_generation: str


@dataclass(frozen=True, slots=True)
class RefusedPrincipal:
    """a resolver's DELIBERATE denial: deny the connection, and tell the principal why.

    :param inbox_prefix: the refused principal's inbox prefix, under which the refusal is published
    :ptype inbox_prefix: str
    :param refusal: what to tell it
    :ptype refusal: CredentialRefusal
    """

    inbox_prefix: str
    refusal: CredentialRefusal
