"""a request, from whoever mints a principal's credential, that the principal renew it now.

A connection's permissions are fixed when it is admitted: nats-server applies the user JWT the
auth-callout minted, and nothing changes a live connection's grant. So when what a principal may
reach changes -- a grant added or withdrawn -- the connection that is open keeps the old grant until
it is replaced. With a long credential that could be a day.

The minter asks instead: it publishes a :class:`CredentialRenewalRequest` to the principal's own
inbox subtree (:meth:`threetears.nats.Subjects.credential_renewal_request`), which the principal is
always granted, and a client armed for it (:meth:`threetears.nats.NatsClient.renew_on_request`)
renews at once -- make-before-break, so nothing in flight is lost -- and its successor is admitted
with the grant as it stands now.

A request is a hint, never authority: renewing is always safe, and a principal that ignores one
keeps its old grant until its next renewal. Taking access AWAY is not done this way -- that is a
kick of the connection and a refused reconnect.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

__all__ = [
    "CREDENTIAL_RENEWAL_SUBJECT_TOKEN",
    "CredentialRenewalReason",
    "CredentialRenewalRequest",
]

#: the token under a principal's inbox prefix a renewal request is published on.
CREDENTIAL_RENEWAL_SUBJECT_TOKEN = "credential-renew"


class CredentialRenewalReason(StrEnum):
    """why a principal is asked to renew now. The wire value is stable."""

    #: what the principal may reach changed since its connection was admitted
    GRANTS_CHANGED = "grants_changed"


class CredentialRenewalRequest(BaseModel):
    """a request that a principal's runner renew its connection now.

    :param reason: why
    :ptype reason: CredentialRenewalReason
    :param pod_id: the pod-session asked, when only one runner of the principal is -- a principal's
        inbox may be shared by several (every pod of one agent); ``None`` asks every runner
    :ptype pod_id: str | None
    """

    reason: CredentialRenewalReason
    pod_id: str | None = None
