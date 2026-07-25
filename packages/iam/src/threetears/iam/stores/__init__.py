"""Storage seams for the short-lived state authentication needs.

Every flow in this package parks something small and temporary somewhere: an
authorization code between redirect and exchange, an OAuth ``state`` between
the two legs of a round trip, a password-reset ticket, a token handoff, a
failed-login counter. All of it is short-lived, all of it is TTL'd, and none of
it belongs in a table -- a row that must be swept by a cron job is a row that
eventually is not.

**This package owns no schema.** The two services it was factored out of
disagree about persistence in every way that matters, and unifying them would
have produced an abstraction neither could use. So state sits behind the
Protocols here, with implementations supplied rather than assumed:

- :mod:`threetears.iam.stores.nats_kv` -- JetStream KV, the production default.
  TTLs are the bucket's job, so expiry is not something application code can
  forget.
- :mod:`threetears.iam.stores.memory` -- in-memory, for tests. Shipped rather
  than left to each consumer, because a hand-rolled fake KV that quietly
  diverges from the real one is how a store bug reaches production green.

**Tickets are stored hashed, and redeemed atomically.** A ticket store holds
SHA-256 of the secret, never the secret, so a store dump is not a set of usable
password-reset links. Redemption is a single atomic claim rather than a
read-then-delete: two concurrent redemptions of one ticket must produce exactly
one winner, and a check-then-act would let both through.
"""

from __future__ import annotations

from threetears.iam.stores.base import (
    AttemptLimiter,
    AttemptWindow,
    SingleUseTicketStore,
    StateStore,
    TicketIssue,
    hash_ticket,
    new_ticket_secret,
)

__all__ = [
    "AttemptLimiter",
    "AttemptWindow",
    "SingleUseTicketStore",
    "StateStore",
    "TicketIssue",
    "hash_ticket",
    "new_ticket_secret",
]
