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
  The bucket TTL reaps storage and each entry carries its own expiry on top, so
  the per-call ``ttl`` means what the Protocol says it means and expiry is not
  something application code can forget.
- :mod:`threetears.iam.stores.postgres` -- for a service that already has a
  pool and wants the expiry predicate inside the claiming statement.
- :mod:`threetears.iam.stores.memory` -- in-memory, for tests. Shipped rather
  than left to each consumer, because a hand-rolled fake KV that quietly
  diverges from the real one is how a store bug reaches production green.

**All three honour the per-call ``ttl`` identically**, and that uniformity is
the point of shipping the double rather than describing it. The KV store once
recorded the argument and ignored it while the double enforced it, so a service
tested against the double shipped an expiry production did not have -- a double
disagreeing with production about a security property, which is the one thing a
double must never do.

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
