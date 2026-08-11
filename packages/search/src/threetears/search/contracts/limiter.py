"""RateLimiterPort -- pacing before the call, keyed ``(provider instance, egress)`` (D8).

D8's ruling is "pace, don't just react", and it names two mechanisms for
one shape: an in-process limiter shipped in this leaf
(:mod:`threetears.search.limiter`, on by default with safe rates per
SR-L6), and a distributed limiter injected by hosts that have a bus. This
module declares the shape; neither implementation lives here.

**The key is the pair, and both halves are required.** Two SearXNG
deployments are two instances, banned and paced separately (SR-N4), and
which exit a request leaves by is what an upstream actually sees, so a
budget keyed on the provider alone paces the wrong thing (D20, SR-N2).
``direct`` is a named egress value, never an absence
(:data:`~threetears.search.contracts.provenance.EGRESS_DIRECT`), which is
why :meth:`RateLimiterPort.acquire` gives ``egress`` no default: an
omitted egress would silently collapse every exit onto one bucket. Both
halves are keyword-only because two adjacent strings are exactly the
argument pair a positional call eventually swaps.

**Shaped so a thin adapter over ``core``'s NATS ``TokenBucket`` satisfies
it.** That primitive's ``claim(key, *, tokens, max_wait_seconds)`` returns
a result the caller inspects and never raises for "not enough tokens", so
a host-side adapter is a key derivation from the pair plus a rename of
three fields -- no policy, no state. The same is true in the other
direction: this leaf must not import ``threetears.core`` (SR-L7), and
structural typing is what lets the sanctioned family primitive be the
distributed implementation anyway, exactly as
:class:`~threetears.search.contracts.transport.SearchTransport` does for
``TracedHttpClient``.

**Acquisition only, deliberately.** SR-H4's honest layering puts the
provider's own server-side limiter behind this one as the backstop that
covers non-cooperating deployments, so the port owes the caller a decision
and nothing else. A release/refund method is not declared: the Gate A
ruling on :class:`~threetears.search.contracts.transport.FetchTransport`
established that widening a protocol later retroactively invalidates every
implementation written against the narrower shape, so an obligation nobody
has yet needed is left out rather than guessed at -- if a refund becomes
load-bearing it arrives as its own protocol, and today's implementers stay
conformant.

Ports are parameters, never payload (SR-L4, P9): a limiter is injected at
construction and appears in no wire type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["RateLimitDecision", "RateLimiterPort"]


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """What a limiter said about one prospective call.

    A seam value, not a wire type -- a plain frozen dataclass, like
    :class:`~threetears.search.contracts.transport.TransportResponse`.

    A denial is a return value rather than an exception: on a paced path it
    is an expected, per-request outcome, and the caller's response to it
    (wait, fall back, refuse) is policy the limiter does not own.
    """

    #: whether the call may proceed now.
    acquired: bool
    #: estimated seconds until an equivalent acquisition would succeed;
    #: ``0.0`` when acquired. Feeds a caller's backoff and, where the
    #: caller gives up, the ``retry_after_seconds`` on the typed
    #: :class:`~threetears.search.contracts.errors.RateLimited` failure.
    retry_after_seconds: float = 0.0


@runtime_checkable
class RateLimiterPort(Protocol):
    """Structural protocol for the injected pacing seam (P9, D8).

    Implementations: this package's in-process token bucket (the default,
    SR-L6), or a host-side adapter over ``core``'s distributed
    ``TokenBucket`` where a bus exists.
    """

    async def acquire(
        self,
        *,
        provider_instance: str,
        egress: str,
        tokens: float = 1.0,
        max_wait_seconds: float = 0.0,
    ) -> RateLimitDecision:
        """Ask for permission to make one call against one paced key.

        :param provider_instance: the configured deployment about to be
            called -- an *instance* name, since two deployments of one
            product are paced separately (D8, SR-N4)
        :ptype provider_instance: str
        :param egress: the exit the call will leave by (D20);
            :data:`~threetears.search.contracts.provenance.EGRESS_DIRECT`
            for the default route, which is a value like any other
        :ptype egress: str
        :param tokens: how much of the key's allowance this call consumes
            -- one call is one token unless a provider weights its
            requests (SR-E4's weighted units have a pacing analogue)
        :ptype tokens: float
        :param max_wait_seconds: how long the caller will block for
            permission; ``0.0`` asks for a single fail-fast answer. A
            caller under a deadline passes what remains of it (SR-G2)
        :ptype max_wait_seconds: float
        :return: the decision; the caller proceeds only when
            :attr:`RateLimitDecision.acquired` is true
        :rtype: RateLimitDecision
        """
        ...
