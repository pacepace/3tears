"""the one issue-time check every single-use proof verifier shares, and the two numbers it uses.

A proof carries a signed issue time (``iat``). A verifier accepts it only when that time is close
enough to the verifier's own clock, and "close enough" is two different questions:

- **How OLD may it be?** A proof is minted, travels, queues behind other work, and is verified.
  The past window is how long a legitimately slow request has to arrive
  (:data:`DEFAULT_PROOF_MAX_AGE`). It is also how long a captured proof stays worth replaying,
  which is why every verifier records the proof's nonce in a replay guard for at least that long.
- **How far AHEAD may it be?** Only as far as the signer's clock can honestly lead the verifier's
  (:data:`ISSUE_TIME_FUTURE_TOLERANCE`). Nothing legitimate is stamped later than its signer's
  clock, so the future side absorbs clock disagreement and nothing else.

They were one symmetric number once, and the future side inherited the past side's full minute.
That minute was not free. A :class:`~threetears.core.coordination.replay_guard.ReplayGuard` keeps
nonces in memory-backed NATS KV, a broker restart wipes it, and after a wipe the guard must refuse
every artifact its verifier could have accepted before the wipe -- which reaches exactly as far
past the new bucket's creation time as the verifier accepts an issue time ahead of its clock, plus
the guard's own clock-drift allowance. A verifier that accepted a minute ahead made every broker
restart cost 65 seconds of refused logins and tool calls. Accepting five seconds ahead makes it
ten.

So the two directions are separate parameters on every verifier, and the future one has a single
owner here. A verifier passes its future tolerance to its guard's
:meth:`~threetears.core.coordination.replay_guard.ReplayGuard.require_covers`, which fails loudly
when the guard was sized for less.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Final

__all__ = ["DEFAULT_PROOF_MAX_AGE", "ISSUE_TIME_FUTURE_TOLERANCE", "issue_time_is_fresh"]

#: how far AHEAD of a verifier's clock a proof's signed issue time may be and still be accepted.
#:
#: The platform's clock-agreement requirement, stated as a number: every host that signs a proof
#: and every host that verifies one must agree to within this. Measured on a live cluster, sixteen
#: pods -- hub, registry, gateway, identity, agents, tool pods and all three NATS brokers -- agreed
#: to within about one second, the error of the measurement itself.
#:
#: The trade-off runs both ways, which is why it is one named number and not a default each
#: verifier picks:
#:
#: - **Every second of it is a second of refused traffic after a broker restart.** A replay guard
#:   refuses, after its bucket is wiped, for this long plus its own drift allowance
#:   (:data:`~threetears.core.coordination.replay_guard.CLOCK_DRIFT_ALLOWANCE`): ten seconds at
#:   this value, where the old symmetric minute made it 65.
#: - **A signer whose clock leads a verifier's by more than this is refused outright**, every
#:   time, as a freshness failure. That is a host with a broken clock, and the refusal is the
#:   signal; widening this to accommodate it buys back the outage above for everyone else.
#:
#: A verifier may accept less than this and never more. Zero is a real cost, not a safe default:
#: an issue time is a whole second, so a signer a fraction of a second ahead of the verifier can
#: stamp a second the verifier has not reached, and refusing that makes a call succeed or fail on
#: sub-second timing.
ISSUE_TIME_FUTURE_TOLERANCE: Final[timedelta] = timedelta(seconds=5)

#: how OLD a proof's signed issue time may be and still be accepted, by default.
#:
#: The time a legitimately slow request has to arrive. Independent of
#: :data:`ISSUE_TIME_FUTURE_TOLERANCE`: lengthening it costs replay-guard memory (a nonce is
#: remembered for the whole accept window) and nothing after a broker restart.
DEFAULT_PROOF_MAX_AGE: Final[timedelta] = timedelta(seconds=60)


def issue_time_is_fresh(issued_at: int, *, now: int, max_age: timedelta, future_tolerance: timedelta) -> bool:
    """whether a signed issue time is close enough to the verifier's clock, each direction on its own bound.

    Accepts ``issued_at`` from ``now - max_age`` to ``now + future_tolerance``, both edges
    included. The caller supplies ``now`` so the verifier's clock is read once, by the verifier.

    :param issued_at: the proof's signed issue time, unix seconds
    :ptype issued_at: int
    :param now: the verifier's clock, unix seconds
    :ptype now: int
    :param max_age: how far behind ``now`` the issue time may be
    :ptype max_age: timedelta
    :param future_tolerance: how far ahead of ``now`` the issue time may be; what the verifier's
        replay guard must be sized for
    :ptype future_tolerance: timedelta
    :return: ``True`` when the issue time is inside the window
    :rtype: bool
    :raises ValueError: when either bound is negative -- a wiring error, not a proof failure
    """
    if max_age < timedelta(0):
        raise ValueError(f"max_age must not be negative, got {max_age}")
    if future_tolerance < timedelta(0):
        raise ValueError(f"future_tolerance must not be negative, got {future_tolerance}")
    lead_seconds = issued_at - now
    return -max_age.total_seconds() <= lead_seconds <= future_tolerance.total_seconds()
