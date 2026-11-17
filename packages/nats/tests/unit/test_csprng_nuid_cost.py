"""a NATS request's token costs about one read of the system's CSPRNG, and every digit is equally likely.

Every request nats-py sends (a KV get, a JetStream API call) names its reply with a fresh token from
:class:`~threetears.nats._nats_py_internals.CsprngNuid`. Drawn a digit at a time with ``secrets.choice`` it took 22
CSPRNG reads, about 17 us, a third of the CPU a follower spent applying one walked grant change, where the keyed
read it applies takes about 75 us (IT3, 2026-11-16).
"""

from __future__ import annotations

import secrets
import string
import time
from collections import Counter

import pytest

from threetears.core.testing.calibrate import budget_problem
from threetears.core.testing.loop_lateness import quietly
from threetears.nats._nats_py_internals import NUID_LENGTH, CsprngNuid

#: tokens timed a run, and runs (the best kept)
_TOKENS = 2_000
_RUNS = 5

#: the most the host's own idle loop may be late while the two sides are timed
_HOST_MAY_BE_LATE = 0.020

#: the bytes the bare read draws: what a token needs, about 22.7 of them, rounded up as the token draws them
_BARE_BYTES = 32


def _best(work: object) -> float:
    best = float("inf")
    for _ in range(_RUNS):
        began = time.perf_counter()
        for _ in range(_TOKENS):
            work()  # type: ignore[operator]
        best = min(best, time.perf_counter() - began)
    return best


@pytest.mark.loop_timing
def test_a_token_costs_within_twice_one_csprng_read() -> None:
    """Flips if a token costs over twice one read of the system's CSPRNG of the bytes it needs (a digit drawn a
    read at a time, as before, is about 20x)."""
    nuid = CsprngNuid()
    bare, took = quietly(
        lambda: (_best(lambda: secrets.token_bytes(_BARE_BYTES)), _best(nuid.next)),
        host_may_be_late=_HOST_MAY_BE_LATE,
    )
    problem = budget_problem(f"{_TOKENS:,} request tokens", took, bare)
    assert problem is None, problem


def test_every_digit_is_equally_likely() -> None:
    """Flips if a token's digits are not uniform over base62 (a remainder taken over every byte value, unrejected,
    makes the first 8 digits 5/4 as likely as the rest): 40,000 tokens, each digit within 6% of its share."""
    nuid = CsprngNuid()
    counts: Counter[int] = Counter()
    for _ in range(40_000):
        counts.update(nuid.next())
    alphabet = (string.digits + string.ascii_uppercase + string.ascii_lowercase).encode()
    assert set(counts) == set(alphabet)
    share = 40_000 * NUID_LENGTH / len(alphabet)
    assert all(abs(count - share) < 0.06 * share for count in counts.values()), sorted(counts.values())
