"""One vocabulary for eviction across the cache tiers: the SQLite L1, the shared-memory Arrow store, the NATS L2.

**Why one.** Each tier holds what it may let go (a cache, refilled from the tier below) beside what it must keep
whole (durable state, or what a reader treats as complete), under a budget. Named once here, the policy, the words
and the metrics read the same in code, docs and panels for every tier (Pace, 2026-11-15: "use the same terms for L1
and L2"); no tier invents its own.

**The policy** (:class:`Policy`), declared per table, entry kind or bucket with a one-line reason
(:class:`Declared`) through :func:`policy_declare` as the tier's module is imported, and never defaulted: a kind set with an
undeclared kind fails the import, and a lookup of one is refused (:func:`policy_lookup`). :func:`policies_declared` lists a tier's.

- ``EVICTABLE``: a cache; dropping it is a miss that refills from the tier below. Evicted least recently used first.
- ``KEEP``: must stay complete, or is durable state; never evicted. It counts first against the budget; a write that
  does not fit beside it is a refusal, and ``KEEP`` alone passing the budget is a sizing failure (counted and
  logged), never an eviction.

**The words**, everywhere: budget (the bytes a tier may hold), held (bytes held now), kept (``KEEP`` bytes),
evictable (``EVICTABLE`` bytes), eviction, evicted bytes, hit, miss, refusal, sizing failure.

**The metrics**, one family labelled by ``tier`` (:class:`Tier`) and ``name`` (the table, store or bucket: declared
ones only, so the cardinality is bounded; ``""`` for a tier-wide value): ``threetears_cache_budget_bytes``,
``threetears_cache_held_bytes`` and ``threetears_cache_held_entries`` (each with ``policy``; entries are rows for
the L1, entries for shared memory, keys for the L2), ``threetears_cache_hits_total``, ``threetears_cache_misses_total``,
``threetears_cache_evictions_total``, ``threetears_cache_evicted_bytes_total``, ``threetears_cache_refusals_total``,
``threetears_cache_sizing_failures_total``. Metrics only: alerting is configured in Grafana from them.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from threetears.observe.metrics import counter, gauge

__all__ = [
    "BUDGET_BYTES",
    "EVICTED_AGE_SECONDS",
    "EVICTED_BYTES",
    "EVICTIONS",
    "HELD_BYTES",
    "HELD_ENTRIES",
    "HITS",
    "MISSES",
    "REFUSALS",
    "SIZING_FAILURES",
    "Declared",
    "Policy",
    "Tier",
    "policy_declare",
    "policies_declared",
    "policy_lookup",
]


class Policy(enum.StrEnum):
    """whether a tier may let a kind of entry go to make room."""

    #: a cache: dropping it is a miss that refills from the tier below; evicted least recently used first
    EVICTABLE = "evictable"
    #: must stay complete, or is durable state: never evicted; counts first against the budget
    KEEP = "keep"


class Tier(enum.StrEnum):
    """the cache tiers, the ``tier`` label of every metric in the family."""

    L1 = "l1"
    SHARED = "shared"
    L2 = "l2"


@dataclass(frozen=True)
class Declared:
    """one kind's declared policy and why.

    :ivar policy: the policy
    :ivar why: the reason, one line, never empty
    :raises ValueError: when the reason is empty
    """

    policy: Policy
    why: str

    def __post_init__(self) -> None:
        if not self.why.strip():
            raise ValueError("an eviction policy is declared with its reason")

    @property
    def evictable(self) -> bool:
        """whether the kind may be evicted.

        :return: True for ``EVICTABLE``
        :rtype: bool
        """
        return self.policy is Policy.EVICTABLE


class _UndeclaredError(LookupError):
    """a kind of entry with no declared eviction policy: there is no default."""


#: each tier's declared kinds, by kind name: what :func:`policies_declared` reads, written by :func:`policy_declare` as each tier's
#: module is imported (code, never data: a restart declares them again)
_DECLARED: dict[Tier, Mapping[str, Declared]] = {}


def policy_declare[K: enum.Enum](tier: Tier, kinds: type[K], policies: Mapping[K, Declared]) -> Mapping[str, Declared]:
    """declare a tier's kinds of entry and their policies, refusing a kind with none: called once per kind set, as
    the tier's module is imported, so an undeclared kind fails the import, never a write at night.

    :param tier: the tier
    :ptype tier: Tier
    :param kinds: every kind the tier holds (an enum: its members are the kinds)
    :ptype kinds: type[K]
    :param policies: each kind's policy
    :ptype policies: Mapping[K, Declared]
    :return: the tier's declarations now, by kind name (``kind.value``)
    :rtype: Mapping[str, Declared]
    :raises LookupError: when a kind has no policy
    :raises ValueError: when a policy names a kind that is not one of ``kinds``
    """
    missing = [kind for kind in kinds if kind not in policies]
    if missing:
        raise _UndeclaredError(f"{tier} kinds declare no eviction policy (EVICTABLE or KEEP, with why): {missing}")
    stray = [kind for kind in policies if not isinstance(kind, kinds)]
    if stray:
        raise ValueError(f"{tier} policies name kinds that are not {kinds.__name__}: {stray}")
    merged = {**_DECLARED.get(tier, {}), **{str(kind.value): each for kind, each in policies.items()}}
    _DECLARED[tier] = MappingProxyType(merged)
    return _DECLARED[tier]


def policies_declared(tier: Tier) -> Mapping[str, Declared]:
    """a tier's declared kinds and their policies, by kind name: what a test lists, and what a tier reads its own
    policy from (:func:`policy_lookup`).

    :param tier: the tier
    :ptype tier: Tier
    :return: kind name -> its policy; empty when the tier declared none
    :rtype: Mapping[str, Declared]
    """
    return _DECLARED.get(tier, MappingProxyType({}))


def policy_lookup(tier: Tier, kind: str) -> Declared:
    """a kind's declared policy, refusing one with none (no default: "never" by omission would hide a cache, and
    "evictable" by omission would drop durable state).

    :param tier: the tier
    :ptype tier: Tier
    :param kind: the kind's name
    :ptype kind: str
    :return: its policy
    :rtype: Declared
    :raises LookupError: when it has none
    """
    try:
        return policies_declared(tier)[kind]
    except KeyError:
        raise _UndeclaredError(
            f"{tier} kind {kind!r} declares no eviction policy (EVICTABLE or KEEP, with why)"
        ) from None


#: the labels every metric of the family carries
_LABELS: Final = ("tier", "name")

BUDGET_BYTES = gauge(
    "threetears_cache_budget_bytes", description="the bytes a cache tier may hold", label_names=_LABELS
)
HELD_BYTES = gauge(
    "threetears_cache_held_bytes",
    description="the bytes a cache tier holds now, by policy: kept (KEEP) and evictable (EVICTABLE)",
    label_names=(*_LABELS, "policy"),
)
HELD_ENTRIES = gauge(
    "threetears_cache_held_entries",
    description="the entries a cache tier holds now, by policy (rows for l1, entries for shared, keys for l2)",
    label_names=(*_LABELS, "policy"),
)
HITS = counter(
    "threetears_cache_hits", description="reads a cache tier answered from what it held", label_names=_LABELS
)
MISSES = counter(
    "threetears_cache_misses", description="reads a cache tier refilled from the tier below", label_names=_LABELS
)
EVICTIONS = counter(
    "threetears_cache_evictions",
    description="EVICTABLE entries a cache tier let go, least recently used first, to make room",
    label_names=_LABELS,
)
EVICTED_BYTES = counter(
    "threetears_cache_evicted_bytes", description="bytes a cache tier's evictions freed", label_names=_LABELS
)
EVICTED_AGE_SECONDS = gauge(
    "threetears_cache_evicted_age_seconds",
    description="how long before its eviction the last entry a cache tier evicted was last used: small means it "
    "evicts what is still in use (too small a budget)",
    label_names=_LABELS,
)
REFUSALS = counter(
    "threetears_cache_refusals",
    description="writes a cache tier refused because what it must KEEP left no room, nothing evictable left",
    label_names=_LABELS,
)
SIZING_FAILURES = counter(
    "threetears_cache_sizing_failures",
    description="times what a cache tier must KEEP alone passed its budget: a sizing failure, never an eviction",
    label_names=_LABELS,
)
