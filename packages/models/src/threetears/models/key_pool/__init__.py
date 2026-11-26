"""A pool of 1..n credentials for one provider, each inside a budget, with sticky rotation.

Requirements: ``docs/model-key-pool-requirements.md``.
"""

from __future__ import annotations

from threetears.models.key_pool.classify import KeyErrorClassifier, Verdict, VerdictKind, VerdictScope, classifier_for
from threetears.models.key_pool.errors import NoKeyAvailableError, NoOutputLimitError
from threetears.models.key_pool.estimate import reservation_estimate
from threetears.models.key_pool.observe import KeyPoolCounters, KeyPoolObserver, MoveReason
from threetears.models.key_pool.period import period_bounds, period_id
from threetears.models.key_pool.pool import KeyPool, KeyPoolLease
from threetears.models.key_pool.secret import KeySpec, Secret
from threetears.models.key_pool.selection import KeyCandidate, SelectionPolicy, StickySelection
from threetears.models.key_pool.status import KeyPoolStatus, KeyState, KeyStatus
from threetears.models.key_pool.store import (
    AccountSnapshot,
    InMemoryKeyPoolStore,
    KeyPoolStore,
    KeyPoolStoreError,
    ReservationAlreadyOpen,
    ReservationNotFound,
    ReservationSnapshot,
)

__all__ = [
    "AccountSnapshot",
    "InMemoryKeyPoolStore",
    "KeyCandidate",
    "KeyErrorClassifier",
    "KeyPool",
    "KeyPoolCounters",
    "KeyPoolLease",
    "KeyPoolObserver",
    "KeyPoolStatus",
    "KeyPoolStore",
    "KeyPoolStoreError",
    "KeySpec",
    "KeyState",
    "KeyStatus",
    "MoveReason",
    "NoKeyAvailableError",
    "NoOutputLimitError",
    "ReservationAlreadyOpen",
    "ReservationNotFound",
    "ReservationSnapshot",
    "Secret",
    "SelectionPolicy",
    "StickySelection",
    "Verdict",
    "VerdictKind",
    "VerdictScope",
    "classifier_for",
    "period_bounds",
    "period_id",
    "reservation_estimate",
]
