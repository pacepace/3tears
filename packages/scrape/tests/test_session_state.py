"""Sealing a human's solve, and refusing to use one that cannot be trusted.

These are live session credentials, so the tests here are mostly about the ways this is
allowed to fail. Every one of them has the same required answer -- ask for a human again --
and the point of testing them separately is that they arrive by different routes: a wrong key,
a tampered token, a format change, a missing expiry, a passed expiry, no key configured at
all. A single "it works" test would pass while any of those quietly sent a dead cookie or,
worse, leaked a live one.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.scrape.health import ScrapeTargetHealthCollection
from threetears.scrape.session_state import (
    DEFAULT_SESSION_STATE_TTL,
    SealedSessionState,
    open_session_state,
    record_session_state,
    seal_session_state,
    usable_session_state,
)

_KEY = SecretStr("an-operator-master-key-from-secret-refs")
_OTHER_KEY = SecretStr("a-different-operators-master-key")
_NOW = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
_T = "warn_oh"
_STATE = {
    "cookies": [{"name": "cf_clearance", "value": "the-thing-a-human-earned", "domain": ".example.gov"}],
    "origins": [],
}


@pytest.fixture()
def health() -> ScrapeTargetHealthCollection:
    return ScrapeTargetHealthCollection(
        CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS"), nats_client=None
    )


def test_a_sealed_state_round_trips() -> None:
    sealed = seal_session_state(_STATE, _KEY, now=_NOW)
    assert open_session_state(sealed.sealed, _KEY) == _STATE
    assert sealed.expires_at == _NOW + DEFAULT_SESSION_STATE_TTL


def test_the_ciphertext_does_not_contain_the_credential() -> None:
    """The obvious check, and the one whose absence would make every other test theatre."""
    sealed = seal_session_state(_STATE, _KEY, now=_NOW)
    assert "the-thing-a-human-earned" not in sealed.sealed
    assert "cf_clearance" not in sealed.sealed


def test_the_same_state_seals_to_different_tokens() -> None:
    """A fresh nonce per call, so equal secrets do not produce equal ciphertext.

    Deterministic sealing would let anyone with read access to the table tell which targets
    share a solve -- and tell when a solve changed -- without opening anything.
    """
    first = seal_session_state(_STATE, _KEY, now=_NOW)
    second = seal_session_state(_STATE, _KEY, now=_NOW)
    assert first.sealed != second.sealed
    assert open_session_state(first.sealed, _KEY) == open_session_state(second.sealed, _KEY)


def test_a_wrong_key_cannot_open_it() -> None:
    sealed = seal_session_state(_STATE, _KEY, now=_NOW)
    assert open_session_state(sealed.sealed, _OTHER_KEY) is None


def test_a_tampered_token_is_rejected_rather_than_partially_read() -> None:
    """GCM authenticates, so a flipped byte fails the tag rather than yielding altered state.

    The failure mode being excluded is the dangerous one: a decrypt that returns something
    plausible-but-modified would put an attacker-chosen cookie into a real fetch.
    """
    sealed = seal_session_state(_STATE, _KEY, now=_NOW)
    for cut in (1, len(sealed.sealed) // 2, len(sealed.sealed) - 2):
        flipped = sealed.sealed[:cut] + ("A" if sealed.sealed[cut] != "A" else "B") + sealed.sealed[cut + 1 :]
        assert open_session_state(flipped, _KEY) is None, f"a token tampered at {cut} was accepted"


def test_garbage_is_refused_without_raising() -> None:
    """A caller can do nothing about an unopenable token, so it degrades rather than fails."""
    for junk in ("", "not-base64!!", "AAAA", "x" * 200):
        assert open_session_state(junk, _KEY) is None


def test_a_state_that_is_not_an_object_is_refused() -> None:
    """Opened with the right key, but not the shape this contract promises.

    The key being right means this is a format change, not a tamper -- same answer, and worth
    not conflating with one.
    """
    from threetears.core.security.encryption import seal

    assert open_session_state(seal(json.dumps(["not", "an", "object"]), _KEY), _KEY) is None
    assert open_session_state(seal("not even json", _KEY), _KEY) is None


def test_the_repr_does_not_carry_the_ciphertext() -> None:
    """A dataclass repr prints every field, and this one would print a credential.

    Ciphertext is not plaintext, but a credential's ciphertext in a log aggregator is still a
    credential in a log aggregator, and an exception rendering is exactly how it gets there.
    """
    sealed = seal_session_state(_STATE, _KEY, now=_NOW)
    assert sealed.sealed not in repr(sealed)
    assert sealed.sealed not in str(sealed)
    assert sealed.sealed not in f"{sealed}"
    assert "redacted" in repr(sealed)


@pytest.mark.asyncio
async def test_a_stored_state_is_used_while_it_is_fresh(health: ScrapeTargetHealthCollection) -> None:
    sealed = seal_session_state(_STATE, _KEY, now=_NOW)
    await record_session_state(health, target_id=_T, state=sealed)

    row = await health.get(_T)
    assert row is not None
    assert usable_session_state(row, _KEY, now=_NOW + timedelta(hours=1)) == _STATE


@pytest.mark.asyncio
async def test_an_expired_state_is_treated_as_absent(health: ScrapeTargetHealthCollection) -> None:
    """Degrades to "ask for help", never to "send a dead cookie and believe the answer".

    A dead cookie does not fail loudly: the target serves a challenge, extraction fails, and
    the circuit records a wall -- so an expiry that was not honoured looks exactly like a
    target that got harder, and the human's solve is blamed for nothing.
    """
    sealed = seal_session_state(_STATE, _KEY, ttl=timedelta(hours=1), now=_NOW)
    await record_session_state(health, target_id=_T, state=sealed)
    row = await health.get(_T)
    assert row is not None

    assert usable_session_state(row, _KEY, now=_NOW + timedelta(minutes=59)) == _STATE
    assert usable_session_state(row, _KEY, now=_NOW + timedelta(hours=1)) is None
    assert usable_session_state(row, _KEY, now=_NOW + timedelta(days=7)) is None


@pytest.mark.asyncio
async def test_a_state_with_no_expiry_is_treated_as_expired(health: ScrapeTargetHealthCollection) -> None:
    """ "I do not know when this stops being valid" reads as "now", not as "never".

    The writer always sets an expiry, so its absence means a hand-edited or half-written row,
    and the safe reading of a credential with no stated lifetime is that it has none left.
    """
    from threetears.scrape.health import _merge_health

    sealed = seal_session_state(_STATE, _KEY, now=_NOW)
    await _merge_health(health, target_id=_T, changes={"session_state_sealed": sealed.sealed})
    row = await health.get(_T)
    assert row is not None
    assert row.session_state_sealed is not None
    assert row.session_state_expires_at is None

    assert usable_session_state(row, _KEY, now=_NOW) is None


@pytest.mark.asyncio
async def test_no_key_configured_means_no_state_is_sent(health: ScrapeTargetHealthCollection) -> None:
    """A deployment with no master key must not somehow send an unopened token as a cookie jar."""
    sealed = seal_session_state(_STATE, _KEY, now=_NOW)
    await record_session_state(health, target_id=_T, state=sealed)
    row = await health.get(_T)
    assert usable_session_state(row, None, now=_NOW) is None


def test_no_row_or_no_stored_state_means_none() -> None:
    assert usable_session_state(None, _KEY, now=_NOW) is None


@pytest.mark.asyncio
async def test_clearing_removes_both_columns_together(health: ScrapeTargetHealthCollection) -> None:
    """Half-cleared is worse than either state.

    A token with no expiry is a credential of unknown lifetime; an expiry with no token guards
    nothing. Same pairing argument `record_circuit_state` makes for its four columns.
    """
    await record_session_state(health, target_id=_T, state=seal_session_state(_STATE, _KEY, now=_NOW))
    await record_session_state(health, target_id=_T, state=None)

    row = await health.get(_T)
    assert row is not None
    assert row.session_state_sealed is None
    assert row.session_state_expires_at is None


@pytest.mark.asyncio
async def test_a_wrong_key_against_a_stored_state_asks_for_a_human(health: ScrapeTargetHealthCollection) -> None:
    """A rotated master key must degrade, not crash the fetch path.

    Every target's stored solve becomes unopenable at once when a key rotates, so this runs on
    the read path of every poll in that window.
    """
    await record_session_state(health, target_id=_T, state=seal_session_state(_STATE, _KEY, now=_NOW))
    row = await health.get(_T)
    assert usable_session_state(row, _OTHER_KEY, now=_NOW) is None
