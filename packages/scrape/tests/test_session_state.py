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
from unittest.mock import patch

from threetears.scrape.session_state import (
    DEFAULT_SESSION_STATE_TTL,
    open_session_state,
    record_session_state,
    seal_session_state,
    usable_session_state,
)

_KEY = SecretStr("an-operator-master-key-from-secret-refs")
_OTHER_KEY = SecretStr("a-different-operators-master-key")
_NOW = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
_T = "warn_oh"
#: Known-good page + strategy + schema, lifted from test_tool.py so extraction succeeds from
#: the seeded recipe and no test in this file ever reaches a model.
_EXTRACTABLE_HTML = "<html><body><table><tr><td>Acme Corp</td><td>42</td></tr></table></body></html>"
_EXTRACTABLE_STRATEGY = {"employer": "td:nth-child(1)", "affected_count": "td:nth-child(2)"}
_EXTRACTABLE_SCHEMA = {"employer": "str", "affected_count": "int"}

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
    nothing. Same pairing argument `record_circuit_state` makes for the columns a trip writes.
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


# ---------------------------------------------------------------------------
# The wiring. These are the tests whose absence let chunk 06 be marked done
# while the capability did not exist: every piece was built and tested, and
# nothing carried a stored solve into an actual fetch.
# ---------------------------------------------------------------------------


class _StateCapturingDriver:
    """A ScrapeDriver stand-in that records the session_state it was handed.

    # parity-with: threetears.scrape.driver.ScrapeDriver
    """

    def __init__(self) -> None:
        self.session_states: list[dict | None] = []

    @property
    def name(self) -> str:
        return "state-capturing"

    async def render(
        self,
        url: str,
        *,
        timeout: float = 30.0,
        wait_for: str | None = None,
        capture_network: bool = False,
        nav_steps: list | None = None,
        session_state: dict | None = None,
    ):
        from threetears.scrape.driver import RenderedPage

        self.session_states.append(session_state)
        # The same html/strategy/schema triple test_tool.py already proves extracts cleanly.
        # Reused rather than invented: a near-miss makes extraction fail, which sends the eval
        # loop to candidate generation and the challenge classifier to a verdict, both against
        # a fake key, and both burn their full retry budget -- 30s per test for a path these
        # tests are not about.
        return RenderedPage(html=_EXTRACTABLE_HTML, status=200, final_url=url, timing_ms=1.0)


def _no_llm():
    """Stub the classifier itself, not the model factory underneath it.

    These tests are about which cookies reach the driver, and supplying a health collection
    opts the tool into challenge classification, which runs whenever extraction misses. The
    classifier is a bounded-retry call -- 6 attempts with 2s linear backoff -- so making the
    MODEL raise does not skip it, it makes it retry: 2+4+6+8+10 = exactly 30 seconds per test,
    which is what was measured before this was moved.

    Patching at the classifier boundary returns immediately and leaves every behaviour these
    tests assert on untouched, because none of them is about classification. ``None`` is the
    classifier's own documented "could not decide" answer, so the tool takes the same path it
    would on a real inconclusive verdict.
    """
    return patch("threetears.scrape.eval_loop.classify_failed_page", return_value=None)


async def _tool(driver, health, key, *, target_id: str):
    """A tool whose recipe already wins, so no test here reaches an LLM.

    Seeding the recipe is not incidental tidiness. Without it the eval loop treats every
    fetch as a target it has never solved, runs candidate generation against a fake API key,
    and spends its full retry-with-backoff budget before failing -- 65 seconds per test,
    measured, for tests that are about which cookies reach the driver and have nothing to do
    with extraction at all.
    """
    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import DefaultCoreConfig
    from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection
    from threetears.scrape.tool import ScrapeTool

    reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
    recipes = ScrapeRecipeCollection(reg, cfg, nats_client=None)
    await recipes.save_entity(
        recipes.create(
            {
                "target_id": target_id,
                "extraction_strategy": _EXTRACTABLE_STRATEGY,
                "won_at": None,
                "last_validated_at": None,
                "consecutive_validation_failures": 0,
            }
        )
    )
    return ScrapeTool(
        recipe_collection=recipes,
        extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
        health_collection=health,
        session_state_key=key,
        drivers={"nodriver": driver},
        api_key="k",
    )


@pytest.mark.asyncio
async def test_the_tool_carries_a_stored_solve_into_the_fetch(health: ScrapeTargetHealthCollection) -> None:
    """The step that makes this chunk a capability rather than plumbing.

    The columns, the sealing, the driver parameter and the sidecar's apply-path all existed
    and were individually tested while NOTHING passed a stored solve to a driver, so a person
    cleared the same challenge on every poll and the chunk's whole purpose was unmet. Chunk 02
    had already written this rule down: "the tool wiring is what makes the parameter a
    capability rather than plumbing."
    """
    url = "https://example.gov/walled"
    schema = _EXTRACTABLE_SCHEMA
    from threetears.scrape.tool import _derive_target_id

    target_id = _derive_target_id(url, schema)
    await record_session_state(health, target_id=target_id, state=seal_session_state(_STATE, _KEY))

    driver = _StateCapturingDriver()
    tool = await _tool(driver, health, _KEY, target_id=target_id)
    with _no_llm():
        await tool.execute(url=url, field_schema=schema)

    assert driver.session_states, "the driver was never called"
    assert driver.session_states[0] == _STATE, (
        "the fetch went out without the human's solve, so the target sees an unauthenticated "
        "request and the person clears the same challenge again"
    )


@pytest.mark.asyncio
async def test_no_stored_solve_means_no_session_state(health: ScrapeTargetHealthCollection) -> None:
    """A target nobody has ever cleared fetches exactly as it always did."""
    from threetears.scrape.tool import _derive_target_id

    url = "https://example.gov/plain"
    schema = _EXTRACTABLE_SCHEMA
    driver = _StateCapturingDriver()
    tool = await _tool(driver, health, _KEY, target_id=_derive_target_id(url, schema))
    with _no_llm():
        await tool.execute(url=url, field_schema=schema)
    assert driver.session_states == [None]


@pytest.mark.asyncio
async def test_an_expired_solve_is_not_sent(health: ScrapeTargetHealthCollection) -> None:
    """The expiry has to be honoured on the path that actually fetches, not only in a helper.

    A dead cookie does not fail loudly: the target serves a challenge, extraction fails, and
    the circuit records a wall -- so an unhonoured expiry looks like a target that got harder.
    """
    url = "https://example.gov/stale"
    schema = _EXTRACTABLE_SCHEMA
    from threetears.scrape.tool import _derive_target_id

    target_id = _derive_target_id(url, schema)
    stale = seal_session_state(_STATE, _KEY, ttl=timedelta(seconds=-1))
    await record_session_state(health, target_id=target_id, state=stale)

    driver = _StateCapturingDriver()
    tool = await _tool(driver, health, _KEY, target_id=target_id)
    with _no_llm():
        await tool.execute(url=url, field_schema=schema)

    assert driver.session_states == [None], "an expired solve was sent as if a human had just earned it"


@pytest.mark.asyncio
async def test_without_a_key_the_stored_solve_is_left_sealed(health: ScrapeTargetHealthCollection) -> None:
    """A deployment with no master key fetches as if no human had cleared the target.

    The safe direction: the alternative is a fetch that never happens over a key problem, or
    worse, ciphertext handed to a driver as if it were a cookie jar.
    """
    url = "https://example.gov/nokey"
    schema = _EXTRACTABLE_SCHEMA
    from threetears.scrape.tool import _derive_target_id

    target_id = _derive_target_id(url, schema)
    await record_session_state(health, target_id=target_id, state=seal_session_state(_STATE, _KEY))

    driver = _StateCapturingDriver()
    tool = await _tool(driver, health, None, target_id=target_id)
    with _no_llm():
        await tool.execute(url=url, field_schema=schema)

    assert driver.session_states == [None]
