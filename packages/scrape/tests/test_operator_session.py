"""Claiming a session's display: that one pod holds it, and knows when it stops.

Every test here runs a real ``KVLease`` over an in-memory bucket, so compare-and-swap, holder
identity and expiry are the genuine article. What is under test is this package's orchestration
on top: that contention refuses instead of queueing, that a claim is renewed, that losing it is
noticed, and that a transport blip is not mistaken for losing it.

The failure being prevented is not a race that settles. Two pods serving one session means two
displays and two browsers, with a human driving whichever one their socket happened to reach.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

import pytest
from _kv_shims import FakeNatsClient
from threetears.core.coordination import KVLease, LeaseUnavailable
from threetears.scrape.operator_session import (
    SESSION_CLAIM_REFRESH,
    SESSION_CLAIM_TTL,
    claim_session,
    session_claim_key,
)

# Whole seconds, because the coordination layer expresses TTL in them and this module refuses
# anything finer rather than truncating it. One second is the shortest claim there is, which
# keeps the deadline tests to about a second each.
_TTL = timedelta(seconds=1)
_REFRESH = timedelta(seconds=0.05)


def _lease(client: FakeNatsClient, pod: str) -> KVLease:
    """A lease factory standing for one pod."""
    return KVLease(client, bucket_name="hitl-claims", pod_id=pod)  # type: ignore[arg-type]


async def _take_over(client: FakeNatsClient, session_id: str, *, by: bytes = b"pod-b") -> None:
    """Rewrite the claim entry's holder, as a stale reclaim by another pod leaves it.

    A compare-and-swap on the real bucket surface, not a back door: this is the same write
    ``KVLease`` itself performs when it reclaims an entry whose holder let it expire.
    """
    bucket = client.buckets["hitl-claims"]
    entry = await bucket.get_entry(key=session_claim_key(session_id))
    assert entry is not None, "the claim wrote no entry to take over"
    value, revision = entry
    written = await bucket.update(
        key=session_claim_key(session_id), value=value.replace(b"pod-a", by), revision=revision
    )
    assert written is not None, "the takeover write lost its own compare-and-swap"


class TestOnlyOnePodHoldsADisplay:
    """The claim exists for this and nothing else."""

    async def test_a_second_pod_is_refused_rather_than_queued(self) -> None:
        """Queueing would hold a caller open for as long as a human takes, which is hours.

        Refusing lets the second pod tell its operator now, which is the only useful answer.
        """
        client = FakeNatsClient()
        async with claim_session(_lease(client, "pod-a"), "session-1", ttl=_TTL, refresh=_REFRESH):
            with pytest.raises(LeaseUnavailable):
                async with claim_session(_lease(client, "pod-b"), "session-1", ttl=_TTL, refresh=_REFRESH):
                    pytest.fail("a second pod claimed a display that was already held")

    async def test_a_released_claim_lets_the_next_pod_in(self) -> None:
        """Any pod becomes the right pod, so a finished session must not lock others out."""
        client = FakeNatsClient()
        async with claim_session(_lease(client, "pod-a"), "session-1", ttl=_TTL, refresh=_REFRESH):
            pass
        async with claim_session(_lease(client, "pod-b"), "session-1", ttl=_TTL, refresh=_REFRESH) as claim:
            assert claim.held, "the display was not free after the first pod released it"

    async def test_different_sessions_do_not_contend(self) -> None:
        """One pod holding one session must not exclude another session entirely."""
        client = FakeNatsClient()
        async with claim_session(_lease(client, "pod-a"), "session-1", ttl=_TTL, refresh=_REFRESH):
            async with claim_session(_lease(client, "pod-b"), "session-2", ttl=_TTL, refresh=_REFRESH) as other:
                assert other.held, "an unrelated session was refused"


class TestLosingTheClaimIsNoticed:
    """A pod that has stopped being the owner and does not know it is the whole hazard."""

    async def test_ownership_taken_by_another_pod_marks_the_claim_lost(self) -> None:
        """Renewal is a compare-and-swap against the holder, so a takeover surfaces on it.

        The takeover is written with an ordinary compare-and-swap rather than a second pod,
        because while this pod's claim is live a second pod cannot legitimately win it. What is
        being modelled is the state AFTER a stale reclaim, where it can -- and the write below
        is byte-for-byte what that reclaim leaves behind.
        """
        client = FakeNatsClient()
        async with claim_session(_lease(client, "pod-a"), "session-1", ttl=_TTL, refresh=_REFRESH) as claim:
            assert claim.held, "the claim was not held to begin with"
            await _take_over(client, "session-1")

            await asyncio.wait_for(claim.until_lost(), timeout=5)
            assert not claim.held, "the claim reported itself held after another pod took it"

    async def test_a_lost_claim_does_not_delete_the_new_owners_entry(self) -> None:
        """Releasing on the way out must not undo the takeover it just lost to.

        Otherwise the pod that correctly gave up ownership frees the display underneath the pod
        that legitimately holds it, and the divergence this whole module prevents arrives by
        way of the cleanup.
        """
        client = FakeNatsClient()
        key = session_claim_key("session-1")
        async with claim_session(_lease(client, "pod-a"), "session-1", ttl=_TTL, refresh=_REFRESH) as claim:
            await _take_over(client, "session-1")
            await asyncio.wait_for(claim.until_lost(), timeout=5)

        surviving = await client.buckets["hitl-claims"].get_entry(key=key)
        assert surviving is not None, "releasing a lost claim deleted the new owner's entry"
        assert b"pod-b" in surviving[0], "the surviving entry is not the new owner's"


class TestATransportBlipIsNotALostClaim:
    """A claim is given up on evidence or on time, never on one failed call."""

    async def test_a_brief_failure_to_renew_does_not_end_the_session(self) -> None:
        """Ending an operator's session over a blip costs them their in-flight work.

        The TTL is a whole second and the bucket is unreachable for a fraction of it, so the
        claim cannot have expired -- and a pod that gave up here would be discarding a claim it
        still holds.
        """
        client = FakeNatsClient()
        async with claim_session(_lease(client, "pod-a"), "session-1", ttl=_TTL, refresh=_REFRESH) as claim:
            bucket = client.buckets["hitl-claims"]
            bucket.unreachable = ConnectionError("kv is briefly unreachable")
            await asyncio.sleep(_REFRESH.total_seconds() * 4)
            bucket.unreachable = None
            assert claim.held, "a transient renewal failure was treated as losing the claim"

            # And the claim recovers rather than merely surviving: renewal resumes.
            await asyncio.sleep(_REFRESH.total_seconds() * 3)
            assert claim.held, "the claim was given up after the bucket came back"

    async def test_a_failure_lasting_past_the_ttl_gives_the_claim_up(self) -> None:
        """A claim un-renewed past its TTL has expired whether or not this pod noticed.

        Another pod may already have taken it, so continuing to serve is the fault. Holding on
        because our own renewals are failing is the exact inversion of the safe reading.
        """
        client = FakeNatsClient()
        async with claim_session(_lease(client, "pod-a"), "session-1", ttl=_TTL, refresh=_REFRESH) as claim:
            client.buckets["hitl-claims"].unreachable = ConnectionError("kv is gone")
            await asyncio.wait_for(claim.until_lost(), timeout=5)
            assert not claim.held, "a claim was kept alive past its own TTL by failing renewals"


class TestCleanupNeverReplacesTheOutcome:
    """Releasing is best-effort, because the way it fails is correlated with why we are here."""

    async def test_a_body_error_survives_a_release_that_cannot_reach_the_bucket(self) -> None:
        """The likeliest reason a release fails is the reason the claim was given up.

        An unreachable coordination layer ends the claim AND breaks the release, so this is the
        ordinary path out of a lost claim rather than an exotic one. If cleanup raised, the
        operator's real failure would be replaced by a KV error every single time -- worst
        exactly when the original exception was the one worth reading.
        """
        client = FakeNatsClient()
        with pytest.raises(RuntimeError, match="what actually went wrong"):
            async with claim_session(_lease(client, "pod-a"), "session-1", ttl=_TTL, refresh=_REFRESH):
                client.buckets["hitl-claims"].unreachable = ConnectionError("kv is gone")
                raise RuntimeError("what actually went wrong")

    async def test_a_clean_exit_is_not_turned_into_a_failure_by_cleanup(self) -> None:
        """A session that ended fine must still end fine when the bucket has gone away."""
        client = FakeNatsClient()
        async with claim_session(_lease(client, "pod-a"), "session-1", ttl=_TTL, refresh=_REFRESH):
            client.buckets["hitl-claims"].unreachable = ConnectionError("kv is gone")


class TestTheContractRefusesWhatItCannotHonour:
    """Configuration that cannot mean what it says is refused, not quietly reinterpreted."""

    async def test_a_sub_second_ttl_is_refused_rather_than_truncated(self) -> None:
        """The coordination layer takes whole seconds, and 0.5 truncates to zero.

        A zero TTL writes an entry stale the instant it lands, so the claim is real for no time
        at all and the next pod to ask takes the display. Nothing about that is visible at the
        call site, which is why it raises.
        """
        client = FakeNatsClient()
        with pytest.raises(ValueError, match="whole number of seconds"):
            async with claim_session(
                _lease(client, "pod-a"), "session-1", ttl=timedelta(seconds=0.5), refresh=timedelta(seconds=0.1)
            ):
                pytest.fail("a sub-second ttl was accepted")

    async def test_a_refresh_slower_than_the_ttl_is_refused(self) -> None:
        """Renewing less often than the claim expires means it lapses under its own holder."""
        client = FakeNatsClient()
        with pytest.raises(ValueError, match="must be shorter than ttl"):
            async with claim_session(
                _lease(client, "pod-a"), "session-1", ttl=timedelta(seconds=1), refresh=timedelta(seconds=2)
            ):
                pytest.fail("a refresh interval longer than the ttl was accepted")

    def test_the_shipped_defaults_satisfy_their_own_contract(self) -> None:
        """The constants are the values every deployment gets, so they are held too.

        Two consecutive renewals must be able to fail without the claim lapsing, which is what
        the refresh interval being a third of the TTL buys.
        """
        assert SESSION_CLAIM_REFRESH < SESSION_CLAIM_TTL
        assert SESSION_CLAIM_REFRESH * 3 <= SESSION_CLAIM_TTL, (
            "the refresh interval leaves no room for a missed renewal"
        )
        assert SESSION_CLAIM_TTL.total_seconds() == int(SESSION_CLAIM_TTL.total_seconds())


class TestASinglePodDeploymentStillWorks:
    """The compose file in this repo runs one pod, and it must not need NATS to work."""

    async def test_no_lease_yields_a_claim_and_says_so_loudly(self, caplog: pytest.LogCaptureFixture) -> None:
        """Silence here is the defect. A platform that meant to pass a lease and did not gets
        no mutual exclusion, and the symptom is two operators on two displays believing they
        share one -- with nothing anywhere saying why.
        """
        with caplog.at_level(logging.WARNING):
            async with claim_session(None, "session-1") as claim:
                assert claim.held, "a claim without a lease reported itself unheld"
        assert any("not claimed" in record.getMessage() for record in caplog.records), (
            "no warning was emitted for a session running without a claim"
        )


class TestTheKeyIsDerivedNotUsedRaw:
    """A session id comes from outside and a KV key admits a restricted character set."""

    def test_an_arbitrary_id_becomes_a_safe_key(self) -> None:
        """Dots, spaces and wildcards are all legal in an id and all trouble in a key."""
        key = session_claim_key("a session.with wildcards * and >")
        assert key.isalnum() and key.islower(), f"the derived key is not subject-safe: {key!r}"

    def test_the_same_id_always_derives_the_same_key(self) -> None:
        """Every pod must reach the same key from the same id, or they do not contend at all."""
        assert session_claim_key("session-1") == session_claim_key("session-1")
        assert session_claim_key("session-1") != session_claim_key("session-2")

    def test_an_empty_id_is_refused(self) -> None:
        """An empty id would collapse every unclaimed session onto one shared key."""
        with pytest.raises(ValueError, match="non-empty"):
            session_claim_key("")
