"""Control messages: that they reach the owning pod, and what they are allowed to carry.

Every test routes through the real ``serve_owner`` and ``forward`` over an in-memory bus, so
both ends derive the session's subject independently through ``Subjects.forward``. Stubbing
either out would remove the one thing that can silently break: two sides disagreeing about
where a session's messages go, which fails as "no owner" rather than as anything diagnosable.

The security claim under test is narrow and absolute. A completed tab's raw export is the cookie
jar of a target a human has just cleared. It must not appear in anything published.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from _bus_shims import FakeBus, RecordingDisplay
from pydantic import SecretStr
from threetears.nats import ForwardedHandlerError, NoOwnerError
from threetears.scrape.operator_control import (
    CLOSE_SESSION,
    COMPLETE_TAB,
    OPEN_TAB,
    READ_STATE,
    close_session,
    complete_tab,
    open_tab,
    read_state,
    serve_session,
)
from threetears.scrape.operator_session import SessionClaim
from threetears.scrape.session_state import open_session_state, seal_session_state

_KEY = SecretStr("a" * 32)
_SESSION = "session-1"

#: A cookie jar shaped like the real thing: what a challenge system sets to say "this one is
#: fine", which is exactly what makes it a credential.
_RAW_STATE = {"cookies": [{"name": "cf_clearance", "value": "s3cr3t-value", "domain": ".example.com"}]}


def _claim() -> SessionClaim:
    """A held claim, which is what entitles a pod to answer."""
    return SessionClaim(session_id=_SESSION)


class TestAMessageReachesThePodHoldingTheSession:
    """Addressed to the session, not to a pod, and the two ends must agree on where that is."""

    async def test_each_message_reaches_the_display(self) -> None:
        """All four, because a contract with one untested member is a contract with a gap."""
        bus, display = FakeBus(), RecordingDisplay()
        async with serve_session(bus, _claim(), display, session_state_key=_KEY):  # type: ignore[arg-type]
            opened = await open_tab(bus, _SESSION, target_id="t-1", url="https://example.com")  # type: ignore[arg-type]
            await complete_tab(bus, _SESSION, tab_id=opened["tab_id"])  # type: ignore[arg-type]
            state = await read_state(bus, _SESSION)  # type: ignore[arg-type]
            await close_session(bus, _SESSION)  # type: ignore[arg-type]

        assert [t.target_id for t in display.opened] == ["t-1"]
        assert display.completed == ["tab-1"]
        assert display.reads == 1
        assert display.closed == 1
        assert state["free_slots"] == 4

    async def test_a_message_for_another_session_finds_no_owner(self) -> None:
        """The subject is derived from the session id, so a different id is a different subject.

        If it were not, a pod serving one session would answer for every session, and the
        routing this whole module exists for would be decorative.
        """
        bus, display = FakeBus(), RecordingDisplay()
        async with serve_session(bus, _claim(), display, session_state_key=_KEY):  # type: ignore[arg-type]
            with pytest.raises(NoOwnerError):
                await read_state(bus, "some-other-session")  # type: ignore[arg-type]
        assert display.reads == 0, "a pod answered for a session it does not hold"

    async def test_no_owner_at_all_is_distinguishable_from_a_failure(self) -> None:
        """A caller retries or self-promotes on "no owner" and must not do that on a fault."""
        bus = FakeBus()
        with pytest.raises(NoOwnerError):
            await read_state(bus, _SESSION)  # type: ignore[arg-type]

    async def test_the_subscription_is_dropped_when_serving_ends(self) -> None:
        """A pod that stopped serving must stop receiving, or it answers after releasing."""
        bus, display = FakeBus(), RecordingDisplay()
        async with serve_session(bus, _claim(), display, session_state_key=_KEY):  # type: ignore[arg-type]
            assert bus.registrations, "nothing subscribed while the session was served"
        assert not bus.registrations, "the subscription outlived the session it served"


class TestTheHumansWorkIsSealedBeforeItTravels:
    """The one claim here that is a security property rather than a behaviour."""

    async def test_a_raw_cookie_jar_never_appears_in_anything_published(self) -> None:
        """Asserted against every payload the bus carried, not against the reply alone.

        Checking the reply would prove the field was renamed. Checking the bus proves the value
        did not travel -- which is the actual claim, and the one that survives somebody later
        adding a second field that happens to include it.
        """
        bus, display = FakeBus(), RecordingDisplay(raw_state=_RAW_STATE)
        async with serve_session(bus, _claim(), display, session_state_key=_KEY):  # type: ignore[arg-type]
            reply = await complete_tab(bus, _SESSION, tab_id="tab-1")  # type: ignore[arg-type]

        assert "session_state" not in reply, "the raw export was returned to the caller"
        for payload in bus.sent:
            assert b"s3cr3t-value" not in payload, "a live cookie value travelled on the bus"
            assert b"cf_clearance" not in payload, "a live cookie name travelled on the bus"

    async def test_the_sealed_reply_opens_back_to_the_original_solve(self) -> None:
        """Sealing that could not be reversed would be a very secure way to lose the work."""
        bus, display = FakeBus(), RecordingDisplay(raw_state=_RAW_STATE)
        async with serve_session(bus, _claim(), display, session_state_key=_KEY):  # type: ignore[arg-type]
            reply = await complete_tab(bus, _SESSION, tab_id="tab-1")  # type: ignore[arg-type]

        assert reply["session_state_sealed"], "no sealed state came back"
        assert reply["session_state_expires_at"], "a sealed solve came back with no expiry"
        assert open_session_state(reply["session_state_sealed"], _KEY) == _RAW_STATE

    async def test_a_tab_that_exported_nothing_carries_no_sealed_field(self) -> None:
        """An export can fail, and an empty seal would be a solve that is not there."""
        bus, display = FakeBus(), RecordingDisplay(raw_state=None)
        async with serve_session(bus, _claim(), display, session_state_key=_KEY):  # type: ignore[arg-type]
            reply = await complete_tab(bus, _SESSION, tab_id="tab-1")  # type: ignore[arg-type]

        assert "session_state_sealed" not in reply
        assert "session_state" not in reply, "an absent export still left the raw key behind"


class TestAPodThatLostTheSessionStopsActing:
    """Dropping the subscription is prompt, not instantaneous, and the gap is the hazard."""

    async def test_a_message_arriving_after_the_claim_is_lost_is_refused(self) -> None:
        """Acting on it would drive a display the new owner is using.

        The claim is lost while the subscription is still live, which is precisely the window
        between another pod taking over and this one noticing.
        """
        bus, display = FakeBus(), RecordingDisplay()
        claim = _claim()
        async with serve_session(bus, claim, display, session_state_key=_KEY):  # type: ignore[arg-type]
            claim.lost.set()
            with pytest.raises(ForwardedHandlerError):
                await open_tab(bus, _SESSION, target_id="t-1", url="https://example.com")  # type: ignore[arg-type]

        assert display.opened == [], "a pod that had lost the session still drove its display"

    async def test_serving_refuses_to_start_on_a_claim_already_lost(self) -> None:
        """Subscribing at all would be advertising ownership this pod does not have."""
        bus, display = FakeBus(), RecordingDisplay()
        claim = _claim()
        claim.lost.set()
        with pytest.raises(RuntimeError, match="not held by this pod"):
            async with serve_session(bus, claim, display, session_state_key=_KEY):  # type: ignore[arg-type]
                pytest.fail("a pod served a session it had already lost")
        assert not bus.registrations, "a lost session was subscribed anyway"


class TestAHandoverReleasesWhatThisPodWasHolding:
    """Stopping serving is not the same as letting go, and only one of them was happening.

    Losing the claim already ended the relay, refused control messages and released the lease.
    None of that touched what this pod still HELD: an x11vnc on a live display, and browser
    contexts carrying the cookies of whatever the human was part way through. They survived until
    the sidecar's own half-hour TTL reaped them -- a browser holding a target's authenticated
    session for a session this pod is no longer entitled to.
    """

    async def test_losing_the_claim_closes_this_pods_own_session(self) -> None:
        """Safe because the pods do not share a display: each has its own sidecar on its own
        loopback, so closing this one cannot touch the display the new owner just brought up."""
        bus, display = FakeBus(), RecordingDisplay()
        claim = _claim()
        async with serve_session(bus, claim, display, session_state_key=_KEY):  # type: ignore[arg-type]
            claim.lost.set()

        assert display.closed == 1, (
            "a pod that lost the session left its display up and its contexts holding the human's "
            "cookies until the sidecar's TTL collected them"
        )

    async def test_an_ordinary_end_does_not_close_it_here(self) -> None:
        """The caller owns a normal teardown. Closing twice would race its own close and, on the
        single-pod deployment where nothing is ever lost, would end sessions that are still live."""
        bus, display = FakeBus(), RecordingDisplay()
        async with serve_session(bus, _claim(), display, session_state_key=_KEY):  # type: ignore[arg-type]
            pass

        assert display.closed == 0, "serving closed a session that ended normally"

    async def test_a_sidecar_that_will_not_close_does_not_mask_the_outcome(self) -> None:
        """This runs in a `finally` on the way out of a session that already went wrong."""

        class _Refuses(RecordingDisplay):
            async def close_session(self) -> dict[str, object]:
                raise RuntimeError("the sidecar is not answering")

        bus, display = FakeBus(), _Refuses()
        claim = _claim()
        # Must not raise: the sidecar's TTL is still the backstop.
        async with serve_session(bus, claim, display, session_state_key=_KEY):  # type: ignore[arg-type]
            claim.lost.set()


class TestTheContractRefusesWhatItCannotCarryOut:
    """Every refusal crosses the wire as a type name, so it has to be the right one."""

    async def test_an_unknown_operation_is_refused_by_name(self) -> None:
        """A caller seeing this is running a different version of this contract."""
        bus, display = FakeBus(), RecordingDisplay()
        async with serve_session(bus, _claim(), display, session_state_key=_KEY):  # type: ignore[arg-type]
            with pytest.raises(ForwardedHandlerError) as caught:
                await _raw_send(bus, {"op": "open_session"})
        assert caught.value.type_name == "UnknownControlMessage"

    async def test_opening_a_session_is_not_on_the_subject(self) -> None:
        """Taking the claim is what MAKES a pod the owner, so routing it would be circular.

        Pinned rather than left implicit: it is the one message somebody will reasonably expect
        to find here, and adding it would make ownership depend on an owner already existing.
        """
        assert "open_session" not in {OPEN_TAB, COMPLETE_TAB, CLOSE_SESSION, READ_STATE}

    async def test_a_malformed_payload_is_refused_rather_than_guessed_at(self) -> None:
        """Half-reading a control message is how a tab opens against the wrong target."""
        bus, display = FakeBus(), RecordingDisplay()
        async with serve_session(bus, _claim(), display, session_state_key=_KEY):  # type: ignore[arg-type]
            with pytest.raises(ForwardedHandlerError):
                await _raw_bytes(bus, b"not json at all")
        assert display.opened == []

    async def test_a_missing_required_field_names_the_field(self) -> None:
        """The message is all a caller gets back, so it has to say which field."""
        bus, display = FakeBus(), RecordingDisplay()
        async with serve_session(bus, _claim(), display, session_state_key=_KEY):  # type: ignore[arg-type]
            with pytest.raises(ForwardedHandlerError, match="url"):
                await _raw_send(bus, {"op": OPEN_TAB, "target_id": "t-1"})
        assert display.opened == []


class TestWhatACallerSends:
    """The caller-side helpers are half the contract; a wrong field is a silent no-op."""

    async def test_nav_steps_are_carried_through(self) -> None:
        """A target is re-driven from url plus nav_steps, so losing them loses the target."""
        bus, display = FakeBus(), RecordingDisplay()
        steps = [{"action": "click", "selector": "#accept"}]
        async with serve_session(bus, _claim(), display, session_state_key=_KEY):  # type: ignore[arg-type]
            await open_tab(bus, _SESSION, target_id="t-1", url="https://example.com", nav_steps=steps)  # type: ignore[arg-type]

        assert display.opened[0].nav_steps == steps


class TestNoCookieJarTravelsInTheClearInEitherDirection:
    """The reply is sealed, so accepting a plaintext jar INBOUND would defeat the whole point.

    That asymmetry is the specific defect these pin: a contract that seals what it sends and
    accepts what it would not send reads as deliberate to whoever finds it, and puts exactly the
    credential the sealing protects onto exactly the bus it protects it from.
    """

    async def test_a_prior_solve_travels_sealed_and_arrives_opened(self) -> None:
        """Sealed on the bus, plaintext only inside the pod that needs it."""
        bus, display = FakeBus(), RecordingDisplay()
        sealed = seal_session_state(_RAW_STATE, _KEY).sealed

        async with serve_session(bus, _claim(), display, session_state_key=_KEY):  # type: ignore[arg-type]
            await open_tab(  # type: ignore[arg-type]
                bus, _SESSION, target_id="t-1", url="https://example.com", session_state_sealed=sealed
            )

        assert display.opened[0].session_state == _RAW_STATE, (
            "the owning pod did not open the solve, so the tab opens unauthenticated"
        )
        for payload in bus.sent:
            assert b"s3cr3t-value" not in payload, "a live cookie value travelled on the bus"
            assert b"cf_clearance" not in payload, "a live cookie name travelled on the bus"

    async def test_a_solve_that_cannot_be_trusted_opens_the_tab_without_one(self) -> None:
        """A wrong key or a tampered token must degrade to "needs a human", never to bad data.

        That is the same failure the stored-solve path already treats as safe: no solve means the
        challenge is cleared again, which costs an operator a minute and cannot produce a wrong
        record.
        """
        bus, display = FakeBus(), RecordingDisplay()
        wrong_key = SecretStr("b" * 32)
        sealed = seal_session_state(_RAW_STATE, wrong_key).sealed

        async with serve_session(bus, _claim(), display, session_state_key=_KEY):  # type: ignore[arg-type]
            await open_tab(  # type: ignore[arg-type]
                bus, _SESSION, target_id="t-1", url="https://example.com", session_state_sealed=sealed
            )

        assert display.opened[0].session_state is None, "an untrusted solve was handed to the browser anyway"
        assert display.opened[0].url == "https://example.com", "the tab was not opened at all"


async def _raw_send(bus: FakeBus, message: dict[str, object]) -> bytes:
    """Send a hand-built control message, to reach shapes the helpers cannot produce."""
    import json

    return await _raw_bytes(bus, json.dumps(message).encode("utf-8"))


async def _raw_bytes(bus: FakeBus, payload: bytes) -> bytes:
    """Send arbitrary bytes to the session's subject."""
    from threetears.nats import forward

    return await forward(bus, _SESSION, payload, timeout=timedelta(seconds=5))  # type: ignore[arg-type]
