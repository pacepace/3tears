"""Tests for the durable per-target fetch circuit.

The claim this chunk has to earn is that a repeatedly blocked target gets fetched less and
less often, and -- separately, because it does not follow -- classified less and less often
too. The second one is not a restatement of the first: classification sits behind its own
verdict cache, and that cache misses on a real interstitial because the page carries a
per-request id, so the only thing that bounds it is the fetch never happening.

So the tests here come in two layers. The state-machine tests pin the transitions and the
backoff arithmetic directly. The decay tests drive a target through many polls with a page
that is different every time -- the shape a Cloudflare interstitial actually has -- and count
what it cost. That second layer is the one that would catch a circuit whose states are all
individually correct while something downstream still fetches on every poll.

No database anywhere: the collections fall back to an in-memory L3.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.models.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)

from threetears.scrape.circuit import BackoffPolicy, TargetCircuit
from threetears.scrape.health import ScrapeTargetHealthCollection, record_circuit_state

_T = "warn_oh"
_NOW = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)


@pytest.fixture()
def health() -> ScrapeTargetHealthCollection:
    """No L3 pool: the in-memory fallback every unit test in this package uses."""
    return ScrapeTargetHealthCollection(
        CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS"), nats_client=None
    )


@pytest.fixture()
def policy() -> BackoffPolicy:
    """Threshold of 2 and a 60s base, so a test can trip a circuit without a long setup."""
    return BackoffPolicy(failure_threshold=2, base_delay_seconds=60.0, max_delay_seconds=480.0)


# ---------------------------------------------------------------------------
# BackoffPolicy
# ---------------------------------------------------------------------------


def test_the_delay_doubles_per_failure_past_the_threshold(policy: BackoffPolicy) -> None:
    """Decay, not a constant.

    A fixed delay would bound a blocked target's cost to a constant rate forever, which is
    a floor rather than a decay. Each probe that finds the wall still standing has to buy a
    longer silence than the last one.
    """
    assert policy.delay_for(2) == 60.0
    assert policy.delay_for(3) == 120.0
    assert policy.delay_for(4) == 240.0


def test_the_delay_is_capped_so_a_target_is_never_abandoned(policy: BackoffPolicy) -> None:
    """A wall is rarely permanent, and a target nobody re-probes stays broken after it lifts."""
    assert policy.delay_for(20) == 480.0


def test_an_absurd_failure_count_still_produces_a_number(policy: BackoffPolicy) -> None:
    """A runaway counter must not turn a delay computation into an OverflowError.

    ``2.0 ** 10000`` raises rather than saturating, so the exponent is capped well past
    where the ceiling has already flattened the curve.
    """
    assert policy.delay_for(10_000) == 480.0


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_closed_circuit_permits_the_fetch(health: ScrapeTargetHealthCollection) -> None:
    """A target nobody has ever had trouble with has no row at all, and must still fetch."""
    decision = await TargetCircuit(health).check(_T, now=_NOW)
    assert decision.permitted
    assert decision.state is CircuitState.CLOSED


@pytest.mark.asyncio
async def test_blocks_below_the_threshold_do_not_suppress_anything(
    health: ScrapeTargetHealthCollection, policy: BackoffPolicy
) -> None:
    """Transient-failure tolerance, matching the recipe path's own posture.

    One block is not a pattern. Suppressing on the first one would back a target off for a
    quarter of an hour over a single bad response.
    """
    circuit = TargetCircuit(health, policy=policy)
    await circuit.record_blocked(_T, now=_NOW)

    row = await health.get(_T)
    assert row is not None
    assert row.consecutive_fetch_failures == 1
    assert row.circuit_state == CircuitState.CLOSED.value
    assert (await circuit.check(_T, now=_NOW)).permitted


@pytest.mark.asyncio
async def test_the_threshold_opens_the_circuit_and_suppresses_the_next_fetch(
    health: ScrapeTargetHealthCollection, policy: BackoffPolicy
) -> None:
    """Closed to open, and the fetch that would have followed does not happen."""
    circuit = TargetCircuit(health, policy=policy)
    await circuit.record_blocked(_T, now=_NOW)
    await circuit.record_blocked(_T, now=_NOW)

    row = await health.get(_T)
    assert row is not None
    assert row.circuit_state == CircuitState.OPEN.value
    assert row.blocked_until == _NOW + timedelta(seconds=60)
    assert row.last_blocked_at == _NOW

    decision = await circuit.check(_T, now=_NOW + timedelta(seconds=30))
    assert not decision.permitted
    assert decision.retry_after_seconds == pytest.approx(30.0, abs=1.0)


@pytest.mark.asyncio
async def test_an_elapsed_window_admits_one_probe_and_records_half_open(
    health: ScrapeTargetHealthCollection, policy: BackoffPolicy
) -> None:
    """Open to half-open, and the promotion is written back rather than only inferred.

    Persisting it is what lets another pod read the target as probing instead of as still
    open, and is what makes the operator index on ``circuit_state`` tell the truth.
    """
    circuit = TargetCircuit(health, policy=policy)
    await circuit.record_blocked(_T, now=_NOW)
    await circuit.record_blocked(_T, now=_NOW)

    decision = await circuit.check(_T, now=_NOW + timedelta(seconds=61))
    assert decision.permitted
    assert decision.is_probe
    assert decision.state is CircuitState.HALF_OPEN

    row = await health.get(_T)
    assert row is not None
    assert row.circuit_state == CircuitState.HALF_OPEN.value


@pytest.mark.asyncio
async def test_a_failed_probe_reopens_with_a_longer_window(
    health: ScrapeTargetHealthCollection, policy: BackoffPolicy
) -> None:
    """Half-open back to open. The wait doubles, which is where the decay comes from."""
    circuit = TargetCircuit(health, policy=policy)
    await circuit.record_blocked(_T, now=_NOW)
    await circuit.record_blocked(_T, now=_NOW)
    probe_at = _NOW + timedelta(seconds=61)
    await circuit.check(_T, now=probe_at)
    await circuit.record_blocked(_T, now=probe_at)

    row = await health.get(_T)
    assert row is not None
    assert row.circuit_state == CircuitState.OPEN.value
    assert row.blocked_until == probe_at + timedelta(seconds=120)


@pytest.mark.asyncio
async def test_a_successful_probe_closes_the_circuit_and_clears_the_window(
    health: ScrapeTargetHealthCollection, policy: BackoffPolicy
) -> None:
    """Half-open back to closed, and every column the trip wrote is cleared.

    Clearing three of the four would leave a target reading as closed while still carrying a
    future ``blocked_until`` that goes on gating it -- a target that looks healthy in every
    query an operator would run and is silently never fetched again.
    """
    circuit = TargetCircuit(health, policy=policy)
    await circuit.record_blocked(_T, now=_NOW)
    await circuit.record_blocked(_T, now=_NOW)
    probe_at = _NOW + timedelta(seconds=61)
    await circuit.check(_T, now=probe_at)
    await circuit.record_reachable(_T, now=probe_at)

    row = await health.get(_T)
    assert row is not None
    assert row.circuit_state == CircuitState.CLOSED.value
    assert row.consecutive_fetch_failures == 0
    assert row.blocked_until is None
    assert (await circuit.check(_T, now=probe_at)).permitted


@pytest.mark.asyncio
async def test_a_transport_failure_backs_off_without_claiming_the_target_was_walled(
    health: ScrapeTargetHealthCollection, policy: BackoffPolicy
) -> None:
    """An unreachable target and a walled one share a circuit but not a diagnosis.

    Both mean "we did not get the content, and retrying now will not change that", so both
    back off. Only a wall stamps ``last_blocked_at``, so the column that answers "when was
    this target last behind a wall" does not quietly become "when did anything last go
    wrong" and send an operator hunting for a challenge page that was really a DNS failure.
    """
    circuit = TargetCircuit(health, policy=policy)
    await circuit.record_unreachable(_T, now=_NOW)
    await circuit.record_unreachable(_T, now=_NOW)

    row = await health.get(_T)
    assert row is not None
    assert row.circuit_state == CircuitState.OPEN.value
    assert row.last_blocked_at is None


@pytest.mark.asyncio
async def test_a_healthy_target_with_a_row_is_not_rewritten_on_every_poll(
    health: ScrapeTargetHealthCollection,
) -> None:
    """The common case must cost a read, not a read-modify-write.

    Deliberately against a target that HAS a health row and is closed at zero, which is
    what every target that has ever validated once looks like. Testing this against a target
    with no row at all would pass on the no-row early return and say nothing about the guard
    it claims to cover -- the first version of this test did exactly that, and only reverting
    the guard and watching this still pass showed it up.

    A fleet of healthy targets polled every few minutes would otherwise spend a write per
    target per poll to record that nothing has changed.
    """
    await record_circuit_state(
        health,
        target_id=_T,
        circuit_state=CircuitState.CLOSED.value,
        consecutive_fetch_failures=0,
        blocked_until=None,
    )

    writes: list[str] = []
    original = health.save_entity

    async def counting_save(entity: object, **kwargs: object) -> object:
        writes.append("write")
        return await original(entity, **kwargs)  # type: ignore[arg-type]

    health.save_entity = counting_save  # type: ignore[method-assign, assignment]
    await TargetCircuit(health).record_reachable(_T, now=_NOW)
    assert writes == []


@pytest.mark.asyncio
async def test_an_uninterpretable_stored_state_is_read_as_closed(
    health: ScrapeTargetHealthCollection, policy: BackoffPolicy
) -> None:
    """A value this version cannot interpret must cost one wasted fetch, not a dead target.

    Reading it as open instead would suppress a target forever on a string nobody can
    explain, with no probe that could ever clear it.
    """
    await record_circuit_state(
        health,
        target_id=_T,
        circuit_state="quarantined",
        consecutive_fetch_failures=9,
        blocked_until=_NOW + timedelta(days=1),
    )
    assert (await TargetCircuit(health, policy=policy).check(_T, now=_NOW)).permitted


@pytest.mark.asyncio
async def test_an_unreadable_health_store_still_lets_the_target_be_fetched(
    health: ScrapeTargetHealthCollection, policy: BackoffPolicy
) -> None:
    """A store outage must degrade to pre-circuit behaviour, not stop scraping everything."""

    async def exploding_get(_entity_id: object, **_kwargs: object) -> object:
        raise RuntimeError("l3 pool is gone")

    health.get = exploding_get  # type: ignore[method-assign, assignment]
    assert (await TargetCircuit(health, policy=policy).check(_T, now=_NOW)).permitted


# ---------------------------------------------------------------------------
# Injected collaborators
# ---------------------------------------------------------------------------


# parity-with: threetears.scrape.circuit.ProbeObservableBreaker
class _FakeBreaker:
    """An in-process breaker that is already open."""

    def __init__(self) -> None:
        self.successes = 0
        self.failures = 0
        self.state = CircuitState.OPEN

    def check(self) -> None:
        raise CircuitOpenError("warn_oh", 12.0)

    def record_success(self) -> None:
        self.successes += 1

    def record_failure(self) -> None:
        self.failures += 1


# parity-with: threetears.core.coordination.windowed_counter.WindowedCounter
class _FakeFleetCounter:
    """Counts blocked observations the way a cross-pod windowed counter would."""

    def __init__(self, *, start: int = 0) -> None:
        self._count = start

    async def record_attempt(self, key: str) -> int:
        del key
        self._count += 1
        return self._count

    async def count(self, key: str) -> int:
        del key
        return self._count

    async def is_over_threshold(self, key: str, *, threshold: int) -> bool:
        del key
        return self._count > threshold


# parity-with: threetears.core.coordination.token_bucket.TokenBucket
class _FakeProbePacer:
    """A capacity-one bucket: the first claim wins, later ones are refused."""

    def __init__(self, *, tokens: int = 1) -> None:
        self.tokens = tokens
        self.claims = 0

    async def claim(self, key: str = "default", **_kwargs: object) -> object:
        del key
        self.claims += 1
        if self.tokens > 0:
            self.tokens -= 1
            return _Claim(claimed=True, retry_after_seconds=0.0)
        return _Claim(claimed=False, retry_after_seconds=45.0)


# parity-with: threetears.core.coordination.token_bucket.TokenClaimResult
class _Claim:
    def __init__(self, *, claimed: bool, retry_after_seconds: float) -> None:
        self.claimed = claimed
        self.tokens_remaining = 0.0
        self.retry_after_seconds = retry_after_seconds


# parity-with: threetears.scrape.circuit.ReprobeScheduler
class _FakeReprobeScheduler:
    def __init__(self, *, explode: bool = False) -> None:
        self.booked: list[tuple[str, float]] = []
        self._explode = explode

    async def schedule_reprobe(self, *, target_id: str, delay_seconds: float) -> None:
        if self._explode:
            raise RuntimeError("scheduler store is down")
        self.booked.append((target_id, delay_seconds))


@pytest.mark.asyncio
async def test_an_open_in_process_breaker_denies_before_any_io(health: ScrapeTargetHealthCollection) -> None:
    """The free question is asked first, exactly as the classifier's cheapest check is."""
    reads: list[str] = []
    original = health.get

    async def counting_get(entity_id: object, **kwargs: object) -> object:
        reads.append("read")
        return await original(entity_id, **kwargs)  # type: ignore[arg-type]

    health.get = counting_get  # type: ignore[method-assign, assignment]
    breaker = _FakeBreaker()
    decision = await TargetCircuit(health, breaker_for=lambda _target: breaker).check(_T, now=_NOW)

    assert not decision.permitted
    assert decision.retry_after_seconds == 12.0
    assert reads == []


@pytest.mark.asyncio
async def test_a_fleet_count_trips_the_circuit_that_one_pod_alone_would_not(
    health: ScrapeTargetHealthCollection, policy: BackoffPolicy
) -> None:
    """Two pods each seeing one block have seen two blocks.

    Without the shared count each pod carries its own share of a total that never reaches
    the threshold, so a target several pods are all being walled by is the one target whose
    circuit never opens. The count is fed in as the breaker's starting failure count rather
    than tripping the circuit separately, so the threshold rule stays in one place.
    """
    circuit = TargetCircuit(health, policy=policy, blocked_attempts=_FakeFleetCounter(start=1))
    await circuit.record_blocked(_T, now=_NOW)

    row = await health.get(_T)
    assert row is not None
    assert row.circuit_state == CircuitState.OPEN.value


@pytest.mark.asyncio
async def test_a_pacer_admits_one_probe_and_refuses_the_rest(
    health: ScrapeTargetHealthCollection, policy: BackoffPolicy
) -> None:
    """Cross-pod single-probe admission, which a restored breaker structurally cannot give.

    ``CircuitBreaker``'s in-flight-probe flag belongs to the process that set it, and a
    breaker hydrated from a row is a fresh object in a different process, so every pod would
    otherwise consider itself the one probe. A capacity-one bucket is the distributed form
    of the same guarantee.

    Two pods, modelled as two circuits that have each already read the row as OPEN, sharing
    nothing but the pacer -- which is exactly what they share in a real deployment. That
    read-before-either-writes window is the only place the pacer can matter: once one pod has
    written HALF_OPEN, the durable probe reservation refuses everyone else on its own, ahead
    of the pacer and without needing it. Driving both checks against one collection would
    therefore prove nothing about the pacer, because the second call would never reach it.
    """
    pacer = _FakeProbePacer(tokens=1)
    pod_a = TargetCircuit(health, policy=policy, probe_pacer=pacer)
    await pod_a.record_blocked(_T, now=_NOW)
    await pod_a.record_blocked(_T, now=_NOW)

    pod_b_health = ScrapeTargetHealthCollection(
        CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS"), nats_client=None
    )
    await record_circuit_state(
        pod_b_health,
        target_id=_T,
        circuit_state=CircuitState.OPEN.value,
        consecutive_fetch_failures=2,
        blocked_until=_NOW + timedelta(seconds=60),
    )
    pod_b = TargetCircuit(pod_b_health, policy=policy, probe_pacer=pacer)

    after = _NOW + timedelta(seconds=61)
    first = await pod_a.check(_T, now=after)
    second = await pod_b.check(_T, now=after)

    assert first.permitted and first.is_probe
    assert not second.permitted, "both pods probed the same target in the same window"
    assert second.retry_after_seconds == 45.0, "the refusal did not come from the pacer"
    assert pacer.claims == 2, "the fleet's probe slot was never consulted"


@pytest.mark.asyncio
async def test_opening_the_circuit_books_a_reprobe_for_when_the_window_expires(
    health: ScrapeTargetHealthCollection, policy: BackoffPolicy
) -> None:
    """An event-driven caller has no next poll, so something has to wake it."""
    scheduler = _FakeReprobeScheduler()
    circuit = TargetCircuit(health, policy=policy, reprobe_scheduler=scheduler)
    await circuit.record_blocked(_T, now=_NOW)
    assert scheduler.booked == []

    await circuit.record_blocked(_T, now=_NOW)
    assert scheduler.booked == [(_T, 60.0)]


@pytest.mark.asyncio
async def test_a_failed_booking_does_not_lose_the_backoff(
    health: ScrapeTargetHealthCollection, policy: BackoffPolicy
) -> None:
    """``blocked_until`` is already durable by then, so a lost booking costs a wake-up only."""
    circuit = TargetCircuit(health, policy=policy, reprobe_scheduler=_FakeReprobeScheduler(explode=True))
    await circuit.record_blocked(_T, now=_NOW)
    await circuit.record_blocked(_T, now=_NOW)

    row = await health.get(_T)
    assert row is not None
    assert row.circuit_state == CircuitState.OPEN.value
    assert row.blocked_until == _NOW + timedelta(seconds=60)


@pytest.mark.asyncio
async def test_a_suppressed_fetch_does_not_strand_the_in_process_breaker(
    health: ScrapeTargetHealthCollection, policy: BackoffPolicy
) -> None:
    """The two circuits run on different clocks, and the fast one must not deadlock.

    The in-process breaker is asked first, and an OPEN one whose own (short) recovery
    timeout has elapsed answers by promoting itself to HALF_OPEN and marking ITS probe in
    flight. If the durable circuit then suppresses the fetch -- which it will, because its
    window is minutes rather than seconds -- that probe never happens, and
    ``CircuitBreakerLike`` has no way to say "never mind": an admitted probe is cleared only
    by an outcome. Left set, the flag outlives the durable window and fast-fails this target
    forever with ``retry_after_seconds`` of zero, telling the caller to retry immediately
    into a circuit that will never admit it.

    Uses the real ``CircuitBreaker`` rather than a fake, because the in-flight flag IS the
    behaviour under test and a fake would be asserting against my own model of it.
    """
    breaker = CircuitBreaker("warn_oh", failure_threshold=1, recovery_timeout_seconds=0.0)
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN

    circuit = TargetCircuit(health, policy=policy, breaker_for=lambda _target: breaker)
    await record_circuit_state(
        health,
        target_id=_T,
        circuit_state=CircuitState.OPEN.value,
        consecutive_fetch_failures=2,
        blocked_until=_NOW + timedelta(seconds=300),
    )

    first = await circuit.check(_T, now=_NOW)
    assert not first.permitted
    assert breaker.state is CircuitState.OPEN, "the in-process breaker was left holding a phantom probe"

    second = await circuit.check(_T, now=_NOW + timedelta(seconds=1))
    assert not second.permitted
    assert second.retry_after_seconds > 1.0, (
        "the caller was told to retry immediately into a circuit that cannot admit it -- "
        "the stranded in-process probe answered instead of the durable window"
    )


@pytest.mark.asyncio
async def test_a_suppressed_fetch_invents_no_failure_on_a_closed_in_process_breaker(
    health: ScrapeTargetHealthCollection, policy: BackoffPolicy
) -> None:
    """The mirror of the test above, and the far more common case of the two.

    A CLOSED breaker's ``check()`` returns without admitting anything, so there is no probe
    to release and reporting a failure to it is inventing one. The two circuits run on very
    different clocks -- seconds against minutes-to-hours -- so a handful of suppressed polls
    is enough to trip a breaker that never once saw a failed fetch. After that the WRONG
    circuit answers, and the caller is told to retry in seconds when the truth is the durable
    window. An LLM holding this tool obeys that hint, so the suppression it was given turns
    into a fixed-cadence poll forever.
    """
    breaker = CircuitBreaker(_T, failure_threshold=2, recovery_timeout_seconds=30.0)
    assert breaker.state is CircuitState.CLOSED

    circuit = TargetCircuit(health, policy=policy, breaker_for=lambda _target: breaker)
    await record_circuit_state(
        health,
        target_id=_T,
        circuit_state=CircuitState.OPEN.value,
        consecutive_fetch_failures=2,
        blocked_until=_NOW + timedelta(seconds=300),
    )

    for poll in range(4):
        decision = await circuit.check(_T, now=_NOW + timedelta(seconds=poll))
        assert not decision.permitted
        assert decision.retry_after_seconds > 250.0, (
            "the in-process breaker answered instead of the durable window, so the caller "
            "was told to retry orders of magnitude too soon"
        )

    assert breaker.failure_count == 0, "suppressed fetches were counted as failures against a breaker that saw none"
    assert breaker.state is CircuitState.CLOSED


@pytest.mark.asyncio
async def test_a_stuck_half_open_row_still_decays_when_a_probe_pacer_is_configured(
    health: ScrapeTargetHealthCollection, policy: BackoffPolicy
) -> None:
    """A token bucket and the probe reservation answer different questions.

    The bucket bounds how MANY pods probe at once, and it refills at a constant rate. The
    reservation bounds how OFTEN a stuck target is probed, on the doubling curve. Deferring
    the reservation to the bucket whenever one is configured therefore swaps the decay for a
    floor: a row left HALF_OPEN by a caller that died before reporting gets re-probed at the
    refill rate forever, which is exactly the fixed-cadence polling this module exists to
    remove -- and it only happens in the deployment that bothered to configure a pacer.

    The pacer here always grants, so the bucket cannot be what suppresses the fetch; if the
    reservation is not honoured, nothing is.
    """
    circuit = TargetCircuit(health, policy=policy, probe_pacer=_FakeProbePacer(tokens=99))
    await circuit.record_blocked(_T, now=_NOW)
    await circuit.record_blocked(_T, now=_NOW)

    probe_at = _NOW + timedelta(seconds=61)
    assert (await circuit.check(_T, now=probe_at)).is_probe
    # The caller dies here, leaving the row HALF_OPEN with its reservation stamped.

    for poll in range(1, 4):
        decision = await circuit.check(_T, now=probe_at + timedelta(seconds=poll))
        assert not decision.permitted, (
            "a configured pacer disabled the probe reservation, so a stuck row was re-probed "
            "at the bucket's refill rate instead of the backoff curve"
        )
        assert decision.retry_after_seconds > 0.0

    # And it is still a reservation, not a life sentence.
    assert (await circuit.check(_T, now=probe_at + timedelta(seconds=61))).is_probe


@pytest.mark.asyncio
async def test_an_abandoned_probe_can_be_released_by_the_caller_that_abandoned_it(
    health: ScrapeTargetHealthCollection, policy: BackoffPolicy
) -> None:
    """The permitted path's version of the stranding the suppressed path already handles.

    ``check`` saying yes can promote the in-process breaker to HALF_OPEN and mark ITS probe
    in flight. If the caller then raises before reporting an outcome, nothing ever clears
    that flag: every later ``check`` fast-fails on the in-process branch before the durable
    row is read, with ``retry_after_seconds`` of 0.0 -- an infinite "retry immediately" into
    a circuit that will never admit anything.

    Uses the real ``CircuitBreaker`` because the in-flight flag IS the behaviour under test.
    """
    breaker = CircuitBreaker(_T, failure_threshold=1, recovery_timeout_seconds=0.0)
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN

    circuit = TargetCircuit(health, policy=policy, breaker_for=lambda _target: breaker)

    # A permitted fetch: the durable circuit is closed, and the in-process breaker promotes
    # itself to HALF_OPEN on the way through, taking its own probe slot.
    assert (await circuit.check(_T, now=_NOW)).permitted
    assert breaker.state is CircuitState.HALF_OPEN

    # The caller raises before reporting any outcome, and says so.
    circuit.release_probe(_T)

    decision = await circuit.check(_T, now=_NOW + timedelta(seconds=1))
    assert decision.permitted, (
        "the abandoned probe was never released, so the in-process breaker fast-failed the "
        "target before the durable row was even read"
    )


@pytest.mark.asyncio
async def test_releasing_a_probe_nobody_admitted_invents_no_failure(
    health: ScrapeTargetHealthCollection, policy: BackoffPolicy
) -> None:
    """``release_probe`` is safe to call blind, which is what lets it live in an except block.

    A caller cannot tell whether its permitted decision promoted the in-process breaker, and
    should not have to: the alternative is an exception handler that has to reason about
    circuit state to avoid corrupting it. Releasing what was never admitted would fabricate
    failures on a healthy breaker at seconds-scale timeouts, and the wrong circuit would then
    be the one answering.
    """
    breaker = CircuitBreaker(_T, failure_threshold=2, recovery_timeout_seconds=30.0)
    circuit = TargetCircuit(health, policy=policy, breaker_for=lambda _target: breaker)

    for _ in range(4):
        circuit.release_probe(_T)

    assert breaker.failure_count == 0, "releasing an unadmitted probe was recorded as a failed fetch"
    assert breaker.state is CircuitState.CLOSED


@pytest.mark.asyncio
async def test_one_targets_in_process_breaker_is_not_another_targets(
    health: ScrapeTargetHealthCollection, policy: BackoffPolicy
) -> None:
    """Everything else here is keyed by target, and this seam must not drop the key.

    One ``TargetCircuit`` serves every target its tool scrapes. A single shared breaker would
    let one walled target fast-fail every other target on the same tool, and would let a
    healthy target's success wipe the failure count a different target had accumulated --
    both of which the registry this delegates to already gets right, per key.
    """
    registry = CircuitBreakerRegistry(failure_threshold=1, recovery_timeout_seconds=300.0)
    other = "some_other_target"
    circuit = TargetCircuit(health, policy=policy, breaker_for=registry.get)

    await circuit.record_unreachable(_T, now=_NOW)
    assert registry.get(_T).state is CircuitState.OPEN

    assert (await circuit.check(other, now=_NOW)).permitted, "one walled target fast-failed an unrelated one"
    await circuit.record_reachable(other, now=_NOW)
    assert registry.get(_T).state is CircuitState.OPEN, "a healthy target's success reset another target's breaker"


@pytest.mark.asyncio
async def test_a_probe_that_never_reports_back_does_not_re_probe_every_poll(
    health: ScrapeTargetHealthCollection, policy: BackoffPolicy
) -> None:
    """The one hole in "the fetch rate decays", and it is reachable from a raising caller.

    ``CircuitBreaker`` bounds concurrent probes with an in-flight flag, but ``restore()``
    cannot carry one across a process boundary, so its HALF_OPEN branch consults no timer:
    a row left HALF_OPEN admits a fresh probe on every single poll. A caller that raises
    between the fetch and the outcome report leaves exactly that row, and the target is then
    fetched at full rate for as long as the fault lasts -- the decay silently gone while
    every individual state transition remains correct.
    """
    circuit = TargetCircuit(health, policy=policy)
    await circuit.record_blocked(_T, now=_NOW)
    await circuit.record_blocked(_T, now=_NOW)

    probe_at = _NOW + timedelta(seconds=61)
    assert (await circuit.check(_T, now=probe_at)).is_probe
    # The caller dies here: no record_blocked, no record_reachable, row left HALF_OPEN.

    for poll in range(1, 4):
        decision = await circuit.check(_T, now=probe_at + timedelta(seconds=poll))
        assert not decision.permitted, "an unreported probe let the target be fetched again on the very next poll"
        assert decision.retry_after_seconds > 0.0

    # And the reservation is a reservation, not a life sentence: once it expires the probe is
    # evidently dead and the target is due another one.
    assert (await circuit.check(_T, now=probe_at + timedelta(seconds=61))).is_probe


@pytest.mark.asyncio
async def test_an_unreadable_health_store_does_not_erase_an_open_circuit(
    health: ScrapeTargetHealthCollection, policy: BackoffPolicy
) -> None:
    """A failed read is the absence of a fact, not the fact that there is nothing.

    ``check`` collapses the two safely -- unreadable or absent, it fetches. A writer must
    not: restoring from "nothing" starts at CLOSED/0 and would then persist a closed circuit
    with no window over a target that is open and backed off, throwing away the suppression
    on the one poll where the store was flaky.

    Only the circuit's OWN read fails here. A store down hard takes the write with it (the
    merge does its own read), so the two failures cancel and nothing is persisted either way
    -- which hides the bug rather than proving there is none. The case that bites is the
    partially degraded one, a read that times out against a store still healthy enough to
    accept the write, so that is what this drives.
    """
    await record_circuit_state(
        health,
        target_id=_T,
        circuit_state=CircuitState.OPEN.value,
        consecutive_fetch_failures=4,
        blocked_until=_NOW + timedelta(seconds=300),
    )
    original = health.get
    failed_once = False

    async def flaky_get(entity_id: object, **kwargs: object) -> object:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise RuntimeError("health store read timed out")
        return await original(entity_id, **kwargs)  # type: ignore[arg-type]

    health.get = flaky_get  # type: ignore[method-assign, assignment]
    await TargetCircuit(health, policy=policy).record_blocked(_T, now=_NOW)
    assert failed_once, "the circuit never read the row, so this test proved nothing"

    health.get = original  # type: ignore[method-assign]
    row = await health.get(_T)
    assert row is not None
    assert row.circuit_state == CircuitState.OPEN.value, "a failed read was written back as a closed circuit"
    assert row.consecutive_fetch_failures == 4
    assert row.blocked_until == _NOW + timedelta(seconds=300)
