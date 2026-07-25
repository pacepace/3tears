"""Stop hammering a wall, and stop paying to look at it.

Telling a bot wall apart from a site redesign keeps a blocked target's recipe intact, but it
does not make a blocked target cheap. A walled target still gets fetched on every poll, and
every one of those fetches produces a page that fails extraction and therefore gets
classified. The classifier's verdict cache does not bound that: it keys on a digest of the
page's visible text, and a real interstitial renders a per-request id into exactly that text,
so the cache misses and the target re-classifies every poll, forever. A target walled before
it ever won a recipe is worse still -- it pays a whole candidate-generation round first.

The only thing that bounds both costs is not fetching the target. So this module gates the
FETCH, and it does so with a durable circuit: repeated blocks trip it open, an open circuit
suppresses the fetch until a backoff window elapses, and the window grows each time a probe
finds the wall still standing. Fetch rate decays; classification rate decays with it, because
classification is downstream of a fetch that no longer happens.

**Nothing here is a new state machine.** The three states, the failure threshold, the
promotion of OPEN to HALF_OPEN after a recovery timeout, and what a probe's outcome does are
all ``threetears.models.circuit_breaker.CircuitBreaker``'s, reached through its ``restore()``
seam: the durable row is hydrated into a real breaker, the transition is driven by calling
that breaker, and the resulting state is written back. What lives here is storage, backoff
arithmetic, and the decision of which outcome counts as a fetch failure.

Four collaborators are optional and injected, never constructed, because each belongs to
infrastructure this package does not own:

- ``breaker`` -- any ``threetears.core.http_client.CircuitBreakerLike``, typically a
  ``CircuitBreaker`` from a ``CircuitBreakerRegistry``. A free in-process fast-fail that
  answers before the health row is even read. The same structural seam ``core`` already uses
  to depend on a breaker without importing ``threetears.models``.
- ``blocked_attempts`` -- a ``threetears.core.coordination.windowed_counter.WindowedCounter``.
  Blocked observations counted across a fleet, so several pods polling one target trip its
  circuit as fast as one pod would rather than each accumulating its own share of a count
  that never reaches the threshold.
- ``probe_pacer`` -- a ``threetears.core.coordination.token_bucket.TokenBucket``. Cross-pod
  single-probe admission: ``CircuitBreaker``'s own in-flight-probe flag is process-local and
  cannot see another pod's probe, so the distributed version of that guarantee comes from a
  bucket of capacity one.
- ``reprobe_scheduler`` -- books the next probe as a ``3tears-scheduled-jobs``
  ``relative_delay`` job (see :mod:`threetears.scrape.reprobe`), for a caller that is
  event-driven rather than polling. No sleep-and-retry loop is written here.

All four are ``None`` by default, and with all four absent the circuit still decays a blocked
target's fetch rate using nothing but the health row. They add cross-pod correctness and
event-driven wake-up, not the core behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

from threetears.core.http_client import CircuitBreakerLike
from threetears.models.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState
from threetears.observe import get_logger

from .health import ScrapeTargetHealth, ScrapeTargetHealthCollection, record_circuit_state

if TYPE_CHECKING:
    # Type-only: both are constructed from a NATS client the caller already holds, and
    # importing them at runtime would make 3tears-nats a hard dependency of this package
    # for a capability that is optional in it.
    from threetears.core.coordination.token_bucket import TokenBucket
    from threetears.core.coordination.windowed_counter import WindowedCounter

__all__ = [
    "BackoffPolicy",
    "FetchDecision",
    "ReprobeScheduler",
    "TargetCircuit",
]

log = get_logger(__name__)

#: Guard on the backoff exponent. ``2.0 ** 1024`` overflows to ``inf`` and beyond that
#: raises, so a target whose failure count has run away must not be able to turn a delay
#: computation into an exception. The cap is far past where :attr:`BackoffPolicy.max_delay_seconds`
#: has already flattened the curve, so it never changes a delay anyone will observe.
_MAX_BACKOFF_DOUBLINGS = 32

#: Stored ``circuit_state`` values this module will act on. A value outside the set is read
#: as CLOSED: it is either from a version that meant something different or it is corruption,
#: and the safe reading of an uninterpretable circuit is "not currently suppressing fetches"
#: -- the failure mode is one wasted fetch, where the opposite is a target suppressed forever
#: on a value nobody can explain.
_KNOWN_CIRCUIT_STATES: dict[str, CircuitState] = {state.value: state for state in CircuitState}


class ReprobeScheduler(Protocol):
    """Books a blocked target's next probe with whatever schedules work in this deployment.

    A Protocol rather than a concrete dependency because ``3tears-scheduled-jobs``, which is
    what satisfies it, brings NATS and APScheduler with it -- real weight for a capability a
    polling caller does not need at all, since a poller's next poll IS the re-probe.
    :class:`threetears.scrape.reprobe.ScheduledJobsReprobeScheduler` is the implementation,
    and it books a ``relative_delay`` job rather than sleeping.
    """

    async def schedule_reprobe(self, *, target_id: str, delay_seconds: float) -> None:
        """Arrange for *target_id* to be probed again in *delay_seconds*."""
        ...


@dataclass(frozen=True)
class BackoffPolicy:
    """How hard to back off a target that keeps coming back walled.

    :param failure_threshold: consecutive blocked or unreachable fetches before the circuit
        opens. Below it nothing is suppressed, so a transient block costs nothing.
    :ptype failure_threshold: int
    :param base_delay_seconds: how long the first trip suppresses fetches for.
    :ptype base_delay_seconds: float
    :param max_delay_seconds: ceiling on the delay. A wall is not usually permanent, and a
        target nobody ever re-probes is a target that stays broken after the block is lifted.
    :ptype max_delay_seconds: float
    """

    failure_threshold: int = 3
    base_delay_seconds: float = 900.0
    max_delay_seconds: float = 21600.0

    def delay_for(self, consecutive_failures: int) -> float:
        """Seconds to suppress fetches for, after *consecutive_failures* failures.

        Doubles per failure past the threshold and then flattens at the ceiling, so the
        trip that first opens the circuit waits :attr:`base_delay_seconds` and each probe
        that finds the wall still up doubles the wait. That is what makes the fetch rate
        decay rather than merely drop to a constant.

        :param consecutive_failures: the failure count that just tripped or re-tripped
        :ptype consecutive_failures: int
        :return: seconds until the next fetch is permitted
        :rtype: float
        """
        doublings = min(max(0, consecutive_failures - self.failure_threshold), _MAX_BACKOFF_DOUBLINGS)
        return min(self.max_delay_seconds, self.base_delay_seconds * (2.0**doublings))


@dataclass(frozen=True)
class FetchDecision:
    """Whether to fetch a target right now, and what the circuit thought.

    Carries the reason and the wait rather than just a boolean, because a caller that
    suppresses a fetch has to tell its own caller something more useful than "no": an
    operator, or an LLM holding this tool, needs to know it is being backed off rather than
    that the target is broken.
    """

    permitted: bool
    state: CircuitState
    retry_after_seconds: float
    is_probe: bool
    reason: str


class TargetCircuit:
    """The durable fetch circuit for a set of scrape targets.

    Read :meth:`check` before fetching; report the outcome afterwards through exactly one of
    :meth:`record_blocked`, :meth:`record_unreachable`, or :meth:`record_reachable`.

    Deliberately sits at the FETCH boundary rather than inside the eval loop. The eval loop
    is handed a page that has already been fetched, so a gate there could only suppress work
    downstream of the cost this exists to avoid. Its caller is the one holding the driver.
    """

    def __init__(
        self,
        health_collection: ScrapeTargetHealthCollection,
        *,
        policy: BackoffPolicy | None = None,
        breaker: CircuitBreakerLike | None = None,
        blocked_attempts: WindowedCounter | None = None,
        probe_pacer: TokenBucket | None = None,
        reprobe_scheduler: ReprobeScheduler | None = None,
    ) -> None:
        """
        :param health_collection: where the durable circuit state lives
        :ptype health_collection: ScrapeTargetHealthCollection
        :param policy: threshold and backoff curve; defaults are used when omitted
        :ptype policy: BackoffPolicy | None
        :param breaker: optional in-process fast-fail, checked before any I/O
        :ptype breaker: CircuitBreakerLike | None
        :param blocked_attempts: optional cross-pod counter of blocked observations
        :ptype blocked_attempts: WindowedCounter | None
        :param probe_pacer: optional capacity-one bucket admitting one probe per fleet
        :ptype probe_pacer: TokenBucket | None
        :param reprobe_scheduler: optional booking of the next probe as a scheduled job
        :ptype reprobe_scheduler: ReprobeScheduler | None
        """
        self._health = health_collection
        self._policy = policy or BackoffPolicy()
        self._breaker = breaker
        self._blocked_attempts = blocked_attempts
        self._probe_pacer = probe_pacer
        self._reprobe_scheduler = reprobe_scheduler

    async def check(self, target_id: str, *, now: datetime | None = None) -> FetchDecision:
        """Decide whether *target_id* may be fetched right now.

        Cheapest question first, exactly as the classifier does it: the in-process breaker
        answers with no I/O at all, then the durable row is read, and only a circuit that is
        actually open does any further work.

        A permitted decision with ``is_probe`` set is the single recovery attempt for an
        expired backoff window. The OPEN-to-HALF_OPEN promotion that produces it is written
        back, so another pod reads the target as probing rather than as still open.

        Never raises. An unreadable health store degrades this target to the behaviour it
        had before the circuit existed -- fetching -- because a store outage must not
        silently stop scraping everything.

        :param target_id: the target about to be fetched
        :ptype target_id: str
        :param now: the current time; injected by tests, defaults to now
        :ptype now: datetime | None
        :return: whether to fetch, and what the circuit thought
        :rtype: FetchDecision
        """
        moment = now or datetime.now(UTC)

        if self._breaker is not None:
            try:
                self._breaker.check()
            except CircuitOpenError as exc:
                return FetchDecision(
                    permitted=False,
                    state=CircuitState.OPEN,
                    retry_after_seconds=exc.remaining_seconds,
                    is_probe=False,
                    reason="in-process circuit breaker is open for this target",
                )

        health = await self._read_health(target_id)
        if health is None or _stored_state(health.circuit_state) is CircuitState.CLOSED:
            return FetchDecision(
                permitted=True,
                state=CircuitState.CLOSED,
                retry_after_seconds=0.0,
                is_probe=False,
                reason="circuit closed",
            )

        was = _stored_state(health.circuit_state)
        restored = self._restore(target_id, health, moment)
        try:
            restored.check()
        except CircuitOpenError as exc:
            log.info(
                "scrape circuit: suppressing the fetch of target %s for another %.0fs",
                target_id,
                exc.remaining_seconds,
                extra={"extra_data": {"target_id": target_id, "circuit_state": CircuitState.OPEN.value}},
            )
            return FetchDecision(
                permitted=False,
                state=CircuitState.OPEN,
                retry_after_seconds=exc.remaining_seconds,
                is_probe=False,
                reason="the target is behind a wall and its backoff window has not elapsed",
            )

        # Past `check()` on a non-closed circuit means the recovery window elapsed and this
        # request is the probe.
        paced = await self._claim_probe(target_id)
        if not paced.permitted:
            return paced

        if was is not CircuitState.HALF_OPEN:
            await self._write(
                target_id,
                state=CircuitState.HALF_OPEN,
                failures=health.consecutive_fetch_failures,
                blocked_until=health.blocked_until,
            )
        log.info(
            "scrape circuit: probing target %s after backoff",
            target_id,
            extra={"extra_data": {"target_id": target_id, "circuit_state": CircuitState.HALF_OPEN.value}},
        )
        return FetchDecision(
            permitted=True,
            state=CircuitState.HALF_OPEN,
            retry_after_seconds=0.0,
            is_probe=True,
            reason="single recovery probe after the backoff window elapsed",
        )

    async def record_blocked(self, target_id: str, *, now: datetime | None = None) -> None:
        """Count a fetch that came back a wall, and open the circuit once that is the pattern.

        :param target_id: the target that came back walled
        :ptype target_id: str
        :param now: the current time; injected by tests, defaults to now
        :ptype now: datetime | None
        """
        await self._record_failure(target_id, now=now, blocked=True)

    async def record_unreachable(self, target_id: str, *, now: datetime | None = None) -> None:
        """Count a fetch that never produced a page: a transport error, a timeout, a refusal.

        Shares the circuit with :meth:`record_blocked` because both mean the same thing to a
        fetch schedule -- we did not receive the content and retrying immediately will not
        change that. They stay distinguishable on the row: only a wall stamps
        ``last_blocked_at``, so "when was this target last walled" keeps its meaning instead
        of quietly becoming "when did anything last go wrong".

        :param target_id: the target that could not be reached
        :ptype target_id: str
        :param now: the current time; injected by tests, defaults to now
        :ptype now: datetime | None
        """
        await self._record_failure(target_id, now=now, blocked=False)

    async def record_reachable(self, target_id: str, *, now: datetime | None = None) -> None:
        """Report that *target_id* served real content, and close its circuit.

        Called for every outcome that is not a wall and not a transport failure, including a
        page whose extraction failed: the fetch worked, and this circuit counts fetches. An
        extraction that keeps failing against a page we can plainly see is the recipe's
        problem, and ``ScrapeRecipe.consecutive_validation_failures`` is already counting it.

        Writes nothing when the row is already closed at zero, which is the overwhelmingly
        common case. A healthy target polled every few minutes must not be paying a
        read-modify-write on a health row for the privilege of still being healthy.

        :param target_id: the target that served content
        :ptype target_id: str
        :param now: the current time; injected by tests, defaults to now
        :ptype now: datetime | None
        """
        moment = now or datetime.now(UTC)
        if self._breaker is not None:
            self._breaker.record_success()

        health = await self._read_health(target_id)
        if health is None:
            return
        if _stored_state(health.circuit_state) is CircuitState.CLOSED and health.consecutive_fetch_failures == 0:
            return

        restored = self._restore(target_id, health, moment)
        restored.record_success()
        log.info(
            "scrape circuit: target %s is reachable again; circuit closed",
            target_id,
            extra={"extra_data": {"target_id": target_id, "circuit_state": restored.state.value}},
        )
        await self._write(target_id, state=restored.state, failures=restored.failure_count, blocked_until=None)

    async def _record_failure(self, target_id: str, *, now: datetime | None, blocked: bool) -> None:
        """Drive one failure through the breaker's rules and persist what it decided."""
        moment = now or datetime.now(UTC)
        if self._breaker is not None:
            self._breaker.record_failure()

        health = await self._read_health(target_id)
        observed = health.consecutive_fetch_failures if health is not None else 0
        if blocked:
            observed = max(observed, await self._fleet_count(target_id))

        restored = self._restore(target_id, health, moment, failure_count=observed)
        restored.record_failure()
        state = restored.state
        failures = restored.failure_count

        blocked_until: datetime | None = None
        if state is CircuitState.OPEN:
            delay = self._policy.delay_for(failures)
            blocked_until = moment + timedelta(seconds=delay)
            log.warning(
                "scrape circuit: target %s opened after %d consecutive failed fetches; suppressing fetches for %.0fs",
                target_id,
                failures,
                delay,
                extra={"extra_data": {"target_id": target_id, "circuit_state": state.value}},
            )

        await self._write(
            target_id,
            state=state,
            failures=failures,
            blocked_until=blocked_until,
            blocked_at=moment if blocked else None,
        )
        if blocked_until is not None:
            await self._book_reprobe(target_id, (blocked_until - moment).total_seconds())

    def _restore(
        self,
        target_id: str,
        health: ScrapeTargetHealth | None,
        now: datetime,
        *,
        failure_count: int | None = None,
    ) -> CircuitBreaker:
        """Hydrate the shipped breaker from the durable row so its rules make the decision.

        The recovery timeout handed to it is this policy's delay for the count the row
        already carries, and the remaining time comes from ``blocked_until``. Those agree by
        construction, because ``blocked_until`` was written as exactly that delay past the
        failure that produced it; where a policy change has made them disagree,
        ``blocked_until`` wins, which is the right way round -- it is the value an operator
        can read, query and reason about.
        """
        failures = failure_count if failure_count is not None else (health.consecutive_fetch_failures if health else 0)
        blocked_until = health.blocked_until if health is not None else None
        remaining = (blocked_until - now).total_seconds() if blocked_until is not None else 0.0
        return CircuitBreaker.restore(
            target_id,
            state=_stored_state(health.circuit_state) if health is not None else CircuitState.CLOSED,
            failure_count=failures,
            seconds_until_probe_permitted=remaining,
            failure_threshold=self._policy.failure_threshold,
            recovery_timeout_seconds=self._policy.delay_for(failures),
        )

    async def _claim_probe(self, target_id: str) -> FetchDecision:
        """Admit at most one probe per fleet, when a pacer is available to say so.

        ``CircuitBreaker`` admits exactly one probe per PROCESS through its in-flight flag,
        which is the guarantee a restored breaker structurally cannot offer: the flag belongs
        to whichever process set it. A capacity-one token bucket is the same guarantee across
        pods.
        """
        if self._probe_pacer is None:
            return FetchDecision(
                permitted=True,
                state=CircuitState.HALF_OPEN,
                retry_after_seconds=0.0,
                is_probe=True,
                reason="no probe pacer configured",
            )
        try:
            claim = await self._probe_pacer.claim(target_id)
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- pacing is a cross-pod refinement; a KV outage must not strand every walled target permanently unprobed, so it degrades to the row-only behaviour. Logged with its traceback below
            log.exception(
                "scrape circuit: probe pacer unavailable for target %s; admitting the probe unpaced",
                target_id,
                extra={"extra_data": {"target_id": target_id}},
            )
            return FetchDecision(
                permitted=True,
                state=CircuitState.HALF_OPEN,
                retry_after_seconds=0.0,
                is_probe=True,
                reason="probe pacer unavailable; probe admitted unpaced",
            )
        if claim.claimed:
            return FetchDecision(
                permitted=True,
                state=CircuitState.HALF_OPEN,
                retry_after_seconds=0.0,
                is_probe=True,
                reason="claimed the fleet's probe slot for this target",
            )
        return FetchDecision(
            permitted=False,
            state=CircuitState.HALF_OPEN,
            retry_after_seconds=claim.retry_after_seconds,
            is_probe=False,
            reason="another pod is already probing this target",
        )

    async def _fleet_count(self, target_id: str) -> int:
        """Blocked observations for *target_id* across the fleet, or 0 with no counter.

        Returned as a count of PRIOR observations, so the caller adds this one itself by
        driving the breaker -- the counter's own return includes the attempt just recorded.

        Feeding this in as the breaker's starting failure count, rather than tripping the
        circuit separately on it, keeps the threshold rule in exactly one place. Two pods
        each seeing two blocks reach four together, which is what should trip a threshold of
        three; two per-pod counters of two never reach it.

        The count is windowed and does not reset on success, so a target that recovers and is
        walled again inside the window re-trips faster than a first-time block. That is
        intended: the window is the memory that a per-target row's "consecutive" counter
        deliberately does not have.
        """
        if self._blocked_attempts is None:
            return 0
        try:
            return max(0, await self._blocked_attempts.record_attempt(target_id) - 1)
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- the fleet count only ever raises the failure count; losing it degrades to the per-row count, never to a missed block. Logged with its traceback below
            log.exception(
                "scrape circuit: fleet blocked-attempt count unavailable for target %s; using the row's own count",
                target_id,
                extra={"extra_data": {"target_id": target_id}},
            )
            return 0

    async def _book_reprobe(self, target_id: str, delay_seconds: float) -> None:
        """Ask the scheduler to wake something up when the backoff window expires."""
        if self._reprobe_scheduler is None:
            return
        try:
            await self._reprobe_scheduler.schedule_reprobe(target_id=target_id, delay_seconds=delay_seconds)
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- the block is already durable in blocked_until, so a failed booking costs a wake-up, not correctness; a polling caller never needed one. Logged with its traceback below
            log.exception(
                "scrape circuit: could not book the re-probe for target %s; its backoff window still stands",
                target_id,
                extra={"extra_data": {"target_id": target_id}},
            )

    async def _read_health(self, target_id: str) -> ScrapeTargetHealth | None:
        """Read the durable row, degrading an unreadable store to "no circuit state"."""
        try:
            return await self._health.get(target_id)
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- an unreadable health store must degrade this target to its pre-circuit behaviour, not fail every poll. Logged with its traceback below
            log.exception(
                "scrape circuit: could not read health for target %s; treating its circuit as closed",
                target_id,
                extra={"extra_data": {"target_id": target_id}},
            )
            return None

    async def _write(
        self,
        target_id: str,
        *,
        state: CircuitState,
        failures: int,
        blocked_until: datetime | None,
        blocked_at: datetime | None = None,
    ) -> None:
        """Persist the circuit, never letting the bookkeeping cost the caller its fetch."""
        try:
            await record_circuit_state(
                self._health,
                target_id=target_id,
                circuit_state=state.value,
                consecutive_fetch_failures=failures,
                blocked_until=blocked_until,
                blocked_at=blocked_at,
            )
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- same posture as every other health write in this package: a bookkeeping failure must not turn a completed fetch into a failed one. Logged with its traceback below
            log.exception(
                "scrape circuit: could not persist the circuit state for target %s",
                target_id,
                extra={"extra_data": {"target_id": target_id}},
            )


def _stored_state(value: str) -> CircuitState:
    """Read a stored ``circuit_state`` string, defaulting an unrecognised one to CLOSED."""
    return _KNOWN_CIRCUIT_STATES.get(value, CircuitState.CLOSED)
