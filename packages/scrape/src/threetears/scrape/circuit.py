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

- ``breaker_for`` -- resolves a :class:`ProbeObservableBreaker` (the
  ``threetears.core.http_client.CircuitBreakerLike`` three-call protocol plus a readable
  ``state``) for one target. A free in-process fast-fail that answers before the health row
  is even read. The same structural seam ``core`` already uses to depend on a breaker
  without importing ``threetears.models``. It is a lookup rather than a single breaker
  because everything else here is keyed by target: one instance serves a whole set of
  targets, so a bare breaker would let one walled target fast-fail every other target the
  same tool scrapes, and let one healthy target's success reset the count a different
  target accumulated.

  The caller owns this lookup, including its lifetime, and for a long-lived process that
  second part matters. ``CircuitBreakerRegistry.get`` fits the signature and is the obvious
  thing to reach for, but the registry holds a plain dict with no eviction: bounded when the
  key is a provider name, unbounded when the key is a scrape target, because
  ``ScrapeTool._derive_target_id`` mints a fresh ``adhoc_<sha256>`` per distinct
  ``(url, field_schema)``. Handed straight to a long-running tool it accumulates one breaker
  per URL ever scraped. A short-lived process can ignore that; a long-lived one should inject
  a lookup bounded the way it wants bounding -- an LRU, a TTL, a registry it prunes. Nothing
  here evicts on the caller's behalf, because a cache policy for someone else's process is
  not this module's to pick, and a wrong guess here silently discards circuit state that a
  walled target is relying on.
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

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, NamedTuple, Protocol

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
    "ProbeObservableBreaker",
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


class ProbeObservableBreaker(CircuitBreakerLike, Protocol):
    """A :class:`~threetears.core.http_client.CircuitBreakerLike` whose state can be read.

    The three-call protocol has no way to say "never mind" about a probe it admitted: an
    in-flight probe is cleared only by an outcome. So resolving a probe that will never
    reach the target means first knowing whether one was admitted at all, and that is what
    ``state`` is for -- reporting a failure to a breaker that admitted nothing invents one,
    and the in-process recovery timeout is seconds where the durable window is minutes to
    hours, so a handful of fabricated failures put the WRONG circuit in charge of the answer.

    Declared rather than probed for. Reading ``state`` off a bare ``CircuitBreakerLike`` and
    treating its absence as "no probe admitted" looks conservative and is not: a breaker that
    satisfies the three calls, has an in-flight-probe concept, and does not expose ``state``
    is never released, so it stays HALF_OPEN with its probe held, every later ``check()``
    raises, no fetch happens, no outcome is recorded, and that target is wedged for the life
    of the process -- with a type signature that disclosed none of it. Requiring the attribute
    makes the constraint checkable at the seam instead of discoverable in production.

    ``threetears.models.circuit_breaker.CircuitBreaker`` satisfies this, so
    ``CircuitBreakerRegistry.get`` still fits the ``breaker_for`` parameter directly, subject
    to the lifetime caveat in the module docstring: that registry never evicts, which is
    bounded by provider name and not by scrape target.
    """

    @property
    def state(self) -> CircuitState:
        """The breaker's current state."""
        ...


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

    async def cancel_reprobe(self, *, target_id: str) -> None:
        """Drop any outstanding re-probe booking for *target_id*.

        Called when the circuit closes, which is the one outcome a re-booking cannot
        supersede: every other outcome books a fresh probe over the same key, and a close
        books nothing, so without this the last booking survives and fires against a target
        that recovered. That wake-up is a whole poll, and it also leaves a row behind per
        target that ever tripped.

        Must be idempotent and must not raise for a booking that is not there: the caller
        cannot know whether one is outstanding without asking, and asking is a round trip to
        answer a question the delete already answers.
        """
        ...


@dataclass(frozen=True)
class BackoffPolicy:
    """How hard to back off a target that keeps coming back walled.

    The three defaults are judgement, not measurement, and are recorded here as such so a
    later reader tunes them against a reason rather than against a number of unknown
    provenance. Each is a `BackoffPolicy` field precisely because the right value is
    deployment-specific; nothing here reads a constant it did not receive.

    - ``failure_threshold=3`` -- two is a coin toss. A single interstitial served to a warm
      cache, a redeploy, one timeout: any of those produces one failure, and a pair of them
      in a row is ordinary. Three consecutive failures against a target that was working is
      the first count that is more cheaply explained by "this target is walled" than by bad
      luck, and the cost of being wrong is one suppressed poll.
    - ``base_delay_seconds=900.0`` (15 minutes) -- sized against the poll interval it
      protects, not against any vendor's documented cooldown, since walls do not publish one.
      A scrape target polled every few minutes needs a first backoff long enough to skip
      several polls, or the circuit reads as open while the fetch rate barely moves. Fifteen
      minutes skips a handful and is still short enough that a target unblocked by a human
      minutes ago is not left waiting an afternoon.
    - ``max_delay_seconds=21600.0`` (6 hours) -- the ceiling exists because the curve doubles
      and would otherwise pass a day within a working shift. Six hours means a target blocked
      overnight is probed by morning without anyone intervening, which is the property that
      matters: a target nobody ever re-probes stays broken after the block is lifted.

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
        breaker_for: Callable[[str], ProbeObservableBreaker] | None = None,
        blocked_attempts: WindowedCounter | None = None,
        probe_pacer: TokenBucket | None = None,
        reprobe_scheduler: ReprobeScheduler | None = None,
    ) -> None:
        """
        :param health_collection: where the durable circuit state lives
        :ptype health_collection: ScrapeTargetHealthCollection
        :param policy: threshold and backoff curve; defaults are used when omitted
        :ptype policy: BackoffPolicy | None
        :param breaker_for: optional per-target in-process fast-fail, checked before any
            I/O. Takes the target id because one instance serves many targets, so a single
            shared breaker would conflate them; ``CircuitBreakerRegistry.get`` fits directly,
            though a long-lived process wants a bounded lookup (see the module docstring).
            Must expose ``state`` (see :class:`ProbeObservableBreaker`), because a probe this
            module cannot see admitted is a probe it cannot resolve
        :ptype breaker_for: Callable[[str], ProbeObservableBreaker] | None
        :param blocked_attempts: optional cross-pod counter of blocked observations
        :ptype blocked_attempts: WindowedCounter | None
        :param probe_pacer: optional capacity-one bucket admitting one probe per fleet
        :ptype probe_pacer: TokenBucket | None
        :param reprobe_scheduler: optional booking of the next probe as a scheduled job
        :ptype reprobe_scheduler: ReprobeScheduler | None
        """
        self._health = health_collection
        self._policy = policy or BackoffPolicy()
        self._breaker_for = breaker_for
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

        An unreadable health store degrades this target to the behaviour it had before the
        circuit existed -- fetching -- because a store outage must not silently stop scraping
        everything. Nothing this module does raises; an injected ``breaker_for`` or its
        ``check()`` still can, and that propagates rather than being swallowed, because a
        broken injection is the caller's bug and hiding it would fetch at full rate while
        looking healthy.

        :param target_id: the target about to be fetched
        :ptype target_id: str
        :param now: the current time; injected by tests, defaults to now
        :ptype now: datetime | None
        :return: whether to fetch, and what the circuit thought
        :rtype: FetchDecision
        """
        moment = now or datetime.now(UTC)

        breaker = self._breaker(target_id)
        if breaker is not None:
            try:
                breaker.check()
            except CircuitOpenError as exc:
                return FetchDecision(
                    permitted=False,
                    state=CircuitState.OPEN,
                    retry_after_seconds=exc.remaining_seconds,
                    is_probe=False,
                    reason="in-process circuit breaker is open for this target",
                )

        health = (await self._read_health(target_id)).row
        if health is None or _stored_state(health.circuit_state) is CircuitState.CLOSED:
            return FetchDecision(
                permitted=True,
                state=CircuitState.CLOSED,
                retry_after_seconds=0.0,
                is_probe=False,
                reason="circuit closed",
            )

        was = _stored_state(health.circuit_state)

        # A HALF_OPEN row means a probe was admitted and has not reported back. `CircuitBreaker`
        # bounds that with an in-flight flag, but `restore()` cannot carry one across a process
        # boundary, so its HALF_OPEN branch consults no timer and would admit a fresh probe on
        # every poll -- a target whose probe never reports (a caller that raised before recording
        # an outcome) would be fetched at full rate, which is the one hole in "the fetch rate
        # decays". The promotion below therefore stamps `blocked_until` as the probe's
        # reservation, and it is honoured here.
        #
        # Honoured whether or not a pacer is configured, because a `TokenBucket` is not the same
        # answer to the same question. The bucket bounds how many pods probe at once, at its
        # own FIXED refill rate; the reservation bounds how OFTEN a stuck target is probed, on
        # the doubling curve. Deferring to the bucket would swap a decay for a floor, and a
        # constant re-probe cadence is the thing this whole module exists to remove.
        if was is CircuitState.HALF_OPEN:
            outstanding = _seconds_until(health.blocked_until, moment)
            if outstanding > 0.0:
                self._release_in_process_probe(breaker)
                return FetchDecision(
                    permitted=False,
                    state=CircuitState.HALF_OPEN,
                    retry_after_seconds=outstanding,
                    is_probe=False,
                    reason="a probe of this target is already outstanding",
                )

        restored = self._restore(target_id, health, moment)
        try:
            restored.check()
        except CircuitOpenError as exc:
            # The in-process breaker was asked first and said yes, which for an OPEN breaker
            # means it just promoted itself to HALF_OPEN and marked ITS probe in flight. That
            # probe is now never going to happen, and its in-flight flag is only cleared by an
            # outcome being recorded -- so leaving it set strands that breaker fast-failing
            # this target forever, outliving the durable window it was waiting on. Reporting
            # the failure it effectively had clears the flag and restarts its own recovery
            # timer, which is the honest description: the attempt did not reach the target.
            self._release_in_process_probe(breaker)
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
            # Same reasoning as the suppressed branch above: another pod holds the fleet's
            # probe slot, so this process's own probe is not happening and must not be left
            # marked as in flight.
            self._release_in_process_probe(breaker)
            return paced

        # Written on every admission, not only the OPEN-to-HALF_OPEN one, because the value
        # being written is this probe's reservation and a re-probe needs a fresh one. It is
        # the same arithmetic the trip itself uses, so a probe that never reports back buys
        # exactly the silence the failure that preceded it had already bought.
        reservation = self._policy.delay_for(health.consecutive_fetch_failures)
        await self._write(
            target_id,
            state=CircuitState.HALF_OPEN,
            failures=health.consecutive_fetch_failures,
            blocked_until=moment + timedelta(seconds=reservation),
        )
        # Booked here as well as at the trip, because a `relative_delay` job is terminal: the
        # job the trip booked has now fired and will not fire again. Without this, an
        # event-driven caller that dies between this probe and its outcome report leaves a
        # HALF_OPEN row with a live reservation and nothing left in the world that will ever
        # look at it again -- the exact crash the reservation was invented for, solved for a
        # poller (whose next poll is the re-probe) and silently not for the deployment
        # `reprobe.py` exists to serve. The booking is keyed by target, so the outcome report
        # that normally follows replaces it rather than adding to it.
        #
        # A recovery is the one outcome that books nothing, and so the one that cannot
        # supersede this by replacing it. `record_reachable` therefore cancels explicitly
        # through `ReprobeScheduler.cancel_reprobe`; without that, the booking made here
        # fires against a target that already came back, which is a whole poll including its
        # eval loop rather than a bare fetch.

        await self._book_reprobe(target_id, reservation)
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

        One inherited behaviour worth naming, since it looks like a bug and is not: a success
        reported against a row still reading OPEN leaves it open. That is ``CircuitBreaker``'s
        own answer to a success from a request it never admitted, and it is adopted rather
        than overridden, because overriding it would mean writing a transition rule here --
        the one thing this module is built not to do.

        What is NOT adopted is clearing the window in that case. A circuit left open keeps its
        ``blocked_until``, because writing ``(OPEN, N, None)`` would be strictly worse than
        either outcome the transition rule is choosing between: :meth:`_restore` reads a
        missing window as nought seconds remaining, so the very next :meth:`check` finds an
        OPEN breaker whose recovery has elapsed, promotes it, and probes -- the backoff gone
        while the state column still says it is being enforced. The window is this module's
        own storage, not a transition, so preserving it writes no rule. Reachable whenever the
        HALF_OPEN promotion failed to persist, and across a fleet without that: pod B trips
        the row while pod A's already-permitted fetch is still in flight, and pod A then
        reports the success it genuinely had.

        With the window kept, that case self-heals on schedule rather than immediately: the
        next :meth:`check` after ``blocked_until`` elapses promotes and probes, which is the
        same treatment the failure that opened the circuit had already bought.

        :param target_id: the target that served content
        :ptype target_id: str
        :param now: the current time; injected by tests, defaults to now
        :ptype now: datetime | None
        """
        moment = now or datetime.now(UTC)
        breaker = self._breaker(target_id)
        if breaker is not None:
            breaker.record_success()

        health = (await self._read_health(target_id)).row
        if health is None:
            return
        if _stored_state(health.circuit_state) is CircuitState.CLOSED and health.consecutive_fetch_failures == 0:
            return

        restored = self._restore(target_id, health, moment)
        restored.record_success()
        closed = restored.state is CircuitState.CLOSED
        log.info(
            "scrape circuit: target %s is reachable again; circuit %s",
            target_id,
            "closed" if closed else f"left {restored.state.value} with its window intact",
            extra={"extra_data": {"target_id": target_id, "circuit_state": restored.state.value}},
        )
        # The window is cleared only by an actual close. A circuit the breaker chose to leave
        # open keeps the `blocked_until` it already had -- see the note above on why writing
        # `(OPEN, N, None)` would erase the backoff while still reading as enforced.
        await self._write(
            target_id,
            state=restored.state,
            failures=restored.failure_count,
            blocked_until=None if closed else health.blocked_until,
        )
        if closed:
            # The only outcome that books nothing, and therefore the only one that leaves the
            # previous booking standing. Every failure path re-books over the same key, so it
            # supersedes itself; a close has to say so explicitly or the last probe fires
            # against a target that already came back -- a whole poll, and a job row per
            # target that ever tripped.
            await self._cancel_reprobe(target_id)

    async def record_human_cleared(self, target_id: str, *, now: datetime | None = None) -> None:
        """A person cleared this target's wall out of band. Stop suppressing it.

        The step that closes the human-in-the-loop loop, and without it the loop does not
        close: a target is walled, its circuit opens for hours, a human clears it in a session
        and their solve is stored -- and the very next poll is still suppressed, so the work
        they just did sits unused until a timer they know nothing about elapses.

        :meth:`record_reachable` cannot do this job, and that is not an oversight in it. It
        reports a FETCH that succeeded, and ``CircuitBreaker``'s answer to a success from a
        request it never admitted is to leave the circuit open -- correct, because a success
        nobody was permitted to attempt is not evidence the target recovered. A human saying
        "I cleared it" is different evidence: it did not come from a fetch, it came from a
        person who looked at the page.

        So this writes the closed state directly rather than driving the breaker. That is the
        one place in this module that does, and the reason is that the breaker models fetch
        outcomes and this is not one. Everything a trip wrote is cleared together --
        ``consecutive_fetch_failures`` back to zero and ``blocked_until`` removed -- because
        clearing three of four is what leaves a target reading closed while a stale window
        goes on gating it.

        ``last_blocked_at`` is deliberately KEPT. It is the evidence this target was walled,
        an operator looking at a recovered target wants to see it, and it is what
        :meth:`~threetears.scrape.health.ScrapeTargetHealthCollection.list_walled` pairs with
        the circuit state to decide the queue -- so keeping it costs nothing and erasing it
        would lose the history.

        Storing the human's sealed session state is the caller's separate step
        (:func:`threetears.scrape.session_state.record_session_state`). Kept separate because
        a human can clear a wall without producing reusable state -- a session that exported
        nothing still un-sticks the target -- and folding them would make the useful half
        conditional on the optional one.

        :param target_id: the target a human just cleared
        :ptype target_id: str
        :param now: current time; injected by tests, defaults to now
        :ptype now: datetime | None
        :return: nothing
        :rtype: None
        """
        del now
        breaker = self._breaker(target_id)
        if breaker is not None:
            # The in-process breaker is fetch-shaped and may be holding its own opinion from
            # before the human arrived. A success is the honest report here: as far as this
            # process is concerned the target is now reachable.
            breaker.record_success()

        await self._write(
            target_id,
            state=CircuitState.CLOSED,
            failures=0,
            blocked_until=None,
        )
        # Nothing is due a probe any more, and a booking that survives would wake a dispatcher
        # for a target that is already working.
        await self._cancel_reprobe(target_id)
        log.info(
            "scrape circuit: target %s was cleared by a human; suppression lifted",
            target_id,
            extra={"extra_data": {"target_id": target_id, "circuit_state": CircuitState.CLOSED.value}},
        )

    def release_probe(self, target_id: str) -> None:
        """Resolve a permitted fetch that will never report an outcome.

        For a caller whose :meth:`check` said yes and which then raised before reaching
        :meth:`record_blocked`, :meth:`record_unreachable` or :meth:`record_reachable`.
        Without it, a permitted decision that promoted the in-process breaker to HALF_OPEN
        leaves that breaker holding a probe no outcome will ever clear, and from then on
        :meth:`check` fast-fails this target on the in-process branch -- before the durable
        row is even read -- with ``retry_after_seconds`` of 0.0, telling the caller to retry
        immediately, forever.

        The durable circuit needs nothing here and is deliberately not touched: its HALF_OPEN
        promotion already stamped ``blocked_until`` as the probe's own reservation, so a
        caller that dies mid-probe is bounded by a value that outlives the process. Only the
        process-local in-flight flag can strand, so only it is released.

        Safe to call when no probe was admitted, and safe to call twice: the release is
        conditional on the breaker actually reading HALF_OPEN.

        Not async -- it touches no store, which is what lets an exception handler call it
        without introducing a new failure of its own.

        :param target_id: the target whose fetch was permitted but never resolved
        :ptype target_id: str
        """
        self._release_in_process_probe(self._breaker(target_id))

    def _breaker(self, target_id: str) -> ProbeObservableBreaker | None:
        """This target's in-process breaker, or ``None`` when no lookup was injected."""
        return None if self._breaker_for is None else self._breaker_for(target_id)

    def _release_in_process_probe(self, breaker: ProbeObservableBreaker | None) -> None:
        """Resolve a probe the in-process breaker admitted but that never reached the target.

        ``CircuitBreakerLike`` has three calls and no way to say "never mind"; an admitted
        probe is cleared only by an outcome. Reporting a failure is the accurate one -- the
        attempt did not reach the target -- and it restarts that breaker's own recovery timer
        rather than leaving it fast-failing indefinitely.

        Only when a probe was actually admitted, which is the whole point: a CLOSED breaker
        admitted nothing, so reporting a failure to it would be inventing one. That is not a
        theoretical tidiness -- the in-process recovery timeout is seconds where the durable
        window is minutes to hours, so an unconditional release lets a handful of suppressed
        polls trip a breaker that never saw a failed fetch, after which :meth:`check` answers
        from the WRONG circuit and tells the caller to retry in seconds when the truth is
        hours.
        """
        if breaker is not None and _admitted_a_probe(breaker):
            breaker.record_failure()

    async def _record_failure(self, target_id: str, *, now: datetime | None, blocked: bool) -> None:
        """Drive one failure through the breaker's rules and persist what it decided."""
        moment = now or datetime.now(UTC)
        breaker = self._breaker(target_id)
        if breaker is not None:
            breaker.record_failure()

        read = await self._read_health(target_id)
        if not read.readable:
            # Unlike `check`, where an unreadable row degrades safely to "fetch it", writing
            # here would persist a state nobody observed: the restore below would start from
            # CLOSED/0 and stamp a closed circuit with no window over a target that may be
            # open and backed off, wiping the backoff the row already carried. Skipping the
            # write costs this one failure; guessing costs the suppression.
            return
        health = read.row
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
        """Ask the scheduler to wake something up when the backoff window expires.

        A failed booking is logged and swallowed rather than failing the fetch that caused
        it, but what that costs depends entirely on the caller, and it is worth being exact
        about which. A poller loses nothing: its next poll IS the re-probe, and
        ``blocked_until`` is already durable. A caller with no tick -- the only kind
        :mod:`threetears.scrape.reprobe` exists for -- loses the re-probe itself, because
        nothing else will ever revisit the row, and the target stays suppressed past the
        window that was supposed to release it. That is the failure
        :attr:`BackoffPolicy.max_delay_seconds` is written to prevent ("a target nobody ever
        re-probes is a target that stays broken after the block is lifted"), so for that
        deployment this is a real gap and not merely a lost convenience.

        It stays swallowed because the alternative is worse -- turning a scheduler outage
        into a failed scrape of a target we already have an answer for -- and ``log.exception``
        makes it detectable. An event-driven deployment that cannot tolerate the gap needs a
        reconciliation sweep over rows whose ``blocked_until`` has passed while
        ``circuit_state`` is still open; that is deliberately not built here, because it is a
        scheduled job about the whole table rather than a decision about one fetch.
        """
        if self._reprobe_scheduler is None:
            return
        try:
            await self._reprobe_scheduler.schedule_reprobe(target_id=target_id, delay_seconds=delay_seconds)
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- a scheduler outage must not fail a fetch whose result we already hold; blocked_until still stands, so a poller loses nothing and an event-driven caller loses this wake-up, which the docstring states and log.exception makes detectable
            log.exception(
                "scrape circuit: could not book the re-probe for target %s; its backoff window still stands",
                target_id,
                extra={"extra_data": {"target_id": target_id}},
            )

    async def _cancel_reprobe(self, target_id: str) -> None:
        """Drop an outstanding booking, never letting the cleanup cost the caller its result."""
        if self._reprobe_scheduler is None:
            return
        try:
            await self._reprobe_scheduler.cancel_reprobe(target_id=target_id)
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- a failed cancel costs one spurious wake-up against a target that has recovered, which is a wasted poll and not a wrong answer; failing the fetch that just succeeded would be worse. Logged with its traceback below
            log.exception(
                "scrape circuit: could not cancel the outstanding re-probe of target %s; it may fire once more",
                target_id,
                extra={"extra_data": {"target_id": target_id}},
            )

    async def forget_target(self, target_id: str) -> None:
        """Discard everything this circuit durably holds about *target_id*.

        For a caller retiring a target, which is the only party that can know a target is
        retired: this module sees polls, not the list of things worth polling.

        Both tables it writes are keyed by target and upserted, never appended, so neither
        grows with time or poll count -- one health row per target observed, one job row per
        target that has tripped, and a re-booking replaces rather than accumulates. What they
        do grow with is DISTINCT targets, and `ScrapeTool._derive_target_id` mints a fresh
        ``adhoc_<sha256>`` per ``(url, field_schema)``, so a long-lived process scraping
        ad-hoc URLs accumulates a row per URL it has ever seen. That is a real bound, just not
        a small one, and until now there was no way to reclaim any of it.

        Deliberately not automatic. A health row is not garbage: it carries the fingerprint
        that stops a target being re-classified on every poll, so evicting one for a target
        still being scraped costs exactly the LLM calls this whole design exists to avoid. No
        TTL can tell those apart, because "still wanted" is the caller's fact, not this
        module's.

        :param target_id: the target being retired
        :ptype target_id: str
        """
        await self._cancel_reprobe(target_id)
        try:
            await self._health.delete(target_id)
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- retiring a target is housekeeping; a store outage must not raise into a caller that is tidying up, and the row is harmless until the next attempt. Logged with its traceback below
            log.exception(
                "scrape circuit: could not delete the health row for target %s",
                target_id,
                extra={"extra_data": {"target_id": target_id}},
            )

    async def _read_health(self, target_id: str) -> _HealthRead:
        """Read the durable row, reporting an unreadable store as distinct from an absent row.

        The two are the same to :meth:`check`, which fetches on either, but they are opposite
        to a writer: "no row" is a fact worth persisting against, and "could not read" is the
        absence of any fact at all.
        """
        try:
            return _HealthRead(row=await self._health.get(target_id), readable=True)
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- an unreadable health store must degrade this target to its pre-circuit behaviour, not fail every poll. Logged with its traceback below
            log.exception(
                "scrape circuit: could not read health for target %s; treating its circuit as closed",
                target_id,
                extra={"extra_data": {"target_id": target_id}},
            )
            return _HealthRead(row=None, readable=False)

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


class _HealthRead(NamedTuple):
    """A health row, and whether the store could be read at all.

    ``row=None, readable=True`` is "this target has no health row yet"; ``readable=False`` is
    "the store did not answer". Collapsing the two is what lets a failed read be written back
    as an observation.
    """

    row: ScrapeTargetHealth | None
    readable: bool


def _stored_state(value: str) -> CircuitState:
    """Read a stored ``circuit_state`` string, defaulting an unrecognised one to CLOSED."""
    return _KNOWN_CIRCUIT_STATES.get(value, CircuitState.CLOSED)


def _seconds_until(moment: datetime | None, now: datetime) -> float:
    """Seconds from *now* until *moment*, or 0.0 when it is absent or already past."""
    if moment is None:
        return 0.0
    return max(0.0, (moment - now).total_seconds())


def _admitted_a_probe(breaker: ProbeObservableBreaker) -> bool:
    """Whether *breaker*'s just-returned ``check()`` admitted a recovery probe.

    A breaker only admits a probe from HALF_OPEN, and it only reaches HALF_OPEN by admitting
    one -- an OPEN breaker whose window elapsed promotes itself and takes the slot, and a
    HALF_OPEN one whose slot is already taken raises instead of returning. So a breaker
    reading HALF_OPEN after a ``check()`` that returned is holding this call's probe, and a
    CLOSED one is holding nothing.

    ``state`` is required by :class:`ProbeObservableBreaker` rather than probed for, so a
    breaker that cannot answer this question is rejected at the seam instead of silently
    never being released.
    """
    return breaker.state is CircuitState.HALF_OPEN
