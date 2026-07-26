"""Honour a site's ``robots.txt``: wait as long as it asks, and escalate when it says no.

Two behaviours, two flags, **both on by default**. A scraper whose politeness is opt-in is a
scraper that is impolite in every deployment nobody configured, and the deployments that
forget are exactly the ones nobody is watching.

The two halves are genuinely different things and conflating them is the trap:

- ``Crawl-delay`` is a request to be **slower**. Honouring it changes scheduling and nothing
  else -- the target still gets fetched, it just gets fetched less often.
- ``Disallow`` is a request **not to fetch at all**. Simply obeying it would make a target
  permanently invisible with no way to say "we have an agreement with this site", and simply
  ignoring it is what gives crawlers their reputation. So it does neither: it **escalates**,
  producing an outcome a human decides on, through the same path a bot wall already takes.

**A human working the page over VNC is not a bot.** The Robots Exclusion Protocol governs
automated agents, not people operating browsers, so a ``Disallow`` that stops the unattended
fetcher does not stop an operator who opens a session and works the target themselves. That
is the position, and it is what makes the escalation close rather than dead-end. Two things
keep it a position rather than a loophole: the exemption is for a session a person is
actually in (not "open a session and let the robot drive through it"), and ``Crawl-delay``
does NOT get the exemption -- load on someone's server is caused equally by either.

**Where this sits relative to the fetch circuit.** Both gate the fetch and they are different
kinds of gate. ``Crawl-delay`` is a FLOOR on politeness that applies to a target working
perfectly; the circuit's ``blocked_until`` is a CEILING on cost that applies to one that is
not. A fetch satisfies both, and neither may be used to weaken the other -- in particular a
circuit probe is not exempt from the crawl delay, or the politeness contract breaks precisely
when a target is already unhappy with us.

Parsing is :mod:`urllib.robotparser` from the standard library rather than a dependency. The
grammar is older and looser than its reputation and implementations genuinely disagree about
wildcards and ``Allow`` precedence; the stdlib's reading is a defensible one, it is already
installed everywhere, and a scraping library adding a package to read a text file it fetches
once per origin is a poor trade.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.core.coordination.token_bucket import TokenBucket
    from threetears.core.egress import EgressDriver

__all__ = [
    "DEFAULT_USER_AGENT",
    "RobotsDecision",
    "RobotsGate",
    "RobotsPolicy",
]

log = get_logger(__name__)

#: What we call ourselves when matching rules. A real token rather than a browser string,
#: because a site that wants to write a rule for us must have something to write it about, and
#: pretending to be a browser in ``robots.txt`` matching while honouring the file is
#: incoherent.
DEFAULT_USER_AGENT = "3tears-scrape"

#: How long a fetched ``robots.txt`` is trusted before re-reading. Per origin, not per target:
#: several targets commonly share one, and re-fetching per target would make politeness cost
#: more requests than it saves.
_DEFAULT_CACHE_SECONDS = 3600.0

#: Budget for fetching ``robots.txt`` itself. Short on purpose -- a site that will not serve
#: its own robots file within this is a site we are about to fetch anyway, and blocking a
#: scrape on it would let an unreachable text file stop real work.
_FETCH_TIMEOUT_SECONDS = 10.0

#: How many origins' parsed files and fetch clocks to retain. A cap rather than unbounded
#: growth: everything held per origin is reconstructible by re-reading a text file, so the
#: cost of evicting the least-recently-used entry is one re-fetch, while the cost of never
#: evicting is a leak proportional to how many distinct sites a process has ever seen.
_DEFAULT_MAX_ORIGINS = 2048


@dataclass(frozen=True)
class RobotsPolicy:
    """What to honour, and as whom. Both behaviours default ON.

    :param respect_crawl_delay: wait at least ``Crawl-delay`` between fetches of one origin
    :ptype respect_crawl_delay: bool
    :param flag_disallowed: escalate a disallowed target for a human rather than fetching it
    :ptype flag_disallowed: bool
    :param user_agent: the token rules are matched against
    :ptype user_agent: str
    :param overrides: origins to skip entirely, for a deployment with a written agreement with
        a site. An explicit, per-origin, recorded exception -- not a global off switch, because
        "we have permission for this one" and "ignore robots everywhere" should not be the
        same setting
    :ptype overrides: frozenset[str]
    :param max_crawl_delay_seconds: ceiling on an honoured delay. A file asking for 86400
        is asking us not to crawl at all, and a scheduler that silently sleeps for a day looks
        identical to one that has hung
    :ptype max_crawl_delay_seconds: float
    """

    respect_crawl_delay: bool = True
    flag_disallowed: bool = True
    user_agent: str = DEFAULT_USER_AGENT
    overrides: frozenset[str] = field(default_factory=frozenset)
    max_crawl_delay_seconds: float = 300.0


@dataclass(frozen=True)
class RobotsDecision:
    """Whether to fetch this url now, and what the file said.

    Carries the reason rather than a bare boolean for the same reason
    :class:`~threetears.scrape.circuit.FetchDecision` does: a caller that declines has to tell
    its own caller something more useful than "no".
    """

    #: Fetch may proceed (possibly after :attr:`wait_seconds`).
    allowed: bool
    #: How long to wait first, honouring ``Crawl-delay``. Zero when nothing is owed.
    wait_seconds: float = 0.0
    #: True when a ``Disallow`` matched and a human should decide.
    needs_human: bool = False
    #: Human-readable explanation.
    reason: str = "no robots.txt restrictions apply"


class RobotsGate:
    """Reads a site's ``robots.txt`` and answers "may I fetch this, and how soon".

    One instance serves many origins; the parsed file is cached per origin with a TTL, because
    several targets commonly share an origin and re-reading per target would spend more
    requests on politeness than politeness saves.

    :param policy: what to honour and as whom
    :ptype policy: RobotsPolicy
    :param egress: exit the DEFAULT fetcher leaves by, so the robots request shares the
        scrape's route. Ignored when ``fetch`` is supplied, since an injected fetcher already
        carries whatever transport its owner chose
    :ptype egress: EgressDriver | None
    :param fetch: how to GET a url, returning ``(status, text)``. Defaults to an httpx
        GET, so a gate built with no arguments genuinely honours a site's file. Inject one
        built on :class:`~threetears.core.http_client.TracedHttpClient` to give the robots
        request the same egress driver, tracing, retry and circuit breaking as the real fetch
        -- worth doing, because a robots request that leaves by a different exit than the
        scrape is asking a different question than the one being answered
    :ptype fetch: Callable[[str], Awaitable[tuple[int, str]]] | None
    :param delay_pacer: optional cross-pod ``TokenBucket``. Without it the crawl delay is
        honoured PER PROCESS, which is a lie in a fleet: five pods each waiting ten seconds
        present a request every two. The same reuse and the same reasoning as the circuit's
        probe pacer
    :ptype delay_pacer: TokenBucket | None
    :param cache_seconds: how long a parsed file is trusted
    :ptype cache_seconds: float
    :param max_origins: how many origins to retain before evicting the least recently used.
        Both per-origin stores are reconstructible from a re-fetch, so this is a cap rather
        than a leak; see :meth:`forget` for retiring one origin explicitly
    :ptype max_origins: int
    """

    def __init__(
        self,
        policy: RobotsPolicy | None = None,
        *,
        fetch: Any = None,
        egress: EgressDriver | None = None,
        delay_pacer: TokenBucket | None = None,
        cache_seconds: float = _DEFAULT_CACHE_SECONDS,
        max_origins: int = _DEFAULT_MAX_ORIGINS,
    ) -> None:
        self._policy = policy or RobotsPolicy()
        # A default fetcher, so a gate constructed with no arguments actually reads robots.txt.
        # Without one, "both behaviours on by default" was true of a policy object and false of
        # every deployment: no fetcher means no file, no file means nothing is ever honoured,
        # and nothing anywhere would have said so.
        # The default fetcher leaves by the SAME exit as the scrape it precedes. Without that
        # it is a bare client on the container's own route, and a deployment that configured
        # TOR would disclose its real address to every target origin moments before the
        # proxied fetch -- worse than no proxying, because it believes it has the property.
        # The robots request also has to ask the same question the scrape asks: a file fetched
        # from a different address can be a different file.
        self._egress = egress
        self._fetch = fetch if fetch is not None else _default_fetch_via(egress)
        self._delay_pacer = delay_pacer
        self._cache_seconds = cache_seconds
        self._max_origins = max_origins
        # Both are keyed by origin and neither is durable: everything here can be rebuilt by
        # re-reading a text file. That is what makes self-bounding the right answer, where the
        # circuit's equivalent state gets a manual `forget_target` instead -- evicting a
        # circuit row would discard a judgement nothing can reconstruct, while evicting one of
        # these costs exactly one re-fetch. A long-lived process scraping a wide set of sites
        # would otherwise hold one entry per origin it had ever touched, forever.
        self._cache: OrderedDict[str, tuple[RobotFileParser | None, float]] = OrderedDict()
        self._last_fetch_at: OrderedDict[str, float] = OrderedDict()
        # Bumped by `forget`. `_parser_for` deliberately does not hold the lock across its
        # fetch -- one slow origin must not stall every other -- which leaves a window where a
        # forget lands mid-load and the completing write would resurrect what was just
        # discarded, silently returning the pre-forget file to a caller who forgot precisely
        # because it changed. A counter rather than a per-origin epoch so this cannot become a
        # third store that grows per origin forever, which is the hazard being fixed two lines
        # up; the cost is that any forget discards any in-flight load, worth one re-fetch.
        self._generation = 0
        self._lock = asyncio.Lock()

    @property
    def max_wait_seconds(self) -> float:
        """The longest this gate can ask a caller to sleep before a fetch.

        Public because a caller that advertises a deadline has to include it: the wait happens
        BEFORE the render, so a budget derived from the render alone is smaller than the call
        it describes, and the executor cancels mid-sleep. Reads the live policy rather than the
        module default, so raising the ceiling raises the declared budget with it.
        """
        return self._policy.max_crawl_delay_seconds if self._policy.respect_crawl_delay else 0.0

    async def check(self, url: str, *, now: float | None = None) -> RobotsDecision:
        """Decide whether *url* may be fetched, and how long to wait first.

        Never raises. Every way of failing to read a site's wishes ends in "allowed", because
        a broken or missing ``robots.txt`` is not a refusal -- treating an unreachable text
        file as a wall would let one 500 stop a scrape the site never objected to.

        :param url: the url about to be fetched
        :ptype url: str
        :param now: current time; injected by tests
        :ptype now: float | None
        :return: whether to fetch, and what the file said
        :rtype: RobotsDecision
        """
        moment = now if now is not None else time.monotonic()
        origin = _origin_of(url)
        if not origin:
            return RobotsDecision(allowed=True, reason="not an absolute http(s) url; nothing to consult")
        if origin in self._policy.overrides:
            return RobotsDecision(allowed=True, reason=f"{origin} is an explicit override in this deployment")

        parser = await self._parser_for(origin, moment)
        if parser is None:
            return RobotsDecision(allowed=True, reason="no usable robots.txt; nothing was asked of us")

        if self._policy.flag_disallowed and not parser.can_fetch(self._policy.user_agent, url):
            return RobotsDecision(
                allowed=False,
                needs_human=True,
                reason=(
                    f"{origin}/robots.txt disallows {self._policy.user_agent} from this path. "
                    "Not fetched unattended; a person may still work it, since the exclusion "
                    "protocol governs automated agents rather than people using browsers."
                ),
            )

        wait = await self._delay_owed(origin, parser, moment)
        if wait > 0:
            return RobotsDecision(
                allowed=True,
                wait_seconds=wait,
                reason=f"{origin}/robots.txt asks for {wait:.0f}s between requests",
            )
        return RobotsDecision(allowed=True)

    def note_fetched(self, url: str, *, now: float | None = None) -> None:
        """Record that *url*'s origin was just fetched, starting its crawl-delay clock.

        Separate from :meth:`check` because the delay is measured between FETCHES, and a check
        that did not lead to one -- the circuit suppressed it, the caller changed its mind --
        must not start the clock. Folding them would make a rejected check consume the site's
        patience.
        """
        origin = _origin_of(url)
        if origin:
            self._last_fetch_at[origin] = now if now is not None else time.monotonic()
            self._last_fetch_at.move_to_end(origin)
            self._evict_if_needed(self._last_fetch_at)

    async def _delay_owed(self, origin: str, parser: RobotFileParser, now: float) -> float:
        """Seconds still owed to *origin* before another fetch is polite."""
        delay = self._capped_delay(origin, parser)
        if delay is None:
            return 0.0

        # The FLEET's turn is deliberately not asked for here. `TokenBucket.claim` consumes a
        # token atomically, and `check` is a question, not a commitment: the circuit can
        # suppress the fetch afterwards, the caller can change its mind, the driver can be
        # missing. Charging the site's fleet-wide budget for a fetch that never happens is the
        # same defect `note_fetched` exists to prevent for the local clock -- and it is worse
        # here, because the token is shared, so polling one walled target inside its backoff
        # delayed every sibling target on that origin. See :meth:`claim_fleet_turn`.
        return self._local_delay_owed(origin, delay, now)

    def forget(self, url_or_origin: str) -> None:
        """Drop everything cached for one origin: its parsed file and its fetch clock.

        For a caller retiring a site, or one that knows the file just changed. Dropping the
        clock is deliberate and is the reason this is not simply a cache expiry -- a caller
        that forgets an origin it is still scraping gets a fresh crawl-delay window rather
        than a preserved one, which is the permissive direction, so this is for retirement
        rather than for reset.

        :param url_or_origin: a full url or a bare ``scheme://host`` origin
        :ptype url_or_origin: str
        """
        origin = _origin_of(url_or_origin) or url_or_origin
        self._cache.pop(origin, None)
        self._last_fetch_at.pop(origin, None)
        self._generation += 1

    def _evict_if_needed(self, store: OrderedDict[str, Any]) -> None:
        """Hold *store* to ``max_origins`` entries, dropping least-recently-touched first."""
        while len(store) > self._max_origins:
            store.popitem(last=False)

    def _capped_delay(self, origin: str, parser: RobotFileParser, *, announce: bool = True) -> float | None:
        """The ``Crawl-delay`` this origin is governed by after capping, or ``None`` if none.

        ``None`` means "this origin is not delay-governed at all" -- the policy has the
        behaviour off, the file declares no delay, or the value will not parse. Shared by the
        local clock and :meth:`claim_fleet_turn` precisely so the two cannot disagree about
        which origins are paced: they did, briefly, and every site that declares no delay was
        being throttled fleet-wide by a pacer the local clock would never have consulted.

        *announce* exists because this is now called twice per fetch -- once deciding the local
        wait and once deciding whether the origin is paced at all -- and both branches log. The
        second caller passes ``False`` so a capped or malformed ``Crawl-delay`` is reported once
        per poll rather than twice, which is the difference between a note and a stutter.
        """
        if not self._policy.respect_crawl_delay:
            return None
        raw = parser.crawl_delay(self._policy.user_agent)
        if raw is None:
            return None
        try:
            requested = float(raw)
        except TypeError, ValueError:
            # A malformed Crawl-delay is not a refusal and not a licence. Ignoring the value
            # while still honouring the rest of the file is the reading that respects what the
            # site could actually express.
            #
            # Unreachable through a real robots.txt: `urllib.robotparser` validates the value
            # itself and returns None for anything non-integer, including `1e3`. This guards
            # the injected-parser case -- `_capped_delay` takes whatever object `_parser_for`
            # produced -- and a future stdlib that returns the raw token. Kept rather than
            # deleted because the cost is three lines and the failure it prevents is a
            # ValueError escaping a method documented never to raise.
            if announce:
                log.info("scrape robots: %s asked for an unparseable crawl delay %r; ignoring it", origin, raw)
            return None
        delay = min(requested, self._policy.max_crawl_delay_seconds)
        if delay < requested and announce:
            log.info(
                "scrape robots: %s asked for %.0fs between requests; capped at %.0fs",
                origin,
                requested,
                delay,
            )
        return delay

    async def claim_fleet_turn(self, url: str) -> float:
        """Take this origin's turn from the cross-pod pacer, and say how long is still owed.

        Split from :meth:`check` because claiming CONSUMES. Call it once the fetch is committed
        -- after every other gate has admitted it -- and honour the returned wait alongside
        :attr:`RobotsDecision.wait_seconds`; the two are different constraints and the longer
        one binds. Without a pacer configured this is a no-op returning ``0.0``, so a caller
        need not branch on whether fleet coordination exists.

        Never raises. A KV outage costs fleet-wide precision and falls back to this pod's own
        clock, which is stricter per pod rather than looser -- the same posture as every other
        optional coordination primitive in this package.

        :param url: the url about to be fetched
        :ptype url: str
        :return: additional seconds to wait before fetching, ``0.0`` when the turn is granted
        :rtype: float
        """
        if self._delay_pacer is None:
            return 0.0
        origin = _origin_of(url)
        if not origin or origin in self._policy.overrides:
            return 0.0

        # The SAME preconditions the local clock applies. Moving the claim out of `check`
        # accidentally dropped them: an origin with a written agreement, one serving no
        # robots.txt, or one declaring no `Crawl-delay` -- which is most sites -- was suddenly
        # paced fleet-wide by a bucket the local clock would never have consulted. A gate that
        # throttles sites which asked for nothing is not politeness.
        # The parser is the cached one `check` already fetched moments earlier on this path, so
        # this is a cache read rather than a second network round-trip.
        parser = await self._parser_for(origin, time.monotonic())
        if parser is None or self._capped_delay(origin, parser, announce=False) is None:
            return 0.0

        try:
            claim = await self._delay_pacer.claim(origin)
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- a KV outage must not stop a scrape; it costs fleet-wide precision and falls back to the per-process clock, which is stricter per pod rather than looser. Logged with its traceback below
            log.exception("scrape robots: crawl-delay pacer unavailable for %s; using this pod's own clock", origin)
            return 0.0
        if claim.claimed:
            return 0.0
        # Capped by the same ceiling as a declared delay. `retry_after_seconds` is the bucket's
        # own number and is otherwise unbounded, which would let it exceed `max_wait_seconds` --
        # the value that sizes `ScrapeTool`'s advertised deadline, so an uncapped wait puts the
        # call back outside the budget it was just taught to declare.
        return min(float(claim.retry_after_seconds), self._policy.max_crawl_delay_seconds)

    def _local_delay_owed(self, origin: str, delay: float, now: float) -> float:
        """Seconds owed by this process's own clock alone, ignoring any fleet coordination."""
        last = self._last_fetch_at.get(origin)
        if last is None:
            return 0.0
        return max(0.0, delay - (now - last))

    async def _parser_for(self, origin: str, now: float) -> RobotFileParser | None:
        """The cached parser for *origin*, fetching and parsing when stale."""
        async with self._lock:
            cached = self._cache.get(origin)
            if cached is not None and now < cached[1]:
                # A read counts as use, or a steadily-scraped origin would be evicted by a
                # burst of one-off ones purely because it was fetched longer ago.
                self._cache.move_to_end(origin)
                return cached[0]
            seen = self._generation

        parser = await self._load(origin)
        async with self._lock:
            if self._generation != seen:
                # Something was forgotten while this was in flight. Hand the caller what was
                # actually fetched, but do not cache it: the write would undo the forget.
                return parser
            self._cache[origin] = (parser, now + self._cache_seconds)
            self._cache.move_to_end(origin)
            self._evict_if_needed(self._cache)
        return parser

    async def _load(self, origin: str) -> RobotFileParser | None:
        """Fetch and parse ``{origin}/robots.txt``, or ``None`` when there is nothing usable.

        ``None`` covers every unusable outcome deliberately: no fetcher injected, a transport
        failure, a 404, a 500, or a body that will not parse. All of them mean the same thing
        -- the site has not told us anything -- and distinguishing them here would produce
        four ways of saying "allowed".
        """
        url = f"{origin}/robots.txt"
        try:
            status, body = await asyncio.wait_for(self._fetch(url), timeout=_FETCH_TIMEOUT_SECONDS)
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- an unreachable robots.txt is not a refusal; letting it raise would turn one 500 on a text file into a stopped scrape the site never objected to. Logged with its traceback below
            log.info("scrape robots: could not read %s; proceeding as unrestricted", url, exc_info=True)
            return None
        if status != 200 or not body:
            log.debug("scrape robots: %s returned %s; proceeding as unrestricted", url, status)
            return None
        parser = RobotFileParser()
        try:
            parser.parse(body.splitlines())
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- the grammar is loose and implementations disagree; an unparseable file is a site that failed to express a restriction, not one that expressed a total ban. Logged with its traceback below
            log.info("scrape robots: %s did not parse; proceeding as unrestricted", url, exc_info=True)
            return None
        return parser


def _default_fetch_via(egress: EgressDriver | None) -> Any:
    """Build the default fetcher, bound to *egress*.

    Deliberately present: this module's whole claim is that politeness is on by default, and a
    default of "no fetcher" made that claim false everywhere while looking correct in the
    configuration. Deliberately egress-bound: a default that ignored the configured exit would
    disclose the container's real address to every origin, on by default, immediately before
    the proxied fetch that was supposed to hide it.

    Errors propagate to :meth:`RobotsGate._load`, which treats every one of them as "the site
    told us nothing".
    """

    async def _fetch(url: str) -> tuple[int, str]:
        import httpx  # noqa: PLC0415 -- deliberate late import; module stays importable without a client

        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT_SECONDS,
            follow_redirects=True,
            transport=egress.httpx_transport() if egress is not None else None,
        ) as client:
            response = await client.get(url, headers={"user-agent": DEFAULT_USER_AGENT})
            return response.status_code, response.text

    return _fetch


def _origin_of(url: str) -> str | None:
    """``scheme://host[:port]`` for *url*, or ``None`` when it has no usable origin.

    ``robots.txt`` is per origin, and scheme and port are part of one: ``https://x`` and
    ``http://x`` are allowed to publish different files, and merging them would apply one
    site's rules to another.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))
