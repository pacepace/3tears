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
    """

    def __init__(
        self,
        policy: RobotsPolicy | None = None,
        *,
        fetch: Any = None,
        egress: EgressDriver | None = None,
        delay_pacer: TokenBucket | None = None,
        cache_seconds: float = _DEFAULT_CACHE_SECONDS,
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
        self._cache: dict[str, tuple[RobotFileParser | None, float]] = {}
        self._last_fetch_at: dict[str, float] = {}
        self._lock = asyncio.Lock()

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

    async def _delay_owed(self, origin: str, parser: RobotFileParser, now: float) -> float:
        """Seconds still owed to *origin* before another fetch is polite."""
        if not self._policy.respect_crawl_delay:
            return 0.0
        raw = parser.crawl_delay(self._policy.user_agent)
        if raw is None:
            return 0.0
        try:
            requested = float(raw)
        except TypeError, ValueError:
            # A malformed Crawl-delay is not a refusal and not a licence. Ignoring the value
            # while still honouring the rest of the file is the reading that respects what the
            # site could actually express.
            log.info("scrape robots: %s asked for an unparseable crawl delay %r; ignoring it", origin, raw)
            return 0.0
        delay = min(requested, self._policy.max_crawl_delay_seconds)
        if delay < requested:
            log.info(
                "scrape robots: %s asked for %.0fs between requests; capped at %.0fs",
                origin,
                requested,
                delay,
            )

        if self._delay_pacer is not None:
            # Fleet-wide: without this each pod honours the delay alone and the site sees the
            # sum. Degrades to the per-process clock below rather than blocking, on the same
            # posture as every other optional coordination primitive in this package.
            try:
                claim = await self._delay_pacer.claim(origin)
            except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- a KV outage must not stop a scrape; it costs fleet-wide precision and falls back to the per-process clock, which is stricter per pod rather than looser. Logged with its traceback below
                log.exception("scrape robots: crawl-delay pacer unavailable for %s; using this pod's own clock", origin)
            else:
                return 0.0 if claim.claimed else float(claim.retry_after_seconds)

        last = self._last_fetch_at.get(origin)
        if last is None:
            return 0.0
        return max(0.0, delay - (now - last))

    async def _parser_for(self, origin: str, now: float) -> RobotFileParser | None:
        """The cached parser for *origin*, fetching and parsing when stale."""
        async with self._lock:
            cached = self._cache.get(origin)
            if cached is not None and now < cached[1]:
                return cached[0]

        parser = await self._load(origin)
        async with self._lock:
            self._cache[origin] = (parser, now + self._cache_seconds)
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
            log.info("scrape robots: could not read %s; proceeding as unrestricted", url)
            return None
        if status != 200 or not body:
            log.debug("scrape robots: %s returned %s; proceeding as unrestricted", url, status)
            return None
        parser = RobotFileParser()
        try:
            parser.parse(body.splitlines())
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- the grammar is loose and implementations disagree; an unparseable file is a site that failed to express a restriction, not one that expressed a total ban. Logged with its traceback below
            log.info("scrape robots: %s did not parse; proceeding as unrestricted", url)
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
