"""Which exit an outbound request leaves by, as a driver rather than a setting.

Every app on this framework eventually wants a request to leave by something other than the
container's default route: a TOR circuit for non-attribution, a VPN for a target that blocks
datacentre ranges, a residential pool, a per-tenant egress IP. Written as a flag, each of
those becomes a branch in whatever happens to make the call. Written as a driver, adding the
fourth is one class and no change to any caller -- which is the whole reason this is a seam
and not a config option.

**Two consumers, one driver, because an exit is not an HTTP concept.** An egress driver has
to be able to answer both "what transport should httpx use" and "what does a browser need on
its command line", or a deployment ends up with its API calls going one way and its scrapes
going another while both report the same configured exit. Those are the two halves every
backend has to supply, and a driver that can only do one of them is not an exit.

**A driver describes an exit; it does not run the daemon that provides it.** Starting `tor` or
`warp-cli` is deployment work -- a container, a sidecar, a host service -- and a library that
tried to own process lifecycle for someone else's network would be wrong about it in every
deployment that already had one.

The one thing that does reach the network is :meth:`EgressDriver.health`, and only when a
caller asks. It fetches an address-reporting endpoint THROUGH the exit rather than merely
opening a socket to the proxy, because a proxy that is listening but forwarding directly
answers a connectivity check perfectly while providing none of the property it was configured
for. Describing an exit and being able to say whether it currently works are not the same
claim, and a deployment needs both.

**Direct is a driver, not a special case.** The no-proxy path going through the same seam is
what stops "direct" quietly meaning "the seam was bypassed", and it means a caller never
branches on whether egress is configured at all.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .config import DEFAULT_EGRESS_HEALTH_TIMEOUT_SECONDS

if TYPE_CHECKING:  # pragma: no cover - import-time only
    import httpx

__all__ = [
    "DirectEgress",
    "EgressDriver",
    "EgressHealth",
    "EgressRegistry",
    "ProxyEgress",
    "SocksEgress",
    "WarpEgress",
]


#: What a probe fetches to learn the address an exit presents. Plain-text and tiny by design.
#: Overridable per deployment is deliberately NOT offered here: a driver that probed an
#: attacker-chosen URL through a customer's proxy would be a more interesting bug than the one
#: this method fixes.
_HEALTH_PROBE_URL = "https://api.ipify.org"


@dataclass(frozen=True)
class EgressHealth:
    """Whether an exit is usable, and what it looked like from outside.

    Carries the observed address because that is the only evidence that traffic actually LEFT
    by this exit rather than merely reaching a proxy that forwarded it directly -- a
    misconfigured chain answers "up" to anything that only asks whether it is reachable.
    """

    #: Whether the exit answered at all.
    reachable: bool
    #: The public address the exit presents, when a probe could determine one.
    observed_address: str | None = None
    #: Why it is unusable, when it is.
    reason: str | None = None


@runtime_checkable
class EgressDriver(Protocol):
    """One way out of the machine.

    Structural rather than a base class, matching :class:`CircuitBreakerLike` in
    :mod:`threetears.core.http_client`: a deployment that already has its own egress object
    should be able to satisfy this without inheriting from anything here.
    """

    @property
    def name(self) -> str:
        """Stable identifier, e.g. ``"direct"``, ``"tor"``, ``"warp"``.

        Recorded alongside a result so "this target is blocked" can be told apart from "this
        target is blocked FROM THIS EXIT" -- without which one blocked exit poisons a target
        permanently and whatever is backing it off learns the wrong lesson.
        """
        ...

    def httpx_transport(self) -> httpx.AsyncBaseTransport | None:
        """The transport an ``httpx`` client should bind, or ``None`` for the default route.

        A transport rather than a proxy URL because that is the seam ``TracedHttpClient``
        already exposes, and because it is the only form that can express an exit httpx has
        no URL scheme for.
        """
        ...

    def browser_proxy_arg(self) -> str | None:
        """What a Chromium-family browser needs, or ``None`` for the default route.

        Returned as the value for ``--proxy-server`` rather than a full argument, so a caller
        that builds its own argument list is not made to string-match this one.
        """
        ...

    async def health(self, *, timeout: float = ...) -> EgressHealth:
        """Is this exit actually usable right now?

        On the protocol because its absence is a DETECTION gap, not a missing convenience. A
        dead ``tor`` or ``warp`` daemon fails every render transport-side; each target's
        circuit then opens and backs off for hours, and those targets are correctly excluded
        from the walled queue -- unreachability never stamps ``last_blocked_at`` -- so the one
        operator-facing list stays EMPTY while the entire fleet decays. Every individual
        signal is behaving correctly and the aggregate is invisible.

        This gives an operator one thing to ask so "every target broke at once" can be told
        apart from "the exit is down", which are the same observation until something
        distinguishes them.
        """
        ...


class DirectEgress:
    """The default route, as a driver.

    Exists so that "no proxy" is a configuration choice rather than the absence of one. A
    caller that special-cased ``None`` would grow a second code path that nothing exercises,
    and that path is where "the seam was bypassed" hides.
    """

    @property
    def name(self) -> str:
        """``"direct"``."""
        return "direct"

    def httpx_transport(self) -> httpx.AsyncBaseTransport | None:
        """``None`` -- httpx's own default."""
        return None

    def browser_proxy_arg(self) -> str | None:
        """``None`` -- no ``--proxy-server``."""
        return None

    async def health(self, *, timeout: float = DEFAULT_EGRESS_HEALTH_TIMEOUT_SECONDS) -> EgressHealth:
        """Always reachable, by definition.

        The default route is whatever the machine already has; if it is down, nothing in this
        process is running to ask. Reporting a probe result here would test the internet
        rather than the exit, and turn an unrelated outage into "your egress is broken".
        """
        return EgressHealth(reachable=True, reason="the default route is not probed")


class ProxyEgress:
    """Any exit reachable as a proxy URL: ``http://``, ``https://``, ``socks5://``.

    One class for all of them because the difference between them is the scheme in a string,
    and three classes that differ by a literal would be three places to fix a bug in one.
    TOR and WARP are both constructed from this rather than subclassing it -- see
    :func:`SocksEgress` and the module docstring's point that a driver describes an exit
    rather than running it.
    """

    def __init__(self, name: str, proxy_url: str) -> None:
        """
        :param name: stable identifier recorded against results
        :ptype name: str
        :param proxy_url: full proxy URL including scheme
        :ptype proxy_url: str
        :raises ValueError: when either argument is empty, because a nameless or
            addressless exit silently becomes the default route -- the failure mode this
            whole module exists to make impossible
        """
        if not name:
            raise ValueError("an egress driver must have a name; results are recorded against it")
        if not proxy_url:
            raise ValueError(f"egress {name!r} has no proxy url; an exit with no address is the default route")
        self._name = name
        self._proxy_url = proxy_url

    @property
    def name(self) -> str:
        """The configured identifier."""
        return self._name

    @property
    def proxy_url(self) -> str:
        """The configured proxy URL."""
        return self._proxy_url

    def httpx_transport(self) -> httpx.AsyncBaseTransport | None:
        """An ``httpx`` transport bound to this proxy.

        Imported inside the method rather than at module scope so this module stays importable
        by anything that only wants the protocol -- the sidecar, for instance, which needs the
        browser half and has no httpx involvement in it.

        ``socks5://`` additionally requires ``httpx[socks]``; that is httpx's own dependency
        boundary and the ``ImportError`` it raises names it clearly enough that wrapping it
        would say less.
        """
        import httpx  # noqa: PLC0415 -- deliberate; see docstring

        return httpx.AsyncHTTPTransport(proxy=self._proxy_url)

    def browser_proxy_arg(self) -> str | None:
        """The proxy URL, which is the form ``--proxy-server`` takes."""
        return self._proxy_url

    async def health(self, *, timeout: float = DEFAULT_EGRESS_HEALTH_TIMEOUT_SECONDS) -> EgressHealth:
        """Ask the exit what address it presents.

        Fetches an address-reporting endpoint THROUGH this driver's own transport rather than
        merely opening a socket to the proxy: a proxy that is listening but forwarding
        directly answers a connectivity check perfectly while providing none of the property
        it was configured for, which is the failure worth catching.

        Never raises. An egress health check that can itself fail a caller has replaced one
        outage with two.
        """
        import httpx  # noqa: PLC0415 -- same import discipline as `httpx_transport`

        try:
            async with httpx.AsyncClient(transport=self.httpx_transport(), timeout=timeout) as client:
                response = await client.get(_HEALTH_PROBE_URL)
                response.raise_for_status()
                return EgressHealth(reachable=True, observed_address=response.text.strip())
        except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- this reports on an outage; raising out of it would make the diagnostic another thing that breaks when the thing it diagnoses does
            return EgressHealth(reachable=False, reason=f"{type(exc).__name__}: {exc}")


def SocksEgress(name: str, host: str = "127.0.0.1", port: int = 9050) -> ProxyEgress:  # noqa: N802 -- constructor-shaped by intent
    """A SOCKS5 exit, which is what both TOR and most VPN sidecars present.

    A function rather than a class because it adds no behaviour -- only the knowledge that
    SOCKS5 is the shape, and a default port. Making it a subclass would invite behaviour to
    accumulate on it that belongs on :class:`ProxyEgress`.

    The defaults are TOR's, since ``9050`` is its conventional SOCKS port and localhost is
    where a sidecar puts it. They are defaults, not assumptions: WARP, a residential pool and
    a second TOR instance are the same call with different arguments.

    :param name: stable identifier, e.g. ``"tor"`` or ``"warp"``
    :ptype name: str
    :param host: where the SOCKS proxy listens
    :ptype host: str
    :param port: SOCKS port
    :ptype port: int
    :return: a driver for that exit
    :rtype: ProxyEgress
    """
    return ProxyEgress(name, f"socks5://{host}:{port}")


def WarpEgress(host: str = "127.0.0.1", port: int = 40000, *, name: str = "warp") -> ProxyEgress:  # noqa: N802 -- constructor-shaped by intent, matching SocksEgress
    """Cloudflare WARP, as an exit.

    A named constructor rather than "you can express it with SocksEgress" -- which was true
    and is not the same thing. A backend nobody can find by name is a backend that gets
    reimplemented by the next person who needs it, and the port is the part everyone gets
    wrong.

    ``warp-cli mode proxy`` puts WARP on a local SOCKS5 listener; ``40000`` is its default,
    which is why it is the default here. This does NOT run ``warp-cli`` -- registering and
    connecting the daemon is deployment work, exactly as it is for TOR, and a library that
    tried to own another network's process lifecycle would be wrong about it in every
    deployment that already had one.

    **What WARP is and is not.** It is a VPN: it changes the address a site sees and hides
    traffic from the local network. It is not anonymity -- Cloudflare can associate the
    traffic with the account -- and its ranges are known Cloudflare ranges, so a site that
    blocks datacentre traffic will block this too. It earns its place for the opposite
    problem to TOR's: WARP addresses are far less challenged than TOR exits, so this is the
    one to reach for when a target simply dislikes the container's own address, while TOR is
    the one for non-attribution.

    :param host: where the WARP SOCKS proxy listens
    :ptype host: str
    :param port: SOCKS port; ``warp-cli mode proxy`` defaults to 40000
    :ptype port: int
    :param name: identifier recorded against results
    :ptype name: str
    :return: a driver for that exit
    :rtype: ProxyEgress
    """
    return ProxyEgress(name, f"socks5://{host}:{port}")


class EgressRegistry:
    """Named exits, so configuration names a driver rather than branching on a string.

    Deliberately tiny. It exists because the alternative is every caller holding a dict and
    re-deciding what a missing name means, and "missing means direct" is exactly the silent
    fallback that makes a deployment believe it is proxied when it is not.
    """

    def __init__(self, drivers: dict[str, EgressDriver] | None = None) -> None:
        """
        :param drivers: named drivers; ``direct`` is always available and cannot be removed
        :ptype drivers: dict[str, EgressDriver] | None
        """
        self._drivers: dict[str, EgressDriver] = {"direct": DirectEgress()}
        if drivers:
            self._drivers.update(drivers)

    def register(self, driver: EgressDriver) -> None:
        """Add or replace a driver under its own name."""
        self._drivers[driver.name] = driver

    def get(self, name: str) -> EgressDriver:
        """The driver called *name*.

        Raises rather than falling back to direct. A deployment that asked for ``tor`` and got
        the default route would be told nothing, would look correct in every log line, and
        would be wrong about the one property it configured this for.

        :param name: driver name
        :ptype name: str
        :return: the driver
        :rtype: EgressDriver
        :raises KeyError: when no driver is registered under that name
        """
        try:
            return self._drivers[name]
        except KeyError:
            available = ", ".join(sorted(self._drivers))
            raise KeyError(f"no egress driver named {name!r}; registered: {available}") from None

    def names(self) -> list[str]:
        """Every registered driver name, sorted."""
        return sorted(self._drivers)

    async def health(self, *, timeout: float = DEFAULT_EGRESS_HEALTH_TIMEOUT_SECONDS) -> dict[str, EgressHealth]:
        """Every registered exit's health, concurrently. The one place an operator asks.

        Exists because the alternative to asking here is inferring it from the symptom, and
        the symptom is indistinguishable from its opposite: a dead exit makes every target
        fail transport-side, which opens every circuit, which fills no walled queue -- since
        unreachability is deliberately not a wall. "All my targets broke at once" and "one
        daemon died" produce identical evidence until something asks the exits directly.
        """

        async def _ask(driver: EgressDriver) -> EgressHealth:
            probe = getattr(driver, "health", None)
            if probe is None:
                # A duck-typed driver written before `health` joined the protocol. Reported as
                # NOT reachable with a reason that says why, rather than defaulting to healthy:
                # this sweep exists so an operator can rule an exit out, and an exit that
                # cannot answer must not be the one that looks fine. Degrading to "true" here
                # would reintroduce exactly the blind spot the method was added to remove.
                return EgressHealth(reachable=False, reason="this driver does not report health")
            result: EgressHealth = await probe(timeout=timeout)
            return result

        names = sorted(self._drivers)
        # `return_exceptions=True` because this is a diagnostic over a `runtime_checkable`
        # Protocol that deliberately invites foreign implementations. `ProxyEgress.health`
        # promises never to raise and explains why; a third-party driver promises nothing, and
        # without this one such driver takes down the whole sweep -- so the operator asking
        # "which of my exits is down" gets an exception instead of the answer, at exactly the
        # moment something is already broken.
        results = await asyncio.gather(*(_ask(self._drivers[n]) for n in names), return_exceptions=True)
        report: dict[str, EgressHealth] = {}
        for name, result in zip(names, results, strict=True):
            if isinstance(result, BaseException):
                report[name] = EgressHealth(
                    reachable=False,
                    reason=f"health check raised: {type(result).__name__}: {result}",
                )
            else:
                report[name] = result
        return report
