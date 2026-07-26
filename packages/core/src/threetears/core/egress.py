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

**Nothing here opens a socket.** A driver describes an exit; it does not run the daemon that
provides it. Starting `tor` or `warp-cli` is deployment work -- a container, a sidecar, a host
service -- and a library that tried to own process lifecycle for someone else's network would
be wrong about it in every deployment that already had one.

**Direct is a driver, not a special case.** The no-proxy path going through the same seam is
what stops "direct" quietly meaning "the seam was bypassed", and it means a caller never
branches on whether egress is configured at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - import-time only
    import httpx

__all__ = [
    "DirectEgress",
    "EgressDriver",
    "EgressRegistry",
    "ProxyEgress",
    "SocksEgress",
]


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
