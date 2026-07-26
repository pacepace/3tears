"""Egress drivers: an exit is a driver, and a misconfigured one must never be silent.

Most of these are about the failure mode this module exists to prevent, which is not "the
proxy is wrong" but "the proxy is absent and nothing said so". A deployment that configured
TOR for non-attribution and silently got the default route is wrong about the one property it
turned this on for, and every log line still looks correct.
"""

from __future__ import annotations

import pytest
from threetears.core.egress import (
    DirectEgress,
    EgressDriver,
    EgressRegistry,
    ProxyEgress,
    SocksEgress,
)


def test_direct_is_a_driver_not_a_special_case() -> None:
    """The no-proxy path goes through the same seam as every other exit.

    A caller that special-cased ``None`` would grow a second code path nothing exercises, and
    that path is exactly where "the seam was bypassed" hides.
    """
    direct = DirectEgress()
    assert isinstance(direct, EgressDriver)
    assert direct.name == "direct"
    assert direct.httpx_transport() is None
    assert direct.browser_proxy_arg() is None


def test_a_proxy_driver_serves_both_consumers() -> None:
    """An exit is not an HTTP concept, so a driver that only does httpx is not an exit.

    A deployment whose API calls proxy and whose scrapes do not is in a worse position than
    one with no proxying at all: it believes it has the property.
    """
    egress = ProxyEgress("corp", "http://proxy.internal:3128")
    assert isinstance(egress, EgressDriver)
    assert egress.browser_proxy_arg() == "http://proxy.internal:3128"
    transport = egress.httpx_transport()
    assert transport is not None


def test_socks_builds_the_shape_tor_and_most_vpn_sidecars_present() -> None:
    tor = SocksEgress("tor")
    assert tor.name == "tor"
    assert tor.proxy_url == "socks5://127.0.0.1:9050"
    assert tor.browser_proxy_arg() == "socks5://127.0.0.1:9050"

    warp = SocksEgress("warp", host="10.0.0.5", port=1080)
    assert warp.proxy_url == "socks5://10.0.0.5:1080"


def test_an_exit_with_no_address_is_refused() -> None:
    """Because it is the default route wearing a name.

    This is the single most dangerous misconfiguration here: it produces a driver that reports
    itself as ``tor`` in every log and leaves by the container's own IP.
    """
    with pytest.raises(ValueError, match="no proxy url"):
        ProxyEgress("tor", "")


def test_an_exit_with_no_name_is_refused() -> None:
    """Results are recorded against the name, so a nameless exit cannot be told apart later."""
    with pytest.raises(ValueError, match="must have a name"):
        ProxyEgress("", "socks5://127.0.0.1:9050")


def test_the_registry_always_has_direct() -> None:
    assert EgressRegistry().get("direct").name == "direct"


def test_an_unknown_exit_raises_rather_than_falling_back() -> None:
    """The whole point. A deployment that asked for ``tor`` and got the default route would be
    told nothing, look correct everywhere, and be wrong about the one property it configured.

    The message names what IS registered, because the realistic cause is a typo or a driver
    that was never wired, and both are answered by that list.
    """
    registry = EgressRegistry({"tor": SocksEgress("tor")})
    with pytest.raises(KeyError) as exc:
        registry.get("torr")
    assert "torr" in str(exc.value)
    assert "tor" in str(exc.value)
    assert "direct" in str(exc.value)


def test_drivers_can_be_registered_and_listed() -> None:
    """Adding a fourth exit is one class and no change to any caller -- the reason for the seam."""
    registry = EgressRegistry()
    registry.register(SocksEgress("tor"))
    registry.register(ProxyEgress("warp", "socks5://127.0.0.1:1080"))
    registry.register(ProxyEgress("residential", "http://pool.example:8080"))

    assert registry.names() == ["direct", "residential", "tor", "warp"]
    assert registry.get("residential").browser_proxy_arg() == "http://pool.example:8080"


def test_a_deployments_own_object_satisfies_the_protocol_structurally() -> None:
    """Structural, like ``CircuitBreakerLike``: an app with its own egress object should not
    have to inherit from anything here to use the seam."""

    class _TheirOwn:
        @property
        def name(self) -> str:
            return "theirs"

        def httpx_transport(self) -> None:
            return None

        def browser_proxy_arg(self) -> str | None:
            return "socks5://their.host:9050"

    assert isinstance(_TheirOwn(), EgressDriver)
    assert EgressRegistry({"theirs": _TheirOwn()}).get("theirs").browser_proxy_arg() is not None
