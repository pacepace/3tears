"""Egress drivers: an exit is a driver, and a misconfigured one must never be silent.

Most of these are about the failure mode this module exists to prevent, which is not "the
proxy is wrong" but "the proxy is absent and nothing said so". A deployment that configured
TOR for non-attribution and silently got the default route is wrong about the one property it
turned this on for, and every log line still looks correct.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from threetears.core.egress import (
    DirectEgress,
    EgressDriver,
    EgressRegistry,
    EgressHealth,
    ProxyEgress,
    SocksEgress,
    WarpEgress,
)


def test_direct_is_a_driver_not_a_special_case() -> None:
    """The no-proxy path goes through the same seam as every other exit.

    A caller that special-cased ``None`` would grow a second code path nothing exercises, and
    that path is exactly where "the seam was bypassed" hides.
    """
    direct = DirectEgress()
    assert isinstance(direct, EgressDriver)
    assert direct.name == "direct"

    # `direct://` rather than `None`, and the two are not interchangeable. A browser consumer
    # that already carries a proxy -- the sidecar applies one container-wide at launch -- reads
    # `None` as "no opinion" and keeps it, so an explicitly-direct request left by the
    # container's proxy while still being reported as `direct`. A value that overrides is the
    # only way for this class to mean what its name says.
    assert direct.browser_proxy_arg() == "direct://"

    # httpx has no inherited proxy to override -- a transport is constructed per client, not
    # applied process-wide -- so `None` there genuinely is the default route. The asymmetry is
    # in the consumers, not in this class.
    assert direct.httpx_transport() is None


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

        async def health(self, *, timeout: float = 10.0) -> EgressHealth:
            return EgressHealth(reachable=True, observed_address="203.0.113.9")

    assert isinstance(_TheirOwn(), EgressDriver)
    assert EgressRegistry({"theirs": _TheirOwn()}).get("theirs").browser_proxy_arg() is not None


async def test_a_driver_that_cannot_report_health_is_unknown_not_healthy() -> None:
    """The sweep exists so an operator can rule an exit OUT.

    A duck-typed driver written before `health` joined the protocol has no answer. Defaulting
    it to reachable would put the one exit nobody can check at the top of the "these are fine"
    list, which is the blind spot the method was added to remove -- so it reports unreachable
    with a reason naming why, and the operator goes and looks.
    """

    class _Older:
        @property
        def name(self) -> str:
            return "older"

        def httpx_transport(self) -> None:
            return None

        def browser_proxy_arg(self) -> str | None:
            return "socks5://old.host:9050"

    report = await EgressRegistry({"older": _Older()}).health()  # type: ignore[dict-item]

    assert report["older"].reachable is False
    assert "does not report health" in (report["older"].reason or "")
    assert report["direct"].reachable is True, "the sweep still answers for the drivers that can"


class TestEgressHealth:
    """A dead exit and a fleet-wide outage produce identical evidence until something asks."""

    async def test_a_reachable_proxy_reports_the_address_it_presents(self):
        """The address, not just reachability. A proxy that is listening but forwarding
        directly passes a connectivity check while providing none of the property it exists
        for, which is precisely the misconfiguration worth catching."""
        import httpx

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="185.220.101.5\n")

        driver = ProxyEgress("tor", "socks5://127.0.0.1:9050")
        with patch.object(driver, "httpx_transport", return_value=httpx.MockTransport(_handler)):
            health = await driver.health()

        assert health.reachable is True
        assert health.observed_address == "185.220.101.5"

    async def test_a_dead_daemon_reports_unreachable_rather_than_raising(self):
        """The check runs to describe an outage; raising would make it a second one.

        This is the whole point of the method: without it a dead tor daemon fails every render
        transport-side, opens every circuit, and fills no walled queue -- because unreachable
        is deliberately not walled -- so the operator queue stays empty while the fleet decays.
        """
        import httpx

        def _refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        driver = ProxyEgress("tor", "socks5://127.0.0.1:9050")
        with patch.object(driver, "httpx_transport", return_value=httpx.MockTransport(_refuse)):
            health = await driver.health()

        assert health.reachable is False
        assert "ConnectError" in (health.reason or "")

    async def test_direct_is_not_probed(self):
        """Probing the default route tests the internet, not the exit."""
        health = await DirectEgress().health()
        assert health.reachable is True

    async def test_the_registry_answers_for_every_exit_at_once(self):
        """One place to ask, because the alternative is inferring it from a symptom that
        looks identical to its opposite."""
        import httpx

        def _refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        dead = ProxyEgress("tor", "socks5://127.0.0.1:9050")
        registry = EgressRegistry({"tor": dead})
        with patch.object(dead, "httpx_transport", return_value=httpx.MockTransport(_refuse)):
            report = await registry.health()

        assert set(report) == {"direct", "tor"}
        assert report["direct"].reachable is True
        assert report["tor"].reachable is False, "a dead exit was reported as healthy"


def test_warp_is_a_named_exit_on_its_own_default_port() -> None:
    """Named rather than "expressible with SocksEgress", which was true and not the same thing.

    A backend nobody can find by name gets reimplemented by the next person who needs it, and
    the port is the part everyone gets wrong: `warp-cli mode proxy` listens on 40000, not
    TOR's 9050.
    """
    warp = WarpEgress()

    assert warp.name == "warp"
    assert warp.browser_proxy_arg() == "socks5://127.0.0.1:40000"
    assert warp.proxy_url != SocksEgress("tor").proxy_url, "warp inherited TOR's port"


def test_two_exits_can_run_side_by_side_under_their_own_names() -> None:
    """The whole point of the registry: results record WHICH exit they came from.

    Without distinct names, "this target is blocked" cannot be told apart from "this target is
    blocked from this exit", and one blocked exit poisons a target permanently.
    """
    registry = EgressRegistry({"tor": SocksEgress("tor"), "warp": WarpEgress()})

    assert registry.names() == ["direct", "tor", "warp"]
    assert registry.get("warp").browser_proxy_arg() != registry.get("tor").browser_proxy_arg()


async def test_one_raising_driver_does_not_take_down_the_sweep() -> None:
    """The sweep is a diagnostic, so it must survive the thing it diagnoses.

    `EgressDriver` is `runtime_checkable` and deliberately invites foreign implementations.
    `ProxyEgress.health` promises never to raise; a third-party driver promises nothing. Without
    `return_exceptions=True` one such driver replaced the whole report with an exception -- so
    an operator asking "which of my exits is down" got nothing back, precisely when something
    already was.
    """

    class _Exploding:
        @property
        def name(self) -> str:
            return "boom"

        def httpx_transport(self) -> None:
            return None

        def browser_proxy_arg(self) -> str | None:
            return None

        async def health(self, *, timeout: float = 10.0) -> EgressHealth:
            raise RuntimeError("the driver itself is broken")

    report = await EgressRegistry({"boom": _Exploding()}).health()

    assert report["boom"].reachable is False
    assert "RuntimeError" in (report["boom"].reason or "")
    assert report["direct"].reachable is True, "one broken driver hid every other exit's status"
