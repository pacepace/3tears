"""the config layer owns the deliver, report and namespace-discovery timeout defaults.

these defaults used to be literals inside the modules that used them, which the hardcoded-timeout
gate could not see because it scanned a directory that did not exist. the pins here are the
behaviour those modules had, now read through the one layer the gate allows: an operator override
wins, and a blank, malformed or non-positive override falls back to the platform default rather than
declaring a timeout every call trips at once.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.agent.tools.config import (
    get_deliver_timeout,
    get_namespace_discovery_request_timeout,
    get_report_timeout,
)
from threetears.agent.tools.namespace_discovery_client import NamespaceDiscoveryClient

_POSITIVE_GETTERS = [
    pytest.param(get_deliver_timeout, "DELIVER_TIMEOUT_SECONDS", 30.0, id="deliver"),
    pytest.param(get_report_timeout, "REPORT_TIMEOUT_SECONDS", 120.0, id="report"),
]


class TestPositiveToolTimeouts:
    """a tool's declared timeout comes from its env var when positive, else the platform default."""

    @pytest.mark.parametrize(("getter", "env_var", "default"), _POSITIVE_GETTERS)
    def test_the_platform_default_applies_when_unset(
        self, monkeypatch: pytest.MonkeyPatch, getter: Any, env_var: str, default: float
    ) -> None:
        monkeypatch.delenv(env_var, raising=False)

        assert getter() == default

    @pytest.mark.parametrize(("getter", "env_var", "default"), _POSITIVE_GETTERS)
    def test_a_positive_override_wins(
        self, monkeypatch: pytest.MonkeyPatch, getter: Any, env_var: str, default: float
    ) -> None:
        monkeypatch.setenv(env_var, "45")

        assert getter() == 45.0

    @pytest.mark.parametrize("raw", ["", "not-a-number", "0", "-5"])
    @pytest.mark.parametrize(("getter", "env_var", "default"), _POSITIVE_GETTERS)
    def test_an_unusable_override_falls_back_to_the_default(
        self, monkeypatch: pytest.MonkeyPatch, getter: Any, env_var: str, default: float, raw: str
    ) -> None:
        monkeypatch.setenv(env_var, raw)

        assert getter() == default


class TestNamespaceDiscoveryTimeout:
    """the discovery client reads its default from the config layer and an explicit value wins."""

    def test_the_platform_default_applies_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("THREETEARS_TOOLSERVER_NAMESPACE_DISCOVERY_REQUEST_TIMEOUT", raising=False)

        assert get_namespace_discovery_request_timeout() == 5.0

    def test_an_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("THREETEARS_TOOLSERVER_NAMESPACE_DISCOVERY_REQUEST_TIMEOUT", "12")

        assert get_namespace_discovery_request_timeout() == 12.0

    def test_a_client_built_without_a_timeout_reads_the_config_layer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("THREETEARS_TOOLSERVER_NAMESPACE_DISCOVERY_REQUEST_TIMEOUT", "7")

        client = NamespaceDiscoveryClient(nats_client=None, namespace="ns")

        assert client.timeout_seconds == 7.0

    def test_an_explicit_timeout_wins_over_the_config_layer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("THREETEARS_TOOLSERVER_NAMESPACE_DISCOVERY_REQUEST_TIMEOUT", "7")

        client = NamespaceDiscoveryClient(nats_client=None, namespace="ns", timeout_seconds=2.5)

        assert client.timeout_seconds == 2.5
