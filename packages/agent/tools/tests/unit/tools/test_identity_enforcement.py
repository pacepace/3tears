"""tests for the pod-side identity-enforcement flag (v0.13.9 auth).

default off (inert); parses the off->warn->enforce ladder; fails loud on a typo'd value (a
tenant-security control must never silently leave the pod unverified).
"""

from __future__ import annotations

import pytest

from threetears.agent.tools.identity_enforcement import (
    ToolIdentityEnforcement,
    get_tool_identity_enforcement,
    get_tool_proxy_assertion_enforcement,
)

_ENV = "THREETEARS_TOOL_IDENTITY_ENFORCEMENT"


class TestToolIdentityEnforcement:
    def test_default_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_ENV, raising=False)
        assert get_tool_identity_enforcement() is ToolIdentityEnforcement.OFF

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("off", ToolIdentityEnforcement.OFF),
            ("warn", ToolIdentityEnforcement.WARN),
            ("enforce", ToolIdentityEnforcement.ENFORCE),
            ("ENFORCE", ToolIdentityEnforcement.ENFORCE),
            ("  Warn  ", ToolIdentityEnforcement.WARN),
        ],
    )
    def test_valid_values(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: ToolIdentityEnforcement
    ) -> None:
        monkeypatch.setenv(_ENV, raw)
        assert get_tool_identity_enforcement() is expected

    def test_invalid_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_ENV, "on")
        with pytest.raises(ValueError):
            get_tool_identity_enforcement()


_PROXY_ENV = "THREETEARS_TOOL_PROXY_ASSERTION_ENFORCEMENT"


class TestToolProxyAssertionEnforcement:
    """the proxy-assertion flag ladders INDEPENDENTLY of the identity-token flag."""

    def test_default_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_PROXY_ENV, raising=False)
        assert get_tool_proxy_assertion_enforcement() is ToolIdentityEnforcement.OFF

    def test_enforce_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_PROXY_ENV, "enforce")
        assert get_tool_proxy_assertion_enforcement() is ToolIdentityEnforcement.ENFORCE

    def test_invalid_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_PROXY_ENV, "on")
        with pytest.raises(ValueError):
            get_tool_proxy_assertion_enforcement()
