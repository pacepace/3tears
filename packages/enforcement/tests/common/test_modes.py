"""tests for ``modes`` module."""

from __future__ import annotations

import pytest

from threetears.enforcement.common.modes import (
    MODE_REPORT,
    MODE_STRICT,
    ModeError,
    resolve_mode,
)


def test_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("X_ENFORCEMENT_MODE", raising=False)
    assert resolve_mode("X_ENFORCEMENT_MODE") == MODE_STRICT


def test_default_can_be_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("X_ENFORCEMENT_MODE", raising=False)
    assert resolve_mode("X_ENFORCEMENT_MODE", default=MODE_REPORT) == MODE_REPORT


def test_strict_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X_ENFORCEMENT_MODE", "strict")
    assert resolve_mode("X_ENFORCEMENT_MODE") == MODE_STRICT


def test_report_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X_ENFORCEMENT_MODE", "report")
    assert resolve_mode("X_ENFORCEMENT_MODE") == MODE_REPORT


def test_uppercase_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X_ENFORCEMENT_MODE", "STRICT")
    assert resolve_mode("X_ENFORCEMENT_MODE") == MODE_STRICT


def test_whitespace_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X_ENFORCEMENT_MODE", "  Report  ")
    assert resolve_mode("X_ENFORCEMENT_MODE") == MODE_REPORT


def test_unknown_value_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X_ENFORCEMENT_MODE", "audit")
    with pytest.raises(ModeError, match="X_ENFORCEMENT_MODE"):
        resolve_mode("X_ENFORCEMENT_MODE")


def test_unknown_default_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("X_ENFORCEMENT_MODE", raising=False)
    with pytest.raises(ModeError, match="X_ENFORCEMENT_MODE"):
        resolve_mode("X_ENFORCEMENT_MODE", default="bogus")


def test_env_var_name_appears_in_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_DOMAIN_MODE", "wrong")
    with pytest.raises(ModeError) as exc_info:
        resolve_mode("MY_DOMAIN_MODE")
    assert "MY_DOMAIN_MODE" in str(exc_info.value)
