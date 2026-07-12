"""tests for registry configuration getters."""

from __future__ import annotations

import pytest

from threetears.registry.config import get_heartbeat_max_misses


class TestHeartbeatMaxMisses:
    """tests for get_heartbeat_max_misses env parsing + clamping."""

    def test_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """returns platform default of 3 when the env var is unset."""
        monkeypatch.delenv("THREETEARS_REGISTRY_HEARTBEAT_MAX_MISSES", raising=False)
        assert get_heartbeat_max_misses() == 3

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """reads an integer override from the env var."""
        monkeypatch.setenv("THREETEARS_REGISTRY_HEARTBEAT_MAX_MISSES", "5")
        assert get_heartbeat_max_misses() == 5

    def test_clamps_below_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """values below 1 clamp to 1 (evict on the first miss)."""
        monkeypatch.setenv("THREETEARS_REGISTRY_HEARTBEAT_MAX_MISSES", "0")
        assert get_heartbeat_max_misses() == 1

    def test_invalid_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a non-integer value falls back to the platform default."""
        monkeypatch.setenv("THREETEARS_REGISTRY_HEARTBEAT_MAX_MISSES", "not-an-int")
        assert get_heartbeat_max_misses() == 3
