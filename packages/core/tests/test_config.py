"""Tests for threetears.core.config."""

from __future__ import annotations

import pytest

from threetears.core.config import CoreConfig, DefaultCoreConfig


def test_default_config_values():
    cfg = DefaultCoreConfig()
    assert cfg.collection_flush == "ON_CHECKPOINT"
    assert cfg.collection_flush_interval == 30
    assert cfg.collection_flush_tables == "messages,token_usage_logs"


def test_protocol_compliance():
    cfg = DefaultCoreConfig()
    assert isinstance(cfg, CoreConfig)


def test_invalid_flush_strategy_raises():
    with pytest.raises(ValueError, match="collection_flush must be one of"):
        DefaultCoreConfig(collection_flush="NEVER")


def test_custom_config_satisfies_protocol():
    class MyConfig:
        collection_flush: str = "ALWAYS"
        collection_flush_interval: int = 10
        collection_flush_tables: str = "events"

    cfg = MyConfig()
    assert isinstance(cfg, CoreConfig)
