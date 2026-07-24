"""Unit tests for threetears.scrape.target_source -- pluggable scrape target
config sources (Python literal, YAML, database) and YAML-into-database
bootstrap.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.scrape.collections import ScrapeTarget, ScrapeTargetCollection
from threetears.scrape.target_source import (
    CollectionTargetSource,
    StaticTargetSource,
    YamlTargetSource,
    bootstrap_targets,
    read_yaml_targets,
)

from .test_collections import FakeDurableStore


def _target(target_id: str = "t1", **overrides) -> ScrapeTarget:
    data = {
        "target_id": target_id,
        "url": "https://example.gov/warn",
        "driver_backend": "nodriver",
        "rate_limit_key": "warn_act_state_sites",
        "cadence": "86400",
        "multi_row": True,
        "field_schema": {"employer": "str", "affected_count": "int"},
    }
    data.update(overrides)
    return ScrapeTarget(data)


@pytest.fixture()
def config() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS")


class TestStaticTargetSource:
    async def test_load_returns_the_wrapped_targets(self):
        targets = {"t1": _target("t1"), "t2": _target("t2")}
        source = StaticTargetSource(targets)

        loaded = await source.load()

        assert set(loaded) == {"t1", "t2"}
        assert loaded["t1"].url == "https://example.gov/warn"

    async def test_ad_hoc_single_target_needs_no_file_or_database(self):
        """The "ad-hoc, one-off scrape" case: a single ScrapeTarget constructed
        inline, no file or database involved at all."""
        source = StaticTargetSource({"once": _target("once", url="https://example.gov/one-off")})

        loaded = await source.load()

        assert list(loaded) == ["once"]
        assert loaded["once"].url == "https://example.gov/one-off"


class TestYamlTargetSource:
    def _write_yaml(self, tmp_path: Path) -> Path:
        path = tmp_path / "targets.yaml"
        path.write_text("""
warn_act_md:
  url: "https://example.gov/md"
  driver_backend: nodriver
  rate_limit_key: warn_act_state_sites
  cadence: "86400"
  multi_row: true
  field_schema:
    employer: str
    affected_count: int
warn_act_ny:
  url: "https://example.gov/ny"
  driver_backend: nodriver
  rate_limit_key: warn_act_state_sites
  cadence: "86400"
  field_schema:
    employer: str
""")
        return path

    async def test_load_parses_every_target(self, tmp_path: Path):
        source = YamlTargetSource(self._write_yaml(tmp_path))

        loaded = await source.load()

        assert set(loaded) == {"warn_act_md", "warn_act_ny"}
        assert loaded["warn_act_md"].url == "https://example.gov/md"
        assert loaded["warn_act_md"].multi_row is True
        assert loaded["warn_act_md"].field_schema == {"employer": str, "affected_count": int}
        assert loaded["warn_act_ny"].field_schema == {"employer": str}

    async def test_target_id_comes_from_the_yaml_key_not_a_field(self, tmp_path: Path):
        loaded = await YamlTargetSource(self._write_yaml(tmp_path)).load()
        assert loaded["warn_act_md"].target_id == "warn_act_md"

    async def test_timeout_seconds_defaults_to_thirty(self, tmp_path: Path):
        loaded = await YamlTargetSource(self._write_yaml(tmp_path)).load()
        assert loaded["warn_act_md"].timeout_seconds == 30.0

    async def test_timeout_seconds_explicit_override(self, tmp_path: Path):
        path = tmp_path / "targets.yaml"
        path.write_text("""
warn_act_ok:
  url: "https://example.gov/ok"
  driver_backend: network_capture
  rate_limit_key: warn_act_state_sites
  cadence: "86400"
  timeout_seconds: 60
  field_schema:
    employer: str
""")
        loaded = await YamlTargetSource(path).load()
        assert loaded["warn_act_ok"].timeout_seconds == 60.0

    def test_read_yaml_targets_is_plain_sync(self, tmp_path: Path):
        """No event loop needed -- safe to call at module import time."""
        targets = read_yaml_targets(self._write_yaml(tmp_path))
        assert set(targets) == {"warn_act_md", "warn_act_ny"}

    def test_missing_file_raises_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            read_yaml_targets(tmp_path / "does_not_exist.yaml")


class TestCollectionTargetSource:
    async def test_load_returns_every_target_in_the_collection(self, config: DefaultCoreConfig):
        registry = CollectionRegistry()
        collection = ScrapeTargetCollection(registry, config, nats_client=None)
        entity = collection.create(_target("t1").to_dict())
        await entity.save()

        loaded = await CollectionTargetSource(collection).load()

        assert set(loaded) == {"t1"}
        assert loaded["t1"].field_schema == {"employer": str, "affected_count": int}

    async def test_load_reflects_writes_from_a_second_collection_instance(self, config: DefaultCoreConfig):
        """A real L3 store (unlike the in-memory fallback, which is
        deliberately not multi-instance-safe -- see Chunk 08) is genuinely
        shared -- a second collection instance's write is visible."""
        registry = CollectionRegistry()
        registry.configure(l3_pool=FakeDurableStore())
        coll_a = ScrapeTargetCollection(registry, config, nats_client=None)
        coll_b = ScrapeTargetCollection(registry, config, nats_client=None)
        entity = coll_a.create(_target("t1").to_dict())
        await entity.save()

        loaded = await CollectionTargetSource(coll_b).load()

        assert set(loaded) == {"t1"}


class TestBootstrapTargets:
    async def test_seeds_every_target_into_an_empty_collection(self, config: DefaultCoreConfig):
        registry = CollectionRegistry()
        collection = ScrapeTargetCollection(registry, config, nats_client=None)
        source = StaticTargetSource({"t1": _target("t1"), "t2": _target("t2")})

        seeded = await bootstrap_targets(source, collection)

        assert seeded == 2
        assert {e.target_id for e in await collection.list_all()} == {"t1", "t2"}
        assert await collection.get("t1") is not None
        assert await collection.get("t2") is not None

    async def test_never_overwrites_a_row_already_present(self, config: DefaultCoreConfig):
        """A target edited directly through the database (not the seed
        source) must survive a bootstrap call untouched."""
        registry = CollectionRegistry()
        collection = ScrapeTargetCollection(registry, config, nats_client=None)
        live_edit = collection.create(_target("t1", url="https://example.gov/live-edited-url").to_dict())
        await live_edit.save()
        source = StaticTargetSource({"t1": _target("t1", url="https://example.gov/seed-url")})

        seeded = await bootstrap_targets(source, collection)

        assert seeded == 0
        current = await collection.get("t1")
        assert current is not None
        assert current.url == "https://example.gov/live-edited-url"

    async def test_safe_to_call_repeatedly(self, config: DefaultCoreConfig):
        """Idempotent -- calling bootstrap again after the first seed adds nothing new."""
        registry = CollectionRegistry()
        collection = ScrapeTargetCollection(registry, config, nats_client=None)
        source = StaticTargetSource({"t1": _target("t1")})

        first = await bootstrap_targets(source, collection)
        second = await bootstrap_targets(source, collection)

        assert first == 1
        assert second == 0

    async def test_bootstrapped_targets_are_visible_via_collection_target_source(self, config: DefaultCoreConfig):
        """The intended real-world composition: bootstrap from YAML, then read
        back via CollectionTargetSource -- the shape WarnActPlugin.connect() uses."""
        registry = CollectionRegistry()
        collection = ScrapeTargetCollection(registry, config, nats_client=None)
        source = StaticTargetSource({"t1": _target("t1")})

        await bootstrap_targets(source, collection)
        loaded = await CollectionTargetSource(collection).load()

        assert set(loaded) == {"t1"}
        assert loaded["t1"].field_schema == {"employer": str, "affected_count": int}
