"""tests for the invalidation-listener walkers."""

from __future__ import annotations

import ast
from pathlib import Path

from threetears.enforcement.invalidation_listener import (
    count_l2_live_registries,
    find_starts_without_stops,
    find_unlistened_registries,
    started_registries,
    starts_without_stopping,
    unlistened_registries,
)


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    return repo_root


class TestAnL2LiveRegistryMustSubscribe:
    """the original defect: a write on replica A is evicted nowhere on replica B."""

    def test_a_configured_client_with_no_listener_is_flagged(self) -> None:
        tree = ast.parse("registry = CollectionRegistry()\nregistry.configure(l2_client=nc, kv_key_scope=scope)\n")

        assert unlistened_registries(tree) == ["registry"]

    def test_a_registry_that_starts_one_is_left_alone(self) -> None:
        tree = ast.parse(
            "registry = CollectionRegistry()\n"
            "registry.configure(l2_client=nc, kv_key_scope=scope)\n"
            "await registry.start_invalidation_listener(nc)\n"
        )

        assert unlistened_registries(tree) == []

    def test_the_per_collection_client_form_is_seen(self) -> None:
        """``nats_client=`` on the COLLECTION wins over the registry default.

        This is the shape a ``configure(l2_client=)`` sweep misses entirely, and it is how
        the devx workspace runtime went L2-live with nobody watching.
        """
        tree = ast.parse(
            "registry = CollectionRegistry()\n"
            "registry.configure(l3_pool=pool, kv_key_scope=scope)\n"
            "coll = WorkspaceCollection(registry=registry, config=cfg, nats_client=nc)\n"
        )

        assert unlistened_registries(tree) == ["registry"]

    def test_an_explicit_none_client_is_not_live(self) -> None:
        """``nats_client=None`` is the deliberate opt-out, not a forgotten listener."""
        tree = ast.parse(
            "registry = CollectionRegistry()\n"
            "registry.configure(l1_backend=backend)\n"
            "coll = PodAffinityCollection(registry=registry, config=cfg, nats_client=None)\n"
        )

        assert unlistened_registries(tree) == []

    def test_the_wrapper_call_shape_counts_as_a_start(self) -> None:
        """the registry can arrive as an ARGUMENT rather than a receiver.

        ``started_registries`` is deliberately OVER-inclusive here: it cannot tell which
        argument is the registry, so it claims every spelling the call names -- ``nc``
        included. That is safe by construction, because the only consumer subtracts it
        from ``l2_live_registries``, and a client name is never a constructed registry.
        Narrowing it would mean guessing argument positions, which breaks the moment a
        wrapper takes them in another order.
        """
        tree = ast.parse(
            "registry = CollectionRegistry()\n"
            "registry.configure(l2_client=nc, kv_key_scope=scope)\n"
            "await start_collection_invalidation_listener(registry, nc)\n"
        )

        assert "registry" in started_registries(tree)
        assert unlistened_registries(tree) == []

    def test_the_over_inclusive_start_set_cannot_mask_a_real_offender(self) -> None:
        """the safety of over-inclusion is worth pinning, not just asserting in prose."""
        tree = ast.parse(
            "registry = CollectionRegistry()\n"
            "registry.configure(l2_client=nc, kv_key_scope=scope)\n"
            "other = CollectionRegistry()\n"
            "other.configure(l2_client=nc, kv_key_scope=scope)\n"
            "await start_collection_invalidation_listener(registry, nc)\n"
        )

        assert unlistened_registries(tree) == ["other"]

    def test_two_registries_in_one_module_are_judged_independently(self) -> None:
        """the unit is the REGISTRY, not the process -- one process may hold two."""
        tree = ast.parse(
            "heartbeat = CollectionRegistry()\n"
            "heartbeat.configure(l2_client=nc, kv_key_scope=scope)\n"
            "await heartbeat.start_invalidation_listener(nc)\n"
            "rbac = CollectionRegistry()\n"
            "rbac.configure(l2_client=nc, kv_key_scope=scope)\n"
        )

        assert unlistened_registries(tree) == ["rbac"]

    def test_an_attribute_registry_is_tracked_by_its_dotted_spelling(self) -> None:
        tree = ast.parse(
            "self._registry = CollectionRegistry()\nself._registry.configure(l2_client=nc, kv_key_scope=scope)\n"
        )

        assert unlistened_registries(tree) == ["self._registry"]


class TestAStartMustBePaired:
    """a subscription that outlives its connection is the unpaired-lifecycle shape."""

    def test_a_start_with_no_stop_is_flagged(self) -> None:
        tree = ast.parse("await registry.start_invalidation_listener(nc)\n")

        assert starts_without_stopping(tree) is True

    def test_a_start_with_a_stop_is_left_alone(self) -> None:
        """the two halves routinely live in different functions, so this is module-scoped."""
        tree = ast.parse(
            "async def up():\n"
            "    await registry.start_invalidation_listener(nc)\n"
            "async def down():\n"
            "    await registry.stop_invalidation_listener()\n"
        )

        assert starts_without_stopping(tree) is False

    def test_a_module_that_starts_nothing_is_left_alone(self) -> None:
        tree = ast.parse("await registry.stop_invalidation_listener()\n")

        assert starts_without_stopping(tree) is False


class TestTheReaderIsNotVacuous:
    """a scan that matches nothing reports what a clean repo reports."""

    def test_the_count_sees_live_registries(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "a.py",
            "registry = CollectionRegistry()\nregistry.configure(l2_client=nc, kv_key_scope=s)\n",
        )
        _write(
            src / "b.py",
            "reg2 = CollectionRegistry()\nreg2.configure(l2_client=nc, kv_key_scope=s)\n",
        )

        assert count_l2_live_registries((src,)) == 2

    def test_the_count_ignores_registries_with_no_l2(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "a.py", "registry = CollectionRegistry()\nregistry.configure(l3_pool=pool)\n")

        assert count_l2_live_registries((src,)) == 0


class TestViolationRecords:
    """the report is the operator's only clue, so it must locate and explain."""

    def test_an_unlistened_registry_is_reported_at_its_construction_line(self, tmp_path: Path) -> None:
        """the construction site stays put while the configure call moves."""
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "m.py",
            "import os\n\nregistry = CollectionRegistry()\nregistry.configure(l2_client=nc, kv_key_scope=s)\n",
        )

        violations = find_unlistened_registries((src,))

        assert len(violations) == 1
        assert violations[0].line == 3
        assert violations[0].category == "invalidation_listener.live_registry_without_a_listener"
        # the symbol is the registry NAME alone; the path is already the file field, and
        # the shared exemption format is `path:line:symbol`, so embedding it would double up
        assert violations[0].symbol == "registry"
        assert "TTL is the wrong remedy" in violations[0].reason

    def test_an_unpaired_start_is_reported(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "m.py", "await registry.start_invalidation_listener(nc)\n")

        violations = find_starts_without_stops((src,))

        assert len(violations) == 1
        assert violations[0].category == "invalidation_listener.start_without_stop"
        # identifier-shaped, so it can actually be exempted -- a path-shaped symbol
        # cannot match the exemption grammar and aborts the whole domain
        assert violations[0].symbol == "m"
        assert "outlives the connection" in violations[0].reason

    def test_skip_basenames_is_honoured(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "m.py",
            "registry = CollectionRegistry()\nregistry.configure(l2_client=nc, kv_key_scope=s)\n",
        )

        assert find_unlistened_registries((src,), frozenset({"m.py"})) == []
