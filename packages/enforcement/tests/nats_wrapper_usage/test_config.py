"""tests for ``NatsWrapperConfig`` dataclass."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from threetears.enforcement.nats_wrapper_usage import NatsWrapperConfig


class TestDefaults:
    def test_minimal_construction(self, tmp_path: Path) -> None:
        config = NatsWrapperConfig(repo_root=tmp_path)
        assert config.repo_root == tmp_path
        assert config.src_roots is None
        assert config.tests_root is None
        assert config.exemptions_path is None
        assert config.mode_env_var == "NATS_ENFORCEMENT_MODE"
        assert config.forbidden_module == "nats"

    def test_is_frozen(self, tmp_path: Path) -> None:
        config = NatsWrapperConfig(repo_root=tmp_path)
        try:
            config.repo_root = tmp_path / "other"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            return
        raise AssertionError("expected FrozenInstanceError")


class TestOverrides:
    def test_explicit_src_roots(self, tmp_path: Path) -> None:
        roots = (tmp_path / "a", tmp_path / "b")
        config = NatsWrapperConfig(repo_root=tmp_path, src_roots=roots)
        assert config.src_roots == roots

    def test_explicit_tests_root(self, tmp_path: Path) -> None:
        tests = tmp_path / "tests"
        config = NatsWrapperConfig(repo_root=tmp_path, tests_root=tests)
        assert config.tests_root == tests

    def test_explicit_exemptions_path(self, tmp_path: Path) -> None:
        ex = tmp_path / "_nats_exemptions.txt"
        config = NatsWrapperConfig(
            repo_root=tmp_path, exemptions_path=ex,
        )
        assert config.exemptions_path == ex

    def test_explicit_mode_env_var(self, tmp_path: Path) -> None:
        config = NatsWrapperConfig(
            repo_root=tmp_path, mode_env_var="MY_MODE",
        )
        assert config.mode_env_var == "MY_MODE"

    def test_explicit_forbidden_module(self, tmp_path: Path) -> None:
        config = NatsWrapperConfig(
            repo_root=tmp_path, forbidden_module="kafka",
        )
        assert config.forbidden_module == "kafka"

    def test_full_roundtrip(self, tmp_path: Path) -> None:
        roots = (tmp_path / "src",)
        tests = tmp_path / "tests"
        ex = tmp_path / "ex.txt"
        config = NatsWrapperConfig(
            repo_root=tmp_path,
            src_roots=roots,
            tests_root=tests,
            exemptions_path=ex,
            mode_env_var="NATS_TEST_MODE",
            forbidden_module="nats",
        )
        assert config.repo_root == tmp_path
        assert config.src_roots == roots
        assert config.tests_root == tests
        assert config.exemptions_path == ex
        assert config.mode_env_var == "NATS_TEST_MODE"
        assert config.forbidden_module == "nats"
