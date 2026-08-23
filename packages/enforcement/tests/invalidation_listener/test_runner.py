"""tests for ``run_invalidation_listener_enforcement`` orchestration.

The floor's FAILING branch is the important one here. It is the guard that makes every
other assertion in this domain mean something -- a scan reaching nothing reports exactly
what a clean repo reports -- and an unverified guard is indistinguishable from an absent
one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.invalidation_listener import (
    InvalidationListenerConfig,
    run_invalidation_listener_enforcement,
)

_VALID_RATIONALE = (
    "constructed here and handed to WiringHolder.registry; service.py starts the listener on it "
    "in start() and stops it in stop(), because the shutdown path lives there and not here"
)


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    return repo_root


_OFFENDER = "registry = CollectionRegistry()\nregistry.configure(l2_client=nc, kv_key_scope=s)\n"
_CLEAN = (
    "registry = CollectionRegistry()\n"
    "registry.configure(l2_client=nc, kv_key_scope=s)\n"
    "await registry.start_invalidation_listener(nc)\n"
    "await registry.stop_invalidation_listener()\n"
)


class TestWalkerSelection:
    """a typo must not silently no-op a walker."""

    def test_an_unknown_walker_raises(self, tmp_path: Path) -> None:
        config = InvalidationListenerConfig(repo_root=_make_repo(tmp_path / "repo"), minimum_live_registries=0)

        with pytest.raises(ValueError, match="walker must be one of"):
            run_invalidation_listener_enforcement(config, walker="unlistend")

    def test_selecting_one_walker_does_not_run_the_other(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """an unpaired start is not an unlistened registry, and vice versa."""
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        # starts a listener (so NOT unlistened) but never stops it (so unpaired)
        _write(
            src / "m.py",
            "registry = CollectionRegistry()\n"
            "registry.configure(l2_client=nc, kv_key_scope=s)\n"
            "await registry.start_invalidation_listener(nc)\n",
        )
        monkeypatch.setenv("IL_TEST_MODE", "strict")
        config = InvalidationListenerConfig(
            repo_root=repo,
            src_roots=(src,),
            minimum_live_registries=0,
            mode_env_var="IL_TEST_MODE",
        )

        run_invalidation_listener_enforcement(config, walker="unlistened")

        with pytest.raises(pytest.fail.Exception, match="1 violation"):
            run_invalidation_listener_enforcement(config, walker="unpaired")


class TestTheNonVacuityFloor:
    """the guard that makes a green report mean something."""

    def test_a_scan_below_the_floor_fails_even_with_no_violations(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """THE case. A broken reader finds no offenders, which reads as a clean repo."""
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "m.py", "x = 1\n")  # nothing for the reader to see at all
        monkeypatch.setenv("IL_TEST_MODE", "strict")
        config = InvalidationListenerConfig(
            repo_root=repo,
            src_roots=(src,),
            minimum_live_registries=3,
            mode_env_var="IL_TEST_MODE",
        )

        with pytest.raises(pytest.fail.Exception, match="BROKEN READER"):
            run_invalidation_listener_enforcement(config)

    def test_the_floor_message_names_the_roots_it_searched(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """the operator's first question is "where did it look", so answer it."""
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "m.py", "x = 1\n")
        monkeypatch.setenv("IL_TEST_MODE", "strict")
        config = InvalidationListenerConfig(
            repo_root=repo,
            src_roots=(src,),
            minimum_live_registries=1,
            mode_env_var="IL_TEST_MODE",
        )

        with pytest.raises(pytest.fail.Exception) as excinfo:
            run_invalidation_listener_enforcement(config)

        assert str(src) in str(excinfo.value)

    def test_the_floor_is_checked_before_findings(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """a broken reader must not be reported as a violation count of zero."""
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "m.py", _OFFENDER)
        monkeypatch.setenv("IL_TEST_MODE", "strict")
        config = InvalidationListenerConfig(
            repo_root=repo,
            src_roots=(src,),
            minimum_live_registries=99,
            mode_env_var="IL_TEST_MODE",
        )

        with pytest.raises(pytest.fail.Exception) as excinfo:
            run_invalidation_listener_enforcement(config)

        assert "BROKEN READER" in str(excinfo.value)
        assert "violation" not in str(excinfo.value)

    def test_a_satisfied_floor_lets_the_run_proceed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "m.py", _CLEAN)
        monkeypatch.setenv("IL_TEST_MODE", "strict")
        config = InvalidationListenerConfig(
            repo_root=repo,
            src_roots=(src,),
            minimum_live_registries=1,
            mode_env_var="IL_TEST_MODE",
        )

        run_invalidation_listener_enforcement(config)


class TestExemptions:
    """an exemption must silence its own entry and nothing else."""

    def test_an_exemption_silences_the_named_registry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "m.py", _OFFENDER)
        exemptions = repo / "_exempt.txt"
        exemptions.write_text(f"# rationale: {_VALID_RATIONALE}\nsrc/m.py:*:registry\n")
        monkeypatch.setenv("IL_TEST_MODE", "strict")
        config = InvalidationListenerConfig(
            repo_root=repo,
            src_roots=(src,),
            minimum_live_registries=0,
            exemptions_path=exemptions,
            mode_env_var="IL_TEST_MODE",
        )

        run_invalidation_listener_enforcement(config, walker="unlistened")

    def test_an_exemption_does_not_silence_a_different_registry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """a blanket effect would make one entry an off switch for the file."""
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "m.py",
            _OFFENDER + "other = CollectionRegistry()\nother.configure(l2_client=nc, kv_key_scope=s)\n",
        )
        exemptions = repo / "_exempt.txt"
        exemptions.write_text(f"# rationale: {_VALID_RATIONALE}\nsrc/m.py:*:registry\n")
        monkeypatch.setenv("IL_TEST_MODE", "strict")
        config = InvalidationListenerConfig(
            repo_root=repo,
            src_roots=(src,),
            minimum_live_registries=0,
            exemptions_path=exemptions,
            mode_env_var="IL_TEST_MODE",
        )

        with pytest.raises(pytest.fail.Exception, match="1 violation"):
            run_invalidation_listener_enforcement(config, walker="unlistened")

    def test_an_unpaired_violation_can_actually_be_exempted(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """the symbol must be identifier-shaped, or the entry aborts the whole domain.

        A path-shaped symbol cannot match the exemption grammar at all: the only entry
        that could name it raises ``ExemptionError`` and takes every other finding down
        with it, so the escape hatch is not merely unused, it is a trap.
        """
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "starter.py", "await registry.start_invalidation_listener(nc)\n")
        exemptions = repo / "_exempt.txt"
        exemptions.write_text(f"# rationale: {_VALID_RATIONALE}\nsrc/starter.py:*:starter\n")
        monkeypatch.setenv("IL_TEST_MODE", "strict")
        config = InvalidationListenerConfig(
            repo_root=repo,
            src_roots=(src,),
            minimum_live_registries=0,
            exemptions_path=exemptions,
            mode_env_var="IL_TEST_MODE",
        )

        run_invalidation_listener_enforcement(config, walker="unpaired")

    def test_an_absent_exemption_file_is_not_an_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """a repo with nothing to exempt should not have to create an empty file."""
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "m.py", _CLEAN)
        monkeypatch.setenv("IL_TEST_MODE", "strict")
        config = InvalidationListenerConfig(
            repo_root=repo,
            src_roots=(src,),
            minimum_live_registries=0,
            exemptions_path=repo / "_absent.txt",
            mode_env_var="IL_TEST_MODE",
        )

        run_invalidation_listener_enforcement(config)


class TestReportMode:
    """report mode is the adoption ramp; it must not fail while it reports."""

    def test_report_mode_returns_despite_violations(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "m.py", _OFFENDER)
        monkeypatch.setenv("IL_TEST_MODE", "report")
        config = InvalidationListenerConfig(
            repo_root=repo,
            src_roots=(src,),
            minimum_live_registries=0,
            mode_env_var="IL_TEST_MODE",
        )

        run_invalidation_listener_enforcement(config)

    def test_report_mode_does_not_soften_the_floor(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """report mode forgives findings; it must not forgive a broken reader.

        A repo adopting in report mode is exactly the one that would never notice its
        scan reaching nothing, because it is not reading the failures yet.
        """
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "m.py", "x = 1\n")
        monkeypatch.setenv("IL_TEST_MODE", "report")
        config = InvalidationListenerConfig(
            repo_root=repo,
            src_roots=(src,),
            minimum_live_registries=2,
            mode_env_var="IL_TEST_MODE",
        )

        with pytest.raises(pytest.fail.Exception, match="BROKEN READER"):
            run_invalidation_listener_enforcement(config)
