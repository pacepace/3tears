"""tests for the logger-coverage ``find_structlog_shaped_log_calls`` walker.

The walker exists because this defect is LEVEL-GATED: ``Logger.info`` guards its
``_log`` call with ``isEnabledFor``, so a structlog-shaped call below the configured
level never executes and never raises. A suite that leaves logging at WARNING stays
green over code that dies the moment a production entry point calls
``configure_logging(level="INFO")``.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.logger_coverage.walkers import (
    STDLIB_LOG_CALL_KWARGS,
    find_structlog_shaped_log_calls,
)

_VAR_NAMES = frozenset({"log", "_logger"})
_SKIP_BASENAMES = frozenset({"__init__.py"})


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    return repo_root


def _scan(
    src: Path,
    repo: Path,
    exempt_files: dict[str, str] | None = None,
) -> list:
    return find_structlog_shaped_log_calls(
        (src,),
        repo,
        exempt_files or {},
        _VAR_NAMES,
        _SKIP_BASENAMES,
    )


class TestFlagsTheStructlogShape:
    """an arbitrary keyword reaches ``Logger._log`` and raises ``TypeError``."""

    def test_bare_module_logger_with_a_data_kwarg_is_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "m.py", 'log.info("configured", kv_key_scope="registry")\n')

        violations = _scan(src, repo)

        assert len(violations) == 1
        assert violations[0].category == "logger_coverage.call_kwargs"
        assert violations[0].line == 1
        assert "kv_key_scope" in violations[0].reason

    def test_self_logger_attribute_receiver_is_flagged(self, tmp_path: Path) -> None:
        """classes holding their own logger are the same contract."""
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "m.py", 'class C:\n    def go(self):\n        self._logger.warning("x", pod_id=1)\n')

        violations = _scan(src, repo)

        assert len(violations) == 1
        assert "pod_id" in violations[0].reason

    def test_every_level_method_is_covered(self, tmp_path: Path) -> None:
        """the guard is on ``_log``, which every level method forwards to."""
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        body = "".join(f'log.{level}("m", key=1)\n' for level in ("debug", "info", "warning", "error", "critical"))
        _write(src / "m.py", body)

        violations = _scan(src, repo)

        assert len(violations) == 5

    def test_each_offending_kwarg_is_named(self, tmp_path: Path) -> None:
        """the message is the operator's fix list, so it must be complete."""
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "m.py", 'log.info("m", alpha=1, extra={"extra_data": {}}, beta=2)\n')

        violations = _scan(src, repo)

        assert len(violations) == 1
        assert "alpha, beta" in violations[0].reason
        assert "extra" not in violations[0].reason.split("--")[0]


class TestAcceptsWhatTheStdlibAccepts:
    """flagging a legal call is how a walker gets switched off."""

    def test_the_four_stdlib_kwargs_are_clean(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        body = "".join(f'log.info("m", {kw}=1)\n' for kw in sorted(STDLIB_LOG_CALL_KWARGS))
        _write(src / "m.py", body)

        assert _scan(src, repo) == []

    def test_printf_style_positional_args_are_clean(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "m.py", 'log.info("scope=%s ns=%s", scope, ns)\n')

        assert _scan(src, repo) == []

    def test_canonical_extra_data_form_is_clean(self, tmp_path: Path) -> None:
        """the form the reason line tells the author to use must itself pass."""
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "m.py", 'log.info("configured", extra={"extra_data": {"kv_key_scope": "registry"}})\n')

        assert _scan(src, repo) == []


class TestDoesNotSweepInDomainLoggers:
    """an object that merely has a ``.log`` method is not a stdlib logger."""

    def test_domain_audit_logger_is_not_flagged(self, tmp_path: Path) -> None:
        """``audit_logger.log(event_id=..., ...)`` has a keyword api of its own."""
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "m.py", 'await audit_logger.log(event_id=1, action="x")\n')

        assert _scan(src, repo) == []

    def test_unrecognised_bare_receiver_is_not_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "m.py", 'metrics.info("m", key=1)\n')

        assert _scan(src, repo) == []

    def test_attribute_receiver_on_a_non_self_object_is_not_flagged(self, tmp_path: Path) -> None:
        """``other.log`` is another object's logger, whose type we cannot see."""
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "m.py", 'client.log.info("m", key=1)\n')

        assert _scan(src, repo) == []


class TestFileFilters:
    """the same skip rules the sibling walker applies."""

    def test_exempt_file_is_skipped(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "m.py", 'log.info("m", key=1)\n')

        assert _scan(src, repo, exempt_files={"src/m.py": "rationale"}) == []

    def test_skip_basename_is_skipped(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "__init__.py", 'log.info("m", key=1)\n')

        assert _scan(src, repo) == []
