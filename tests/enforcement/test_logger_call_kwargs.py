"""thin shell — walker logic in :mod:`threetears.enforcement.logger_coverage`.

Runs the ``call_kwargs`` walker over every workspace package's src tree: a
``log.info("msg", key=value)`` is structlog shape, and :func:`threetears.observe.get_logger`
returns a ``logging.Logger`` subclass, so the keyword reaches ``Logger._log`` and raises
``TypeError``.

**Why this is a walker and not a test.** ``Logger.info`` guards the ``_log`` call with
``isEnabledFor``, so an offending call below the configured level never executes. The whole
suite runs at the root logger's WARNING and stays green; the call raises the moment a
process calls ``configure_logging(level="INFO")``, which only production entry points do.
The defect is invisible to every test that does not deliberately raise the level on the
module under test, and it reached the registry's default RBAC boot path exactly that way.

**Scoped to ``call_kwargs`` deliberately.** The sibling ``missing`` walker (every module
declares a module-level logger) is a contract the three consumer repos adopted with curated
per-repo exemption dicts; 3tears has never run it and adopting it here is a separate
decision with its own exemption-curation cost, not a side effect of closing this bug class.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.logger_coverage import (
    LoggerCoverageConfig,
    run_logger_coverage_enforcement,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _workspace_src_roots() -> tuple[Path, ...]:
    """every ``src`` tree under ``packages/``, at either nesting depth.

    Discovered rather than listed: a hand-maintained tuple silently stops covering the
    next package added to the workspace, which is the same class of gap this shell exists
    to close.

    :return: every workspace package src root, sorted for stable reporting
    :ptype: tuple[Path, ...]
    :rtype: tuple[Path, ...]
    """
    packages = _REPO_ROOT / "packages"
    return tuple(
        sorted(
            p
            for p in packages.glob("*/**/src")
            # a bare glob also matches `.venv/.../licenses/src` and `.mypy_cache/3.14/src`:
            # harmless to scan but they bloat the report the operator has to read.
            if p.is_dir() and not any(part.startswith(".") for part in p.relative_to(packages).parts)
        )
    )


_CONFIG = LoggerCoverageConfig(
    repo_root=_REPO_ROOT,
    src_roots=_workspace_src_roots(),
)


class TestLoggerCallKwargs:
    """no module passes the stdlib logger keywords it will reject at runtime."""

    def test_no_structlog_shaped_log_calls(self) -> None:
        """``log.info("m", key=v)`` raises TypeError once the level admits the call."""
        run_logger_coverage_enforcement(_CONFIG, walker="call_kwargs")

    def test_scan_actually_reaches_the_workspace(self) -> None:
        """a walker over zero files passes vacuously and reports nothing wrong.

        The glob is the fragile part of this shell -- a layout change that stops matching
        would turn the gate green rather than red -- so pin that it found the packages.
        """
        roots = _workspace_src_roots()

        assert len(roots) > 10, f"expected the whole workspace, found {len(roots)}: {roots}"
        assert any(r.parts[-2] == "core" for r in roots), "core package src root not discovered"
