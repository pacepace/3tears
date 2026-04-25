"""
walker invocation -- runs migration yugabyte-safety enforcement against
every migration directory shipped by the 3tears repo.

3tears packages with migration directories:

- ``packages/agent-memory/src/threetears/agent/memory/migrations/``
- ``packages/agent-tools/src/threetears/agent/tools/migrations/``
  (only if present)
- ``packages/agent-workspace/src/threetears/agent/workspace/migrations/``
  (only if present)
- ``packages/conversations/src/threetears/conversations/migrations/``
  (only if present)
- ``packages/langgraph/src/threetears/langgraph/migrations/``
  (only if present)

mode is controlled by ``MIGRATION_ENFORCEMENT_MODE``. defaults to
``report`` during the retro-rewrite window; flips to ``strict`` once
every migration has been reshaped onto helpers or carries a
rationale-tagged exemption (sub-task 7).

exemption file: ``tests/enforcement/_migration_exemptions.txt`` in the
package root. each entry must carry a ``# rationale: <reason>`` suffix;
blank rationales are rejected by :func:`load_exemptions`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from threetears.core.data.migrations.enforcement import (
    WalkerConfig,
    get_enforcement_mode,
    load_exemptions,
    walk_migration_directory,
)

__all__: list[str] = []


_REPO_ROOT = Path(__file__).resolve().parents[4]


_MIGRATION_DIRS_CANDIDATES: tuple[Path, ...] = (
    _REPO_ROOT
    / "packages" / "agent-memory" / "src" / "threetears"
    / "agent" / "memory" / "migrations",
    _REPO_ROOT
    / "packages" / "agent-tools" / "src" / "threetears"
    / "agent" / "tools" / "migrations",
    _REPO_ROOT
    / "packages" / "agent-workspace" / "src" / "threetears"
    / "agent" / "workspace" / "migrations",
    _REPO_ROOT
    / "packages" / "conversations" / "src" / "threetears"
    / "conversations" / "migrations",
    _REPO_ROOT
    / "packages" / "langgraph" / "src" / "threetears"
    / "langgraph" / "migrations",
)


_EXEMPTION_FILE = (
    Path(__file__).resolve().parent / "_migration_exemptions.txt"
)


def _existing_dirs() -> tuple[Path, ...]:
    """
    return the subset of candidate migration dirs that exist on disk.

    :return: tuple of existing migration directory paths
    :rtype: tuple[Path, ...]
    """
    result = tuple(d for d in _MIGRATION_DIRS_CANDIDATES if d.exists())
    return result


def test_migration_yugabyte_safety_enforcement_3tears() -> None:
    """
    walk every 3tears-package migration directory; verify yugabyte-safe shape.

    :return: nothing
    :rtype: None
    :raises AssertionError: when ``MIGRATION_ENFORCEMENT_MODE`` is
        ``strict`` and at least one violation is not exempted
    """
    exemptions, missing = load_exemptions(_EXEMPTION_FILE)
    assert missing == [], (
        "exemption entries lack a non-blank ``# rationale: ...`` "
        f"suffix: {missing}"
    )
    config = WalkerConfig(
        migration_dirs=_existing_dirs(),
        exemptions=exemptions,
        repo_root=_REPO_ROOT,
    )
    violations = walk_migration_directory(config)
    if not violations:
        return
    mode = get_enforcement_mode()
    formatted = "\n".join(v.format() for v in violations)
    if mode == "report":
        pytest.skip(
            f"3tears migration yugabyte-safety: {len(violations)} "
            f"violation(s) (mode=report)\n{formatted}",
        )
        return
    raise AssertionError(
        f"3tears migration yugabyte-safety: {len(violations)} "
        f"strict violation(s) (mode={mode})\n{formatted}",
    )


def test_walker_runs_under_15_seconds() -> None:
    """
    walker runtime must stay well under the CLAUDE.md 15s budget.

    :return: nothing
    :rtype: None
    """
    import time

    config = WalkerConfig(
        migration_dirs=_existing_dirs(),
        exemptions=frozenset(),
        repo_root=_REPO_ROOT,
    )
    start = time.monotonic()
    walk_migration_directory(config)
    elapsed = time.monotonic() - start
    assert elapsed < 15.0, (
        f"walker took {elapsed:.2f}s (budget 15s)"
    )


def test_migration_dirs_resolved() -> None:
    """
    sanity check: at least one migration directory exists on disk.

    :return: nothing
    :rtype: None
    """
    dirs = _existing_dirs()
    assert len(dirs) > 0, (
        f"no migration directories resolved from candidates: "
        f"{_MIGRATION_DIRS_CANDIDATES}"
    )


def test_enforcement_mode_env_var_recognized() -> None:
    """
    setting ``MIGRATION_ENFORCEMENT_MODE`` flips ``get_enforcement_mode``.

    :return: nothing
    :rtype: None
    """
    prior = os.environ.get("MIGRATION_ENFORCEMENT_MODE")
    try:
        os.environ["MIGRATION_ENFORCEMENT_MODE"] = "strict"
        assert get_enforcement_mode() == "strict"
        os.environ["MIGRATION_ENFORCEMENT_MODE"] = "report"
        assert get_enforcement_mode() == "report"
    finally:
        if prior is None:
            os.environ.pop("MIGRATION_ENFORCEMENT_MODE", None)
        else:
            os.environ["MIGRATION_ENFORCEMENT_MODE"] = prior
