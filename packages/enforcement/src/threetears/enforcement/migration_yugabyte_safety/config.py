"""configuration dataclass for migration-yugabyte-safety enforcement.

the migration-yugabyte-safety domain is a thin pytest adapter around
the existing walker logic in
:mod:`threetears.core.data.migrations.enforcement`. that walker
enforces five rules (M-1 .. M-5) protecting yugabyte's transactional
semantics across alembic-style migration files: no DDL/DML mixing
inside ``DO`` blocks (M-1), backfill ``UPDATE`` must carry a
replay-guard predicate (M-2), bare ``CREATE TABLE`` / ``CREATE
INDEX`` / ``ADD COLUMN`` are non-idempotent (M-3), composite-PK adds
must preserve a ``UNIQUE`` index on the legacy id column (M-4), and
``TRUNCATE TABLE`` is forbidden inside migrations (M-5).

per-repo configuration goes through :class:`MigrationYugabyteConfig`;
the runner orchestrates the walker invocation, applies exemptions,
emits the standard report, and either fails the test or returns
silently according to mode.

defaults:

- :attr:`mode_env_var` defaults to ``MIGRATION_ENFORCEMENT_MODE`` —
  the env-var name the canonical hub harness already reads.
- :attr:`default_mode` defaults to :data:`MODE_REPORT
  <threetears.enforcement.common.modes.MODE_REPORT>` per the
  task-00 spec (the retro-rewrite-window posture documented in the
  canonical's header comments). the canonical's underlying
  :func:`threetears.core.data.migrations.enforcement.get_enforcement_mode`
  defaults to ``"strict"`` and is invoked unconditionally; the
  package adapter restores the documented retro-rewrite default by
  routing through :func:`resolve_mode
  <threetears.enforcement.common.modes.resolve_mode>` with this
  field as the fallback. consumers that want hard-fail flip this to
  :data:`MODE_STRICT
  <threetears.enforcement.common.modes.MODE_STRICT>` per repo.
- :attr:`migration_dirs` is explicit and per-repo — the canonical
  hard-coded ``src/3tears/hub/migrations/``; consumers in other repos
  point to their own migration trees here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from threetears.enforcement.common import MODE_REPORT

__all__ = ["MigrationYugabyteConfig"]


@dataclass(frozen=True)
class MigrationYugabyteConfig:
    """per-repo config for the migration-yugabyte-safety enforcement domain.

    :ivar repo_root: absolute path to the consumer repo's root (the
        directory containing its top-level ``pyproject.toml``). used
        for relative-path rendering in the report and as the anchor
        the walker uses to compute exemption-match keys.
    :ivar migration_dirs: absolute paths to migration directories the
        walker should scan (e.g.
        ``(repo_root / "src" / "3tears" / "hub" / "migrations",)``).
        directories that do not exist on disk are silently skipped at
        runtime — preserving the canonical ``_existing_dirs()``
        behaviour so a repo that ships no migrations today can declare
        the future path without a probe.
    :ivar exemptions_path: path to ``_migration_exemptions.txt``;
        ``None`` means "no exemptions file". format is
        ``<file_path>:<rule_name> # rationale: <reason>`` per line, as
        parsed by
        :func:`threetears.core.data.migrations.enforcement.load_exemptions`.
        blank rationales are rejected at load time.
    :ivar mode_env_var: environment variable controlling strict vs
        report mode. defaults to ``MIGRATION_ENFORCEMENT_MODE`` —
        the canonical hub harness reads this exact name.
    :ivar default_mode: fallback when the env var is unset. defaults
        to :data:`MODE_REPORT
        <threetears.enforcement.common.modes.MODE_REPORT>` to preserve
        the canonical's retro-rewrite-window posture; consumers that
        want hard-fail set this to :data:`MODE_STRICT
        <threetears.enforcement.common.modes.MODE_STRICT>` per repo.
    """

    repo_root: Path
    migration_dirs: tuple[Path, ...]
    exemptions_path: Path | None = None
    mode_env_var: str = "MIGRATION_ENFORCEMENT_MODE"
    default_mode: str = MODE_REPORT
