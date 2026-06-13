"""per-repo configuration for dependency-alignment enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["DependencyAlignmentConfig"]


@dataclass(frozen=True)
class DependencyAlignmentConfig:
    """knobs the consuming repo's thin test shell injects.

    :ivar repo_root: workspace root (the directory holding ``packages/``)
    :ivar package_globs: glob patterns (relative to ``repo_root``) whose
        matches are workspace package directories -- each must contain a
        ``pyproject.toml`` and a ``src/threetears`` tree to participate
    :ivar exemptions_path: rationale-required exemption file, or ``None``
        for no exemptions
    :ivar mode_env_var: environment variable selecting strict/report mode
    :ivar contract_packages: package directories (relative to
        ``repo_root``) designated as *contracts* packages -- their
        ``src/`` trees may import only the stdlib, their own namespace,
        and :attr:`contract_extra_allowed` prefixes
    :ivar contract_extra_allowed: additional import prefixes contracts
        packages may use (e.g. ``("pydantic",)`` for validated DTOs)
    """

    repo_root: Path
    package_globs: tuple[str, ...] = ("packages/*", "packages/agent/*")
    exemptions_path: Path | None = None
    mode_env_var: str = "DEPENDENCY_ALIGNMENT_ENFORCEMENT_MODE"
    contract_packages: tuple[str, ...] = ()
    contract_extra_allowed: tuple[str, ...] = ()
