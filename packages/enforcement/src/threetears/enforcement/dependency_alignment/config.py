"""per-repo configuration for dependency-alignment enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["DependencyAlignmentConfig", "DependencyFloor"]


@dataclass(frozen=True)
class DependencyFloor:
    """an exact permitted hard-dependency set for one package.

    the sibling of :attr:`DependencyAlignmentConfig.contract_packages`
    for packages that are *not* dependency-free but whose floor was
    argued and ruled. a contracts package's rule is "nothing"; a floor's
    rule is "these and no others", which is the same guarantee stated
    for a package that needs a short list to exist at all.

    the pin is on ``[project] dependencies`` only. optional extras are
    weight a host opts into and are deliberately not pinned here -- the
    floor is what every consumer pays unconditionally, and that is the
    number a constrained host (a Pi, a container with a ``MemoryMax``)
    budgets against.

    :ivar package: package directory relative to the repo root
        (``packages/search``)
    :ivar allowed: the exact distribution names ``[project]
        dependencies`` may contain -- a superset and a subset are both
        violations, because a floor that drifts either way stops being
        the thing that was ruled
    :ivar rationale: why this list and not another, naming the ruling it
        implements. required: an unexplained floor cannot be reviewed
    """

    package: str
    allowed: tuple[str, ...]
    rationale: str


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
    :ivar dependency_floors: packages whose hard-dependency list is
        pinned to an exact ruled set -- see :class:`DependencyFloor`
    """

    repo_root: Path
    package_globs: tuple[str, ...] = ("packages/*", "packages/agent/*")
    exemptions_path: Path | None = None
    mode_env_var: str = "DEPENDENCY_ALIGNMENT_ENFORCEMENT_MODE"
    contract_packages: tuple[str, ...] = ()
    contract_extra_allowed: tuple[str, ...] = ()
    dependency_floors: tuple[DependencyFloor, ...] = ()
