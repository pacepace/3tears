"""per-repo configuration for the declared-imports gate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ImportsDeclaredConfig"]


@dataclass(frozen=True)
class ImportsDeclaredConfig:
    """knobs the consuming repo's thin test shell injects.

    :ivar repo_root: the repo root, holding ``pyproject.toml``
    :ivar first_party: import roots that are this repo's own packages; an import of one is
        never a dependency (``{"aibots"}``, ``{"aibots_agents"}``)
    :ivar source_roots: directories, relative to ``repo_root``, whose ``*.py`` files are
        walked for imports. Name the shipped package only: a tree that also holds tests
        (``src/tests``) would hold test-only imports to the runtime rule
    :ivar required_import_roots: top-level modules the walk MUST find, the import walk's
        non-vacuity guard. A filter that started excluding what it should govern would
        otherwise report zero offenders and pass. Name two the source imports throughout
    :ivar required_owned_modules: modules the installed-distribution owner map MUST know,
        the other input's non-vacuity guard. An editable install lists no module files, so
        an owner map that stops reading checkouts empties silently -- and an empty map has
        no offenders to report
    :ivar pyproject: the manifest whose declarations are read; ``repo_root /
        "pyproject.toml"`` when ``None``
    :ivar undeclarable_import_roots: top-level import roots the repo imports but must NOT
        declare, each mapped to why -- the SDK lazily importing the hub, which hard-depends
        on the SDK, so declaring it back would be a cycle. Never a convenience: a rationale
        is required, and an entry no import uses is itself a finding, so the list cannot
        outlive its reason unnoticed
    :raises ValueError: when an undeclarable root carries no rationale
    """

    repo_root: Path
    first_party: frozenset[str]
    source_roots: tuple[str, ...] = ("src",)
    required_import_roots: frozenset[str] = field(default_factory=frozenset)
    required_owned_modules: frozenset[str] = frozenset({"threetears"})
    pyproject: Path | None = None
    undeclarable_import_roots: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """refuse an undeclarable root with no stated reason.

        :return: nothing
        :rtype: None
        :raises ValueError: when a rationale is empty
        """
        unexplained = sorted(root for root, why in self.undeclarable_import_roots.items() if not why.strip())
        if unexplained:
            raise ValueError(
                f"undeclarable_import_roots {unexplained} carry no rationale; say why each is undeclarable"
            )

    @property
    def manifest(self) -> Path:
        """the manifest whose declarations are read.

        :return: :attr:`pyproject`, or ``repo_root / "pyproject.toml"``
        :rtype: Path
        """
        return self.pyproject if self.pyproject is not None else self.repo_root / "pyproject.toml"
