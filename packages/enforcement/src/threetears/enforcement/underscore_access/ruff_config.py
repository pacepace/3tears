"""one definition of "which files carry a per-file ``SLF001`` ignore, and from which config".

the underscore-access ledger records why each private access on an exempted path was judged
acceptable, so every consumer of that ledger has to agree on which paths ARE exempted. two
callers each answered it their own way and disagreed in the direction that loses information:
one walked every ruff config in the tree, the other read only the root ``pyproject.toml``. a
nested ``ruff.toml`` is a FULL override -- a root key for a path beneath it is never read -- so
anything reading only the root is blind to whatever that subtree exempts. that blindness
deleted a set of reviewed ledger entries once, and no check caught it, because a checker built
on the root config cannot miss what it cannot see.

ruff reads per-file ignores from three places -- ``pyproject.toml`` under ``[tool.ruff]``,
``ruff.toml``, and ``.ruff.toml`` -- and resolves each key relative to the config declaring it,
matching a separator-free pattern against the basename anywhere beneath it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

__all__ = ["all_exempted_files", "exempted_files", "ruff_configs", "slf001_globs"]

#: directories whose contents are somebody else's code and not this repo's to enforce.
_VENDORED = {".venv", "node_modules", ".git", "__pycache__"}


def _is_vendored(path: Path, repo_root: Path) -> bool:
    """whether *path* sits under a vendored directory INSIDE the repo.

    relative to the root deliberately: testing an absolute path's parts means a checkout living
    under a directory named ``.venv`` or ``.git`` excludes the whole tree, and every caller then
    scans nothing while reporting success.
    """
    return any(part in _VENDORED for part in path.relative_to(repo_root).parts)


def ruff_configs(repo_root: Path) -> list[Path]:
    """every file ruff would read a per-file ignore from, under *repo_root*."""
    configs: list[Path] = []
    for name in ("pyproject.toml", "ruff.toml", ".ruff.toml"):
        for path in repo_root.rglob(name):
            if _is_vendored(path, repo_root):
                continue
            if path.name == "pyproject.toml" and "[tool.ruff" not in path.read_text(errors="replace"):
                continue  # a pyproject with no ruff section configures nothing
            configs.append(path)
    return sorted(configs)


def slf001_globs(config: Path) -> list[str]:
    """the ``per-file-ignores`` keys in *config* whose code list includes ``SLF001``."""
    data = tomllib.loads(config.read_text())
    section = (
        data.get("tool", {}).get("ruff", {}).get("lint", {})
        if config.name == "pyproject.toml"
        else data.get("lint", {})
    )
    return [key for key, codes in section.get("per-file-ignores", {}).items() if "SLF001" in codes]


def exempted_files(config: Path, pattern: str, repo_root: Path) -> list[Path]:
    """python files matched by *pattern*, following ruff's own two-way matching.

    a pattern with no separator matches the BASENAME anywhere beneath the config -- which is why
    ruff's documented ``"__init__.py"`` example covers a whole tree -- and one with a separator
    matches the relative path. comparing only the second way leaves a bare-name key covering
    whatever single file sits at the config's root and every other file of that name unscanned.
    """
    base = config.parent
    matched = base.glob(pattern) if "/" in pattern else base.rglob(pattern)
    return sorted(p for p in matched if p.suffix == ".py" and not _is_vendored(p, repo_root))


def all_exempted_files(repo_root: Path) -> list[Path]:
    """every file carrying a per-file ``SLF001`` ignore, from every config ruff reads.

    not filtered to ``/tests/``: production files carry per-file ignores too, and their accesses
    are precisely the kind the ledger exists to record.
    """
    found: list[Path] = []
    for config in ruff_configs(repo_root):
        for pattern in slf001_globs(config):
            found.extend(exempted_files(config, pattern, repo_root))
    return sorted(set(found))
