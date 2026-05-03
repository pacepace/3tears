"""path-dep src-root discovery — walk pyproject.toml files transitively.

walkers that need the full inheritance chain (cache primitive
detection, codebase-conventions cross-repo audits) cannot rely solely
on a repo's own src/ trees: classes inherit from canonical bases
declared in sibling packages installed via path-dep. this module
discovers every transitively-reachable path-dep target's src/ tree by
parsing pyproject.toml files for the three shapes actually used in the
3tears ecosystem:

- **poetry** path deps under ``[tool.poetry.dependencies]`` and
  ``[tool.poetry.group.*.dependencies]`` (metallm uses this).
- **uv workspace** members declared via ``[tool.uv.workspace] members =
  [...]`` glob patterns (3tears workspace root uses this).
- **uv sources** under ``[tool.uv.sources]`` resolving either to a
  workspace member (``{workspace = true}``) or to an explicit
  ``{path = "..."}`` (the bot trio uses this).

raising :class:`PyprojectError` on unsupported shapes is intentional —
the alternative is silently skipping a real path dep, which would let
the consumer drift out of sync with what the walker enforces.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Any

from threetears.enforcement.common.repo_layout import find_local_src_roots

__all__ = [
    "PyprojectError",
    "discover_src_roots",
]


class PyprojectError(Exception):
    """raised when a ``pyproject.toml`` cannot be parsed or has unsupported shape."""


def discover_src_roots(repo_root: Path) -> tuple[Path, ...]:
    """return every src root reachable from ``repo_root`` via path-deps.

    starts with :func:`find_local_src_roots(repo_root) <threetears.enforcement.common.repo_layout.find_local_src_roots>`,
    then transitively follows path-deps to every reachable target,
    accumulating each target's local src roots. cycle-safe (visited set
    keyed by resolved absolute path). returns the deduplicated union
    sorted by path string for stable report output.

    :param repo_root: absolute repo root path containing ``pyproject.toml``
    :ptype repo_root: Path
    :return: sorted tuple of absolute src-root paths
    :rtype: tuple[Path, ...]
    :raises PyprojectError: a ``pyproject.toml`` could not be parsed,
        had a shape this discoverer does not recognise, or referenced
        a path-dep target whose own ``pyproject.toml`` did not exist
    """
    if sys.version_info < (3, 11):
        raise PyprojectError("discover_src_roots requires Python 3.11+ (tomllib stdlib)")
    visited: set[Path] = set()
    roots: set[Path] = set()
    _visit(repo_root.resolve(), visited, roots)
    return tuple(sorted(roots))


def _visit(target: Path, visited: set[Path], roots: set[Path]) -> None:
    """recurse into ``target`` accumulating src roots into ``roots``.

    :param target: directory whose ``pyproject.toml`` to read
    :ptype target: Path
    :param visited: resolved absolute paths already processed
    :ptype visited: set[Path]
    :param roots: accumulator of discovered src-root paths
    :ptype roots: set[Path]
    :raises PyprojectError: see :func:`discover_src_roots`
    """
    if target in visited:
        return
    visited.add(target)

    if not target.is_dir():
        raise PyprojectError(f"path-dep target is not a directory: {target}")
    pyproject = target / "pyproject.toml"
    if not pyproject.is_file():
        raise PyprojectError(f"no pyproject.toml in {target}")

    for src_root in find_local_src_roots(target):
        roots.add(src_root)

    data = _load_toml(pyproject)

    for dep_path in _collect_path_deps(target, data, pyproject):
        _visit(dep_path.resolve(), visited, roots)


def _load_toml(path: Path) -> dict[str, Any]:
    """read ``path`` as TOML; wrap parse errors in :class:`PyprojectError`.

    :param path: path to ``pyproject.toml``
    :ptype path: Path
    :return: parsed TOML mapping
    :rtype: dict[str, Any]
    :raises PyprojectError: file cannot be read or is not valid TOML
    """
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except OSError as exc:
        raise PyprojectError(f"cannot read {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise PyprojectError(f"invalid TOML in {path}: {exc}") from exc


def _collect_path_deps(
    target: Path,
    data: dict[str, Any],
    pyproject_path: Path,
) -> list[Path]:
    """resolve every path-dep target referenced by ``data`` to an absolute path.

    handles the three shapes documented in the module docstring; each
    helper returns a ``list[Path]`` of resolved targets and raises
    :class:`PyprojectError` if a recognised top-level table holds an
    unrecognised value type.

    :param target: directory the pyproject lives in (anchor for
        relative paths)
    :ptype target: Path
    :param data: parsed pyproject TOML
    :ptype data: dict[str, Any]
    :param pyproject_path: path to the pyproject for error context
    :ptype pyproject_path: Path
    :return: list of resolved path-dep target directories
    :rtype: list[Path]
    :raises PyprojectError: an unsupported shape was encountered
    """
    deps: list[Path] = []
    workspace_members = _resolve_uv_workspace_members(target, data, pyproject_path)
    deps.extend(workspace_members)

    deps.extend(_resolve_poetry_path_deps(target, data, pyproject_path))
    deps.extend(_resolve_uv_sources(target, data, pyproject_path, workspace_members))
    return deps


def _resolve_uv_workspace_members(
    target: Path,
    data: dict[str, Any],
    pyproject_path: Path,
) -> list[Path]:
    """expand ``[tool.uv.workspace] members = [...]`` globs to absolute paths.

    each entry is either a literal directory name or a glob (``packages/*``).
    relative entries are anchored to ``target``. results are sorted for
    determinism.

    :param target: directory the pyproject lives in
    :ptype target: Path
    :param data: parsed pyproject TOML
    :ptype data: dict[str, Any]
    :param pyproject_path: path to the pyproject for error context
    :ptype pyproject_path: Path
    :return: sorted list of resolved workspace-member directories
    :rtype: list[Path]
    :raises PyprojectError: ``members`` is present but not a list of strings
    """
    workspace = data.get("tool", {}).get("uv", {}).get("workspace")
    if workspace is None:
        return []
    if not isinstance(workspace, dict):
        raise PyprojectError(f"{pyproject_path}: [tool.uv.workspace] must be a table, got {type(workspace).__name__}")
    members = workspace.get("members")
    if members is None:
        return []
    if not isinstance(members, list):
        raise PyprojectError(f"{pyproject_path}: [tool.uv.workspace].members must be a list of strings")
    resolved: list[Path] = []
    seen: set[Path] = set()
    for entry in members:
        if not isinstance(entry, str):
            raise PyprojectError(
                f"{pyproject_path}: [tool.uv.workspace].members entries must be strings, got {entry!r}"
            )
        for match in sorted(target.glob(entry)):
            if not match.is_dir():
                continue
            resolved_path = match.resolve()
            if resolved_path in seen:
                continue
            seen.add(resolved_path)
            resolved.append(resolved_path)
    return resolved


def _resolve_poetry_path_deps(
    target: Path,
    data: dict[str, Any],
    pyproject_path: Path,
) -> list[Path]:
    """resolve every ``{path = "..."}`` dep under ``[tool.poetry.*]``.

    walks ``[tool.poetry.dependencies]`` and every
    ``[tool.poetry.group.<name>.dependencies]`` table; each entry whose
    value is a table containing a ``path`` key is treated as a path
    dep. version-string deps and url/git deps are skipped.

    :param target: directory the pyproject lives in
    :ptype target: Path
    :param data: parsed pyproject TOML
    :ptype data: dict[str, Any]
    :param pyproject_path: path to the pyproject for error context
    :ptype pyproject_path: Path
    :return: list of resolved poetry path-dep targets in declaration order
    :rtype: list[Path]
    :raises PyprojectError: an entry has a malformed ``path`` value
    """
    poetry = data.get("tool", {}).get("poetry")
    if poetry is None:
        return []
    if not isinstance(poetry, dict):
        raise PyprojectError(f"{pyproject_path}: [tool.poetry] must be a table, got {type(poetry).__name__}")
    resolved: list[Path] = []
    deps_tables: list[tuple[str, Any]] = []
    if "dependencies" in poetry:
        deps_tables.append(("dependencies", poetry["dependencies"]))
    groups = poetry.get("group")
    if groups is not None:
        if not isinstance(groups, dict):
            raise PyprojectError(f"{pyproject_path}: [tool.poetry.group] must be a table")
        for group_name, group_data in groups.items():
            if not isinstance(group_data, dict):
                raise PyprojectError(f"{pyproject_path}: [tool.poetry.group.{group_name}] must be a table")
            sub_deps = group_data.get("dependencies")
            if sub_deps is not None:
                deps_tables.append((f"group.{group_name}.dependencies", sub_deps))
    for label, deps in deps_tables:
        if not isinstance(deps, dict):
            raise PyprojectError(f"{pyproject_path}: [tool.poetry.{label}] must be a table")
        for name, value in deps.items():
            if not isinstance(value, dict):
                continue
            path_value = value.get("path")
            if path_value is None:
                continue
            if not isinstance(path_value, str):
                raise PyprojectError(f"{pyproject_path}: [tool.poetry.{label}].{name}.path must be a string")
            candidate = (target / path_value).resolve()
            resolved.append(candidate)
    return resolved


def _resolve_uv_sources(
    target: Path,
    data: dict[str, Any],
    pyproject_path: Path,
    workspace_members: list[Path],
) -> list[Path]:
    """resolve every entry under ``[tool.uv.sources]`` to an absolute path.

    supports two shapes per entry:

    - ``{path = "..."}``: anchored to ``target``.
    - ``{workspace = true}``: looks up the dep name against the
      enclosing workspace's members. resolution strategy: search local
      ``workspace_members`` (when ``target`` itself declares the
      workspace) for a member whose ``[project] name`` (or whose
      directory basename, as a fallback) matches the dep name. when
      ``target`` does not declare a workspace, walk upward from
      ``target`` until a pyproject with ``[tool.uv.workspace]`` is
      found, expand its members, and search there. this matches uv's
      own resolution semantics: a workspace-true source on a member
      points to a sibling member of the enclosing workspace.

    other shapes (``{git = "..."}``, ``{index = "..."}``) are treated
    as non-path-dep and skipped.

    :param target: directory the pyproject lives in
    :ptype target: Path
    :param data: parsed pyproject TOML
    :ptype data: dict[str, Any]
    :param pyproject_path: path to the pyproject for error context
    :ptype pyproject_path: Path
    :param workspace_members: resolved workspace member directories from
        ``target``'s own pyproject (empty when ``target`` is a workspace
        member rather than the root)
    :ptype workspace_members: list[Path]
    :return: list of resolved uv-source path-dep targets
    :rtype: list[Path]
    :raises PyprojectError: entry has a malformed shape or
        ``workspace = true`` references a name that cannot be resolved
        to a workspace member
    """
    sources = data.get("tool", {}).get("uv", {}).get("sources")
    if sources is None:
        return []
    if not isinstance(sources, dict):
        raise PyprojectError(f"{pyproject_path}: [tool.uv.sources] must be a table")
    resolved: list[Path] = []
    enclosing_members: list[Path] | None = None
    for name, value in sources.items():
        if not isinstance(value, dict):
            raise PyprojectError(
                f"{pyproject_path}: [tool.uv.sources].{name} must be a table, got {type(value).__name__}"
            )
        path_value = value.get("path")
        workspace_flag = value.get("workspace")
        if path_value is not None:
            if not isinstance(path_value, str):
                raise PyprojectError(f"{pyproject_path}: [tool.uv.sources].{name}.path must be a string")
            resolved.append((target / path_value).resolve())
            continue
        if workspace_flag is True:
            members = workspace_members
            if not members:
                if enclosing_members is None:
                    enclosing_members = _find_enclosing_workspace_members(target)
                members = enclosing_members
            member = _match_workspace_member(name, members)
            if member is None:
                raise PyprojectError(
                    f"{pyproject_path}: [tool.uv.sources].{name} declares workspace=true "
                    f"but no enclosing [tool.uv.workspace].members entry resolves to a "
                    f"package named {name!r}"
                )
            resolved.append(member)
            continue
        if "git" in value or "url" in value or "index" in value:
            continue
    return resolved


def _find_enclosing_workspace_members(start: Path) -> list[Path]:
    """walk upward from ``start`` to the first pyproject with a uv workspace.

    returns the resolved member directories declared by that workspace,
    or an empty list if none is found before the filesystem root.

    :param start: directory to begin the upward walk
    :ptype start: Path
    :return: workspace member directories
    :rtype: list[Path]
    :raises PyprojectError: an enclosing pyproject has malformed
        workspace shape
    """
    current = start.resolve()
    for candidate in [*current.parents]:
        pyproject = candidate / "pyproject.toml"
        if not pyproject.is_file():
            continue
        data = _load_toml(pyproject)
        members = _resolve_uv_workspace_members(candidate, data, pyproject)
        if members:
            return members
    return []


def _match_workspace_member(name: str, members: list[Path]) -> Path | None:
    """find the member directory whose package name matches ``name``.

    matching strategy (tried in order):

    1. read each member's ``pyproject.toml`` ``[project] name`` and
       compare exactly.
    2. fall back to the directory basename match — used by older or
       hatchling-only pyprojects that don't declare ``[project] name``.

    :param name: dep name to look up
    :ptype name: str
    :param members: candidate workspace member directories
    :ptype members: list[Path]
    :return: matching member directory, or None
    :rtype: Path | None
    """
    for member in members:
        pyproject = member / "pyproject.toml"
        if not pyproject.is_file():
            continue
        data = _load_toml(pyproject)
        project = data.get("project")
        if isinstance(project, dict):
            project_name = project.get("name")
            if isinstance(project_name, str) and project_name == name:
                return member
    for member in members:
        if member.name == name:
            return member
    return None
