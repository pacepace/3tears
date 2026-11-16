"""every place a version string lives, rewritten and verified as one.

Each rewrite is pure text in, text out, and touches only the lines that carry
the version: the ``[project] version`` line, ``__version__ == "X.Y.Z"`` (and its
aliased ``core_version == "X.Y.Z"`` shape, which slipped past 3tears v0.10.2's
first bump), a bake file's ``VERSION`` default and ``:vX.Y.Z`` image refs, and the
intra-family ``>=X.Y.0,<X.Y+1.0`` bounds -- a family at 0.20.0 still requiring
siblings ``<0.20.0`` cannot install itself. The family's npm packages (``@<prefix>/*``) carry the
same version: each ``package.json``'s own ``version``, its dependencies on siblings, and a
``package-lock.json``'s entries for the workspace's own packages.
"""

from __future__ import annotations

import json
import re

from threetears.enforcement.release.config import ReleaseConfig
from threetears.enforcement.release.versions import parse_version, project_version

__all__ = ["lockstep_mismatches", "rewrite_lockstep_file", "write_project_version"]

#: a smoke assertion's version, with any PEP 440 local label (``"0.80.0+analyst.1"``) read as part of it, so
#: ``verify`` names a labelled assertion against a plain release and a bump rewrites it.
_SMOKE = re.compile(r'(__version__|[A-Za-z_]+_version) == "(\d+\.\d+\.\d+(?:\+[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)?)"')
_BAKE_DEFAULT = re.compile(r'^(\s*default = )"v(\d+\.\d+\.\d+)"$', re.MULTILINE)
_BAKE_REF = re.compile(r":v(\d+\.\d+\.\d+)")
#: a TOML table header: a LINE that is a header, never the header's text quoted in prose.
_TABLE_HEADER = re.compile(r"^\s*\[\[?[^\[\]=\"']+\]\]?\s*(#.*)?$")


def _bound(prefix: str) -> re.Pattern[str]:
    """matches one intra-family bound, ``"<prefix>-x[extra]>=A.B.C,<D.E.F"``.

    :param prefix: family package prefix
    :ptype prefix: str
    :return: pattern capturing the requirement name and both bounds
    :rtype: re.Pattern[str]
    """
    return re.compile(
        r'("' + re.escape(prefix) + r'[a-z0-9-]*(?:\[[a-z0-9,_-]+\])?)>=(\d+\.\d+\.\d+),<(\d+\.\d+\.\d+)"'
    )


def _line_bounds(version: str) -> tuple[str, str]:
    """the floor and ceiling of *version*'s minor line.

    :param version: ``X.Y.Z``
    :ptype version: str
    :return: ``(X.Y.0, X.Y+1.0)``
    :rtype: tuple[str, str]
    """
    major, minor, _ = parse_version(version)
    return f"{major}.{minor}.0", f"{major}.{minor + 1}.0"


def write_project_version(text: str, version: str) -> str:
    """rewrites the first ``version = "..."`` line inside ``[project]``; nothing else changes.

    :param text: ``pyproject.toml`` contents
    :ptype text: str
    :param version: ``X.Y.Z`` to write
    :ptype version: str
    :return: the rewritten contents
    :rtype: str
    :raises ValueError: if ``[project]`` declares no version line, or the result
        does not read back as *version*
    """
    out: list[str] = []
    in_project = False
    done = False
    for line in text.split("\n"):
        if _TABLE_HEADER.match(line):
            in_project = line.split("#", 1)[0].strip() == "[project]"
        elif in_project and not done and re.match(r"^version\s*=", line):
            line = f'version = "{version}"'
            done = True
        out.append(line)
    result = "\n".join(out)
    if not done or project_version(result) != parse_version(version):
        raise ValueError(f"could not write version {version} into [project]")
    return result


#: the dependency tables of a package.json whose sibling entries follow the version
_NPM_DEPENDENCY_TABLES = ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies")


def _npm_versions(path: str, data: dict[str, object], scope: str | None) -> list[tuple[str, str, dict[str, object]]]:
    """every version an npm file carries for the family: ``(where, key, the dict holding it)``.

    A ``package.json`` carries its own ``version`` and, under each dependency table, the exact
    versions of sibling ``@<scope>/*`` packages. A ``package-lock.json`` carries, in ``packages``,
    the entry of each workspace package (a key outside ``node_modules``), and in each entry's
    dependency tables the siblings again. A lock's root entry (``""``) has no version.

    :param path: repo-relative path, to tell a lock from a manifest
    :ptype path: str
    :param data: the parsed file
    :ptype data: dict[str, object]
    :param scope: the npm scope (``@3tears/``), or ``None`` for no sibling entries
    :ptype scope: str | None
    :return: the places, in file order
    :rtype: list[tuple[str, str, dict[str, object]]]
    """

    def manifest(where: str, entry: dict[str, object]) -> list[tuple[str, str, dict[str, object]]]:
        found: list[tuple[str, str, dict[str, object]]] = []
        if "version" in entry:
            found.append((where, "version", entry))
        for table in _NPM_DEPENDENCY_TABLES:
            deps = entry.get(table)
            if scope and isinstance(deps, dict):
                found.extend((f"{where} {table}", name, deps) for name in deps if name.startswith(scope))
        return found

    if not path.endswith("package-lock.json"):
        return manifest(path, data)
    places: list[tuple[str, str, dict[str, object]]] = []
    packages = data.get("packages")
    if isinstance(packages, dict):
        for key, entry in packages.items():
            if key and "node_modules" not in key and isinstance(entry, dict):
                places.extend(manifest(f"{path} {key}", entry))
    return places


def _npm_scope(config: ReleaseConfig) -> str | None:
    return f"@{config.family_prefix}/" if config.family_prefix else None


def _rewrite_npm_file(config: ReleaseConfig, path: str, text: str, version: str) -> str:
    """brings one npm ``package.json`` or ``package-lock.json`` to *version*.

    Written back as npm writes it: two-space indent, a trailing newline.

    :param config: the repository's release configuration
    :ptype config: ReleaseConfig
    :param path: repo-relative path of the file
    :ptype path: str
    :param text: its contents
    :ptype text: str
    :param version: ``X.Y.Z``
    :ptype version: str
    :return: the rewritten contents
    :rtype: str
    """
    data = json.loads(text)
    for _where, key, holder in _npm_versions(path, data, _npm_scope(config)):
        holder[key] = version
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def rewrite_lockstep_file(config: ReleaseConfig, path: str, text: str, version: str) -> str:
    """brings one lockstep file to *version*.

    :param config: the repository's release configuration
    :ptype config: ReleaseConfig
    :param path: repo-relative path of the file
    :ptype path: str
    :param text: its contents
    :ptype text: str
    :param version: ``X.Y.Z``
    :ptype version: str
    :return: the rewritten contents
    :rtype: str
    """
    if path in config.npm_files:
        return _rewrite_npm_file(config, path, text, version)
    result = text
    if path in config.version_files:
        result = write_project_version(result, version)
        if config.family_prefix:
            floor, ceiling = _line_bounds(version)
            result = _bound(config.family_prefix).sub(rf'\g<1>>={floor},<{ceiling}"', result)
    if path in config.smoke_tests:
        result = _SMOKE.sub(rf'\g<1> == "{version}"', result)
    if path == config.bake_file:
        result = _BAKE_DEFAULT.sub(rf'\g<1>"v{version}"', result)
        result = _BAKE_REF.sub(f":v{version}", result)
    return result


def lockstep_mismatches(config: ReleaseConfig, version: str) -> list[str]:
    """every place a version string does not say *version*.

    :param config: the repository's release configuration
    :ptype config: ReleaseConfig
    :param version: ``X.Y.Z`` everything should carry
    :ptype version: str
    :return: one line per mismatch; empty when everything agrees
    :rtype: list[str]
    """
    mismatches: list[str] = []
    floor, ceiling = _line_bounds(version)
    for path in config.version_files:
        text = (config.root / path).read_text(encoding="utf-8")
        try:
            declared = ".".join(map(str, project_version(text)))
        except ValueError as exc:
            declared = f"<{exc}>"
        if declared != version:
            mismatches.append(f"{path}: version {declared}, expected {version}")
        if config.family_prefix:
            for matched in _bound(config.family_prefix).finditer(text):
                if (matched[2], matched[3]) != (floor, ceiling):
                    mismatches.append(f"{path}: {matched[0]} (expected >={floor},<{ceiling})")
    for path in config.smoke_tests:
        for matched in _SMOKE.finditer((config.root / path).read_text(encoding="utf-8")):
            if matched[2] != version:
                mismatches.append(f"{path}: {matched[0]} (expected {version})")
    for path in config.npm_files:
        data = json.loads((config.root / path).read_text(encoding="utf-8"))
        mismatches.extend(
            f"{where}: {key} {holder[key]}, expected {version}"
            for where, key, holder in _npm_versions(path, data, _npm_scope(config))
            if holder[key] != version
        )
    if config.bake_file:
        text = (config.root / config.bake_file).read_text(encoding="utf-8")
        defaults = [matched[2] for matched in _BAKE_DEFAULT.finditer(text)]
        if defaults[:1] != [version]:
            mismatches.append(
                f"{config.bake_file} VERSION: {defaults[0] if defaults else '<missing>'}, expected {version}"
            )
        mismatches.extend(
            f"{config.bake_file} image-tag :v{matched[1]} (expected :v{version})"
            for matched in _BAKE_REF.finditer(text)
            if matched[1] != version
        )
    return mismatches
