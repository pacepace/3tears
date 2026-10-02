"""the API-growth gate: a release whose public API grew may not be a patch.

**The rule** (3tears backlog BLD-7QM3, 2026-07-26; extended to every aibots repo
by owner ruling, 2026-09-30): a **minor** bump for any new public API or new
feature, a **patch** for everything else, a **major** only on the owner's word.
3tears v0.24.7 shipped eleven new public names as a patch with every gate green;
this is the guard that would have refused it.

**The comparison.** The public surface (:mod:`threetears.enforcement.release.surface`)
at the highest ``vX.Y.Z`` tag against the WORKING TREE. Growth while the declared
version is still on that tag's minor line is a finding. A minor or major bump
moves the line, so growth there is what the bump is for.

**The baseline is never optional.** With no tag visible, the gate passes in ONE
state only: the declared version IS the repo's first release, so nothing was ever
released to compare against. Past it, a missing tag means a release was never
tagged or the clone cannot see tags (a CI checkout needs ``fetch-depth: 0``), and
the gate refuses rather than passing. A version BELOW the latest tag is refused.
A tag whose tree or objects this clone lacks is refused, not read as empty.

**It is never vacuous.** Every run reads the working tree and requires each
configured anchor -- a name the repo is known to export -- so a reader that stops
matching fails here rather than comparing two empty sets. A module whose surface
cannot be read (see :class:`~threetears.enforcement.release.surface.SurfaceReadError`)
is a finding naming it, on either side of the comparison.

**What it cannot see**, recorded rather than papered over: a new field on an
existing model, a route added without a decorator, a NATS subject, and a method
added to a private class a public function returns. A repo whose API lives
somewhere else supplies an extractor for it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from threetears.enforcement.release.surface import (
    BaselineUnreadableError,
    SurfaceExtractor,
    TreeSurface,
    surface_at,
    surface_now,
)
from threetears.enforcement.release.versions import (
    Version,
    format_version,
    project_version,
    release_tags,
)

__all__ = ["ApiGrowthConfig", "api_growth_findings"]


@dataclass(frozen=True)
class ApiGrowthConfig:
    """one repository's API-growth gate.

    :param repo_root: repository root
    :ptype repo_root: Path
    :param source_roots: repo-relative source trees whose public surface is the API
    :ptype source_roots: tuple[str, ...]
    :param first_release: the version this repo first released; the only version at
        which no release tag is an acceptable state
    :ptype first_release: Version
    :param known_exports: ``(module path, name)`` anchors the working tree must
        export; at least one is required
    :ptype known_exports: tuple[tuple[str, str], ...]
    :param version_file: repo-relative ``pyproject.toml`` declaring the version
    :ptype version_file: str
    :param extractors: extra surface readers (routes, tool actions, ...)
    :ptype extractors: Sequence[SurfaceExtractor]
    :param remedy: the command a finding tells the developer to run
    :ptype remedy: str
    """

    repo_root: Path
    source_roots: tuple[str, ...]
    first_release: Version
    known_exports: tuple[tuple[str, str], ...]
    version_file: str = "pyproject.toml"
    extractors: Sequence[SurfaceExtractor] = field(default_factory=tuple)
    remedy: str = "./scripts/bump-version.sh minor"


def _working_tree(config: ApiGrowthConfig) -> tuple[dict[str, TreeSurface], list[str]]:
    """the working tree's surface per source root, and every finding about reading it.

    :param config: the gate's configuration
    :ptype config: ApiGrowthConfig
    :return: surfaces by source root, and findings
    :rtype: tuple[dict[str, TreeSurface], list[str]]
    """
    findings: list[str] = []
    surfaces: dict[str, TreeSurface] = {}
    for source_root in config.source_roots:
        tree = surface_now(config.repo_root, source_root, config.extractors)
        surfaces[source_root] = tree
        if not tree.present:
            findings.append(
                f"the source root {source_root} does not exist in the working tree; the gate's config is wrong."
            )
        findings.extend(f"cannot read the public surface of {error}" for error in tree.errors)
    if not config.known_exports:
        findings.append("the gate's config names no known export, so nothing proves the surface reader still matches.")
    for module_path, name in config.known_exports:
        found = any(name in tree.modules.get(module_path, set()) for tree in surfaces.values())
        if not found:
            findings.append(
                f"the surface reader no longer sees {name!r} in {module_path} -- it has stopped matching, so every "
                f"comparison would be between empty sets. Fix the reader (or this anchor) before trusting a green run."
            )
    return surfaces, findings


def _growth_since(config: ApiGrowthConfig, tag_name: str, now: dict[str, TreeSurface], shown: str) -> list[str]:
    """findings for public surface added since *tag_name*, for a version on the tag's minor line.

    A source root absent at the tag is wholly new, so all of its surface is growth.

    :param config: the gate's configuration
    :ptype config: ApiGrowthConfig
    :param tag_name: the baseline release tag
    :ptype tag_name: str
    :param now: the working tree's surfaces by source root
    :ptype now: dict[str, TreeSurface]
    :param shown: the declared version, for the message
    :ptype shown: str
    :return: findings; empty when nothing grew
    :rtype: list[str]
    """
    findings: list[str] = []
    grown: list[str] = []
    for source_root in config.source_roots:
        try:
            before = surface_at(config.repo_root, tag_name, source_root, config.extractors)
        except BaselineUnreadableError as exc:
            findings.append(
                f"{source_root} could not be read at {tag_name}: {exc}. Fetch the full history (`git fetch --tags "
                f"--unshallow`; CI needs fetch-depth: 0). Nothing was compared, so this is a refusal, not a pass."
            )
            continue
        findings.extend(
            f"cannot read the public surface at {tag_name} of {error}. That module cannot be compared on this "
            f"minor line; a minor bump moves the baseline past it."
            for error in before.errors
        )
        for module_path, names in sorted(now[source_root].modules.items()):
            added = names - before.modules.get(module_path, set())
            if added:
                grown.append(f"  {module_path}: {', '.join(sorted(added))}")
    if grown:
        findings.append(
            f"the public API grew since {tag_name}, but the version {shown} is on the same minor line. New public "
            f"API or a new feature is a MINOR release: run `{config.remedy}`.\n" + "\n".join(grown)
        )
    return findings


def api_growth_findings(config: ApiGrowthConfig) -> list[str]:
    """everything wrong with a repository's declared version against its public surface.

    :param config: the gate's configuration
    :ptype config: ApiGrowthConfig
    :return: findings; empty when the version is honest about the surface
    :rtype: list[str]
    """
    now, findings = _working_tree(config)
    current = project_version((config.repo_root / config.version_file).read_text(encoding="utf-8"))
    shown = format_version(current)
    tags = release_tags(config.repo_root)
    if not tags and current != config.first_release:
        findings.append(
            f"no vX.Y.Z release tag is visible, but the version {shown} is past the first release "
            f"{format_version(config.first_release)}. Either a release was never tagged (tag its main merge "
            f"commit) or this clone cannot see tags (`git fetch --tags`; a CI checkout needs fetch-depth: 0). "
            f"Without a baseline this gate cannot tell a patch from a minor, so it refuses rather than passing."
        )
    elif tags:
        tag_name, tag_version = max(tags.items(), key=lambda item: item[1])
        if current < tag_version:
            findings.append(f"the version {shown} is BELOW the latest release {tag_name}; versions only move forward.")
        elif current[:2] == tag_version[:2]:
            findings.extend(_growth_since(config, tag_name, now, shown))
    return findings
