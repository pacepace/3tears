"""release tooling with one owner: the API-growth gate and the version bump.

Every 3tears and aibots repository versions by one rule -- minor for new public
API or a new feature, patch for everything else, major on the owner's word -- and
used to carry its own copy of the gate and of ``scripts/bump-version.sh`` to hold
it. Those copies drifted and were re-synced by hand. They live here now:

- :func:`api_growth_findings`, called from each repository's enforcement test
  with an :class:`ApiGrowthConfig` naming its source roots, first release,
  anchors and any extra surface extractors (:func:`http_routes`, or the repo's own);
- :func:`untagged_checkout_findings`, which binds every CI job that runs the
  gate to a checkout that can see the release tags;
- ``threetears-release`` (:func:`main`), which each repository's
  ``scripts/bump-version.sh`` wraps in one line.
"""

from __future__ import annotations

from threetears.enforcement.release.checkouts import untagged_checkout_findings
from threetears.enforcement.release.bump import BUMP_KINDS, Refusal, next_version, run_release
from threetears.enforcement.release.cli import main
from threetears.enforcement.release.config import ConfigError, ReleaseConfig, load_release_config
from threetears.enforcement.release.growth import ApiGrowthConfig, api_growth_findings
from threetears.enforcement.release.lockstep import lockstep_mismatches, rewrite_lockstep_file, write_project_version
from threetears.enforcement.release.notes import (
    KEEPACHANGELOG,
    THREETEARS,
    ChangelogStyle,
    NotesRefusal,
    release_changelog,
    release_prawduct_log,
)
from threetears.enforcement.release.surface import (
    BaselineUnreadableError,
    SurfaceExtractor,
    SurfaceReadError,
    TreeSurface,
    http_routes,
    is_private_module,
    module_surface,
    surface_at,
    surface_now,
)
from threetears.enforcement.release.versions import (
    Version,
    format_version,
    parse_version,
    project_version,
    release_tags,
)

__all__ = [
    "BUMP_KINDS",
    "KEEPACHANGELOG",
    "THREETEARS",
    "ApiGrowthConfig",
    "BaselineUnreadableError",
    "ChangelogStyle",
    "ConfigError",
    "NotesRefusal",
    "Refusal",
    "ReleaseConfig",
    "SurfaceExtractor",
    "SurfaceReadError",
    "TreeSurface",
    "Version",
    "api_growth_findings",
    "format_version",
    "http_routes",
    "is_private_module",
    "load_release_config",
    "lockstep_mismatches",
    "main",
    "module_surface",
    "next_version",
    "parse_version",
    "project_version",
    "release_changelog",
    "release_prawduct_log",
    "release_tags",
    "rewrite_lockstep_file",
    "run_release",
    "surface_at",
    "surface_now",
    "untagged_checkout_findings",
    "write_project_version",
]
