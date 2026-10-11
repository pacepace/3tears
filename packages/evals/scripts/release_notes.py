"""Print the release notes for one version of ``3tears-evals``, from ``MIGRATIONS.toml``.

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is not installed.

    # The notes for a tag (the leading "v" is optional):
    uv run python packages/evals/scripts/release_notes.py v0.66.0-evaladopt.6

    # The release being prepared:
    uv run python packages/evals/scripts/release_notes.py unreleased

The notes are the old-to-new map a porter needs: every name that left a public root in that release, each with
its new home or ``removed``, grouped under why it moved, then the section's notes on changes that are not a name.
They are printed as Markdown for the release's description. Nothing is written here but headings; every line comes
from ``MIGRATIONS.toml``, which ``tests/test_api_surface.py`` holds to the package.
"""

from __future__ import annotations

import argparse
import sys
import typing
from collections.abc import Iterator

from public_surface import MIGRATIONS_PATH, REMOVED, Release, load_migrations


def render(release: Release) -> Iterator[str]:
    """The notes for one release, line by line."""
    yield f"## 3tears-evals {release.version}: names that moved or were removed"
    yield ""
    if release.summary:
        yield " ".join(release.summary.split())
        yield ""
    if not release.changes:
        yield "No public name moved or was removed."
        yield ""
    for change in release.changes:
        target = "removed" if change.to == REMOVED else f"`{change.to}`"
        yield f"### `{change.source}` → {target}"
        yield ""
        yield " ".join(change.why.split())
        yield ""
        for migration in change.migrations():
            new = "removed" if migration.new == REMOVED else f"`{migration.new}`"
            yield f"- `{migration.old}` → {new}"
        yield ""
    if release.notes:
        yield "### Other breaking changes"
        yield ""
        for note in release.notes:
            yield f"- {' '.join(note.split())}"
        yield ""


def main(argv: list[str] | None = None) -> int:
    """Print one release's notes, or name the versions there are."""
    parser = argparse.ArgumentParser(description=typing.cast(str, __doc__).splitlines()[0])
    parser.add_argument("version", help="the version or tag, e.g. v0.66.0-evaladopt.6, or 'unreleased'")
    args = parser.parse_args(argv)
    migrations = load_migrations(MIGRATIONS_PATH)
    release = migrations.release(args.version)
    if release is None:
        known = ", ".join(r.version for r in migrations.releases)
        print(f"MIGRATIONS.toml has no section for {args.version!r}; it has: {known}", file=sys.stderr)
        return 1
    print("\n".join(render(release)).rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
