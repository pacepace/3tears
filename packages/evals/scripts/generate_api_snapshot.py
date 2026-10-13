"""Generate ``packages/evals/api/surface.json``, the checked-in snapshot of the package's public surface.

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is not installed.

    # Rewrite the snapshot (refused while a name has left a root unrecorded):
    uv run python packages/evals/scripts/generate_api_snapshot.py

    # Exit 1 when the snapshot or its record disagrees with the package, without writing:
    uv run python packages/evals/scripts/generate_api_snapshot.py --check

The snapshot is every public root's ``__all__`` (the roots are ``threetears.evals.PUBLIC_ROOTS``), one entry per
``root.name`` with its kind and signature, read by the same walk that renders ``docs/reference.md``
(``public_surface.py``). ``tests/test_api_surface.py`` holds the installed package to it, so a change to the
surface is a diff in review rather than a surprise in an adopter's import.

**What regenerating records, and what it refuses to.** A name added, or a kind or signature changed, is recorded
by regenerating: the diff is the record. A name that left a root (moved, renamed or removed) is not, when the
release before shipped it: it must first be written down in ``MIGRATIONS.toml``'s newest section (its new home or
``removed``, and why), which is what the release notes are printed from (``release_notes.py``), and until it is
this script writes nothing. A name no release shipped leaves with a regeneration alone. A name exported from two
roots must be dropped from all but its home, or declared in ``api/reexports.toml``.
"""

from __future__ import annotations

import argparse
import sys
import typing
from pathlib import Path

from public_surface import (
    MIGRATIONS_PATH,
    REEXPORTS_PATH,
    SNAPSHOT_PATH,
    Finding,
    SurfaceEntry,
    all_findings,
    installed_surface,
    load_migrations,
    load_reexports,
    load_snapshot,
    render_snapshot,
)


def write_snapshot(
    installed: dict[str, SurfaceEntry],
    findings: list[Finding],
    path: Path = SNAPSHOT_PATH,
) -> bool:
    """Write ``installed`` as the snapshot unless a finding needs more than a regeneration.

    Args:
        installed: The surface to record.
        findings: Every finding over it (:func:`public_surface.all_findings`).
        path: Where the snapshot is written.

    Returns:
        Whether it was written: ``False`` while any finding is one regenerating does not fix.
    """
    if any(not f.regenerating_fixes for f in findings):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_snapshot(installed), encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    """Write the snapshot, or check it."""
    parser = argparse.ArgumentParser(description=typing.cast(str, __doc__).splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 on any finding; write nothing")
    args = parser.parse_args(argv)
    installed = installed_surface()
    findings = all_findings(
        installed, load_snapshot(), load_migrations(MIGRATIONS_PATH), load_reexports(REEXPORTS_PATH)
    )
    if args.check:
        for finding in findings:
            print(finding.problem, file=sys.stderr)
        return 1 if findings else 0
    if not write_snapshot(installed, findings):
        for finding in findings:
            if not finding.regenerating_fixes:
                print(finding.problem, file=sys.stderr)
        print(f"{SNAPSHOT_PATH} not written: resolve the findings above first.", file=sys.stderr)
        return 1
    print(f"wrote {SNAPSHOT_PATH} ({len(installed)} names; {len(findings)} differences recorded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
