"""``threetears-release``: the one release tool every 3tears and aibots repository runs.

Each repository's ``scripts/bump-version.sh`` is a one-line wrapper around this
entry point, so the logic has exactly one owner and no copy to drift::

    threetears-release [--root DIR] patch|minor|major|release|sync
    threetears-release [--root DIR] verify X.Y.Z

Exit status: 0 done, 1 refused or failed (nothing changed), 2 a wrong invocation.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from threetears.enforcement.release.bump import BUMP_KINDS, run_release
from threetears.enforcement.release.config import ConfigError, load_release_config
from threetears.enforcement.release.versions import parse_version

__all__ = ["main"]

_ACTIONS = (*BUMP_KINDS, "release", "sync", "verify")

_EPILOG = """\
  patch      everything that is not new public API or a new feature
  minor      any new public API or new feature
  major      only when the owner says so
  release    release the version already declared (dates its notes; no version change)
  sync       bring every lockstep location back to the declared version
  verify     check every lockstep location says VERSION; edits nothing

It checks everything before it edits anything, and restores every file it touched
if a step fails. See threetears.enforcement.release.bump for what it refuses.
"""


def _parser() -> argparse.ArgumentParser:
    """the command-line parser.

    :return: parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="threetears-release",
        description="Bump a release version and release its notes, in one step.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root (default: the cwd)")
    parser.add_argument("action", choices=_ACTIONS)
    parser.add_argument("version", nargs="?", help="X.Y.Z, for verify only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """runs the tool.

    :param argv: arguments, without the program name; ``sys.argv[1:]`` when ``None``
    :ptype argv: Sequence[str] | None
    :return: exit status
    :rtype: int
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if (args.action == "verify") != (args.version is not None):
        parser.error("verify takes one X.Y.Z, and no other action takes a version")
    if args.version is not None:
        try:
            parse_version(args.version)
        except ValueError as exc:
            parser.error(str(exc))
    status = 1
    try:
        config = load_release_config(args.root.resolve())
    except (ConfigError, OSError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
    else:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        status = run_release(config, args.action, today, sys.stdout, sys.stderr, args.version)
    return status
