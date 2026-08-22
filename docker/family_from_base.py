"""prepare the exported requirements when the family comes from the BASE image.

WHY THIS EXISTS
---------------

A consumer image installs its dependencies from ``uv.lock`` via ``uv export |
uv pip install -r``. On a RELEASED lock the ``3tears*`` family resolves from an
index and installs normally, and the Dockerfile deliberately does NOT exclude it
so that a base image built from a different family version is caught.

This module ships INSIDE ``threetears-base`` at
``/opt/threetears/family_from_base.py`` rather than being copied into each
consumer's build context. Every consumer that inherits the base needs the same
check against the same family, and a per-repo copy is a second source of truth
that drifts silently -- the exact failure this file exists to prevent one layer
down.

On a DEVELOPMENT branch the same lock resolves the family as EDITABLE PATH
dependencies into sibling checkouts::

    -e ../14-eng-ai-bot-agents-ripple/.3tears/packages/datasources

Those paths are outside the consumer's build context. Docker cannot read them, so
the install fails and the image cannot be built at all -- which is what forced a
3tears release before any local end-to-end test could run.

The family is in exactly the position ``aibots-agents`` is already in: a path
dependency the build context cannot see, ALREADY INSTALLED in the inherited base
venv. ``threetears-base`` builds the family from source
(``uv build --all-packages --wheel``) and installs it, so when that base is built
from the same worktree the venv already holds precisely the code under test.

WHAT IT DOES NOT GIVE UP
------------------------

Excluding the family would drop the guarantee the Dockerfile's comment names:
installing from the lock "catches a mismatched base image". So this module does
not merely strip -- it REPLACES that check with a stricter one. It asserts that
the version installed in the base venv equals the version ``uv.lock`` records
for every family package, and fails the build naming both when they disagree.

The default path is untouched. ``--mode 0`` writes nothing and asserts nothing,
so a released build behaves exactly as before.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from importlib import metadata
from pathlib import Path

__all__ = ["family_versions_from_lock", "installed_version", "main", "strip_family_requirements"]

#: distribution-name prefix of every package in the framework family. The family
#: releases in lockstep, so one prefix covers it and a new member needs no edit
#: here -- which is the point, because a hand-listed member set drifts.
_FAMILY_PREFIX = "3tears"

#: a requirement line naming a family distribution, editable or not. Matched on
#: the DISTRIBUTION rather than on the ``-e`` marker: stripping every editable
#: line would also strip a legitimately-editable non-family dependency, and
#: nothing here should depend on the family being the only one.
_FAMILY_LINE = re.compile(rf"^\s*(-e\s+)?\S*({_FAMILY_PREFIX}[a-z0-9-]*)\b", re.IGNORECASE)


def family_versions_from_lock(lock: Path) -> dict[str, str]:
    """the version ``uv.lock`` records for every family package.

    :param lock: path to ``uv.lock``
    :ptype lock: pathlib.Path
    :return: distribution name to version, for family packages only
    :rtype: dict[str, str]
    """
    parsed = tomllib.loads(lock.read_text(encoding="utf-8"))
    return {
        package["name"]: package["version"]
        for package in parsed.get("package", [])
        if str(package.get("name", "")).startswith(_FAMILY_PREFIX) and package.get("version")
    }


def installed_version(distribution: str) -> str | None:
    """the version of one distribution installed in the current environment.

    :param distribution: distribution name
    :ptype distribution: str
    :return: installed version, or ``None`` when it is not installed
    :rtype: str | None
    """
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def strip_family_requirements(text: str) -> tuple[str, list[str]]:
    """drop every family requirement from an exported requirements file.

    A requirements file carries continuation lines and comments; only lines
    that NAME a family distribution are removed, and the comment block uv
    emits beneath a requirement is left alone because it starts with ``#``
    and pip ignores it.

    A dropped requirement takes its ``--hash=`` continuations with it. Leaving
    them orphaned produces a file uv refuses with ``Unexpected '-', expected
    '-c', '-e', '-r' or the start of a requirement``, which fails the image
    build at install time -- after the export step succeeded, so the export
    looks fine and the error names neither the family nor this function.

    :param text: contents of the exported requirements file
    :ptype text: str
    :return: the filtered text, and the family lines removed
    :rtype: tuple[str, list[str]]
    """
    kept: list[str] = []
    dropped: list[str] = []
    in_dropped_continuation = False
    for line in text.splitlines():
        is_comment = line.lstrip().startswith("#")
        if _FAMILY_LINE.match(line) and not is_comment:
            dropped.append(line.strip())
            in_dropped_continuation = line.rstrip().endswith("\\")
        elif in_dropped_continuation and not is_comment:
            # a continuation of the requirement just dropped; it ends when a
            # line does not itself continue.
            in_dropped_continuation = line.rstrip().endswith("\\")
        else:
            in_dropped_continuation = False
            kept.append(line)
    return "\n".join(kept) + "\n", dropped


def _verify(lock: Path) -> list[str]:
    """check every family package the lock names against what is installed.

    This is the REPLACEMENT for installing the family from the lock. The
    default build catches a mismatched base image by letting uv resolve the
    recorded version; when the family is taken from the base instead, that
    same mismatch has to be caught here or it is not caught at all.

    :param lock: path to ``uv.lock``
    :ptype lock: pathlib.Path
    :return: human-readable mismatches, empty when the base agrees with the lock
    :rtype: list[str]
    """
    problems: list[str] = []
    for distribution, expected in sorted(family_versions_from_lock(lock).items()):
        present = installed_version(distribution)
        if present is None:
            problems.append(f"{distribution}: uv.lock records {expected}, the base image does not carry it at all")
        elif present != expected:
            problems.append(f"{distribution}: uv.lock records {expected}, the base image carries {present}")
    return problems


def main(argv: list[str] | None = None) -> int:
    """strip the family from the requirements, then prove the base matches.

    :param argv: command-line arguments, or ``None`` for ``sys.argv``
    :ptype argv: list[str] | None
    :return: 0 on success, 1 when the base image and the lock disagree
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["0", "1"], help="1 takes the family from the base image")
    parser.add_argument("--requirements", type=Path, required=True, help="exported requirements file, rewritten")
    parser.add_argument("--lock", type=Path, required=True, help="uv.lock, read for the recorded family versions")
    arguments = parser.parse_args(argv)

    if arguments.mode == "0":
        print("family-from-base: off; the family installs from the lock as usual")
        return 0

    filtered, dropped = strip_family_requirements(arguments.requirements.read_text(encoding="utf-8"))
    arguments.requirements.write_text(filtered, encoding="utf-8")
    print(f"family-from-base: on; {len(dropped)} family requirement(s) will come from the base image")

    problems = _verify(arguments.lock)
    if problems:
        print("FAMILY MISMATCH between the base image and uv.lock:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "\nthe base image was built from different framework source than this lock records. "
            "rebuild threetears-base from the SAME checkout this lock resolves, or drop --mode 1 "
            "and install the family from the lock.",
            file=sys.stderr,
        )
        return 1
    print(f"family-from-base: verified {len(family_versions_from_lock(arguments.lock))} package(s) match the lock")
    return 0


if __name__ == "__main__":  # pragma: no cover - build-time entry point
    sys.exit(main())
