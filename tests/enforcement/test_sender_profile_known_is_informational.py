"""enforcement: ``ChannelMessage.sender_profile_known`` is read only where it is informational.

The flag says a channel adapter could not read the sender's profile (#878). It exists so a host
can tell the agent "this sender's profile couldn't be read" instead of letting an unknown sender
read as a sparse profile. It is NOT evidence about who the sender is: a profile lookup fails
for reasons that have nothing to do with the sender, and nothing may authorize, deny, or
reconcile an identity on it (owner ruling on #878).

A docstring saying so does not stop the first package that finds it handy. So outside
``threetears.channels``, which sets it, any read of the name is refused unless the file is
listed in :data:`_REVIEWED_READERS` with what it does with the flag. A new reader is a reviewed
edit to this file, not one line in a package.

Static parsing only -- no imports executed -- consistent with the rest of ``tests/enforcement``.
"""

from __future__ import annotations

import ast
from pathlib import Path

from threetears.enforcement.common import parse_python_path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_GLOBS = ("packages/*/src", "packages/agent/*/src")
_FIELD = "sender_profile_known"

#: The package that sets the flag. Repo-relative posix prefix.
_SETTER = "packages/channels/src/"

#: Files outside the setter that may read the flag, each with what it does with it. Empty:
#: nothing in this workspace reads it. A host shows it to its agent; that lives in the host.
_REVIEWED_READERS: dict[str, str] = {}


def _mentions(tree: ast.AST) -> list[int]:
    """Return the lines where a module names the field: an attribute, a keyword, or a string.

    A string catches ``getattr(msg, "sender_profile_known")`` and a serialized message read as a
    dict.

    :param tree: parsed module
    :ptype tree: ast.AST
    :return: line numbers that name the field, sorted
    :rtype: list[int]
    """
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == _FIELD:
            lines.add(node.lineno)
        elif isinstance(node, ast.keyword) and node.arg == _FIELD:
            lines.add(node.value.lineno)
        elif isinstance(node, ast.Constant) and node.value == _FIELD:
            lines.add(node.lineno)
    return sorted(lines)


def _source_files() -> list[Path]:
    """Return every shipped python file across the workspace's source trees.

    :return: python files under each package's ``src``
    :rtype: list[Path]
    """
    found: list[Path] = []
    for glob in _PACKAGE_GLOBS:
        for root in sorted(_REPO_ROOT.glob(glob)):
            found.extend(sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts))
    return found


def test_the_source_trees_were_discovered() -> None:
    """The globs must actually match -- a silent zero would pass every test below.

    :return: none
    :rtype: None
    :raises AssertionError: if almost nothing was discovered
    """
    assert len(_source_files()) > 100, (
        f"only {len(_source_files())} source files matched {_PACKAGE_GLOBS}. The layout changed "
        "and this guard is now inspecting almost nothing."
    )


def test_the_setter_is_where_it_is_looked_for() -> None:
    """The adapter that sets the flag is seen by the scan, so a moved field fails here first.

    :return: none
    :rtype: None
    :raises AssertionError: if no file under the setter names the field
    """
    setters = [
        path
        for path in _source_files()
        if path.relative_to(_REPO_ROOT).as_posix().startswith(_SETTER) and _mentions(parse_python_path(path))
    ]
    assert setters, f"nothing under {_SETTER} names {_FIELD}; the field moved and this guard checks nothing"


def test_each_way_of_reading_it_is_seen() -> None:
    """An attribute, a keyword and a string each count as naming the field.

    :return: none
    :rtype: None
    :raises AssertionError: if a form of read is missed
    """
    source = "\n".join(
        [
            "a = msg.sender_profile_known",
            "b = Thing(sender_profile_known=False)",
            'c = getattr(msg, "sender_profile_known")',
            "d = msg.sender_profile",
        ]
    )
    assert _mentions(ast.parse(source)) == [1, 2, 3]


def test_only_a_reviewed_reader_reads_it_outside_channels() -> None:
    """No package outside ``threetears.channels`` reads the flag without a reviewed entry.

    :return: none
    :rtype: None
    :raises AssertionError: if an unlisted file names the field
    """
    violations: list[str] = []
    for path in _source_files():
        relative = path.relative_to(_REPO_ROOT).as_posix()
        if relative.startswith(_SETTER) or relative in _REVIEWED_READERS:
            continue
        violations.extend(f"{relative}:{line}" for line in _mentions(parse_python_path(path)))
    assert not violations, (
        f"these files read {_FIELD} outside threetears.channels:\n  "
        + "\n  ".join(violations)
        + "\n\nThe flag is informational: a failed profile lookup says nothing about who the sender "
        "is, so nothing may authorize, deny or reconcile an identity on it. If this file only "
        "passes it on for the agent to see, add it to _REVIEWED_READERS in this test with what it "
        "does with the flag."
    )


def test_every_reviewed_reader_still_reads_it() -> None:
    """A listed file that no longer names the field is a stale entry, so the list only shrinks.

    :return: none
    :rtype: None
    :raises AssertionError: if a listed file is missing or no longer names the field
    """
    stale = [
        relative
        for relative in _REVIEWED_READERS
        if not (_REPO_ROOT / relative).is_file() or not _mentions(parse_python_path(_REPO_ROOT / relative))
    ]
    assert not stale, f"these _REVIEWED_READERS entries no longer read {_FIELD}; remove them: {stale}"
