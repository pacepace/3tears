"""enforcement: prose must not qualify a hub table with the ``platform`` schema.

The hub writes to the schema named by its ``FOURTEENAIBOTS_DB_SCHEMA`` setting.
On the compose stack that is ``aibots``. It used to be ``platform``, and the
hardcoded default was removed precisely because every consumer that forgot to
read the setting fell back to ``platform`` -- which is CORRECT on one deployment
and wrong on every other one. The code was corrected when the default went away.
**The prose was not.** Hundreds of docstrings and comments across the estate went
on saying ``platform.namespaces``, ``platform.agents``, ``platform.roles``.

That is the removed default by another route, and it has already cost real time:
a downstream engineer read those docstrings, queried ``platform.namespaces`` for
three days against a stale leftover schema, and filed a confident bug report
against working code. It also reached a downstream spec, where it will keep
being true-looking until someone loses days to it again.

So: **prose names the table, never the schema.** ``namespaces``, not
``platform.namespaces``. Where a sentence needs to say WHOSE table it is, say so
in words -- "the hub's ``namespaces`` table" -- because ownership is the thing
the qualifier was actually carrying, and a schema name is a bad way to carry it.
Where a sentence needs to name the schema itself, "the configured platform
schema" is the phrase; the concrete name comes from ``FOURTEENAIBOTS_DB_SCHEMA``
at deploy time and nothing in this repo may assume it.

FOUR classes of ``platform.<something>`` are NOT this defect, and each is
excluded here on its own stated grounds rather than by one blanket regex:

1. **``system.platform.*`` is a NATS subject / proxy-pool namespace, not a
   schema-qualified table.** ``system.platform.rbac`` is the broker's read-only
   carve-out, spelled that way on the wire. Rewriting one breaks routing.
2. **Attribute access on a local variable named ``platform``.** Excluded
   STRUCTURALLY: this guard reads comments and docstrings only, never executable
   code, so ``origin=platform.id`` is never even looked at.
3. **SQL that really does name a schema.** Migrations that ran cross-schema, and
   assertions pinning those statements, are historical fact -- rewriting the
   statement changes what ran, and rewriting only the docstring beside it makes
   the docstring wrong. Non-docstring string literals are excluded structurally
   (same mechanism as class 2); the handful of modules whose *docstrings*
   describe such SQL are named in :data:`_ALLOWED`, each with its reason.
4. **Hostnames and URLs.** ``platform.invalid`` (a reserved TLD used in test
   fixtures), ``platform.openai.com``, ``nats://platform.local:4222``. These are
   network names that merely start with the same word.

Scope is the shipped source trees plus package READMEs. Test trees are
deliberately out of scope: their fixtures create a literal ``platform`` schema
to migrate against, and their assertions pin SQL strings by exact text, so the
qualifier there is describing the fixture rather than misdescribing production.

Static parsing only -- no imports executed, no network -- consistent with the
rest of ``tests/enforcement``.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_GLOBS = ("packages/*/src", "packages/agent/*/src")
_README_GLOBS = ("packages/*/README.md", "packages/agent/*/README.md")

#: ``platform.`` followed by a table-shaped identifier or the ``*`` wildcard.
#: The leading guard keeps ``system.platform.rbac`` and ``foo_platform.bar``
#: from matching at the ``platform`` boundary.
_QUALIFIER = re.compile(r"(?<![\w.])platform\.(?P<label>\*|[a-z_][a-z0-9_]*)")

#: Class 1. A match whose immediately preceding text is one of these is a NATS
#: subject, not a schema. ``system.`` is the live one; the tuple is a list so a
#: second subject family does not need a second mechanism.
_SUBJECT_PREFIXES = ("system.",)

#: Class 4. Labels that make ``platform.<label>`` a hostname rather than a
#: table. ``invalid`` / ``test`` / ``example`` / ``localhost`` are reserved by
#: RFC 2606 + RFC 6761 and turn up in fixtures; the rest are ordinary TLDs.
_HOST_LABELS = frozenset(
    {"invalid", "test", "example", "localhost", "local", "internal", "com", "net", "org", "io", "dev", "ai"}
)

#: Class 4, second form. A match sitting inside a URL is a host, whatever its
#: label. Anything ending in one of these immediately before ``platform`` means
#: the token is being used as a network name.
_URL_LEAD_INS = ("//", "@", "://")

#: Class 3. Modules whose DOCSTRINGS legitimately name a ``platform``-qualified
#: table, each with the reason. Repo-relative posix paths.
_ALLOWED: dict[str, str] = {
    "packages/core/src/threetears/core/data/migrations/drift.py": (
        "documents a schema-STRIPPING helper: ``platform.agents`` -> ``agents`` is the "
        "worked example of what the function does, so dropping the prefix deletes the example"
    ),
    "packages/enforcement/src/threetears/enforcement/cache/walkers.py": (
        "same as drift.py -- the docstring's ``platform.customers`` -> ``customers`` line "
        "is the walker's documented transform, not a claim about where a table lives"
    ),
    "packages/agent/workspace/src/threetears/agent/workspace/migrations/v003_workspace_namespace_backfill.py": (
        "historical DDL: this migration's SQL literally reads ``INSERT INTO platform.namespaces``. "
        "The docstring describes the statement that ran; correcting the prose alone would make it lie"
    ),
    "packages/agent/workspace/src/threetears/agent/workspace/migrations/__init__.py": (
        "indexes v003 by what it does, so it inherits v003's historical-DDL wording"
    ),
    "packages/agent/workspace/src/threetears/agent/workspace/tools/helpers.py": (
        "quotes ``_SELECT_NAMESPACE_CUSTOMER_SQL``, a fully-qualified fallback literal in this "
        "same module for pools with no ``namespace=`` support. The docstring names a real string"
    ),
}


def _is_excluded(line: str, start: int, label: str) -> bool:
    """Return whether a match at `start` in `line` is one of the four false-positive classes.

    :param line: the source line the match sits on
    :ptype line: str
    :param start: column offset of the ``platform`` token
    :ptype start: int
    :param label: the identifier (or ``*``) immediately after the dot
    :ptype label: str
    :return: True when the match is a subject, a host, or a URL rather than a table
    :rtype: bool
    """
    before = line[:start]
    if any(before.endswith(prefix) for prefix in _SUBJECT_PREFIXES):
        return True
    if label in _HOST_LABELS:
        return True
    return any(before.endswith(lead) for lead in _URL_LEAD_INS)


def _hits(text: str, first_line: int) -> list[tuple[int, str]]:
    """Return every stale qualifier in a block of prose, as ``(line number, matched text)``.

    :param text: prose block (a docstring, a comment, or a whole markdown file)
    :ptype text: str
    :param first_line: 1-based line number the block starts on
    :ptype first_line: int
    :return: line number and matched token for each surviving hit
    :rtype: list[tuple[int, str]]
    """
    found: list[tuple[int, str]] = []
    for offset, line in enumerate(text.splitlines()):
        for match in _QUALIFIER.finditer(line):
            if not _is_excluded(line, match.start(), match.group("label")):
                found.append((first_line + offset, match.group(0)))
    return found


def _prose_blocks(source: str) -> list[tuple[int, str]]:
    """Return every comment and docstring in a python module, as ``(line number, text)``.

    Executable code and non-docstring string literals are never returned -- that
    is what excludes false-positive classes 2 and 3 structurally.

    :param source: python source text
    :ptype source: str
    :return: prose blocks with their starting line numbers
    :rtype: list[tuple[int, str]]
    """
    blocks: list[tuple[int, str]] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            blocks.append((token.start[0], token.string))
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        doc = ast.get_docstring(node, clean=False)
        if doc is None:
            continue
        constant = node.body[0]
        assert isinstance(constant, ast.Expr)
        blocks.append((constant.lineno, doc))
    return blocks


def _scanned_files() -> list[Path]:
    """Return every shipped python file and package README in the workspace.

    :return: files whose prose this guard reads
    :rtype: list[Path]
    """
    found: list[Path] = []
    for glob in _PACKAGE_GLOBS:
        for root in sorted(_REPO_ROOT.glob(glob)):
            found.extend(sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts))
    for glob in _README_GLOBS:
        found.extend(sorted(_REPO_ROOT.glob(glob)))
    return found


def test_the_source_trees_were_discovered() -> None:
    """The globs must actually match -- a silent zero would pass the guard below.

    :return: none
    :rtype: None
    :raises AssertionError: if almost nothing was discovered
    """
    assert len(_scanned_files()) > 100, (
        f"only {len(_scanned_files())} files matched {_PACKAGE_GLOBS + _README_GLOBS}. "
        "The layout changed and this guard is now inspecting almost nothing."
    )


def test_every_allowlisted_path_still_exists() -> None:
    """An allowlist entry that no longer resolves is silently granting nothing.

    :return: none
    :rtype: None
    :raises AssertionError: if an allowlisted path was moved or deleted
    """
    missing = [path for path in _ALLOWED if not (_REPO_ROOT / path).is_file()]
    assert not missing, (
        f"allowlisted paths no longer exist: {missing}. Delete the entry if the module went away, "
        "or update it if the module moved -- a stale entry exempts nothing and hides the next one."
    )


def test_no_prose_qualifies_a_hub_table_with_the_platform_schema() -> None:
    """Docstrings, comments and READMEs name the table, not the schema it happens to sit in.

    :return: none
    :rtype: None
    :raises AssertionError: if a stale ``platform.<table>`` qualifier reached shipped prose
    """
    violations: list[str] = []
    for path in _scanned_files():
        relative = path.relative_to(_REPO_ROOT).as_posix()
        if relative in _ALLOWED:
            continue
        text = path.read_text(encoding="utf-8")
        blocks = [(1, text)] if path.suffix == ".md" else _prose_blocks(text)
        for first_line, block in blocks:
            violations.extend(f"{relative}:{line}: {token}" for line, token in _hits(block, first_line))

    assert not violations, (
        "prose still qualifies a table with the ``platform`` schema:\n  "
        + "\n  ".join(violations)
        + "\n\nThe hub's schema is whatever FOURTEENAIBOTS_DB_SCHEMA names -- ``aibots`` on the "
        "compose stack, ``platform`` on nothing that ships. Write the bare table name "
        '(``namespaces``), or name the owner in words ("the hub\'s ``namespaces`` table") when '
        "the point is whose table it is. See this module's docstring for the three days somebody "
        "already lost to the old wording, and for the four exclusions this guard encodes."
    )
