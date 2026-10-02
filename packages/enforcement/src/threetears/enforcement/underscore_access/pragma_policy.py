"""the SLF001 suppression policy: no inline pragma, and a per-file ignore only on a recorded src module.

Owner ruling, 2026-10-01: a leading underscore is a stability contract in ``src/`` and in
``tests/`` alike. A test reaches an object through its front door, and reaching into a
third-party library's private members is confined to one ``src`` module per library, recorded in
the exemptions ledger with a specific rationale. Two spellings defeated that contract while every
gate stayed green, and this module refuses both:

- **an inline pragma.** ``# noqa: SLF001`` (or a bare ``# noqa`` on a line that reads a private
  name, or a file-level ``# ruff: noqa``) silences ruff for that access, and nothing records why.
  Inline pragmas outnumbered recorded exemptions by about an order of magnitude when this landed.
- **a per-file ignore on a test file.** ``"tests/test_x.py" = ["SLF001"]`` turns every private
  access in that file, present and future, into an unreviewed one. A per-file ignore is
  legitimate only on a ``src`` module that confines a third-party library's private surface, and
  only alongside an exemptions-ledger entry for that module.

The ledger is held to the same line: an entry naming a test file is the "tests need access"
exemption the ruling bans, wherever it is written.

Discovery reads every ruff config ruff itself would read -- the same
:func:`~threetears.enforcement.underscore_access.ruff_config.ruff_configs` the ledger checks use --
and every python file that is this repo's own, excluding exactly what
:func:`~threetears.enforcement.underscore_access.ruff_config.is_vendored` excludes. Both inputs
can come back empty for a reason that is not "clean", so a consumer's shell asserts a floor on
each (see :func:`scanned_python_files` and :func:`slf001_ignored_files`).

**Enabling it in a consumer repo** is one thin test module under ``tests/enforcement/``::

    from pathlib import Path

    from threetears.enforcement.underscore_access import (
        scanned_python_files,
        slf001_ignored_files,
        slf001_policy_findings,
    )

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _LEDGER = _REPO_ROOT / "tests" / "enforcement" / "_underscore_exemptions.txt"


    def test_the_scan_covers_this_repo() -> None:
        assert len(scanned_python_files(_REPO_ROOT)) > 50  # a floor sized to the repo


    def test_no_slf001_suppression_outside_a_recorded_src_module() -> None:
        findings = slf001_policy_findings(_REPO_ROOT, _LEDGER)
        assert not findings, "\\n".join(findings)
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

from threetears.enforcement.underscore_access.ledger import ledger_paths, private_accesses
from threetears.enforcement.underscore_access.ruff_config import (
    exempted_files,
    is_vendored,
    ruff_configs,
    slf001_globs,
)

__all__ = [
    "TEST_DIRECTORIES",
    "is_src_module",
    "ledger_entries_outside_src",
    "scanned_python_files",
    "slf001_ignored_files",
    "slf001_ignores_outside_src",
    "slf001_ignores_without_a_ledger_entry",
    "slf001_policy_findings",
    "slf001_pragma_offenders",
]

#: a ``noqa`` whose code list names SLF001, or the whole SLF group. Matched against COMMENT tokens
#: only, so a string or docstring quoting the spelling -- this module's own, for one -- is not a
#: pragma.
_SLF_CODE_PRAGMA = re.compile(r"#\s*(?:ruff\s*:\s*)?noqa\s*:[^#\n]*\bSLF(?:001)?\b")

#: a ``noqa`` with no code list: on a line it silences every rule there, SLF001 included.
_BARE_LINE_PRAGMA = re.compile(r"#\s*noqa(?!\s*:)")

#: the file-level ruff directive with no code list: it silences every rule in the whole file.
#: (spelled out only in the pattern below, since ruff reads the spelling in any comment.)
_BARE_FILE_PRAGMA = re.compile(r"#\s*ruff\s*:\s*noqa(?!\s*:)")

#: path segments that make a file a test file wherever they appear.
TEST_DIRECTORIES: frozenset[str] = frozenset({"tests", "test"})


def is_src_module(relative_path: str) -> bool:
    """whether a repo-relative path is production source rather than a test file.

    :param relative_path: posix path relative to the repo root
    :ptype relative_path: str
    :return: ``True`` for a file under a ``src`` directory, below no ``tests``/``test`` directory,
        and not itself a ``test_*.py`` or ``conftest.py``
    :rtype: bool
    """
    parts = Path(relative_path).parts
    name = parts[-1] if parts else ""
    in_src = "src" in parts[:-1]
    in_tests = any(part in TEST_DIRECTORIES for part in parts[:-1])
    is_test_file = name.startswith("test_") or name == "conftest.py"
    return in_src and not in_tests and not is_test_file


def scanned_python_files(repo_root: Path) -> list[Path]:
    """every python file that is this repo's own: vendored trees and nested checkouts excluded.

    :param repo_root: the repo's root
    :ptype repo_root: Path
    :return: the files, sorted
    :rtype: list[Path]
    """
    return sorted(path for path in repo_root.rglob("*.py") if not is_vendored(path, repo_root))


def _comments(source: str) -> list[tuple[int, str]]:
    """every comment in ``source`` with the line it sits on.

    :param source: python source text
    :ptype source: str
    :return: ``(line, comment text)`` pairs in file order
    :rtype: list[tuple[int, str]]
    """
    found: list[tuple[int, str]] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                found.append((token.start[0], token.string))
    except (
        tokenize.TokenError,
        SyntaxError,
    ):  # NOSILENT: an untokenizable file is reported by nothing ruff runs either; what was read so far is kept
        pass
    return found


def slf001_pragma_offenders(repo_root: Path) -> list[str]:
    """``path:line`` for every comment that suppresses SLF001, anywhere in the repo.

    three spellings: a ``noqa`` code list naming SLF001 (or the SLF group), on any line; a bare
    ``# noqa`` on a line that reads a private name; a bare file-level ``# ruff: noqa`` in a file
    that reads one. A bare ``noqa`` on a line with no private access suppresses no SLF001 and is
    left to whatever owns its own rule.

    :param repo_root: the repo's root
    :ptype repo_root: Path
    :return: one ``path:line -- comment`` per offender, sorted
    :rtype: list[str]
    """
    offenders: list[str] = []
    for path in scanned_python_files(repo_root):
        source = path.read_text(errors="replace")
        if "noqa" not in source:
            continue
        rel = path.relative_to(repo_root).as_posix()
        private_lines = {line for line, _symbol in private_accesses(path)}
        for line, comment in _comments(source):
            names_slf = _SLF_CODE_PRAGMA.search(comment) is not None
            bare_on_access = _BARE_LINE_PRAGMA.search(comment) is not None and line in private_lines
            bare_file_wide = _BARE_FILE_PRAGMA.search(comment) is not None and bool(private_lines)
            if names_slf or bare_on_access or bare_file_wide:
                offenders.append(f"{rel}:{line} -- {comment.strip()}")
    return offenders


def slf001_ignored_files(repo_root: Path) -> list[tuple[str, str, str]]:
    """every file a per-file SLF001 ignore covers, with the config and the key that cover it.

    :param repo_root: the repo's root
    :ptype repo_root: Path
    :return: ``(config, key, file)``, each repo-relative, sorted
    :rtype: list[tuple[str, str, str]]
    """
    found: list[tuple[str, str, str]] = []
    for config in ruff_configs(repo_root):
        config_rel = config.relative_to(repo_root).as_posix()
        for key in slf001_globs(config):
            for path in exempted_files(config, key, repo_root):
                found.append((config_rel, key, path.relative_to(repo_root).as_posix()))
    return sorted(found)


def slf001_ignores_outside_src(repo_root: Path) -> list[str]:
    """per-file SLF001 ignores that cover a file which is not a ``src`` module.

    :param repo_root: the repo's root
    :ptype repo_root: Path
    :return: one line per covered file, sorted
    :rtype: list[str]
    """
    return [
        f"{config}: {key!r} covers {path}, which is not a src module"
        for config, key, path in slf001_ignored_files(repo_root)
        if not is_src_module(path)
    ]


def slf001_ignores_without_a_ledger_entry(repo_root: Path, exemptions_path: Path) -> list[str]:
    """``src`` modules with a per-file SLF001 ignore and no exemptions-ledger entry.

    :param repo_root: the repo's root
    :ptype repo_root: Path
    :param exemptions_path: the underscore-access exemptions ledger
    :ptype exemptions_path: Path
    :return: one line per unrecorded module, sorted
    :rtype: list[str]
    """
    recorded = set(ledger_paths(exemptions_path))
    return [
        f"{config}: {key!r} covers {path}, which has no entry in {exemptions_path.name}"
        for config, key, path in slf001_ignored_files(repo_root)
        if is_src_module(path) and path not in recorded
    ]


def ledger_entries_outside_src(exemptions_path: Path) -> list[str]:
    """exemptions-ledger paths that are not ``src`` modules: a test's private access, recorded.

    :param exemptions_path: the underscore-access exemptions ledger
    :ptype exemptions_path: Path
    :return: each offending path once, sorted
    :rtype: list[str]
    """
    return sorted({path for path in ledger_paths(exemptions_path) if not is_src_module(path)})


def slf001_policy_findings(repo_root: Path, exemptions_path: Path) -> list[str]:
    """every breach of the SLF001 suppression policy, each line naming its fix.

    :param repo_root: the repo's root
    :ptype repo_root: Path
    :param exemptions_path: the underscore-access exemptions ledger
    :ptype exemptions_path: Path
    :return: findings; empty when the repo complies
    :rtype: list[str]
    """
    findings = [
        f"inline SLF001 suppression: {offender}. Reach the object through its front door, or "
        "confine a third-party library's private members to its one src module."
        for offender in slf001_pragma_offenders(repo_root)
    ]
    findings += [
        f"per-file SLF001 ignore on a test file: {finding}. Rewrite the test against the front door "
        "and delete the ignore."
        for finding in slf001_ignores_outside_src(repo_root)
    ]
    findings += [
        f"unrecorded per-file SLF001 ignore: {finding}. Record each access with a specific "
        "rationale, or remove the ignore."
        for finding in slf001_ignores_without_a_ledger_entry(repo_root, exemptions_path)
    ]
    findings += [
        f"exemptions-ledger entry for a test file: {path}. A test reaches what it asserts through "
        "the front door; delete the entry with the access."
        for path in ledger_entries_outside_src(exemptions_path)
    ]
    return findings
