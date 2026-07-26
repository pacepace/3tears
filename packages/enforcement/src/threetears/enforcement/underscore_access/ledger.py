"""reconciliation between ``_underscore_exemptions.txt`` and the code it claims to describe.

the ledger is a list of ``path:line:symbol`` triples, each preceded by a rationale recording why
one private access was judged acceptable. the underscore walkers scan a repo's ``src`` roots and
never enter a ``tests/`` tree, so for every exempted test file the ledger is documentation that
nothing reads back -- and it rots in both directions:

- **stale.** an entry pointing at a line that no longer holds that symbol, or at a file that has
  been deleted outright. worse than a missing one, because it reads as a reviewed decision about
  a specific access and a reader trusts the rationale for code that is not there.
- **missing.** an access on an exempted path with no entry at all, so nothing records why it was
  judged acceptable.

**both directions are needed, and neither implies the other.** a stale-entry check cannot see a
missing entry, because a missing entry points at nothing. and the missing-direction check has to
see accesses ruff would not report: an inline ``# noqa: SLF001`` on a per-file-ignored path
suppresses the finding, so the access reaches neither ruff nor the ledger. everything here walks
the AST for that reason; :func:`blanket_noqa_offenders` keeps the pragmas out regardless.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from threetears.enforcement.underscore_access.ruff_config import all_exempted_files

__all__ = [
    "blanket_noqa_offenders",
    "ledger_entries",
    "missing_files",
    "orphan_rationales",
    "private_accesses",
    "unlisted_accesses",
    "unresolved_entries",
]

#: any comment that would stop ruff reporting an ``SLF001`` on that line, not merely the
#: fully-spelled form. a bare suppression with no code list covers everything, and the
#: file-scoped `ruff:` spelling covers a whole file. both blind the ledger regeneration
#: identically, so matching only the explicit spelling would leave the cheapest one through.
_BLANKET_NOQA = re.compile(
    # a bare suppression with no code list, in either the plain or the `ruff:` spelling
    r"#\s*(?:ruff:\s*)?noqa(?!\s*:)"
    # or a code list that names SLF001 (or the bare SLF group)
    r"|#\s*(?:ruff:\s*)?noqa\s*:[^#\n]*\bSLF(?:001)?\b"
)

_RATIONALE_PREFIX = "# rationale:"


def ledger_entries(exemptions_path: Path) -> list[tuple[str, int, str]]:
    """every ``path:line:symbol`` triple in the ledger, comments and blanks dropped."""
    found: list[tuple[str, int, str]] = []
    for raw in exemptions_path.read_text().split("\n"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        path, _, rest = line.partition(":")
        number, _, symbol = rest.partition(":")
        if path and number.isdigit() and symbol:
            found.append((path, int(number), symbol))
    return found


def private_accesses(path: Path) -> set[tuple[int, str]]:
    """every ``obj._private`` read in *path* that is not ``self``/``cls``.

    AST rather than ruff, deliberately: ruff honours an inline ``noqa`` and would report nothing
    for exactly the accesses that go missing from the ledger. dunders are excluded because
    ``__init__`` and friends are public protocol, not private state.
    """
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:  # a file the repo cannot parse is not this module's business
        return set()

    found: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not node.attr.startswith("_") or node.attr.startswith("__"):
            continue
        if isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"}:
            continue
        found.add((node.lineno, node.attr))
    return found


def orphan_rationales(exemptions_path: Path) -> list[int]:
    """1-indexed line numbers of rationales not followed by the entry they justify.

    note the direction: this finds a RATIONALE with no entry, not an entry with no rationale.
    the converse is the ledger header's stated rule and is a different check.

    this direction caught a real bug: a regeneration script's header scan stopped at the first
    ENTRY rather than the first rationale, so a rationale line was captured as header AND
    re-emitted with its entry, leaving one orphan at the top per run.
    """
    lines = exemptions_path.read_text().split("\n")
    orphans: list[int] = []
    for number, raw in enumerate(lines, 1):
        # `strip()` first, matching a regeneration script's own header scan: an indented
        # rationale is invisible to a bare `startswith`, and the two disagreeing is how a line
        # goes unseen by one of them.
        if not raw.strip().startswith(_RATIONALE_PREFIX):
            continue
        following = lines[number] if number < len(lines) else ""
        if not following.strip() or following.strip().startswith("#"):
            orphans.append(number)
    return orphans


def missing_files(exemptions_path: Path, repo_root: Path) -> list[str]:
    """ledger paths that do not exist, so the entry documents a decision about nothing."""
    return sorted({path for path, _, _ in ledger_entries(exemptions_path) if not (repo_root / path).exists()})


def unresolved_entries(exemptions_path: Path, repo_root: Path) -> list[str]:
    """``path:line:symbol`` triples whose line no longer contains the symbol they name.

    files that do not exist at all are skipped, so :func:`missing_files` reports them once
    instead of both checks counting the same entry.
    """
    unresolved: list[str] = []
    for path, number, symbol in ledger_entries(exemptions_path):
        source = repo_root / path
        if not source.exists():
            continue
        lines = source.read_text(errors="replace").split("\n")
        if number > len(lines) or symbol not in lines[number - 1]:
            unresolved.append(f"{path}:{number}:{symbol}")
    return unresolved


def unlisted_accesses(exemptions_path: Path, repo_root: Path) -> list[str]:
    """private accesses on an exempted path that have no ledger entry."""
    entries = set(ledger_entries(exemptions_path))
    unlisted: list[str] = []
    for source in all_exempted_files(repo_root):
        rel = source.relative_to(repo_root).as_posix()
        for number, symbol in sorted(private_accesses(source)):
            if (rel, number, symbol) not in entries:
                unlisted.append(f"{rel}:{number}:{symbol}")
    return unlisted


def blanket_noqa_offenders(repo_root: Path) -> list[str]:
    """``path:line`` for every ``noqa`` suppressing ``SLF001`` on an already-exempted path.

    the pragma is redundant to ruff, so on its own this would be a tidiness rule. it matters
    because of what the redundancy does to the ledger: a regeneration built on ruff's output
    never sees the access, and it silently drops out of the record.
    """
    offenders: list[str] = []
    for path in all_exempted_files(repo_root):
        for number, line in enumerate(path.read_text(errors="replace").split("\n"), 1):
            if _BLANKET_NOQA.search(line):
                offenders.append(f"{path.relative_to(repo_root)}:{number}")
    return offenders
