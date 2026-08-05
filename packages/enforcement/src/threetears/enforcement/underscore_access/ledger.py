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
import subprocess
from collections import Counter
from pathlib import Path

from threetears.enforcement.underscore_access.ruff_config import all_exempted_files

__all__ = [
    "blanket_noqa_offenders",
    "carry_forward_rationales",
    "enclosing_scopes",
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


def enclosing_scopes(path: Path) -> dict[int, str]:
    """Map each line in *path* to the dotted name of the function or class enclosing it.

    Exists so a rationale can be tied to the ACCESS it describes rather than to the symbol name.
    A file commonly touches the same private name from several places -- three tests each
    reaching for the same helper, for different reasons -- and a ledger keyed on
    ``(path, symbol)`` collapses those into one, silently giving every entry the first
    entry's reason.

    That failure is not correctable by hand: rewriting the wrong one is reverted by the next
    regeneration, because the regeneration is what applied it. A scope is the smallest thing
    that distinguishes them and survives the line drift the ledger exists to absorb.

    Innermost wins, so a nested helper is named rather than the function containing it. Lines
    outside any function or class are absent from the mapping; module-level accesses key on the
    symbol alone, which is unambiguous there because there is only one such scope.
    """
    return _scopes_from_source(path.read_text(errors="replace"))


def _scopes_from_source(text: str) -> dict[int, str]:
    """:func:`enclosing_scopes` for source that is not on disk -- a git blob, mainly."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}

    scopes: dict[int, str] = {}

    def _walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                qualname = f"{prefix}.{child.name}" if prefix else child.name
                end = child.end_lineno or child.lineno
                for line in range(child.lineno, end + 1):
                    # Assigned unconditionally so an inner definition overwrites the outer one
                    # it sits inside; iteration is outside-in, so innermost wins.
                    scopes[line] = qualname
                _walk(child, qualname)
            else:
                _walk(child, prefix)

    _walk(tree, "")
    return scopes


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


def _git_stdout(repo_root: Path, *args: str) -> str | None:
    """stdout of a git command run in *repo_root*, or ``None`` when git cannot answer.

    ``None`` covers every way the answer can be unavailable -- git absent, not a repository,
    an object that does not exist -- because the callers all degrade the same way regardless
    of which one it was.
    """
    try:
        proc = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True, check=False)
    except OSError:  # NOSILENT: a machine without git is a supported environment here -- None is this function's documented "cannot answer", and the caller degrades to current-file resolution
        return None
    return proc.stdout if proc.returncode == 0 else None


def _source_when_ledger_was_written(exemptions_path: Path, repo_root: Path, rel_path: str) -> str | None:
    """*rel_path*'s content at the last commit that touched the ledger, or ``None``.

    The moment the ledger was last written is the moment its line numbers were last known to
    match the code, so that commit's tree is the one a drifted entry's number indexes into.
    ``None`` when there is no such snapshot to consult: no git, a ledger that has never been
    committed, or a file absent from that commit.

    The pathspec and the blob path both carry an explicit ``./`` so git resolves them against
    *repo_root* (via ``-C``) rather than against the repository toplevel -- the two differ
    whenever the caller's root is a subdirectory of the actual repository.
    """
    try:
        ledger_rel = exemptions_path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:  # NOSILENT: a ledger outside repo_root has no repo-relative path to ask git about -- None is this function's documented "no snapshot", and the caller degrades to current-file resolution
        return None
    commit = _git_stdout(repo_root, "rev-list", "-1", "HEAD", "--", f"./{ledger_rel}")
    if commit is None or not commit.strip():
        return None
    return _git_stdout(repo_root, "show", f"{commit.strip()}:./{rel_path}")


def carry_forward_rationales(exemptions_path: Path, repo_root: Path) -> dict[tuple[str, str, str, int], str]:
    """Map ``(path, scope, symbol, occurrence)`` to the rationale recorded for it.

    Keyed on the enclosing scope as well as the symbol because a file routinely reaches for the
    same private name from several places for different reasons. Keyed on ``(path, symbol)``
    alone -- which this was once -- the first rationale is applied to every access of that name in
    the file, so several unrelated entries all document whichever reason came first.

    Keyed on the OCCURRENCE within that scope as well, because the scope was not enough either.
    One function routinely touches the same private twice for two reasons -- an arrange line that
    reads a value and an assert line that compares it, which is the commonest shape in a test --
    and those collapse to one key. The observed instance: two ``_x11vnc`` reads in one
    ``test_start_is_idempotent``, whose second hand-written rationale was silently replaced by a
    copy of the first one's on the next regeneration.

    That failure is worse than ordinary staleness because it is self-repairing in the wrong
    direction: correcting one of the wrong entries by hand is reverted by the next regeneration,
    since the regeneration is what wrote it. The ledger then has a class of entry that is
    permanently and silently wrong about the code it describes.

    The occurrence index is the entry's ordinal among same-scope, same-symbol entries in ledger
    order, which is line order. That is stable across the line shifts carry-forward exists to
    survive, and only mismatches if the accesses are REORDERED within their function -- at which
    point which rationale belongs to which access is a genuine question rather than one this can
    answer.

    Lives here rather than in the regeneration script so it can be tested. Both widenings were
    bugs in the keying, and a test that pins only the scope walker leaves the keying free to be
    simplified back with the suite still green.

    The scope of a FRESH entry -- its recorded line still holds its symbol -- is resolved
    against the current file, which is exact. A DRIFTED entry is resolved against the file as
    it stood at the last commit that touched the ledger itself: the ledger's numbers were
    correct when it was last written, so that snapshot is the one they index into. Resolving a
    drifted line against the current file instead -- which this once did -- returned whichever
    function had slid under the old number, so a single edit longer than a function turned
    every reviewed rationale below it into a placeholder at the next regeneration. When no
    snapshot exists to consult (no git, a never-committed ledger, staleness accumulated across
    uncommitted regenerations) the current file is still used, and a mapping it cannot make
    surfaces as a placeholder rather than as a wrong rationale.
    """
    scopes_by_path: dict[str, dict[int, str]] = {}
    lines_by_path: dict[str, list[str]] = {}
    recorded_scopes_by_path: dict[str, dict[int, str] | None] = {}
    found: dict[tuple[str, str, str, int], str] = {}
    seen: Counter[tuple[str, str, str]] = Counter()
    rationale: str | None = None
    for raw in exemptions_path.read_text().split("\n"):
        line = raw.strip()
        if line.startswith(_RATIONALE_PREFIX):
            rationale = line[len(_RATIONALE_PREFIX) :].strip()
            continue
        if not line or line.startswith("#"):
            continue
        path, _, rest = line.partition(":")
        number, _, symbol = rest.partition(":")
        if not number.isdigit() or not rationale:
            continue
        source = repo_root / path
        if path not in scopes_by_path:
            text = source.read_text(errors="replace") if source.exists() else ""
            scopes_by_path[path] = _scopes_from_source(text)
            lines_by_path[path] = text.split("\n")
        entry_line = int(number)
        lines = lines_by_path[path]
        drifted = entry_line > len(lines) or symbol not in lines[entry_line - 1]
        scopes = scopes_by_path[path]
        if drifted:
            if path not in recorded_scopes_by_path:
                recorded = _source_when_ledger_was_written(exemptions_path, repo_root, path)
                recorded_scopes_by_path[path] = _scopes_from_source(recorded) if recorded is not None else None
            scopes = recorded_scopes_by_path[path] or scopes
        group = (path, scopes.get(entry_line, ""), symbol)
        found.setdefault((*group, seen[group]), rationale)
        seen[group] += 1
    return found


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
