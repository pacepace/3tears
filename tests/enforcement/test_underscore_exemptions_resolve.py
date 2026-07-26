"""Every exemption names code that exists, and every exempted access is named.

``_underscore_exemptions.txt`` is a list of ``path:line:symbol`` triples, each with a rationale
recording why one private access was judged acceptable. Nothing verified either half of that
until now, and both had rotted:

- **Stale.** 54 of 122 entries pointed at code that had moved or gone -- 49 at a line no longer
  containing the symbol, 5 at a file under a ``packages/agent-tools/`` directory that no longer
  exists. A stale exemption is worse than a missing one: it reads as a reviewed decision about a
  specific access, so a reader trusts a rationale for code that is not there.
- **Missing.** Six accesses in the scrape suites had no entry at all.

Neither is caught by the underscore walker, which scans ``packages/*/src`` only and never enters
a ``tests/`` tree -- so for every exempted test file the ledger is documentation that nothing
reads back.

**Why this needs both directions.** A stale-entry check alone would have missed all six missing
ones, because a missing entry is not a stale entry. And the missing-direction check has to see
accesses that ruff would not report: an inline ``# noqa: SLF001`` on a per-file-ignored path
suppresses the finding, so the access would reach neither ruff nor this ledger. This module and
``scripts/regen-underscore-exemptions.py`` both walk the AST, so neither inherits that blind
spot; ``test_no_redundant_slf001_pragmas.py`` keeps the pragmas out regardless.
"""

from __future__ import annotations

import ast
from pathlib import Path

from _ruff_config_discovery import exempted_files, ruff_configs, slf001_globs

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXEMPTIONS = _REPO_ROOT / "tests" / "enforcement" / "_underscore_exemptions.txt"


def _entries() -> list[tuple[str, int, str]]:
    """Every ``path:line:symbol`` triple in the ledger, comments and blanks dropped."""
    found: list[tuple[str, int, str]] = []
    for raw in _EXEMPTIONS.read_text().split("\n"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        path, _, rest = line.partition(":")
        number, _, symbol = rest.partition(":")
        if path and number.isdigit() and symbol:
            found.append((path, int(number), symbol))
    return found


def _accesses(path: Path) -> set[tuple[int, str]]:
    """Every ``obj._private`` read in *path* that is not ``self``/``cls``.

    AST rather than ruff, deliberately: ruff honours an inline ``noqa`` and would report nothing
    for exactly the accesses that go missing from the ledger. Dunders are excluded because
    ``__init__`` and friends are public protocol, not private state.
    """
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:  # a file this repo cannot parse is not this test's business
        return set()

    found: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not node.attr.startswith("_") or node.attr.startswith("__"):
            continue
        if isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"}:
            continue
        found.add((node.lineno, node.attr))
    return found


def _exempted_paths() -> list[Path]:
    """Every file carrying a per-file ``SLF001`` ignore, from every config ruff reads.

    Not just the root ``pyproject.toml``, and not just ``/tests/`` paths. Reading only the root
    is exactly how five reviewed sidecar entries were deleted without anything noticing: the
    nested ``packages/scrape/sidecar/ruff.toml`` is a full override, so a checker built on the
    root config cannot see what it governs -- and the underscore walker never enters the
    sidecar either, since it scans ``packages/*/src``.

    Filtering to ``/tests/`` had the same shape: `sidecar/hitl.py` and `main.py` are production
    files that carry per-file ignores, and their accesses are precisely the kind the ledger
    exists to record.
    """
    found: list[Path] = []
    for config in ruff_configs(_REPO_ROOT):
        for pattern in slf001_globs(config):
            found.extend(exempted_files(config, pattern, _REPO_ROOT))
    return sorted(set(found))


class TestUnderscoreExemptionsResolve:
    def test_every_rationale_is_attached_to_an_entry(self) -> None:
        """The file's own "each entry requires a preceding rationale" rule, enforced.

        It was enforced by nothing, and that gap let a bug in the regeneration script
        accumulate silently: `_header()` stopped at the first ENTRY rather than the first
        rationale, so a rationale line was captured as header and re-emitted with its entry,
        leaving one orphan at the top of the file per run. Three had built up. Neither
        direction of the resolve check pairs a rationale with an entry, so neither could see it.
        """
        lines = _EXEMPTIONS.read_text().split("\n")
        orphans: list[str] = []
        for number, raw in enumerate(lines, 1):
            if not raw.startswith("# rationale:"):
                continue
            following = lines[number] if number < len(lines) else ""
            if not following.strip() or following.startswith("#"):
                orphans.append(f"line {number}")

        assert not orphans, (
            "these rationale lines are not followed by the entry they justify, so they document "
            f"nothing and accumulate on every regeneration: {orphans}"
        )

    def test_every_entry_names_a_file_that_exists(self) -> None:
        """Five entries named a directory deleted from the tree entirely."""
        missing = sorted({path for path, _, _ in _entries() if not (_REPO_ROOT / path).exists()})

        assert not missing, (
            f"these exemptions name files that do not exist, so they document decisions about "
            f"code that is gone: {missing}"
        )

    def test_every_entry_resolves_to_the_symbol_it_claims(self) -> None:
        """A triple whose line no longer holds that symbol is a rationale for the wrong code."""
        unresolved: list[str] = []
        for path, number, symbol in _entries():
            source = _REPO_ROOT / path
            if not source.exists():
                continue  # reported by the test above; not double-counted here
            lines = source.read_text(errors="replace").split("\n")
            if number > len(lines) or symbol not in lines[number - 1]:
                unresolved.append(f"{path}:{number}:{symbol}")

        assert not unresolved, (
            "these exemptions point at a line that no longer contains the symbol they name. "
            "Regenerate that path's entries -- see the procedure in the exemptions file header "
            f"-- rather than editing the numbers by hand: {unresolved}"
        )

    def test_every_exempted_access_has_an_entry(self) -> None:
        """The other direction, which a stale-entry check structurally cannot cover.

        A missing entry is not a stale entry: it points at nothing because it does not exist.
        Six went missing behind inline pragmas before anything looked for them.
        """
        entries = {(path, number, symbol) for path, number, symbol in _entries()}
        unlisted: list[str] = []
        for source in _exempted_paths():
            rel = source.relative_to(_REPO_ROOT).as_posix()
            for number, symbol in sorted(_accesses(source)):
                if (rel, number, symbol) not in entries:
                    unlisted.append(f"{rel}:{number}:{symbol}")

        assert not unlisted, (
            "these private accesses sit on a per-file-exempted path and have no ledger entry, "
            f"so nothing records why they were judged acceptable: {unlisted}"
        )
