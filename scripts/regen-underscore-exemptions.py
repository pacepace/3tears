#!/usr/bin/env python3
"""Rewrite `tests/enforcement/_underscore_exemptions.txt` from the code it describes.

The ledger is `path:line:symbol` triples, so it goes stale whenever a file is edited above an
entry -- which happens constantly and has nothing to do with the access itself. Two enforcement
tests now fail when it does, in both directions, so regeneration is routine maintenance rather
than a rescue operation and deserves one command instead of an ad-hoc script each time.

Rationales are carried forward by `(path, enclosing scope, symbol)`, so the reasoning survives a
line shift and
only a genuinely new access needs new text. Any access it cannot map is reported and given a
placeholder, so it is visible rather than silently templated.

Discovery and AST walking come from `threetears.enforcement.underscore_access`, the same
canonical domain the enforcement tests are thin shells over, so this script and the checks that
police its output cannot answer "which files are exempted" differently. That discovery reads
EVERY ruff config, not just the root `pyproject.toml` -- reading only the root is how a set of
reviewed sidecar entries were deleted without anything noticing, since a nested `ruff.toml` is a
full override the root cannot reach past.

    uv run python scripts/regen-underscore-exemptions.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from threetears.enforcement.underscore_access import (
    all_exempted_files,
    carry_forward_rationales,
    enclosing_scopes,
    private_accesses,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

_LEDGER = _REPO_ROOT / "tests" / "enforcement" / "_underscore_exemptions.txt"
_PLACEHOLDER = "TODO: no rationale carried forward -- write why this access is acceptable"


def _header() -> list[str]:
    """The file's leading commentary, stopping BEFORE the first entry's rationale.

    Stopping at the first entry is not the same thing, and the difference compounds: a
    `# rationale:` line is a comment, so it was captured as header AND re-emitted with its
    entry, leaving one orphan rationale at the top of the file per run. Three had accumulated
    before anything noticed, because nothing pairs a rationale with an entry.
    """
    lines: list[str] = []
    for raw in _LEDGER.read_text().split("\n"):
        if raw.strip().startswith("# rationale:"):
            break
        if raw.strip().startswith("#") or not raw.strip():
            lines.append(raw)
        else:
            break
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def main() -> int:
    rationales = carry_forward_rationales(_LEDGER, _REPO_ROOT)
    paths = all_exempted_files(_REPO_ROOT)

    out = _header()
    unmapped: list[str] = []
    for source in paths:
        rel = source.relative_to(_REPO_ROOT).as_posix()
        accesses = sorted(private_accesses(source))
        if not accesses:
            continue
        scopes = enclosing_scopes(source)
        out.append("")
        # Counted the same way `carry_forward_rationales` counts, and in the same order, so the
        # nth access of one private inside one function keeps ITS OWN rationale rather than a copy
        # of the first one's. A test reading a value and then asserting on it is two accesses of
        # one name in one scope, which is the commonest shape there is.
        seen: Counter[tuple[str, str]] = Counter()
        for number, symbol in accesses:
            scope = scopes.get(number, "")
            reason = rationales.get((rel, scope, symbol, seen[scope, symbol]))
            seen[scope, symbol] += 1
            if reason is None:
                unmapped.append(f"{rel}:{number}:{symbol}")
                reason = _PLACEHOLDER
            out.append(f"# rationale: {reason}")
            out.append(f"{rel}:{number}:{symbol}")

    _LEDGER.write_text("\n".join(out) + "\n")
    entries = sum(1 for line in out if line and not line.startswith("#"))
    print(f"wrote {entries} entries across {len(paths)} exempted files")
    if unmapped:
        print(f"\n{len(unmapped)} access(es) had no rationale to carry forward -- write one for each:")
        for item in unmapped:
            print(f"  {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
