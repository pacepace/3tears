#!/usr/bin/env python3
"""Rewrite `tests/enforcement/_underscore_exemptions.txt` from the code it describes.

The ledger is `path:line:symbol` triples, so it goes stale whenever a file is edited above an
entry -- which happens constantly and has nothing to do with the access itself. Two enforcement
tests now fail when it does, in both directions, so regeneration is routine maintenance rather
than a rescue operation and deserves one command instead of an ad-hoc script each time.

Rationales are carried forward by `(path, symbol)`, so the reasoning survives a line shift and
only a genuinely new access needs new text. Any access it cannot map is reported and given a
placeholder, so it is visible rather than silently templated.

Discovery is shared with the enforcement tests (`tests/enforcement/_ruff_config_discovery.py`)
and reads EVERY ruff config, not just the root `pyproject.toml`. Reading only the root is how
five reviewed sidecar entries were deleted without anything noticing: the nested
`packages/scrape/sidecar/ruff.toml` is a full override the root cannot reach past.

    uv run python scripts/regen-underscore-exemptions.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "tests" / "enforcement"))

from _ruff_config_discovery import exempted_files, ruff_configs, slf001_globs  # noqa: E402

_LEDGER = _REPO_ROOT / "tests" / "enforcement" / "_underscore_exemptions.txt"
_PLACEHOLDER = "TODO: no rationale carried forward -- write why this access is acceptable"


def _existing_rationales() -> dict[tuple[str, str], str]:
    """Map ``(path, symbol)`` to its recorded rationale, so a line shift loses nothing."""
    found: dict[tuple[str, str], str] = {}
    rationale: str | None = None
    for raw in _LEDGER.read_text().split("\n"):
        line = raw.strip()
        if line.startswith("# rationale:"):
            rationale = line[len("# rationale:") :].strip()
            continue
        if not line or line.startswith("#"):
            continue
        path, _, rest = line.partition(":")
        number, _, symbol = rest.partition(":")
        if number.isdigit() and rationale:
            found.setdefault((path, symbol), rationale)
    return found


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


def _accesses(path: Path) -> set[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return set()
    found: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not node.attr.startswith("_") or node.attr.startswith("__"):
            continue
        if isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"}:
            continue
        found.add((node.lineno, node.attr))
    return found


def main() -> int:
    rationales = _existing_rationales()
    paths = sorted(
        {p for c in ruff_configs(_REPO_ROOT) for g in slf001_globs(c) for p in exempted_files(c, g, _REPO_ROOT)}
    )

    out = _header()
    unmapped: list[str] = []
    for source in paths:
        rel = source.relative_to(_REPO_ROOT).as_posix()
        accesses = sorted(_accesses(source))
        if not accesses:
            continue
        out.append("")
        for number, symbol in accesses:
            reason = rationales.get((rel, symbol))
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
