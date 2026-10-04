"""The retired "co-running is the cleanest design" claim stays retired.

The arm is the unit: a run carries exactly one arm, the candidate model among its settings. The
cleanest design was once said to be two arms co-run inside one run, and that claim lived in
docstrings, prompt rules and test names across the engine. A sweep removed it; a sweep is only as
good as the next person who does not re-derive the old claim from an old comment, so the sweep is a
test.

**The shape of the search, so an absence here is read for what it is:** case-insensitive substring
matches of the phrases below over ``*.py`` under this package's ``src/`` and ``tests/``. It cannot see a
paraphrase. A
browser kit that renders the engine's reports is scanned on the same terms where it ships.
"""

from __future__ import annotations

from pathlib import Path


#: The package root: its ``src`` and ``tests`` trees are what is scanned.
_REPO = Path(__file__).resolve().parents[1]

#: Phrasings of the retired claim.
_RETIRED = (
    "co-running is",
    "co-running two arms",
    "two arms inside one run",
    "two arms co-run",
    "co-run inside one run",
    "cleanest comparison the bundle",
    "within-run arm",
    "within-run a/b",
)

#: Lines allowed to carry a retired phrase, as (path, a fragment of the line), and why. Each is a
#: record of what the code USED to do, which is the one place the old claim belongs.
_HISTORY: dict[tuple[str, str], str] = {}

#: This file names every phrase in order to forbid it.
_SELF = Path(__file__).resolve().relative_to(_REPO).as_posix()


def _sources() -> list[Path]:
    """Every scanned source file, build output excluded.

    Returns:
        The paths, sorted.
    """
    found: list[Path] = []
    for tree, globs in (
        (_REPO / "src", ("*.py",)),
        (_REPO / "tests", ("*.py",)),
    ):
        for pattern in globs:
            found.extend(
                path
                for path in tree.rglob(pattern)
                if "node_modules" not in path.parts and "dist" not in path.parts and "__pycache__" not in path.parts
            )
    return sorted(found)


def _hits() -> list[tuple[str, str]]:
    """Every (path, line) carrying a retired phrase, this file excluded.

    Returns:
        The hits, in path order.
    """
    hits: list[tuple[str, str]] = []
    for path in _sources():
        relative = path.relative_to(_REPO).as_posix()
        if relative == _SELF:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            lowered = line.lower()
            if any(phrase in lowered for phrase in _RETIRED):
                hits.append((relative, line.strip()))
    return hits


def test_no_source_repeats_the_retired_co_run_claim() -> None:
    """A run is one arm; nothing live says two arms in one run are the cleanest design."""
    offenders = [
        f"{path}: {line}"
        for path, line in _hits()
        if not any(path == allowed_path and fragment in line for allowed_path, fragment in _HISTORY)
    ]

    assert offenders == [], (
        "the arm is the unit: a run carries exactly one arm, so these repeat a retired claim —\n" + "\n".join(offenders)
    )


def test_every_history_exemption_still_matches_a_line() -> None:
    """An exemption whose line has gone is a stale hole the next reintroduction could hide behind."""
    hits = _hits()
    stale = [
        f"{path}: {fragment}"
        for path, fragment in _HISTORY
        if not any(hit_path == path and fragment in line for hit_path, line in hits)
    ]

    assert stale == [], f"history exemptions matching no line: {stale}"


def test_the_scan_reaches_both_trees() -> None:
    """A walk that silently found nothing would make the test above vacuous."""
    roots = {path.relative_to(_REPO).parts[0] for path in _sources()}

    assert {"src", "tests"} <= roots
