"""3tears' serving code, as the dict-state and await-race checks walk it: every package's source, less the trees
that are not serving code (each with its why; Pace approved the three below). One list, so the two checks never
disagree about what serves.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["NON_SERVING", "NON_SERVING_ROOTS", "REPO_ROOT", "SRC_ROOTS"]

REPO_ROOT = Path(__file__).resolve().parents[2]

#: every package's source: a package added later is walked without anyone remembering to list it
SRC_ROOTS: tuple[Path, ...] = tuple(
    sorted(
        root
        for pattern in ("packages/*/src", "packages/agent/*/src")
        for root in REPO_ROOT.glob(pattern)
        if root.is_dir()
    )
)

#: the code that is not serving code, each with its why
NON_SERVING: tuple[tuple[str, str], ...] = (
    (
        "packages/enforcement",
        "the rule engine: run by pytest in each repo's enforcement suite, never imported by a serving process",
    ),
    (
        "packages/core/src/threetears/core/testing",
        "test helpers (fakes, fixtures, harnesses) imported only by test suites",
    ),
    (
        "packages/evals",
        "the evaluation harness: run offline against recorded cases, never part of a serving pod",
    ),
)

NON_SERVING_ROOTS: tuple[Path, ...] = tuple(REPO_ROOT / path for path, _why in NON_SERVING)
