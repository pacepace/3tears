"""A path with a per-file ``SLF001`` ignore must not also carry inline ``noqa: SLF001``.

The pragma is redundant to ruff -- the per-file ignore already covers it -- so on its own this
would be a tidiness rule not worth a test. It is here because of what the redundancy does to
the exemptions ledger.

``tests/enforcement/_underscore_exemptions.txt`` is maintained by re-running ruff over an
exempted path and rewriting that path's entries from the output. Ruff honours an inline
``noqa`` even under ``--isolated``, so any access carrying one is reported by nothing and
silently never reaches the ledger. Six accesses in the scrape suites went missing exactly that
way, and the loss is invisible from both ends: the gate passes (the underscore walker scans
``packages/*/src`` only and never enters a ``tests/`` tree), and the regeneration command
produces a short list without complaining.

That is why this guards the ledger rather than style. A bidirectional consistency check --
every entry resolves, every flagged access has an entry -- cannot catch it either, because the
blinded access is never flagged in the first place. This has to be checked at the source.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INLINE_SLF001 = re.compile(r"#\s*noqa:\s*SLF001(?![\w-])")


def _paths_with_per_file_slf001_ignore() -> list[str]:
    config = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    per_file = config["tool"]["ruff"]["lint"]["per-file-ignores"]
    return [path for path, codes in per_file.items() if "SLF001" in codes]


class TestNoRedundantSlf001Pragmas:
    def test_exempted_paths_carry_no_inline_pragmas(self) -> None:
        """Named per offender, so the failure says which file and which line to fix."""
        offenders: list[str] = []
        for rel in _paths_with_per_file_slf001_ignore():
            path = _REPO_ROOT / rel
            if not path.exists():
                # A per-file ignore for a deleted path is its own defect, tracked separately;
                # this test is about live files and should not fail for that reason.
                continue
            for number, line in enumerate(path.read_text(errors="replace").split("\n"), 1):
                if _INLINE_SLF001.search(line):
                    offenders.append(f"{rel}:{number}")

        assert not offenders, (
            "these lines carry an inline `noqa: SLF001` on a path that already has a per-file "
            "ignore. The pragma is redundant to ruff AND hides the access from the exemptions "
            f"ledger regeneration, so it silently drops out of the record: {offenders}"
        )
