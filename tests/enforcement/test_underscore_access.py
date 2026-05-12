"""thin shell — actual walker logic in
:mod:`threetears.enforcement.underscore_access`.

the underscore-access domain is universal: there are no per-repo
allowlists, dictionaries, or configuration knobs. the consumer
declares only the repo root and the exemptions file path; every
shape walker (A through E), the rationale-required exemption
parser, the mode resolver, the report emitter, and the ruff
shell-out for shape B all live in the package.

the test class / method names preserve the canonical shape so any
external CI looking for them continues to find them.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.underscore_access import (
    UnderscoreAccessConfig,
    run_underscore_enforcement,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


_CONFIG = UnderscoreAccessConfig(
    repo_root=_REPO_ROOT,
    exemptions_path=_REPO_ROOT / "tests" / "enforcement" / "_underscore_exemptions.txt",
)


class TestUnderscoreAccess:
    """aggregate test: five shapes, one assertion, exemptions applied."""

    def test_no_underscore_violations(self) -> None:
        """every private access crosses a public API boundary or is exempted."""
        run_underscore_enforcement(_CONFIG, walker="all")
