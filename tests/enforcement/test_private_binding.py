"""thin shell -- the rules live in :mod:`threetears.enforcement.underscore_access.private_binding`.

No private name or module bound by an import (shape G) or by a string -- ``patch``,
``patch.object``, ``monkeypatch.setattr``/``delattr``, ``mocker.spy``, ``import_module`` (shape H)
-- outside its owner, in src, tests and scripts alike. Owner ruling, 2026-10-01; ``CLAUDE.md``
("Leading underscores are a stability contract").

**Expected to fail until this repo's findings are fixed.** The gate landed with the findings it
reports still in the tree, on purpose: they are fixed in follow-up work rather than exempted, and
nothing here exempts any of them. ``test_no_private_binding_outside_its_owner`` is red until then;
the two non-vacuity tests are green.

The floors below are this repo's, which is why they live here rather than in the package: a scan
of nothing, and recognisers that matched nothing, both report what a compliant repo reports.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.underscore_access import (
    PrivateBindingScan,
    private_binding_findings,
    scan_private_bindings,
    undetected_planted_controls,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LEDGER = _REPO_ROOT / "tests" / "enforcement" / "_underscore_exemptions.txt"


@pytest.fixture(scope="module")
def scan() -> PrivateBindingScan:
    return scan_private_bindings(_REPO_ROOT, _LEDGER)


class TestPrivateBinding:
    def test_the_scan_covers_the_workspace(self, scan: PrivateBindingScan) -> None:
        """non-vacuity on both inputs: the files read, and the imports and binder calls recognised."""
        assert scan.files_scanned > 1500, f"only {scan.files_scanned} python files scanned; discovery has collapsed"
        assert scan.imports_examined > 12000, f"only {scan.imports_examined} imports read"
        assert scan.binding_calls_examined > 900, f"only {scan.binding_calls_examined} binder calls recognised"

    def test_every_planted_shape_is_detected(self, tmp_path: Path) -> None:
        """the positive control: the installed walker still reports one planted instance of each shape."""
        undetected = undetected_planted_controls(tmp_path)

        assert not undetected, f"the walker no longer detects: {undetected}"

    def test_no_private_binding_outside_its_owner(self, scan: PrivateBindingScan) -> None:
        findings = private_binding_findings(scan, _REPO_ROOT)

        assert not findings, f"{len(findings)} private binding(s) outside their owner:\n" + "\n".join(findings)
