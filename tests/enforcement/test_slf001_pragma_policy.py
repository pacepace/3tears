"""thin shell -- the policy lives in :mod:`threetears.enforcement.underscore_access.pragma_policy`.

No inline ``# noqa: SLF001`` anywhere in the repo, and a per-file SLF001 ignore only on a ``src``
module that confines a third-party library's private members and has an entry in
``_underscore_exemptions.txt``. Owner ruling, 2026-10-01; ``CLAUDE.md`` ("Leading underscores are
a stability contract").

The floors below are this repo's, which is why they live here rather than in the package: a scan
of nothing and a config read of nothing both report what a compliant repo reports.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.underscore_access import (
    scanned_python_files,
    slf001_ignored_files,
    slf001_policy_findings,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LEDGER = _REPO_ROOT / "tests" / "enforcement" / "_underscore_exemptions.txt"


class TestSlf001PragmaPolicy:
    def test_the_scan_covers_the_workspace(self) -> None:
        """non-vacuity on the pragma input: the file walk reaches every package, src and tests."""
        scanned = {path.relative_to(_REPO_ROOT).as_posix() for path in scanned_python_files(_REPO_ROOT)}

        assert len(scanned) > 1000, f"only {len(scanned)} python files scanned; discovery has collapsed"
        assert "packages/nats/src/threetears/nats/client.py" in scanned
        assert "packages/nats/tests/unit/test_client.py" in scanned
        assert not any(path.startswith(".venv/") for path in scanned), "the virtualenv was scanned as this repo"

    def test_the_config_read_finds_the_confinement_modules(self) -> None:
        """non-vacuity on the ignore input: the two third-party confinement modules are read back."""
        covered = {path for _config, _key, path in slf001_ignored_files(_REPO_ROOT)}

        assert "packages/nats/src/threetears/nats/_nats_py_internals.py" in covered
        assert "packages/observe/src/threetears/observe/_otel_internals.py" in covered

    def test_no_slf001_suppression_outside_a_recorded_src_module(self) -> None:
        findings = slf001_policy_findings(_REPO_ROOT, _LEDGER)

        assert not findings, f"{len(findings)} SLF001 suppression(s) outside policy:\n" + "\n".join(findings)
