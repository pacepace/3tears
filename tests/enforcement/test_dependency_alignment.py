"""thin shell — walker logic in :mod:`threetears.enforcement.dependency_alignment`.

the 3tears self-test consumes the shared ``3tears-enforcement``
workspace package and injects only the per-repo configuration. the
walkers, exemption parser, mode resolver, and report emitter live in
the package; this file declares the knobs and calls the runner.

mode is controlled by ``DEPENDENCY_ALIGNMENT_ENFORCEMENT_MODE`` —
defaults to ``strict``. catches the drift class where the uv workspace
masks undeclared (or stale) cross-package dependencies until a
standalone ``pip install`` of one package ImportErrors in a consumer.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.dependency_alignment import (
    DependencyAlignmentConfig,
    run_dependency_alignment_enforcement,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_CONFIG = DependencyAlignmentConfig(
    repo_root=_REPO_ROOT,
    package_globs=("packages/*", "packages/agent/*"),
    exemptions_path=_REPO_ROOT / "tests" / "enforcement" / "_dependency_alignment_exemptions.txt",
    contract_packages=("packages/media-contracts",),
)


class TestDependencyAlignment:
    """declared deps match actual imports for every workspace package."""

    def test_declared_dependencies_match_actual_imports(self) -> None:
        """no undeclared module-top imports; no stale 3tears declarations."""
        run_dependency_alignment_enforcement(_CONFIG, walker="alignment")

    def test_contracts_packages_stay_dependency_free(self) -> None:
        """contracts packages import only stdlib + their own namespace."""
        run_dependency_alignment_enforcement(_CONFIG, walker="contract_purity")
