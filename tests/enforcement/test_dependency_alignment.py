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
    DependencyFloor,
    run_dependency_alignment_enforcement,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: packages whose hard-dependency list is an argued ruling, not an accretion.
#:
#: ``media-contracts`` needs no entry: its floor is *nothing*, which
#: ``contract_packages`` above already states more strongly. This list is for the
#: packages that cannot be dependency-free but whose floor was still ruled -- where
#: "nothing" is unavailable and "these three" is the promise instead.
_DEPENDENCY_FLOORS = (
    DependencyFloor(
        package="packages/search",
        allowed=("3tears-media-contracts", "3tears-observe", "pydantic"),
        rationale=(
            "search-spec.md D24, matching SR-L7's permitted floor exactly. The leaf's whole "
            "value to a constrained host (samsung on a Pi; any embedded consumer) is that "
            "taking it costs pydantic plus two dependency-free family leaves and nothing "
            "else -- so httpx and trafilatura ride the [standalone] and [extract] extras, and "
            "the provider adapters, being pure logic over an injected transport, add no "
            "weight at all. A fourth hard dependency arriving here is exactly the drift that "
            "makes a leaf untakeable, and it would arrive one innocent import at a time"
        ),
    ),
)

_CONFIG = DependencyAlignmentConfig(
    repo_root=_REPO_ROOT,
    package_globs=("packages/*", "packages/agent/*"),
    exemptions_path=_REPO_ROOT / "tests" / "enforcement" / "_dependency_alignment_exemptions.txt",
    contract_packages=("packages/media-contracts",),
    dependency_floors=_DEPENDENCY_FLOORS,
)


class TestDependencyAlignment:
    """declared deps match actual imports for every workspace package."""

    def test_declared_dependencies_match_actual_imports(self) -> None:
        """no undeclared module-top imports; no stale 3tears declarations."""
        run_dependency_alignment_enforcement(_CONFIG, walker="alignment")

    def test_contracts_packages_stay_dependency_free(self) -> None:
        """contracts packages import only stdlib + their own namespace."""
        run_dependency_alignment_enforcement(_CONFIG, walker="contract_purity")

    def test_floor_pinned_packages_declare_exactly_their_ruled_floor(self) -> None:
        """a pinned package's hard deps are the ruled list -- no more, no fewer."""
        run_dependency_alignment_enforcement(_CONFIG, walker="dependency_floor")

    def test_every_floor_pin_names_a_real_package(self) -> None:
        """guard the guard: a pin over a path that holds nothing passes vacuously,
        so the pinned directories are asserted to exist as packages here rather
        than only inside the walker's own report."""
        for floor in _DEPENDENCY_FLOORS:
            assert (_REPO_ROOT / floor.package / "pyproject.toml").is_file(), floor.package
            assert floor.allowed, f"{floor.package} pins an empty floor"
