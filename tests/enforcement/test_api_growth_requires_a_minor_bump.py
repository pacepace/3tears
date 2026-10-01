"""
enforcement: a package whose public API grew may not ship as a patch release.

**The rule this holds is a recorded product decision, not an inference.** Backlog
item BLD-7QM3, decided 2026-07-26: *an intra-family API addition requires a minor
bump.* Two other directions were considered there and rejected -- raising each
bound's floor to the release that carries the new API (more precise, but the
bound cannot be written until the release version is known, which puts a moving
part inside the very mechanism whose failure mode is an unresolvable family), and
guarding the imports to degrade (wrong wherever the seam is not optional).

**Why the rule exists.** Every intra-family dependency is bounded to a MINOR line
-- ``3tears>=0.24.0,<0.25.0`` -- which is what makes a mixed family unresolvable
rather than merely unlikely (``test_intra_family_version_bounds.py``). That bound's
floor is the minor. So when API is added inside a minor, every earlier patch in
the range satisfies the bound and lacks the symbol: pip resolves a family that
installs clean, builds clean, and raises ``ImportError`` on first use. v0.24.7
did exactly that, with every gate green.

**The gate itself lives in** :mod:`threetears.enforcement.release.growth`, the one
copy every aibots repository calls too; this file is 3tears' configuration of it.
What it counts, what it refuses to read as empty, and what it cannot see are
recorded there.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.release import ApiGrowthConfig, api_growth_findings, untagged_checkout_findings

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: every workspace member's source tree, per the root pyproject's workspace globs.
_SOURCE_ROOTS = tuple(
    sorted(
        path.relative_to(_REPO_ROOT).as_posix()
        for pattern in ("packages/*/src", "packages/agent/*/src")
        for path in _REPO_ROOT.glob(pattern)
        if (path.parent / "pyproject.toml").is_file()
    )
)

_CONFIG = ApiGrowthConfig(
    repo_root=_REPO_ROOT,
    source_roots=_SOURCE_ROOTS,
    first_release=(0, 1, 0),
    known_exports=(
        ("packages/core/src/threetears/core/collections/__init__.py", "BaseCollection"),
        ("packages/enforcement/src/threetears/enforcement/release/__init__.py", "api_growth_findings"),
    ),
    version_file="packages/core/pyproject.toml",
)


class TestApiGrowthRequiresAMinorBump:
    """A patch release may fix things. It may not add things a consumer can import."""

    def test_the_workspace_declares_an_honest_version(self) -> None:
        """
        no growth on the latest release's minor line, and every module's surface readable.

        :return: nothing
        :rtype: None
        """
        assert len(_SOURCE_ROOTS) > 20, f"the workspace globs found only {_SOURCE_ROOTS}"
        findings = api_growth_findings(_CONFIG)
        assert not findings, "\n\n".join(findings)

    def test_every_ci_job_running_this_gate_sees_the_tags(self) -> None:
        """
        a job running ``tests/`` checks this repo out with ``fetch-depth: 0``, or the gate refuses there.

        :return: nothing
        :rtype: None
        """
        findings = untagged_checkout_findings(_REPO_ROOT / ".github" / "workflows", "pytest packages/ tests/")
        assert not findings, "\n".join(findings)
