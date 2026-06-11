"""thin shell -- actual logic in
:mod:`threetears.enforcement.integration_skill`.

the package-shipped integration skill
(``packages/core/src/threetears/core/SKILL.md``) is consumer-facing
guidance that ships in the ``3tears`` wheel. three contracts are
pinned here: the exact sentinel disclaimer opens the body, every
``threetears.*`` import in its fenced code blocks resolves, and every
component section's public-surface attestation is current. surface
drift fails with instructions to re-verify the section and re-attest
(``python -m threetears.enforcement.integration_skill --update <skill_path>``).

design rationale: ``docs/experiments/integration-skill/DESIGN.md``.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.integration_skill import (
    IntegrationSkillConfig,
    run_integration_skill_enforcement,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_CONFIG = IntegrationSkillConfig(
    skill_path=_REPO_ROOT / "packages" / "core" / "src" / "threetears" / "core" / "SKILL.md",
)


class TestIntegrationSkillIntegrity:
    """the shipped SKILL.md is sentinel-marked, import-true, and attested."""

    def test_skill_integrity(self) -> None:
        """sentinel byte-match + referential integrity + surface attestation."""
        run_integration_skill_enforcement(_CONFIG)
