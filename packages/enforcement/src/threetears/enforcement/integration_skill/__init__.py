"""integration-skill integrity enforcement domain.

the package-shipped integration skill (``threetears/core/SKILL.md``)
is third-party-authored guidance for *consuming* agents. guidance that
lies is worse than no guidance, so this domain pins three contracts:

- the document opens with the exact :data:`SENTINEL` disclaimer line
  (a format marker for harnesses, not a security boundary -- see
  ``docs/experiments/integration-skill/DESIGN.md``);
- every ``threetears.*`` import in its fenced code blocks resolves
  against the installed workspace (referential integrity);
- every component section is attested against a public-surface
  fingerprint; surface drift fails CI until a human re-verifies the
  section and re-attests via
  ``python -m threetears.enforcement.integration_skill --update``.

per-repo configuration goes through :class:`IntegrationSkillConfig`;
:func:`run_integration_skill_enforcement` is the pytest-friendly
entry point.
"""

from threetears.enforcement.integration_skill.runner import (
    IntegrationSkillConfig,
    run_integration_skill_enforcement,
)
from threetears.enforcement.integration_skill.skill_doc import (
    SENTINEL,
    AttestedSurface,
    SkillDocument,
    SkillParseError,
    parse_skill_document,
)
from threetears.enforcement.integration_skill.surface import (
    COMPONENT_DISTRIBUTIONS,
    compute_surface_fingerprint,
    resolve_component_dir,
)

__all__ = [
    "COMPONENT_DISTRIBUTIONS",
    "SENTINEL",
    "AttestedSurface",
    "IntegrationSkillConfig",
    "SkillDocument",
    "SkillParseError",
    "compute_surface_fingerprint",
    "parse_skill_document",
    "resolve_component_dir",
    "run_integration_skill_enforcement",
]
