"""pytest-friendly runner for integration-skill integrity checks."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from threetears.enforcement.integration_skill.skill_doc import parse_skill_document
from threetears.enforcement.integration_skill.surface import (
    compute_surface_fingerprint,
    resolve_component_dir,
)

__all__ = [
    "IntegrationSkillConfig",
    "run_integration_skill_enforcement",
]


@dataclass(frozen=True)
class IntegrationSkillConfig:
    """per-repo configuration.

    :ivar skill_path: the SKILL.md under enforcement
    """

    skill_path: Path


def _sentinel_violations(config: IntegrationSkillConfig) -> list[str]:
    doc = parse_skill_document(config.skill_path)
    if doc.opens_with_sentinel:
        return []
    return [
        f"{config.skill_path}: body must open with the exact SENTINEL disclaimer "
        "(threetears.enforcement.integration_skill.SENTINEL)"
    ]


def _attestation_violations(config: IntegrationSkillConfig) -> list[str]:
    doc = parse_skill_document(config.skill_path)
    violations: list[str] = []
    if not doc.attested:
        return [f"{config.skill_path}: frontmatter has no 'attested:' block"]
    for component, attested in sorted(doc.attested.items()):
        computed = compute_surface_fingerprint(resolve_component_dir(component))
        if computed != attested.fingerprint:
            violations.append(
                f"public surface of {component} changed since attestation "
                f"(attested {attested.fingerprint}, computed {computed}). "
                f"re-verify that component's section of {config.skill_path.name}, then re-attest: "
                "uv run python -m threetears.enforcement.integration_skill --update "
                f"{config.skill_path}"
            )
    return violations


def _import_violations(config: IntegrationSkillConfig) -> list[str]:
    doc = parse_skill_document(config.skill_path)
    violations: list[str] = []
    for module_name, symbol in doc.threetears_imports():
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            violations.append(f"skill imports {module_name!r} which does not import: {exc}")
            continue
        if symbol is not None and symbol != "*" and not hasattr(module, symbol):
            violations.append(f"skill imports {symbol!r} from {module_name!r} but it does not exist")
    return violations


def run_integration_skill_enforcement(config: IntegrationSkillConfig) -> None:
    """run all three integrity checks; raise on any violation.

    :param config: per-repo configuration
    :ptype config: IntegrationSkillConfig
    :raises AssertionError: listing every violation found
    """
    violations = _sentinel_violations(config) + _attestation_violations(config) + _import_violations(config)
    if violations:
        details = "\n  - ".join(violations)
        raise AssertionError(f"integration-skill integrity violations:\n  - {details}")
