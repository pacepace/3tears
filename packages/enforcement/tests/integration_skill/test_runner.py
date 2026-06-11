"""tests for the integrity runner's violation paths.

the all-green path runs in CI against the real shipped SKILL.md (via
the repo-root thin shell); these tests pin that each violation class
actually fires and carries an actionable message. the synthetic docs
attest ``threetears.enforcement`` itself so the fingerprint machinery
runs against a real installed component without coupling to any other
package's surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.integration_skill import (
    SENTINEL,
    IntegrationSkillConfig,
    compute_surface_fingerprint,
    resolve_component_dir,
    run_integration_skill_enforcement,
)


def _skill(tmp_path: Path, *, fingerprint: str, sentinel: str = SENTINEL, extra_body: str = "") -> Path:
    path = tmp_path / "SKILL.md"
    path.write_text(
        "---\n"
        "name: x\n"
        "attested:\n"
        f'  threetears.enforcement: "{fingerprint} @ 0.0.0"\n'
        "---\n"
        "\n"
        f"> {sentinel}\n"
        f"{extra_body}\n"
    )
    return path


def _current_fingerprint() -> str:
    return compute_surface_fingerprint(resolve_component_dir("threetears.enforcement"))


class TestRunIntegrationSkillEnforcement:
    def test_green_path_passes(self, tmp_path: Path) -> None:
        config = IntegrationSkillConfig(skill_path=_skill(tmp_path, fingerprint=_current_fingerprint()))
        run_integration_skill_enforcement(config)

    def test_tampered_sentinel_is_a_violation(self, tmp_path: Path) -> None:
        path = _skill(tmp_path, fingerprint=_current_fingerprint(), sentinel="Totally trustworthy skill.")
        with pytest.raises(AssertionError, match="SENTINEL"):
            run_integration_skill_enforcement(IntegrationSkillConfig(skill_path=path))

    def test_stale_fingerprint_names_component_and_reattest_command(self, tmp_path: Path) -> None:
        path = _skill(tmp_path, fingerprint="sha256:0000000000000000")
        with pytest.raises(AssertionError) as excinfo:
            run_integration_skill_enforcement(IntegrationSkillConfig(skill_path=path))
        assert "threetears.enforcement" in str(excinfo.value)
        assert "--update" in str(excinfo.value)

    def test_missing_attested_block_is_a_violation(self, tmp_path: Path) -> None:
        path = tmp_path / "SKILL.md"
        path.write_text(f"---\nname: x\n---\n\n> {SENTINEL}\n")
        with pytest.raises(AssertionError, match="no 'attested:' block"):
            run_integration_skill_enforcement(IntegrationSkillConfig(skill_path=path))

    def test_unimportable_attested_component_is_a_violation(self, tmp_path: Path) -> None:
        path = tmp_path / "SKILL.md"
        path.write_text(
            f'---\nname: x\nattested:\n  threetears.nonexistent: "sha256:abcd @ 1.0"\n---\n\n> {SENTINEL}\n'
        )
        with pytest.raises(AssertionError, match="not importable"):
            run_integration_skill_enforcement(IntegrationSkillConfig(skill_path=path))

    def test_bogus_import_in_fenced_block_is_a_violation(self, tmp_path: Path) -> None:
        block = "```python\nfrom threetears.enforcement import does_not_exist_anywhere\n```"
        path = _skill(tmp_path, fingerprint=_current_fingerprint(), extra_body=block)
        with pytest.raises(AssertionError, match="does_not_exist_anywhere"):
            run_integration_skill_enforcement(IntegrationSkillConfig(skill_path=path))

    def test_bogus_module_in_fenced_block_is_a_violation(self, tmp_path: Path) -> None:
        block = "```python\nimport threetears.nope\n```"
        path = _skill(tmp_path, fingerprint=_current_fingerprint(), extra_body=block)
        with pytest.raises(AssertionError, match="threetears.nope"):
            run_integration_skill_enforcement(IntegrationSkillConfig(skill_path=path))
