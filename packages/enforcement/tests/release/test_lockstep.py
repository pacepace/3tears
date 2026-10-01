"""a lockstep workspace: every place the version lives moves, and ``verify`` sees each one.

The fixture has the shape of 3tears itself -- members one and two levels deep, a
nested deployable with its own version, smoke tests with the aliased assertion
that slipped past v0.10.2's first bump, a bake file, intra-family bounds, and a
3tears-style changelog -- configured through ``[tool.threetears-release]``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from packages.enforcement.tests.release._scratch import TODAY, commit_all, git, release

_CONFIG = """[tool.uv.workspace]
members = ["packages/*", "packages/agent/*"]

[tool.threetears-release]
version-file = "packages/core/pyproject.toml"
lockstep-files = ["packages/*/pyproject.toml", "packages/agent/*/pyproject.toml"]
smoke-tests = ["packages/*/tests/test_smoke.py", "packages/agent/*/tests/test_smoke.py"]
bake-file = "docker-bake.hcl"
family-prefix = "fam"
notes = "CHANGELOG.md"
notes-style = "3tears"
"""

_BAKE = """variable "VERSION" {
  default = "v0.1.0"
}

target "base" {
  contexts = {
    base = "docker-image://ghcr.io/example/base:v0.1.0"
  }
}
"""


def _member(name: str, version: str = "0.1.0", requires: str = "") -> str:
    """a member pyproject.

    :param name: package name
    :ptype name: str
    :param version: its version
    :ptype version: str
    :param requires: a dependency line, if any
    :ptype requires: str
    :return: the file
    :rtype: str
    """
    deps = f'dependencies = [\n    "{requires}",\n]\n' if requires else ""
    return f'[project]\nname = "{name}"\nversion = "{version}"\n{deps}'


@pytest.fixture
def workspace(scratch: Path) -> Path:
    """a committed, tagged lockstep workspace at 0.1.0.

    :param scratch: the scratch parent
    :ptype scratch: Path
    :return: the workspace root
    :rtype: Path
    """
    root = scratch / "family"
    files = {
        "pyproject.toml": _CONFIG,
        "packages/core/pyproject.toml": _member("fam"),
        "packages/core/tests/test_smoke.py": (
            "from fam import __version__\nfrom fam import __version__ as core_version\n\n"
            'def test_version() -> None:\n    assert __version__ == "0.1.0"\n    assert core_version == "0.1.0"\n'
        ),
        "packages/extra/pyproject.toml": _member("fam-extra", requires="fam[x]>=0.1.0,<0.2.0"),
        "packages/extra/sidecar/pyproject.toml": _member("sidecar", version="9.9.9"),
        "packages/agent/tools/pyproject.toml": _member("fam-agent-tools", requires="fam-extra>=0.1.0,<0.2.0"),
        "docker-bake.hcl": _BAKE,
        "CHANGELOG.md": "# Changelog\n\n## Unreleased\n\n### A new thing\n\n- it\n\n## v0.1.0 -- 2026-01-01\n\n- first\n",
        "uv.lock": 'version = 1\n\n[[package]]\nname = "fam"\nversion = "0.1.0"\nsource = { editable = "packages/core" }\n',
    }
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    git(root, "init", "-q", "-b", "main")
    commit_all(root, "initial")
    git(root, "tag", "-a", "v0.1.0", "-m", "v0.1.0")
    return root


def _drift(root: Path) -> None:
    """puts every lockstep target but the canonical version line out of step, and commits it.

    :param root: workspace root
    :ptype root: Path
    :return: nothing
    :rtype: None
    """
    replacements = {
        "packages/extra/pyproject.toml": [('version = "0.1.0"', 'version = "0.0.99"')],
        "packages/agent/tools/pyproject.toml": [(">=0.1.0,<0.2.0", ">=0.0.0,<0.1.0")],
        "packages/core/tests/test_smoke.py": [('core_version == "0.1.0"', 'core_version == "0.0.99"')],
        "docker-bake.hcl": [('"v0.1.0"', '"v0.0.99"'), (":v0.1.0", ":v0.0.99")],
    }
    for path, pairs in replacements.items():
        text = (root / path).read_text()
        for old, new in pairs:
            text = text.replace(old, new)
        (root / path).write_text(text)
    commit_all(root, "drift")


class TestVerify:
    """``verify`` reports every location that disagrees, and edits nothing."""

    def test_a_consistent_workspace_passes(self, workspace: Path) -> None:
        """
        everything at 0.1.0.

        :param workspace: the workspace
        :ptype workspace: Path
        :return: nothing
        :rtype: None
        """
        status, out, err = release(workspace, "verify", "0.1.0")
        assert status == 0, err
        assert "All version locations at 0.1.0." in out

    def test_each_kind_of_drift_is_named(self, workspace: Path) -> None:
        """
        member version, bound, aliased smoke assertion, bake default and bake image tag.

        :param workspace: the workspace
        :ptype workspace: Path
        :return: nothing
        :rtype: None
        """
        _drift(workspace)
        status, _, err = release(workspace, "verify", "0.1.0")
        assert status == 1
        assert "packages/extra/pyproject.toml: version 0.0.99, expected 0.1.0" in err
        assert '"fam-extra>=0.0.0,<0.1.0" (expected >=0.1.0,<0.2.0)' in err
        assert 'packages/core/tests/test_smoke.py: core_version == "0.0.99"' in err
        assert "docker-bake.hcl VERSION: 0.0.99" in err
        assert "docker-bake.hcl image-tag :v0.0.99" in err
        assert "sidecar" not in err


class TestBumpAndSync:
    """a bump moves every target; ``sync`` heals drift at the declared version."""

    def test_sync_heals_every_drifted_target(self, workspace: Path) -> None:
        """
        drift is brought back to the canonical version.

        :param workspace: the workspace
        :ptype workspace: Path
        :return: nothing
        :rtype: None
        """
        _drift(workspace)
        status, _, err = release(workspace, "sync")
        assert status == 0, err
        status, _, err = release(workspace, "verify", "0.1.0")
        assert status == 0, err

    def test_a_minor_moves_every_target_and_the_notes(self, workspace: Path) -> None:
        """
        versions, bounds, smoke (alias kept), bake, changelog; the nested deployable is untouched.

        :param workspace: the workspace
        :ptype workspace: Path
        :return: nothing
        :rtype: None
        """
        status, out, err = release(workspace, "minor")
        assert status == 0, err
        status, _, err = release(workspace, "verify", "0.2.0")
        assert status == 0, err
        smoke = (workspace / "packages/core/tests/test_smoke.py").read_text()
        assert 'assert core_version == "0.2.0"' in smoke
        assert "from fam import __version__ as core_version" in smoke
        assert '"fam[x]>=0.2.0,<0.3.0"' in (workspace / "packages/extra/pyproject.toml").read_text()
        assert 'version = "9.9.9"' in (workspace / "packages/extra/sidecar/pyproject.toml").read_text()
        changelog = (workspace / "CHANGELOG.md").read_text()
        assert (
            f"## Unreleased\n\n## v0.2.0 -- {TODAY}\n\n### A new thing\n\n- it\n\n## v0.1.0 -- 2026-01-01" in changelog
        )
        assert "packages/agent/tools/pyproject.toml" in out
