"""end-to-end fixture test for ``scripts/bump-version.sh``.

we let the script be the source of truth for "every place the version
string lives" -- this test confirms the bump path edits each target
correctly, the verify path catches drift on each target, and the
two stay in sync (anyone who adds a new lockstep target updates the
script and gets coverage from these tests for free).

invocation:

    uv run pytest tests/test_bump_version_script.py -v

the test does not rely on this repo's current version values: it
builds a self-contained fixture under ``tmp_path`` carrying one of
each file pattern at a known starting version, runs the script
against ``tmp_path`` (via the REPO_ROOT calculation in the script),
then asserts the bump landed and the verify pass agrees.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "bump-version.sh"

# pyproject.toml fixture at the canonical-version starting state.
# the script's REPO_ROOT comes from the script's own directory; the
# fixture below stages the script in tmp_path/scripts/ so the
# resolved REPO_ROOT is tmp_path.
_FIXTURE_PYPROJECT = (
    textwrap.dedent(
        """\
    [project]
    name = "fixture-pkg"
    version = "0.1.0"
    """
    ).strip()
    + "\n"
)

_FIXTURE_SMOKE = (
    textwrap.dedent(
        """\
    from fixture_pkg import __version__


    def test_version() -> None:
        assert __version__ == "0.1.0"
    """
    ).strip()
    + "\n"
)

# docker-bake.hcl mimicking the shape of the real bake file: the
# VERSION variable + two hardcoded image-tag references the script
# is meant to keep in sync with VERSION.
_FIXTURE_BAKE = (
    textwrap.dedent(
        """\
    variable "VERSION" {
      default = "v0.1.0"
    }

    target "threetears-base" {
      contexts = {
        threetears-base = "docker-image://ghcr.io/pacepace/threetears-base:v0.1.0"
      }
    }

    target "aibots-base" {
      contexts = {
        aibots-base = "docker-image://ghcr.io/pacepace/aibots-base:v0.1.0"
      }
    }
    """
    ).strip()
    + "\n"
)


def _build_fixture(root: Path) -> None:
    """stage a self-contained mini-repo under ``root`` with one of each
    lockstep target file pattern at version ``0.1.0``.

    layout matches the production repo well enough that the script's
    REPO_ROOT discovery + glob patterns work unchanged.
    """
    (root / "scripts").mkdir()
    shutil.copy2(_SCRIPT, root / "scripts" / "bump-version.sh")
    (root / "scripts" / "bump-version.sh").chmod(0o755)

    # one pyproject inside packages/core (so the canonical-version
    # lookup at the start of the script resolves cleanly)
    (root / "packages" / "core").mkdir(parents=True)
    (root / "packages" / "core" / "pyproject.toml").write_text(_FIXTURE_PYPROJECT)
    (root / "packages" / "core" / "tests").mkdir()
    (root / "packages" / "core" / "tests" / "test_smoke.py").write_text(_FIXTURE_SMOKE)

    # a second package's pyproject so the multi-target bump path
    # exercises more than one file
    (root / "packages" / "extra").mkdir(parents=True)
    (root / "packages" / "extra" / "pyproject.toml").write_text(_FIXTURE_PYPROJECT)

    (root / "docker-bake.hcl").write_text(_FIXTURE_BAKE)


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """run the staged script with cwd = root. ``--no-lock`` is passed
    to skip the ``uv lock`` step that requires a real uv workspace."""
    return subprocess.run(
        [str(root / "scripts" / "bump-version.sh"), "--no-lock", *args],
        cwd=root,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": os.environ.get("PATH", "")},
    )


class TestBumpVersionScript:
    """exercise bump + verify modes against a self-contained fixture."""

    def test_verify_passes_on_consistent_fixture(self, tmp_path: Path) -> None:
        """verify mode against a fresh fixture (everything at 0.1.0)
        succeeds when asked to verify ``0.1.0``."""
        _build_fixture(tmp_path)
        result = _run(tmp_path, "--verify", "0.1.0")
        assert result.returncode == 0, f"--verify against unchanged fixture failed: {result.stdout}{result.stderr}"
        assert "All version locations at 0.1.0" in result.stdout

    def test_verify_catches_drift_on_each_target(self, tmp_path: Path) -> None:
        """drift one file at a time; verify must surface each drift
        as a MISMATCH line. This is the regression guard: anyone who
        adds a new lockstep target without wiring verify coverage will
        see this drop coverage in the assertions below.
        """
        _build_fixture(tmp_path)
        # drift packages/extra/pyproject.toml
        pkg_extra = tmp_path / "packages" / "extra" / "pyproject.toml"
        pkg_extra.write_text(pkg_extra.read_text().replace('"0.1.0"', '"0.0.99"'))
        # drift the smoke test
        smoke = tmp_path / "packages" / "core" / "tests" / "test_smoke.py"
        smoke.write_text(smoke.read_text().replace('"0.1.0"', '"0.0.99"'))
        # drift the bake VERSION + image tag
        bake = tmp_path / "docker-bake.hcl"
        bake.write_text(bake.read_text().replace('"v0.1.0"', '"v0.0.99"').replace(":v0.1.0", ":v0.0.99"))

        result = _run(tmp_path, "--verify", "0.1.0")
        assert result.returncode != 0
        # one mismatch line per drifted target type -- if any of these
        # disappear from the verify output, the verify mode has lost
        # coverage of that target and needs the script's verify
        # section extended in lockstep with whatever bump-mode change
        # caused the regression.
        combined = result.stdout + result.stderr
        assert "packages/extra/pyproject.toml" in combined
        assert "packages/core/tests/test_smoke.py" in combined
        assert "docker-bake.hcl VERSION" in combined
        assert "docker-bake.hcl image-tag" in combined

    def test_bump_fixes_every_drifted_target(self, tmp_path: Path) -> None:
        """bump path applied against a drifted state heals every
        target. proves bump-mode coverage stays in lockstep with
        verify-mode coverage.
        """
        _build_fixture(tmp_path)
        # introduce the SAME drift as above on every target
        pkg_extra = tmp_path / "packages" / "extra" / "pyproject.toml"
        pkg_extra.write_text(pkg_extra.read_text().replace('"0.1.0"', '"0.0.99"'))
        smoke = tmp_path / "packages" / "core" / "tests" / "test_smoke.py"
        smoke.write_text(smoke.read_text().replace('"0.1.0"', '"0.0.99"'))
        bake = tmp_path / "docker-bake.hcl"
        bake.write_text(bake.read_text().replace('"v0.1.0"', '"v0.0.99"').replace(":v0.1.0", ":v0.0.99"))

        result = _run(tmp_path, "0.1.0")
        assert result.returncode == 0, f"bump failed: {result.stdout}{result.stderr}"

        verify = _run(tmp_path, "--verify", "0.1.0")
        assert verify.returncode == 0, f"verify after bump still failed: {verify.stdout}{verify.stderr}"

    def test_bump_to_new_version_moves_every_target(self, tmp_path: Path) -> None:
        """canonical 0.1.0 -> 0.2.0 bump from a clean fixture: every
        lockstep target should land at 0.2.0.
        """
        _build_fixture(tmp_path)
        result = _run(tmp_path, "0.2.0")
        assert result.returncode == 0, result.stderr

        # spot-check each file shape ended up at 0.2.0
        assert 'version = "0.2.0"' in (tmp_path / "packages" / "core" / "pyproject.toml").read_text()
        assert 'version = "0.2.0"' in (tmp_path / "packages" / "extra" / "pyproject.toml").read_text()
        assert '"0.2.0"' in (tmp_path / "packages" / "core" / "tests" / "test_smoke.py").read_text()
        bake_after = (tmp_path / "docker-bake.hcl").read_text()
        assert '"v0.2.0"' in bake_after
        assert "threetears-base:v0.2.0" in bake_after
        assert "aibots-base:v0.2.0" in bake_after
        # and the verify pass should agree
        verify = _run(tmp_path, "--verify", "0.2.0")
        assert verify.returncode == 0, f"{verify.stdout}{verify.stderr}"

    def test_verify_requires_explicit_target(self, tmp_path: Path) -> None:
        """``--verify`` without an X.Y.Z target is a usage error
        (we don't want a silent default that could mask drift).
        """
        _build_fixture(tmp_path)
        result = _run(tmp_path, "--verify", "patch")
        assert result.returncode != 0
        assert "explicit X.Y.Z" in result.stderr or "Usage" in result.stderr

    def test_verify_catches_aliased_version_drift(self, tmp_path: Path) -> None:
        """``assert <alias>_version == "X.Y.Z"`` drift is caught.

        Regression guard for v0.10.2: the cross-package smoke test in
        ``packages/core/tests/test_smoke.py`` uses
        ``from threetears.core import __version__ as core_version``
        and then ``assert core_version == "X.Y.Z"``. Pre-fix the
        verify regex matched only the literal ``__version__`` token,
        so the aliased assertions silently held the old value and CI
        only caught the drift after the bump PR shipped. This test
        pins the aliased shape so the regression cannot recur.
        """
        _build_fixture(tmp_path)
        # rewrite the fixture smoke to use the aliased shape that
        # broke v0.10.2. one aliased line at the old version, the
        # rest at the canonical -- exactly what packages/core looked
        # like when the v0.10.2 bump went out.
        smoke = tmp_path / "packages" / "core" / "tests" / "test_smoke.py"
        smoke.write_text(
            textwrap.dedent(
                """\
                def test_aliased_import() -> None:
                    from fixture_pkg import __version__ as core_version
                    assert core_version == "0.0.99"
                """,
            ).strip()
            + "\n",
        )

        result = _run(tmp_path, "--verify", "0.1.0")
        assert result.returncode != 0, (
            "--verify failed to catch aliased ``core_version`` drift; "
            "regex must match both ``__version__`` and ``<alias>_version``"
        )
        combined = result.stdout + result.stderr
        assert "core_version" in combined
        assert "packages/core/tests/test_smoke.py" in combined

    def test_bump_rewrites_aliased_version_assertions(self, tmp_path: Path) -> None:
        """bump mode rewrites ``<alias>_version == "..."`` and preserves
        the alias name on the LHS.

        Co-regression-guard for v0.10.2. The bump regex's capture group
        must preserve ``core_version`` / ``memory_version`` / etc. on
        rewrite -- a naive ``s/.../__version__/g`` would have
        unilaterally replaced the alias with ``__version__`` and
        broken the imports.
        """
        _build_fixture(tmp_path)
        smoke = tmp_path / "packages" / "core" / "tests" / "test_smoke.py"
        smoke.write_text(
            textwrap.dedent(
                """\
                def test_aliased_import() -> None:
                    from fixture_pkg import __version__ as core_version
                    assert core_version == "0.0.99"
                """,
            ).strip()
            + "\n",
        )

        result = _run(tmp_path, "0.2.0")
        assert result.returncode == 0, f"{result.stdout}{result.stderr}"

        smoke_after = smoke.read_text()
        # the LHS alias is preserved; the RHS is bumped to the new version
        assert 'assert core_version == "0.2.0"' in smoke_after, smoke_after
        # the import line is untouched (the script must not rewrite
        # the import alias to ``__version__`` -- that would break the
        # cross-package import contract)
        assert "from fixture_pkg import __version__ as core_version" in smoke_after

    @pytest.mark.parametrize(
        "keyword,starting,expected",
        [
            ("patch", "0.1.0", "0.1.1"),
            ("minor", "0.1.5", "0.2.0"),
            ("major", "0.9.9", "1.0.0"),
        ],
    )
    def test_bump_keywords_resolve_correctly(
        self,
        tmp_path: Path,
        keyword: str,
        starting: str,
        expected: str,
    ) -> None:
        """patch / minor / major SemVer arithmetic produces the
        expected resolved target.
        """
        _build_fixture(tmp_path)
        pkg_core = tmp_path / "packages" / "core" / "pyproject.toml"
        pkg_core.write_text(pkg_core.read_text().replace('"0.1.0"', f'"{starting}"'))
        pkg_extra = tmp_path / "packages" / "extra" / "pyproject.toml"
        pkg_extra.write_text(pkg_extra.read_text().replace('"0.1.0"', f'"{starting}"'))
        smoke = tmp_path / "packages" / "core" / "tests" / "test_smoke.py"
        smoke.write_text(smoke.read_text().replace('"0.1.0"', f'"{starting}"'))
        bake = tmp_path / "docker-bake.hcl"
        bake.write_text(bake.read_text().replace("v0.1.0", f"v{starting}"))

        result = _run(tmp_path, keyword)
        assert result.returncode == 0, result.stderr
        assert f"Done. New version: {expected}" in result.stdout
