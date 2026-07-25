"""end-to-end fixture test for ``scripts/verify-dist-complete.sh``.

The script exists because v0.18.0 published 26 of its 27 packages: a step in
``release.yml`` deleted ``3tears-scrape`` from ``dist/`` between build and upload,
and nothing objected. The guard turns that class of omission into a failed build.

A guard nobody tests is the same shape of problem as the comment it replaced -- it
looks like protection right up until the moment it silently stops working -- so
these tests pin both directions: a complete dist passes, and each way of being
incomplete fails and names what is missing.

Self-contained: builds a fake workspace under ``tmp_path`` with the script staged
in ``tmp_path/scripts/`` (the script derives REPO_ROOT from its own location), so
nothing here depends on this repo's real package set or version.

    uv run pytest tests/test_verify_dist_complete_script.py -v
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "verify-dist-complete.sh"


def _pyproject(name: str) -> str:
    return textwrap.dedent(
        f"""\
        [project]
        name = "{name}"
        version = "1.2.3"
        """
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A miniature workspace: two top-level packages and one under agent/."""
    (tmp_path / "scripts").mkdir()
    shutil.copy(_SCRIPT, tmp_path / "scripts" / "verify-dist-complete.sh")
    (tmp_path / "scripts" / "verify-dist-complete.sh").chmod(0o755)

    for rel, name in (
        ("packages/core", "3tears"),
        ("packages/scrape", "3tears-scrape"),
        ("packages/agent/memory", "3tears-agent-memory"),
    ):
        pkg = tmp_path / rel
        pkg.mkdir(parents=True)
        (pkg / "pyproject.toml").write_text(_pyproject(name))

    # A nested pyproject that is NOT a workspace member (the real one is
    # packages/scrape/sidecar) -- it must not be demanded in dist/.
    sidecar = tmp_path / "packages/scrape/sidecar"
    sidecar.mkdir(parents=True)
    (sidecar / "pyproject.toml").write_text(_pyproject("nodriver-sidecar"))

    (tmp_path / "dist").mkdir()
    return tmp_path


def _add_artifacts(workspace: Path, normalized: str, *, wheel: bool = True, sdist: bool = True) -> None:
    dist = workspace / "dist"
    if wheel:
        (dist / f"{normalized}-1.2.3-py3-none-any.whl").write_text("x")
    if sdist:
        (dist / f"{normalized}-1.2.3.tar.gz").write_text("x")


def _run(workspace: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(workspace / "scripts" / "verify-dist-complete.sh")],
        capture_output=True,
        text=True,
        check=False,
    )


def _complete(workspace: Path) -> None:
    _add_artifacts(workspace, "3tears")
    _add_artifacts(workspace, "3tears_scrape")
    _add_artifacts(workspace, "3tears_agent_memory")


def test_a_complete_dist_passes(workspace: Path) -> None:
    """Every member present as both sdist and wheel, including the agent/ tier."""
    _complete(workspace)

    result = _run(workspace)

    assert result.returncode == 0, result.stderr
    assert "all 3 workspace packages" in result.stdout


def test_the_v0180_omission_is_caught(workspace: Path) -> None:
    """The exact failure this guard was written for: one package dropped from dist/.

    Named for the incident rather than for the mechanism, because the mechanism
    (a withhold step) is gone and the thing worth never repeating is the outcome.
    """
    _add_artifacts(workspace, "3tears")
    _add_artifacts(workspace, "3tears_agent_memory")
    # 3tears-scrape built fine and was removed before upload.

    result = _run(workspace)

    assert result.returncode == 1
    assert "3tears-scrape" in result.stderr
    assert "3tears" in result.stderr


def test_a_wheel_without_an_sdist_is_incomplete(workspace: Path) -> None:
    """Both artifact kinds are required; a half-built package is still a gap."""
    _add_artifacts(workspace, "3tears")
    _add_artifacts(workspace, "3tears_agent_memory")
    _add_artifacts(workspace, "3tears_scrape", sdist=False)

    result = _run(workspace)

    assert result.returncode == 1
    assert "no sdist" in result.stderr
    assert "no wheel" not in result.stderr


def test_a_sidecar_pyproject_is_not_demanded(workspace: Path) -> None:
    """Only workspace members are required.

    ``packages/scrape/sidecar`` has its own pyproject and its own version and is a
    separate deployable, exactly as bump-version.sh's lockstep scope treats it. If
    this guard demanded it, every release would fail on an artifact that is never
    built.
    """
    _complete(workspace)

    result = _run(workspace)

    assert result.returncode == 0, result.stderr
    assert "nodriver" not in result.stdout


def test_a_missing_dist_directory_fails_loudly(workspace: Path) -> None:
    """Nothing built at all is a failure, not a vacuous pass over zero packages."""
    shutil.rmtree(workspace / "dist")

    result = _run(workspace)

    assert result.returncode == 1
    assert "does not exist" in result.stderr


def test_an_empty_workspace_fails_rather_than_passing_vacuously(tmp_path: Path) -> None:
    """A guard that finds no packages has been mis-wired, and must say so.

    Without this, a glob that stopped matching would report success on an empty
    check -- the guard would still be green while protecting nothing, which is
    precisely the failure mode it exists to prevent.
    """
    (tmp_path / "scripts").mkdir()
    shutil.copy(_SCRIPT, tmp_path / "scripts" / "verify-dist-complete.sh")
    (tmp_path / "scripts" / "verify-dist-complete.sh").chmod(0o755)
    (tmp_path / "dist").mkdir()

    result = subprocess.run(
        [str(tmp_path / "scripts" / "verify-dist-complete.sh")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "no workspace members" in result.stderr
