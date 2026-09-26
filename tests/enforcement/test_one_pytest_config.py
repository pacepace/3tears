"""
enforcement: the workspace root's pytest configuration is the only one.

pytest picks its rootdir and inifile from the closest ancestor of the paths it
is given that carries a ``[tool.pytest.ini_options]`` table. A package whose own
``pyproject.toml`` carries one therefore wins for every run pointed at that
package -- ``./scripts/test.sh agent/acl`` -- and the root's settings silently
stop applying: ``--import-mode=importlib``, ``pythonpath = ["."]``, namespace
packages, the marker list. Test modules then lose their ``packages.<pkg>...``
names and every relative helper import (``from ._fake_loaders import ...``)
fails at collection, while the whole-workspace run, rooted at the repo, stays
green and hides it.

Four packages carried such a table, each holding only settings the root already
had (``asyncio_mode = "auto"`` and the ``integration`` marker). Run on their own,
two failed collection outright and the other two ran under a configuration the
workspace run never uses. Several other packages already say, in a comment, why
they have none; nothing enforced it.

The scrape sidecar is exempt: it is a separate deployable with its own venv,
``--ignore``d by the workspace configuration, and its own pytest run.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEPARATE_DEPLOYABLES = frozenset({Path("packages/scrape/sidecar/pyproject.toml")})


def _package_manifests() -> list[Path]:
    """every workspace package manifest, found by walking rather than by a fixed depth.

    :return: repo-relative manifest paths
    :rtype: list[Path]
    """
    return sorted(
        path.relative_to(_REPO_ROOT)
        for path in (_REPO_ROOT / "packages").rglob("pyproject.toml")
        if ".venv" not in path.parts and "node_modules" not in path.parts
    )


def test_the_root_carries_the_pytest_configuration() -> None:
    root = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    assert "ini_options" in root.get("tool", {}).get("pytest", {})


def test_no_workspace_package_carries_its_own_pytest_configuration() -> None:
    manifests = _package_manifests()
    assert len(manifests) > len(_SEPARATE_DEPLOYABLES), "found no package manifests to check"

    offenders = [
        str(path)
        for path in manifests
        if path not in _SEPARATE_DEPLOYABLES
        and "pytest" in tomllib.loads((_REPO_ROOT / path).read_text()).get("tool", {})
    ]

    assert offenders == [], (
        "these packages carry a [tool.pytest...] table, which makes pytest root a per-package run "
        "there and drop the workspace configuration (import mode, pythonpath, markers); put any "
        f"setting they need in the root pyproject.toml instead: {offenders}"
    )
