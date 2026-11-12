"""``MIGRATIONS.toml`` ships inside the installed ``3tears-evals`` package, so a host reads it from what it installed."""

from __future__ import annotations

import shutil
import subprocess
import tomllib
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "packages" / "evals" / "MIGRATIONS.toml"
PACKAGED = "threetears/evals/MIGRATIONS.toml"


def test_migrations_toml_reaches_a_built_wheel_inside_the_package(tmp_path: Path) -> None:
    """A wheel built from the workspace root, as the release builds, holds the repo's notes at the packaged path."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not available, so nothing can be built")
    built = subprocess.run(  # noqa: S603 - fixed argv, repo-local paths
        [uv, "build", "--package", "3tears-evals", "--wheel", "--out-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )
    assert built.returncode == 0, f"the wheel would not build: {built.stderr}"
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {[w.name for w in wheels]}"
    with zipfile.ZipFile(wheels[0]) as wheel:
        assert PACKAGED in wheel.namelist(), "MIGRATIONS.toml is not in the wheel"
        shipped = wheel.read(PACKAGED)
    assert tomllib.loads(shipped.decode("utf-8")) == tomllib.loads(SOURCE.read_text(encoding="utf-8"))
