"""Enforcement: the testament image's Chromium is the one the locked playwright drives.

``docker/testament.Dockerfile`` installs headless Chromium for the evals render tests (the ``browser`` marker) with
``playwright==<version> playwright install``: that playwright's ``browsers.json`` names the Chromium build it fetches,
and a playwright looks for its own build only. The workspace venv the tests run in is built from ``uv.lock``. When the
two versions part, the image holds a Chromium the tests' playwright does not look for, and every browser test on the
pool fails at launch. The image is rebuilt only when the Dockerfile's text changes (testament keys it on its inputs),
so the version written there is also what rebuilds it.

The manifest's env is held here too: it points playwright at the image's browsers and requires Chromium, so a pool
runner that lost the browser fails the render tests instead of skipping them.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DOCKERFILE = _REPO_ROOT / "docker" / "testament.Dockerfile"


def _locked_playwright() -> str:
    lock = tomllib.loads((_REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    versions = {package["version"] for package in lock["package"] if package["name"] == "playwright"}
    assert len(versions) == 1, f"uv.lock locks playwright at {sorted(versions)}, not at one version"
    return versions.pop()


def test_the_image_installs_chromium_with_the_locked_playwright() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")
    pins = re.findall(r"playwright==([0-9][^\s\"']*)", dockerfile)
    assert pins, "docker/testament.Dockerfile no longer installs Chromium with a pinned playwright (playwright==X.Y.Z)"
    locked = _locked_playwright()
    assert set(pins) == {locked}, (
        f"docker/testament.Dockerfile installs Chromium with playwright {sorted(set(pins))}, but uv.lock locks "
        f"{locked}: move the Dockerfile's pin to {locked}, so the image holds the Chromium build the tests' playwright "
        "drives"
    )


def test_the_manifest_points_playwright_at_the_image_s_browsers_and_requires_them() -> None:
    manifest = tomllib.loads((_REPO_ROOT / ".testament.toml").read_text(encoding="utf-8"))
    env = manifest["env"]
    installed = re.findall(r"^ENV PLAYWRIGHT_BROWSERS_PATH=(\S+)$", _DOCKERFILE.read_text(encoding="utf-8"), re.M)
    assert installed == [env.get("PLAYWRIGHT_BROWSERS_PATH")], (installed, env.get("PLAYWRIGHT_BROWSERS_PATH"))
    assert env.get("THREETEARS_REQUIRE_CHROMIUM") == "1", "the pool must fail a browser test whose Chromium is missing"


def test_the_image_checks_its_fallback_fonts_are_dejavu() -> None:
    """The package's pages draw in the machine's faces (a local font stack), so the image fails its build unless
    fontconfig resolves them to DejaVu, as on the dev box (scripts/check-fallback-fonts.sh says why); the script is
    an image input, so a change to it rebuilds the image."""
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY --chmod=0755 scripts/check-fallback-fonts.sh /usr/local/bin/check-fallback-fonts" in dockerfile
    run = dockerfile[dockerfile.index("apt-get install -y --no-install-recommends fonts-dejavu-core") :]
    assert "&& check-fallback-fonts" in run.split("\n\n", 1)[0], "the build must run the check after installing DejaVu"
    manifest = tomllib.loads((_REPO_ROOT / ".testament.toml").read_text(encoding="utf-8"))
    assert "scripts/check-fallback-fonts.sh" in manifest["image"]["inputs"]
