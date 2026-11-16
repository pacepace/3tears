"""The browser packages (web/, @3tears/*) carry the Python family's version, and the release tool knows them.

One tag releases both (release.yml publishes PyPI, then npm), so there is one version to
track. The release tool's ``verify`` (the release workflow's pre-flight) checks every file
``npm-lockstep-files`` names; this holds the other half, that it names every web/ package, so a new
package cannot ship at a stale version because nothing listed it.
"""

from __future__ import annotations

import json
from pathlib import Path

from threetears.enforcement.release import load_release_config, lockstep_mismatches, parse_version, project_version

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _family_version() -> str:
    config = load_release_config(_REPO_ROOT)
    text = (_REPO_ROOT / config.version_file).read_text(encoding="utf-8")
    return ".".join(map(str, project_version(text)))


def test_every_web_package_is_a_lockstep_file() -> None:
    """each ``web/packages/*/package.json`` and the workspace lock are in ``npm-lockstep-files``.

    :return: nothing
    :rtype: None
    """
    config = load_release_config(_REPO_ROOT)
    manifests = sorted(p.relative_to(_REPO_ROOT).as_posix() for p in _REPO_ROOT.glob("web/packages/*/package.json"))
    assert manifests, "web/packages has no package; the workspace layout changed"
    missing = [path for path in [*manifests, "web/package-lock.json"] if path not in config.npm_files]
    assert missing == [], f"not in [tool.threetears-release] npm-lockstep-files: {missing}"


def test_web_packages_are_at_the_family_version() -> None:
    """every web/ version location says the family version.

    :return: nothing
    :rtype: None
    """
    config = load_release_config(_REPO_ROOT)
    version = _family_version()
    parse_version(version)
    web = [line for line in lockstep_mismatches(config, version) if line.startswith("web/")]
    assert web == [], "\n".join(web)


def test_web_packages_are_scoped_to_the_family() -> None:
    """a web/ package is ``@3tears/web-<name>``: the scope the release tool moves sibling pins in, and the
    ``web-`` prefix that keeps the browser packages apart from anything else under the scope.

    :return: nothing
    :rtype: None
    """
    for manifest in _REPO_ROOT.glob("web/packages/*/package.json"):
        name = json.loads(manifest.read_text(encoding="utf-8"))["name"]
        assert name.startswith("@3tears/web-"), f"{manifest.relative_to(_REPO_ROOT)}: {name}"
