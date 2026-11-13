"""Enforcement: the testament image holds the services the integration tests name, at the versions they name.

The pool's container runs no Docker daemon, so the integration suite there starts Postgres and nats-server as child
processes, from binaries ``docker/testament.Dockerfile`` installs. They stand in for the docker images the tests name
elsewhere: ``threetears.core.testing.fixtures``' default ``pgvector/pgvector:pg<major>``, and each
``nats:<version>-alpine`` a test or fixture pins. A test that pins a nats version the image lacks, or a fixture whose
Postgres major moves without the image, would fail only on the pool, and only once the image is rebuilt; the image is
rebuilt only when its inputs change. So the Dockerfile's pins are held to the tree's here:

- every ``nats:<version>-alpine`` named under ``packages/`` has a sha256-pinned tarball for amd64 and for arm64, and
  ``check-test-services`` checks exactly those versions at build;
- the ``postgresql-<major>`` packages, ``check-test-services``' major and the fixtures' default image agree;
- ``scripts/check-test-services.sh`` is an ``[image] input`` (it is COPYed, so a change to it must rebuild);
- the services' layers come after the browser layer, so changing them keeps the cached Chromium layer.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DOCKERFILE = _REPO_ROOT / "docker" / "testament.Dockerfile"
_FIXTURES = _REPO_ROOT / "packages" / "core" / "src" / "threetears" / "core" / "testing" / "fixtures.py"
_ARCHES = {"amd64", "arm64"}


def _dockerfile() -> str:
    return _DOCKERFILE.read_text(encoding="utf-8")


def _check_invocation() -> tuple[str, list[str]]:
    """The major and nats versions the build's ``check-test-services`` run is given."""
    runs = re.findall(r"^RUN check-test-services (\d+)((?: [0-9.]+)+)$", _dockerfile(), re.M)
    assert len(runs) == 1, f"docker/testament.Dockerfile must run `check-test-services MAJOR VERSION...` once: {runs}"
    major, versions = runs[0]
    return major, versions.split()


def _nats_pins() -> dict[str, set[str]]:
    """Each nats-server version the Dockerfile pins, with the architectures it has a sha256 for."""
    pins: dict[str, set[str]] = {}
    for version, arch in re.findall(r'^\s*"([0-9.]+) (amd64|arm64) [0-9a-f]{64}"', _dockerfile(), re.M):
        pins.setdefault(version, set()).add(arch)
    return pins


def _nats_versions_named() -> dict[str, str]:
    """Every ``nats:<version>-alpine`` named in a Python file under ``packages/``, with one file naming it."""
    named: dict[str, str] = {}
    for path in sorted((_REPO_ROOT / "packages").glob("**/*.py")):
        if "__pycache__" in path.parts:
            continue
        for version in re.findall(r"nats:(\d+\.\d+\.\d+)-alpine", path.read_text(encoding="utf-8")):
            named.setdefault(version, str(path.relative_to(_REPO_ROOT)))
    return named


def _fixtures_default_major() -> str:
    """The Postgres major of ``db_image``'s default, ``pgvector/pgvector:pg<major>``."""
    tree = ast.parse(_FIXTURES.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "db_image":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    found = re.fullmatch(r"pgvector/pgvector:pg(\d+)", sub.value)
                    if found:
                        return found.group(1)
    raise AssertionError(f"{_FIXTURES.relative_to(_REPO_ROOT)}: db_image no longer defaults to pgvector/pgvector:pgN")


def test_the_tree_names_nats_versions_the_walk_can_see() -> None:
    """The control: the pins the nats tests are known to carry are found, so an empty walk can't pass."""
    assert {"2.14.2", "2.12.6"} <= set(_nats_versions_named())


def test_every_nats_version_the_tests_name_is_in_the_image_for_both_architectures() -> None:
    pins = _nats_pins()
    missing = {version: path for version, path in _nats_versions_named().items() if pins.get(version) != _ARCHES}
    assert not missing, (
        f"nats versions named in the tree with no sha256-pinned tarball for both {sorted(_ARCHES)} in "
        f"docker/testament.Dockerfile: {missing} (pinned: { {v: sorted(a) for v, a in pins.items()} }). Add a "
        '"<version> <arch> <sha256>" line per architecture, from the release\'s SHA256SUMS'
    )


def test_the_build_checks_exactly_the_pinned_nats_versions() -> None:
    _, checked = _check_invocation()
    assert sorted(checked) == sorted(_nats_pins()), (checked, sorted(_nats_pins()))


def test_the_image_s_postgres_major_is_the_fixtures_default() -> None:
    dockerfile = _dockerfile()
    installed = set(re.findall(r"\bpostgresql-(\d+)(?:-pgvector)?\b", dockerfile))
    clients = set(re.findall(r"\bpostgresql-client-(\d+)\b", dockerfile))
    major, _ = _check_invocation()
    default = _fixtures_default_major()
    assert installed == clients == {major} == {default}, (
        f"server packages {sorted(installed)}, client packages {sorted(clients)}, check-test-services {major} and "
        f"the fixtures' default pgvector/pgvector:pg{default} must name one Postgres major"
    )


def test_the_check_script_is_an_image_input_and_runs_at_build() -> None:
    dockerfile = _dockerfile()
    assert "COPY --chmod=0755 scripts/check-test-services.sh /usr/local/bin/check-test-services" in dockerfile
    manifest = tomllib.loads((_REPO_ROOT / ".testament.toml").read_text(encoding="utf-8"))
    assert "scripts/check-test-services.sh" in manifest["image"]["inputs"]


def test_the_services_layers_come_after_the_browser_layer() -> None:
    dockerfile = _dockerfile()
    order = [
        dockerfile.index("playwright install"),
        dockerfile.index("apt.postgresql.org.sh"),
        dockerfile.index("nats-server-v"),
        dockerfile.index("RUN check-test-services"),
        dockerfile.index("RUN useradd"),
    ]
    assert order == sorted(order), "the services' layers go after the playwright layer and before `useradd`"
