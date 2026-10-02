"""
enforcement: the OpenTelemetry floor 3tears declares is one floor, and CI tests the code at it.

``threetears.observe._otel_internals`` imports ``LogRecordExporter`` from
``opentelemetry.sdk._logs.export``, a name that first shipped in opentelemetry-sdk 1.39.0. The
``otel`` extra of 3tears-observe still said ``>=1.28``. Nothing went red, because the workspace
suite only ever ran against the OpenTelemetry ``uv.lock`` pinned, which was 1.43. A consumer that
pinned 1.36 resolved cleanly and then failed at import, in thirteen of its own tests.

The check that a floor matches the code cannot be static: the only way to know whether a name
exists at 1.39 is to import it from 1.39. ``scripts/test-otel-floor.sh`` does that, running the
OpenTelemetry-touching observe tests in an isolated environment pinned to exactly the declared
floors. What this module guards is everything around that run that can rot silently:

- every OpenTelemetry requirement in the workspace states a ``>=`` floor the script can read;
- api, sdk and exporter declared together share ONE floor -- they release in lockstep (each sdk
  pins its api exactly, each exporter its sdk minor), so a split floor is either unresolvable or
  a stale line that looks deliberate;
- the repo's dev install never declares a lower floor than 3tears-observe itself;
- CI's ``check`` job still runs the floor script, so the run cannot drop out of the gate.

Static parsing only (no network, no install), consistent with the rest of ``tests/enforcement``.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_GLOBS = ("packages/*", "packages/agent/*")
_OBSERVE_PYPROJECT = _REPO_ROOT / "packages" / "observe" / "pyproject.toml"
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_FLOOR_SCRIPT = "scripts/test-otel-floor.sh"

#: the release-train members 3tears declares; they version together.
_LOCKSTEP = ("opentelemetry-api", "opentelemetry-sdk", "opentelemetry-exporter-otlp")

#: ``opentelemetry-sdk>=1.39`` or ``opentelemetry-sdk>=1.39,<2`` -> name, floor.
_OTEL_REQUIREMENT = re.compile(r"^(?P<name>opentelemetry-[a-z0-9-]+)(?:\[[a-z0-9,_-]+\])?(?P<spec>.*)$")
_FLOOR = re.compile(r"^>=(?P<floor>[0-9][0-9A-Za-z.]*)(?:,<[0-9A-Za-z.]+)?$")


def _declared_lists() -> dict[str, list[str]]:
    """return every requirement list in the workspace that could name OpenTelemetry.

    covers each package's dependencies and every extra, and the root's dev-dependencies, keyed by
    a label naming the file and the list.

    :return: label to raw requirement strings
    :rtype: dict[str, list[str]]
    """
    lists: dict[str, list[str]] = {}
    pyprojects = [_REPO_ROOT / "pyproject.toml"]
    for glob in _PACKAGE_GLOBS:
        pyprojects.extend(sorted(path / "pyproject.toml" for path in _REPO_ROOT.glob(glob)))
    for pyproject in pyprojects:
        if not pyproject.is_file():
            continue
        label = str(pyproject.relative_to(_REPO_ROOT))
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = data.get("project") or {}
        lists[f"{label} dependencies"] = list(project.get("dependencies") or [])
        for extra, deps in (project.get("optional-dependencies") or {}).items():
            lists[f"{label} optional-dependencies.{extra}"] = list(deps or [])
        uv = (data.get("tool") or {}).get("uv") or {}
        lists[f"{label} tool.uv.dev-dependencies"] = list(uv.get("dev-dependencies") or [])
    return lists


def _otel_requirements(requirements: list[str]) -> dict[str, str]:
    """return the OpenTelemetry requirements of one list as name to version specifier.

    :param requirements: raw requirement strings
    :ptype requirements: list[str]
    :return: package name to its specifier text, stripped
    :rtype: dict[str, str]
    """
    found: dict[str, str] = {}
    for requirement in requirements:
        match = _OTEL_REQUIREMENT.match(requirement.strip())
        if match is not None:
            found[match["name"]] = match["spec"].strip()
    return found


def _floor(spec: str) -> str | None:
    """return the ``>=`` floor of a specifier, or ``None`` when it states none this guard reads.

    :param spec: a specifier such as ``>=1.39`` or ``>=1.39,<2``
    :ptype spec: str
    :return: the floor version text
    :rtype: str | None
    """
    match = _FLOOR.match(spec)
    return match["floor"] if match is not None else None


_LISTS = _declared_lists()
_OBSERVE_OTEL = _otel_requirements(
    tomllib.loads(_OBSERVE_PYPROJECT.read_text(encoding="utf-8"))["project"]["optional-dependencies"]["otel"]
)


def test_the_declarations_were_found() -> None:
    """both inputs are non-empty: a silent zero would pass every comparison below."""
    with_otel = [label for label, deps in _LISTS.items() if _otel_requirements(deps)]
    assert len(_LISTS) > 20, f"only {len(_LISTS)} requirement lists found; the package layout changed"
    assert set(_OBSERVE_OTEL) == set(_LOCKSTEP), (
        f"3tears-observe's otel extra declares {sorted(_OBSERVE_OTEL)}; expected exactly {list(_LOCKSTEP)}"
    )
    assert "pyproject.toml tool.uv.dev-dependencies" in with_otel, (
        "the root dev-dependencies no longer name OpenTelemetry; this guard compares them to observe's floor"
    )


def test_every_opentelemetry_requirement_states_a_floor() -> None:
    """a floor is what the floor script pins; a requirement without one cannot be tested at it."""
    floorless = [
        f"{label}: {name}{spec}"
        for label, deps in _LISTS.items()
        for name, spec in _otel_requirements(deps).items()
        if _floor(spec) is None
    ]
    assert not floorless, (
        "OpenTelemetry requirements with no `>=` floor:\n  "
        + "\n  ".join(floorless)
        + "\n\nDeclare the lowest release whose names the code imports, e.g. `opentelemetry-sdk>=1.39`."
    )


def test_lockstep_members_declared_together_share_one_floor() -> None:
    """api, sdk and exporter in one list name one floor; OpenTelemetry releases them as a train."""
    split: list[str] = []
    for label, deps in _LISTS.items():
        floors = {name: _floor(spec) for name, spec in _otel_requirements(deps).items() if name in _LOCKSTEP}
        if len(set(floors.values())) > 1:
            split.append(f"{label}: {floors}")
    assert not split, (
        "OpenTelemetry api/sdk/exporter declared with different floors:\n  "
        + "\n  ".join(split)
        + "\n\nEach sdk pins its api exactly and each exporter pins its sdk minor, so raise all three "
        "together to the floor the code needs."
    )


def test_the_dev_install_floor_is_observes_floor() -> None:
    """the repo's dev install declares the same OpenTelemetry floor 3tears-observe ships with."""
    dev = _otel_requirements(_LISTS["pyproject.toml tool.uv.dev-dependencies"])
    mismatched = {
        name: (_floor(dev.get(name, "")), _floor(spec))
        for name, spec in _OBSERVE_OTEL.items()
        if _floor(dev.get(name, "")) != _floor(spec)
    }
    assert not mismatched, (
        f"root dev-dependencies floors differ from 3tears-observe's otel extra (dev, observe): {mismatched}. "
        "Move them together; the floor is set in packages/observe/pyproject.toml."
    )


def test_ci_runs_the_floor_script() -> None:
    """the check job runs the code at the declared floor; removing that step removes the guard."""
    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")
    assert (_REPO_ROOT / _FLOOR_SCRIPT).is_file(), f"{_FLOOR_SCRIPT} is missing"
    assert f"./{_FLOOR_SCRIPT}" in workflow, (
        f"{_CI_WORKFLOW.relative_to(_REPO_ROOT)} no longer runs ./{_FLOOR_SCRIPT}; the workspace suite "
        "only sees the locked OpenTelemetry, so nothing else would notice a floor falling behind the code."
    )
