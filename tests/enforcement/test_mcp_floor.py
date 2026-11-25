"""enforcement: the MCP SDK floors 3tears declares are ones the floor script can read, and CI tests the code at them.

3tears supports MCP SDK 2.x only: 3tears-mcp and 3tears-models' ``claude-cli`` extra declare
``mcp>=2.0,<3``, and 3tears-evals' ``fastmcp`` extra ``fastmcp>=4.0,<5`` (fastmcp 4 is the first
major on mcp 2). The workspace suite only ever runs against the versions ``uv.lock`` pins, which are
always recent, so code that uses a name a later release added would pass every check and break at
import or at runtime for a consumer that resolves the floor. ``scripts/test-mcp-floor.sh`` runs
those packages' MCP tests on exactly the declared floors. What this module guards is everything
around that run that can rot silently:

- 3tears-mcp states exactly one ``mcp>=X,<Y`` requirement, the shape the script reads;
- 3tears-models states the same mcp requirement, so the one floor the script pins is both of theirs;
- the evals ``fastmcp`` extra states one ``fastmcp>=F,<G`` and one ``fastmcp-slim`` with the same
  bounds, and the repo's dev install declares the same fastmcp bounds;
- CI's ``check`` job still runs the floor script, so the run cannot drop out of the gate
  (``test_check_all_runs_ci_suites.py`` then keeps check-all running it too).

Static parsing only (no network, no install), consistent with the rest of ``tests/enforcement``.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MCP_PYPROJECT = _REPO_ROOT / "packages" / "mcp" / "pyproject.toml"
_MODELS_PYPROJECT = _REPO_ROOT / "packages" / "models" / "pyproject.toml"
_EVALS_PYPROJECT = _REPO_ROOT / "packages" / "evals" / "pyproject.toml"
_ROOT_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_FLOOR_SCRIPT = "scripts/test-mcp-floor.sh"

#: the shape ``scripts/test-mcp-floor.sh`` reads; keep the two in step.
_MCP_REQUIREMENT = re.compile(r"^mcp>=(?P<floor>[0-9][0-9A-Za-z.]*),<(?P<ceiling>[0-9A-Za-z.]+)$")


def _project(path: Path) -> dict[str, Any]:
    """the ``[project]`` table of a pyproject.

    :param path: the pyproject
    :ptype path: Path
    :return: its project table
    :rtype: dict[str, Any]
    """
    project: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
    return project


def _named(requirements: list[str], name: str) -> list[str]:
    """the requirements naming exactly ``name`` (not a package whose name it begins).

    :param requirements: raw requirement strings
    :ptype requirements: list[str]
    :param name: a distribution name
    :ptype name: str
    :return: the matching requirements, stripped
    :rtype: list[str]
    """
    return [dep.strip() for dep in requirements if re.match(rf"^{re.escape(name)}(?![A-Za-z0-9_.-])", dep.strip())]


def test_3tears_mcp_states_one_readable_mcp_floor() -> None:
    """the floor script pins the one ``mcp>=X,<Y`` requirement; any other shape leaves it nothing to pin."""
    named_mcp = _named(_project(_MCP_PYPROJECT)["dependencies"], "mcp")
    readable = [dep for dep in named_mcp if _MCP_REQUIREMENT.match(dep.strip())]
    assert len(named_mcp) == 1 and readable == named_mcp, (
        f"packages/mcp/pyproject.toml names mcp as {named_mcp}; declare exactly one `mcp>=<floor>,<<ceiling>` "
        f"so {_FLOOR_SCRIPT} can test the code at the floor"
    )


def test_3tears_models_states_the_same_mcp_requirement() -> None:
    """the script pins 3tears-mcp's floor; models must declare that floor for the run to test it."""
    mcp = _named(_project(_MCP_PYPROJECT)["dependencies"], "mcp")
    extra = _project(_MODELS_PYPROJECT)["optional-dependencies"]["claude-cli"]
    assert _named(extra, "mcp") == mcp, (
        f"3tears-models' claude-cli extra names mcp as {_named(extra, 'mcp')}, 3tears-mcp as {mcp}; "
        "declare the same requirement in both, or the floor run tests only one of them"
    )


def test_the_evals_fastmcp_extra_states_readable_floors_the_dev_install_shares() -> None:
    """fastmcp and fastmcp-slim each have one readable ``>=F,<G``, the same bounds, and the dev install agrees."""
    extra = _project(_EVALS_PYPROJECT)["optional-dependencies"]["fastmcp"]
    fastmcp, slim = _named(extra, "fastmcp"), _named(extra, "fastmcp-slim")
    assert len(fastmcp) == 1 and len(slim) == 1, f"the fastmcp extra names {fastmcp} and {slim}; declare one of each"
    bounds = _MCP_REQUIREMENT.pattern.replace("^mcp", "")
    assert re.match(rf"^fastmcp{bounds}", fastmcp[0]) and re.match(rf"^fastmcp-slim{bounds}", slim[0]), (
        f"{fastmcp[0]!r} and {slim[0]!r} must each read `>=<floor>,<<ceiling>` for {_FLOOR_SCRIPT}"
    )
    assert fastmcp[0].removeprefix("fastmcp") == slim[0].removeprefix("fastmcp-slim"), (
        "fastmcp and fastmcp-slim release together; give them the same bounds"
    )
    dev = _named(
        tomllib.loads(_ROOT_PYPROJECT.read_text(encoding="utf-8"))["tool"]["uv"]["dev-dependencies"], "fastmcp"
    )
    assert dev == fastmcp, (
        f"the root dev install names fastmcp as {dev}, the evals extra as {fastmcp}; move them together"
    )


def test_ci_runs_the_mcp_floor_script() -> None:
    """the check job runs the mcp tests at the declared floor; removing that step removes the guard."""
    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")
    assert (_REPO_ROOT / _FLOOR_SCRIPT).is_file(), f"{_FLOOR_SCRIPT} is missing"
    assert f"./{_FLOOR_SCRIPT}" in workflow, (
        f"{_CI_WORKFLOW.relative_to(_REPO_ROOT)} no longer runs ./{_FLOOR_SCRIPT}; the workspace suite "
        "only sees the locked mcp, so nothing else would notice the floor falling behind the code."
    )
