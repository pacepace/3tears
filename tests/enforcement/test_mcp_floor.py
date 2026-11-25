"""enforcement: 3tears-mcp's mcp floor is one the floor script can read, and CI tests the code at it.

3tears-mcp declares ``mcp>=2.0,<3``. The workspace suite only ever runs it against the mcp
``uv.lock`` pins, which is always recent, so code that uses a name a later 2.x added would pass
every check and break at import or at runtime for a consumer that resolves the floor.
``scripts/test-mcp-floor.sh`` runs the mcp unit tests on exactly the declared floor. What this
module guards is everything around that run that can rot silently:

- 3tears-mcp states exactly one ``mcp>=X,<Y`` requirement, the shape the script reads;
- CI's ``check`` job still runs the floor script, so the run cannot drop out of the gate
  (``test_check_all_runs_ci_suites.py`` then keeps check-all running it too).

Static parsing only (no network, no install), consistent with the rest of ``tests/enforcement``.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MCP_PYPROJECT = _REPO_ROOT / "packages" / "mcp" / "pyproject.toml"
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_FLOOR_SCRIPT = "scripts/test-mcp-floor.sh"

#: the shape ``scripts/test-mcp-floor.sh`` reads; keep the two in step.
_MCP_REQUIREMENT = re.compile(r"^mcp>=(?P<floor>[0-9][0-9A-Za-z.]*),<(?P<ceiling>[0-9A-Za-z.]+)$")


def test_3tears_mcp_states_one_readable_mcp_floor() -> None:
    """the floor script pins the one ``mcp>=X,<Y`` requirement; any other shape leaves it nothing to pin."""
    dependencies = tomllib.loads(_MCP_PYPROJECT.read_text(encoding="utf-8"))["project"]["dependencies"]
    named_mcp = [dep for dep in dependencies if re.match(r"^mcp\b(?![-_])", dep.strip())]
    readable = [dep for dep in named_mcp if _MCP_REQUIREMENT.match(dep.strip())]
    assert len(named_mcp) == 1 and readable == named_mcp, (
        f"packages/mcp/pyproject.toml names mcp as {named_mcp}; declare exactly one `mcp>=<floor>,<<ceiling>` "
        f"so {_FLOOR_SCRIPT} can test the code at the floor"
    )


def test_ci_runs_the_mcp_floor_script() -> None:
    """the check job runs the mcp tests at the declared floor; removing that step removes the guard."""
    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")
    assert (_REPO_ROOT / _FLOOR_SCRIPT).is_file(), f"{_FLOOR_SCRIPT} is missing"
    assert f"./{_FLOOR_SCRIPT}" in workflow, (
        f"{_CI_WORKFLOW.relative_to(_REPO_ROOT)} no longer runs ./{_FLOOR_SCRIPT}; the workspace suite "
        "only sees the locked mcp, so nothing else would notice the floor falling behind the code."
    )
