"""Enforcement: the typecheck suite checks every workspace package, and a bare mypy checks the same set.

The typecheck suite (``typecheck/``, testament's ``typecheck`` suite) reads each package's import roots
off its ``src/threetears/`` tree. A bare ``mypy`` (an editor, anyone invoking it directly) reads the root
pyproject's ``[tool.mypy] files`` instead, and resolves imports through its ``mypy_path``. When the two
disagree they give different answers on the same code, and whichever one runs looks authoritative: they
did disagree once, 28 targets against 21, so a bare mypy skipped six packages the gate checked. So:

- every workspace member is a package the suite checks: a member whose tree the walk cannot read is
  a package nobody checks, while the gate stays green;
- ``files`` and ``mypy_path`` name exactly those packages' source trees;
- the unit suite never collects ``typecheck/``, which runs once, as its own suite.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from typecheck.targets import REPO_ROOT, packages

_PYPROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
_MANIFEST = tomllib.loads((REPO_ROOT / ".testament.toml").read_text())


def _members() -> set[str]:
    """Return the uv workspace's members with a pyproject, repo-relative.

    :return: each member's directory
    :rtype: set[str]
    """
    workspace = _PYPROJECT["tool"]["uv"]["workspace"]
    excluded = set(workspace.get("exclude", []))
    found: set[str] = set()
    for pattern in workspace["members"]:
        for directory in REPO_ROOT.glob(pattern):
            path = directory.relative_to(REPO_ROOT).as_posix()
            if path not in excluded and (directory / "pyproject.toml").is_file():
                found.add(path)
    return found


def test_the_suite_checks_every_workspace_member() -> None:
    """a member the walk misses would be checked by nothing."""
    members = _members()
    assert len(members) > 20, f"the workspace walk found only {sorted(members)}"
    checked = {package.path for package in packages()}
    assert members == checked, (
        f"members the typecheck suite does not check: {sorted(members - checked)}; "
        f"checked but not members: {sorted(checked - members)}"
    )


def test_a_bare_mypy_reads_the_same_source_trees() -> None:
    """``[tool.mypy] files`` and ``mypy_path`` name exactly the checked packages' ``src/`` trees."""
    want = {f"{package.path}/src" for package in packages()}
    mypy = _PYPROJECT["tool"]["mypy"]
    for key in ("files", "mypy_path"):
        have = set(mypy[key])
        assert have == want, f"[tool.mypy] {key}: missing {sorted(want - have)}, extra {sorted(have - want)}"


def test_the_typecheck_tests_run_in_their_own_suite_only() -> None:
    """the unit suite selects ``packages/`` and ``tests/``; ``typecheck/`` is the typecheck suite's alone."""
    suites = _MANIFEST["suites"]
    assert suites["typecheck"]["default_selection"] == ["typecheck/"]
    for name, suite in suites.items():
        if name == "typecheck":
            continue
        for path in suite.get("default_selection", []):
            assert not Path(path).as_posix().startswith("typecheck"), f"suite {name} selects {path}"
