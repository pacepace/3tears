"""behavioural tests for ``scripts/typecheck.sh``: what it checks, and where.

The typecheck is testament's ``typecheck`` suite (``typecheck/test_mypy.py``). Where testament is
installed the script hands the run to the pool, whose cache serves a tree that already passed, so a run
here and CI's run of the same tree check it once; CI, and ``TESTAMENT=off`` with a stated reason, run the
same tests here. The suite ships no release tags (``[suites.typecheck.git] tags = []``), so nothing about
this checkout's tags matters to the cache, and the script reads none.

Self-contained: the script is staged in a miniature repo under ``tmp_path`` (it derives REPO_ROOT from
its own location), and ``testament``, ``uv`` and ``git`` are stand-ins on PATH that record their argv, so
nothing here runs mypy or reaches the pool.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: a pool run of the whole suite
_POOL = ["testament", "run", "--suite", "typecheck", "--"]

#: a stand-in that appends its name and each argument, one per line, then a blank line; exits ``$<NAME>_EXIT``
_RECORDER = """#!/bin/sh
{{ basename "$0"; printf '%s\\n' "$@"; echo; }} >> "{calls}"
exit "${{{var}:-0}}"
"""

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="typecheck.sh is bash; none on this host")


@dataclass(frozen=True)
class _Run:
    """what one invocation of the staged script did.

    :param returncode: the script's exit status
    :ptype returncode: int
    :param stderr: what it printed to stderr
    :ptype stderr: str
    :param calls: each stand-in call, as its name followed by its arguments
    :ptype calls: list[list[str]]
    """

    returncode: int
    stderr: str
    calls: list[list[str]]

    def of(self, name: str) -> list[list[str]]:
        """the calls one stand-in received.

        :param name: ``testament``, ``uv`` or ``git``
        :ptype name: str
        :return: those calls
        :rtype: list[list[str]]
        """
        return [call for call in self.calls if call[0] == name]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A miniature repo: the script and its helper, and three packages with ``src/threetears`` trees.

    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :return: the repo root
    :rtype: Path
    """
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    for name in ("typecheck.sh", "_tool.sh"):
        shutil.copy(_REPO_ROOT / "scripts" / name, root / "scripts" / name)
    for package, leaf in (("core", "core"), ("evals", "evals"), ("agent/memory", "agent/memory")):
        module = root / "packages" / package / "src" / "threetears" / leaf
        module.mkdir(parents=True)
        (module / "__init__.py").write_text("")
    return root


def _run(
    repo: Path,
    args: list[str],
    *,
    testament: bool = True,
    env: dict[str, str] | None = None,
    testament_exit: int = 0,
) -> _Run:
    """Run the staged script with recording stand-ins for ``uv``, ``git`` and, optionally, ``testament``.

    :param repo: the miniature repo root
    :ptype repo: Path
    :param args: the script's arguments
    :ptype args: list[str]
    :param testament: whether a ``testament`` is on PATH
    :ptype testament: bool
    :param env: extra environment for the run
    :ptype env: dict[str, str] | None
    :param testament_exit: the exit status the ``testament`` stand-in returns
    :ptype testament_exit: int
    :return: what the run did
    :rtype: _Run
    """
    stage = repo.parent
    bin_dir = stage / "bin"
    bin_dir.mkdir(exist_ok=True)
    calls = stage / "calls.txt"
    calls.unlink(missing_ok=True)
    for name in ("uv", "testament") if testament else ("uv",):
        (bin_dir / name).write_text(_RECORDER.format(calls=calls, var=f"{name.upper()}_EXIT"))
        (bin_dir / name).chmod(0o755)
    if not testament:
        (bin_dir / "testament").unlink(missing_ok=True)
    (bin_dir / "git").write_text(_RECORDER.format(calls=calls, var="GIT_EXIT"))
    (bin_dir / "git").chmod(0o755)
    result = subprocess.run(
        ["bash", str(repo / "scripts" / "typecheck.sh"), *args],
        cwd=repo,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "TESTAMENT_EXIT": str(testament_exit), **(env or {})},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    text = calls.read_text() if calls.exists() else ""
    recorded = [block.splitlines() for block in text.split("\n\n") if block.strip()]
    return _Run(result.returncode, result.stderr, recorded)


class TestThePool:
    """where testament is installed, the typecheck suite runs on the pool, selected by package."""

    def test_every_package_runs_as_the_typecheck_suite(self, repo: Path) -> None:
        run = _run(repo, [])
        assert run.returncode == 0, run.stderr
        assert run.of("testament") == [[*_POOL, "typecheck/"]]
        assert run.of("uv") == [], "a pool run checks nothing here"
        assert run.of("git") == [], "the suite ships no tags, so the script reads none"

    @pytest.mark.parametrize(
        ("package", "selection"),
        [
            pytest.param("core", ["typecheck/test_mypy.py::test_package_typechecks[core]"], id="core"),
            pytest.param(
                "agent/memory", ["typecheck/test_mypy.py::test_package_typechecks[agent.memory]"], id="agent-memory"
            ),
            pytest.param(
                "evals",
                [
                    "typecheck/test_mypy.py::test_package_typechecks[evals]",
                    "typecheck/test_mypy.py::test_evals_example_hosts_typecheck",
                ],
                id="evals-and-its-example-hosts",
            ),
        ],
    )
    def test_a_package_runs_its_own_test(self, repo: Path, package: str, selection: list[str]) -> None:
        """a package name selects its test, by the id typecheck/targets.py gives it (dots for slashes).

        :param repo: the miniature repo
        :ptype repo: Path
        :param package: the package argument
        :ptype package: str
        :param selection: the node ids the run must select
        :ptype selection: list[str]
        :return: nothing
        :rtype: None
        """
        run = _run(repo, [package])
        assert run.of("testament") == [[*_POOL, *selection]], run.stderr

    def test_the_arguments_after_a_package_go_to_pytest(self, repo: Path) -> None:
        run = _run(repo, ["core", "-v"])
        assert run.of("testament") == [[*_POOL, "typecheck/test_mypy.py::test_package_typechecks[core]", "-v"]]

    def test_an_unknown_package_is_refused(self, repo: Path) -> None:
        run = _run(repo, ["nosuch"])
        assert run.returncode == 2
        assert "no package 'packages/nosuch'" in run.stderr
        assert run.of("testament") == [] and run.of("uv") == []

    @pytest.mark.parametrize(
        ("code", "said"),
        [
            pytest.param(75, "testament exit 75: the pool could not serve this run", id="pool-could-not-serve"),
            pytest.param(2, "testament gave no verdict (exit 2)", id="no-verdict"),
        ],
    )
    def test_no_verdict_exits_2_and_says_how_to_go_on(self, repo: Path, code: int, said: str) -> None:
        """the pool's no-verdict endings are this script's exit 2, as test.sh's are, and run nothing here.

        :param repo: the miniature repo
        :ptype repo: Path
        :param code: testament's exit status
        :ptype code: int
        :param said: what the script must say
        :ptype said: str
        :return: nothing
        :rtype: None
        """
        run = _run(repo, [], testament_exit=code)
        assert run.returncode == 2
        assert said in run.stderr
        assert "TESTAMENT=off" in run.stderr
        assert run.of("uv") == []

    def test_a_type_error_fails_the_run(self, repo: Path) -> None:
        run = _run(repo, [], testament_exit=1)
        assert run.returncode == 1
        assert run.of("uv") == []


class TestHere:
    """CI, TESTAMENT=off with a reason, and a machine without testament run the same tests here."""

    @pytest.mark.parametrize(
        ("testament", "env"),
        [
            pytest.param(True, {"CI": "true"}, id="ci"),
            pytest.param(True, {"TESTAMENT": "off", "TESTAMENT_OFF_REASON": "pool down"}, id="testament-off"),
            pytest.param(False, {}, id="no-testament"),
        ],
    )
    def test_the_suite_runs_here(self, repo: Path, testament: bool, env: dict[str, str]) -> None:
        """the run here is pytest over the same selection.

        :param repo: the miniature repo
        :ptype repo: Path
        :param testament: whether a ``testament`` is on PATH
        :ptype testament: bool
        :param env: the environment that sends the run here
        :ptype env: dict[str, str]
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["agent/memory", "-q"], testament=testament, env=env)
        assert run.returncode == 0, run.stderr
        assert run.of("testament") == []
        assert run.of("uv") == [
            ["uv", "run", "pytest", "typecheck/test_mypy.py::test_package_typechecks[agent.memory]", "-q"]
        ]
        assert run.of("git") == []

    def test_testament_off_needs_a_reason(self, repo: Path) -> None:
        run = _run(repo, [], env={"TESTAMENT": "off"})
        assert run.returncode == 2
        assert "TESTAMENT=off needs a reason" in run.stderr
        assert run.of("uv") == [] and run.of("testament") == []

    def test_the_reason_is_said(self, repo: Path) -> None:
        run = _run(repo, [], env={"TESTAMENT": "off", "TESTAMENT_OFF_REASON": "pool down"})
        assert "running locally, not via testament: pool down" in run.stderr
