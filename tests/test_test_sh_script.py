"""behavioural tests for ``scripts/test.sh``'s routing: what it runs, and where.

Two gaps sent runs local by accident. ``./scripts/test.sh evals <file>`` handed the pool the
whole evals package beside the file, so a one-file run took minutes and people stepped around
the script with a raw ``pytest`` or ``TESTAMENT=off``. And ``TESTAMENT=off`` went local without
saying why. These tests pin both: named test paths run alone on either route, and
``TESTAMENT=off`` demands ``TESTAMENT_OFF_REASON`` unless the run was going local anyway.
Testament's ``--area`` and ``--not-area`` reach ``testament run`` ahead of the pytest selection,
and a local run, which knows no areas, refuses them.

Self-contained: the script is staged in a miniature repo under ``tmp_path`` (it derives
REPO_ROOT from its own location), and ``testament`` and ``uv`` are stand-in executables on PATH
that record their argv, so nothing here runs a test or reaches the pool.

    ./scripts/test.sh tests/test_test_sh_script.py
"""

from __future__ import annotations

import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "test.sh"

_EVALS_FILE = "packages/evals/tests/test_transport.py"

#: a stand-in executable: writes its own name and then each argument, one per line
_RECORDER = '#!/bin/sh\n{{ basename "$0"; printf "%s\\n" "$@"; }} > "{calls}"\n'

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="test.sh is bash; none on this host")


@dataclass(frozen=True)
class _Run:
    """what one invocation of the staged script did.

    :param returncode: the script's exit status
    :ptype returncode: int
    :param stderr: what it printed to stderr
    :ptype stderr: str
    :param argv: the stand-in's name and arguments, or empty when neither was called
    :ptype argv: list[str]
    """

    returncode: int
    stderr: str
    argv: list[str]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A miniature repo: the script, two packages with tests, and a root pyproject.

    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :return: the repo root
    :rtype: Path
    """
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(_SCRIPT, root / "scripts" / "test.sh")
    (root / "scripts" / "test.sh").chmod(0o755)
    for package in ("evals", "core"):
        (root / "packages" / package / "tests").mkdir(parents=True)
    (root / _EVALS_FILE).write_text("def test_y() -> None:\n    pass\n")
    (root / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
    # files and a directory that options name as their values, all existing
    (root / "packages" / "core" / "src").mkdir()
    for name in ("out.log", "report.jsonl", "report.html", ".coveragerc"):
        (root / name).write_text("")
    (root / "packages" / "core" / "tests" / "out.log").write_text("")
    return root


def _run(repo: Path, args: list[str], *, testament: bool = True, env: dict[str, str] | None = None) -> _Run:
    """Run the staged script with recording stand-ins for ``uv`` and, optionally, ``testament``.

    :param repo: the miniature repo root
    :ptype repo: Path
    :param args: the script's arguments
    :ptype args: list[str]
    :param testament: whether a ``testament`` is on PATH
    :ptype testament: bool
    :param env: extra environment for the run
    :ptype env: dict[str, str] | None
    :return: what the run did
    :rtype: _Run
    """
    bin_dir = repo.parent / "bin"
    bin_dir.mkdir(exist_ok=True)
    calls = repo.parent / "calls.txt"
    calls.unlink(missing_ok=True)
    for name in ("uv", "testament") if testament else ("uv",):
        stand_in = bin_dir / name
        stand_in.write_text(_RECORDER.format(calls=calls))
        stand_in.chmod(0o755)
    result = subprocess.run(
        ["bash", str(repo / "scripts" / "test.sh"), *args],
        cwd=repo,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", **(env or {})},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    argv = calls.read_text().splitlines() if calls.exists() else []
    return _Run(result.returncode, result.stderr, argv)


class TestNamedTestPathsRunAlone:
    """A package name with test paths runs those paths, not the package's whole ``tests/``."""

    def test_a_file_goes_to_the_pool_alone(self, repo: Path) -> None:
        """the file, and not ``packages/evals/tests/`` beside it.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["evals", _EVALS_FILE, "-q"])
        assert run.returncode == 0, run.stderr
        assert run.argv == ["testament", "run", "--", _EVALS_FILE, "-q"]

    def test_a_node_id_goes_to_the_pool_alone(self, repo: Path) -> None:
        """``file.py::name`` counts as a test path when the file exists.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        node = f"{_EVALS_FILE}::test_y"
        run = _run(repo, ["evals", node])
        assert run.argv == ["testament", "run", "--", node]

    def test_a_file_runs_alone_on_the_local_route(self, repo: Path) -> None:
        """without testament, the same file runs locally and alone.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["evals", _EVALS_FILE, "-q"], testament=False)
        assert run.returncode == 0, run.stderr
        assert run.argv == ["uv", "run", "pytest", _EVALS_FILE, "-q"]

    def test_a_file_runs_alone_on_an_env_gated_local_run(self, repo: Path) -> None:
        """a recording run of one file records that file, not the whole suite.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, [_EVALS_FILE], env={"RECORD": "1"})
        assert run.argv == ["uv", "run", "pytest", _EVALS_FILE]
        assert "running locally, not via testament: an environment gate is set" in run.stderr


class TestAPackageWithOnlyFlagsRunsTheWholePackage:
    """Flags, and the values of flags, are not test paths."""

    def test_flags_only(self, repo: Path) -> None:
        """``evals -q`` is the whole package, as before.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["evals", "-q"])
        assert run.argv == ["testament", "run", "--", "packages/evals/tests/", "-q"]

    def test_a_flag_value_that_names_a_file_is_not_a_test_path(self, repo: Path) -> None:
        """``-c pyproject.toml`` names an existing file, but as the option's value.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["core", "-c", "pyproject.toml", "-k", "lease"])
        assert run.argv == ["testament", "run", "--", "packages/core/tests/", "-c", "pyproject.toml", "-k", "lease"]

    def test_a_word_that_is_not_a_path_is_not_a_test_path(self, repo: Path) -> None:
        """a ``-k`` expression's operand is passed through, not taken for a path.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["core", "-k", "lease", "nonexistent_test.py"])
        assert run.argv[:4] == ["testament", "run", "--", "packages/core/tests/"]


class TestAnOptionsValueIsNotATestPath:
    """An existing file or directory given as an option's value leaves the package's ``tests/`` in place.

    Taking one for a test path would drop ``packages/<package>/tests/`` and hand the pool every test
    the run's other arguments reach, or none, with the package name silently ignored.
    """

    @pytest.mark.parametrize(
        "options",
        [
            pytest.param(["--log-file", "out.log"], id="log-file"),
            pytest.param(["--report-log", "report.jsonl"], id="report-log"),
            pytest.param(["--html", "report.html"], id="html"),
            pytest.param(["--cov", "packages/core/src"], id="cov-dir-under-packages"),
            pytest.param(["--cov-config", ".coveragerc"], id="cov-config"),
            pytest.param(["--log-file", "packages/core/tests/out.log"], id="log-file-under-tests"),
            pytest.param(["--an-unlisted-option", "report.html"], id="unlisted-option"),
        ],
    )
    def test_the_package_still_runs_whole(self, repo: Path, options: list[str]) -> None:
        """each value names something that exists, and none is a test path.

        :param repo: the miniature repo root
        :ptype repo: Path
        :param options: an option and its value, written apart
        :ptype options: list[str]
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["core", *options, "-q"])
        assert run.argv == ["testament", "run", "--", "packages/core/tests/", *options, "-q"]

    def test_the_same_on_the_local_route(self, repo: Path) -> None:
        """without testament, ``--cov <dir>`` still runs the package's tests.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["core", "--cov", "packages/core/src"], testament=False)
        assert run.argv == ["uv", "run", "pytest", "packages/core/tests/", "--cov", "packages/core/src"]

    def test_an_option_after_cov_is_not_its_value(self, repo: Path) -> None:
        """``--cov``'s value is optional: ``--cov -q <file>`` still names the file.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["evals", "--cov", "-q", _EVALS_FILE])
        assert run.argv == ["testament", "run", "--", "--cov", "-q", _EVALS_FILE]


class TestWhatCountsAsATestPath:
    """A ``.py`` file, a node id or a directory, under ``packages/`` or ``tests/``, however it is spelled."""

    @pytest.mark.parametrize(
        "path",
        [
            pytest.param(f"./{_EVALS_FILE}", id="dot-slash"),
            pytest.param("packages/evals/tests", id="directory"),
            pytest.param("ABSOLUTE", id="absolute"),
        ],
    )
    def test_it_runs_alone(self, repo: Path, path: str) -> None:
        """each spelling replaces the package's ``tests/``.

        :param repo: the miniature repo root
        :ptype repo: Path
        :param path: the test path as given, ``ABSOLUTE`` for the file's absolute path
        :ptype path: str
        :return: nothing
        :rtype: None
        """
        if path == "ABSOLUTE":
            path = str(repo / _EVALS_FILE)
        run = _run(repo, ["evals", path])
        assert run.argv == ["testament", "run", "--", path]

    def test_a_non_python_file_outside_the_test_trees_is_not_one(self, repo: Path) -> None:
        """``pyproject.toml`` exists, but it names no tests.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["core", "pyproject.toml"])
        assert run.argv == ["testament", "run", "--", "packages/core/tests/", "pyproject.toml"]


class TestTestamentOffNeedsAReason:
    """``TESTAMENT=off`` takes a run off the pool only with ``TESTAMENT_OFF_REASON`` set."""

    def test_without_a_reason_it_exits_2_and_runs_nothing(self, repo: Path) -> None:
        """the message says how to run on the pool and how to state a reason.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["evals", _EVALS_FILE], env={"TESTAMENT": "off"})
        assert run.returncode == 2
        assert run.argv == []
        assert "TESTAMENT_OFF_REASON" in run.stderr
        assert "./scripts/test.sh <package> <test file>" in run.stderr
        assert len(run.stderr.strip().splitlines()) == 1

    def test_an_empty_reason_is_no_reason(self, repo: Path) -> None:
        """an empty ``TESTAMENT_OFF_REASON`` is refused like a missing one.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["core"], env={"TESTAMENT": "off", "TESTAMENT_OFF_REASON": ""})
        assert run.returncode == 2
        assert run.argv == []

    def test_with_a_reason_it_runs_locally_and_says_why(self, repo: Path) -> None:
        """the reason is printed on the line every local-only run prints.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["core"], env={"TESTAMENT": "off", "TESTAMENT_OFF_REASON": "debugging under pdb"})
        assert run.returncode == 0, run.stderr
        assert "test.sh: running locally, not via testament: debugging under pdb" in run.stderr
        assert run.argv == ["uv", "run", "pytest", "packages/core/tests/"]

    @pytest.mark.parametrize(
        ("args", "env", "testament"),
        [
            pytest.param(["core"], {"RECORD": "1"}, True, id="env-gate"),
            pytest.param(["core"], {"CI": "true"}, True, id="ci"),
            pytest.param(["core"], {}, False, id="no-testament"),
        ],
    )
    def test_the_automatic_local_routes_need_no_reason(
        self, repo: Path, args: list[str], env: dict[str, str], testament: bool
    ) -> None:
        """a run that goes local on its own does not ask why.

        :param repo: the miniature repo root
        :ptype repo: Path
        :param args: the script's arguments
        :ptype args: list[str]
        :param env: extra environment for the run
        :ptype env: dict[str, str]
        :param testament: whether a ``testament`` is on PATH
        :ptype testament: bool
        :return: nothing
        :rtype: None
        """
        run = _run(repo, args, testament=testament, env={"TESTAMENT": "off", **env})
        assert run.returncode == 0, run.stderr
        assert run.argv[:3] == ["uv", "run", "pytest"]


class TestAreasGoToTestament:
    """``--area`` and ``--not-area`` are testament's flags: they go ahead of the pytest selection."""

    @pytest.mark.parametrize(
        ("args", "argv"),
        [
            pytest.param(["--not-area", "evals"], ["testament", "run", "--not-area", "evals", "--"], id="not-area"),
            pytest.param(["--area=core", "-q"], ["testament", "run", "--area=core", "--", "-q"], id="joined-value"),
            pytest.param(
                ["--not-area", "evals", "--not-area", "enforcement", "-q"],
                ["testament", "run", "--not-area", "evals", "--not-area", "enforcement", "--", "-q"],
                id="repeated",
            ),
            pytest.param(
                ["core", "--area", "core"],
                ["testament", "run", "--area", "core", "--", "packages/core/tests/"],
                id="with-a-package",
            ),
            pytest.param(
                ["evals", _EVALS_FILE, "--area", "evals"],
                ["testament", "run", "--area", "evals", "--", _EVALS_FILE],
                id="with-a-test-path",
            ),
        ],
    )
    def test_they_reach_testament_not_pytest(self, repo: Path, args: list[str], argv: list[str]) -> None:
        """the area flags and their values come before ``--``; everything else after it, as before.

        :param repo: the miniature repo root
        :ptype repo: Path
        :param args: the script's arguments
        :ptype args: list[str]
        :param argv: what testament is called with
        :ptype argv: list[str]
        :return: nothing
        :rtype: None
        """
        run = _run(repo, args)
        assert run.returncode == 0, run.stderr
        assert run.argv == argv

    def test_a_flag_without_its_name_runs_nothing(self, repo: Path) -> None:
        """``--area`` last on the line names no area, so nothing runs.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["-q", "--area"])
        assert run.returncode == 2
        assert run.argv == []
        assert "--area needs an area name" in run.stderr

    @pytest.mark.parametrize(
        ("args", "env", "testament"),
        [
            pytest.param(["--not-area", "evals"], {}, False, id="no-testament"),
            pytest.param(["--not-area", "evals"], {"CI": "true"}, True, id="ci"),
            pytest.param(["--not-area", "evals"], {"RECORD": "1"}, True, id="env-gate"),
            pytest.param(
                ["--not-area", "evals"], {"TESTAMENT": "off", "TESTAMENT_OFF_REASON": "debugging"}, True, id="off"
            ),
        ],
    )
    def test_a_local_run_refuses_them(self, repo: Path, args: list[str], env: dict[str, str], testament: bool) -> None:
        """only the pool knows the areas, so a local run refuses rather than run the whole suite.

        :param repo: the miniature repo root
        :ptype repo: Path
        :param args: the script's arguments
        :ptype args: list[str]
        :param env: extra environment for the run
        :ptype env: dict[str, str]
        :param testament: whether a ``testament`` is on PATH
        :ptype testament: bool
        :return: nothing
        :rtype: None
        """
        run = _run(repo, args, testament=testament, env=env)
        assert run.returncode == 2
        assert run.argv == []
        assert "--area and --not-area select on the testament pool" in run.stderr


class TestDashMGoesToThePool:
    """testament narrows the manifest's marker expression by a ``-m`` given here, so ``-m`` no longer keeps a run local."""

    def test_a_package_run_with_dash_m(self, repo: Path) -> None:
        """the ``-m`` rides after the package's tests, for testament to AND with the manifest's own.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["core", "-m", "slow"])
        assert run.returncode == 0, run.stderr
        assert run.argv == ["testament", "run", "--", "packages/core/tests/", "-m", "slow"]

    def test_with_areas(self, repo: Path) -> None:
        """areas and ``-m`` together reach testament.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["--not-area", "evals", "-m", "slow"])
        assert run.returncode == 0, run.stderr
        assert run.argv == ["testament", "run", "--not-area", "evals", "--", "-m", "slow"]

    def test_a_local_whole_suite_run_keeps_the_callers_marks(self, repo: Path) -> None:
        """on the local route the caller's ``-m`` is the selection, as before.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["-m", "slow"], env={"TESTAMENT": "off", "TESTAMENT_OFF_REASON": "debugging"})
        assert run.returncode == 0, run.stderr
        assert run.argv[:4] == ["uv", "run", "pytest", "packages/"]
        assert run.argv[-2:] == ["-m", "slow"]


class TestFlagsThatNeedOneLocalProcess:
    """``-x``, ``--maxfail``, ``--lf``, ``-n`` and their kin: the pool can't take them, and the manifest is pool-only.

    A run naming only test files runs here with them; a wider one is refused rather than load this machine.
    """

    @pytest.mark.parametrize(
        "flags",
        [
            pytest.param(["-x"], id="x"),
            pytest.param(["-vx"], id="bundle"),
            pytest.param(["--maxfail", "2"], id="maxfail"),
            pytest.param(["--maxfail=2"], id="maxfail-joined"),
            pytest.param(["--lf"], id="lf"),
            pytest.param(["--last-failed"], id="last-failed"),
            pytest.param(["-n", "4"], id="xdist"),
            pytest.param(["-n4"], id="xdist-joined"),
            pytest.param(["--numprocesses=auto"], id="numprocesses"),
            pytest.param(["--ff"], id="failed-first"),
            pytest.param(["--sw"], id="stepwise"),
            pytest.param(["--sw-skip"], id="stepwise-skip"),
            pytest.param(["-xvs"], id="bundle-x-first"),
            pytest.param(["-xrs"], id="bundle-before-a-value"),
            pytest.param(["-s"], id="no-capture"),
            pytest.param(["-vs"], id="no-capture-bundle"),
            pytest.param(["-srx"], id="no-capture-before-a-value"),
            pytest.param(["--capture=no"], id="capture-no"),
            pytest.param(["--capture", "no"], id="capture-no-apart"),
            pytest.param(["--pdb"], id="pdb"),
            pytest.param(["--trace"], id="trace"),
        ],
    )
    def test_a_package_run_is_refused(self, repo: Path, flags: list[str]) -> None:
        """a whole package with such a flag runs nothing, and says how to run it.

        :param repo: the miniature repo root
        :ptype repo: Path
        :param flags: the flags given
        :ptype flags: list[str]
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["core", *flags])
        assert run.returncode == 2
        assert run.argv == []
        assert "needs one local pytest process" in run.stderr
        assert "name the files" in run.stderr

    def test_a_directory_is_wider_than_named_files(self, repo: Path) -> None:
        """a directory selection with ``-x`` is refused too.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["evals", "packages/evals/tests", "-x"])
        assert run.returncode == 2
        assert run.argv == []

    @pytest.mark.parametrize(
        "selection",
        [pytest.param([_EVALS_FILE], id="file"), pytest.param([f"{_EVALS_FILE}::test_y"], id="node-id")],
    )
    def test_named_test_files_run_here_with_them(self, repo: Path, selection: list[str]) -> None:
        """named files with ``-x`` run locally, exactly as named, and say why.

        :param repo: the miniature repo root
        :ptype repo: Path
        :param selection: the named tests
        :ptype selection: list[str]
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["evals", *selection, "-x"])
        assert run.returncode == 0, run.stderr
        assert run.argv == ["uv", "run", "pytest", *selection, "-x"]
        assert "running locally, not via testament: -x needs one local pytest process" in run.stderr

    @pytest.mark.parametrize(
        "flags",
        [
            pytest.param(["-rx"], id="report-chars"),
            pytest.param(["-rsx"], id="report-chars-with-s"),
            pytest.param(["-vv"], id="verbose"),
            pytest.param(["--capture", "sys"], id="capture-sys"),
            pytest.param(["-p", "no:randomly"], id="plugin"),
        ],
    )
    def test_flags_that_keep_nothing_local_go_to_the_pool(self, repo: Path, flags: list[str]) -> None:
        """``-rx`` asks for a report of xfails, not a stop at the first failure; these go to the pool.

        :param repo: the miniature repo root
        :ptype repo: Path
        :param flags: the flags given
        :ptype flags: list[str]
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["core", *flags])
        assert run.returncode == 0, run.stderr
        assert run.argv[:2] == ["testament", "run"]

    def test_areas_with_a_local_flag_are_refused(self, repo: Path) -> None:
        """named files with ``-x`` run here, where no areas exist, so an area flag beside them is refused.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["evals", _EVALS_FILE, "--area", "evals", "-x"])
        assert run.returncode == 2
        assert run.argv == []
        assert "-x keeps this run local; name test files instead" in run.stderr

    @pytest.mark.parametrize(
        ("env", "testament"),
        [
            pytest.param({"TESTAMENT": "off", "TESTAMENT_OFF_REASON": "debugging"}, True, id="off-with-reason"),
            pytest.param({"RECORD": "1"}, True, id="env-gate"),
            pytest.param({"CI": "true"}, True, id="ci"),
            pytest.param({}, False, id="no-testament"),
        ],
    )
    def test_a_chosen_local_run_keeps_them(self, repo: Path, env: dict[str, str], testament: bool) -> None:
        """a run already going local takes the flag as it always did.

        :param repo: the miniature repo root
        :ptype repo: Path
        :param env: extra environment for the run
        :ptype env: dict[str, str]
        :param testament: whether a ``testament`` is on PATH
        :ptype testament: bool
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["core", "-x"], testament=testament, env=env)
        assert run.returncode == 0, run.stderr
        assert run.argv == ["uv", "run", "pytest", "packages/core/tests/", "-x"]


class TestTheManifest:
    """the pool's own settings that this script's routing relies on."""

    def test_the_pool_is_pool_only_and_leaves_out_integration_and_loop_timing(self) -> None:
        """a pool outage fails rather than run the suite here, and neither local-only kind reaches a runner.

        :return: nothing
        :rtype: None
        """
        manifest = tomllib.loads((_REPO_ROOT / ".testament.toml").read_text())
        assert manifest["fallback"] == "pool-only"
        command = manifest["command"]
        marks = command[command.index("-m") + 1]
        assert "not integration" in marks and "not loop_timing" in marks
