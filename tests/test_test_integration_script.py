"""behavioural tests for ``scripts/test-integration.sh``'s routing: which integration tests run on the pool, which here.

The integration suite moves to the testament pool one package at a time. Its paths are the manifest's
``[suites.integration] default_selection``, and they go to the pool through ``scripts/test.sh --suite
integration``; the script's ``LOCAL_DIRS`` run here through testcontainers, as every integration test once
did. These tests pin the split: a package goes to one route or the other, a package with paths on both
goes to both without running anything twice, named paths follow the path they lie in, and where the pool
route is not taken (no testament, CI, ``TESTAMENT=off`` with a reason) everything runs here.

Self-contained: both scripts are staged in a miniature repo under ``tmp_path`` beside a manifest of its
own, and ``testament`` and ``uv`` are stand-in executables on PATH that record their argv, so nothing
here runs a test or reaches the pool.

    ./scripts/test.sh tests/test_test_integration_script.py
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

_SUITE_FILE = "packages/core/tests/integration/test_process_services.py"
_SUITE_DIR = "packages/scheduled-jobs/tests/integration"
_CORE_DIR = "packages/core/tests/integration"
_NATS_DIR = "packages/nats/tests/integration"

_MANIFEST = f"""manifest_version = 3
name = "fixture"

[suites.default]
adapter = "pytest"
command = ["uv", "run", "pytest", "-m", "not integration"]
default_selection = ["packages/", "tests/"]
default_for_paths = true

[suites.integration]
adapter = "pytest"
command = ["uv", "run", "pytest", "-m", "integration"]
default_selection = ["{_SUITE_FILE}", "{_SUITE_DIR}/"]
"""

#: a stand-in executable: appends one line to ``calls``, its name and then each argument tab-separated, and one
#: to ``gates``, its name and then every variable it was given named like one of test.sh's environment gates
#: (starting RECORD, TAVILY or THREETEARS_LIVE); then exits ``code``
_RECORDER = (
    "#!/bin/sh\n"
    '{{ printf "%s" "$(basename "$0")"; printf "\\t%s" "$@"; printf "\\n"; }} >> "{calls}"\n'
    '{{ printf "%s" "$(basename "$0")"; for gate in $(env | cut -d= -f1 | grep -E "^(RECORD|TAVILY|THREETEARS_LIVE)" | sort); do printf "\\t%s" "$gate"; done; printf "\\n"; }} >> "{gates}"\n'
    "exit {code}\n"
)

#: a test file the scripts count as an integration test
_MARKED = "import pytest\n\npytestmark = pytest.mark.integration\n\n\ndef test_y() -> None:\n    pass\n"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="the scripts are bash; none on this host")


@dataclass(frozen=True)
class _Run:
    """what one invocation of the staged script did.

    :param returncode: the script's exit status
    :ptype returncode: int
    :param stderr: what it printed to stderr
    :ptype stderr: str
    :param calls: each stand-in call, its name first, in order
    :ptype calls: list[list[str]]
    :param gates: each stand-in call's name, then the environment gates it saw
    :ptype gates: list[list[str]]
    :param stdout: what it printed to stdout
    :ptype stdout: str
    """

    returncode: int
    stderr: str
    calls: list[list[str]]
    gates: list[list[str]]
    stdout: str = ""

    def gates_seen(self, name: str) -> list[str]:
        """the environment gates the stand-in ``name`` saw when it was called.

        :param name: ``testament`` or ``uv``
        :ptype name: str
        :return: the gates' names
        :rtype: list[str]
        """
        found = [call[1:] for call in self.gates if call[0] == name]
        assert len(found) == 1, f"{name} called {len(found)} times"
        return found[0]

    def call(self, name: str) -> list[str] | None:
        """the one call of the stand-in ``name``, or None when it was not called.

        :param name: ``testament`` or ``uv``
        :ptype name: str
        :return: its argv after its name
        :rtype: list[str] | None
        """
        found = [call[1:] for call in self.calls if call[0] == name]
        assert len(found) <= 1, f"{name} called more than once: {found}"
        return found[0] if found else None


def _local_dirs() -> list[str]:
    """the script's own ``LOCAL_DIRS``, as written.

    :return: the paths
    :rtype: list[str]
    """
    text = (_REPO_ROOT / "scripts" / "test-integration.sh").read_text()
    block = text.split("LOCAL_DIRS=(\n", 1)[1].split("\n)\n", 1)[0]
    return [line.strip() for line in block.splitlines() if line.strip()]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A miniature repo: both scripts, a manifest with an integration suite, and the paths it names.

    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :return: the repo root
    :rtype: Path
    """
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    for name in ("test.sh", "test-integration.sh"):
        shutil.copy(_REPO_ROOT / "scripts" / name, root / "scripts" / name)
        (root / "scripts" / name).chmod(0o755)
    (root / ".testament.toml").write_text(_MANIFEST)
    for directory in (_CORE_DIR, _SUITE_DIR, _NATS_DIR, "packages/models/tests"):
        (root / directory).mkdir(parents=True)
    for path in (_SUITE_FILE, f"{_CORE_DIR}/test_other.py", f"{_SUITE_DIR}/test_jobs.py", f"{_NATS_DIR}/test_bus.py"):
        (root / path).write_text(_MARKED)
    # every local path the script lists holds an integration test, as in the repo
    for local in _local_dirs():
        target = root / local if local.endswith(".py") else root / local / "test_local.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(_MARKED)
    # a package's own unit tests, beside its integration directory
    (root / "packages" / "scheduled-jobs" / "tests" / "test_unit.py").write_text("def test_u() -> None:\n    pass\n")
    return root


def _run(
    repo: Path,
    args: list[str],
    *,
    testament: bool = True,
    env: dict[str, str] | None = None,
    testament_exit: int = 0,
    uv_exit: int = 0,
    script: str = "test-integration.sh",
) -> _Run:
    """Run a staged script with recording stand-ins for ``uv`` and, optionally, ``testament``.

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
    :param uv_exit: the exit status the ``uv`` stand-in returns
    :ptype uv_exit: int
    :param script: the staged script to run, ``test-integration.sh`` or ``test.sh``
    :ptype script: str
    :return: what the run did
    :rtype: _Run
    """
    bin_dir = repo.parent / "bin"
    bin_dir.mkdir(exist_ok=True)
    calls = repo.parent / "calls.txt"
    calls.unlink(missing_ok=True)
    gates = repo.parent / "gates.txt"
    gates.unlink(missing_ok=True)
    codes = {"uv": uv_exit, "testament": testament_exit}
    for name in ("uv", "testament") if testament else ("uv",):
        stand_in = bin_dir / name
        stand_in.write_text(_RECORDER.format(calls=calls, gates=gates, code=codes[name]))
        stand_in.chmod(0o755)
    python = shutil.which("python3")
    assert python is not None
    (bin_dir / "python3").unlink(missing_ok=True)
    (bin_dir / "python3").symlink_to(python)
    result = subprocess.run(
        ["bash", str(repo / "scripts" / script), *args],
        cwd=repo,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", **(env or {})},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    lines = calls.read_text().splitlines() if calls.exists() else []
    seen = gates.read_text().splitlines() if gates.exists() else []
    return _Run(
        result.returncode,
        result.stderr,
        [line.split("\t") for line in lines],
        [line.split("\t") for line in seen],
        result.stdout,
    )


_POOL = ["run", "--suite", "integration", "--"]


class TestAPackageGoesWhereItsTestsAre:
    """a package in the suite runs on the pool, one in ``LOCAL_DIRS`` runs here, and one in both runs in both."""

    def test_a_package_in_the_suite_runs_on_the_pool_only(self, repo: Path) -> None:
        """scheduled-jobs' integration directory is the suite's: the pool runs it, nothing runs here.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["scheduled-jobs", "-rs"])
        assert run.returncode == 0, run.stderr
        assert run.call("testament") == [*_POOL, _SUITE_DIR, "-rs"]
        assert run.call("uv") is None

    def test_a_package_not_moved_runs_here_only(self, repo: Path) -> None:
        """nats is in ``LOCAL_DIRS``: it runs here through testcontainers, as before.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["nats"])
        assert run.returncode == 0, run.stderr
        assert run.call("testament") is None
        assert run.call("uv") == ["run", "pytest", _NATS_DIR, "-m", "integration"]

    def test_a_package_on_both_runs_each_part_once(self, repo: Path) -> None:
        """core's suite file runs on the pool and is left out of core's local run.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["core", "-v"])
        assert run.returncode == 0, run.stderr
        assert run.call("testament") == [*_POOL, _SUITE_FILE, "-v"]
        assert run.call("uv") == ["run", "pytest", _CORE_DIR, "-m", "integration", f"--ignore={_SUITE_FILE}", "-v"]

    def test_a_package_with_no_integration_tests_runs_nothing(self, repo: Path) -> None:
        """models has no integration test: nothing runs, and the exit is pytest's "no tests" 5.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["models"])
        assert run.returncode == 5
        assert run.calls == []
        assert "no integration tests in what was selected" in run.stderr

    def test_a_package_root_whose_integration_tests_are_the_suite_s_runs_on_the_pool_only(self, repo: Path) -> None:
        """``packages/scheduled-jobs`` named as a path: its integration directory goes to the pool, and the rest of
        the package holds no integration test, so nothing runs here (no local run exiting 5 after a green pool).

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["packages/scheduled-jobs"])
        assert run.returncode == 0, run.stderr
        assert run.call("testament") == [*_POOL, _SUITE_DIR]
        assert run.call("uv") is None


class TestWhatCountsAsAnIntegrationTest:
    """the script reads marks as the selection enforcement does: a ``pytest.mark.integration`` applied, not the words."""

    def test_a_package_in_neither_list_runs_here_whole(self, repo: Path) -> None:
        """a package whose integration test is in neither list runs here whole, as before the pool; its mark sits
        on a class, which counts.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        (repo / "packages" / "newpkg" / "tests").mkdir(parents=True)
        (repo / "packages" / "newpkg" / "tests" / "test_live.py").write_text(
            "import pytest\n\n\n@pytest.mark.integration\nclass TestLive:\n    def test_y(self) -> None:\n        pass\n"
        )
        run = _run(repo, ["newpkg", "-q"])
        assert run.returncode == 0, run.stderr
        assert run.call("testament") is None
        assert run.call("uv") == ["run", "pytest", "packages/newpkg", "-m", "integration", "-q"]

    def test_the_words_in_a_docstring_are_no_mark(self, repo: Path) -> None:
        """a module that only mentions ``pytest.mark.integration`` holds no integration test: nothing runs.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        (repo / "packages" / "prose" / "tests").mkdir(parents=True)
        (repo / "packages" / "prose" / "tests" / "test_words.py").write_text(
            '"""these are unit tests; the live ones carry ``pytest.mark.integration``."""\n\n\n'
            "def test_y() -> None:\n    pass\n"
        )
        run = _run(repo, ["prose"])
        assert run.returncode == 5
        assert run.calls == []


class TestTheDryRun:
    """``--dry-run`` prints what each route would run, and runs nothing."""

    def test_both_commands_and_nothing_run(self, repo: Path) -> None:
        """the pool's command and the local one, with the options on each.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["--dry-run", "-rs"])
        assert run.returncode == 0, run.stderr
        assert run.calls == []
        assert run.stdout.splitlines() == [
            f"on the pool: ./scripts/test.sh --suite integration {_SUITE_FILE} {_SUITE_DIR} -rs",
            f"here: uv run pytest {' '.join(_local_dirs())} -m integration --ignore={_SUITE_FILE} -rs",
        ]

    def test_a_pool_only_package_prints_one_line(self, repo: Path) -> None:
        """scheduled-jobs has nothing local.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["--dry-run", "scheduled-jobs"])
        assert run.stdout.splitlines() == [f"on the pool: ./scripts/test.sh --suite integration {_SUITE_DIR}"]


class TestAWholeRun:
    """no package: the suite's paths on the pool, then every local directory here."""

    def test_both_routes_with_the_options_on_each(self, repo: Path) -> None:
        """the pool takes the manifest's paths; here, every ``LOCAL_DIRS`` entry, less the suite's file.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["-rs"])
        assert run.returncode == 0, run.stderr
        assert run.call("testament") == [*_POOL, _SUITE_FILE, _SUITE_DIR, "-rs"]
        assert run.call("uv") == [
            "run",
            "pytest",
            *_local_dirs(),
            "-m",
            "integration",
            f"--ignore={_SUITE_FILE}",
            "-rs",
        ]
        assert "pool exit 0, local exit 0" in run.stderr

    @pytest.mark.parametrize(("pool", "local", "status"), [(1, 0, 1), (0, 1, 1), (2, 0, 2), (2, 1, 2)])
    def test_either_route_failing_fails_the_run_and_no_verdict_wins(
        self, repo: Path, pool: int, local: int, status: int
    ) -> None:
        """a red on either route is a red; the pool's exit 2 (no verdict) is what the run reports.

        :param repo: the miniature repo root
        :ptype repo: Path
        :param pool: the pool's exit status
        :ptype pool: int
        :param local: the local run's exit status
        :ptype local: int
        :param status: the script's expected exit status
        :ptype status: int
        :return: nothing
        :rtype: None
        """
        run = _run(repo, [], testament_exit=pool, uv_exit=local)
        assert run.returncode == status
        assert run.call("uv") is not None, "the local part runs whatever the pool said"


class TestNamedPaths:
    """test paths follow the route of the path they lie in."""

    def test_a_file_in_a_suite_directory_runs_on_the_pool_alone(self, repo: Path) -> None:
        """one scheduled-jobs file: the pool, that file only.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        path = f"{_SUITE_DIR}/test_jobs.py"
        run = _run(repo, ["scheduled-jobs", path, "-q"])
        assert run.call("testament") == [*_POOL, path, "-q"]
        assert run.call("uv") is None

    def test_a_suite_path_named_itself_runs_once(self, repo: Path) -> None:
        """naming the suite's file directly sends it once.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, [_SUITE_FILE])
        assert run.call("testament") == [*_POOL, _SUITE_FILE]
        assert run.call("uv") is None

    def test_a_local_file_runs_here_alone(self, repo: Path) -> None:
        """a nats file runs here, alone.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        path = f"{_NATS_DIR}/test_bus.py"
        run = _run(repo, [path])
        assert run.call("testament") is None
        assert run.call("uv") == ["run", "pytest", path, "-m", "integration"]

    def test_a_directory_holding_a_suite_file_splits(self, repo: Path) -> None:
        """core's integration directory, named: its suite file on the pool, the rest here.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, [_CORE_DIR])
        assert run.call("testament") == [*_POOL, _SUITE_FILE]
        assert run.call("uv") == ["run", "pytest", _CORE_DIR, "-m", "integration", f"--ignore={_SUITE_FILE}"]

    @pytest.mark.parametrize("path", ["README.md", "packages/core/tests/integration/test_missing.py"])
    def test_a_path_that_is_not_a_test_path_is_refused(self, repo: Path, path: str) -> None:
        """outside packages/ and tests/, or missing: refused, nothing run.

        :param repo: the miniature repo root
        :ptype repo: Path
        :param path: the bad path
        :ptype path: str
        :return: nothing
        :rtype: None
        """
        run = _run(repo, [path])
        assert run.returncode == 2
        assert run.calls == []


class TestWhenThePoolRouteIsNotTaken:
    """no testament, CI, or TESTAMENT=off with a reason: everything selected runs here, as before the pool."""

    @pytest.mark.parametrize(
        ("env", "testament"),
        [
            pytest.param({"TESTAMENT": "off", "TESTAMENT_OFF_REASON": "the pool is down"}, True, id="off-with-reason"),
            pytest.param({"CI": "true"}, True, id="ci"),
            pytest.param({}, False, id="no-testament"),
        ],
    )
    def test_the_suite_s_paths_run_here(self, repo: Path, env: dict[str, str], testament: bool) -> None:
        """scheduled-jobs runs here through testcontainers.

        :param repo: the miniature repo root
        :ptype repo: Path
        :param env: extra environment for the run
        :ptype env: dict[str, str]
        :param testament: whether a ``testament`` is on PATH
        :ptype testament: bool
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["scheduled-jobs"], testament=testament, env=env)
        assert run.returncode == 0, run.stderr
        assert run.call("testament") is None
        assert run.call("uv") == ["run", "pytest", _SUITE_DIR, "-m", "integration"]

    def test_a_suite_file_inside_a_local_directory_is_not_named_on_its_own(self, repo: Path) -> None:
        """off the pool, core's directory runs whole and its conftest decides about the suite's file there; named
        on its own, pytest would collect it whatever the conftest says.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["core"], testament=False)
        assert run.returncode == 0, run.stderr
        assert run.call("uv") == ["run", "pytest", _CORE_DIR, "-m", "integration"]

    def test_off_without_a_reason_is_refused(self, repo: Path) -> None:
        """TESTAMENT=off needs a stated reason, as test.sh's does.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["scheduled-jobs"], env={"TESTAMENT": "off"})
        assert run.returncode == 2
        assert run.calls == []
        assert "TESTAMENT=off needs a reason" in run.stderr


class TestEnvironmentGates:
    """test.sh keeps a run local under an environment gate; the integration suite's pool part must not be kept."""

    _GATES = {"RECORD": "1", "TAVILY_API_KEY": "a-key", "THREETEARS_LIVE_CLAUDE_CLI": "1"}

    def test_the_pool_part_runs_without_the_gates_and_the_local_part_keeps_them(self, repo: Path) -> None:
        """with every kind of gate set, the suite still goes to the pool, which sees none of them; the local
        directories run here with all of them.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["-rs"], env=self._GATES)
        assert run.returncode == 0, run.stderr
        assert run.call("testament") == [*_POOL, _SUITE_FILE, _SUITE_DIR, "-rs"]
        assert run.gates_seen("testament") == []
        assert run.gates_seen("uv") == sorted(self._GATES)

    @pytest.mark.parametrize(
        ("name", "is_a_gate"),
        [
            ("RECORD", True),
            ("TAVILY_API_KEY", True),
            ("THREETEARS_LIVE_CLAUDE_CLI", True),
            ("THREETEARS_LIVE_price2", True),
            ("RECORDING", False),
            ("TAVILY_API_KEY_OLD", False),
            ("THREETEARS_LIVEX", False),
        ],
    )
    def test_the_two_scripts_agree_on_what_a_gate_is(self, repo: Path, name: str, is_a_gate: bool) -> None:
        """a variable test.sh takes for a gate (the run stays local) is one test-integration.sh clears for its pool
        call, and one it does not take for a gate reaches the pool untouched.

        :param repo: the miniature repo root
        :ptype repo: Path
        :param name: the variable
        :ptype name: str
        :param is_a_gate: whether test.sh should take it for one
        :ptype is_a_gate: bool
        :return: nothing
        :rtype: None
        """
        alone = _run(repo, [f"{_SUITE_DIR}/test_jobs.py"], env={name: "1"}, script="test.sh")
        kept_local = alone.call("testament") is None
        through = _run(repo, ["scheduled-jobs"], env={name: "1"})
        assert through.returncode == 0, through.stderr
        cleared = name not in through.gates_seen("testament")
        assert kept_local == cleared == is_a_gate

    def test_a_package_on_the_pool_is_never_sent_local_by_a_gate(self, repo: Path) -> None:
        """scheduled-jobs moved to the pool; a gate does not bring it back to the laptop.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["scheduled-jobs"], env={"TAVILY_API_KEY": "a-key"})
        assert run.returncode == 0, run.stderr
        assert run.call("testament") == [*_POOL, _SUITE_DIR]
        assert run.call("uv") is None


class TestTestShRulesHold:
    """the pool part goes through test.sh, so its rules hold."""

    def test_a_local_only_flag_on_a_pool_directory_is_refused(self, repo: Path) -> None:
        """``-x`` needs one local process, and a directory is wider than named files: test.sh refuses.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["scheduled-jobs", "-x"])
        assert run.returncode == 2
        assert run.calls == []

    def test_no_verdict_ends_with_the_runbook(self, repo: Path) -> None:
        """the pool's exit 2 passes through with test.sh's closing line.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["scheduled-jobs"], testament_exit=2)
        assert run.returncode == 2
        assert 'docs/testing.md, "When the pool is down"' in run.stderr
