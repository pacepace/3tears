"""behavioural tests for ``scripts/test-integration.sh``: the suite on the pool, ``--here`` on this machine.

The integration suite runs on the testament pool: the manifest's ``[suites.integration] default_selection``,
through ``scripts/test.sh --suite integration``, and nothing on this machine. ``--here`` runs the tests marked
``docker`` or ``secrets`` (the script's ``HERE_FILES``) locally. Where the pool route is not taken (no testament,
CI, ``TESTAMENT=off`` with a reason) the suite's paths run here through testcontainers, less the ``process_only``
modules. These tests pin each route, what a package or a named path selects, the dry run, the environment gates
the pool call clears, and test.sh's rules the pool call inherits.

Self-contained: both scripts are staged in a miniature repo under ``tmp_path`` beside a manifest of its
own, and ``testament`` and ``uv`` are stand-in executables on PATH that record their argv, so nothing
here runs a test or reaches the pool.

    ./scripts/test.sh tests/test_test_integration_script.py
"""

from __future__ import annotations

import shlex
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


def _here_files() -> list[str]:
    """the script's ``HERE_FILES``, as written.

    :return: the paths
    :rtype: list[str]
    """
    text = (_REPO_ROOT / "scripts" / "test-integration.sh").read_text()
    block = text.split("\nHERE_FILES=(\n", 1)[1].split("\n)\n", 1)[0]
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
    shutil.copy(_REPO_ROOT / "scripts" / "integration_marks.py", root / "scripts" / "integration_marks.py")
    (root / ".testament.toml").write_text(_MANIFEST)
    for directory in (_CORE_DIR, _SUITE_DIR, _NATS_DIR, "packages/models/tests"):
        (root / directory).mkdir(parents=True)
    for path in (_SUITE_FILE, f"{_CORE_DIR}/test_other.py", f"{_SUITE_DIR}/test_jobs.py", f"{_NATS_DIR}/test_bus.py"):
        (root / path).write_text(_MARKED)
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
    old_python: bool = False,
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
    :param old_python: whether the ``python3`` on PATH is one older than 3.11 (a stand-in that fails the check)
    :ptype old_python: bool
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
    if old_python:
        (bin_dir / "python3").write_text("#!/bin/sh\nexit 1\n")
        (bin_dir / "python3").chmod(0o755)
    else:
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


#: the off-pool command's marker expression: the pool suite's
_LOCAL_MARKS = ["-m", "integration and not docker and not secrets"]


class TestThePoolRunsTheSuite:
    """the suite's paths go to the pool, and nothing runs on this machine."""

    def test_a_whole_run_is_the_whole_suite_on_the_pool(self, repo: Path) -> None:
        """no package: every suite path, with the options; ``uv`` never runs here.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["-rs"])
        assert run.returncode == 0, run.stderr
        assert run.call("testament") == [*_POOL, _SUITE_FILE, _SUITE_DIR, "-rs"]
        assert run.call("uv") is None

    def test_a_package_runs_its_part_of_the_suite(self, repo: Path) -> None:
        """scheduled-jobs: its suite directory alone.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["scheduled-jobs", "-v"])
        assert run.call("testament") == [*_POOL, _SUITE_DIR, "-v"]
        assert run.call("uv") is None

    def test_a_package_with_nothing_in_the_suite_runs_nothing_and_says_where_to_look(self, repo: Path) -> None:
        """models has no suite path: exit 5, nothing run, and the message names ``--here``.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["models"])
        assert run.returncode == 5
        assert run.calls == []
        assert "--here" in run.stderr

    @pytest.mark.parametrize(("pool", "status"), [(0, 0), (1, 1), (2, 2)])
    def test_the_pool_s_status_is_the_run_s(self, repo: Path, pool: int, status: int) -> None:
        """a red is a red, and the pool's exit 2 (no verdict) passes through with test.sh's runbook line.

        :param repo: the miniature repo root
        :ptype repo: Path
        :param pool: the pool's exit status
        :ptype pool: int
        :param status: the script's expected exit status
        :ptype status: int
        :return: nothing
        :rtype: None
        """
        run = _run(repo, [], testament_exit=pool)
        assert run.returncode == status
        assert ("testament gave no verdict" in run.stderr) == (pool == 2)


class TestNamedPaths:
    """named paths select within the suite; one outside it is refused, naming ``--here``."""

    def test_a_file_in_a_suite_directory_runs_alone(self, repo: Path) -> None:
        """one scheduled-jobs file: that file only.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        path = f"{_SUITE_DIR}/test_jobs.py"
        run = _run(repo, ["scheduled-jobs", path, "-q"])
        assert run.call("testament") == [*_POOL, path, "-q"]

    def test_a_directory_holding_a_suite_file_runs_that_file(self, repo: Path) -> None:
        """core's integration directory holds one suite file: that file runs.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, [_CORE_DIR])
        assert run.call("testament") == [*_POOL, _SUITE_FILE]

    def test_a_suite_path_named_itself_runs_once(self, repo: Path) -> None:
        """naming the suite's file directly sends it once.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, [_SUITE_FILE])
        assert run.call("testament") == [*_POOL, _SUITE_FILE]

    def test_a_path_outside_the_suite_is_refused_naming_here(self, repo: Path) -> None:
        """a nats file is in no suite path: refused, nothing run.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, [f"{_NATS_DIR}/test_bus.py"])
        assert run.returncode == 2
        assert run.calls == []
        assert "--here" in run.stderr

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


class TestHere:
    """``--here`` runs exactly the docker and secrets tests on this machine."""

    def test_it_runs_the_here_files_with_their_marks_and_keeps_the_environment(self, repo: Path) -> None:
        """HERE_FILES, ``-m "docker or secrets"``, the options; a credential reaches the run.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["--here", "-rs"], env={"TAVILY_API_KEY": "a-key"})
        assert run.returncode == 0, run.stderr
        assert run.call("uv") == ["run", "pytest", *_here_files(), "-m", "docker or secrets", "-rs"]
        assert run.call("testament") is None
        assert run.gates_seen("uv") == ["TAVILY_API_KEY"]

    def test_its_dry_run_prints_the_command_and_runs_nothing(self, repo: Path) -> None:
        """``-rs --here --dry-run``, in any order.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["-rs", "--dry-run", "--here"])
        assert run.returncode == 0, run.stderr
        assert run.calls == []
        (line,) = run.stdout.splitlines()
        assert shlex.split(line.removeprefix("here: ")) == [
            "uv",
            "run",
            "pytest",
            *_here_files(),
            "-m",
            "docker or secrets",
            "-rs",
        ]


class TestTheDryRun:
    """``--dry-run``, anywhere among the arguments, prints the command a run would run, quoted, and runs nothing."""

    def test_the_pool_command_and_nothing_run(self, repo: Path) -> None:
        """the pool's command, with the options.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["-rs", "--dry-run"])
        assert run.returncode == 0, run.stderr
        assert run.calls == []
        assert run.stdout.splitlines() == [
            f"on the pool: ./scripts/test.sh --suite integration {_SUITE_FILE} {_SUITE_DIR} -rs"
        ]

    def test_the_flag_counts_after_a_package(self, repo: Path) -> None:
        """``core --dry-run`` is a dry run too.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["core", "--dry-run"])
        assert run.calls == []
        assert run.stdout.splitlines() == [f"on the pool: ./scripts/test.sh --suite integration {_SUITE_FILE}"]

    def test_it_prints_the_real_command_quoted_with_the_gates_it_clears(self, repo: Path) -> None:
        """the pool line carries the ``env -u`` of a set gate, and an argument with a space comes back as one word.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["--dry-run", "scheduled-jobs", "-k", "a or b"], env={"TAVILY_API_KEY": "a-key"})
        assert run.returncode == 0, run.stderr
        (line,) = run.stdout.splitlines()
        assert shlex.split(line.removeprefix("on the pool: ")) == [
            "env",
            "-u",
            "TAVILY_API_KEY",
            "./scripts/test.sh",
            "--suite",
            "integration",
            _SUITE_DIR,
            "-k",
            "a or b",
        ]


class TestWhenThePoolRouteIsNotTaken:
    """no testament, CI, or TESTAMENT=off with a reason: the suite's paths run here, through testcontainers."""

    @pytest.mark.parametrize(
        ("env", "testament"),
        [
            pytest.param({"TESTAMENT": "off", "TESTAMENT_OFF_REASON": "the pool is down"}, True, id="off-with-reason"),
            pytest.param({"CI": "true"}, True, id="ci"),
            pytest.param({}, False, id="no-testament"),
        ],
    )
    def test_the_suite_s_paths_run_here_with_the_suite_s_marks(
        self, repo: Path, env: dict[str, str], testament: bool
    ) -> None:
        """scheduled-jobs runs here with the pool suite's marker expression.

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
        assert run.call("uv") == ["run", "pytest", _SUITE_DIR, *_LOCAL_MARKS]

    def test_a_process_only_module_is_left_out_here_and_kept_on_the_pool(self, repo: Path) -> None:
        """a suite module marked ``process_only`` checks child processes, which testcontainers can't stand in for.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        (repo / _SUITE_FILE).write_text(
            "import pytest\n\npytestmark = [pytest.mark.integration, pytest.mark.process_only]\n\n\n"
            "def test_y() -> None:\n    pass\n"
        )
        off = _run(repo, ["-rs"], testament=False)
        assert off.call("uv") == ["run", "pytest", _SUITE_DIR, *_LOCAL_MARKS, "-rs"]
        on = _run(repo, ["-rs"])
        assert on.call("testament") == [*_POOL, _SUITE_FILE, _SUITE_DIR, "-rs"]

    def test_a_selection_that_is_all_process_only_runs_nothing_here(self, repo: Path) -> None:
        """only the process-mode module selected, off the pool: nothing to run, exit 5.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        (repo / _SUITE_FILE).write_text(
            "import pytest\n\npytestmark = [pytest.mark.integration, pytest.mark.process_only]\n"
        )
        run = _run(repo, ["core"], testament=False)
        assert run.returncode == 5
        assert run.calls == []

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

    def test_a_python_older_than_3_11_is_refused(self, repo: Path) -> None:
        """the manifest is read with tomllib: an older ``python3`` stops the run, naming the floor.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["scheduled-jobs"], old_python=True)
        assert run.returncode == 2
        assert run.calls == []
        assert "python3 3.11 or later" in run.stderr


class TestEnvironmentGates:
    """test.sh keeps a run local under an environment gate; the integration suite's pool call must not be kept."""

    _GATES = {"RECORD": "1", "TAVILY_API_KEY": "a-key", "THREETEARS_LIVE_CLAUDE_CLI": "1"}

    def test_the_pool_call_runs_without_the_gates(self, repo: Path) -> None:
        """with every kind of gate set, the suite still goes to the pool, which sees none of them.

        :param repo: the miniature repo root
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        run = _run(repo, ["-rs"], env=self._GATES)
        assert run.returncode == 0, run.stderr
        assert run.call("testament") == [*_POOL, _SUITE_FILE, _SUITE_DIR, "-rs"]
        assert run.gates_seen("testament") == []
        assert run.call("uv") is None

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
        # kept local means it ran here, not that it was refused
        assert (alone.call("uv") is not None) == kept_local, alone.stderr
        through = _run(repo, ["scheduled-jobs"], env={name: "1"})
        assert through.returncode == 0, through.stderr
        cleared = name not in through.gates_seen("testament")
        assert kept_local == cleared == is_a_gate


class TestTestShRulesHold:
    """the pool call goes through test.sh, so its rules hold."""

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
