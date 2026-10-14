"""The remote test lane's dispatcher, and test.sh's choice between lanes.

Nothing here reaches a host. ``THREETEARS_TEST_REMOTE_DRY_RUN=1`` makes
``scripts/test-remote.sh`` print each ssh/rsync instead of running it, and a
stub ``ssh`` on ``PATH`` answers ``ssh -G`` (and fails every connection, for
the fallback case); a stub ``uv`` stands in for the local pytest run. The
commands the dispatcher prints are then fed to the real shim, so the two halves
cannot drift apart on what the lane sends.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

_BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(_BASH is None or shutil.which("git") is None, reason="needs bash and git")

_REPO_ROOT = Path(__file__).resolve().parents[2]
# ``ssh -G`` runs after the lane lock is taken, so with STUB_LOCK_LOG set the stub
# records what the lock directory holds at that moment, and with STUB_BLOCK set it
# then waits there (holding the dispatcher inside its locked section) until killed.
_SSH_STUB = """#!{bash}
if [ "$1" = -G ]; then
    if [ -n "${{STUB_LOCK_LOG:-}}" ]; then
        cat "$XDG_STATE_HOME"/threetears-testlane/*.lock/owner >>"$STUB_LOCK_LOG" 2>/dev/null || echo none >>"$STUB_LOCK_LOG"
    fi
    if [ -n "${{STUB_BLOCK:-}}" ]; then sleep 60; fi
    printf 'hostname lane.internal\\nuser tester\\nport 2222\\n'
    exit 0
fi
exit 255
"""


def _stubs(tmp_path: Path) -> Path:
    stubs = tmp_path / "stubs"
    stubs.mkdir(exist_ok=True)
    (stubs / "ssh").write_text(_SSH_STUB.format(bash=_BASH))
    (stubs / "uv").write_text(f'#!{_BASH}\nprintf "%s\\n" "STUB:uv" "$@"\n')
    for stub in stubs.iterdir():
        stub.chmod(0o755)
    return stubs


def _env(tmp_path: Path, **env: str) -> dict[str, str]:
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True, exist_ok=True)
    (home / ".ssh" / "threetears-testlane.pub").write_text("ssh-ed25519 AAAA test\n")
    return {
        "PATH": f"{_stubs(tmp_path)}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(home),
        **env,
    }


def _run(tmp_path: Path, script: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    assert _BASH is not None
    return subprocess.run(  # noqa: S603
        [_BASH, str(script), *args], env=_env(tmp_path, **env), capture_output=True, text=True, check=False
    )


def _stage(tmp_path: Path, dirname: str) -> Path:
    """a copy of the dispatcher and its runner inside a worktree-shaped directory."""
    root = tmp_path / dirname
    (root / "scripts").mkdir(parents=True)
    for script in ("test-remote.sh", "remote-test.sh"):
        shutil.copy2(_REPO_ROOT / "scripts" / script, root / "scripts" / script)
    return root


def _dry(tmp_path: Path, root: Path, *args: str, **env: str) -> list[list[str]]:
    result = _run(
        tmp_path, root / "scripts" / "test-remote.sh", "gpu-x", *args, THREETEARS_TEST_REMOTE_DRY_RUN="1", **env
    )
    assert result.returncode == 0, result.stderr
    return [shlex.split(line[2:]) for line in result.stdout.splitlines() if line.startswith("+ ")]


def _remote_name(commands: list[list[str]]) -> str:
    mkdir = commands[0][-1]
    assert mkdir.startswith("mkdir -p 'testruns-3tears/")
    return mkdir.removeprefix("mkdir -p 'testruns-3tears/").removesuffix("'")


def test_the_name_is_a_sanitised_basename_plus_a_path_hash(tmp_path: Path) -> None:
    one = _remote_name(_dry(tmp_path, _stage(tmp_path / "a", "my wt!")))
    same = _remote_name(_dry(tmp_path, tmp_path / "a" / "my wt!"))
    other = _remote_name(_dry(tmp_path, _stage(tmp_path / "b", "my wt!")))
    assert one.startswith("my_wt_-")
    assert one == same
    assert other != one


_STARTED = "2026-10-14 09:00:00"


def _hold(tmp_path: Path, name: str, pid: int) -> Path:
    """a lane lock on ``name`` as another dispatch (``pid``) would have left it."""
    lock = tmp_path / "state" / "threetears-testlane" / f"{name}.lock"
    lock.mkdir(parents=True)
    (lock / "owner").write_text(f"{pid}\n{_STARTED}\n")
    return lock


def _dead_pid() -> int:
    child = subprocess.Popen(["true"])  # noqa: S607
    child.wait()
    return child.pid


def test_a_dispatch_holds_its_trees_lock_and_releases_it(tmp_path: Path) -> None:
    root = _stage(tmp_path, "wt")
    seen = tmp_path / "seen"
    state = tmp_path / "state"
    _dry(tmp_path, root, XDG_STATE_HOME=str(state), STUB_LOCK_LOG=str(seen))
    pid, started = seen.read_text().splitlines()
    assert pid.isdigit()
    assert len(started) == len(_STARTED)
    assert list((state / "threetears-testlane").iterdir()) == []
    # Without XDG_STATE_HOME the lock lives under ~/.local/state; the probe takes it too.
    probe = _run(tmp_path, root / "scripts" / "test-remote.sh", "--probe", "gpu-x", THREETEARS_TEST_REMOTE_DRY_RUN="1")
    assert probe.returncode == 0, probe.stderr
    assert list((tmp_path / "home" / ".local" / "state" / "threetears-testlane").iterdir()) == []


def test_a_second_dispatch_of_a_held_tree_exits_75_without_touching_the_lock(tmp_path: Path) -> None:
    root = _stage(tmp_path, "wt")
    name = _remote_name(_dry(tmp_path, root))
    lock = _hold(tmp_path, name, os.getpid())
    env = {
        "THREETEARS_TEST_REMOTE_DRY_RUN": "1",
        "THREETEARS_LANE_WAIT_S": "0",
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }
    for mode in ((), ("--probe",)):
        result = _run(tmp_path, root / "scripts" / "test-remote.sh", *mode, "gpu-x", **env)
        assert result.returncode == 75, result.stderr
        assert f"remote lane: {name} is already being tested from {os.getpid()} (started {_STARTED})" in result.stderr
        assert "+ " not in result.stdout
        assert (lock / "owner").read_text() == f"{os.getpid()}\n{_STARTED}\n"


def test_test_sh_stops_rather_than_run_locally_when_the_tree_stays_held(tmp_path: Path) -> None:
    name = _remote_name(_dry(tmp_path, _REPO_ROOT))
    _hold(tmp_path, name, os.getpid())
    result = _run(
        tmp_path,
        _REPO_ROOT / "scripts" / "test.sh",
        "core",
        THREETEARS_TEST_REMOTE="gpu-x",
        THREETEARS_TEST_REMOTE_DRY_RUN="1",
        THREETEARS_LANE_WAIT_S="0",
        XDG_STATE_HOME=str(tmp_path / "state"),
    )
    assert result.returncode == 75
    assert "remote lane gpu-x is still busy with this tree: not running" in result.stderr
    assert "STUB:uv" not in result.stdout


def test_a_stale_lock_is_removed_and_the_dispatch_proceeds(tmp_path: Path) -> None:
    root = _stage(tmp_path, "wt")
    name = _remote_name(_dry(tmp_path, root))
    dead = _dead_pid()
    lock = _hold(tmp_path, name, dead)
    seen = tmp_path / "seen"
    result = _run(
        tmp_path,
        root / "scripts" / "test-remote.sh",
        "gpu-x",
        THREETEARS_TEST_REMOTE_DRY_RUN="1",
        THREETEARS_LANE_WAIT_S="0",
        XDG_STATE_HOME=str(tmp_path / "state"),
        STUB_LOCK_LOG=str(seen),
    )
    assert result.returncode == 0, result.stderr
    assert f"removing the stale lock on {name} (pid {dead} is gone)" in result.stderr
    assert seen.read_text().splitlines()[0] != str(dead)
    assert not lock.exists()


def test_two_trees_never_wait_on_each_other(tmp_path: Path) -> None:
    held = _remote_name(_dry(tmp_path, _stage(tmp_path, "one")))
    _hold(tmp_path, held, os.getpid())
    commands = _dry(
        tmp_path, _stage(tmp_path, "two"), THREETEARS_LANE_WAIT_S="0", XDG_STATE_HOME=str(tmp_path / "state")
    )
    assert _remote_name(commands) != held


def test_an_interrupted_dispatch_releases_its_lock(tmp_path: Path) -> None:
    root = _stage(tmp_path, "wt")
    seen = tmp_path / "seen"
    state = tmp_path / "state"
    env = _env(
        tmp_path,
        THREETEARS_TEST_REMOTE_DRY_RUN="1",
        XDG_STATE_HOME=str(state),
        STUB_LOCK_LOG=str(seen),
        STUB_BLOCK="1",
    )
    assert _BASH is not None
    dispatch = subprocess.Popen(  # noqa: S603
        [_BASH, str(root / "scripts" / "test-remote.sh"), "gpu-x"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not seen.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert seen.read_text().splitlines()[0] == str(dispatch.pid)
        assert len(list((state / "threetears-testlane").iterdir())) == 1
        os.killpg(dispatch.pid, signal.SIGINT)
        assert dispatch.wait(timeout=10) == 130
    finally:
        if dispatch.poll() is None:
            os.killpg(dispatch.pid, signal.SIGKILL)
    assert list((state / "threetears-testlane").iterdir()) == []


@pytest.mark.parametrize("wait", ["soon", "-1", "1.5"])
def test_a_wait_that_is_not_whole_seconds_is_refused(tmp_path: Path, wait: str) -> None:
    result = _run(
        tmp_path,
        _REPO_ROOT / "scripts" / "test-remote.sh",
        "gpu-x",
        THREETEARS_TEST_REMOTE_DRY_RUN="1",
        THREETEARS_LANE_WAIT_S=wait,
    )
    assert result.returncode == 64
    assert "THREETEARS_LANE_WAIT_S must be a whole number of seconds" in result.stderr


def test_only_the_lane_key_is_offered_and_config_is_bypassed(tmp_path: Path) -> None:
    commands = _dry(tmp_path, _stage(tmp_path, "wt"))
    ssh = commands[0]
    assert ssh[:3] == ["ssh", "-F", "none"]
    assert ssh[ssh.index("-i") + 1].endswith(".ssh/threetears-testlane.pub")
    for option in ("IdentitiesOnly=yes", "BatchMode=yes", "ConnectTimeout=5", "ControlPath=none"):
        assert option in ssh
    assert ssh[ssh.index("-l") + 1] == "tester"
    assert ssh[ssh.index("-p") + 1] == "2222"
    assert "lane.internal" in ssh


def test_the_rsync_carries_no_links_and_skips_the_local_state(tmp_path: Path) -> None:
    rsync = _dry(tmp_path, _stage(tmp_path, "wt"))[1]
    assert rsync[:3] == ["rsync", "-rtpz", "--delete"]
    excluded = {rsync[i + 1] for i, word in enumerate(rsync) if word == "--exclude"}
    assert {".git", ".venv", ".worktrees", "**/__pycache__", ".pytest_cache", "*.egg-info"} <= excluded


def _rsync_ends(rsync: list[str]) -> tuple[str, str]:
    """(local source, remote destination) of a dry-run rsync."""
    return rsync[-2], rsync[-1]


def test_the_tree_rsync_leaves_the_runner_to_the_dispatcher(tmp_path: Path) -> None:
    root = _stage(tmp_path, "wt")
    commands = _dry(tmp_path, root)
    name = _remote_name(commands)
    tree, runner = commands[1], commands[2]
    assert "/scripts/remote-test.sh" in {tree[i + 1] for i, word in enumerate(tree) if word == "--exclude"}
    # The second rsync sends exactly the dispatcher's runner, into the tree root
    # the shim already admits, and deletes nothing.
    assert runner[0] == "rsync"
    assert not any(word.startswith("--delete") for word in runner)
    filters = [(word, runner[i + 1]) for i, word in enumerate(runner) if word in ("--include", "--exclude")]
    assert filters == [("--include", "/scripts/"), ("--include", "/scripts/remote-test.sh"), ("--exclude", "*")]
    assert _rsync_ends(runner) == (f"{root}/", f"lane.internal:testruns-3tears/{name}/")
    assert commands.index(runner) < next(i for i, c in enumerate(commands) if "docker exec" in c[-1])


def test_tree_syncs_another_checkout_with_this_dispatchers_runner(tmp_path: Path) -> None:
    dispatcher = _stage(tmp_path, "lane")
    old_branch = tmp_path / "old-branch"
    (old_branch / "scripts").mkdir(parents=True)
    result = _run(
        tmp_path,
        dispatcher / "scripts" / "test-remote.sh",
        "--tree",
        str(old_branch),
        "gpu-x",
        "core",
        THREETEARS_TEST_REMOTE_DRY_RUN="1",
    )
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line[2:]) for line in result.stdout.splitlines() if line.startswith("+ ")]
    name = _remote_name(commands)
    assert name.startswith("old-branch-")
    assert _rsync_ends(commands[1]) == (f"{old_branch}/", f"lane.internal:testruns-3tears/{name}/")
    assert _rsync_ends(commands[2]) == (f"{dispatcher}/", f"lane.internal:testruns-3tears/{name}/")


def test_a_dispatcher_without_its_runner_refuses_before_connecting(tmp_path: Path) -> None:
    root = _stage(tmp_path, "wt")
    (root / "scripts" / "remote-test.sh").unlink()
    result = _run(tmp_path, root / "scripts" / "test-remote.sh", "gpu-x", THREETEARS_TEST_REMOTE_DRY_RUN="1")
    assert result.returncode == 66
    assert "no runner beside the dispatcher" in result.stderr
    assert "+ " not in result.stdout


def test_every_remote_command_the_dispatcher_sends_passes_the_shim(tmp_path: Path) -> None:
    commands = _dry(tmp_path, _stage(tmp_path, "wt"), "core", "-k", "not slow and remote", "-x")
    name = _remote_name(commands)
    remote = [command[-1] for command in commands if command[0] == "ssh"]
    remote.append(f"rsync --server -tprze.iLsfxCIvu --delete . testruns-3tears/{name}/")
    # the runner rsync: same destination, no --delete
    remote.append(f"rsync --server -tprze.iLsfxCIvu . testruns-3tears/{name}/")
    remote.append(f"rsync --server -tprze.iLsfxCIvu . testruns-3tears/{name}/.git/")
    assert remote[1] == (
        f"docker exec -i threetears-test bash /testruns/{name}/scripts/remote-test.sh 'core' '-k' 'not slow and remote' '-x'"
    )
    stubs = tmp_path / "shim-stubs"
    stubs.mkdir()
    for binary in ("docker", "rsync", "mkdir"):
        (stubs / binary).write_text(f"#!{_BASH}\nexit 0\n")
        (stubs / binary).chmod(0o755)
    for command in remote:
        assert _BASH is not None
        result = subprocess.run(  # noqa: S603
            [_BASH, str(_REPO_ROOT / "scripts" / "threetears-testlane-shim")],
            env={"PATH": str(stubs), "HOME": str(tmp_path), "SSH_ORIGINAL_COMMAND": command},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (command, result.stderr)


@pytest.mark.parametrize("arg", ["a;b", "$(id)", "x'y", "--", ""])
def test_an_argument_the_shim_would_refuse_is_refused_here_first(tmp_path: Path, arg: str) -> None:
    result = _run(tmp_path, _REPO_ROOT / "scripts" / "test-remote.sh", "--probe", "gpu-x", "core", arg)
    assert result.returncode == 64
    assert "cannot cross the lane" in result.stderr


def test_test_sh_dispatches_when_the_probe_answers(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        _REPO_ROOT / "scripts" / "test.sh",
        "core",
        "-k",
        "a or b",
        THREETEARS_TEST_REMOTE="gpu-x",
        THREETEARS_TEST_REMOTE_DRY_RUN="1",
    )
    assert result.returncode == 0, result.stderr
    remote = [shlex.split(line[2:])[-1] for line in result.stdout.splitlines() if line.startswith("+ ssh ")]
    assert remote[-1].endswith("/scripts/remote-test.sh 'core' '-k' 'a or b'")
    assert result.stdout.splitlines()[-1] == "tests ran on gpu-x (remote)"
    assert "STUB:uv" not in result.stdout


@pytest.mark.parametrize(("remote", "where"), [("gpu-x", "on gpu-x (remote)"), (None, "locally")])
def test_test_sh_tells_check_all_where_the_tests_ran(tmp_path: Path, remote: str | None, where: str) -> None:
    where_file = tmp_path / "where"
    env = {"THREETEARS_TEST_WHERE_FILE": str(where_file), "THREETEARS_TEST_REMOTE_DRY_RUN": "1"}
    if remote is not None:
        env["THREETEARS_TEST_REMOTE"] = remote
    result = _run(tmp_path, _REPO_ROOT / "scripts" / "test.sh", "core", **env)
    assert result.returncode == 0, result.stderr
    assert where_file.read_text() == f"{where}\n"


def test_test_sh_falls_back_loudly_and_keeps_its_arguments(tmp_path: Path) -> None:
    result = _run(tmp_path, _REPO_ROOT / "scripts" / "test.sh", "core", "-k", "a or b", THREETEARS_TEST_REMOTE="gpu-x")
    assert result.returncode == 0, result.stderr
    assert "remote lane gpu-x unreachable: running locally" in result.stderr
    assert result.stdout.splitlines() == ["STUB:uv", "run", "pytest", "packages/core/tests/", "-k", "a or b"]


def test_test_sh_unset_never_probes(tmp_path: Path) -> None:
    result = _run(tmp_path, _REPO_ROOT / "scripts" / "test.sh", "-x")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.splitlines() == [
        "STUB:uv",
        "run",
        "pytest",
        "packages/",
        "tests/",
        "-m",
        "not integration",
        "-x",
    ]
