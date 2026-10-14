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
import subprocess
from pathlib import Path

import pytest

_BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(_BASH is None or shutil.which("git") is None, reason="needs bash and git")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SSH_STUB = """#!{bash}
if [ "$1" = -G ]; then printf 'hostname lane.internal\\nuser tester\\nport 2222\\n'; exit 0; fi
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


def _run(tmp_path: Path, script: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True, exist_ok=True)
    (home / ".ssh" / "threetears-testlane.pub").write_text("ssh-ed25519 AAAA test\n")
    full_env = {
        "PATH": f"{_stubs(tmp_path)}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(home),
        **env,
    }
    assert _BASH is not None
    return subprocess.run([_BASH, str(script), *args], env=full_env, capture_output=True, text=True, check=False)  # noqa: S603


def _stage(tmp_path: Path, dirname: str) -> Path:
    """a copy of the dispatcher inside a worktree-shaped directory."""
    root = tmp_path / dirname
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(_REPO_ROOT / "scripts" / "test-remote.sh", root / "scripts" / "test-remote.sh")
    return root


def _dry(tmp_path: Path, root: Path, *args: str) -> list[list[str]]:
    result = _run(tmp_path, root / "scripts" / "test-remote.sh", "gpu-x", *args, THREETEARS_TEST_REMOTE_DRY_RUN="1")
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


def test_every_remote_command_the_dispatcher_sends_passes_the_shim(tmp_path: Path) -> None:
    commands = _dry(tmp_path, _stage(tmp_path, "wt"), "core", "-k", "not slow and remote", "-x")
    name = _remote_name(commands)
    remote = [command[-1] for command in commands if command[0] == "ssh"]
    remote.append(f"rsync --server -tprze.iLsfxCIvu --delete . testruns-3tears/{name}/")
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
