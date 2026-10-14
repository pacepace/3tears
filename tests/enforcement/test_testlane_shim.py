"""The remote test lane's forced command admits its four requests and nothing else.

``scripts/threetears-testlane-shim`` is the whole of the lane key's reach on a
host whose docker access is effectively root. Each case runs the shim with
``SSH_ORIGINAL_COMMAND`` set, ``PATH`` pointing at stub ``docker``/``rsync``/
``mkdir`` executables that print their argv, and ``HOME`` in a temp dir: nothing
real is executed, and because the shim ``exec``s argv directly, the stubs see
exactly what docker or rsync would have.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(_BASH is None, reason="needs bash")

_SHIM = Path(__file__).resolve().parents[2] / "scripts" / "threetears-testlane-shim"
_RUN = "docker exec -i threetears-test bash /testruns/wt-1f/scripts/remote-test.sh"
_FLAGS = "-tprze.iLsfxCIvu"


def _shim(tmp_path: Path, command: str | None) -> subprocess.CompletedProcess[str]:
    stubs = tmp_path / "stubs"
    if not stubs.exists():
        stubs.mkdir()
        for name in ("docker", "rsync", "mkdir"):
            stub = stubs / name
            stub.write_text(f'#!{_BASH}\nprintf "%s\\n" "STUB:${{0##*/}}" "$@"\n')
            stub.chmod(0o755)
    env = {"PATH": str(stubs), "HOME": str(tmp_path)}
    if command is not None:
        env["SSH_ORIGINAL_COMMAND"] = command
    assert _BASH is not None
    return subprocess.run([_BASH, str(_SHIM)], env=env, capture_output=True, text=True, check=False)  # noqa: S603


_ALLOWED = [
    pytest.param(
        "docker inspect -f {{.State.Running}} threetears-test",
        ["docker", "inspect", "-f", "{{.State.Running}}", "threetears-test"],
        id="probe",
    ),
    pytest.param(
        "docker inspect -f '{{.State.Running}}' threetears-test",
        ["docker", "inspect", "-f", "{{.State.Running}}", "threetears-test"],
        id="probe-quoted",
    ),
    pytest.param("mkdir -p 'testruns-3tears/wt-1f'", ["mkdir", "-p", "testruns-3tears/wt-1f"], id="mkdir"),
    pytest.param(
        f"rsync --server {_FLAGS} --delete . testruns-3tears/wt-1f/",
        ["rsync", "--server", _FLAGS, "--delete", ".", "testruns-3tears/wt-1f/"],
        id="rsync-tree",
    ),
    pytest.param(
        f"rsync --server {_FLAGS} . testruns-3tears/wt-1f/.git/",
        ["rsync", "--server", _FLAGS, ".", "testruns-3tears/wt-1f/.git/"],
        id="rsync-git",
    ),
    pytest.param(
        f"{_RUN} 'core' '-k' 'not slow and remote' '--maxprocesses=4' '-x' 'packages/core/tests/test_a.py::test_b'",
        [
            "docker",
            "exec",
            "-i",
            "threetears-test",
            "bash",
            "/testruns/wt-1f/scripts/remote-test.sh",
            "core",
            "-k",
            "not slow and remote",
            "--maxprocesses=4",
            "-x",
            "packages/core/tests/test_a.py::test_b",
        ],
        id="run-with-args",
    ),
    pytest.param(
        "docker exec threetears-test bash /testruns/wt-1f/scripts/remote-test.sh",
        ["docker", "exec", "threetears-test", "bash", "/testruns/wt-1f/scripts/remote-test.sh"],
        id="run-bare",
    ),
]


@pytest.mark.parametrize(("command", "argv"), _ALLOWED)
def test_allowed_requests_reach_the_binary_with_exact_argv(tmp_path: Path, command: str, argv: list[str]) -> None:
    result = _shim(tmp_path, command)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [f"STUB:{argv[0]}", *argv[1:]]


_REFUSED = [
    pytest.param(None, id="interactive-shell"),
    pytest.param("id", id="arbitrary-command"),
    pytest.param(f"{_RUN} core; id", id="semicolon"),
    pytest.param(f"{_RUN} core && id", id="ampersand"),
    pytest.param(f"{_RUN} core\nid", id="newline"),
    pytest.param(f"{_RUN} core\tid", id="tab"),
    pytest.param(f"{_RUN} '$(id)'", id="substitution"),
    pytest.param(f"{_RUN} `id`", id="backtick"),
    pytest.param(f"{_RUN} 'unbalanced", id="unbalanced-quote"),
    pytest.param(f"{_RUN} -- -c id", id="double-dash"),
    pytest.param("mkdir -p 'testruns-3tears/../x'", id="mkdir-traversal"),
    pytest.param("mkdir -p 'testruns-3tears/k' '/home/u/.ssh/evil'", id="mkdir-extra-operand"),
    pytest.param("mkdir -p '.ssh'", id="mkdir-outside-root"),
    pytest.param("mkdir -p 'testruns-3tears/'", id="mkdir-bare-root"),
    pytest.param("docker exec -i homeassistant bash /testruns/wt-1f/scripts/remote-test.sh", id="other-container"),
    pytest.param(
        "docker exec -u 0 -i threetears-test bash /testruns/wt-1f/scripts/remote-test.sh", id="exec-flag-user"
    ),
    pytest.param(
        "docker exec -i --privileged threetears-test bash /testruns/wt-1f/scripts/remote-test.sh", id="exec-privileged"
    ),
    pytest.param("docker exec -i threetears-test bash -c id", id="exec-bash-c"),
    pytest.param("docker exec -i threetears-test sh /testruns/wt-1f/scripts/remote-test.sh", id="exec-not-bash"),
    pytest.param("docker exec -i threetears-test bash /testruns/../etc/scripts/remote-test.sh", id="exec-traversal"),
    pytest.param("docker exec -i threetears-test bash /testruns/wt-1f/scripts/other.sh", id="exec-other-script"),
    pytest.param(f"rsync --server {_FLAGS} --delete . .ssh/", id="rsync-outside-root"),
    pytest.param(f"rsync --server {_FLAGS} --delete . testruns-3tears/", id="rsync-bare-root"),
    pytest.param(f"rsync --server {_FLAGS} . testruns-3tears/../.ssh/", id="rsync-traversal"),
    pytest.param("rsync --server -logDtprze.iLsfxCIvu --delete . testruns-3tears/wt-1f/", id="rsync-links"),
    pytest.param(f"rsync --server --sender {_FLAGS} . testruns-3tears/wt-1f/", id="rsync-sender"),
    pytest.param(f"rsync --server {_FLAGS} --partial-dir=x . testruns-3tears/wt-1f/", id="rsync-partial-dir"),
    pytest.param(f"rsync --server {_FLAGS} . testruns-3tears/wt-1f/ testruns-3tears/b/", id="rsync-two-dests"),
    pytest.param("docker inspect -f {{.State.Running}} threetears-test; id", id="probe-plus-command"),
    pytest.param("docker inspect threetears-test", id="inspect-other-shape"),
    pytest.param("docker run --rm -v /:/host alpine", id="docker-run"),
]


@pytest.mark.parametrize("command", _REFUSED)
def test_everything_else_is_refused_before_any_binary_runs(tmp_path: Path, command: str | None) -> None:
    result = _shim(tmp_path, command)
    assert result.returncode == 77, (result.stdout, result.stderr)
    assert "refused:" in result.stderr
    assert "STUB:" not in result.stdout


def test_a_symlinked_destination_is_refused(tmp_path: Path) -> None:
    (tmp_path / "testruns-3tears").mkdir()
    os.symlink(tmp_path / "elsewhere", tmp_path / "testruns-3tears" / "wt-1f")
    result = _shim(tmp_path, f"rsync --server {_FLAGS} --delete . testruns-3tears/wt-1f/")
    assert result.returncode == 77
    assert "symlink" in result.stderr
