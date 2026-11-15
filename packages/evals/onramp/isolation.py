"""Where the agent's commands run: a container with no network, as a user that is not root, seeing only its trial.

Every process an agent causes to run (its ``python`` calls, and the regrade's re-runs of what it wrote) runs in a
fresh :class:`Container` of the trial image, one container per command, removed when the command ends:

- no network (``--network none``), no capability, no new privileges, a read-only root filesystem;
- user 10001, not root, with an environment of five named variables and nothing of this process's;
- nothing mounted from this machine: the trial's directory is copied in as a tar stream before the command and copied
  back out after it (members that would land outside it, links included, are dropped), into a ``/work`` of
  :data:`WORK_MB` MB held in memory; the docs build is installed in the image, read-only;
- :data:`CPUS` CPU, :data:`MEMORY_MB` MB of memory and no swap, :data:`PIDS` processes, and a wall-clock limit per
  command, held inside the container (``timeout -s KILL``) and again outside it.

No bind mount is used, so the Docker daemon may be on another machine (this dev container talks to its host's): the
CLI streams every byte in and out. The image is built from ``container/Dockerfile`` at set-up (:func:`trial_image`),
with the network, never inside a trial.

:class:`HostRunner` runs a command on this machine instead. It is for the scripted agent only:
:func:`~.agent.attempt` refuses it for any other brain, and the command line offers it only beside ``--scripted``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import IO, Protocol

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
DOCKERFILE = HERE / "container" / "Dockerfile"
RERUN = HERE / "rerun.py"

#: The image's name; its tag is a digest of the wheel and the Dockerfile.
IMAGE = "onramp-trial"
#: The label every image and container made here carries, so a leftover is found by it.
LABEL = "onramp.trial=1"
#: The user every command runs as.
USER = "10001:10001"
#: Where the trial's directory is, inside the container.
WORK = "/work"
#: The interpreter the build is installed in, inside the container.
PYTHON = "/opt/build/bin/python"
#: Where the regrade's ``rerun.py`` is put, only for the regrade's own commands: no trial command can read it.
RERUN_AT = "/tmp/onramp/rerun.py"
#: Limits on one command.
CPUS = 1
MEMORY_MB = 1024
PIDS = 128
WORK_MB = 256
TMP_MB = 64
#: How long past its own limit a command may take to be killed and copied out before the harness gives up on it.
SLACK_S = 30.0
#: Bytes of a command's output kept, from its head and from its tail each: the agent is shown far less.
KEPT_BYTES = 256 * 1024
#: The environment every command runs with: nothing of this process's.
ENVIRONMENT: Mapping[str, str] = {
    "PATH": "/opt/build/bin:/usr/local/bin:/usr/bin:/bin",
    "HOME": WORK,
    "LANG": "C.UTF-8",
    "PYTHONUTF8": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}
#: What the ``docker`` command itself is given of this process's environment: how to reach the daemon, nothing else.
_DOCKER_VARIABLES = (
    "PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "DOCKER_CERT_PATH", "DOCKER_TLS_VERIFY",
)  # fmt: skip


class IsolationUnavailable(RuntimeError):
    """The container a command must run in could not be had: no ``docker``, no daemon, or no image."""


@dataclass(frozen=True)
class Ran:
    """What one command did.

    :ivar exit_code: Its exit code; ``None`` when it was killed at its time limit.
    :ivar stdout: Its output, its head and tail when it was long.
    :ivar stderr: Its errors, the same.
    :ivar cwd: The directory it ran in, as it saw it: what the agent is shown as ``.``.
    """

    exit_code: int | None
    stdout: str
    stderr: str
    cwd: str


class Runner(Protocol):
    """Where a command runs."""

    #: Whether a command it runs is cut off from the network, the keys and the rest of this machine.
    isolated: bool
    #: The build's interpreter, as a command names it.
    python: str
    #: ``rerun.py``, as a regrade's command names it.
    rerun: str

    async def run(self, argv: Sequence[str], *, cwd: Path, timeout_s: float, grading: bool = False) -> Ran:
        """Run ``argv`` in ``cwd`` for at most ``timeout_s``; ``grading`` puts :data:`RERUN` where :attr:`rerun` says."""
        ...


def docker_environment() -> dict[str, str]:
    """The environment the ``docker`` command runs with: how to reach the daemon, and no key."""
    return {name: os.environ[name] for name in _DOCKER_VARIABLES if name in os.environ}


def _docker(*args: str, stdin: bytes | None = None, timeout_s: float = 120.0) -> subprocess.CompletedProcess[bytes]:
    """``docker args``, at set-up.

    :raises IsolationUnavailable: There is no ``docker`` command, or it did not answer within ``timeout_s``.
    """
    try:
        return subprocess.run(
            ["docker", *args], input=stdin, capture_output=True, env=docker_environment(), timeout=timeout_s
        )
    except FileNotFoundError as missing:
        raise IsolationUnavailable("there is no docker command on PATH") from missing
    except subprocess.TimeoutExpired as slow:
        raise IsolationUnavailable(f"docker {args[0]} did not answer within {timeout_s:.0f}s") from slow


def _said(done: subprocess.CompletedProcess[bytes]) -> str:
    """A failed docker command's first line of errors."""
    text = done.stderr.decode("utf-8", "replace")
    return next((line.strip() for line in text.splitlines() if line.strip()), f"exit {done.returncode}")


def unavailable() -> str | None:
    """Why no trial container can run here, or ``None`` when the Docker daemon answers."""
    try:
        done = _docker("version", "--format", "{{.Server.Version}}", timeout_s=30.0)
    except IsolationUnavailable as why:
        return str(why)
    if done.returncode != 0:
        return f"the Docker daemon did not answer: {_said(done)}"
    return None


def trial_image(wheel: Path) -> str:
    """The trial image with ``wheel`` installed, built when it is not there yet; its tag names the wheel's digest.

    :raises IsolationUnavailable: Docker is unavailable, or the image did not build.
    """
    if (why := unavailable()) is not None:
        raise IsolationUnavailable(why)
    digest = hashlib.sha256(wheel.read_bytes() + DOCKERFILE.read_bytes()).hexdigest()
    tag = f"{IMAGE}:{digest[:16]}"
    if _docker("image", "inspect", tag).returncode == 0:
        return tag
    with tempfile.TemporaryDirectory(prefix="onramp-image-") as context:
        shutil.copy2(wheel, Path(context) / wheel.name)
        shutil.copy2(DOCKERFILE, Path(context) / "Dockerfile")
        done = _docker("build", "--quiet", "--label", LABEL, "--tag", tag, context, timeout_s=900.0)
    if done.returncode != 0:
        raise IsolationUnavailable(f"the trial image did not build: {_said(done)}")
    return tag


def package_dir(image: str) -> str:
    """Where ``threetears.evals`` is installed in ``image``, asked of the image with no network.

    :raises IsolationUnavailable: The image could not be asked.
    """
    done = _docker(
        "run", "--rm", "--network", "none", "--user", USER, image,
        PYTHON, "-I", "-c", "import pathlib, threetears.evals as e; print(pathlib.Path(e.__file__).parent)",
    )  # fmt: skip
    if done.returncode != 0:
        raise IsolationUnavailable(f"the trial image {image} does not import threetears.evals: {_said(done)}")
    return done.stdout.decode().strip()


def copy_out_of_image(image: str, path: str, dest: Path) -> Path:
    """A copy of the directory ``path`` in ``image`` at ``dest``, which must not exist: the docs the agent reads.

    :raises IsolationUnavailable: The image could not be read.
    """
    name = f"onramp-read-{uuid.uuid4().hex[:12]}"
    made = _docker("create", "--name", name, "--label", LABEL, "--network", "none", image)
    if made.returncode != 0:
        raise IsolationUnavailable(f"could not read the trial image {image}: {_said(made)}")
    try:
        done = _docker("cp", f"{name}:{path}/.", "-")
        if done.returncode != 0:
            raise IsolationUnavailable(f"could not read {path} from the trial image {image}: {_said(done)}")
    finally:
        _docker("rm", "-f", name)
    dest.mkdir(parents=True)
    _extract(BytesIO(done.stdout), dest)
    return dest


def _safe(member: tarfile.TarInfo, dest: str) -> tarfile.TarInfo | None:
    """``member`` as the ``data`` filter passes it, or ``None`` (dropped) when it would land outside ``dest``."""
    try:
        return tarfile.data_filter(member, dest)
    except tarfile.FilterError:
        return None


def _extract(stream: IO[bytes], dest: Path) -> None:
    """Extract a tar stream into ``dest``, dropping every member that would land outside it (links included)."""
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        archive.extractall(dest, filter=_safe)


def _tar_of(directory: Path) -> bytes:
    """``directory``'s contents as a tar stream whose every member belongs to the container's user."""
    uid, gid = (int(part) for part in USER.split(":"))

    def owned(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid, info.gid, info.uname, info.gname = uid, gid, "agent", "agent"
        return info

    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in sorted(directory.iterdir()):
            archive.add(path, arcname=path.name, filter=owned)
    return buffer.getvalue()


def _rerun_tar() -> bytes:
    """``rerun.py`` as a tar stream to unpack at ``/``, so it lands at :data:`RERUN_AT`."""
    uid, gid = (int(part) for part in USER.split(":"))
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        data = RERUN.read_bytes()
        info = tarfile.TarInfo(RERUN_AT.lstrip("/").removeprefix("tmp/"))
        info.size, info.mode, info.uid, info.gid = len(data), 0o644, uid, gid
        archive.addfile(info, BytesIO(data))
    return buffer.getvalue()


async def _drain(stream: asyncio.StreamReader) -> bytes:
    """All of ``stream`` read, keeping its first and last :data:`KEPT_BYTES` and saying how much was cut between."""
    head = bytearray()
    tail = bytearray()
    cut = 0
    while chunk := await stream.read(65536):
        room = KEPT_BYTES - len(head)
        head += chunk[:room]
        tail += chunk[room:] if room > 0 else chunk
        if len(tail) > KEPT_BYTES:
            cut += len(tail) - KEPT_BYTES
            del tail[: len(tail) - KEPT_BYTES]
    middle = f"\n[... {cut} bytes cut ...]\n".encode() if cut else b""
    return bytes(head) + middle + bytes(tail)


async def _communicate(
    process: asyncio.subprocess.Process, timeout_s: float, stdin: bytes | None = None
) -> tuple[bytes, bytes] | None:
    """The process's output, its head and tail kept; ``None`` when it outlived ``timeout_s`` and was killed.

    The process is killed whenever this returns without it having ended, cancelled or not.
    """
    assert process.stdout is not None and process.stderr is not None

    async def both() -> tuple[bytes, bytes]:
        if stdin is not None and process.stdin is not None:
            process.stdin.write(stdin)
            await process.stdin.drain()
            process.stdin.close()
        out, err = await asyncio.gather(_drain(process.stdout), _drain(process.stderr))  # type: ignore[arg-type]
        await process.wait()
        return out, err

    try:
        return await asyncio.wait_for(both(), timeout_s)
    except TimeoutError:
        return None
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


class HostRunner:
    """A command run on this machine, as this user, with the network: for the scripted agent only.

    The environment holds no key, and that is all it holds back: a process run here can read anything this user can.
    """

    isolated = False

    def __init__(self, python: Path) -> None:
        self.python = str(python)
        self.rerun = str(RERUN)

    async def run(self, argv: Sequence[str], *, cwd: Path, timeout_s: float, grading: bool = False) -> Ran:
        environment = {
            "PATH": f"{Path(self.python).parent}{os.pathsep}/usr/bin{os.pathsep}/bin",
            "HOME": str(cwd),
            "LANG": "C.UTF-8",
            "PYTHONUTF8": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        said = await _communicate(process, timeout_s)
        if said is None:
            return Ran(None, "", "", str(cwd))
        out, err = said
        code = process.returncode if process.returncode is not None else -1
        return Ran(code, out.decode("utf-8", "replace"), err.decode("utf-8", "replace"), str(cwd))


class Container:
    """A command run in a fresh container of ``image``, with no network, and removed when the command ends."""

    isolated = True
    python = PYTHON
    rerun = RERUN_AT

    def __init__(self, image: str) -> None:
        self.image = image

    def flags(self, name: str) -> list[str]:
        """How a trial's container is started: every limit and every closed door, and no mount from this machine."""
        uid, gid = USER.split(":")
        return [
            "--name", name,
            "--label", LABEL,
            "--rm",
            "--network", "none",
            "--user", USER,
            "--read-only",
            "--tmpfs", f"{WORK}:rw,nosuid,nodev,size={WORK_MB}m,uid={uid},gid={gid},mode=0700",
            "--tmpfs", f"/tmp:rw,nosuid,nodev,size={TMP_MB}m,uid={uid},gid={gid},mode=0700",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--cpus", str(CPUS),
            "--memory", f"{MEMORY_MB}m",
            "--memory-swap", f"{MEMORY_MB}m",
            "--pids-limit", str(PIDS),
            "--init",
            "--workdir", WORK,
            *(part for key, value in ENVIRONMENT.items() for part in ("--env", f"{key}={value}")),
        ]  # fmt: skip

    async def _docker(
        self, *args: str, stdin: bytes | None = None, stdout: int | IO[bytes] = asyncio.subprocess.PIPE
    ) -> asyncio.subprocess.Process:
        try:
            return await asyncio.create_subprocess_exec(
                "docker",
                *args,
                env=docker_environment(),
                stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                stdout=stdout,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as missing:
            raise IsolationUnavailable("there is no docker command on PATH") from missing

    async def _must(self, *args: str, stdin: bytes | None = None, timeout_s: float = 60.0) -> bytes:
        """``docker args``, its output.

        :raises IsolationUnavailable: It failed, or did not end within ``timeout_s``.
        """
        process = await self._docker(*args, stdin=stdin)
        said = await _communicate(process, timeout_s, stdin)
        if said is None:
            raise IsolationUnavailable(f"docker {args[0]} did not end within {timeout_s:.0f}s")
        out, err = said
        if process.returncode != 0:
            first = next((line for line in err.decode("utf-8", "replace").splitlines() if line.strip()), "")
            raise IsolationUnavailable(f"docker {args[0]} failed: {first or f'exit {process.returncode}'}")
        return out

    async def run(self, argv: Sequence[str], *, cwd: Path, timeout_s: float, grading: bool = False) -> Ran:
        """Copy ``cwd`` in, run ``argv`` in it for at most ``timeout_s``, and copy what it left back over ``cwd``.

        :raises IsolationUnavailable: The container could not be started, filled or read.
        """
        name = f"onramp-trial-{uuid.uuid4().hex[:16]}"
        # The container's own process ends on its own past this, and --rm removes it, should the harness die.
        lifetime = int(timeout_s + 4 * SLACK_S)
        await self._must("run", "--detach", *self.flags(name), self.image, "sleep", str(lifetime))
        try:
            await self._must("exec", "-i", name, "tar", "-x", "-f", "-", "-C", WORK, stdin=_tar_of(cwd))
            if grading:
                await self._must("exec", "-i", name, "tar", "-x", "-f", "-", "-C", "/tmp", stdin=_rerun_tar())
            limit = str(max(1, int(timeout_s)))
            started = time.monotonic()
            process = await self._docker("exec", "--workdir", WORK, name, "timeout", "-s", "KILL", limit, *argv)
            said = await _communicate(process, timeout_s + SLACK_S)
            elapsed = time.monotonic() - started
            code = process.returncode
            killed = said is None or (code == 137 and elapsed >= timeout_s - 1)
            await self._copy_out(name, cwd)
        finally:
            removed = await self._docker("rm", "-f", name)
            if await _communicate(removed, SLACK_S) is None or removed.returncode != 0:
                log.warning("onramp: container %s was not removed; it removes itself within %ss", name, lifetime)
        if killed or said is None:
            return Ran(None, "", "", WORK)
        out, err = said
        return Ran(code, out.decode("utf-8", "replace"), err.decode("utf-8", "replace"), WORK)

    async def _copy_out(self, name: str, cwd: Path) -> None:
        """What the command left in ``/work``, laid in place of ``cwd``; members that would land outside it dropped."""
        fresh = cwd.parent / f".{cwd.name}.{uuid.uuid4().hex[:8]}"
        with tempfile.TemporaryFile() as archive:
            process = await self._docker("exec", name, "tar", "-c", "-f", "-", "-C", WORK, ".", stdout=archive)
            assert process.stderr is not None
            try:
                err = await asyncio.wait_for(process.stderr.read(), SLACK_S * 2)
                await asyncio.wait_for(process.wait(), SLACK_S)
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
            if process.returncode != 0:
                said = err.decode("utf-8", "replace").strip() or f"exit {process.returncode}"
                raise IsolationUnavailable(f"could not copy the trial out of its container: {said}")
            archive.seek(0)
            fresh.mkdir()
            _extract(archive, fresh)
        fresh.chmod(0o755)
        shutil.rmtree(cwd)
        fresh.rename(cwd)
