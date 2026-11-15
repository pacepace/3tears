"""Where the agent's commands run: a container with no network, as a user that is not root, seeing only its trial.

Every process an agent causes to run (its ``python`` calls, and the regrade's re-runs of what it wrote) runs in a
fresh :class:`Container` of the trial image, one container per command, removed when the command ends. How it is
started is :meth:`Container.settings`, the one place it is said:

- no network (``network_mode="none"``: only loopback, no route), no capability, no new privileges, a read-only root
  filesystem, and no Docker socket or any other path of this machine mounted;
- user 10001, not root, with the environment :data:`ENVIRONMENT` and nothing of this process's (Docker adds
  ``HOSTNAME``, and the base image its ``PYTHON_VERSION`` and ``PYTHON_SHA256``);
- ``/work``, the trial's directory, is an anonymous in-memory volume of :data:`WORK_MB` MB, removed with the
  container: the directory is copied in as a tar stream before the command and back out after it (members that would
  land outside it, links included, are dropped); the docs build is installed in the image, root-owned and read-only;
- :data:`CPUS` CPU, :data:`MEMORY_MB` MB of memory and no swap, :data:`PIDS` processes, and a wall-clock limit per
  command, held inside the container (``timeout -s KILL``) and again outside it.

Docker is driven through its API (the ``docker`` package, as the repo's testcontainers drive it), never the CLI: the
daemon may be on another machine (this dev container talks to its host's), so nothing is bind-mounted and every byte
goes through the API; and the CLI here loses the output and exit code of any command that runs past a fraction of a
second (seen 2026-11-15). The image is built from ``container/Dockerfile`` at set-up (:func:`trial_image`), with the
network, never inside a trial.

:class:`HostRunner` runs a command on this machine instead. It is for the scripted agent only:
:func:`~.agent.attempt` refuses it for any other brain, and the command line offers it only beside ``--scripted``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import tarfile
import tempfile
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import IO, Any, Protocol

import docker
import requests
from docker.errors import DockerException
from docker.types import DriverConfig, Mount

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
DOCKERFILE = HERE / "container" / "Dockerfile"
RERUN = HERE / "rerun.py"

#: The image's name; its tag is a digest of the wheel and the Dockerfile.
IMAGE = "onramp-trial"
#: The label every image and container made here carries, so a leftover is found by it.
LABELS: Mapping[str, str] = {"onramp.trial": "1"}
#: The user every command runs as.
USER = "10001:10001"
#: Where the trial's directory is, inside the container.
WORK = "/work"
#: The interpreter the build is installed in, inside the container.
PYTHON = "/opt/build/bin/python"
#: Where the regrade's own reading puts ``rerun.py``: a volume only its containers have, so no agent command sees it.
GRADING = "/grading"
RERUN_AT = f"{GRADING}/rerun.py"
#: Limits on one command.
CPUS = 1
MEMORY_MB = 1024
PIDS = 128
WORK_MB = 256
TMP_MB = 64
#: How long past its own limit a command may take to be killed before the harness gives up on it.
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
#: The anonymous in-memory volume ``/work`` (and ``/grading``) is.
_IN_MEMORY = {"type": "tmpfs", "device": "tmpfs", "o": f"size={WORK_MB}m,uid=10001,gid=10001,mode=0700,nosuid,nodev"}

#: What a Docker call raises when the daemon cannot be reached or refuses: the SDK's own, and the transport's.
DOCKER_FAILURES: tuple[type[Exception], ...] = (DockerException, requests.RequestException)


class IsolationUnavailable(RuntimeError):
    """The container a command must run in could not be had: no daemon, no image, or the daemon failed it."""


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


def _client(timeout_s: float = 120.0) -> docker.DockerClient:
    """A client of the daemon the environment names (``DOCKER_HOST``, or the local socket).

    :raises IsolationUnavailable: No daemon answered.
    """
    try:
        client = docker.from_env(timeout=int(timeout_s))
        client.ping()
    except DOCKER_FAILURES as why:
        raise IsolationUnavailable(f"no Docker daemon answered: {why}") from why
    return client


def unavailable() -> str | None:
    """Why no trial container can run here, or ``None`` when the Docker daemon answers."""
    try:
        _client(30.0).close()
    except IsolationUnavailable as why:
        return str(why)
    return None


def trial_image(wheel: Path) -> str:
    """The trial image with ``wheel`` installed, built when it is not there yet; its tag names the wheel's digest.

    :raises IsolationUnavailable: Docker is unavailable, or the image did not build.
    """
    digest = hashlib.sha256(wheel.read_bytes() + DOCKERFILE.read_bytes()).hexdigest()
    tag = f"{IMAGE}:{digest[:16]}"
    client = _client()
    try:
        try:
            client.images.get(tag)
            return tag
        except docker.errors.ImageNotFound:
            pass  # built below
        with tempfile.TemporaryDirectory(prefix="onramp-image-") as context:
            shutil.copy2(wheel, Path(context) / wheel.name)
            shutil.copy2(DOCKERFILE, Path(context) / "Dockerfile")
            client.images.build(path=context, tag=tag, labels=dict(LABELS), rm=True, forcerm=True)
    except DOCKER_FAILURES as failed:
        raise IsolationUnavailable(f"the trial image did not build: {failed}") from failed
    finally:
        client.close()
    return tag


def package_dir(image: str) -> str:
    """Where ``threetears.evals`` is installed in ``image``, asked of the image with no network.

    :raises IsolationUnavailable: The image could not be asked, or does not import the package.
    """
    asked = "import pathlib, threetears.evals as e; print(pathlib.Path(e.__file__).parent)"
    client = _client()
    try:
        container = client.containers.run(
            image, ["sleep", "120"], detach=True, network_mode="none", user=USER,
            labels=dict(LABELS),
        )  # fmt: skip
        try:
            code, out = container.exec_run([PYTHON, "-I", "-c", asked], demux=True)
        finally:
            container.remove(force=True)
    except DOCKER_FAILURES as failed:
        raise IsolationUnavailable(f"could not ask the trial image {image}: {failed}") from failed
    finally:
        client.close()
    stdout, stderr = out
    answer = (stdout or b"").decode().strip()
    if code != 0 or not answer.startswith("/"):
        why = (stderr or b"").decode("utf-8", "replace").strip().splitlines()[-1:] or [f"it answered {answer!r}"]
        raise IsolationUnavailable(f"the trial image {image} does not import threetears.evals: {why[0]}")
    return answer


def copy_out_of_image(image: str, path: str, dest: Path) -> Path:
    """A copy of the directory ``path`` in ``image`` at ``dest``, which must not exist: the docs the agent reads.

    :raises IsolationUnavailable: The image could not be read.
    """
    client = _client()
    try:
        container = client.containers.create(image, ["true"], network_mode="none", labels=dict(LABELS))
        try:
            with tempfile.TemporaryFile() as archive:
                chunks, _ = container.get_archive(f"{path}/.")
                for chunk in chunks:
                    archive.write(chunk)
                archive.seek(0)
                dest.mkdir(parents=True)
                _extract(archive, dest)
        finally:
            container.remove(force=True)
    except DOCKER_FAILURES as failed:
        raise IsolationUnavailable(f"could not read {path} from the trial image {image}: {failed}") from failed
    finally:
        client.close()
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


def _owned(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """``info`` as the container's user owns it."""
    uid, gid = (int(part) for part in USER.split(":"))
    info.uid, info.gid, info.uname, info.gname = uid, gid, "agent", "agent"
    return info


def _tar_of(directory: Path) -> bytes:
    """``directory``'s contents as a tar stream whose every member, directories included, belongs to the container's
    user."""
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in sorted(directory.iterdir()):
            archive.add(path, arcname=path.name, filter=_owned)
    return buffer.getvalue()


def _rerun_tar() -> bytes:
    """``rerun.py`` as a tar stream to unpack in :data:`GRADING`."""
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        data = RERUN.read_bytes()
        info = _owned(tarfile.TarInfo(Path(RERUN_AT).name))
        info.size, info.mode = len(data), 0o644
        archive.addfile(info, BytesIO(data))
    return buffer.getvalue()


def _kept(chunks: Iterable[tuple[bytes | None, bytes | None]]) -> tuple[bytes, bytes]:
    """A command's demultiplexed output, each stream's first and last :data:`KEPT_BYTES` kept, saying how much was cut
    between."""
    kept: list[tuple[bytearray, bytearray, list[int]]] = [
        (bytearray(), bytearray(), [0]),
        (bytearray(), bytearray(), [0]),
    ]
    for pair in chunks:
        for chunk, (head, tail, cut) in zip(pair, kept, strict=True):
            if not chunk:
                continue
            room = KEPT_BYTES - len(head)
            head += chunk[:room]
            tail += chunk[room:] if room > 0 else chunk
            if len(tail) > KEPT_BYTES:
                cut[0] += len(tail) - KEPT_BYTES
                del tail[: len(tail) - KEPT_BYTES]
    out, err = (
        bytes(head) + (f"\n[... {cut[0]} bytes cut ...]\n".encode() if cut[0] else b"") + bytes(tail)
        for head, tail, cut in kept
    )
    return out, err


async def _communicate(process: asyncio.subprocess.Process, timeout_s: float) -> tuple[bytes, bytes] | None:
    """A process's output; ``None`` when it outlived ``timeout_s`` and was killed. The process is killed whenever this
    returns without it having ended, cancelled or not."""
    try:
        return await asyncio.wait_for(process.communicate(), timeout_s)
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

    def settings(self, name: str, *, lifetime_s: int, grading: bool) -> dict[str, Any]:  # Any: the SDK's options
        """How a trial's container is started: every limit and every closed door, and no path of this machine.

        Its own process is ``sleep lifetime_s``, so it ends on its own, and is removed (``auto_remove``), should the
        harness die with it running.
        """
        uid, gid = USER.split(":")
        mounts = [Mount(WORK, None, type="volume", driver_config=DriverConfig("local", dict(_IN_MEMORY)))]
        if grading:
            mounts.append(Mount(GRADING, None, type="volume", driver_config=DriverConfig("local", dict(_IN_MEMORY))))
        return {
            "image": self.image,
            "command": ["sleep", str(lifetime_s)],
            "name": name,
            "labels": dict(LABELS),
            "auto_remove": True,
            "network_mode": "none",
            "user": USER,
            "read_only": True,
            "mounts": mounts,
            "tmpfs": {"/tmp": f"rw,nosuid,nodev,size={TMP_MB}m,uid={uid},gid={gid},mode=0700"},
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges"],
            "privileged": False,
            "nano_cpus": int(CPUS * 1_000_000_000),
            "mem_limit": f"{MEMORY_MB}m",
            "memswap_limit": f"{MEMORY_MB}m",
            "pids_limit": PIDS,
            "init": True,
            "working_dir": WORK,
            "environment": dict(ENVIRONMENT),
        }

    async def run(self, argv: Sequence[str], *, cwd: Path, timeout_s: float, grading: bool = False) -> Ran:
        """Copy ``cwd`` in, run ``argv`` in it for at most ``timeout_s``, and copy what it left back over ``cwd``.

        :raises IsolationUnavailable: The container could not be started, filled or read.
        """
        name = f"onramp-trial-{uuid.uuid4().hex[:16]}"
        settings = self.settings(name, lifetime_s=int(timeout_s + 4 * SLACK_S), grading=grading)
        client = await asyncio.to_thread(_client)
        try:
            container = await asyncio.to_thread(client.containers.run, detach=True, **settings)
            try:
                await asyncio.to_thread(container.put_archive, WORK, _tar_of(cwd))
                if grading:
                    await asyncio.to_thread(container.put_archive, GRADING, _rerun_tar())
                limit = str(max(1, int(timeout_s)))
                started = time.monotonic()
                try:
                    code, out, err = await asyncio.wait_for(
                        asyncio.to_thread(self._exec, client, container.id, ["timeout", "-s", "KILL", limit, *argv]),
                        timeout_s + SLACK_S,
                    )
                except TimeoutError:
                    return Ran(None, "", "", WORK)
                await asyncio.to_thread(self._copy_out, container, cwd)
            finally:
                # Removing it ends a command still running, and the call waiting on it.
                await asyncio.to_thread(self._remove, container)
        except DOCKER_FAILURES as failed:
            raise IsolationUnavailable(f"the trial container failed: {failed}") from failed
        finally:
            client.close()
        if code == 137 and time.monotonic() - started >= timeout_s - 1:
            return Ran(None, "", "", WORK)
        return Ran(code, out.decode("utf-8", "replace"), err.decode("utf-8", "replace"), WORK)

    @staticmethod
    def _exec(client: docker.DockerClient, container: str, argv: list[str]) -> tuple[int, bytes, bytes]:
        """Run ``argv`` in ``container``'s ``/work``: its exit code and its output, head and tail kept."""
        started = client.api.exec_create(container, argv, workdir=WORK)
        out, err = _kept(client.api.exec_start(started["Id"], stream=True, demux=True))
        code = client.api.exec_inspect(started["Id"])["ExitCode"]
        return (code if code is not None else -1), out, err

    @staticmethod
    def _copy_out(container: Any, cwd: Path) -> None:  # Any: the SDK's container
        """What the command left in ``/work``, laid in place of ``cwd``; members that would land outside it dropped."""
        fresh = cwd.parent / f".{cwd.name}.{uuid.uuid4().hex[:8]}"
        with tempfile.TemporaryFile() as archive:
            chunks, _ = container.get_archive(f"{WORK}/.")
            for chunk in chunks:
                archive.write(chunk)
            archive.seek(0)
            fresh.mkdir()
            _extract(archive, fresh)
        fresh.chmod(0o755)
        shutil.rmtree(cwd)
        fresh.rename(cwd)

    @staticmethod
    def _remove(container: Any) -> None:  # Any: the SDK's container
        """Remove ``container`` and its volumes; one already gone is removed."""
        try:
            container.remove(force=True, v=True)
        except docker.errors.NotFound:
            return  # it removed itself
        except docker.errors.APIError as failed:
            if failed.status_code == 409:  # its own removal (auto_remove) is under way
                return
            log.warning(
                "onramp: container %s was not removed (%s); it removes itself when it ends", container.name, failed
            )
        except DOCKER_FAILURES as failed:
            log.warning(
                "onramp: container %s was not removed (%s); it removes itself when it ends", container.name, failed
            )
