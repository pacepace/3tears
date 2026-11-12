"""the services an integration test talks to -- Postgres, NATS, S3 -- as child processes or as containers.

:mod:`threetears.core.testing.fixtures` starts its session services through these. Where a service
comes from is decided in one order:

1. an explicit URL (``THREETEARS_TEST_POSTGRES_URL``, ``THREETEARS_TEST_NATS_URL``): the fixture uses
   the server already running there (CI's service containers). The fixtures read those; nothing here
   does;
2. **process mode**, ``THREETEARS_TEST_SERVICES=process``: each service runs as a child process of the
   test process, from binaries the testament image installs (``docker/testament.Dockerfile``):
   Postgres from the PGDG packages under ``/usr/lib/postgresql/<major>/bin``, nats-server from
   ``/opt/nats/<version>/nats-server``, and S3 from moto's server (``moto[server]``, in the lockfile).
   The pool's container has no Docker daemon, so this is how the integration suite runs there;
3. otherwise a testcontainer, as on a laptop with Docker.

**Process mode, in detail.**

- Postgres: the first start of a major in a process runs ``initdb`` once (``--no-sync``, the
  ``en_US.UTF-8`` locale the ``pgvector/pgvector`` image's initdb uses, scram for TCP clients), and
  every start copies that cluster, so a start costs a copy and a server boot. The server runs in the
  foreground as a child, with durability off (``fsync``, ``synchronous_commit``,
  ``full_page_writes``) and nothing that needs ``/dev/shm`` (``dynamic_shared_memory_type=mmap``,
  ``jit=off``), since every batch in a person's pool container shares a 64 MB ``/dev/shm``. It
  listens on 127.0.0.1 on a free port, chosen again when another process took it first, and the URL
  is ``postgresql://test:test@127.0.0.1:<port>/test``, as the docker image's is.
- NATS: the image tag ``nats:<version>-alpine`` names the binary; a version the image lacks is a
  missing service. The server reads the test's configuration, and the command line moves its client
  and monitoring ports to free ones (``-p -1 -m -1``, read back from a ports file) and, when the
  configuration enables JetStream, its store into the service's own directory (``-sd``, or the
  configuration's own ``store_dir`` rewritten to it), so two servers never share a store.
- S3: ``python -m moto.server`` on a socket this process binds first and hands it, so what answers
  is that server and no other; ready once it answers an HTTP request, with the bucket created as
  the container route creates it.

Each process-mode service lives in its own directory under ``/tmp/threetears-svc/<owner pid>-<id>/``,
named for the test process that started it. A batch killed at its timeout leaves its servers running
in the person's long-lived pool container, so every start first sweeps the directories of owners
that are gone: it kills the servers each recorded (only while the pid still names the same process)
and removes the directory. The test process also raises its own soft limit on open files, which the
servers inherit: one pool host's is 1,024.

**Missing services.** :func:`require_services` is the one gate: a test that cannot get its service
skips, unless ``THREETEARS_REQUIRE_SERVICES=1`` (the pool's integration suite), where it fails,
since a skip reads like a pass.
"""

from __future__ import annotations

import atexit
import contextlib
import importlib.util
import json
import os
import re
import resource
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import pytest

from threetears.core.testing.containers import check_docker_available, stagger_container_start
from threetears.observe import contain, get_logger

if TYPE_CHECKING:
    from testcontainers.core.container import DockerContainer

log = get_logger(__name__)

__all__ = [
    "require_services",
    "running_nats",
    "running_postgres",
    "running_s3",
]

#: ``process`` starts the services as child processes; unset (or empty) uses testcontainers.
_SERVICES_ENV = "THREETEARS_TEST_SERVICES"
_PROCESS = "process"
#: ``1`` turns a missing service from a skip into a failure.
_REQUIRE_ENV = "THREETEARS_REQUIRE_SERVICES"

#: where every process-mode service keeps its files. Short, because a Postgres socket path must fit
#: in 107 bytes, and shared by every test process in the container, so the sweep can find orphans.
_SERVICE_ROOT = Path("/tmp/threetears-svc")  # noqa: S108 - a fixed root is what lets one process sweep another's
_OWNER_NAME = re.compile(r"(\d+)-[0-9a-f]+")

_PG_ROOT = Path("/usr/lib/postgresql")
_PG_USER = "test"
_PG_PASSWORD = "test"  # noqa: S105 - the throwaway password the pgvector image's test user has too
_PG_DATABASE = "test"
_PG_LOCALE = "en_US.UTF-8"
#: the server's settings: durability off, and nothing that needs /dev/shm
_PG_SETTINGS: tuple[str, ...] = (
    "listen_addresses=127.0.0.1",
    "fsync=off",
    "synchronous_commit=off",
    "full_page_writes=off",
    "jit=off",
    "dynamic_shared_memory_type=mmap",
    "shared_buffers=32MB",
)

_NATS_ROOT = Path("/opt/nats")
#: a configured JetStream store directory, with its value, in each spelling nats-server reads: ``store_dir:
#: /tmp/js-store``, ``storedir = "x"``, ``store x``. Only as a key: not inside a name (``max_file_store``), a
#: subject (``kv.store.get``) or a quoted string.
_STORE_DIR = re.compile(r"""(?<![\w."'$-])(?:store_dir|storedir|store)(?:\s*[:=]\s*|\s+)(?:"[^"]*"|'[^']*'|[^\s,}]+)""")


def _code_of(line: str) -> str:
    """the part of a nats-server configuration line before its comment (``#`` or ``//``, outside quotes).

    ``//`` starts a comment only at the start of the line or after whitespace, so the ``//`` of an
    unquoted URL (``nats-route://host:6222``) is not one.

    :param line: one line
    :ptype line: str
    :return: the line up to its comment
    :rtype: str
    """
    quote = ""
    for index, char in enumerate(line):
        if quote:
            quote = "" if char == quote else quote
        elif char in "\"'":
            quote = char
        elif char == "#" or (line.startswith("//", index) and (index == 0 or line[index - 1].isspace())):
            return line[:index]
    return line


def _own_store(config: str, store: Path) -> tuple[str, bool]:
    """``config`` with every JetStream store it names, outside comments, made ``store``.

    The per-test configurations name the same store (``/tmp/js-store``), and nats-server refuses ``-sd``
    beside a configured one, so a named store is rewritten rather than overridden.

    :param config: the server configuration
    :ptype config: str
    :param store: this server's own store directory
    :ptype store: Path
    :return: the configuration, and whether it named a store
    :rtype: tuple[str, bool]
    """
    named = False
    lines: list[str] = []
    for line in config.splitlines(keepends=True):
        code = _code_of(line)
        rewritten, count = _STORE_DIR.subn(f'store_dir: "{store}"', code)
        named = named or count > 0
        lines.append(rewritten + line[len(code) :])
    return "".join(lines), named


_NATS_TAG = re.compile(r"(?:docker\.io/)?(?:library/)?nats:(\d+\.\d+\.\d+)(?:-alpine)?")

#: the soft open-files limit a process-mode service start raises this process to (and its servers inherit)
_OPEN_FILES = 65536
#: attempts at a free port before a Postgres start gives up, when another process keeps taking the one picked
_PORT_ATTEMPTS = 5
_START_TIMEOUT_S = 60.0
_STOP_TIMEOUT_S = 15.0


def _services_mode() -> str:
    """which way services start: ``process`` or ``docker``.

    :return: ``process`` under ``THREETEARS_TEST_SERVICES=process``, else ``docker``
    :rtype: str
    :raises ValueError: the variable names a mode that does not exist
    """
    mode = os.environ.get(_SERVICES_ENV, "")
    if mode == _PROCESS:
        return _PROCESS
    if mode == "":
        return "docker"
    raise ValueError(f"{_SERVICES_ENV}={mode!r}: the only mode is {_PROCESS!r} (unset uses testcontainers)")


def require_services(reason: str) -> NoReturn:
    """stop a test whose service is not there: skip it, or fail it under ``THREETEARS_REQUIRE_SERVICES=1``.

    The pool's integration suite sets the flag, because there a skip would read as a pass: every
    service it names is meant to be in the image.

    :param reason: what is missing, and where it was looked for
    :ptype reason: str
    :return: never
    :rtype: NoReturn
    """
    if os.environ.get(_REQUIRE_ENV) == "1":
        pytest.fail(f"{reason} ({_REQUIRE_ENV}=1 makes a missing service a failure, not a skip)", pytrace=False)
    pytest.skip(reason)


@dataclass(frozen=True, slots=True)
class RunningNats:
    """a nats-server that :func:`running_nats` started.

    :param uri: the ``nats://`` URI it serves clients on
    :ptype uri: str
    :param cluster_url: the ``nats://`` URL of its cluster listener, or None when it has none (or runs
        in a container, whose cluster port is not published)
    :ptype cluster_url: str | None
    """

    uri: str
    cluster_url: str | None
    _lame_duck: Callable[[], None] = field(repr=False)

    def enter_lame_duck(self) -> None:
        """put the server in lame-duck mode, as a rolling restart does: it stops accepting clients
        and closes the ones it has over its lame-duck duration, then exits.

        :return: None
        :rtype: None
        """
        self._lame_duck()


# --------------------------------------------------------------------------------------------------
# process mode: owner directories, the orphan sweep, child processes
# --------------------------------------------------------------------------------------------------


class _ProcessServices:
    """this process's share of ``/tmp/threetears-svc``: its owner directory and its Postgres templates."""

    def __init__(self) -> None:
        """start with no directory: the first service start makes it."""
        self.lock = threading.Lock()
        self.owner: Path | None = None
        self.owner_pid = 0
        self.templates: dict[int, Path] = {}


_STATE = _ProcessServices()


def _start_time(pid: int) -> str | None:
    """the kernel's start time of ``pid`` (``/proc/<pid>/stat`` field 22), or None when no live process has it.

    A zombie counts as gone: its pid is still taken, but it runs nothing.

    :param pid: the process id
    :ptype pid: int
    :return: the start time in clock ticks since boot, as written, or None
    :rtype: str | None
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except FileNotFoundError, ProcessLookupError:
        # NOSILENT: no /proc entry is the answer this asks for: no such process
        return None
    fields = stat.rsplit(")", 1)[1].split()
    if fields[0] in ("Z", "X"):
        return None
    return fields[19]


def _raise_open_files() -> None:
    """raise this process's soft open-files limit, which the servers it starts inherit.

    :return: None
    :rtype: None
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = _OPEN_FILES if hard == resource.RLIM_INFINITY else min(hard, _OPEN_FILES)
    if soft != resource.RLIM_INFINITY and soft < want:
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))


def _remove_tree(path: Path) -> None:
    """remove ``path`` and everything under it; a part already gone (another sweep took it) is fine.

    :param path: the directory
    :ptype path: Path
    :return: None
    :rtype: None
    """

    def gone_already(_function: Callable[..., object], _path: str, exc: BaseException) -> None:
        if not isinstance(exc, FileNotFoundError):
            raise exc

    shutil.rmtree(path, onexc=gone_already)


def _owner_alive(directory: Path, pid: int) -> bool:
    """whether the test process that owns ``directory`` still runs.

    :param directory: an owner directory
    :ptype directory: Path
    :param pid: the pid its name records
    :ptype pid: int
    :return: True while that very process (same pid, same start time) runs
    :rtype: bool
    """
    started = _start_time(pid)
    if started is None:
        return False
    try:
        recorded = (directory / "owner").read_text(encoding="ascii").strip()
    except FileNotFoundError:
        # NOSILENT: an owner that is still writing its file is alive, which the live pid says already
        return True
    return recorded == started


def _sweep_orphans() -> None:
    """stop the servers of every owner directory whose test process is gone, and remove the directory.

    A server is killed only while its pid still names the process that was recorded (same start
    time), so a pid the kernel has since handed to something else is left alone.

    :return: None
    :rtype: None
    """
    for directory in _SERVICE_ROOT.iterdir():
        name = _OWNER_NAME.fullmatch(directory.name)
        if name is None or directory == _STATE.owner:
            continue
        try:
            _sweep_one(directory, int(name.group(1)))
        except Exception as exc:  # noqa: BLE001 - contained: one bad orphan must not fail an unrelated test
            contain(exc, log=log, what=f"sweeping the orphaned test services in {directory}")


def _sweep_one(directory: Path, owner_pid: int) -> None:
    """stop the servers ``directory`` recorded and remove it, unless its owner still runs.

    :param directory: an owner directory other than this process's
    :ptype directory: Path
    :param owner_pid: the pid its name records
    :ptype owner_pid: int
    :return: None
    :rtype: None
    """
    if _owner_alive(directory, owner_pid):
        return
    try:
        recorded = (directory / "pids").read_text(encoding="ascii").split("\n")
    except FileNotFoundError:
        # NOSILENT: an owner that never started a server (or another sweep removed it) left none to stop
        recorded = []
    for line in recorded:
        if not line.strip():
            continue
        pid, started = line.split()
        if _start_time(int(pid)) == started:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                # NOSILENT: it exited between the check and the kill, which is the outcome wanted
                pass
            log.warning(
                "killed an orphaned test service",
                extra={"extra_data": {"pid": int(pid), "owner_dir": str(directory)}},
            )
    _remove_tree(directory)


def _remove_owner(owner: Path, pid: int) -> None:
    """remove this process's owner directory (its Postgres templates) as the process exits.

    Its servers have stopped by then, each with its fixture. A forked child inherits this exit hook
    but not the directory, so it does nothing there. A removal that fails is contained: the next
    start in the container sweeps the directory once this process is gone.

    :param owner: the owner directory
    :ptype owner: Path
    :param pid: the process that made it
    :ptype pid: int
    :return: None
    :rtype: None
    """
    if os.getpid() != pid:
        return
    try:
        _remove_tree(owner)
    except Exception as exc:  # noqa: BLE001 - contained: the sweep removes it later
        contain(exc, log=log, what=f"removing the test services' owner directory {owner}")


def _own_service_root() -> None:
    """make ``/tmp/threetears-svc`` this user's private directory, or refuse one that is not.

    The sweep kills the pids the directories under it record, so a root another user could write
    to would let them name any of this user's processes.

    :return: None
    :rtype: None
    :raises RuntimeError: the root is a symlink, not a directory, or another user's
    """
    try:
        _SERVICE_ROOT.mkdir(mode=0o700)
    except FileExistsError:
        # NOSILENT: an existing root is checked below, which is the point
        pass
    found = os.lstat(_SERVICE_ROOT)
    if stat.S_ISLNK(found.st_mode) or not stat.S_ISDIR(found.st_mode) or found.st_uid != os.getuid():
        raise RuntimeError(
            f"{_SERVICE_ROOT} must be a directory of this user's (uid {os.getuid()}), not a symlink or another "
            f"user's (mode {stat.filemode(found.st_mode)}, uid {found.st_uid}); remove it, and the next start makes it"
        )
    if stat.S_IMODE(found.st_mode) != 0o700:
        os.chmod(_SERVICE_ROOT, 0o700)


def _service_dir(kind: str) -> Path:
    """a fresh directory for one service, under this process's owner directory; sweeps orphans first.

    :param kind: the service, for the directory's name (``pg``, ``nats``, ``s3``)
    :ptype kind: str
    :return: the new directory
    :rtype: Path
    """
    _raise_open_files()
    with _STATE.lock:
        _own_service_root()
        if _STATE.owner is None or _STATE.owner_pid != os.getpid():
            pid = os.getpid()
            owner = _SERVICE_ROOT / f"{pid}-{secrets.token_hex(6)}"
            owner.mkdir(mode=0o700)
            # written whole, then renamed into place: a sweep never reads a half-written start time
            # and takes a live owner for a dead one
            (owner / "owner.tmp").write_text(f"{_start_time(pid)}\n", encoding="ascii")
            os.replace(owner / "owner.tmp", owner / "owner")
            _STATE.owner, _STATE.owner_pid, _STATE.templates = owner, pid, {}
            atexit.register(_remove_owner, owner, pid)
        _sweep_orphans()
        return Path(tempfile.mkdtemp(prefix=f"{kind}-", dir=_STATE.owner))


def _spawn(
    argv: list[str], *, log_path: Path, env: dict[str, str] | None = None, pass_fds: tuple[int, ...] = ()
) -> subprocess.Popen[bytes]:
    """start a server as a child in its own session, its output in ``log_path``, and record it for the sweep.

    :param argv: the command
    :ptype argv: list[str]
    :param log_path: where its stdout and stderr go
    :ptype log_path: Path
    :param env: its environment, or None for this process's
    :ptype env: dict[str, str] | None
    :param pass_fds: descriptors it inherits
    :ptype pass_fds: tuple[int, ...]
    :return: the child
    :rtype: subprocess.Popen[bytes]
    """
    assert _STATE.owner is not None, "a service directory comes first"
    with log_path.open("ab") as out:
        child = subprocess.Popen(  # noqa: S603 - argv is built here from fixed binaries
            argv,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
            pass_fds=pass_fds,
        )
    with _STATE.lock, (_STATE.owner / "pids").open("a", encoding="ascii") as pids:
        pids.write(f"{child.pid} {_start_time(child.pid)}\n")
    return child


def _stop(child: subprocess.Popen[bytes], stop_signal: signal.Signals) -> None:
    """stop a server: ``stop_signal``, then a kill once :data:`_STOP_TIMEOUT_S` passes.

    :param child: the server
    :ptype child: subprocess.Popen[bytes]
    :param stop_signal: the signal that shuts it down cleanly
    :ptype stop_signal: signal.Signals
    :return: None
    :rtype: None
    """
    if child.poll() is not None:
        return
    try:
        child.send_signal(stop_signal)
        child.wait(timeout=_STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        log.warning("a test service ignored its stop signal; killing it", extra={"extra_data": {"pid": child.pid}})
        child.kill()
        child.wait(timeout=_STOP_TIMEOUT_S)


def _tail(path: Path, lines: int = 30) -> str:
    """the last ``lines`` lines of a server's log, for a failure message.

    :param path: the log
    :ptype path: Path
    :param lines: how many
    :ptype lines: int
    :return: the lines, or a note that there is no log
    :rtype: str
    """
    if not path.exists():
        return "(no log)"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


@contextlib.contextmanager
def _teardown(directory: Path, what: str) -> Iterator[None]:
    """remove a service's directory once the block (the server's stop) is done, whatever it raised.

    A failure to remove it is contained, not raised: the server is already stopped, and the next
    sweep cannot take a live owner's directory, so it is logged where the run shows it.

    :param directory: the service's directory
    :ptype directory: Path
    :param what: the service, for the log
    :ptype what: str
    :yield: nothing
    :rtype: Iterator[None]
    """
    try:
        yield
    finally:
        try:
            _remove_tree(directory)
        except Exception as exc:  # noqa: BLE001 - contained: teardown must not mask the test's own outcome
            contain(exc, log=log, what=f"removing the {what} test service's directory {directory}")


def _free_port() -> int:
    """a TCP port on 127.0.0.1 that nothing listens on at this moment.

    :return: the port
    :rtype: int
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


# --------------------------------------------------------------------------------------------------
# Postgres
# --------------------------------------------------------------------------------------------------


def _postgres_major(image: str) -> int:
    """the Postgres major an image tag names: ``pgvector/pgvector:pg16`` and ``postgres:16-alpine`` are 16.

    :param image: the image the test asked for
    :ptype image: str
    :return: the major
    :rtype: int
    """
    found = re.fullmatch(r"(?:[\w.-]+/)*(?:postgres|pgvector):(?:pg)?(\d+)(?:[.-][\w.-]*)?", image)
    if found is None:
        pytest.fail(
            f"process-mode Postgres stands in for postgres:<major> and pgvector/pgvector:pg<major> images; "
            f"{image!r} is neither",
            pytrace=False,
        )
    return int(found.group(1))


def _postgres_bin(major: int) -> Path:
    """the directory holding Postgres ``major``'s server binaries, or the missing-service stop.

    :param major: the Postgres major
    :ptype major: int
    :return: the ``bin`` directory
    :rtype: Path
    """
    bin_dir = _PG_ROOT / str(major) / "bin"
    if not (bin_dir / "postgres").is_file() or not (bin_dir / "initdb").is_file():
        require_services(
            f"Postgres {major} is not installed here ({bin_dir}/postgres): process-mode services start "
            f"Postgres from the testament image's PGDG packages"
        )
    return bin_dir


def _run_checked(argv: list[str], *, what: str, stdin: str | None = None) -> None:
    """run a setup command to completion; fail with its output if it fails.

    :param argv: the command
    :ptype argv: list[str]
    :param what: what it does, for the failure
    :ptype what: str
    :param stdin: text for its standard input
    :ptype stdin: str | None
    :return: None
    :rtype: None
    :raises RuntimeError: the command exited non-zero
    """
    done = subprocess.run(  # noqa: S603 - argv is built here from fixed binaries
        argv, input=stdin, capture_output=True, text=True, timeout=_START_TIMEOUT_S, check=False
    )
    if done.returncode != 0:
        raise RuntimeError(f"{what} failed (exit {done.returncode}): {argv}\n{done.stdout}\n{done.stderr}")


def _postgres_template(major: int) -> Path:
    """this process's initdb'd Postgres ``major`` cluster, with the ``test`` database, made on first use.

    :param major: the Postgres major
    :ptype major: int
    :return: the cluster's data directory, which each start copies
    :rtype: Path
    """
    bin_dir = _postgres_bin(major)
    with _STATE.lock:
        cached = _STATE.templates.get(major)
        if cached is not None:
            return cached
        assert _STATE.owner is not None, "a service directory comes first"
        data = _STATE.owner / f"pg{major}-template"
        password = _STATE.owner / f"pg{major}-password"
        password.write_text(f"{_PG_PASSWORD}\n", encoding="ascii")
        try:
            _initdb(bin_dir, data, password, major)
        except BaseException:
            # a half-made cluster would make every later start in this process fail on a non-empty directory
            if data.exists():
                _remove_tree(data)
            raise
        _STATE.templates[major] = data
        return data


def _initdb(bin_dir: Path, data: Path, password: Path, major: int) -> None:
    """initdb a cluster into ``data`` and create the ``test`` database in it.

    :param bin_dir: Postgres ``major``'s binaries
    :ptype bin_dir: Path
    :param data: the data directory to make
    :ptype data: Path
    :param password: a file holding the ``test`` user's password
    :ptype password: Path
    :param major: the Postgres major, for failures
    :ptype major: int
    :return: None
    :rtype: None
    """
    _run_checked(
        [
            str(bin_dir / "initdb"),
            "-D",
            str(data),
            "-U",
            _PG_USER,
            f"--pwfile={password}",
            "--auth-local=trust",
            "--auth-host=scram-sha-256",
            f"--locale={_PG_LOCALE}",
            "--encoding=UTF8",
            "--no-sync",
            "--no-instructions",
        ],
        what=f"initdb (Postgres {major})",
    )
    # the database the URL names, as the image's POSTGRES_DB makes it
    _run_checked(
        [str(bin_dir / "postgres"), "--single", "-D", str(data), "-c", "fsync=off", "postgres"],
        what=f"creating the {_PG_DATABASE} database (Postgres {major})",
        stdin=f"CREATE DATABASE {_PG_DATABASE};\n",
    )


def _await_postgres(child: subprocess.Popen[bytes], log_path: Path) -> bool:
    """wait until the server says it accepts connections.

    :param child: the server
    :ptype child: subprocess.Popen[bytes]
    :param log_path: its log
    :ptype log_path: Path
    :return: True once ready; False when it exited because its port was taken (another start may retry)
    :rtype: bool
    """
    deadline = time.monotonic() + _START_TIMEOUT_S
    while True:
        said = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        if "ready to accept connections" in said:
            return True
        if child.poll() is not None:
            if "Address already in use" in said:
                return False
            pytest.fail(f"Postgres exited (code {child.returncode}) before it was ready:\n{_tail(log_path)}")
        if time.monotonic() > deadline:
            _stop(child, signal.SIGQUIT)
            pytest.fail(f"Postgres was not ready within {_START_TIMEOUT_S:.0f}s:\n{_tail(log_path)}")
        time.sleep(0.05)


@contextlib.contextmanager
def _postgres_process(image: str) -> Iterator[str]:
    """a Postgres server as a child process, standing in for ``image``; yields its URL.

    :param image: the image the test asked for, which names the major
    :ptype image: str
    :yield: the ``postgresql://`` URL
    :rtype: Iterator[str]
    """
    major = _postgres_major(image)
    _postgres_bin(major)
    directory = _service_dir("pg")
    with _teardown(directory, f"Postgres {major}"):
        template = _postgres_template(major)
        bin_dir = _PG_ROOT / str(major) / "bin"
        data = directory / "data"
        shutil.copytree(template, data, symlinks=True)
        for attempt in range(_PORT_ATTEMPTS):
            log_path = directory / f"postgres-{attempt}.log"
            port = _free_port()
            argv = [str(bin_dir / "postgres"), "-D", str(data), "-c", f"port={port}"]
            argv += ["-c", f"unix_socket_directories={directory}"]
            for setting in _PG_SETTINGS:
                argv += ["-c", setting]
            child = _spawn(argv, log_path=log_path)
            try:
                ready = _await_postgres(child, log_path)
            except BaseException:
                _stop(child, signal.SIGQUIT)
                raise
            if ready:
                break
        else:
            pytest.fail(f"Postgres {major} found no free port in {_PORT_ATTEMPTS} tries:\n{_tail(log_path)}")
        try:
            yield f"postgresql://{_PG_USER}:{_PG_PASSWORD}@127.0.0.1:{port}/{_PG_DATABASE}"
        finally:
            # SIGINT is Postgres's fast shutdown: it ends every session and stops
            _stop(child, signal.SIGINT)


@contextlib.contextmanager
def _postgres_container(image: str) -> Iterator[str]:
    """a Postgres testcontainer running ``image``; yields its asyncpg-compatible URL.

    :param image: the image
    :ptype image: str
    :yield: the ``postgresql://`` URL (testcontainers' ``postgresql+psycopg2://`` normalised)
    :rtype: Iterator[str]
    """
    if not check_docker_available():
        require_services("Docker not available")
    stagger_container_start()

    from testcontainers.community.postgres import PostgresContainer  # noqa: PLC0415

    with PostgresContainer(image) as container:
        raw_url = container.get_connection_url()
        normalised = raw_url
        if normalised.startswith("postgresql+psycopg2://"):
            normalised = normalised.replace(
                "postgresql+psycopg2://",
                "postgresql://",
                1,
            )
        yield normalised


def running_postgres(image: str) -> contextlib.AbstractContextManager[str]:
    """a Postgres server for the block: a child process in process mode, else a testcontainer of ``image``.

    In process mode ``image`` names only the major (``pgvector/pgvector:pg16``, ``postgres:16``): the
    server is the image's PGDG Postgres of that major, which carries pgvector either way.

    :param image: the docker image the server stands for
    :ptype image: str
    :return: a context manager yielding the server's ``postgresql://`` URL
    :rtype: contextlib.AbstractContextManager[str]
    """
    if _services_mode() == _PROCESS:
        return _postgres_process(image)
    return _postgres_container(image)


# --------------------------------------------------------------------------------------------------
# NATS
# --------------------------------------------------------------------------------------------------


def _nats_binary(image: str) -> Path:
    """the nats-server binary standing in for ``image`` (``nats:<version>-alpine``).

    :param image: the image the test asked for
    :ptype image: str
    :return: the binary
    :rtype: Path
    """
    found = _NATS_TAG.fullmatch(image)
    if found is None:
        pytest.fail(
            f"process-mode NATS runs {_NATS_ROOT}/<version>/nats-server, and {image!r} names no nats:<version> release",
            pytrace=False,
        )
    server = _NATS_ROOT / found.group(1) / "nats-server"
    if not server.is_file():
        require_services(
            f"nats-server {found.group(1)} ({image}) is not installed at {server}: docker/testament.Dockerfile "
            f"installs one per version the tests pin"
        )
    return server


def _read_ports(ports_dir: Path) -> dict[str, list[str]] | None:
    """the ports file nats-server writes once it is ready, or None before it has.

    :param ports_dir: the ``--ports_file_dir``
    :ptype ports_dir: Path
    :return: the parsed file
    :rtype: dict[str, list[str]] | None
    """
    for path in ports_dir.glob("*.ports"):
        try:
            ports: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # NOSILENT: a file still being written; the next poll reads it whole
            return None
        return ports
    return None


@contextlib.contextmanager
def _nats_process(config: str | None, image: str) -> Iterator[RunningNats]:
    """a nats-server as a child process, standing in for ``image``; yields it.

    :param config: the server configuration, or None for the defaults
    :ptype config: str | None
    :param image: the image the test asked for, which names the version
    :ptype image: str
    :yield: the running server
    :rtype: Iterator[RunningNats]
    """
    server = _nats_binary(image)
    directory = _service_dir("nats")
    with _teardown(directory, "NATS"):
        ports_dir = directory / "ports"
        ports_dir.mkdir()
        argv = [str(server), "-a", "127.0.0.1", "-p", "-1", "-m", "-1", "--ports_file_dir", str(ports_dir)]
        argv += ["-P", str(directory / "nats.pid")]
        if config is not None:
            store = directory / "jetstream"
            config, named_a_store = _own_store(config, store)
            if not named_a_store and any(re.match(r"\s*jetstream\b", _code_of(line)) for line in config.splitlines()):
                # the server's own store, never the shared default /tmp/nats/jetstream
                argv += ["-sd", str(store)]
            conf = directory / "nats.conf"
            conf.write_text(config, encoding="utf-8")
            argv += ["-c", str(conf)]
        log_path = directory / "nats.log"
        child = _spawn(argv, log_path=log_path)
        try:
            deadline = time.monotonic() + _START_TIMEOUT_S
            while (ports := _read_ports(ports_dir)) is None or not ports.get("nats"):
                if child.poll() is not None:
                    pytest.fail(f"nats-server exited (code {child.returncode}) before it was ready:\n{_tail(log_path)}")
                if time.monotonic() > deadline:
                    pytest.fail(f"nats-server was not ready within {_START_TIMEOUT_S:.0f}s:\n{_tail(log_path)}")
                time.sleep(0.02)
            clusters = ports.get("cluster") or []

            def lame_duck() -> None:
                _run_checked([str(server), "--signal", f"ldm={child.pid}"], what="nats-server --signal ldm")

            yield RunningNats(uri=ports["nats"][0], cluster_url=clusters[0] if clusters else None, _lame_duck=lame_duck)
        finally:
            _stop(child, signal.SIGTERM)


@contextlib.contextmanager
def _nats_container(config: str | None, image: str) -> Iterator[RunningNats]:
    """a nats-server testcontainer of ``image``; yields it.

    :param config: the server configuration, or None for the image's defaults
    :ptype config: str | None
    :param image: the image; with ``config``, one that carries a shell
    :ptype image: str
    :yield: the running server
    :rtype: Iterator[RunningNats]
    """
    if not check_docker_available():
        require_services("Docker not available")
    stagger_container_start()

    from threetears.core.testing.fixtures import nats_server, nats_server_uri  # noqa: PLC0415 - fixtures imports this module

    container: DockerContainer = nats_server(image, config=config)
    with container:
        container_id = container.get_wrapped_container().id

        def lame_duck() -> None:
            _run_checked(
                ["docker", "exec", container_id, "nats-server", "--signal", "ldm=1"],
                what="nats-server --signal ldm in the container",
            )

        yield RunningNats(uri=nats_server_uri(container), cluster_url=None, _lame_duck=lame_duck)


def running_nats(config: str | None, *, image: str) -> contextlib.AbstractContextManager[RunningNats]:
    """a nats-server for the block: a child process in process mode, else a testcontainer of ``image``.

    In process mode the configuration's own ``port`` and ``http`` give way to free ports, so a
    configuration written for a container (``port: 4222``) serves as it is.

    :param config: the server configuration, or None for the defaults
    :ptype config: str | None
    :param image: the ``nats:<version>-alpine`` image the server stands for
    :ptype image: str
    :return: a context manager yielding the running server
    :rtype: contextlib.AbstractContextManager[RunningNats]
    """
    if _services_mode() == _PROCESS:
        return _nats_process(config, image)
    return _nats_container(config, image)


# --------------------------------------------------------------------------------------------------
# S3
# --------------------------------------------------------------------------------------------------

#: the bucket every S3 service is created with
_S3_BUCKET = "threetears-test-objects"


@contextlib.contextmanager
def _direct_to(host: str) -> Iterator[None]:
    """Reach ``host`` without the environment's HTTP proxy while the block runs.

    A test talks to its own server, never through an egress proxy: behind one
    (a dev container with ``HTTP_PROXY`` set), urllib, boto3 and aiobotocore all
    sent the request for the S3 container's bridge address to the proxy, which
    cannot reach it, and the fixture failed with "never answered". Adding the
    host to ``NO_PROXY`` is what every one of those clients honours.

    :param host: the server's host address
    :ptype host: str
    :yield: nothing
    :rtype: Iterator[None]
    """
    saved = {name: os.environ.get(name) for name in ("NO_PROXY", "no_proxy")}
    for name, value in saved.items():
        os.environ[name] = f"{value},{host}" if value else host
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _s3_answers(endpoint: str, *, stopped: Callable[[], bool]) -> bool:
    """wait until the S3 server at ``endpoint`` answers an HTTP request.

    Readiness is "answers an S3 call", not "the port accepts a socket". The listener binds before the
    app is ready to route, so a bare connect check hands the first real request a 502.

    :param endpoint: the server's URL
    :ptype endpoint: str
    :param stopped: whether the server has exited (a child that lost its port), polled between tries
    :ptype stopped: Callable[[], bool]
    :return: True once it answers; False when it stopped first
    :rtype: bool
    """
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    deadline = time.monotonic() + _START_TIMEOUT_S
    while True:
        try:
            with urllib.request.urlopen(endpoint, timeout=5) as probe:  # noqa: S310
                if probe.status < 500:
                    return True
        except urllib.error.HTTPError as exc:
            # An HTTP error IS a served response: the app is up and simply
            # dislikes a bare GET at the root, which is all this probe
            # needed to learn.
            if exc.code < 500:
                return True
        except urllib.error.URLError, OSError:
            # NOSILENT: a refused or reset connection IS the not-ready
            # signal this loop polls for. Never becoming ready is the
            # failure that matters, and it is raised at the deadline.
            pass
        if stopped():
            return False
        if time.monotonic() > deadline:
            pytest.fail(f"S3 server at {endpoint} never answered")
        time.sleep(0.2)


def _create_bucket(endpoint: str, credentials: tuple[str, str]) -> None:
    """create the test bucket on the S3 server at ``endpoint``.

    :param endpoint: the server's URL
    :ptype endpoint: str
    :param credentials: the access and secret key
    :ptype credentials: tuple[str, str]
    :return: None
    :rtype: None
    """
    import boto3  # noqa: PLC0415

    access_key, secret_key = credentials
    boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
    ).create_bucket(Bucket=_S3_BUCKET)


@contextlib.contextmanager
def _s3_process(credentials: tuple[str, str]) -> Iterator[tuple[str, str]]:
    """moto's S3 server as a child process; yields (endpoint, bucket).

    :param credentials: the access and secret key
    :ptype credentials: tuple[str, str]
    :yield: the endpoint URL and the created bucket
    :rtype: Iterator[tuple[str, str]]
    """
    if importlib.util.find_spec("moto") is None:
        require_services("moto is not installed (moto[server], in the workspace's dev dependencies)")
    directory = _service_dir("s3")
    with _teardown(directory, "S3"), _direct_to("127.0.0.1"):
        log_path = directory / "moto.log"
        # The listening socket is bound here and handed to the server, which werkzeug (moto's server)
        # takes through its reloader's protocol: WERKZEUG_RUN_MAIN=true and WERKZEUG_SERVER_FD. The port
        # is this process's from before the server starts, so whatever answers on it is this server,
        # never another batch's that won a race for a port both had probed as free.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            port: int = listener.getsockname()[1]
            env = {**os.environ, "WERKZEUG_RUN_MAIN": "true", "WERKZEUG_SERVER_FD": str(listener.fileno())}
            argv = [sys.executable, "-m", "moto.server", "-H", "127.0.0.1", "-p", str(port)]
            child = _spawn(argv, log_path=log_path, env=env, pass_fds=(listener.fileno(),))
        endpoint = f"http://127.0.0.1:{port}"
        try:
            if not _s3_answers(endpoint, stopped=lambda: child.poll() is not None):
                pytest.fail(f"moto's server exited (code {child.returncode}) before it answered:\n{_tail(log_path)}")
            _create_bucket(endpoint, credentials)
            yield endpoint, _S3_BUCKET
        finally:
            _stop(child, signal.SIGTERM)


@contextlib.contextmanager
def _s3_container(credentials: tuple[str, str]) -> Iterator[tuple[str, str]]:
    """moto's S3 server in a testcontainer; yields (endpoint, bucket).

    :param credentials: the access and secret key
    :ptype credentials: tuple[str, str]
    :yield: the endpoint URL and the created bucket
    :rtype: Iterator[tuple[str, str]]
    """
    if not check_docker_available():
        require_services("Docker not available")
    stagger_container_start()

    from testcontainers.core.container import DockerContainer  # noqa: PLC0415

    with (
        DockerContainer("motoserver/moto:latest").with_exposed_ports(5000) as container,
        _direct_to(container.get_container_host_ip()),
    ):
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5000)
        endpoint = f"http://{host}:{port}"
        _s3_answers(endpoint, stopped=lambda: False)
        _create_bucket(endpoint, credentials)
        yield endpoint, _S3_BUCKET


def running_s3(credentials: tuple[str, str]) -> contextlib.AbstractContextManager[tuple[str, str]]:
    """an S3-compatible server (moto) for the block, with its bucket made: a child process in process
    mode, else a testcontainer.

    :param credentials: the access and secret key it accepts
    :ptype credentials: tuple[str, str]
    :return: a context manager yielding the endpoint URL and the bucket's name
    :rtype: contextlib.AbstractContextManager[tuple[str, str]]
    """
    if _services_mode() == _PROCESS:
        return _s3_process(credentials)
    return _s3_container(credentials)
