"""Certify the evals reading's error-rate claims under the null, and write ``packages/evals/tests/fixtures/stats_certificate.json``.

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is not installed. Run it from the
repository's root whenever ``test_stats_certificate.py`` says the certificate is stale:

    uv run python packages/evals/scripts/certify_stats.py

It runs every shard of every claim (``packages/evals/tests/stats_certificate.py``), restated or through the product, on
the testament pool as one request of the ``simulation`` suite (``test_certified_simulations.py``), and reads each
shard's counts back from the metrics it records under this run's key. Nothing runs here but the request. Then it writes
the certificate (every claim, its count, its Clopper-Pearson bounds, its seeds and the digest of the files that decide
it) and exits 1 if any claim does not clear its target. Commit the certificate it writes.

It never hangs silently. It prints, as it goes, its start, the pool request's id once testament accepts it, the
request's progress every minute, the metrics it reads back and the end of each phase. A ``testament run`` past
:data:`RUN_TIMEOUT_S` or a ``testament metrics`` past :data:`METRICS_TIMEOUT_S` is stopped and fails the run, naming
which and how long it ran. A request the pool holds queued behind others' work counts as alive up to the run's limit,
:data:`RUN_TIMEOUT_S`, which stops it like any other run. The run stalls, writing every thread's stack, naming the phase
and why, and exiting 3, when the coordinator stops answering for ``--stall-minutes``, when the request leaves the
coordinator with no verdict for as long, when the request has batches running and none finishes for
:data:`STUCK_MINUTES`, or, anywhere else, when nothing moves for ``--stall-minutes``. ``kill -USR1 <pid>`` writes every
thread's stack at any time from the first line on, and ``py-spy dump --pid <pid>`` shows the same from outside.

    # Run every shard here instead, when the pool can't take them (an hour or more, and it loads this machine):
    uv run python packages/evals/scripts/certify_stats.py --local
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH", "")]))

#: How long nothing may move before the run is called stalled: no shard done, no line from testament, the coordinator
#: not answering. The slowest shard takes about three minutes on the pool.
STALL_MINUTES = 10.0

#: How long a request may have batches running on the pool with none of them finishing before it is called stuck.
STUCK_MINUTES = 30.0

#: What the run exits with when it stalls, fails before a verdict, or is stopped by a signal (128 plus its number).
STALLED, FAILED = 3, 2


def _watch_loading(stall_s: float) -> threading.Timer:
    """Stall the run, with every thread's stack, if the certification's modules have not loaded in ``stall_s``."""

    def fire() -> None:
        print(f"stalled in phase loading: the certification's modules have not loaded in {stall_s:.0f}s", flush=True)
        faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
        sys.stdout.flush()
        print("certification failed: stalled in phase loading", flush=True)
        os._exit(STALLED)

    timer = threading.Timer(stall_s, fire)
    timer.daemon = True
    timer.start()
    return timer


_LOADING: threading.Timer | None = None
if __name__ == "__main__":
    # Before anything slow: a stack on demand and the stall clock run from the first line, which gives the pid to send
    # SIGUSR1 to. The modules below take a while to load, and a run that never gets past them still says so.
    faulthandler.register(signal.SIGUSR1, file=sys.stdout, all_threads=True)
    _LOADING = _watch_loading(STALL_MINUTES * 60)
    if "--shard" not in sys.argv:
        print(f"certify: loading the certification's modules, pid {os.getpid()}", flush=True)

from packages.evals.tests import stats_certificate as cert  # noqa: E402

if _LOADING is not None:
    _LOADING.cancel()

SHARD_FILE = "packages/evals/tests/test_certified_simulations.py"

#: The command that reaches the pool; a test puts a stub in its place.
TESTAMENT: tuple[str, ...] = ("testament",)

#: The command each worker of ``--local`` runs, with a shard's id after it; a test puts a stub in its place.
WORKER: tuple[str, ...] = (sys.executable, str(Path(__file__).resolve()), "--shard")

_DROPPED = ("connection_lost", "coordinator_unreachable")

#: The line testament prints on stderr once the pool has taken the request.
_ACCEPTED = re.compile(r"^request ([0-9a-f]+) accepted\b")

#: How long the one ``testament run`` may take, every shard of it: a healthy run takes ten to twenty minutes.
RUN_TIMEOUT_S = 3600.0

#: How long one ``testament metrics`` or ``testament status`` call may take.
METRICS_TIMEOUT_S = 120.0
STATUS_TIMEOUT_S = 60.0

#: How long one shard may run under ``--local``: the pool's own limit on each (``test_certified_simulations.py``).
SHARD_TIMEOUT_S = 1500.0

#: How often the command says where the pool's request stands while it waits.
PROGRESS_EVERY_S = 60.0

#: How long to wait for the pool's metrics once its request has passed, and how often to ask again meanwhile.
_METRICS_WAIT_S, _METRICS_EVERY_S = 600, 15


class CertificationFailed(Exception):
    """The run could not reach a verdict: a shard failed, the pool gave no result, or a count never arrived."""


class CommandTimeout(CertificationFailed):
    """A subprocess ran past its limit and was stopped."""

    def __init__(self, command: str, ran_s: float, limit_s: float) -> None:
        super().__init__(f"{command} ran {ran_s:.0f}s, past its limit of {limit_s:.0f}s, and was stopped")
        self.command = command
        self.ran_s = ran_s


class Stopped(Exception):
    """A signal (SIGTERM, SIGHUP) asked the run to stop."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"signal {signum}")
        self.signum = signum


class Out:
    """Where the run says what it does: every line written whole and flushed at once, so a log read mid-run, or after
    a kill, holds every line said so far."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self._lock = threading.Lock()

    def say(self, line: str) -> None:
        with self._lock:
            self.stream.write(line + "\n")
            self.stream.flush()


class Watchdog:
    """Fails the run when nothing moves for ``stall_s``: it names the phase, writes every thread's stack, runs each
    cleanup (stopping the subprocesses, removing the run's key file), and exits with :data:`STALLED`."""

    def __init__(
        self,
        stall_s: float,
        out: Out,
        *,
        stop: Callable[[int], object] = os._exit,
        every_s: float | None = None,
    ) -> None:
        self.stall_s = stall_s
        self.out = out
        self._stop = stop
        self._every_s = min(5.0, stall_s / 4) if every_s is None else every_s
        self._lock = threading.Lock()
        self._phase = "start"
        self._last = time.monotonic()
        self._cleanups: dict[int, Callable[[], None]] = {}
        self._next = 0
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._watch, name="stall-watchdog", daemon=True)
        self._waiting = ""
        self.fired = False

    def start(self) -> Watchdog:
        self._thread.start()
        return self

    def close(self) -> None:
        self._done.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join()

    @property
    def phase_name(self) -> str:
        return self._phase

    def beat(self) -> None:
        """Something moved: a shard finished, the pool's request is alive, testament said a line."""
        with self._lock:
            self._last = time.monotonic()

    def waiting_on(self, what: str) -> None:
        """Say what the run is waiting on, for the stall line if nothing moves ("" for nothing in particular)."""
        with self._lock:
            self._waiting = what

    def stall(self, why: str) -> None:
        """Stall the run now, saying ``why``: a phase that judged itself stuck (the coordinator silent, the request
        gone, batches that never finish)."""
        self._fire(self._phase, why)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Run a phase: say its start and its end (or its failure) with the time it took."""
        started = time.monotonic()
        with self._lock:
            self._phase, self._last = name, started
        self.out.say(f"{name}: started")
        try:
            yield
        except BaseException:
            self.out.say(f"{name}: failed after {time.monotonic() - started:.0f}s")
            raise
        self.out.say(f"{name}: done in {time.monotonic() - started:.0f}s")

    def on_stall(self, cleanup: Callable[[], None]) -> Callable[[], None]:
        """Run ``cleanup`` if the run stalls; returns what withdraws it."""
        with self._lock:
            handle, self._next = self._next, self._next + 1
            self._cleanups[handle] = cleanup

        def withdraw() -> None:
            with self._lock:
                self._cleanups.pop(handle, None)

        return withdraw

    def _watch(self) -> None:
        while not self._done.wait(self._every_s):
            with self._lock:
                idle, phase, waiting = time.monotonic() - self._last, self._phase, self._waiting
            if idle >= self.stall_s:
                why = f"nothing moved for {idle:.0f}s (the limit is {self.stall_s:.0f}s)"
                self._fire(phase, why + (f", waiting for {waiting}" if waiting else ""))
                return

    def _fire(self, phase: str, why: str) -> None:
        with self._lock:
            if self.fired:
                return
            self.fired = True
        self.out.say(f"stalled in phase {phase}: {why}; every thread's stack follows")
        faulthandler.dump_traceback(file=self.out.stream, all_threads=True)
        self.out.stream.flush()
        with self._lock:
            cleanups = list(self._cleanups.values())
        for cleanup in reversed(cleanups):
            cleanup()
        self.out.say(f"certification failed: stalled in phase {phase}")
        self._stop(STALLED)


def _kill(process: subprocess.Popen[str]) -> Callable[[], None]:
    def kill() -> None:
        if process.poll() is None:
            process.kill()

    return kill


@dataclass(frozen=True)
class Standing:
    """Where the pool's request stood at one ``testament status``: whether the coordinator answered (and if not, what
    testament said), whether it lists the request, and the request's batches."""

    answered: bool
    error: str = ""
    listed: bool = False
    state: str = ""
    items: int | None = None
    done: int = 0
    passed: int = 0
    failed: int = 0
    running: int = 0
    waiting: int = 0

    def said(self, request: str) -> str:
        if not self.answered:
            return f"the coordinator did not answer ({self.error})"
        if not self.listed:
            return f"request {request[:8]} is not at the coordinator"
        if self.items is None:
            return f"request {request[:8]} {self.state}"
        return (
            f"request {request[:8]} {self.state}: {self.done} of {self.items} done, {self.running} batches running, "
            f"{self.waiting} tests waiting, {self.passed} passed, {self.failed} failed"
        )


def _said_by(error: subprocess.CalledProcessError) -> str:
    """What a failed ``testament status`` said: its error's code and message, else the last of its stderr."""
    try:
        told = json.loads(error.stdout or "")
    except json.JSONDecodeError:
        told = None
    if isinstance(told, dict) and isinstance(told.get("error"), dict):
        return f"exit {error.returncode}, {told['error'].get('code')}: {told['error'].get('message')}"
    tail = " / ".join((error.stderr or "").strip().splitlines()[-3:])
    return f"exit {error.returncode}" + (f": {tail}" if tail else "")


def _status(testament: Sequence[str], request: str) -> Standing:
    """Where the pool's request stands (:class:`Standing`)."""
    command = [*testament, "status", "--json"]
    try:
        asked = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True, timeout=STATUS_TIMEOUT_S)
        told = json.loads(asked.stdout)
    except subprocess.TimeoutExpired:
        return Standing(answered=False, error=f"testament status gave no answer in {STATUS_TIMEOUT_S:.0f}s")
    except subprocess.CalledProcessError as error:
        return Standing(answered=False, error=_said_by(error))
    except json.JSONDecodeError:
        return Standing(answered=False, error="testament status said no JSON")
    mine = [entry for entry in told.get("requests") or [] if str(entry.get("run_id", "")) == request]
    if not mine:
        return Standing(answered=True)
    entry = mine[0]
    batches = [batch for batch in told.get("batches") or [] if request in (batch.get("requests") or [])]
    in_flight = sum(int(batch.get("items") or 0) - int(batch.get("done") or 0) for batch in batches)
    items, done = entry.get("items"), int(entry.get("done") or 0)
    return Standing(
        answered=True,
        listed=True,
        state=str(entry.get("state")),
        items=items,
        done=done,
        passed=int(entry.get("passed") or 0),
        failed=int(entry.get("failed") or 0),
        running=len(batches),
        waiting=max(0, int(items) - done - in_flight) if items is not None else 0,
    )


class Liveness:
    """Whether the pool's request is alive, from one :class:`Standing` after another. A request the coordinator lists
    is alive, queued or running, up to the run's limit (:data:`RUN_TIMEOUT_S`) while others' work keeps it waiting. It
    is stalled only when the coordinator stops answering for ``stall_s``, when the request is gone from it for
    ``stall_s`` with no verdict, or when it has batches running and none finishes for ``stuck_s``."""

    def __init__(self, request: str, stall_s: float, stuck_s: float) -> None:
        self.request = request
        self.stall_s = stall_s
        self.stuck_s = stuck_s
        self._silent_since: float | None = None
        self._silent_at = ""
        self._gone_since: float | None = None
        self._done = -1
        self._finished_at = time.monotonic()

    def stalled(self, standing: Standing, now: float | None = None) -> str:
        """Why the request is stalled, or "" while it is alive."""
        now = time.monotonic() if now is None else now
        if not standing.answered:
            if self._silent_since is None:
                self._silent_since, self._silent_at = now, f"{datetime.now(UTC):%H:%M:%S} UTC"
            if now - self._silent_since >= self.stall_s:
                return (
                    f"the coordinator stopped answering: testament status has failed since {self._silent_at}, "
                    f"{now - self._silent_since:.0f}s (the limit is {self.stall_s:.0f}s); last: {standing.error}"
                )
            return ""
        self._silent_since = None
        if not standing.listed:
            if self._gone_since is None:
                self._gone_since = now
            if now - self._gone_since >= self.stall_s:
                return (
                    f"request {self.request} is gone from the coordinator with no verdict, for "
                    f"{now - self._gone_since:.0f}s (the limit is {self.stall_s:.0f}s)"
                )
            return ""
        self._gone_since = None
        if standing.done != self._done or not standing.running:
            self._done, self._finished_at = standing.done, now
        if now - self._finished_at >= self.stuck_s:
            return (
                f"request {self.request} has had batches running ({standing.running} now) and none finished for "
                f"{now - self._finished_at:.0f}s (the limit is {self.stuck_s:.0f}s)"
            )
        return ""


def _run_streaming(
    command: Sequence[str],
    *,
    testament: Sequence[str],
    watchdog: Watchdog,
    out: Out,
    timeout_s: float,
    every_s: float,
    stuck_s: float,
) -> tuple[int, str, str, str]:
    """Run ``testament run``: echo each line it says on stderr as it says it, the request's id the moment it is
    accepted, and where the request stands every ``every_s``. Returns the exit code, stdout, stderr and the request's
    id ("" if none was accepted). Past ``timeout_s`` it stops the run and raises :class:`CommandTimeout`."""
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    withdraw = watchdog.on_stall(_kill(process))
    said: list[str] = []
    heard: list[str] = []
    request: list[str] = []

    def read_stdout() -> None:
        assert process.stdout is not None
        said.append(process.stdout.read())

    def read_stderr() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            heard.append(line)
            watchdog.beat()
            accepted = _ACCEPTED.match(line)
            if accepted and not request:
                request.append(accepted.group(1))
                out.say(f"pool: request {accepted.group(1)} accepted after {time.monotonic() - started:.0f}s")
            else:
                out.say(f"pool: testament says: {line.rstrip()}")

    readers = [threading.Thread(target=read, daemon=True) for read in (read_stdout, read_stderr)]
    for reader in readers:
        reader.start()
    watchdog.waiting_on("testament to accept the request")
    liveness: Liveness | None = None
    try:
        while True:
            ran = time.monotonic() - started
            if ran >= timeout_s:
                process.kill()
                process.wait()
                raise CommandTimeout("testament run", ran, timeout_s)
            try:
                process.wait(timeout=min(every_s, timeout_s - ran))
                break
            except subprocess.TimeoutExpired:
                pass
            if not request:
                out.say(f"pool: {time.monotonic() - started:.0f}s; waiting for testament to accept the request")
                continue
            if liveness is None:
                watchdog.waiting_on("")
                liveness = Liveness(request[0], watchdog.stall_s, stuck_s)
            standing = _status(testament, request[0])
            out.say(f"pool: {time.monotonic() - started:.0f}s; {standing.said(request[0])}")
            stalled = liveness.stalled(standing)
            if stalled:
                watchdog.stall(stalled)
            else:
                watchdog.beat()
    finally:
        watchdog.waiting_on("")
        if process.poll() is None:
            process.kill()
            process.wait()
        withdraw()
    for reader in readers:
        reader.join()
    return process.returncode, "".join(said), "".join(heard), request[0] if request else ""


def pool(
    key: str,
    watchdog: Watchdog,
    out: Out,
    *,
    testament: Sequence[str] = TESTAMENT,
    timeout_s: float = RUN_TIMEOUT_S,
    every_s: float = PROGRESS_EVERY_S,
    stuck_s: float = STUCK_MINUTES * 60,
    retry_after_s: float = 60.0,
) -> tuple[dict[str, dict[str, int]], str]:
    """Run every shard on the pool as one request and read each one's counts back. Returns them by shard id, and the
    request's id."""
    command = [
        *testament, "run", "--no-fallback", "--no-cache", "--suite", "simulation", "--json=summary",
        "--", SHARD_FILE, "-q",
    ]  # fmt: skip
    with watchdog.phase("pool request"):
        for attempt in range(1, 5):
            code, stdout, stderr, request = _run_streaming(
                command,
                testament=testament,
                watchdog=watchdog,
                out=out,
                timeout_s=timeout_s,
                every_s=every_s,
                stuck_s=stuck_s,
            )
            if not any(word in stdout + stderr for word in _DROPPED) or attempt == 4:
                break
            out.say(
                f"pool: the connection dropped (attempt {attempt}, exit {code}: {why(stdout, stderr)}); trying again "
                f"in {retry_after_s:.0f}s"
            )
            time.sleep(retry_after_s)
            watchdog.beat()
        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError:
            raise CertificationFailed(f"pool: no result envelope (exit {code}):\n{(stdout + stderr)[-4000:]}") from None
        report = envelope.get("report") or {}
        failed = [item for item in report.get("items", []) if item.get("outcome") != "passed"]
        if envelope.get("verdict") != "passed" or failed:
            lines = [f"  {item.get('id')}: {item.get('outcome')} {item.get('message', '')[:2000]}" for item in failed]
            raise CertificationFailed(
                f"pool: the shards did not all pass (exit {code}: {why(stdout, stderr)}):\n" + "\n".join(lines)
            )
        request = str(envelope.get("request_id", "")) or request
        out.say(f"pool: request {request} passed")
    with watchdog.phase("metrics read-back"):
        counts = counts_of(key, watchdog, out, testament=testament)
    return counts, request


def why(stdout: str, stderr: str) -> str:
    """What testament said of a run that did not pass: its envelope's verdict, reason and error, and the last of what
    it said on stderr."""
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError:
        envelope = {}
    error = envelope.get("error") if isinstance(envelope, dict) else None
    said = [f"verdict {envelope.get('verdict')}, {envelope.get('verdict_reason')}"] if envelope else ["no envelope"]
    if isinstance(error, dict):
        said.append(f"error {error.get('code')}: {str(error.get('message', ''))[:500]}")
    tail = stderr.strip().splitlines()[-3:]
    if tail:
        said.append("stderr ended " + " / ".join(tail))
    return "; ".join(said)


def metrics(name: str, *, testament: Sequence[str] = TESTAMENT, timeout_s: float = METRICS_TIMEOUT_S) -> list[Any]:
    """Every series the pool holds of the metric ``name`` from the last two days. Past ``timeout_s`` the call is
    stopped and raises :class:`CommandTimeout`; a call that returns as many series as it may refuses to be read as
    complete."""
    command = [
        *testament, "metrics", "--name", name, "--since", "2d", "--limit", str(cert.METRICS_PER_CALL), "--json",
    ]  # fmt: skip
    started = time.monotonic()
    try:
        asked = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise CommandTimeout(f"testament metrics --name {name}", time.monotonic() - started, timeout_s) from None
    except subprocess.CalledProcessError as error:
        raise CertificationFailed(
            f"testament metrics --name {name} failed (exit {error.returncode}): {error.stderr[-2000:]}"
        ) from None
    series: list[Any] = json.loads(asked.stdout)["result"]["series"]
    if len(series) >= cert.METRICS_PER_CALL:
        raise CertificationFailed(
            f"metrics: {name} returned {len(series)} series, the most one call returns, so some may be cut off; "
            f"stats_certificate.METRIC_BUCKET ({cert.METRIC_BUCKET}) must stay under it"
        )
    return series


def wanted(key: str) -> dict[str, tuple[str, list[str]]]:
    """Every metric the shards of the run keyed ``key`` record, by name: the field it counts, and the shards that
    record it."""
    by_name: dict[str, tuple[str, list[str]]] = {}
    for shard in cert.SHARDS:
        for field in cert.recorded_fields(shard):
            name = cert.metric_name(key, cert.metric_bucket(shard), field)
            by_name.setdefault(name, (field, []))[1].append(shard.id)
    return by_name


def item_id(shard_id: str) -> str:
    """The pool's id of one shard's test: what its metrics are recorded under."""
    return f"{SHARD_FILE}::test_certified_simulation_shard[{shard_id}]"


def counts_of(
    key: str,
    watchdog: Watchdog,
    out: Out,
    *,
    testament: Sequence[str] = TESTAMENT,
    wait_s: float = _METRICS_WAIT_S,
    every_s: float = _METRICS_EVERY_S,
) -> dict[str, dict[str, int]]:
    """Each shard's counts, read back from the metrics once every shard's every recorded field is there. The pool's
    request passes before its metrics are all taken in, so this asks again every ``every_s`` for the metrics still
    short, up to ``wait_s``; past that it fails naming each shard still missing a count."""
    names = wanted(key)
    counts: dict[str, dict[str, int]] = {}
    short = dict(names)
    waited_from = time.monotonic()
    while True:
        with ThreadPoolExecutor(max_workers=4) as asking:
            answers = dict(zip(short, asking.map(lambda name: metrics(name, testament=testament), short), strict=True))
        for name, series in answers.items():
            field = names[name][0]
            for each in series:
                counts.setdefault(each["item_id"], {})[field] = int(each["days"][-1]["last"])
        short = {
            name: (field, shard_ids)
            for name, (field, shard_ids) in short.items()
            if any(field not in counts.get(item_id(shard_id), {}) for shard_id in shard_ids)
        }
        watchdog.beat()
        missing = {
            shard.id: [field for field in cert.recorded_fields(shard) if field not in counts.get(item_id(shard.id), {})]
            for shard in cert.SHARDS
        }
        missing = {shard_id: fields for shard_id, fields in missing.items() if fields}
        out.say(
            f"metrics: asked {len(answers)} metrics; {len(cert.SHARDS) - len(missing)} of {len(cert.SHARDS)} shards "
            "have every count"
        )
        if not missing:
            return {shard.id: counts[item_id(shard.id)] for shard in cert.SHARDS}
        waited = time.monotonic() - waited_from
        if waited >= wait_s:
            named = "\n".join(f"  {shard_id}: {', '.join(fields)}" for shard_id, fields in sorted(missing.items()))
            raise CertificationFailed(
                f"pool: every shard passed, but after waiting {wait_s:.0f}s for the metrics {len(missing)} of "
                f"{len(cert.SHARDS)} shards still have no count under key {key}:\n{named}"
            )
        out.say(f"metrics: {len(missing)} shards' counts not yet taken in; asking again ({waited:.0f}s waited)")
        time.sleep(every_s)


def here(
    shards: Sequence[cert.Shard],
    workers: int,
    watchdog: Watchdog,
    out: Out,
    *,
    worker: Sequence[str] = WORKER,
    timeout_s: float = SHARD_TIMEOUT_S,
) -> dict[str, dict[str, int]]:
    """Run ``shards`` on this machine, ``workers`` at a time, each in a process of its own (``--shard``): saying each
    as it finishes, with its time. A shard past ``timeout_s`` is stopped and raises :class:`CommandTimeout`."""
    running: dict[int, subprocess.Popen[str]] = {}
    lock = threading.Lock()
    halted = threading.Event()

    def stop_all() -> None:
        halted.set()
        with lock:
            processes = list(running.values())
        for process in processes:
            _kill(process)()

    def one(shard: cert.Shard) -> tuple[dict[str, int], float]:
        started = time.monotonic()
        process = subprocess.Popen([*worker, shard.id], cwd=ROOT, stdout=subprocess.PIPE, text=True)
        with lock:
            running[process.pid] = process
        if halted.is_set():
            process.kill()
        try:
            said, _ = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise CommandTimeout(
                f"shard {shard.id} (pid {process.pid})", time.monotonic() - started, timeout_s
            ) from None
        finally:
            with lock:
                running.pop(process.pid, None)
        if process.returncode != 0:
            raise CertificationFailed(
                f"local: shard {shard.id} failed (exit {process.returncode}); its stderr is above"
            )
        lines = said.strip().splitlines()
        if not lines:
            raise CertificationFailed(f"local: shard {shard.id} printed no counts")
        counts: dict[str, int] = json.loads(lines[-1])
        return counts, time.monotonic() - started

    withdraw = watchdog.on_stall(stop_all)
    results: dict[str, dict[str, int]] = {}
    workers_pool = ThreadPoolExecutor(max_workers=workers)
    try:
        with watchdog.phase("local shards"):
            futures = {workers_pool.submit(one, shard): shard for shard in shards}
            for finished, future in enumerate(as_completed(futures), 1):
                shard = futures[future]
                counts, took = future.result()
                results[shard.id] = counts
                watchdog.beat()
                out.say(f"local: {shard.id} done in {took:.1f}s ({finished} of {len(shards)})")
    finally:
        workers_pool.shutdown(wait=False, cancel_futures=True)
        stop_all()
        withdraw()
    return results


def _stop_on(signum: int, _frame: FrameType | None) -> None:
    raise Stopped(signum)


def main(
    argv: list[str] | None = None,
    *,
    log: TextIO | None = None,
    testament: Sequence[str] = TESTAMENT,
    worker: Sequence[str] = WORKER,
    key_file: Path = cert.RUN_KEY,
    stop: Callable[[int], object] = os._exit,
    pool_options: Mapping[str, Any] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", action="store_true", help="run every shard here instead of on the pool")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="processes for --local")
    parser.add_argument("--out", type=Path, default=cert.CERTIFICATE, help="where to write the certificate")
    parser.add_argument(
        "--stall-minutes",
        type=float,
        default=STALL_MINUTES,
        help="stall when the coordinator stops answering, or nothing moves, for this long",
    )
    parser.add_argument("--shard", help="run one shard here and print its counts (what each --local worker runs)")
    options = parser.parse_args(argv)
    if options.shard:
        print(json.dumps(cert.run_shard(cert.shard_by_id(options.shard))), flush=True)
        return 0
    out = Out(log or sys.stdout)
    started = time.monotonic()
    where = f"here, {options.workers} at a time" if options.local else "on the testament pool, as one request"
    out.say(
        f"certify: started {datetime.now(UTC):%Y-%m-%d %H:%M:%S} UTC, pid {os.getpid()}: {len(cert.SHARDS)} shards "
        f"of {len(cert.CLAIMS)} claims, {where}; on a stall: kill -USR1 {os.getpid()} or py-spy dump --pid "
        f"{os.getpid()}"
    )
    watchdog = Watchdog(options.stall_minutes * 60, out, stop=stop).start()
    watchdog.on_stall(lambda: key_file.unlink(missing_ok=True))
    key = secrets.token_hex(6)
    about: dict[str, Any] = {}
    faulthandler.register(signal.SIGUSR1, file=out.stream, all_threads=True)
    stopping = {sig: signal.signal(sig, _stop_on) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        if key_file.exists():
            out.say(f"certify: replacing the key file an earlier run left behind ({key_file.read_text().strip()})")
        key_file.write_text(key + "\n")
        out.say(f"certify: this run's key is {key}")
        if options.local:
            results = here(cert.SHARDS, options.workers, watchdog, out, worker=worker)
            about["pool_request"] = None
        else:
            results, about["pool_request"] = pool(key, watchdog, out, testament=testament, **(pool_options or {}))
    except CertificationFailed as error:
        out.say(f"certification failed after {time.monotonic() - started:.0f}s: {error}")
        return FAILED
    except Stopped as stopped:
        out.say(
            f"certification stopped by signal {stopped.signum} in phase {watchdog.phase_name} after "
            f"{time.monotonic() - started:.0f}s"
        )
        return 128 + stopped.signum
    finally:
        watchdog.close()
        key_file.unlink(missing_ok=True)
        for sig, previous in stopping.items():
            signal.signal(sig, previous)
        faulthandler.unregister(signal.SIGUSR1)
    about["wall_s"] = round(time.monotonic() - started)
    document = cert.certificate(results, **about)
    cert.write(document, options.out)
    failing = [claim for claim in document["claims"] if not claim["holds"]]
    for claim in document["claims"]:
        mark = "holds" if claim["holds"] else "FAILS"
        low, high = claim["bounds"]
        out.say(
            f"{mark}  {claim['id']}: {claim['count']}/{claim['replicates']} "
            f"(95% {low:.4f} to {high:.4f}; {claim['test']} {claim['target']:.5g})"
        )
    out.say(
        f"certificate {document['digest'][:16]} written to {options.out} in {about['wall_s']}s: "
        f"{len(document['claims']) - len(failing)} of {len(document['claims'])} claims hold"
    )
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
