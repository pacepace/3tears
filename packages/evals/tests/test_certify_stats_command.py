"""The certification command (``packages/evals/scripts/certify_stats.py``) sends every shard to the pool, says what it
does as it goes, and never hangs silently: a stall writes every thread's stack, names its phase and exits; a subprocess
past its limit is stopped and named; a signal stops it with its last line written.

Each test runs the command against a stub of ``testament`` (a small script that plays the pool's run, status and
metrics), never the pool itself, and a stub of the ``--local`` worker.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from packages.evals.tests import stats_certificate as cert
from packages.evals.tests.script_loader import load_script
from packages.evals.tests.test_certified_simulations import PARAMS

#: Plays ``testament`` from a JSON plan: ``run`` writes the plan's stderr lines (``stderr_every`` apart), records the
#: key file it saw and its arguments, sleeps, then prints the envelope; ``status`` reports the request as the plan's
#: ``status`` says: ``advance`` (a test further each call), ``queued`` (waiting behind others, nothing running),
#: ``stuck`` (a batch running, nothing finishing), ``gone`` (not listed) or ``silent`` (the coordinator not answering);
#: ``metrics`` answers every name whose bucket and field the plan lists, after an optional sleep.
_STUB = r"""
import json, sys, time
from pathlib import Path
plan_path = Path(sys.argv[1])
plan = json.loads(plan_path.read_text())
command, args = sys.argv[2], sys.argv[3:]
if command == "run":
    Path(plan["seen"]).write_text(json.dumps({"args": args, "key": Path(plan["key_file"]).read_text().strip()}))
    for line in plan.get("stderr", []):
        print(line, file=sys.stderr, flush=True)
        time.sleep(plan.get("stderr_every", 0))
    time.sleep(plan.get("run_sleep", 0))
    print(json.dumps(plan["envelope"]), flush=True)
elif command == "status":
    mode = plan.get("status", "advance")
    if mode == "silent":
        print(json.dumps({"status": "coordinator_unreachable", "exit_code": 2,
                          "error": {"code": "coordinator_unreachable", "message": "TimeoutError"}}))
        sys.exit(2)
    ticks = Path(plan["ticks"])
    done = int(ticks.read_text()) + 1 if ticks.exists() else 1
    ticks.write_text(str(done))
    done = done if mode == "advance" else 0
    batches = [{"batch_id": "b1", "items": 1, "done": 0, "requests": [plan["request"]]}] if mode == "stuck" else []
    requests = [] if mode == "gone" else [{"run_id": plan["request"], "state": "running", "done": done,
                                           "items": plan["items"], "passed": done, "failed": 0}]
    print(json.dumps({"requests": requests, "batches": batches}))
elif command == "metrics":
    time.sleep(plan.get("metrics_sleep", 0))
    name = args[args.index("--name") + 1]
    suffix = name.split("_", 2)[2]
    items = plan["metrics"].get(suffix, [])
    value = lambda item: 1 if suffix.split("_", 1)[1] in ("n", "held") else 0
    series = [{"item_id": item, "days": [{"last": value(item)}]} for item in items]
    print(json.dumps({"ok": True, "result": {"series": series}}))
"""

_REQUEST = "6d260b0aba9648f68a0eb6789dee8ba4"


@pytest.fixture(scope="module")
def command() -> ModuleType:
    return load_script("certify_stats.py")


def _every_metric(command: ModuleType) -> dict[str, list[str]]:
    """Every (bucket, field) the shards record, as the stub's names end, with the pool test ids recording it."""
    answers: dict[str, list[str]] = {}
    for name, (_, shard_ids) in command.wanted("k").items():
        answers[name.split("_", 2)[2]] = [command.item_id(shard_id) for shard_id in shard_ids]
    return answers


def _stub(tmp_path: Path, **plan: Any) -> tuple[tuple[str, ...], Path]:
    script = tmp_path / "testament_stub.py"
    script.write_text(_STUB)
    plan_path = tmp_path / "plan.json"
    full = {
        "seen": str(tmp_path / "seen.json"),
        "ticks": str(tmp_path / "ticks.txt"),
        "key_file": str(tmp_path / "run_key.txt"),
        "request": _REQUEST,
        "items": len(cert.SHARDS),
        "stderr": [
            f"request {_REQUEST} accepted for 3tears at tree 61845b5002b1",
            f"collected {len(cert.SHARDS)} items",
        ],
        "envelope": {"verdict": "passed", "request_id": _REQUEST, "report": {"items": []}},
        "metrics": {},
        **plan,
    }
    plan_path.write_text(json.dumps(full))
    return (sys.executable, str(script), str(plan_path)), tmp_path / "run_key.txt"


#: A ``--local`` worker that must never run.
_NEVER_HERE = (sys.executable, "-c", "import sys; sys.exit('a shard ran on this machine')")

#: A ``--local`` worker that blocks.
_BLOCKS = (sys.executable, "-c", "import time; time.sleep(120)")


def _run(command: ModuleType, tmp_path: Path, argv: list[str], **kwargs: Any) -> tuple[int, str, list[int]]:
    stops: list[int] = []
    log_path = tmp_path / "log.txt"
    with log_path.open("w+") as log:
        code = command.main(["--out", str(tmp_path / "certificate.json"), *argv], log=log, stop=stops.append, **kwargs)
    return code, log_path.read_text(), stops


def test_every_shard_is_a_pool_shard() -> None:
    """The pool's test holds one parameter per shard, restated and product alike, so the one request runs them all."""
    assert [param.id for param in PARAMS] == [shard.id for shard in cert.SHARDS]
    assert {shard.through for shard in cert.SHARDS} == {"restated", "product"}


def test_the_default_run_sends_every_shard_to_the_pool_and_runs_none_here(command: ModuleType, tmp_path: Path) -> None:
    """With no ``--local`` the one request carries the simulation suite's whole file, every shard's counts come back
    from the metrics, and no worker starts here. A key file an earlier run left behind is replaced before the request
    goes out, and removed at the end."""
    testament, key_file = _stub(tmp_path, metrics=_every_metric(command), run_sleep=1.0)
    key_file.write_text("deadbeef0000\n")
    code, log, stops = _run(
        command,
        tmp_path,
        ["--stall-minutes", "1"],
        testament=testament,
        worker=_NEVER_HERE,
        key_file=key_file,
        pool_options={"every_s": 0.2},
    )
    seen = json.loads((tmp_path / "seen.json").read_text())
    assert seen["args"][:6] == ["--no-fallback", "--no-cache", "--suite", "simulation", "--json=summary", "--"]
    assert seen["args"][6] == command.SHARD_FILE
    assert seen["key"] != "deadbeef0000" and len(seen["key"]) == 12
    assert "replacing the key file an earlier run left behind (deadbeef0000)" in log
    assert not key_file.exists()
    written = json.loads((tmp_path / "certificate.json").read_text())
    assert written["pool_request"] == _REQUEST
    assert [claim["id"] for claim in written["claims"]] == [claim.id for claim in cert.CLAIMS]
    assert code in (0, 1) and not stops, log
    assert "local:" not in log


def test_the_run_says_what_it_does_as_it_goes(command: ModuleType, tmp_path: Path) -> None:
    """A start line naming the pid and how to dump its stacks; the request's id as soon as testament accepts it;
    where the request stands while it waits; the metrics read back; each phase's end; and the certificate's line."""
    testament, key_file = _stub(tmp_path, metrics=_every_metric(command), run_sleep=1.0)
    code, log, _ = _run(
        command, tmp_path, [], testament=testament, worker=_NEVER_HERE, key_file=key_file, pool_options={"every_s": 0.2}
    )
    lines = log.splitlines()
    assert lines[0].startswith("certify: started ") and f"pid {os.getpid()}" in lines[0], lines[0]
    assert f"kill -USR1 {os.getpid()}" in lines[0] and f"py-spy dump --pid {os.getpid()}" in lines[0]
    in_order = [
        "pool request: started",
        f"pool: request {_REQUEST} accepted after ",
        f"pool: testament says: collected {len(cert.SHARDS)} items",
        f"request {_REQUEST[:8]} running: ",
        f"pool: request {_REQUEST} passed",
        "pool request: done in ",
        "metrics read-back: started",
        f"metrics: asked {len(command.wanted('k'))} metrics; {len(cert.SHARDS)} of {len(cert.SHARDS)} shards have",
        "metrics read-back: done in ",
        "certificate ",
    ]
    at = 0
    for expected in in_order:
        found = next((index for index in range(at, len(lines)) if expected in lines[index]), None)
        assert found is not None, f"{expected!r} not after line {at}:\n{log}"
        at = found + 1
    assert code in (0, 1)


def test_the_watchdog_fires_on_a_pool_request_that_blocks(command: ModuleType, tmp_path: Path) -> None:
    """A ``testament run`` that never says a word and never ends: past the stall limit the watchdog names the phase,
    writes every thread's stack, stops the subprocess, removes the key file and exits with its own code."""
    testament, key_file = _stub(tmp_path, stderr=[], run_sleep=120)
    started = time.monotonic()
    code, log, stops = _run(
        command, tmp_path, ["--stall-minutes", "0.02"], testament=testament, worker=_NEVER_HERE, key_file=key_file
    )
    assert stops == [command.STALLED], log
    assert "stalled in phase pool request: nothing moved for " in log
    assert "waiting for testament to accept the request" in log
    assert "most recent call first" in log, log
    assert "certification failed: stalled in phase pool request" in log
    assert not key_file.exists()
    assert time.monotonic() - started < 60, "the stalled subprocess was not stopped"
    assert code == command.FAILED


def test_the_watchdog_covers_the_local_shards(command: ModuleType, tmp_path: Path) -> None:
    """Under ``--local``, workers that block stall the run in the local phase, named, and are stopped."""
    testament, key_file = _stub(tmp_path)
    started = time.monotonic()
    code, log, stops = _run(
        command,
        tmp_path,
        ["--local", "--workers", "2", "--stall-minutes", "0.02"],
        testament=testament,
        worker=_BLOCKS,
        key_file=key_file,
    )
    assert stops == [command.STALLED], log
    assert "stalled in phase local shards: nothing moved for " in log
    assert "most recent call first" in log
    assert not key_file.exists()
    assert time.monotonic() - started < 60, "the blocked workers were not stopped"
    assert code == command.FAILED
    assert not (tmp_path / "seen.json").exists(), "--local sent a pool request"


def test_a_testament_run_past_its_limit_is_stopped_and_named(command: ModuleType, tmp_path: Path) -> None:
    testament, key_file = _stub(tmp_path, run_sleep=120)
    code, log, stops = _run(
        command,
        tmp_path,
        [],
        testament=testament,
        worker=_NEVER_HERE,
        key_file=key_file,
        pool_options={"timeout_s": 1.0, "every_s": 0.2},
    )
    assert code == command.FAILED and not stops
    assert "certification failed after " in log
    assert "testament run ran 1s, past its limit of 1s, and was stopped" in log, log


def test_a_testament_metrics_past_its_limit_raises_naming_it(command: ModuleType, tmp_path: Path) -> None:
    testament, _ = _stub(tmp_path, metrics_sleep=30)
    started = time.monotonic()
    with pytest.raises(command.CommandTimeout, match=r"testament metrics --name simcert_k_0_n ran 1s, past its limit"):
        command.metrics("simcert_k_0_n", testament=testament, timeout_s=0.5)
    assert time.monotonic() - started < 20


def test_a_metrics_call_that_may_be_cut_off_is_refused(command: ModuleType, tmp_path: Path) -> None:
    """A call answering as many series as it may has perhaps cut some off: refused, never read as complete."""
    items = [f"item-{index}" for index in range(cert.METRICS_PER_CALL)]
    testament, _ = _stub(tmp_path, metrics={"0_n": items})
    with pytest.raises(command.CertificationFailed, match="some may be cut off"):
        command.metrics("simcert_k_0_n", testament=testament)


def test_the_read_back_covers_every_shard_s_every_count(command: ModuleType) -> None:
    """Every field each shard records is asked for by name, and no name is shared by more shards than one call
    returns."""
    asked = {(shard_id, field) for field, shard_ids in command.wanted("k").values() for shard_id in shard_ids}
    recorded = {(shard.id, field) for shard in cert.SHARDS for field in cert.recorded_fields(shard)}
    assert asked == recorded
    assert max(Counter(cert.metric_bucket(shard) for shard in cert.SHARDS).values()) < cert.METRICS_PER_CALL
    for claim in cert.CLAIMS:
        for shard in cert.SHARDS:
            if shard.group == claim.group:
                assert claim.field in cert.recorded_fields(shard), (claim.id, shard.id)


def _when_said(log_path: Path, words: str, then: Any) -> threading.Thread:
    def wait() -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if log_path.exists() and words in log_path.read_text():
                then()
                return
            time.sleep(0.05)

    thread = threading.Thread(target=wait, daemon=True)
    thread.start()
    return thread


def test_sigusr1_writes_every_thread_s_stack(command: ModuleType, tmp_path: Path) -> None:
    testament, key_file = _stub(tmp_path, metrics=_every_metric(command), run_sleep=2.0)
    _when_said(tmp_path / "log.txt", "accepted after", lambda: os.kill(os.getpid(), signal.SIGUSR1))
    _, log, _ = _run(command, tmp_path, [], testament=testament, worker=_NEVER_HERE, key_file=key_file)
    assert "most recent call first" in log, log


def test_a_terminated_run_says_so_and_leaves_no_key_file(command: ModuleType, tmp_path: Path) -> None:
    """SIGTERM (``timeout``'s signal) stops the run with its last line written, its subprocess stopped and the key
    file removed."""
    testament, key_file = _stub(tmp_path, run_sleep=120)
    _when_said(tmp_path / "log.txt", "accepted after", lambda: os.kill(os.getpid(), signal.SIGTERM))
    started = time.monotonic()
    code, log, stops = _run(command, tmp_path, [], testament=testament, worker=_NEVER_HERE, key_file=key_file)
    assert code == 128 + signal.SIGTERM and not stops
    assert log.splitlines()[-1].startswith("certification stopped by signal 15 in phase pool request after "), log
    assert not key_file.exists()
    assert time.monotonic() - started < 60


def test_a_dropped_connection_says_why_each_time(command: ModuleType, tmp_path: Path) -> None:
    """A run the pool could not give a verdict says testament's reason and error on each retry, and in its failure."""
    envelope = {
        "verdict": "unavailable",
        "verdict_reason": "coordinator_unreachable",
        "error": {"code": "coordinator_unreachable", "message": "no answer from the coordinator"},
        "report": {"items": []},
    }
    testament, key_file = _stub(tmp_path, stderr=["testament: coordinator_unreachable"], envelope=envelope)
    code, log, _ = _run(
        command,
        tmp_path,
        [],
        testament=testament,
        worker=_NEVER_HERE,
        key_file=key_file,
        pool_options={"retry_after_s": 0.0},
    )
    why = (
        "verdict unavailable, coordinator_unreachable; error coordinator_unreachable: no answer from the coordinator; "
        "stderr ended testament: coordinator_unreachable"
    )
    for attempt in (1, 2, 3):
        assert f"pool: the connection dropped (attempt {attempt}, exit 0: {why}); trying again in 0s" in log, log
    assert f"pool: the shards did not all pass (exit 0: {why})" in log
    assert code == command.FAILED


#: A stall limit of about one second, as ``--stall-minutes`` takes it.
_ONE_SECOND = ["--stall-minutes", str(1 / 60)]


def test_a_run_whose_testament_keeps_talking_never_stalls(command: ModuleType, tmp_path: Path) -> None:
    """Before the request is accepted, each line testament says is progress: lines a third of a second apart, for three
    times the stall limit, never trip it."""
    lines = [f"shipping the tree: part {index}" for index in range(10)]
    accepted = f"request {_REQUEST} accepted for 3tears at tree 61845b5002b1"
    testament, key_file = _stub(tmp_path, metrics=_every_metric(command), stderr=[*lines, accepted], stderr_every=0.3)
    code, log, stops = _run(command, tmp_path, _ONE_SECOND, testament=testament, worker=_NEVER_HERE, key_file=key_file)
    assert not stops and "stalled" not in log, log
    assert "pool: testament says: shipping the tree: part 9" in log
    assert code in (0, 1)


def test_a_run_the_pool_keeps_moving_never_stalls(command: ModuleType, tmp_path: Path) -> None:
    """After the request is accepted, a request the coordinator lists, finishing tests, is alive: three times the
    stall limit passes with no line from testament and no stall."""
    testament, key_file = _stub(tmp_path, metrics=_every_metric(command), run_sleep=3.0)
    code, log, stops = _run(
        command,
        tmp_path,
        _ONE_SECOND,
        testament=testament,
        worker=_NEVER_HERE,
        key_file=key_file,
        pool_options={"every_s": 0.2},
    )
    assert not stops and "stalled" not in log, log
    assert code in (0, 1)


def test_a_request_queued_behind_others_never_stalls(command: ModuleType, tmp_path: Path) -> None:
    """The pool busy with others' work: the request is listed, nothing of it runs or finishes, for three times the
    stall limit and past the stuck limit; it is waiting, not stalled."""
    testament, key_file = _stub(tmp_path, metrics=_every_metric(command), run_sleep=3.0, status="queued")
    code, log, stops = _run(
        command,
        tmp_path,
        _ONE_SECOND,
        testament=testament,
        worker=_NEVER_HERE,
        key_file=key_file,
        pool_options={"every_s": 0.2, "stuck_s": 1.0},
    )
    assert not stops and "stalled" not in log, log
    assert f"request {_REQUEST[:8]} running: 0 of {len(cert.SHARDS)} done, 0 batches running" in log
    assert code in (0, 1)


def test_a_silent_coordinator_stalls_saying_it_stopped_answering(command: ModuleType, tmp_path: Path) -> None:
    """Every ``testament status`` fails: past the stall limit the run stalls, saying the coordinator stopped answering,
    since when, for how long, and testament's own error."""
    testament, key_file = _stub(tmp_path, run_sleep=120, status="silent")
    code, log, stops = _run(
        command,
        tmp_path,
        _ONE_SECOND,
        testament=testament,
        worker=_NEVER_HERE,
        key_file=key_file,
        pool_options={"every_s": 0.2},
    )
    assert stops == [command.STALLED], log
    assert "stalled in phase pool request: the coordinator stopped answering: testament status has failed since " in log
    assert "last: exit 2, coordinator_unreachable: TimeoutError" in log
    assert "most recent call first" in log
    assert code == command.FAILED


def test_a_request_gone_from_the_coordinator_stalls(command: ModuleType, tmp_path: Path) -> None:
    testament, key_file = _stub(tmp_path, run_sleep=120, status="gone")
    code, log, stops = _run(
        command,
        tmp_path,
        _ONE_SECOND,
        testament=testament,
        worker=_NEVER_HERE,
        key_file=key_file,
        pool_options={"every_s": 0.2},
    )
    assert stops == [command.STALLED], log
    assert f"stalled in phase pool request: request {_REQUEST} is gone from the coordinator with no verdict" in log
    assert code == command.FAILED


def test_batches_that_run_and_never_finish_stall_past_the_stuck_limit(command: ModuleType, tmp_path: Path) -> None:
    testament, key_file = _stub(tmp_path, run_sleep=120, status="stuck")
    code, log, stops = _run(
        command,
        tmp_path,
        ["--stall-minutes", "1"],
        testament=testament,
        worker=_NEVER_HERE,
        key_file=key_file,
        pool_options={"every_s": 0.2, "stuck_s": 1.0},
    )
    assert stops == [command.STALLED], log
    assert f"request {_REQUEST} has had batches running (1 now) and none finished for " in log
    assert code == command.FAILED


def test_liveness_at_the_real_limits(command: ModuleType) -> None:
    """At the command's own limits: a request queued for eleven minutes is alive; one with batches running is alive
    while tests finish, and stuck once none has finished for thirty minutes; a silent coordinator stalls at ten."""
    stall_s, stuck_s = command.STALL_MINUTES * 60, command.STUCK_MINUTES * 60
    queued = command.Standing(answered=True, listed=True, state="running", items=1217, done=0, running=0, waiting=1217)
    alive = command.Liveness(_REQUEST, stall_s, stuck_s)
    assert all(alive.stalled(queued, now=float(minute * 60)) == "" for minute in range(12))

    running = command.Liveness(_REQUEST, stall_s, stuck_s)
    for minute in range(0, 60, 5):
        moving = command.Standing(answered=True, listed=True, state="running", items=1217, done=minute, running=40)
        assert running.stalled(moving, now=float(minute * 60)) == ""
    still = command.Standing(answered=True, listed=True, state="running", items=1217, done=55, running=40)
    assert running.stalled(still, now=float(84 * 60)) == ""
    assert "none finished for 1800s" in running.stalled(still, now=float(85 * 60))

    silent = command.Liveness(_REQUEST, stall_s, stuck_s)
    gone = command.Standing(answered=False, error="exit 2, coordinator_unreachable: TimeoutError")
    assert silent.stalled(gone, now=0.0) == "" and silent.stalled(gone, now=599.0) == ""
    assert "the coordinator stopped answering" in silent.stalled(gone, now=600.0)


def test_sigusr1_at_once_dumps_the_stacks_and_the_run_goes_on(command: ModuleType) -> None:
    """The first line gives the pid; a SIGUSR1 sent the moment it appears, while the modules still load, writes every
    thread's stack and the command carries on (here to its help, and exit 0)."""
    process = subprocess.Popen(
        [sys.executable, str(command.__file__), "--help"],
        cwd=cert.ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    first = process.stdout.readline()
    assert first.startswith(f"certify: loading the certification's modules, pid {process.pid}"), first
    process.send_signal(signal.SIGUSR1)
    rest, _ = process.communicate(timeout=300)
    assert process.returncode == 0, rest
    assert "most recent call first" in rest and "usage:" in rest


def test_a_shard_still_missing_a_count_is_refused_by_name(command: ModuleType, tmp_path: Path) -> None:
    """The read-back asks again for a count still missing, and past its wait refuses the run naming the shard and the
    field, never reading it as 0."""
    answers = _every_metric(command)
    withheld = cert.SHARDS[0]
    suffix = f"{cert.metric_bucket(withheld)}_hits"
    answers[suffix] = [item for item in answers[suffix] if item != command.item_id(withheld.id)]
    testament, _ = _stub(tmp_path, metrics=answers)
    with (tmp_path / "log.txt").open("w+") as log:
        out = command.Out(log)
        watchdog = command.Watchdog(60.0, out, stop=lambda code: None)
        with pytest.raises(command.CertificationFailed, match=rf"1 of {len(cert.SHARDS)} shards") as refused:
            command.counts_of("k", watchdog, out, testament=testament, wait_s=0.3, every_s=0.1)
    assert f"  {withheld.id}: hits" in str(refused.value)


def test_the_tally_refuses_a_missing_count_and_reads_a_present_one_as_it_is() -> None:
    results = {
        shard.id: {field: index % 7 + 1 for index, field in enumerate(cert.recorded_fields(shard))}
        for shard in cert.SHARDS
    }
    expected = {
        claim.id: (
            sum(results[shard.id][claim.field] for shard in cert.SHARDS if shard.group == claim.group),
            sum(results[shard.id]["n"] for shard in cert.SHARDS if shard.group == claim.group),
        )
        for claim in cert.CLAIMS
    }
    assert cert.tally(results) == expected
    claim = cert.CLAIMS[0]
    shard = next(shard for shard in cert.SHARDS if shard.group == claim.group)
    del results[shard.id][claim.field]
    with pytest.raises(KeyError, match=rf"claim {claim.id} has no count for {shard.id} \({claim.field}\)"):
        cert.tally(results)
