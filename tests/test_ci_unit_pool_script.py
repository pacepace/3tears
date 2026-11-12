"""behavioural tests for ``scripts/ci-unit-pool.sh``: ci.yml's unit route and its pool run.

The script decides two things CI's required ``check`` rests on, so both are pinned here by running
it, not by reading it:

- ``route``: the testament pool, or the GitHub shards and why. The shards run when the repo
  variable ``TESTAMENT_CI`` is off, no key reached the run, the event is not a push or a pull
  request from a branch of this repo, the client cannot be had, or ``testament run --probe`` says
  no run is possible. The key is never sent to a console that is not https.
- ``run``: the pool's outcome, ``passed``, ``failed`` or ``unreachable``, from testament's exit code
  and envelope. ``unreachable`` (the pool down or lost, no test failed) is what makes ci.yml run the
  shards; a genuine test failure never does. A retryable ending runs once more.

Self-contained: the script is staged in a miniature repo under ``tmp_path`` (it derives REPO_ROOT
from its own location), and ``curl``, ``uv`` and ``testament`` are stand-in executables that serve
canned answers and record how they were called, so nothing here downloads, installs or reaches a
pool. ``testament`` exists only in the directory ``uv tool dir --bin`` names, so the script can
reach it only after installing the client.

    ./scripts/test.sh tests/test_ci_unit_pool_script.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "ci-unit-pool.sh"

_KEY = "tk_ci_secret_value"
_WHEEL = b"PK\x03\x04 a stand-in wheel"

#: serves the canned download: the headers and body files the test wrote, and the HTTP status
_CURL = """#!/bin/sh
cat > "$STAGE/curl.stdin"
printf '%s\\n' "$@" > "$STAGE/curl.argv"
headers=""
body=""
while [ $# -gt 0 ]; do
    case "$1" in
        -D) headers="$2"; shift ;;
        -o) body="$2"; shift ;;
    esac
    shift
done
if [ -f "$STAGE/curl.fail" ]; then
    echo "curl: (7) Failed to connect" >&2
    printf '000'
    exit 7
fi
cp "$STAGE/wheel.headers" "$headers"
cp "$STAGE/wheel.body" "$body"
cat "$STAGE/curl.status"
"""

#: `uv tool install` succeeds unless told not to; `uv tool dir --bin` names where testament is
_UV = """#!/bin/sh
printf '%s\\n' "$*" >> "$STAGE/uv.calls"
case "$1 $2" in
    "tool install") [ -f "$STAGE/uv.fail" ] && exit 1; exit 0 ;;
    "tool dir") echo "$STAGE/tool-bin" ;;
esac
"""

#: a probe answers from probe.json/probe.code, or never with probe.hang; the Nth run from runN.json/runN.code
_TESTAMENT = """#!/bin/sh
printf '%s\\n' "$*" >> "$STAGE/testament.calls"
if [ "$2" = "--probe" ]; then
    [ -f "$STAGE/probe.hang" ] && exec sleep 30
    cat "$STAGE/probe.json"
    exit "$(cat "$STAGE/probe.code")"
fi
n=$(( $(cat "$STAGE/runs" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$STAGE/runs"
cat "$STAGE/run$n.json" 2>/dev/null
exit "$(cat "$STAGE/run$n.code")"
"""

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("sha256sum") is None,
    reason="ci-unit-pool.sh is bash and checks the download with sha256sum; this host lacks one",
)


@dataclass(frozen=True)
class _Run:
    """what one invocation of the staged script did.

    :param returncode: the script's exit status
    :ptype returncode: int
    :param stdout: what it printed to stdout (annotations among it)
    :ptype stdout: str
    :param outputs: the ``name=value`` lines it wrote to ``$GITHUB_OUTPUT``
    :ptype outputs: dict[str, str]
    :param summary: what it wrote to ``$GITHUB_STEP_SUMMARY``
    :ptype summary: str
    :param testament: each ``testament`` call's arguments, one string a call
    :ptype testament: list[str]
    :param curl_called: whether the download was attempted
    :ptype curl_called: bool
    :param installed: whether ``uv tool install`` was called
    :ptype installed: bool
    """

    returncode: int
    stdout: str
    outputs: dict[str, str]
    summary: str
    testament: list[str]
    curl_called: bool
    installed: bool


class _Stage:
    """a miniature repo with the script, its stand-ins, and the answers they serve.

    :param root: the directory everything lives under
    :ptype root: Path
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.repo = root / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        shutil.copy(_SCRIPT, self.repo / "scripts" / "ci-unit-pool.sh")
        self.stage = root / "stage"
        (self.stage / "tool-bin").mkdir(parents=True)
        self.bin = root / "bin"
        self.bin.mkdir()
        for directory, name, body in (
            (self.bin, "curl", _CURL),
            (self.bin, "uv", _UV),
            (self.stage / "tool-bin", "testament", _TESTAMENT),
        ):
            (directory / name).write_text(body)
            (directory / name).chmod(0o755)
        # the script's python3 is whatever runs these tests, wherever its host keeps it
        (self.bin / "python3").symlink_to(sys.executable)
        self.serve_wheel()
        self.probe(0, {"ok": True, "probe": {"ready": True}})

    def serve_wheel(
        self,
        *,
        status: str = "200",
        name: str = "testament-0.44.1-py3-none-any.whl",
        sha256: str | None = None,
    ) -> None:
        """What the console's download answers.

        :param status: the HTTP status curl reports
        :ptype status: str
        :param name: the file name the Content-Disposition header names
        :ptype name: str
        :param sha256: the X-Testament-Wheel-SHA256 header; the body's real digest when None,
            and no header at all when empty
        :ptype sha256: str | None
        :return: nothing
        :rtype: None
        """
        digest = hashlib.sha256(_WHEEL).hexdigest() if sha256 is None else sha256
        headers = [f"HTTP/2 {status}", f'content-disposition: attachment; filename="{name}"']
        if digest:
            headers.append(f"X-Testament-Wheel-SHA256: {digest}")
        (self.stage / "wheel.headers").write_text("\r\n".join(headers) + "\r\n\r\n")
        (self.stage / "wheel.body").write_bytes(_WHEEL)
        (self.stage / "curl.status").write_text(status)

    def unreachable_console(self) -> None:
        """the download cannot connect at all.

        :return: nothing
        :rtype: None
        """
        (self.stage / "curl.fail").write_text("")

    def failing_install(self) -> None:
        """``uv tool install`` fails.

        :return: nothing
        :rtype: None
        """
        (self.stage / "uv.fail").write_text("")

    def probe(self, code: int, envelope: dict[str, object]) -> None:
        """What ``testament run --probe --json`` answers.

        :param code: its exit code
        :ptype code: int
        :param envelope: the JSON it prints
        :ptype envelope: dict[str, object]
        :return: nothing
        :rtype: None
        """
        (self.stage / "probe.json").write_text(json.dumps(envelope))
        (self.stage / "probe.code").write_text(str(code))

    def hanging_probe(self) -> None:
        """``testament run --probe`` takes the connection and never answers.

        :return: nothing
        :rtype: None
        """
        (self.stage / "probe.hang").write_text("")

    def runs(self, *answers: tuple[int, dict[str, object] | None]) -> None:
        """What each ``testament run`` answers, in order.

        :param answers: (exit code, envelope or None for no output) per run
        :ptype answers: tuple[int, dict[str, object] | None]
        :return: nothing
        :rtype: None
        """
        for n, (code, envelope) in enumerate(answers, start=1):
            if envelope is not None:
                (self.stage / f"run{n}.json").write_text(json.dumps(envelope))
            (self.stage / f"run{n}.code").write_text(str(code))

    def go(self, mode: str, **env: str) -> _Run:
        """Run the staged script.

        :param mode: ``route`` or ``run``
        :ptype mode: str
        :param env: the environment beyond PATH and the GitHub files; overrides the defaults
        :ptype env: str
        :return: what it did
        :rtype: _Run
        """
        output = self.root / "github_output"
        summary = self.root / "step_summary"
        output.write_text("")
        summary.write_text("")
        environment = {
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "STAGE": str(self.stage),
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(summary),
            "TESTAMENT_KEY": _KEY,
            "TESTAMENT_COORDINATOR": "coord.example:7621",
            "TESTAMENT_CONSOLE": "https://console.example/",
            "EVENT_NAME": "pull_request",
            "HEAD_REPO": "owner/3tears",
            "REPO": "owner/3tears",
            "OUT": str(self.root / "out"),
            **env,
        }
        result = subprocess.run(
            ["bash", str(self.repo / "scripts" / "ci-unit-pool.sh"), mode],
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        outputs = dict(line.split("=", 1) for line in output.read_text().splitlines() if "=" in line)
        calls = self.stage / "testament.calls"
        return _Run(
            returncode=result.returncode,
            stdout=result.stdout + result.stderr,
            outputs=outputs,
            summary=summary.read_text(),
            testament=calls.read_text().splitlines() if calls.exists() else [],
            curl_called=(self.stage / "curl.argv").exists(),
            installed=(self.stage / "uv.calls").exists() and "tool install" in (self.stage / "uv.calls").read_text(),
        )


@pytest.fixture
def stage(tmp_path: Path) -> _Stage:
    """The staged script with stand-ins that serve a good wheel and a ready pool.

    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :return: the stage
    :rtype: _Stage
    """
    return _Stage(tmp_path)


def _envelope(
    code: int, *, error: str | None = None, retryable: bool = False, counts: dict[str, int] | None = None
) -> dict[str, object]:
    """A testament ``--json=summary`` envelope.

    :param code: its exit_code
    :ptype code: int
    :param error: its error.code, when it ended in one
    :ptype error: str | None
    :param retryable: its error.retryable
    :ptype retryable: bool
    :param counts: its report.counts
    :ptype counts: dict[str, int] | None
    :return: the envelope
    :rtype: dict[str, object]
    """
    envelope: dict[str, object] = {"status": error or "done", "exit_code": code, "request_id": "r1"}
    if counts is not None:
        envelope["report"] = {"counts": counts, "items": []}
    if error is not None:
        envelope["error"] = {"code": error, "message": f"{error} happened", "retryable": retryable, "ran": 0}
    return envelope


_PASSED = (0, _envelope(0, counts={"passed": 40}))
_FAILED = (1, _envelope(1, counts={"passed": 38, "failed": 2}))
_LOST = (2, _envelope(2, error="connection_lost", retryable=True))
_UNREACHABLE = (2, _envelope(2, error="coordinator_unreachable", retryable=True))
_BATCHES_LOST = (2, _envelope(2, error="batches_lost", retryable=True, counts={"passed": 30, "missing": 10}))
_NO_RUNNER = (2, _envelope(2, error="no_runner"))
_LOST_AFTER_A_FAILURE = (2, _envelope(2, error="connection_lost", retryable=True, counts={"passed": 3, "failed": 1}))
_FORBIDDEN = (2, _envelope(2, error="forbidden"))
_UPGRADE = (2, _envelope(2, error="upgrade_required"))
_NOTHING_COLLECTED = (5, _envelope(5, counts={}))


def _unavailable(error: str, *, retryable: bool, counts: dict[str, int] | None = None) -> tuple[int, dict[str, object]]:
    """testament 0.46's ending for a run the pool could not serve: exit 75 and verdict ``unavailable``.

    :param error: its error.code
    :ptype error: str
    :param retryable: its error.retryable
    :ptype retryable: bool
    :param counts: its report.counts
    :ptype counts: dict[str, int] | None
    :return: the exit status and the envelope
    :rtype: tuple[int, dict[str, object]]
    """
    envelope = _envelope(75, error=error, retryable=retryable, counts=counts)
    envelope["verdict"] = "unavailable"
    return 75, envelope


_LOST_75 = _unavailable("connection_lost", retryable=True)
_STORE_FULL_75 = _unavailable("store_full", retryable=False)
_LOST_75_AFTER_A_FAILURE = _unavailable("connection_lost", retryable=True, counts={"passed": 3, "failed": 1})
_NO_ENVELOPE = (1, None)


class TestTheRunsOutcome:
    """``run`` tells a test failure from a pool that is down or was lost, and only the latter falls back."""

    @pytest.mark.parametrize(
        ("answers", "outcome", "exit_code", "runs"),
        [
            pytest.param([_PASSED], "passed", 0, 1, id="passed"),
            pytest.param([_FAILED], "failed", 1, 1, id="a-test-failed"),
            pytest.param([_LOST, _PASSED], "passed", 0, 2, id="lost-then-passed-on-the-retry"),
            pytest.param([_LOST, _FAILED], "failed", 1, 2, id="lost-then-failed-on-the-retry"),
            pytest.param([_LOST, _LOST], "unreachable", 0, 2, id="connection-lost-twice"),
            pytest.param([_UNREACHABLE, _UNREACHABLE], "unreachable", 0, 2, id="coordinator-unreachable-twice"),
            pytest.param([_UNREACHABLE, _LOST], "unreachable", 0, 2, id="unreachable-then-lost"),
            pytest.param([_BATCHES_LOST, _BATCHES_LOST], "unreachable", 0, 2, id="batches-lost-twice"),
            pytest.param([_NO_RUNNER], "unreachable", 0, 1, id="no-runner-is-not-retried"),
            pytest.param([_LOST_AFTER_A_FAILURE], "failed", 1, 1, id="lost-after-a-test-failed"),
            pytest.param([_FORBIDDEN], "failed", 1, 1, id="forbidden"),
            pytest.param([_UPGRADE], "failed", 1, 1, id="upgrade-required"),
            pytest.param([_NOTHING_COLLECTED], "failed", 1, 1, id="nothing-collected"),
            pytest.param([_NO_ENVELOPE], "failed", 1, 1, id="no-envelope"),
            pytest.param([_LOST_75, _PASSED], "passed", 0, 2, id="exit-75-lost-then-passed-on-the-retry"),
            pytest.param([_LOST_75, _LOST_75], "unreachable", 0, 2, id="exit-75-lost-twice"),
            pytest.param([_STORE_FULL_75], "unreachable", 0, 1, id="exit-75-store-full-is-not-retried"),
            pytest.param([_LOST_75_AFTER_A_FAILURE], "failed", 1, 1, id="exit-75-after-a-test-failed"),
        ],
    )
    def test_the_outcome(
        self,
        stage: _Stage,
        answers: list[tuple[int, dict[str, object] | None]],
        outcome: str,
        exit_code: int,
        runs: int,
    ) -> None:
        """each ending's outcome, exit status and number of attempts.

        :param stage: the staged script
        :ptype stage: _Stage
        :param answers: what each run answers
        :ptype answers: list[tuple[int, dict[str, object] | None]]
        :param outcome: the outcome it must report
        :ptype outcome: str
        :param exit_code: the status it must exit with
        :ptype exit_code: int
        :param runs: how many times it must run the suite
        :ptype runs: int
        :return: nothing
        :rtype: None
        """
        stage.runs(*answers)
        run = stage.go("run")
        assert run.outputs.get("outcome") == outcome, run.stdout
        assert run.returncode == exit_code, run.stdout
        assert len(run.testament) == runs, run.testament

    def test_the_run_is_fresh_uncached_and_never_falls_back(self, stage: _Stage) -> None:
        """the required check gets a cold container, every test run, and no local fallback.

        :param stage: the staged script
        :ptype stage: _Stage
        :return: nothing
        :rtype: None
        """
        stage.runs(_PASSED)
        run = stage.go("run")
        args = run.testament[0].split()
        assert args[0] == "run"
        assert {"--fresh", "--no-cache", "--no-fallback", "--json=summary"} <= set(args)

    def test_the_run_leaves_the_loop_timing_tests_to_github(self, stage: _Stage) -> None:
        """the pool run narrows the manifest's marker expression by ``not loop_timing`` and names no path.

        testament ANDs a selection's ``-m`` with the suite's own (``not integration``), so the pool runs
        every unit test but the wall-clock ones, which ci.yml's ``unit-pool`` job then runs serially.
        No path: the manifest's ``default_selection`` (``packages/`` and ``tests/``) stays the selection.

        :param stage: the staged script
        :ptype stage: _Stage
        :return: nothing
        :rtype: None
        """
        stage.runs(_PASSED)
        run = stage.go("run")
        selection = run.testament[0].partition(" -- ")[2]
        assert selection == "-m not loop_timing", run.testament[0]

    def test_a_lost_pool_says_the_shards_run(self, stage: _Stage) -> None:
        """unreachable is never silent: an annotation and the summary say the shards run instead.

        :param stage: the staged script
        :ptype stage: _Stage
        :return: nothing
        :rtype: None
        """
        stage.runs(_LOST, _LOST)
        run = stage.go("run")
        assert "::warning title=unit route::pool unreachable during the run" in run.stdout
        assert "running the GitHub shards" in run.stdout
        assert "GitHub shards run instead" in run.summary
        assert "connection_lost" in run.outputs["why"]

    def test_a_retry_is_announced(self, stage: _Stage) -> None:
        """the second attempt says why it runs.

        :param stage: the staged script
        :ptype stage: _Stage
        :return: nothing
        :rtype: None
        """
        stage.runs(_LOST, _PASSED)
        run = stage.go("run")
        assert "ended retryable" in run.stdout and "running it once more" in run.stdout

    def test_a_test_failure_is_named(self, stage: _Stage) -> None:
        """a failure is an error annotation and in the summary, not a fallback.

        :param stage: the staged script
        :ptype stage: _Stage
        :return: nothing
        :rtype: None
        """
        stage.runs(_FAILED)
        run = stage.go("run")
        assert "::error title=unit (testament pool)::2 test(s) failed or errored" in run.stdout
        assert "running the GitHub shards" not in run.stdout
        assert "Failed on the pool" in run.summary


class TestTheClientDownload:
    """the client is installed only from an https console, only when the download checks out."""

    def test_a_good_download_is_installed_and_the_key_stays_out_of_argv(self, stage: _Stage) -> None:
        """the key reaches curl on stdin, never in its arguments.

        :param stage: the staged script
        :ptype stage: _Stage
        :return: nothing
        :rtype: None
        """
        stage.runs(_PASSED)
        run = stage.go("run")
        assert run.installed and run.outputs["outcome"] == "passed"
        assert _KEY not in (stage.stage / "curl.argv").read_text()
        assert f"Bearer {_KEY}" in (stage.stage / "curl.stdin").read_text()
        assert "https://console.example/api/v1/cli/wheel" in (stage.stage / "curl.argv").read_text()

    @pytest.mark.parametrize(
        ("console", "mode"),
        [
            pytest.param("http://console.example/", "run", id="http-run"),
            pytest.param("http://console.example/", "route", id="http-route"),
            pytest.param("console.example", "route", id="no-scheme"),
        ],
    )
    def test_a_console_that_is_not_https_never_gets_the_key(self, stage: _Stage, console: str, mode: str) -> None:
        """refused before curl runs, so the key never leaves.

        :param stage: the staged script
        :ptype stage: _Stage
        :param console: the TESTAMENT_CONSOLE value
        :ptype console: str
        :param mode: route or run
        :ptype mode: str
        :return: nothing
        :rtype: None
        """
        stage.runs(_PASSED)
        run = stage.go(mode, TESTAMENT_CONSOLE=console)
        assert not run.curl_called
        assert not run.testament
        assert "is not an https URL" in run.outputs["why"]

    @pytest.mark.parametrize(
        ("serve", "said"),
        [
            pytest.param({"status": "403"}, "HTTP 403", id="refused-403"),
            pytest.param({"status": "502"}, "HTTP 502", id="console-down-502"),
            pytest.param({"name": "evil.sh"}, "named no testament wheel", id="not-a-testament-wheel"),
            pytest.param({"name": "testament-0.44.1.tar.gz"}, "named no testament wheel", id="not-a-wheel"),
            pytest.param({"sha256": "0" * 64}, "sha256 is", id="digest-mismatch"),
            pytest.param({"sha256": ""}, "sha256 is", id="no-digest-header"),
        ],
    )
    def test_a_bad_download_is_never_installed(self, stage: _Stage, serve: dict[str, str], said: str) -> None:
        """a non-200, a file that is not a testament wheel, or one whose digest differs is refused.

        On ``run`` that is the outcome ``unreachable``: the pool cannot be used, so the shards decide.

        :param stage: the staged script
        :ptype stage: _Stage
        :param serve: how the download misbehaves
        :ptype serve: dict[str, str]
        :param said: what the reason must say
        :ptype said: str
        :return: nothing
        :rtype: None
        """
        stage.serve_wheel(**serve)
        stage.runs(_PASSED)
        run = stage.go("run")
        assert not run.installed
        assert not run.testament
        assert run.outputs["outcome"] == "unreachable"
        assert said in run.outputs["why"]

    def test_an_unreachable_console_is_unreachable(self, stage: _Stage) -> None:
        """curl failing to connect is an unreachable pool.

        :param stage: the staged script
        :ptype stage: _Stage
        :return: nothing
        :rtype: None
        """
        stage.unreachable_console()
        run = stage.go("run")
        assert run.outputs["outcome"] == "unreachable"
        assert "HTTP unreachable" in run.outputs["why"]

    def test_a_failed_install_runs_nothing(self, stage: _Stage) -> None:
        """uv failing to install the wheel leaves no testament to run.

        :param stage: the staged script
        :ptype stage: _Stage
        :return: nothing
        :rtype: None
        """
        stage.failing_install()
        run = stage.go("run")
        assert not run.testament
        assert "uv could not install" in run.outputs["why"]


class TestTheRoute:
    """``route`` picks the pool only when it may and the pool answers its probe."""

    def test_the_pool_when_it_answers_the_probe(self, stage: _Stage) -> None:
        """a same-repo pull request with a key and a ready pool goes to the pool.

        :param stage: the staged script
        :ptype stage: _Stage
        :return: nothing
        :rtype: None
        """
        run = stage.go("route")
        assert run.returncode == 0, run.stdout
        assert run.outputs == {"pool": "true", "why": "the pool answered the probe"}
        assert run.testament == ["run --probe --json"]
        assert "unit route: the testament pool" in run.summary

    def test_a_push_goes_to_the_pool(self, stage: _Stage) -> None:
        """a push to this repo is not a pull request and needs no head repo.

        :param stage: the staged script
        :ptype stage: _Stage
        :return: nothing
        :rtype: None
        """
        run = stage.go("route", EVENT_NAME="push", HEAD_REPO="")
        assert run.outputs["pool"] == "true"

    @pytest.mark.parametrize(
        ("env", "said"),
        [
            pytest.param({"TESTAMENT_CI": "off"}, "TESTAMENT_CI is off", id="variable-off"),
            pytest.param({"TESTAMENT_CI": "OFF"}, "TESTAMENT_CI is off", id="variable-OFF"),
            pytest.param({"TESTAMENT_KEY": ""}, "no TESTAMENT_KEY", id="no-key"),
            pytest.param({"HEAD_REPO": "someone/3tears"}, "not a push or a pull request", id="fork-pr"),
            pytest.param({"HEAD_REPO": ""}, "not a push or a pull request", id="pr-without-head-repo"),
            pytest.param({"EVENT_NAME": "workflow_dispatch"}, "not a push or a pull request", id="other-event"),
            pytest.param({"TESTAMENT_COORDINATOR": ""}, "no TESTAMENT_COORDINATOR", id="no-coordinator"),
        ],
    )
    def test_the_shards_without_asking_the_pool(self, stage: _Stage, env: dict[str, str], said: str) -> None:
        """each reason not to use the pool routes to the shards before any download or probe.

        :param stage: the staged script
        :ptype stage: _Stage
        :param env: what differs from a same-repo pull request with a key
        :ptype env: dict[str, str]
        :param said: what the reason must say
        :ptype said: str
        :return: nothing
        :rtype: None
        """
        run = stage.go("route", **env)
        assert run.returncode == 0, run.stdout
        assert run.outputs["pool"] == "false"
        assert said in run.outputs["why"]
        assert not run.curl_called and not run.testament
        assert "running the GitHub shards" in run.stdout
        assert "unit route: the GitHub shards" in run.summary

    @pytest.mark.parametrize(
        ("code", "error"),
        [
            pytest.param(2, "coordinator_unreachable", id="coordinator-unreachable"),
            pytest.param(2, "no_runner", id="no-runner"),
            pytest.param(2, "forbidden", id="forbidden"),
            pytest.param(1, None, id="no-envelope"),
        ],
    )
    def test_a_failed_probe_routes_to_the_shards(self, stage: _Stage, code: int, error: str | None) -> None:
        """the pool down at the probe is the shards, with a warning naming testament's answer.

        :param stage: the staged script
        :ptype stage: _Stage
        :param code: the probe's exit code
        :ptype code: int
        :param error: its error.code, or None for no envelope at all
        :ptype error: str | None
        :return: nothing
        :rtype: None
        """
        stage.probe(code, _envelope(code, error=error) if error else {})
        run = stage.go("route")
        assert run.returncode == 0, run.stdout
        assert run.outputs["pool"] == "false"
        assert run.outputs["why"].startswith(f"pool unavailable at probe: {error or 'no answer'}:")
        assert "::warning title=unit route::pool unavailable at probe" in run.stdout
        assert "running the GitHub shards" in run.stdout

    def test_a_probe_that_never_answers_routes_to_the_shards_in_time(self, stage: _Stage) -> None:
        """a pool that accepts the connection and never answers is stopped at the bound: the shards, with a warning.

        Unbounded, the probe would hold ``route`` to its job timeout, and ``check`` would fail rather than
        fall back.

        :param stage: the staged script
        :ptype stage: _Stage
        :return: nothing
        :rtype: None
        """
        stage.hanging_probe()
        began = time.monotonic()
        run = stage.go("route", TESTAMENT_PROBE_TIMEOUT="1")
        assert time.monotonic() - began < 20, "the probe was not stopped at its bound"
        assert run.returncode == 0, run.stdout
        assert run.outputs["pool"] == "false"
        assert run.outputs["why"] == "pool unavailable at probe: no answer within 1s (exit 124)"
        assert "::warning title=unit route::pool unavailable at probe: no answer within 1s" in run.stdout
        assert "unit route: the GitHub shards" in run.summary
        assert "no answer within 1s" in run.summary

    def test_a_console_that_is_down_routes_to_the_shards(self, stage: _Stage) -> None:
        """no client, no probe: the shards.

        :param stage: the staged script
        :ptype stage: _Stage
        :return: nothing
        :rtype: None
        """
        stage.unreachable_console()
        run = stage.go("route")
        assert run.outputs["pool"] == "false"
        assert run.outputs["why"].startswith("pool unavailable at probe: the testament console did not hand over")
        assert not run.testament

    def test_a_digest_mismatch_routes_to_the_shards_as_an_error(self, stage: _Stage) -> None:
        """an integrity refusal is never installed, and is an error annotation, not a quiet warning.

        :param stage: the staged script
        :ptype stage: _Stage
        :return: nothing
        :rtype: None
        """
        stage.serve_wheel(sha256="f" * 64)
        run = stage.go("route")
        assert run.outputs["pool"] == "false"
        assert not run.installed
        assert "::error title=unit route::pool unavailable at probe" in run.stdout


def test_an_unknown_mode_is_refused(stage: _Stage) -> None:
    """the script takes ``route`` or ``run`` and nothing else.

    :param stage: the staged script
    :ptype stage: _Stage
    :return: nothing
    :rtype: None
    """
    run = stage.go("probe")
    assert run.returncode == 2
    assert "usage" in run.stdout
