"""The on-ramp's trial container, held by canaries: what an agent's commands run in cannot reach a key, a file of this
machine, the repository or the network.

These drive a Docker daemon, so they run on this machine, never on the testament pool:
``./scripts/test-integration.sh --here -k onramp -rs``. The image is built once (set-up needs the network; a trial has
none) from the control arm's wheel, as the live baseline builds it.

Each canary runs through the real path with the scripted agent: a transcript writes a probe, runs it as the agent's own
command, adds a working eval that imports the probe, and reports; the harness then regrades it, which runs the probe
again as a replay, and in every container the eval file runs in (the door's, and ``rerun.py``'s grading container).
The probe looks for everything the container must not show and writes what it found beside itself; the test reads it
from every run. The same probe run on this machine finds every leak it looks for (the control), so a probe that sees
nothing in the container has looked. And a sparse file far past the volume's size fails its command, and no more than
a copy's bound is written here.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from packages.evals.onramp.agent import Caps, ScriptedBrain
from packages.evals.onramp.campaign import CONTROL_REF, adopter, isolation_refusal, onramp_eval, preflight, trials
from packages.evals.onramp.isolation import COPY_BYTES, ENVIRONMENT, GRADING
from packages.evals.onramp.sandbox import HERE, DocsBuild, Sandbox

pytestmark = [pytest.mark.integration, pytest.mark.docker]

REPO = HERE.parents[2]
#: The name of the variable set in the harness's own environment, and its value, which nothing in a trial may see.
CANARY_VARIABLE = "ONRAMP_CANARY_KEY"
CANARY_VALUE = "canary-4b1d0c2e-not-a-key"
#: Names that would mean the harness, its graders or a secret is reachable from inside a trial.
#: What a container's environment holds besides the harness's own: Docker's HOSTNAME, and the base image's build.
IMAGE_VARIABLES = ("HOSTNAME", "PYTHON_VERSION", "PYTHON_SHA256")
FORBIDDEN_NAMES = ("twins", "transcripts", "apps.jsonl", "rerun.py", "probe.jsonl", ".env", "canary.txt", "onramp")

#: Where the container's probe tries to connect: the internet, by address and by name, and the Docker daemon.
NETWORK = [("1.1.1.1", 443), ("pypi.org", 443), ("host.docker.internal", 2375)]

_PROBE = """\
import json, os, pathlib, socket, sys

found = {"uid": os.getuid(), "variable": os.environ.get(VARIABLE), "variables": sorted(os.environ)}
found["environ_holding_canary"] = []
for proc in pathlib.Path("/proc").iterdir():
    if proc.name.isdigit():
        try:
            if VALUE.encode() in (proc / "environ").read_bytes():
                found["environ_holding_canary"].append(proc.name)
        except OSError:
            pass
found["exists"] = {path: os.path.exists(path) for path in PATHS}
found["readable"] = {}
for path in PATHS:
    try:
        found["readable"][path] = VALUE in pathlib.Path(path).read_text()
    except OSError as why:
        found["readable"][path] = type(why).__name__


def connect(host, port):
    try:
        socket.create_connection((host, port), timeout=3).close()
        return "connected"
    except OSError as why:
        return type(why).__name__


found["network"] = {f"{h}:{p}": connect(h, p) for h, p in NETWORK}
found["interfaces_up"] = []
for _, dev in socket.if_nameindex():
    try:
        if int(open(f"/sys/class/net/{dev}/flags").read(), 16) & 1:
            found["interfaces_up"].append(dev)
    except OSError as why:
        found["interfaces_up"].append(f"{dev}: {type(why).__name__}")
found["routes"] = open("/proc/net/route").read().splitlines()[1:]
status = {line.split(":")[0]: line.split()[1] for line in open("/proc/self/status") if line.startswith("Cap")}
found["capabilities"], found["bounding"] = status["CapEff"], status["CapBnd"]
hits = []
for root, dirs, files in os.walk(WALK):
    if root == "/":
        dirs[:] = [d for d in dirs if d not in ("proc", "sys", "dev")]
    hits += [os.path.join(root, name) for name in dirs + files if name in FORBIDDEN or VALUE in name]
found["forbidden_names"] = hits
# Where it ran: as the agent's own command, or imported by the eval file in the door's or rerun.py's process.
if __name__ == "__main__":
    out = "probe.json"
else:
    out = "probe-rerun.json" if pathlib.Path(sys.argv[0]).name == "rerun.py" else "probe-door.json"
pathlib.Path(out).write_text(json.dumps(found))
if __name__ == "__main__":
    print(json.dumps(found))
"""


def _probe(paths: list[str], *, walk: str = "/", network: list[tuple[str, int]] = NETWORK) -> str:
    """The probe's source, with what it looks for written in: ``walk`` is the tree searched for the forbidden names
    (the container's whole filesystem; on this machine, a planted tree), ``network`` where it tries to connect."""
    header = (
        f"VARIABLE = {CANARY_VARIABLE!r}\nVALUE = {CANARY_VALUE!r}\nPATHS = {paths!r}\nFORBIDDEN = {FORBIDDEN_NAMES!r}\n"
        f"WALK = {walk!r}\nNETWORK = {network!r}\n"
    )
    return header + _PROBE


@pytest.fixture
def keyholder() -> Iterator[None]:
    """A process on this machine started with the canary in its environment, as the harness is started with a key:
    ``/proc/<pid>/environ`` shows a process's starting environment, which a later ``setenv`` does not change."""
    process = subprocess.Popen(["sleep", "600"], env={"PATH": "/usr/bin:/bin", CANARY_VARIABLE: CANARY_VALUE})
    try:
        yield
    finally:
        process.kill()
        process.wait()


def _transcript(dest: Path, probe: str) -> Path:
    """A scripted agent: it reads the guide, writes and runs the probe, adds a working eval to holds that imports the
    probe (so it runs wherever the eval file runs), and reports."""
    evals = (REPO / "packages/evals/onramp/transcripts/holds.jsonl").read_text(encoding="utf-8").splitlines()
    write_eval, door, done = (json.loads(line) for line in evals[4:7])
    write_eval["text"] = "import probe  # the canary: it looks around wherever this file runs\n" + write_eval["text"]
    acts = [
        {"tool": "read", "path": "{llms}"},
        {"tool": "write", "path": "probe.py", "text": probe},
        {"tool": "run", "command": "python probe.py"},
        write_eval,
        door,
        done,
    ]
    dest.write_text("".join(json.dumps(act) + "\n" for act in acts), encoding="utf-8")
    return dest


@pytest.fixture(scope="module")
def control(tmp_path_factory: pytest.TempPathFactory) -> DocsBuild:
    """The control arm, as the live baseline builds it: the wheel of ``CONTROL_REF`` installed in a trial image."""
    if (why := isolation_refusal()) is not None:
        pytest.fail(f"these canaries need a Docker daemon: {why}")
    wheels = HERE / "runs" / "wheels" / CONTROL_REF
    found = sorted(wheels.glob("3tears_evals-*.whl")) if wheels.is_dir() else []
    if not found:
        source = tmp_path_factory.mktemp("control-source")
        archive = subprocess.run(
            ["git", "archive", CONTROL_REF, "packages/evals"], cwd=REPO, capture_output=True, check=True
        )
        subprocess.run(["tar", "-x", "-C", str(source)], input=archive.stdout, check=True)
        out = tmp_path_factory.mktemp("control-wheel")
        subprocess.run(
            ["uv", "build", "--quiet", "--wheel", "--out-dir", str(out), str(source / "packages/evals")], check=True
        )
        found = sorted(out.glob("3tears_evals-*.whl"))
    return DocsBuild.in_image("docs@control", found[0], tmp_path_factory.mktemp("images"))


@pytest.mark.timeout(1200)
async def test_nothing_an_agent_runs_reaches_a_key_a_file_here_the_repo_or_the_network(
    control: DocsBuild, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keyholder: None
) -> None:
    monkeypatch.setenv(CANARY_VARIABLE, CANARY_VALUE)
    canary_file = tmp_path / "canary.txt"
    canary_file.write_text(CANARY_VALUE, encoding="utf-8")
    paths = [str(canary_file), str(REPO / ".env"), str(REPO), "/var/run/docker.sock", "/run/docker.sock"]
    transcript = _transcript(tmp_path / "probe.jsonl", _probe(paths))

    def brains(sandbox: Sandbox) -> ScriptedBrain:
        return ScriptedBrain(transcript, sandbox)

    arms = {control.name: adopter(control, brains, workdir=tmp_path / "trials", caps=Caps())}
    result = await onramp_eval(arms, k=1, max_cost_usd=None, apps=["holds"]).run(store=tmp_path / "onramp.sqlite")
    (row,) = trials(result)
    record: dict[str, Any] = row["record"]  # Any: a trial's record
    assert record["build"]["image"] == control.image and record["stopped"] == "done"

    trial = Path(record["sandbox"]).parent
    runs = {
        "the agent's command": trial / "app" / "probe.json",
        "the agent's door run of its eval": trial / "app" / "probe-door.json",
        "the regrade's replay of the agent's command": trial / "regrade-twin" / "probe.json",
        "the regrade's door run of the eval": trial / "regrade" / "probe-door.json",
        "rerun.py's grading run of the eval on the app": trial / "regrade" / "probe-rerun.json",
        "rerun.py's grading run of the eval on the twin": trial / "regrade-twin" / "probe-rerun.json",
    }
    for where, written in runs.items():
        found = json.loads(written.read_text(encoding="utf-8"))
        assert found["uid"] == 10001, where
        # The harness's five, and what Docker and the base image set: the container's name and the Python build.
        assert found["variable"] is None, where
        assert set(found["variables"]) <= {*ENVIRONMENT, *IMAGE_VARIABLES}, (where, found["variables"])
        assert found["environ_holding_canary"] == [], where
        assert not any(found["exists"].values()), (where, found["exists"])
        assert all(value != "connected" for value in found["network"].values()), (where, found["network"])
        # Only loopback is up, and there is no route anywhere: --network none.
        assert found["interfaces_up"] == ["lo"] and found["routes"] == [], (where, found["interfaces_up"])
        # No capability held, and none that could ever be gained.
        assert int(found["capabilities"], 16) == 0 and int(found["bounding"], 16) == 0, where
        # The grading container holds rerun.py, which reads the eval; nothing else of the harness is anywhere.
        grading = "rerun.py" in where
        assert found["forbidden_names"] == ([f"{GRADING}/rerun.py"] if grading else []), (
            where,
            found["forbidden_names"],
        )

    # The control arm is the wheel and the llms.txt inside it, and the agent is shown where the image holds them.
    assert record["build"]["package_dir"] == str(control.shown_package_dir)
    assert {"page": "llms.txt", "lines": record["pages_read"][0]["lines"], "turn": 1} in record["pages_read"]
    assert control.shown_llms_txt.is_relative_to(control.shown_package_dir)
    # The work itself ran in the container, and the graders read it: the agent's eval passes all three.
    grades = {name: row["scores"][name] for name in ("eval_runs", "measures_the_app", "read_right")}
    assert grades == {"eval_runs": 1.0, "measures_the_app": 1.0, "read_right": 1.0}, record["checks"]


@pytest.fixture
def listener() -> Iterator[tuple[str, int]]:
    """A port on this machine that takes connections: what the probe connects to in the control."""
    server = socket.create_server(("127.0.0.1", 0))
    server.settimeout(5)
    try:
        yield server.getsockname()[:2]
    finally:
        server.close()


def test_the_probe_finds_every_leak_it_looks_for_where_there_is_one(
    tmp_path: Path, keyholder: None, listener: tuple[str, int]
) -> None:
    """The control: on this machine, started with the canary in its environment, the probe finds every leak it looks
    for: the variable, a key-holding process's environment through ``/proc``, the canary file and the repository, a
    connection, an interface other than loopback and a route, the user it runs as and the capabilities it could be
    granted, and the forbidden names in a planted tree. Each check that finds nothing in a container has looked."""
    canary_file = tmp_path / "canary.txt"
    canary_file.write_text(CANARY_VALUE, encoding="utf-8")
    planted = tmp_path / "planted"
    (planted / "deep" / "twins").mkdir(parents=True)
    (planted / f"held-{CANARY_VALUE}").write_text("", encoding="utf-8")
    (planted / "deep" / "rerun.py").write_text("", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    (work / "probe.py").write_text(
        _probe([str(canary_file), str(REPO)], walk=str(planted), network=[listener]), encoding="utf-8"
    )
    environment = {"PATH": os.environ["PATH"], CANARY_VARIABLE: CANARY_VALUE}
    subprocess.run([sys.executable, "-I", "probe.py"], cwd=work, env=environment, check=True, capture_output=True)
    found = json.loads((work / "probe.json").read_text(encoding="utf-8"))

    assert found["variable"] == CANARY_VALUE and CANARY_VARIABLE in found["variables"]
    assert found["environ_holding_canary"], "a key-holding process's environment is open through /proc here"
    assert found["exists"] == {str(canary_file): True, str(REPO): True}
    assert found["readable"][str(canary_file)] is True
    assert found["network"] == {f"{listener[0]}:{listener[1]}": "connected"}
    assert set(found["interfaces_up"]) - {"lo"} and found["routes"], "this machine has a network"
    assert found["uid"] == os.getuid()
    assert int(found["bounding"], 16) != 0, "a process here could be granted capabilities"
    assert sorted(found["forbidden_names"]) == sorted(
        [str(planted / "deep" / "twins"), str(planted / "deep" / "rerun.py"), str(planted / f"held-{CANARY_VALUE}")]
    )


def _written() -> int:
    """Bytes this process has handed to ``write`` and its kind, so far."""
    counters = Path("/proc/self/io").read_text(encoding="utf-8").splitlines()
    return int(next(line.split()[1] for line in counters if line.startswith("wchar:")))


#: Sparse files the agent's command makes: each takes no room in the volume and streams all of its size.
SPARSE = {
    "one of 2 GiB": [2 * 2**30],
    "three of 120 MiB, each under the bound and together over it": [120 * 2**20] * 3,
}


@pytest.mark.timeout(600)
@pytest.mark.parametrize("sizes", list(SPARSE.values()), ids=list(SPARSE))
async def test_a_sparse_file_past_the_volume_fails_its_command_and_writes_no_more_than_the_bound_here(
    control: DocsBuild, tmp_path: Path, sizes: list[int]
) -> None:
    sandbox = Sandbox.create(tmp_path / "app", "holds", control)
    shipped = sorted(path.relative_to(sandbox.root) for path in sandbox.root.rglob("*"))
    made = "".join(f"open('sparse-{n}', 'wb').truncate({size})\n" for n, size in enumerate(sizes))
    assert sandbox.write("sparse.py", made).ok
    before = _written()
    observation = await sandbox.run("python sparse.py")
    written = _written() - before
    assert not observation.ok and observation.exit_code is None, observation.text
    assert "larger than" in observation.first_error and "copied back" in observation.first_error
    assert written <= COPY_BYTES + 8 * 2**20, f"{written} bytes written here, past the bound"
    # The directory is as it was before the command: nothing it made came back, and no partial copy is left.
    left = sorted(path.relative_to(sandbox.root) for path in sandbox.root.rglob("*"))
    assert left == sorted([*shipped, Path("sparse.py")]), left
    assert [path.name for path in tmp_path.iterdir()] == ["app"]


@pytest.mark.timeout(1200)
def test_the_scripted_pre_flight_passes_every_app_on_the_control_wheel(control: DocsBuild, tmp_path: Path) -> None:
    """The guard every agent run passes first: the fixed transcripts succeed on each arm's wheel, in its containers,
    3 of 3 apps on every check; a wheel the harness reads wrong is refused there (the offline test holds the refusal)."""
    assert preflight([control], tmp_path) is None
