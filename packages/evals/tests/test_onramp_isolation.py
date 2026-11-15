"""The on-ramp's trial container, held by canaries: what an agent's commands run in cannot reach a key, a file of this
machine, the repository or the network.

These drive a Docker daemon, so they run on this machine, never on the testament pool:
``./scripts/test-integration.sh --here -k onramp -rs``. The image is built once (set-up needs the network; a trial has
none) from the control arm's wheel, as the live baseline builds it.

Each canary runs through the real path with the scripted agent: a transcript writes a probe, runs it as the agent's own
command, adds a working eval, and reports; the harness then regrades it, which runs the probe again. The probe looks for
everything the container must not show and writes what it found beside itself; the test reads it from both runs. The
same probe run on this machine (the scripted agent's ``HostRunner``) finds the canaries, so a probe that sees nothing
in the container has looked.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from packages.evals.onramp.agent import Caps, ScriptedBrain
from packages.evals.onramp.campaign import CONTROL_REF, adopter, isolation_refusal, onramp_eval, trials
from packages.evals.onramp.isolation import ENVIRONMENT, HostRunner
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

_PROBE = """\
import json, os, pathlib, socket

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


found["network"] = {f"{h}:{p}": connect(h, p) for h, p in (("1.1.1.1", 443), ("pypi.org", 443), ("host.docker.internal", 2375))}
found["interfaces_up"] = []
for _, dev in socket.if_nameindex():
    try:
        if int(open(f"/sys/class/net/{dev}/flags").read(), 16) & 1:
            found["interfaces_up"].append(dev)
    except OSError as why:
        found["interfaces_up"].append(f"{dev}: {type(why).__name__}")
found["routes"] = open("/proc/net/route").read().splitlines()[1:]
found["capabilities"] = next(line.split()[1] for line in open("/proc/self/status") if line.startswith("CapEff"))
hits = []
for root, dirs, files in os.walk("/") if WALK else ():
    if root == "/":
        dirs[:] = [d for d in dirs if d not in ("proc", "sys", "dev")]
    hits += [os.path.join(root, name) for name in dirs + files if name in FORBIDDEN or VALUE in name]
found["forbidden_names"] = hits
pathlib.Path("probe.json").write_text(json.dumps(found))
print(json.dumps(found))
"""


def _probe(paths: list[str], *, walk: bool = True) -> str:
    """The probe's source, with what it looks for written in; ``walk`` searches the whole filesystem for the
    forbidden names (in the container; this machine's is too large to walk in a test)."""
    header = (
        f"VARIABLE = {CANARY_VARIABLE!r}\nVALUE = {CANARY_VALUE!r}\nPATHS = {paths!r}\nFORBIDDEN = {FORBIDDEN_NAMES!r}\n"
        f"WALK = {walk!r}\n"
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
    """A scripted agent: it reads the guide, writes and runs the probe, adds a working eval to holds, and reports."""
    evals = (REPO / "packages/evals/onramp/transcripts/holds.jsonl").read_text(encoding="utf-8").splitlines()
    write_eval, door, done = (json.loads(line) for line in evals[4:7])
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
    agent_run = json.loads((trial / "app" / "probe.json").read_text(encoding="utf-8"))
    regrade_run = json.loads((trial / "regrade-twin" / "probe.json").read_text(encoding="utf-8"))
    for where, found in (("the agent's command", agent_run), ("the regrade's replay of it", regrade_run)):
        assert found["uid"] != 0, where
        # The harness's five, and what Docker and the base image set: the container's name and the Python build.
        assert found["variable"] is None, where
        assert set(found["variables"]) <= {*ENVIRONMENT, *IMAGE_VARIABLES}, (where, found["variables"])
        assert found["environ_holding_canary"] == [], where
        assert not any(found["exists"].values()), (where, found["exists"])
        assert all(value != "connected" for value in found["network"].values()), (where, found["network"])
        # Only loopback is up, and there is no route anywhere: --network none.
        assert found["interfaces_up"] == ["lo"] and found["routes"] == [], (where, found["interfaces_up"])
        assert int(found["capabilities"], 16) == 0, where
        assert found["forbidden_names"] == [], (where, found["forbidden_names"])

    # The control arm is the wheel and the llms.txt inside it, and the agent is shown where the image holds them.
    assert record["build"]["package_dir"] == str(control.shown_package_dir)
    assert {"page": "llms.txt", "lines": record["pages_read"][0]["lines"], "turn": 1} in record["pages_read"]
    assert control.shown_llms_txt.is_relative_to(control.shown_package_dir)
    # The work itself ran in the container, and the graders read it: the agent's eval passes all three.
    grades = {name: row["scores"][name] for name in ("eval_runs", "measures_the_app", "read_right")}
    assert grades == {"eval_runs": 1.0, "measures_the_app": 1.0, "read_right": 1.0}, record["checks"]


@pytest.mark.timeout(600)
async def test_the_probe_finds_the_canaries_where_they_are_there(tmp_path: Path, keyholder: None) -> None:
    """The control: on this machine, as the scripted agent's ``--host`` runs it, the probe sees the canary file, the
    repository and a key-holding process's environment through ``/proc``: what it finds nothing of in the container,
    it would have found."""
    canary_file = tmp_path / "canary.txt"
    canary_file.write_text(CANARY_VALUE, encoding="utf-8")
    here = DocsBuild.here("docs@here")
    sandbox = Sandbox.create(tmp_path / "app", "holds", here, HostRunner(here.python))
    assert sandbox.write("probe.py", _probe([str(canary_file), str(REPO)], walk=False)).ok
    observation = await sandbox.run("python probe.py")
    assert observation.ok, observation.text
    found = json.loads((sandbox.root / "probe.json").read_text(encoding="utf-8"))
    assert found["exists"] == {str(canary_file): True, str(REPO): True}
    assert found["readable"][str(canary_file)] is True
    assert found["environ_holding_canary"], "the harness's environment is open through /proc on this machine"
