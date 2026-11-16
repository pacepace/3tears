"""The on-ramp's agent test, piece by piece, offline: the apps and their twins, the sandbox, the run's caps, what a run
records, and the live path's guards. ``test_onramp_end_to_end.py`` runs the whole harness."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import math
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

from threetears.evals import Model
from threetears.evals.negotiation import MAX_RESENDS
from threetears.evals.providers import Catalogue, DeclinedRetry, ProviderError
from threetears.evals.run.runner import DEFAULT_CELL_TIMEOUT_S
from packages.evals.onramp.agent import (
    BILLED_SENDS_AT_MOST,
    REPLY_TOKENS,
    SYSTEM,
    Act,
    Caps,
    ModelBrain,
    ScriptedBrain,
    SendMeter,
    SendsSpent,
    Step,
    Turn,
    Usage,
    attempt,
    cause,
    done_report,
    errors_hit,
    metered_call,
    prompt,
    read_reply,
    parse_report,
    scored_door_turns,
    stuck_point,
)
from packages.evals.onramp.checks import REGRADE_BUDGET_S, eval_runs, measures_the_app, read_right
from packages.evals.onramp.rescore import read_again
from packages.evals.onramp.campaign import (
    CELL_BUDGET_S,
    CONTROL_REF,
    TIERS,
    Tier,
    census,
    clopper_pearson,
    failure_causes,
    preflight_refusal,
    spend_line,
    live_refusal,
    main,
    quote,
)
from packages.evals.onramp import isolation
from packages.evals.onramp.isolation import (
    COPY_STOP_S,
    ENVIRONMENT,
    RERUN,
    WORK,
    Container,
    CopyRefused,
    HostRunner,
    Ran,
    Runner,
    pack,
    stoppable,
    unpack,
)
from packages.evals.onramp.sandbox import APPS, READ_BYTES, TWINS, DocsBuild, Observation, Sandbox

FIXTURES = {
    "recipes": "tests/fixtures/scaling.jsonl",
    "holds": "tests/fixtures/requests.jsonl",
    "glossator": "tests/fixtures/glossary.jsonl",
}


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _passes(app: str, root: Path, name: str) -> list[bool]:
    """Whether each of the app's fixture cases comes out as the fixture expects, on the app at ``root``."""
    rows = [json.loads(line) for line in (APPS / app / FIXTURES[app]).read_text(encoding="utf-8").splitlines()]
    if app == "recipes":
        kitchen = _load(root / "kitchen.py", name).Kitchen()
        return [kitchen.scale(row["recipe"], row["servings"]) == row["expected"] for row in rows]
    if app == "holds":
        service = _load(root / "service.py", name)
        return [
            sorted(h.id for h in service.handle(row["request"]).holds if h.expired) == row["expired"] for row in rows
        ]
    glossator = _load(root / "glossator.py", name)
    return [glossator.gloss(row) == {key: row[key] for key in ("id", "lang", "headword", "slug")} for row in rows]


@pytest.mark.parametrize("app", sorted(FIXTURES))
def test_each_app_fails_only_its_planted_slice_and_its_twin_passes_every_case(app: str, tmp_path: Path) -> None:
    # Each loaded from a copy of its own, with no bytecode beside it: the app and its twin can be the same size with
    # the same timestamp, and a cached compile of one would then be run for the other.
    no_cache = shutil.ignore_patterns("__pycache__")
    shipped_root, twin_root = tmp_path / "shipped", tmp_path / "twin"
    shutil.copytree(APPS / app, shipped_root, ignore=no_cache)
    shutil.copytree(APPS / app, twin_root, ignore=no_cache)
    shutil.copytree(TWINS / app, twin_root, ignore=no_cache, dirs_exist_ok=True)
    shipped = _passes(app, shipped_root, f"onramp_{app}_shipped")
    assert all(_passes(app, twin_root, f"onramp_{app}_twin")), "the twin is the app with its defect fixed"
    assert 0 < shipped.count(False) < len(shipped), "the defect is planted on a slice, not everywhere"


def _build(tmp_path: Path) -> DocsBuild:
    return DocsBuild.here("docs@test")


class _Answering(Runner):
    """Where a model-driven test's commands go: nowhere. It runs nothing, so nothing it is handed can reach a key or
    the network, and answers each command as ``pwd`` would, from the directory a trial container shows."""

    isolated = True
    python = Container.python
    rerun = Container.rerun

    def __init__(self) -> None:
        self.ran: list[list[str]] = []

    async def run(self, argv: Sequence[str], *, cwd: Path, timeout_s: float, grading: bool = False) -> Ran:
        self.ran.append(list(argv))
        return Ran(0, f"{WORK}\n", "", WORK)


def _isolated(tmp_path: Path, app: str = "recipes", name: str = "app") -> Sandbox:
    """A sandbox whose commands go to :class:`_Answering`, as a model-driven run's go to a trial container."""
    return Sandbox.create(tmp_path / name, app, _build(tmp_path), _Answering())


def test_the_sandbox_holds_the_agent_to_the_app_and_the_installed_package(tmp_path: Path) -> None:
    sandbox = Sandbox.create(tmp_path / "app", "recipes", _build(tmp_path))
    assert not sandbox.read("/etc/hostname").ok
    assert not sandbox.read("../../outside.txt").ok
    assert sandbox.read("kitchen.py").ok
    sandbox.turn = 3
    assert sandbox.read(str(sandbox.build.llms_txt), 1).ok
    assert sandbox.pages[-1]["page"] == "llms.txt" and sandbox.pages[-1]["turn"] == 3
    assert not sandbox.write("/tmp/escape.py", "x").ok
    assert not sandbox.write("../escape.py", "x").ok
    assert sandbox.write("evals.py", "x = 1\n").ok and sandbox.written == ["evals.py"]
    assert sandbox.made() == ["evals.py"]


async def test_run_admits_only_python_ls_and_cat_with_no_shell(tmp_path: Path) -> None:
    sandbox = Sandbox.create(tmp_path / "app", "holds", _build(tmp_path))
    for command in ("curl https://example.com", "bash -c ls", "rm -rf .", "ls; cat service.py", ""):
        observation = await sandbox.run(command)
        assert not observation.ok and observation.exit_code is None, command
    listing = await sandbox.run("ls")
    assert listing.ok and "service.py" in listing.text.splitlines()
    assert (await sandbox.run(f"cat {sandbox.build.package_dir / '__init__.py'}")).ok
    assert sandbox.pages[-1]["page"] == "__init__.py"


async def test_a_command_on_this_machine_runs_with_no_key_from_the_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-key")
    sandbox = Sandbox.create(tmp_path / "app", "holds", _build(tmp_path))
    assert isinstance(sandbox.runner, HostRunner) and not sandbox.runner.isolated
    shown = await sandbox.run('python -c "import json, os; print(json.dumps(sorted(os.environ)))"')
    assert json.loads(shown.text.splitlines()[1]) == sorted(ENVIRONMENT)


#: Every option a trial container is started with. A new one (a volume, a device, a host network, a shared PID
#: namespace) fails this test until it is read and added here.
_CONTAINER_OPTIONS = {
    "image", "command", "name", "labels", "auto_remove", "network_mode", "user", "read_only", "mounts", "tmpfs",
    "cap_drop", "security_opt", "privileged", "nano_cpus", "mem_limit", "memswap_limit", "pids_limit", "init",
    "working_dir", "environment",
}  # fmt: skip


@pytest.mark.parametrize("grading", [False, True])
def test_a_trial_container_has_no_network_no_path_of_this_machine_no_key_and_limits(grading: bool) -> None:
    settings = Container("onramp-trial:test").settings("onramp-trial-x", lifetime_s=300, grading=grading)
    assert set(settings) == _CONTAINER_OPTIONS
    assert settings["network_mode"] == "none" and settings["privileged"] is False and settings["read_only"] is True
    assert not settings["user"].startswith("0") and settings["user"] != "root"
    assert settings["cap_drop"] == ["ALL"] and settings["security_opt"] == ["no-new-privileges"]
    assert settings["environment"] == dict(ENVIRONMENT), "the harness's own environment is never passed"
    # Every mount is an anonymous in-memory volume: nothing of this machine, the Docker socket included.
    assert [mount["Target"] for mount in settings["mounts"]] == (["/work", "/grading"] if grading else ["/work"])
    for mount in settings["mounts"]:
        assert mount["Type"] == "volume" and mount.get("Source") is None
        assert mount["VolumeOptions"]["DriverConfig"]["Options"]["type"] == "tmpfs"
    assert settings["nano_cpus"] > 0 and settings["mem_limit"] == settings["memswap_limit"]
    assert settings["pids_limit"] > 0 and settings["command"] == ["sleep", "300"] and settings["auto_remove"] is True


def _step(turn: int, tool: str, ok: bool, text: str = "", command: str = "", first_error: str = "") -> Step:
    act = Act(tool=tool, path="", start=1, text="", command=command)  # type: ignore[arg-type]
    return Step(turn, act, Observation(ok=ok, text=text, first_error=first_error))


def test_the_stuck_point_is_the_longest_run_of_failing_turns_and_the_message_that_opened_it() -> None:
    steps = [
        _step(1, "run", False, command="python a.py", first_error="first"),
        _step(2, "run", True, command="ls"),
        _step(3, "run", False, command="python -m threetears.evals e.py", first_error="opened it"),
        _step(4, "write", True),  # a fix tried: neither extends nor ends the span
        _step(5, "read", False, first_error="no such file"),
        _step(6, "run", False, command="python -m threetears.evals e.py", first_error="again"),
        _step(7, "run", True, command="python -m threetears.evals e.py"),
    ]
    # The ls at turn 2 is the agent looking around, not a way out: only the door working at turn 7 ends the span.
    assert stuck_point(steps) == {
        "from_turn": 1, "to_turn": 6, "failed_turns": 4, "opened_by": "first", "cause": "cannot tell"
    }  # fmt: skip
    assert stuck_point([_step(1, "run", True, command="ls")]) is None


def test_errors_hit_names_each_newcomer_error_the_package_printed_with_its_first_turn() -> None:
    steps = [
        _step(1, "run", True, command="python -m threetears.evals --help"),
        _step(2, "run", False, "exit 2\n[stderr]\nevals.py: the cases file cases.jsonl is not there (in .)", "x"),
        _step(3, "run", False, "a grader's __name__ names its measure, and <lambda> has none", "x"),
        _step(4, "run", False, "error: case 0: the candidate raised TypeError: f() takes 0 arguments", "x"),
        _step(5, "run", False, "the cases file cases.jsonl is not there", "x"),
    ]
    assert errors_hit(steps) == {"help_first": 1, "cases_missing": 2, "grader_unnamed": 3, "runner_raised": 4}


def test_the_report_is_read_from_the_last_json_object_the_agent_wrote() -> None:
    assert parse_report('Here it is:\n{"eval_file": "e.py", "pass_rate": 0.5}') == (
        {"eval_file": "e.py", "pass_rate": 0.5},
        "",
    )
    assert parse_report(None)[0] is None
    assert parse_report("no json here")[0] is None


class _Priced:
    """A brain whose every turn lists the app and costs ``each``, bounded by ``bound`` (``None``: unbounded), each call
    sent ``resends`` more times after a drop that may have been billed."""

    model = "priced"

    def __init__(self, each: float | None, bound: float | None, *, resends: int = 0) -> None:
        self.each, self.bound, self.resends = each, bound, resends

    async def ceiling(self, system: str, user: str) -> float | None:
        return self.bound

    async def next(self, system: str, user: str, steps: Sequence[Step]) -> Turn:
        act = Act(tool="run", path="", start=1, text="", command="ls")
        return Turn(act, Usage(100, 10, self.each, maybe_billed_sends=self.resends))


async def test_a_run_stops_before_a_turn_that_could_take_it_past_its_cap(tmp_path: Path) -> None:
    # A call may be billed once per request the negotiation sends and once more for the last (MAX_RESENDS + 2), so it
    # is sent only while the run's charge plus that many of its ceilings stays within the cap: after 0.20 is charged,
    # a third call could reach 0.20 + 6 x 0.10, past 0.75.
    assert BILLED_SENDS_AT_MOST == MAX_RESENDS + 2 == 6
    run = await attempt(_isolated(tmp_path), _Priced(0.10, 0.10), Caps(cost_usd=0.75))
    assert run.stopped == "cap" and len(run.usages) == 2
    assert run.spent() == pytest.approx(0.20) and run.charged() == pytest.approx(0.20)
    assert run.tokens("input_tokens") == 200


@pytest.mark.parametrize("resends", [0, 1, BILLED_SENDS_AT_MOST - 1])
@pytest.mark.parametrize("cap", [0.05, 0.35, 0.50, 1.00, 2.00])
async def test_what_a_run_can_be_billed_never_passes_its_cap_even_when_every_call_was_resent(
    tmp_path: Path, resends: int, cap: float
) -> None:
    """Each call billed at its ceiling, and each maybe-billed resend at its ceiling again: the most the provider can
    have charged stays within the cap."""
    run = await attempt(_isolated(tmp_path), _Priced(0.10, 0.10, resends=resends), Caps(cost_usd=cap))
    charged = run.charged()
    assert run.stopped == "cap" and charged is not None
    assert charged == pytest.approx(len(run.usages) * 0.10 * (1 + resends))
    assert charged <= cap + 1e-9 and run.maybe_billed_sends() == resends * len(run.usages)


async def test_a_turn_no_price_can_bound_is_never_sent_under_a_cap(tmp_path: Path) -> None:
    sandbox = _isolated(tmp_path)
    run = await attempt(sandbox, _Priced(0.01, None), Caps(cost_usd=1.0))
    assert run.stopped == "cap" and run.usages == []


async def test_an_unpriced_turn_makes_the_run_unpriced_and_stops_it_under_a_cap(tmp_path: Path) -> None:
    sandbox = _isolated(tmp_path)
    run = await attempt(sandbox, _Priced(None, 0.01), Caps(cost_usd=1.0))
    assert run.spent() is None and run.stopped == "cap" and len(run.usages) == 1


async def test_a_run_stops_at_its_turn_cap(tmp_path: Path) -> None:
    sandbox = _isolated(tmp_path)
    run = await attempt(sandbox, _Priced(0.0, 0.0), Caps(turns=3))
    assert run.stopped == "turns" and [step.turn for step in run.steps] == [1, 2, 3]


async def test_a_scripted_brain_fills_in_the_build_and_says_done_past_its_transcript(tmp_path: Path) -> None:
    sandbox = Sandbox.create(tmp_path / "app", "recipes", _build(tmp_path))
    script = tmp_path / "script.jsonl"
    script.write_text('{"tool": "read", "path": "{llms}"}\n', encoding="utf-8")
    run = await attempt(sandbox, ScriptedBrain(script, sandbox), Caps())
    assert run.steps[0].act is not None and run.steps[0].act.path == str(sandbox.build.llms_txt)
    assert run.stopped == "done" and run.spent() == 0.0


#: The cheap tier's catalogue row, as OpenRouter lists it: what bounds each turn before it is sent.
_CHEAP_ROW: dict[str, Any] = {  # Any: JSON
    "id": "deepseek/deepseek-v4-flash",
    "pricing": {"prompt": "0.0000001", "completion": "0.0000004"},
    "supported_parameters": ["max_tokens", "reasoning", "response_format", "structured_outputs"],
}


def _act_reply(content: str) -> dict[str, Any]:  # Any: OpenRouter's reply shape
    return {
        "model": "deepseek/deepseek-v4-flash",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "cost": 0.00014, "is_byok": False},
    }


async def test_the_live_brain_is_priced_by_the_package_and_held_to_its_cap(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []  # Any: request bodies

    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": [_CHEAP_ROW]})
        sent.append(json.loads(request.content))
        act = {"tool": "run", "path": "", "start": 1, "text": "", "command": "ls"}
        return httpx.Response(200, json=_act_reply(json.dumps(act)))

    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    brain = ModelBrain(
        lambda meter: TIERS["cheap"].model(None, api_key="scripted", transport=meter, catalogue=Catalogue(http=http)),
        inner=httpx.MockTransport(respond),
    )
    sandbox = _isolated(tmp_path)
    try:
        run = await attempt(sandbox, brain, Caps(cost_usd=0.042))
    finally:
        await brain.aclose()
        await http.aclose()
    assert run.stopped == "cap" and sent, "it ran turns, then stopped before one that could pass the cap"
    spent, charged = run.spent(), run.charged()
    assert spent is not None and spent == pytest.approx(0.00014 * len(sent)) and spent <= 0.042
    assert charged == spent and all(usage.ceiling_usd for usage in run.usages), "each call was bounded before it went"
    assert run.tokens("input_tokens") == 1000 * len(sent)
    assert all("temperature" not in body for body in sent)
    assert all(body["max_tokens"] == REPLY_TOKENS for body in sent), "the limit the agent is told is the one sent"


async def test_the_ceiling_tier_goes_direct_to_anthropic_with_no_temperature(tmp_path: Path) -> None:
    hosts: list[str] = []
    bodies: list[dict[str, Any]] = []  # Any: request bodies

    def respond(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        bodies.append(json.loads(request.content))
        text = json.dumps({"tool": "done", "path": "", "start": 1, "text": "{}", "command": ""})
        reply = {
            "model": TIERS["ceiling"].model_id,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 2000, "output_tokens": 50},
        }
        return httpx.Response(200, json=reply)

    brain = ModelBrain(
        lambda meter: TIERS["ceiling"].model(None, api_key="scripted", transport=meter),
        inner=httpx.MockTransport(respond),
    )
    sandbox = _isolated(tmp_path, "holds")
    try:
        run = await attempt(sandbox, brain, Caps(cost_usd=1.50))
    finally:
        await brain.aclose()
    assert run.stopped == "done" and hosts == ["api.anthropic.com"]
    assert "temperature" not in bodies[0] and bodies[0]["max_tokens"] == REPLY_TOKENS
    assert run.spent() == pytest.approx(2000 * 0.10 / 1e6 + 50 * 0.50 / 1e6), "Anthropic's published rates"


class _Worst:
    """A provider that does the most billing a call can meet: it drops the first send of every request with no
    response (which may have been served and billed), refuses a setting the request carried while the negotiation
    can answer one (``MAX_RESENDS`` times), and then serves the last request, or loses its resend too. It keeps the
    most it can have billed: each dropped or lost send at the call's ceiling, and a served one at its cost."""

    def __init__(self, ending: str) -> None:
        self.ending, self.seen, self.refused = ending, set[bytes](), 0
        self.calls: list[list[str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": [_REFUSABLE_ROW]})
        body = json.loads(request.content)
        if request.content not in self.seen:
            self.seen.add(request.content)
            if not self.calls or self.calls[-1][-1] in ("served", "lost"):
                self.calls.append([])
            self.calls[-1].append("dropped")
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.", request=request)
        if self.refused < MAX_RESENDS and ("temperature" in body or "reasoning" in body):
            self.refused += 1
            param = "temperature" if "temperature" in body else "reasoning_effort"
            self.calls[-1].append(f"refused {param}")
            return _upstream_refusal(param, "unsupported_parameter" if param == "temperature" else "unsupported_value")
        self.refused = 0
        if self.ending == "lost":
            self.calls[-1].append("lost")
            raise httpx.ReadError("the connection reset", request=request)
        self.calls[-1].append("served")
        return httpx.Response(200, json=_act_reply(json.dumps(_LS)))

    def most_billed(self, ceilings: Sequence[float | None]) -> float:
        billed = 0.0
        for sends, ceiling in zip(self.calls, ceilings, strict=True):
            assert ceiling is not None
            billed += sum(ceiling for send in sends if send in ("dropped", "lost"))
            billed += 0.00014 * sends.count("served")
        return billed


#: A row whose model takes a temperature and reasoning, so a call carries settings the negotiation can be refused.
_REFUSABLE_ROW: dict[str, Any] = {  # Any: JSON
    **_CHEAP_ROW,
    "supported_parameters": ["max_tokens", "temperature", "reasoning", "response_format", "structured_outputs"],
}
_LS = {"tool": "run", "path": "", "start": 1, "text": "", "command": "ls"}


def _upstream_refusal(param: str, code: str) -> httpx.Response:
    """OpenRouter relaying an upstream's 400 that refuses ``param``."""
    raw = json.dumps({"error": {"message": f"{param} is not supported", "param": param, "code": code}})
    error = {"code": 400, "message": "Provider returned error", "metadata": {"raw": raw, "provider_name": "x"}}
    return httpx.Response(400, json={"error": error})


async def _no_wait(seconds: float) -> None:
    return None


@pytest.mark.parametrize("ending", ["served", "lost"])
async def test_the_most_a_call_can_be_billed_is_reserved_before_it_is_sent_and_never_passes_the_cap(
    tmp_path: Path, ending: str
) -> None:
    """The worst case, scripted: every request's first send dropped, the negotiation refused ``MAX_RESENDS`` times,
    and the last request served, or lost on its resend. A call then makes ``MAX_RESENDS + 2`` sends that may have been
    billed, exactly what is reserved before it goes; the meter counts each, so the run is charged at least what the
    provider can have billed, and that stays within the cap."""
    provider = _Worst(ending)
    catalogue = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    brain = ModelBrain(
        lambda meter: Model(
            _CHEAP_ROW["id"],
            max_tokens=4096,
            effort="max",
            temperature=0.2,
            api_key="scripted",
            transport=meter,
            catalogue=Catalogue(http=catalogue),
            retry=DeclinedRetry(sleep=_no_wait),
        ),  # fmt: skip
        inner=httpx.MockTransport(provider),
    )
    cap = 0.10
    try:
        run = await attempt(_isolated(tmp_path), brain, Caps(cost_usd=cap))
    finally:
        await brain.aclose()
        await catalogue.aclose()
    first = provider.calls[0]
    assert (
        first.count("dropped") == MAX_RESENDS + 1 and len([s for s in first if s.startswith("refused")]) == MAX_RESENDS
    )
    assert len(first) - first.count("refused temperature") - first.count("refused reasoning_effort") == (
        BILLED_SENDS_AT_MOST
    ), first
    charged = run.charged()
    assert charged is not None and charged <= cap + 1e-12
    assert provider.most_billed([usage.ceiling_usd for usage in run.usages]) <= charged + 1e-12
    if ending == "served":
        assert run.usages[0].maybe_billed_sends == BILLED_SENDS_AT_MOST - 1 and run.stopped == "cap"
        assert len(run.usages) > 1, "the cap leaves room for more than one call"
    else:
        assert run.stopped == "error" and len(run.usages) == 1


async def test_the_meter_refuses_a_send_past_what_the_call_reserved() -> None:
    """However a client resends, a call makes no more sends that may be billed than its reservation covers."""
    reached: list[int] = []

    def served(request: httpx.Request) -> httpx.Response:
        reached.append(1)
        return httpx.Response(200, json={})

    url = "https://openrouter.ai/api/v1/chat/completions"
    async with httpx.AsyncClient(transport=SendMeter(httpx.MockTransport(served))) as client:
        with metered_call() as sends:
            for _ in range(BILLED_SENDS_AT_MOST):
                await client.post(url, json={})
            with pytest.raises(SendsSpent):
                await client.post(url, json={})
        await client.post(url, json={})  # outside a call: not counted
    assert (sends.made, sends.maybe_billed) == (BILLED_SENDS_AT_MOST, BILLED_SENDS_AT_MOST)
    assert len(reached) == BILLED_SENDS_AT_MOST + 1


def test_a_claude_model_is_refused_any_route_but_anthropic_direct() -> None:
    with pytest.raises(ValueError, match="direct to Anthropic"):
        Tier("x", f"anthropic/{TIERS['ceiling'].model_id}", None, 1.0, 0.1, 0.5, "")
    with pytest.raises(ValueError, match="direct to Anthropic"):
        Tier("x", TIERS["ceiling"].model_id, None, 1.0, 0.1, 0.5, "")
    assert TIERS["ceiling"].provider == "anthropic" and TIERS["cheap"].provider is None


def test_the_quote_bounds_every_run_at_its_share_of_the_cap() -> None:
    baseline = quote(TIERS["cheap"], runs=9, cap_usd=None)
    assert baseline.per_run_usd == 0.50 and baseline.bound_usd == pytest.approx(4.50)
    assert 0 < baseline.expected_usd < baseline.bound_usd
    held = quote(TIERS["cheap"], runs=9, cap_usd=1.80)
    assert held.per_run_usd == pytest.approx(0.20) and held.bound_usd == pytest.approx(1.80)


def test_a_live_run_is_refused_without_its_switch_and_its_cap() -> None:
    assert live_refusal({}, 4.5) is not None
    assert live_refusal({"ONRAMP_LIVE": "1"}, None) is not None
    assert live_refusal({"ONRAMP_LIVE": "1"}, 0.0) is not None
    assert live_refusal({"ONRAMP_LIVE": "1"}, 4.5) is None
    # A fallback key would let one call be resent on a second key, past the sends the cap reserves.
    refused = live_refusal({"ONRAMP_LIVE": "1", "OPENROUTER_API_KEY_FALLBACKS": "X"}, 4.5)
    assert refused is not None and "OPENROUTER_API_KEY_FALLBACKS" in refused


def test_a_live_run_refuses_to_start_where_no_trial_container_can_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setenv("ONRAMP_LIVE", "1")
    monkeypatch.delenv("RECORD", raising=False)
    monkeypatch.setenv("DOCKER_HOST", f"unix://{tmp_path / 'no-daemon.sock'}")
    wheel = tmp_path / "never-built.whl"
    assert main(["--arm", f"docs@control={wheel}", "--cap", "4.5", "--workdir", str(tmp_path)]) == 2
    said = capsys.readouterr().err
    assert "run only in a trial container" in said and "no Docker daemon answered" in said
    assert not (tmp_path / "images").exists() and not any(tmp_path.glob("*/recordings")), "nothing built or sent"


def test_commands_on_this_machine_are_only_for_the_scripted_agent(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as refused:
        main(["--host", "--arm", f"docs@control={tmp_path / 'w.whl'}", "--cap", "1", "--workdir", str(tmp_path)])
    assert refused.value.code == 2 and "only the scripted agent" in capsys.readouterr().err
    assert main(["--scripted", "succeeds", "--workdir", str(tmp_path)]) == 2, "no wheel and no --host: nothing to run"
    assert "names its wheel" in capsys.readouterr().err


async def test_a_model_driven_run_never_runs_its_commands_on_this_machine(tmp_path: Path) -> None:
    host = Sandbox.create(tmp_path / "app", "recipes", _build(tmp_path))
    with pytest.raises(ValueError, match="only the scripted agent"):
        await attempt(host, _Priced(0.0, 0.0), Caps())


def test_the_command_quotes_first_and_refuses_to_spend_unless_told(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.delenv("ONRAMP_LIVE", raising=False)
    wheel = f"docs@control={tmp_path / 'never-built.whl'}"
    assert main(["--arm", wheel, "--dry-run", "--workdir", str(tmp_path)]) == 0
    said = capsys.readouterr().out
    assert "at most: $4.50" in said and "deepseek/deepseek-v4-flash" in said
    assert main(["--arm", wheel, "--cap", "4.5", "--workdir", str(tmp_path)]) == 2
    assert "ONRAMP_LIVE=1" in capsys.readouterr().err
    monkeypatch.setenv("ONRAMP_LIVE", "1")
    assert main(["--arm", wheel, "--workdir", str(tmp_path)]) == 2
    assert "--cap" in capsys.readouterr().err
    assert not (tmp_path / "venvs").exists(), "nothing was built: every refusal came first"
    assert len(CONTROL_REF) >= 7


def test_doing_nothing_is_counted_failed_on_a_grader_returning_false_not_only_on_one_that_raises(
    tmp_path: Path,
) -> None:
    (tmp_path / "cases.jsonl").write_text('{"x": 1}\n{"x": 2}\n{"x": 3}\n', encoding="utf-8")
    (tmp_path / "evals.py").write_text(
        "from threetears.evals import Eval\n\n\n"
        "def echo(case):\n"
        '    """Echo the number."""\n'
        '    return case["x"]\n\n\n'
        "def odd(case, out):\n"
        '    """The answer is odd."""\n'
        "    return out is not None and out % 2 == 1\n\n\n"
        'numbers = Eval("cases.jsonl", runner=echo, grader=[odd], k=1)\n',
        encoding="utf-8",
    )
    done = subprocess.run(
        [sys.executable, str(RERUN), "evals.py", "--nothing"], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    reading = json.loads(done.stdout.splitlines()[-1])
    assert reading["means"] == {"odd": pytest.approx(2 / 3)} and reading["finished"] is True
    assert reading["nothing_unpassed"] == 1.0, "a None answer graded False fails its case, though no check failed"
    assert reading["scored"] == 3, "every case's answer was graded"

    (tmp_path / "evals.py").write_text(
        "from threetears.evals import Eval\n\n\n"
        "def broken(case):\n"
        '    """Fail every case."""\n'
        '    return case["missing"]\n\n\n'
        "def odd(case, out):\n"
        '    """The answer is odd."""\n'
        "    return out % 2 == 1\n\n\n"
        'numbers = Eval("cases.jsonl", runner=broken, grader=[odd], k=1)\n',
        encoding="utf-8",
    )
    done = subprocess.run([sys.executable, str(RERUN), "evals.py"], cwd=tmp_path, capture_output=True, text=True)
    reading = json.loads(done.stdout.splitlines()[-1])
    assert reading["finished"] is True and reading["scored"] == 0, "a runner failing every case scores none"


async def test_a_recorded_agent_run_replays_offline_turn_for_turn_in_another_sandbox(tmp_path: Path) -> None:
    acts = [
        {"tool": "run", "command": "ls"},
        {"tool": "read", "path": "README.md"},
        {"tool": "write", "path": "probe.py", "text": "import pathlib\nprint(pathlib.Path.cwd())\n"},
        {"tool": "run", "command": "python probe.py"},
        {"tool": "done", "text": '{"eval_file": "probe.py"}'},
    ]

    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": [_CHEAP_ROW]})
        user = next(m["content"] for m in json.loads(request.content)["messages"] if m["role"] == "user")
        act = {"path": "", "start": 1, "text": "", "command": "", **acts[user.count("[turn ")]}
        return httpx.Response(200, json=_act_reply(json.dumps(act)))

    recording = tmp_path / "agent.jsonl"
    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    live = ModelBrain(
        lambda meter: TIERS["cheap"].model(
            recording, api_key="scripted", transport=meter, catalogue=Catalogue(http=http)
        ),
        inner=httpx.MockTransport(respond),
    )
    first = _isolated(tmp_path, "glossator", "first")
    try:
        recorded = await attempt(first, live, Caps(cost_usd=1.0, wall_s=600.0))
    finally:
        await live.aclose()
        await http.aclose()
    assert recorded.stopped == "done" and recording.is_file()

    # No key, and no client: a replay serves every call, and the meter counts no send.
    replaying = ModelBrain(lambda meter: TIERS["cheap"].model(recording, transport=meter))
    second = _isolated(tmp_path, "glossator", "second")
    replayed = await attempt(second, replaying, Caps(cost_usd=1.0, wall_s=600.0))
    assert replayed.maybe_billed_sends() == 0 and replayed.charged() == replayed.spent()
    assert replayed.stopped == "done" and replayed.report == recorded.report
    assert [(s.act, s.observation) for s in replayed.steps] == [(s.act, s.observation) for s in recorded.steps]
    assert replayed.steps[3].observation.text.splitlines()[1] == ".", "the container's path is shown as ."
    assert isinstance(second.runner, _Answering) and second.runner.ran == [[Container.python, "probe.py"]]


def test_a_run_whose_every_arm_is_recorded_replays_and_needs_no_switch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.delenv("ONRAMP_LIVE", raising=False)
    monkeypatch.delenv("RECORD", raising=False)
    (tmp_path / "docs-control.jsonl").write_text("", encoding="utf-8")
    argv = ["--arm", f"docs@control={tmp_path / 'w.whl'}", "--recordings", str(tmp_path), "--dry-run"]
    assert main(argv) == 0 and "nothing spent" in capsys.readouterr().out
    monkeypatch.setenv("RECORD", "1")  # records them again, live
    assert main(argv) == 0 and "nothing spent" not in capsys.readouterr().out
    assert main(argv[:-1]) == 2 and "ONRAMP_LIVE=1" in capsys.readouterr().err


class _Failing:
    """A brain whose first turn lists the app and whose second call fails at the provider, or never comes back."""

    model = "failing"

    def __init__(self, *, hang: bool = False) -> None:
        self.calls, self.hang = 0, hang

    async def ceiling(self, system: str, user: str) -> float | None:
        return 0.01

    async def next(self, system: str, user: str, steps: Sequence[Step]) -> Turn:
        self.calls += 1
        if self.calls == 1:
            return Turn(Act(tool="run", path="", start=1, text="", command="ls"), Usage(100, 10, 0.005))
        if self.hang:
            await asyncio.sleep(60)
        raise ProviderError("openrouter", 502, "upstream failed")


async def test_a_run_whose_model_call_fails_keeps_its_record_and_charges_the_failed_call(tmp_path: Path) -> None:
    log = tmp_path / "steps.jsonl"
    run = await attempt(_isolated(tmp_path), _Failing(), Caps(cost_usd=1.0), log=log)
    assert run.stopped == "error" and len(run.steps) == 2
    assert run.steps[-1].observation.first_error.startswith("the model call failed")
    assert run.charged() == pytest.approx(0.005 + BILLED_SENDS_AT_MOST * 0.01), (
        "the failed call at every send's ceiling"
    )
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [row["turn"] for row in rows] == [1, 2, 2] and rows[-1]["stopped"] == "error"
    assert rows[0]["act"]["command"] == "ls" and rows[0]["usage"]["cost_usd"] == 0.005


async def test_the_wall_cap_cuts_off_a_call_in_flight_and_charges_it(tmp_path: Path) -> None:
    log = tmp_path / "steps.jsonl"
    run = await attempt(_isolated(tmp_path), _Failing(hang=True), Caps(cost_usd=1.0, wall_s=0.5), log=log)
    assert run.stopped == "wall" and run.wall_s < 5
    assert run.usages[-1].cost_usd is None and run.charged() == pytest.approx(0.005 + BILLED_SENDS_AT_MOST * 0.01)
    assert json.loads(log.read_text(encoding="utf-8").splitlines()[-1]) == {"turn": 2, "stopped": "wall"}


def test_a_run_and_its_regrade_fit_inside_the_engines_cell_budget() -> None:
    assert CELL_BUDGET_S == DEFAULT_CELL_TIMEOUT_S
    # Each is cut off at its own budget, and a copy under way when it is then takes at most COPY_STOP_S to stop.
    assert Caps().wall_s + REGRADE_BUDGET_S + 2 * COPY_STOP_S < CELL_BUDGET_S


def _tar(*members: tuple[tarfile.TarInfo, bytes]) -> bytes:
    """A tar stream of ``members``, each a header and its data."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for info, data in members:
            info.size = len(data) if info.isfile() else 0
            archive.addfile(info, io.BytesIO(data) if info.isfile() else None)
    return buffer.getvalue()


def _member(name: str, kind: bytes = tarfile.REGTYPE, *, link: str = "") -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type, info.linkname, info.mode = kind, link, 0o644
    return info


def _chunked(data: bytes, size: int = 4096) -> Iterator[bytes]:
    return (data[at : at + size] for at in range(0, len(data), size))


def _unpacked(chunks: Iterator[bytes] | bytes, dest: Path, stop: threading.Event | None = None) -> str:
    """Why unpacking ``chunks`` into ``dest`` was refused; the test fails when it was not."""
    stream = _chunked(chunks) if isinstance(chunks, bytes) else chunks
    with pytest.raises(CopyRefused) as refused:
        unpack(stream, dest, stop or threading.Event(), time.monotonic() + 60)
    return str(refused.value)


def test_a_copy_back_takes_regular_files_and_directories_and_nothing_else(tmp_path: Path) -> None:
    good = _tar((_member("d", tarfile.DIRTYPE), b""), (_member("d/a.py"), b"print(1)\n"))
    (tmp_path / "ok").mkdir()
    unpack(_chunked(good), tmp_path / "ok", threading.Event(), time.monotonic() + 60)
    assert (tmp_path / "ok" / "d" / "a.py").read_bytes() == b"print(1)\n"
    refused = {
        "a symlink": _member("s", tarfile.SYMTYPE, link="a.py"),
        "a hard link": _member("h", tarfile.LNKTYPE, link="a.py"),
        "a FIFO": _member("f", tarfile.FIFOTYPE),
        "a device": _member("c", tarfile.CHRTYPE),
        "a path leaving the directory": _member("../escaped.py"),
    }
    for what, member in refused.items():
        dest = tmp_path / what.replace(" ", "-")
        dest.mkdir()
        assert _unpacked(_tar((_member("a.py"), b"x"), (member, b"y")), dest), what
    assert not (tmp_path / "escaped.py").exists()
    # An absolute path is read as relative to the directory: it lands inside it, never at the path it names.
    (tmp_path / "absolute").mkdir()
    unpack(_chunked(_tar((_member(f"/{tmp_path.name}-x.py"), b"y"))), tmp_path / "absolute", threading.Event(), 1e12)
    assert (tmp_path / "absolute" / f"{tmp_path.name}-x.py").is_file() and not Path(f"/{tmp_path.name}-x.py").exists()


def test_a_copy_back_is_held_to_its_bytes_as_they_stream_never_to_a_size_it_is_told(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A header claiming more than a whole copy (a sparse file's, which takes no room in the volume) is refused at
    once. And three files each under the bound, together over it: the copy stops when the bytes streamed pass the
    bound, having read no more than the bound and a chunk."""
    sparse = _member("sparse")
    sparse.size = 2 * 2**30
    (tmp_path / "b").mkdir()
    assert "larger than" in _unpacked(_chunked(sparse.tobuf(tarfile.GNU_FORMAT)), tmp_path / "b")
    assert not any((tmp_path / "b").iterdir())

    monkeypatch.setattr(isolation, "COPY_BYTES", 256 * 1024)
    pulled: list[int] = []

    def counted(data: bytes) -> Iterator[bytes]:
        for chunk in _chunked(data):
            pulled.append(len(chunk))
            yield chunk

    data = _tar(*((_member(f"f{n}"), bytes(100 * 1024)) for n in range(3)))
    (tmp_path / "a").mkdir()
    assert "larger than" in _unpacked(counted(data), tmp_path / "a")
    assert sum(pulled) <= 256 * 1024 + 4096


def test_a_copy_back_is_held_to_its_members_and_their_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(isolation, "COPY_MEMBERS", 5)
    (tmp_path / "many").mkdir()
    assert "more than 5" in _unpacked(_tar(*((_member(f"f{n}"), b"") for n in range(6))), tmp_path / "many")
    (tmp_path / "long").mkdir()
    assert "longer than" in _unpacked(_tar((_member("d/" * 520 + "x"), b"")), tmp_path / "long")


def test_a_stopped_copy_stops_at_its_next_chunk(tmp_path: Path) -> None:
    stop = threading.Event()
    data = _tar((_member("a"), bytes(64 * 1024)))

    def stopped_after_one() -> Iterator[bytes]:
        for at, chunk in enumerate(_chunked(data)):
            if at == 1:
                stop.set()
            yield chunk

    (tmp_path / "x").mkdir()
    assert "stopped" in _unpacked(stopped_after_one(), tmp_path / "x", stop)


async def test_a_cancelled_copy_has_stopped_before_its_cancellation_ends() -> None:
    """The wall cap cancels the run: the copy's thread is told, and the cancellation waits for it to end."""
    ended = threading.Event()

    def copy(stop: threading.Event) -> None:
        try:
            while not stop.wait(0.01):
                pass
            raise CopyRefused("the copy was stopped")
        finally:
            ended.set()

    task = asyncio.create_task(stoppable(copy))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ended.is_set()


def test_a_copy_in_is_held_to_the_same_bounds_and_takes_no_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = tmp_path / "app"
    app.mkdir()
    (app / "a.py").write_text("x", encoding="utf-8")
    (app / "link").symlink_to(app / "a.py")
    with pytest.raises(CopyRefused, match="link"):
        pack(app, io.BytesIO())
    (app / "link").unlink()
    monkeypatch.setattr(isolation, "COPY_BYTES", 64 * 1024)
    (app / "big").write_bytes(bytes(100 * 1024))
    with pytest.raises(CopyRefused, match="larger than"):
        pack(app, io.BytesIO())
    (app / "big").unlink()
    out = io.BytesIO()
    pack(app, out)
    assert tarfile.open(fileobj=io.BytesIO(out.getvalue())).getnames() == ["a.py"]


class _Refusing(_Answering):
    """A runner whose every command ran and left more than may be copied back."""

    async def run(self, argv: Sequence[str], *, cwd: Path, timeout_s: float, grading: bool = False) -> Ran:
        return Ran(0, "made it\n", "", WORK, "what the command left could not be copied back: too large")


async def test_a_command_whose_directory_cannot_come_back_fails_saying_why_and_is_never_replayed(
    tmp_path: Path,
) -> None:
    sandbox = Sandbox.create(tmp_path / "app", "recipes", _build(tmp_path), _Refusing())
    observation = await sandbox.run("python make.py")
    assert not observation.ok and observation.exit_code is None
    assert observation.first_error == "what the command left could not be copied back: too large"
    assert observation.text.startswith("refused: what the command left") and "made it" in observation.text


def test_a_read_never_holds_more_of_a_file_than_it_takes(tmp_path: Path) -> None:
    sandbox = Sandbox.create(tmp_path / "app", "recipes", _build(tmp_path))
    (sandbox.root / "big.txt").write_bytes(b"x\n" * (READ_BYTES // 2 + 1))
    observation = sandbox.read("big.txt")
    assert not observation.ok and "larger than" in observation.first_error
    assert sandbox.read("kitchen.py").ok


def test_the_field_an_answer_lacks_is_found_whatever_the_field_is_called() -> None:
    said = "error: case 2: the candidate answered no expired it can be graded on"
    assert errors_hit([_step(1, "run", False, said, "python -m threetears.evals e.py", "x")]) == {
        "field_not_in_answer": 1
    }


def test_only_a_door_run_that_worked_ends_a_stuck_span() -> None:
    door = "python -m threetears.evals e.py"
    steps = [
        _step(1, "run", False, command=door, first_error="opened it"),
        _step(2, "run", True, command="ls"),
        _step(3, "run", True, command="python probe.py"),
        _step(4, "run", False, command=door, first_error="again"),
        _step(5, "run", True, command=door),
        _step(6, "run", False, command=door, first_error="later"),
    ]
    assert stuck_point(steps) == {
        "from_turn": 1, "to_turn": 4, "failed_turns": 2, "opened_by": "opened it", "cause": "cannot tell"
    }  # fmt: skip


def _facts(app: float, twin: float, higher_is_better: bool | None) -> dict[str, Any]:  # Any: the regrade's facts
    reading = {"misses": higher_is_better}
    return {
        "checks": {
            "problem": "",
            "app": {"means": {"misses": app}, "higher_is_better": reading, "nothing_unpassed": 1.0},
            "twin": {"means": {"misses": twin}, "higher_is_better": reading},
        }
    }


def test_measuring_the_app_reads_the_measures_own_direction() -> None:
    assert measures_the_app({}, _facts(0.4, 0.0, higher_is_better=False)), "fewer misses on the fixed twin"
    assert not measures_the_app({}, _facts(0.0, 0.4, higher_is_better=False))
    assert measures_the_app({}, _facts(0.6, 1.0, higher_is_better=True))
    assert not measures_the_app({}, _facts(0.6, 1.0, higher_is_better=None)), "no stated direction counts for neither"


def test_the_census_states_each_checks_tally_with_its_exact_range() -> None:
    assert clopper_pearson(0, 9) == (0.0, pytest.approx(0.3363, abs=1e-4))
    assert clopper_pearson(3, 9) == (pytest.approx(0.0749, abs=1e-4), pytest.approx(0.7007, abs=1e-4))
    assert clopper_pearson(9, 9) == (pytest.approx(0.6637, abs=1e-4), 1.0)
    rows = [{"scores": {"eval_runs": float(i < 3), "measures_the_app": 0.0, "read_right": 0.0}} for i in range(9)]
    lines = census(rows)
    assert any(line.startswith("  eval_runs: 3 of 9 trials, 95% range 0.07 to 0.70; 6 failed") for line in lines)
    assert any(line.startswith("  measures_the_app: 0 of 9 trials, 95% range 0.00 to 0.34; 9 failed") for line in lines)
    assert any("cannot show" in line for line in lines)


def test_the_census_never_pools_arms_and_tallies_each_check_app_by_app() -> None:
    """Two arms are tallied apart, each line headed by its arm; with more than one app, each line ends with its tally
    by app, the unit a claim beyond these apps rests on. One arm on one app prints as before."""

    def row(arm: str, app: str, passed: bool) -> dict[str, Any]:
        scores = {"eval_runs": 1.0, "measures_the_app": float(passed), "read_right": 1.0}
        return {"arm": arm, "scores": scores, "record": {"app": app, "stopped": "done"}}

    rows = [
        row("docs@control", "holds", False), row("docs@control", "holds", False), row("docs@control", "recipes", True),
        row("docs@candidate", "holds", True), row("docs@candidate", "recipes", True), row("docs@candidate", "recipes", True),
    ]  # fmt: skip
    lines = census(rows)
    measured = [line for line in lines if "measures_the_app:" in line]
    assert measured == [
        "  docs@control measures_the_app: 1 of 3 trials, 95% range 0.01 to 0.91; 2 failed: 2 cannot tell; "
        "by app: holds 0 of 2, recipes 1 of 1",
        "  docs@candidate measures_the_app: 3 of 3 trials, 95% range 0.29 to 1.00; by app: holds 1 of 1, recipes 2 of 2",
    ], measured
    assert not any(line.startswith("  measures_the_app: ") for line in lines), "no line pools the two arms"
    one = census([row("docs@control", "holds", False), row("docs@control", "holds", False)])
    assert "  measures_the_app: 0 of 2 trials, 95% range 0.00 to 0.84; 2 failed: 2 cannot tell" in one, one
    assert any("trials on one app are repeats of that app" in line for line in lines)


def test_pytest_collects_nothing_from_the_run_directories(pytester: pytest.Pytester) -> None:
    """Agents write ``test_*.py`` into the copies of the apps under ``runs/``; the harness's conftest keeps a sweep of
    the tree from collecting them, and collects the rest of it as before."""
    onramp = pytester.mkpydir("onramp")
    shutil.copy(Path(__file__).parents[1] / "onramp" / "conftest.py", onramp / "conftest.py")
    trial = onramp / "runs" / "20261116T030232Z" / "docs-control" / "holds-1" / "app"
    trial.mkdir(parents=True)
    (trial / "test_eval.py").write_text("def test_an_agents_file():\n    raise AssertionError('collected')\n")
    (onramp / "test_kept.py").write_text("def test_kept():\n    pass\n")
    pytester.runpytest_inprocess(str(onramp), "-p", "no:cacheprovider").assert_outcomes(passed=1)


def _worst_coverage(n: int, interval: Callable[[int, int], tuple[float, float]]) -> float:
    """The least often ``interval`` covers the true share, over shares 0.01 to 0.99: at each, every outcome of ``n``
    trials weighed by its chance."""
    ranges = [interval(k, n) for k in range(n + 1)]
    return min(
        sum(
            math.comb(n, k) * share**k * (1 - share) ** (n - k)
            for k, (low, high) in enumerate(ranges)
            if low <= share <= high
        )
        for share in (i / 100 for i in range(1, 100))
    )


def _wald(k: int, n: int) -> tuple[float, float]:
    p = k / n
    half = 1.96 * math.sqrt(p * (1 - p) / n)
    return p - half, p + half


def test_the_census_range_covers_the_true_share_at_least_95_percent_of_the_time() -> None:
    """Exact coverage, every outcome weighed (no sampling error): the Clopper-Pearson range holds 95% at every share on
    a grid and at the n a baseline has; the normal-approximation range, a wrong formula here, does not."""
    for n in (3, 9, 27):
        worst = _worst_coverage(n, clopper_pearson)
        assert worst >= 0.95, (n, worst)
    assert _worst_coverage(9, _wald) < 0.90


# --- the harness's own rules, after the baseline (H1 to H8) -----------------------------------------------------------

_REPORT = {"eval_file": "e.py", "first_line": "x: x passed 1 of 2 cases.", "measure": "grader", "pass_rate": 0.5,
           "finished": True}  # fmt: skip


def test_a_report_given_bare_or_in_the_wrong_field_is_taken_and_an_ambiguous_reply_is_refused_saying_the_shape() -> (
    None
):
    """H1 and H2: where the intent is plain the reply is read as meant; where it is not, nothing is guessed and the
    refusal names what was wrong and the shape."""
    act, problem, how = read_reply(json.dumps(_REPORT))
    assert act is not None and act.tool == "done" and json.loads(act.text) == _REPORT and how == "bare report"
    act, _, how = read_reply(json.dumps({"tool": "done", **_REPORT}))
    assert act is not None and act.tool == "done" and how == "bare report"
    # A field the tool does not read may be left out; one it reads may not.
    act, problem, _ = read_reply('{"tool": "write", "path": "e.py", "text": "x = 1"}')
    assert act is not None and (act.tool, act.path, act.text, act.command, act.start) == (
        "write",
        "e.py",
        "x = 1",
        "",
        1,
    )
    act, problem, _ = read_reply('{"tool": "write", "path": "e.py"}')
    assert act is None and problem.startswith("write reads text, which the reply left out") and '"tool"' in problem
    refused = {
        "two calls": '{"tool": "run", "command": "ls"}\n{"tool": "run", "command": "ls"}',
        "a report beside another tool": json.dumps({"tool": "run", "command": "ls", **_REPORT}),
        "a field no tool has": '{"tool": "run", "command": "ls", "cwd": "."}',
        "no tool": '{"tool": "edit", "path": "e.py"}',
        "no JSON": "I will now run the eval.",
        "a wrong type": '{"tool": "read", "path": "e.py", "start": "top"}',
    }
    for what, reply in refused.items():
        act, problem, _ = read_reply(reply)
        assert act is None and "send one JSON object a turn" in problem, what
    assert "2 JSON values" in read_reply(refused["two calls"])[1]
    assert "cwd is no field" in read_reply(refused["a field no tool has"])[1]
    assert "start" in read_reply(refused["a wrong type"])[1]

    text = json.dumps(_REPORT)
    blank = {"tool": "done", "path": "", "start": 1}
    assert done_report(Act(**blank, text=text, command="")) == (text, "text")
    assert done_report(Act(**blank, text="", command=text)) == (text, "command")
    assert done_report(Act(**blank, text=text, command='{"other": 1}')) == (text, "text"), "the asked field wins"
    assert done_report(Act(**blank, text="", command="ls")) == ("", "text")


async def test_a_bare_report_ends_the_run_and_the_record_says_where_it_was_read_from(tmp_path: Path) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": [_CHEAP_ROW]})
        return httpx.Response(200, json=_act_reply(json.dumps(_REPORT)))

    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    brain = ModelBrain(
        lambda meter: TIERS["cheap"].model(None, api_key="scripted", transport=meter, catalogue=Catalogue(http=http)),
        inner=httpx.MockTransport(respond),
    )
    try:
        run = await attempt(_isolated(tmp_path), brain, Caps(cost_usd=1.0), log=tmp_path / "steps.jsonl")
    finally:
        await brain.aclose()
        await http.aclose()
    assert run.stopped == "done" and run.report_from == "bare report" and json.loads(run.report or "") == _REPORT
    logged = json.loads((tmp_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert json.loads(logged["reply"]) == _REPORT, "the reply as given is kept, so the run can be read again"


def test_the_agent_is_told_its_working_directory_and_its_token_limit_and_the_tier_holds_to_it() -> None:
    """H3 and H6: what the sandbox is, said as it is."""
    assert "Your working directory is the app's directory, shown as `.`" in SYSTEM
    assert "{tokens}" in SYSTEM and REPLY_TOKENS >= 16_384


def test_a_read_past_the_end_fails_and_says_how_long_the_file_is(tmp_path: Path) -> None:
    """H4."""
    sandbox = Sandbox.create(tmp_path / "app", "recipes", _build(tmp_path))
    lines = len((sandbox.root / "kitchen.py").read_text(encoding="utf-8").splitlines())
    observation = sandbox.read("kitchen.py", lines + 50)
    assert (
        not observation.ok
        and observation.first_error == f"kitchen.py has {lines} lines; start at a line from 1 to {lines}"
    )
    assert sandbox.read("kitchen.py", lines).ok


def _run_facts(**app: Any) -> dict[str, Any]:  # Any: the regrade's facts
    reading = {"readings": ["accuracy", "slug"], "finished": True, "scored": 3,
               "means": {"accuracy": 0.6667, "slug": 1.0}, **app}  # fmt: skip
    return {
        "problem": "",
        "door_exit": 0,
        "stored": True,
        "door_first_line": "g: r passed 2 of 3 cases.",
        "app": reading,
    }


def test_an_eval_that_scored_nothing_does_not_run_and_one_that_scored_any_case_does() -> None:
    """H7: a runner that failed every case finished a run that measured nothing."""
    assert eval_runs({}, {"checks": _run_facts()})
    assert eval_runs({}, {"checks": _run_facts(means={"accuracy": 0.0, "slug": None}, scored=1)})
    assert not eval_runs({}, {"checks": _run_facts(scored=0)})


def test_the_pass_rate_is_read_against_the_measure_named_or_the_evals_only_one() -> None:
    """H8: the report names the measure it reads, as the console printed it or as the eval declared it, and is held to
    that measure's value. Unnamed (or named as none the run printed) it is read against the eval's one measure, and of
    an eval with several it is right against none: a value that equals another measure by chance is not a reading."""
    line = "g: r passed 2 of 3 cases."
    report = {"first_line": line, "finished": True}

    def reads(facts: dict[str, Any] | None = None, **given: Any) -> bool:
        return read_right({}, {"report": {**report, **given}, "checks": facts or _run_facts()})

    assert reads(pass_rate=0.6667, measure="Accuracy") and reads(pass_rate=1.0, measure="Slug score")
    assert not reads(pass_rate=1.0), "unnamed, of two measures: 1.0 is slug's by coincidence, a case share misread"
    assert not reads(pass_rate=0.667, measure="headword"), "a name the run did not print, of two measures"
    assert not reads(pass_rate=1.0, measure="accuracy"), "named, and another measure's value given"
    assert not reads(pass_rate=0.6667, measure="accuracy", first_line="g: r passed 2 of 3"), "the line cut short"
    one = _run_facts(readings=["grader"], means={"grader": 0.6})
    assert reads(one, pass_rate=0.6) and reads(one, pass_rate=0.6, measure="Grader score"), (
        "one measure: no name needed"
    )
    assert not reads(one, pass_rate=1.0), "a case share read off the first line, against a grader's 0.60"


def test_what_the_harness_caused_is_told_apart_where_it_can_be_told_mechanically() -> None:
    """Item 2: a format error, a sandbox path, a refused command or a limit is the harness's; a file the guide names
    and the build lacks is the docs'; anything else cannot be told without reading the run."""
    guide = "THE FIRST EXAMPLE (1-triage/triage.py of the examples fetched below)"
    refused_reply = Step(3, None, Observation(ok=False, text="the reply was not one tool call: ..."))
    assert cause(refused_reply) == "harness"
    assert cause(_step(2, "run", False, command="ls /app", first_error="ls: /app is not in the app's directory or the "
                       "installed package")) == "harness"  # fmt: skip
    assert cause(_step(4, "read", False, first_error="examples/1-triage/triage.py: no such file"), guide) == "docs"
    assert cause(_step(4, "read", False, first_error="tests/__init__.py: no such file"), guide) == "cannot tell"
    assert cause(_step(5, "run", False, command="python -m threetears.evals e.py", first_error="KeyError: 'id'")) == (
        "cannot tell"
    )
    assert stuck_point([refused_reply])["cause"] == "harness"  # type: ignore[index]

    door = {"turn": 6, "command": "python -m threetears.evals e.py", "door": True}
    failed = {"eval_runs": 0.0, "measures_the_app": 0.0, "read_right": 0.0}
    # The wall is the harness's when the run had a working eval it had no time to report; else cannot tell.
    walled = {"scores": failed, "record": {"stopped": "wall", "commands": [door], "scored_door_turns": [6]}}
    assert failure_causes(walled) == dict.fromkeys(failed, "harness")
    stuck = {"scores": failed, "record": {"stopped": "wall", "commands": [door], "scored_door_turns": []}}
    assert failure_causes(stuck) == dict.fromkeys(failed, "cannot tell")
    made_up = {"scores": failed, "record": {"stopped": "done", "commands": [], "checks": {
        "eval_file": "evals.py", "problem": "evals.py is not in the app's directory"}}}  # fmt: skip
    assert failure_causes(made_up) == dict.fromkeys(failed, "agent"), "a report sent with no eval written"
    crashed = {"scores": failed, "record": {"stopped": "done", "commands": [door], "report": {}, "checks": {
        "problem": "", "door_exit": 0, "app": {"refused": "Traceback (most recent call last):"}}}}  # fmt: skip
    assert failure_causes(crashed) == dict.fromkeys(failed, "harness"), "the regrade crashed on an eval the door ran"
    lost = {"scores": {**failed, "eval_runs": 1.0}, "record": {"stopped": "done", "commands": [door], "report": None,
            "format_error_turns": [7]}}  # fmt: skip
    assert failure_causes(lost) == {"measures_the_app": "cannot tell", "read_right": "harness"}
    early = {**lost, "record": {**lost["record"], "format_error_turns": [3]}}
    assert failure_causes(early)["read_right"] == "cannot tell", (
        "a refused reply before the last door run lost no report"
    )

    rows = [walled, lost, {"scores": dict.fromkeys(failed, 1.0), "record": {"stopped": "done"}}]
    lines = census(rows)
    assert any(line.startswith("  read_right: 1 of 3 trials") and "2 failed: 2 harness" in line
               and "1 of 1 the harness failed none of" in line for line in lines), lines  # fmt: skip


def test_a_stored_run_is_read_again_with_todays_rules_and_ends_where_a_report_was_first_given() -> None:
    """Item 3's reading: a bare report refused then is the done it meant now; a command after it is named."""
    steps = [
        {"turn": 1, "act": {**_BLANK_ACT, "tool": "run", "command": "python -m threetears.evals e.py"}},
        {"turn": 2, "act": None},
        {"turn": 3, "act": None},
        {"turn": 4, "act": {**_BLANK_ACT, "tool": "done", "text": ""}},
    ]
    replies = ['{"tool": "run"}', "not json", json.dumps(_REPORT), "{}"]
    again = read_again(steps, replies)
    assert again["ended"] == 3 and again["report_from"] == "bare report" and json.loads(again["report"]) == _REPORT
    assert again["refused"] == [2] and again["unmade"] == []
    command = {**_BLANK_ACT, "tool": "done", "text": "", "command": json.dumps(_REPORT)}
    assert read_again([{"turn": 1, "act": command}], [])["report_from"] == "command"


_BLANK_ACT: dict[str, Any] = {"path": "", "start": 1, "text": "", "command": ""}  # Any: an act's fields


def test_a_build_whose_results_lack_a_field_the_reading_needs_is_refused_by_name_and_fails_the_pre_flight(
    tmp_path: Path,
) -> None:
    """Item 2 of the Critic's round 3: a wheel whose results carry no ``outcome`` is read as an error naming the build
    and the field, never as zero scored; that reading fails ``eval_runs``, and the pre-flight refuses its arm by name."""
    shaped = tmp_path / "wheel" / "threetears" / "evals"
    shaped.mkdir(parents=True)
    (shaped / "__init__.py").write_text(
        "from dataclasses import dataclass\n\n\n"
        "@dataclass\nclass Summary:\n    status: str = 'completed'\n\n\n"
        "@dataclass\nclass Measure:\n    name: str\n    mean: float\n\n\n"
        "@dataclass\nclass Arm:\n    measures: list\n    status: str = 'completed'\n\n\n"
        "@dataclass\nclass Trial:\n    case: str\n    state: str = 'passed'\n    scores: dict | None = None\n\n\n"
        "class Result:\n"
        "    name, control, grader_readings = 'renamed', 'runner', ['ok']\n"
        "    arms = {'runner': Arm([Measure('ok', 1.0)])}\n\n"
        "    def results(self, arm=None):\n"
        "        return [Trial('0', scores={'ok': True})]\n\n\n"
        "class Eval:\n"
        "    arms = {'runner': None}\n\n"
        "    async def run(self, store=None):\n"
        "        return Result()\n",
        encoding="utf-8",
    )
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "evals.py").write_text("from threetears.evals import Eval\n\nev = Eval()\n", encoding="utf-8")
    environment = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(tmp_path / "wheel")}
    done = subprocess.run(
        [sys.executable, str(RERUN), "evals.py"], cwd=tmp_path / "work", capture_output=True, text=True, env=environment
    )
    reading = json.loads(done.stdout.splitlines()[-1])
    assert done.returncode == 2 and "Trial.outcome" in reading["refused"], reading
    assert str(tmp_path / "wheel" / "threetears" / "evals") in reading["refused"], "the build is named by where it is"

    facts = {"problem": "", "door_exit": 0, "stored": True, "door_first_line": "x", "app": reading}
    assert not eval_runs({}, {"checks": facts})
    rows = [
        {"scores": {"eval_runs": 0.0, "measures_the_app": 0.0, "read_right": 0.0},
         "record": {"app": app, "checks": facts}}
        for app in ("recipes", "holds", "glossator")
    ]  # fmt: skip
    refused = preflight_refusal("docs@renamed", rows)
    assert refused is not None and "arm docs@renamed" in refused and "0 of 3 apps" in refused
    assert "Trial.outcome" in refused, "the reason is named, not only the count"
    passing = [{**row, "scores": dict.fromkeys(row["scores"], 1.0)} for row in rows]
    assert preflight_refusal("docs@control", passing) is None
    assert preflight_refusal("docs@control", passing[:2]) is not None, "every app, not two of three"


async def test_a_run_logs_its_directorys_state_after_every_turn_that_may_change_it(tmp_path: Path) -> None:
    """So a run can be graded again as it stood at any turn (item 1 of the Critic's round 3)."""
    sandbox = Sandbox.create(tmp_path / "app", "recipes", _build(tmp_path))
    script = tmp_path / "script.jsonl"
    acts = [
        {"tool": "write", "path": "a.py", "text": "print(1)\n"},
        {"tool": "read", "path": "a.py"},
        {"tool": "run", "command": "python a.py"},
        {"tool": "write", "path": "a.py", "text": "print(2)\n"},
    ]
    script.write_text("".join(json.dumps(act) + "\n" for act in acts), encoding="utf-8")
    await attempt(sandbox, ScriptedBrain(script, sandbox), Caps(), log=tmp_path / "steps.jsonl")
    rows = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    logged = {row["turn"]: row.get("files") for row in rows if "observation" in row}
    assert logged[2] is None, "a read changes nothing"
    assert logged[1] is not None and logged[3] is not None and logged[4] is not None
    first, last = logged[1]["a.py"], logged[4]["a.py"]
    assert first != last and last == sandbox.manifest()["a.py"]


# --- the comparison's harness causes (analysis 1116, section 5) --------------------------------------------------------


async def test_ls_takes_its_flags_and_several_paths_and_refuses_other_flags_naming_those_it_takes(
    tmp_path: Path,
) -> None:
    sandbox = Sandbox.create(tmp_path / "app", "glossator", _build(tmp_path))
    listing = await sandbox.run("ls -la")
    assert listing.ok and "tests/" in listing.text and "glossator.py" in listing.text
    sized = (sandbox.root / "glossator.py").stat().st_size
    assert any(
        line.startswith("- ") and line.endswith("glossator.py") and str(sized) in line
        for line in listing.text.splitlines()
    )
    both = await sandbox.run("ls . tests")
    assert both.ok and ".:\n" in both.text and "tests:\nfixtures/" in both.text, both.text
    assert (await sandbox.run("ls -1 tests/fixtures")).ok
    refused = await sandbox.run("ls -R")
    assert not refused.ok and refused.first_error == "ls: -R is not taken; ls takes -a, -l, -1"
    outside = await sandbox.run("ls /app")
    assert not outside.ok and "is not in the app's directory" in outside.first_error
    mixed = await sandbox.run("ls tests /etc")
    assert not mixed.ok and "fixtures/" in mixed.text, "what can be listed is listed, and what cannot is said"


def test_reading_a_directory_says_it_is_one(tmp_path: Path) -> None:
    sandbox = Sandbox.create(tmp_path / "app", "glossator", _build(tmp_path))
    observation = sandbox.read("tests")
    assert not observation.ok and observation.first_error == "tests is a directory: `ls tests` lists it"
    assert "is a directory" in sandbox.read("tests/fixtures").first_error


def test_an_eval_kept_in_a_subdirectory_is_re_run_from_the_apps_directory_as_the_door_runs_it(tmp_path: Path) -> None:
    """The re-grade crashed on a right eval kept in tests/ (cand glossator r3): it ran from the file's directory, where
    the app does not import. It now runs from the app's, as the door does."""
    app = tmp_path / "app"
    (app / "tests" / "fixtures").mkdir(parents=True)
    (app / "shop.py").write_text("def price(case):\n    return case['n'] * 2\n", encoding="utf-8")
    (app / "tests" / "fixtures" / "cases.jsonl").write_text('{"n": 1, "want": 2}\n{"n": 2, "want": 5}\n', "utf-8")
    (app / "tests" / "eval_shop.py").write_text(
        "from threetears.evals import Eval\n\nfrom shop import price\n\n\n"
        "def right(case, out) -> bool:\n"
        '    """The price is the one wanted."""\n'
        "    return out == case['want']\n\n\n"
        'shop = Eval("fixtures/cases.jsonl", runner=price, grader=[right], k=1)\n',
        encoding="utf-8",
    )
    done = subprocess.run(
        [sys.executable, str(RERUN), "tests/eval_shop.py"], cwd=app, capture_output=True, text=True, check=True
    )
    reading = json.loads(done.stdout.splitlines()[-1])
    assert reading["means"] == {"right": 0.5} and reading["scored"] == 2, reading


async def test_the_agent_is_told_how_long_it_has_and_warned_before_the_wall(tmp_path: Path) -> None:
    assert "The task stops after {wall:.0f} seconds or {turns} turns" in SYSTEM
    assert "[time]" not in prompt([]) and "[time] 42 seconds are left" in prompt([], 42.4)
    asked: list[str] = []

    class Slow:
        model = "slow"

        async def ceiling(self, system: str, user: str) -> float | None:
            return 0.0

        async def next(self, system: str, user: str, steps: Sequence[Step]) -> Turn:
            asked.append(user)
            await asyncio.sleep(0.25)
            return Turn(Act(tool="run", path="", start=1, text="", command="ls"), Usage(1, 1, 0.0))

    run = await attempt(_isolated(tmp_path), Slow(), Caps(wall_s=1.5))
    assert run.stopped == "wall"
    assert "[time]" not in asked[0] and any("[time]" in user for user in asked[-2:]), "warned past 80% of the wall"


def _door_step(turn: int, text: str, ok: bool = True) -> Step:
    return _step(turn, "run", ok, text=text, command="python -m threetears.evals e.py")


def test_a_door_run_that_scored_nothing_is_a_failed_turn_and_the_stuck_point_names_it() -> None:
    """H10 of the analysis: a door run exiting 0 with every case failed by the candidate is where a run is stuck."""
    nothing = "exit 0\n3 result(s): 0 scored, 3 failed by the candidate\nKeyError: 'id'"
    steps = [
        _step(1, "read", False, first_error="surface.md: no such file"),
        _step(2, "read", True),
        _door_step(3, nothing),
        _door_step(4, nothing),
        _door_step(5, nothing),
        _door_step(6, "exit 0\n3 result(s): 3 scored"),
    ]
    stuck = stuck_point(steps)
    assert stuck is not None and (stuck["from_turn"], stuck["to_turn"], stuck["failed_turns"]) == (1, 5, 4)
    assert scored_door_turns(steps) == [6]


def test_a_door_run_that_exits_1_having_scored_nothing_fails_and_is_named_by_its_first_line() -> None:
    """Builds from 2026-11-16 exit 1 when the runner raised on every case and say so on the first line, to stdout: the
    turn fails on its exit code, and the stuck point names that line, not the ``exit 1`` header."""
    first = "scaling: scale failed on all 10 cases, so no answer was graded: the runner raised RuntimeError: boom."
    said = f"exit 1\n{first}\n10 result(s): 0 scored, 10 failed by the candidate"
    steps = [
        _door_step(1, said, ok=False),
        _door_step(2, said, ok=False),
        _door_step(3, "exit 0\n3 result(s): 3 scored"),
    ]
    stuck = stuck_point(steps)
    assert stuck is not None and (stuck["from_turn"], stuck["failed_turns"], stuck["opened_by"]) == (1, 2, first)
    assert scored_door_turns(steps) == [3]
    assert errors_hit(steps) == {"runner_raised": 1}


def test_the_doors_help_neither_scores_nor_ends_a_stuck_span() -> None:
    """``python -m threetears.evals --help`` exits 0 and runs no eval: it is the agent looking for help, so a wall stop
    after it is not a run that had a working eval (run 20261116T030232Z counted two such turns as scored)."""
    nothing = "exit 0\n3 result(s): 0 scored, 3 failed by the candidate"
    helped = _step(3, "run", True, "exit 0\nusage: python -m threetears.evals ...", "python -m threetears.evals --help")
    steps = [_door_step(1, nothing), _door_step(2, nothing), helped, _door_step(4, nothing)]
    assert scored_door_turns(steps) == []
    stuck = stuck_point(steps)
    assert stuck is not None and (stuck["from_turn"], stuck["to_turn"], stuck["failed_turns"]) == (1, 4, 3)


def test_errors_hit_reads_the_wording_of_every_build_measured() -> None:
    """The control build says "the candidate raised" and "the scorer X raised"; builds from 2026-11-16 say "the runner
    raised" on their first line and "the grader X raised" on their error lines."""
    for runner, grader in (
        ("error: case 0: the candidate raised KeyError: 'x'", "the scorer right raised KeyError: 'x'"),
        ("s: r failed on all 3 cases, so no answer was graded: the runner raised KeyError: 'x'.", "the grader right raised KeyError: 'x'"),
    ):  # fmt: skip
        steps = [_door_step(1, f"exit 1\n{runner}", ok=False), _door_step(2, f"exit 1\n{grader}", ok=False)]
        assert errors_hit(steps) == {"runner_raised": 1, "grader_raised": 2}


def test_the_spend_line_states_the_charged_bound_beside_the_reported_cost() -> None:
    rows = [{"record": {"cost_usd": 0.01, "charged_usd": 0.01}}, {"record": {"cost_usd": 0.02, "charged_usd": 0.05}}]
    assert (
        spend_line(rows)
        == "spend: $0.030 reported by the provider; at most $0.060 billed (each cut-off call at its ceiling)"
    )
    assert "unpriced" in spend_line([{"record": {"cost_usd": None, "charged_usd": 0.05}}])
