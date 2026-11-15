"""The on-ramp's agent test, piece by piece, offline: the apps and their twins, the sandbox, the run's caps, what a run
records, and the live path's guards. ``test_onramp_end_to_end.py`` runs the whole harness."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

from threetears.evals.providers import Catalogue
from packages.evals.onramp.agent import (
    Act,
    Caps,
    ModelBrain,
    ScriptedBrain,
    Step,
    Turn,
    Usage,
    attempt,
    errors_hit,
    parse_report,
    stuck_point,
)
from packages.evals.onramp.checks import RERUN
from packages.evals.onramp.campaign import CONTROL_REF, TIERS, Tier, live_refusal, main, quote
from packages.evals.onramp.sandbox import APPS, TWINS, DocsBuild, Observation, Sandbox

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


def test_a_command_runs_with_no_key_from_the_harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-key")
    environment = Sandbox.create(tmp_path / "app", "holds", _build(tmp_path)).environment()
    assert set(environment) == {"PATH", "HOME", "LANG", "PYTHONUTF8", "PYTHONDONTWRITEBYTECODE"}


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
    assert stuck_point(steps) == {"from_turn": 3, "to_turn": 6, "failed_turns": 3, "opened_by": "opened it"}
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
    """A brain whose every turn lists the app and costs ``each``, bounded by ``bound`` (``None``: unbounded)."""

    model = "priced"

    def __init__(self, each: float | None, bound: float | None) -> None:
        self.each, self.bound = each, bound

    async def ceiling(self, system: str, user: str) -> float | None:
        return self.bound

    async def next(self, system: str, user: str, steps: Sequence[Step]) -> Turn:
        act = Act(tool="run", path="", start=1, text="", command="ls")
        return Turn(act, Usage(100, 10, self.each))


async def test_a_run_stops_before_a_turn_that_could_take_it_past_its_cap(tmp_path: Path) -> None:
    sandbox = Sandbox.create(tmp_path / "app", "recipes", _build(tmp_path))
    run = await attempt(sandbox, _Priced(0.10, 0.10), Caps(cost_usd=0.35))
    assert run.stopped == "cap" and len(run.usages) == 3, "0.30 spent, and a fourth turn could reach 0.40"
    assert run.spent() == pytest.approx(0.30) and run.tokens("input_tokens") == 300


async def test_a_turn_no_price_can_bound_is_never_sent_under_a_cap(tmp_path: Path) -> None:
    sandbox = Sandbox.create(tmp_path / "app", "recipes", _build(tmp_path))
    run = await attempt(sandbox, _Priced(0.01, None), Caps(cost_usd=1.0))
    assert run.stopped == "cap" and run.usages == []


async def test_an_unpriced_turn_makes_the_run_unpriced_and_stops_it_under_a_cap(tmp_path: Path) -> None:
    sandbox = Sandbox.create(tmp_path / "app", "recipes", _build(tmp_path))
    run = await attempt(sandbox, _Priced(None, 0.01), Caps(cost_usd=1.0))
    assert run.spent() is None and run.stopped == "cap" and len(run.usages) == 1


async def test_a_run_stops_at_its_turn_cap(tmp_path: Path) -> None:
    sandbox = Sandbox.create(tmp_path / "app", "recipes", _build(tmp_path))
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
    model = TIERS["cheap"].model(None, api_key="scripted", http=http, catalogue=Catalogue(http=http))
    sandbox = Sandbox.create(tmp_path / "app", "recipes", _build(tmp_path))
    try:
        run = await attempt(sandbox, ModelBrain(model), Caps(cost_usd=0.004))
    finally:
        await http.aclose()
    assert run.stopped == "cap" and sent, "it ran turns, then stopped before one that could pass the cap"
    spent = run.spent()
    assert spent is not None and spent == pytest.approx(0.00014 * len(sent)) and spent <= 0.004
    assert run.tokens("input_tokens") == 1000 * len(sent)
    assert all("temperature" not in body for body in sent)


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

    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    model = TIERS["ceiling"].model(None, api_key="scripted", http=http)
    sandbox = Sandbox.create(tmp_path / "app", "holds", _build(tmp_path))
    try:
        run = await attempt(sandbox, ModelBrain(model), Caps(cost_usd=1.50))
    finally:
        await http.aclose()
    assert run.stopped == "done" and hosts == ["api.anthropic.com"]
    assert "temperature" not in bodies[0]
    assert run.spent() == pytest.approx(2000 * 0.10 / 1e6 + 50 * 0.50 / 1e6), "Anthropic's published rates"


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
    live = TIERS["cheap"].model(recording, api_key="scripted", http=http, catalogue=Catalogue(http=http))
    first = Sandbox.create(tmp_path / "first", "glossator", _build(tmp_path))
    try:
        recorded = await attempt(first, ModelBrain(live), Caps(cost_usd=1.0, wall_s=600.0))
    finally:
        await live.aclose()
        await http.aclose()
    assert recorded.stopped == "done" and recording.is_file()

    replaying = TIERS["cheap"].model(recording)  # no key, no client: a replay serves every call
    second = Sandbox.create(tmp_path / "second", "glossator", _build(tmp_path))
    replayed = await attempt(second, ModelBrain(replaying), Caps(cost_usd=1.0, wall_s=600.0))
    assert replayed.stopped == "done" and replayed.report == recorded.report
    assert [(s.act, s.observation) for s in replayed.steps] == [(s.act, s.observation) for s in recorded.steps]
    assert replayed.steps[3].observation.text.splitlines()[1] == ".", "the sandbox's path is shown as ."


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
