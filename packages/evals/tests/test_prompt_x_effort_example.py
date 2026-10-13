"""``examples/prompt_x_effort.py`` runs a 2x2 as a newcomer would, with no API key, and reads both factors.

With no key the example replays its recording: four arms, the prompt crossed with the reasoning effort, each effort
one client replaying its own calls from the one file. The arms are read by their keys, never by the words a report
names them by, and nothing here assumes what Claude answered: each printed line is held to the contrast it reads.
"""

from __future__ import annotations

import re
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from threetears.evals.kernel.host import CANDIDATE_MODEL_LEVER
from threetears.evals.quick import Comparison
from threetears.evals.testing import RecordedPrompt
from packages.evals.tests.example_loader import EXAMPLES, check_covers, load_example, replaying
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations

NAME = "prompt_x_effort.py"


def _load() -> ModuleType:
    return load_example(NAME)


async def test_with_no_api_key_it_replays_four_arms_keyed_by_every_factor(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load()
    asked = [RecordedPrompt(text, case["ticket"]) for text in module.PROMPTS.values() for case in module.CASES] * 2
    for settings in module.EFFORTS.values():
        check_covers(module, asked, settings=settings, model=module.MODEL)
    replay = replaying(module, monkeypatch)

    comparison = await module.main()

    assert replay.served() == 112, "14 tickets x 2 repeats x 4 arms, as the docstring says"
    low, medium, model = *module.EFFORTS, module.MODEL
    assert isinstance(comparison, Comparison)
    assert comparison.factors == ("effort", "prompt", "model")
    assert list(comparison.arms) == [
        (low, "v1", model),
        (low, "v2", model),
        (medium, "v1", model),
        (medium, "v2", model),
    ]
    assert comparison.control == (low, "v1", model)
    assert all(summary.status == "completed" and summary.n_scored == 28 for summary in comparison.arms.values())
    assert {summary.candidate_model for summary in comparison.arms.values()} == {model}

    campaign = off_loop(comparison.host.storage.load_campaign, comparison.campaign_id, comparison.scope_id)
    assert campaign is not None and campaign.declared_design is not None
    axes = {axis.axis_id: [value.display for value in axis.values] for axis in campaign.declared_design.axes}
    assert axes == {"callable.effort": [low, medium], "callable.prompt": ["v1", "v2"], CANDIDATE_MODEL_LEVER: [model]}

    out = capsys.readouterr().out
    assert out.startswith("ANTHROPIC_API_KEY is not set: replaying recordings/prompt_x_effort.jsonl, Claude (")
    # Each effort's line reads v2 against v1 at that effort: against (low, v1) at low, and against() at medium.
    for effort, read in ((low, comparison), (medium, await comparison.against((medium, "v1", model)))):
        (row,) = [row for row in read.contrasts("accuracy") if row["arm"] == (effort, "v2", model)]
        assert (
            f"\nAt {effort} effort, v2 vs v1 on Accuracy: {row['control_mean']:.2f} -> {row['arm_mean']:.2f}, " in out
        )
        assert re.search(rf"\nAt {effort} effort, v2 vs v1 on .*: {re.escape(row['verdict'])}\n", out)
    assert comparison.render() not in out
    assert out.rstrip().endswith("The full report: print(comparison.render()), or reports.py to write it to files.")


async def test_a_candidate_sends_its_arm_s_prompt_and_reads_the_label() -> None:
    module = _load()
    sent: list[str] = []

    async def generate(*, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        sent.append(system)
        return SimpleNamespace(content=" Billing\n")

    llm = SimpleNamespace(generate=generate)
    case = module.CASES[0]
    assert await module.classifier(llm, "v1")(case) == "billing"
    assert await module.classifier(llm, "v2")(case) == "billing"
    assert sent == [module.PROMPTS["v1"], module.PROMPTS["v2"]]


def test_the_two_efforts_differ_in_exactly_one_request_setting() -> None:
    module = _load()
    low, medium = (settings.model_dump() for settings in module.EFFORTS.values())
    assert {name for name in low if low[name] != medium[name]} == {"reasoning_effort"}


def test_the_example_reaches_the_engine_only_through_public_roots() -> None:
    assert public_root_violations(SOURCE_ROOT, [(NAME, EXAMPLES / NAME)], consumer_root=REPO_ROOT) == []
