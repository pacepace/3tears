"""The prototype's analyst command line: what it prints is what the mounted analyst tool answers.

Contracts held here (``tests/fixtures/report_plans/analyst_cli.py``):

- **the text is the tool's**: a call's printed text is, byte for byte, the text the analyst tool cut for that role
  answers over the same fixture snapshots, for help, a reference page, a read and a paged read; ``--json`` carries the
  structured content beside it;
- **a refusal teaches**: a refused call exits 1 and prints its recovery line last, as the tool writes it;
- **the ids are fixed**: ``--list-snapshots`` names every fixture snapshot by the id ``fixture_snapshots`` gives it,
  the same on every invocation;
- **the primer is the server's**: ``--primer`` prints the primer of the tool it mounts;
- **usage is refused**: arguments that are not a JSON object, or a fixture that does not exist, exit 2.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from packages.evals.tests.analyst_support import KAIRO, fixture_snapshots
from packages.evals.tests.fixtures.report_plans.analyst_cli import main
from packages.evals.tests.ops_support import CALLER, ops_fixture
from threetears.evals.actions import ActionOutcome, MountedTool, analyst_tools, catalogue_primer, eval_catalogue


def _printed(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    """What the command line printed for ``argv``, its trailing newline dropped, and its exit status."""
    status = main(list(argv))
    return status, capsys.readouterr().out.removesuffix("\n")


def _tool(role: str) -> MountedTool:
    (spec,) = analyst_tools(writes=False, role="editor" if role == "editor" else "analyst")
    return eval_catalogue().mount(spec)


def _answered(role: str, arguments: dict[str, Any]) -> ActionOutcome:
    """What the analyst tool cut for ``role`` answers, over a host seeded with every fixture snapshot."""
    fixture = ops_fixture(generation=False)
    for snapshot in fixture_snapshots(CALLER.scope_id).values():
        fixture.host.eval_host.storage.save_report_snapshot(snapshot)
    return asyncio.run(_tool(role).call(arguments, host=fixture.host, caller=CALLER))


def _ids(capsys: pytest.CaptureFixture[str]) -> dict[str, str]:
    status, listed = _printed(capsys, "--list-snapshots")
    assert status == 0
    return dict(line.split("\t") for line in listed.splitlines())


def test_list_snapshots_names_each_fixture_by_its_snapshot_id_the_same_every_time(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ids = _ids(capsys)
    expected = {("kairo" if name == KAIRO else name): s.id for name, s in fixture_snapshots(CALLER.scope_id).items()}
    assert ids == expected
    assert _ids(capsys) == ids, "a second invocation names the same ids"


@pytest.mark.parametrize(
    ("role", "argv", "arguments"),
    [
        pytest.param("analyst", (), {"action": "help"}, id="help"),
        pytest.param("analyst", (), {"action": "help", "topic": "figure_keys.cost"}, id="a reference page"),
        pytest.param("editor", (), {"action": "help"}, id="the editor's help"),
        pytest.param("analyst", ("--snapshot-of", "r1"), {"action": "campaign_overview"}, id="a read"),
        pytest.param("editor", ("--snapshot-of", "kairo"), {"action": "options_dominance"}, id="the editor's read"),
        pytest.param("analyst", ("--snapshot-of", "r5"), {"action": "contrasts_list", "limit": 2}, id="a paged read"),
    ],
)
def test_a_call_prints_the_text_the_mounted_tool_answers(
    capsys: pytest.CaptureFixture[str], role: str, argv: tuple[str, ...], arguments: dict[str, Any]
) -> None:
    ids = _ids(capsys)
    sent = {**arguments, "snapshot_id": ids[argv[1]]} if argv else arguments
    answered = _answered(role, sent)
    assert answered.ending == "ok", answered.text
    status, text = _printed(capsys, "--role", role, *argv, json.dumps(arguments))
    assert (status, text) == (0, answered.text)
    status, printed = _printed(capsys, "--role", role, "--json", *argv, json.dumps(arguments))
    assert status == 0
    assert json.loads(printed) == {"text": answered.text, "structured_content": answered.structured, "is_error": False}


def test_a_paged_read_ends_on_the_call_to_its_next_page(capsys: pytest.CaptureFixture[str]) -> None:
    ids = _ids(capsys)
    status, text = _printed(capsys, "--snapshot-of", "r5", '{"action": "contrasts_list", "limit": 2}')
    assert status == 0
    assert text.splitlines()[-1].startswith(f"next: action='contrasts_list', snapshot_id='{ids['r5']}', limit=2")


def test_a_refused_call_exits_1_and_prints_its_recovery_line_last(capsys: pytest.CaptureFixture[str]) -> None:
    ids = _ids(capsys)
    answered = _answered("analyst", {"action": "measure_read", "snapshot_id": ids["r2"]})
    assert answered.is_error and answered.recovery is not None
    status, text = _printed(capsys, "--snapshot-of", "r2", '{"action": "measure_read"}')
    assert (status, text) == (1, answered.text)
    last = text.splitlines()[-1]
    assert last.startswith("recovery: ")
    assert json.loads(last.removeprefix("recovery: "))["parameters"] == ["measure_key"]
    status, printed = _printed(capsys, "--json", "--snapshot-of", "r2", '{"action": "measure_read"}')
    structured = json.loads(printed)["structured_content"]
    assert status == 1 and structured["ending"] == "refused" and structured["recovery"]["parameters"] == ["measure_key"]


@pytest.mark.parametrize("role", ["analyst", "editor"])
def test_primer_prints_the_primer_of_the_tool_it_mounts(capsys: pytest.CaptureFixture[str], role: str) -> None:
    assert _printed(capsys, "--role", role, "--primer") == (0, catalogue_primer((_tool(role),)))


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(("[1, 2]",), id="arguments that are no object"),
        pytest.param(("{not json",), id="arguments that are no JSON"),
        pytest.param((), id="no arguments"),
        pytest.param(("--snapshot-of", "r9", '{"action": "help"}'), id="a fixture that does not exist"),
        pytest.param(
            ("--snapshot-of", "r1", '{"action": "caveats_list", "snapshot_id": "report-snapshot:other"}'),
            id="a snapshot_id other than the fixture's",
        ),
    ],
)
def test_a_usage_error_exits_2_saying_what_is_wrong(capsys: pytest.CaptureFixture[str], argv: tuple[str, ...]) -> None:
    assert main(list(argv)) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.startswith("analyst_cli.py: ")
