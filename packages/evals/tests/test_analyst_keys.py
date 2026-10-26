"""A key check: each figure key a report cites, read on the snapshot or refused with what to write instead.

Contracts held here (``threetears.evals.analysis.analyst.keys``, ``ops.analyst_keys``, ``actions.analyst_keys``, and
``figure_keys``' ``key_form`` and ``figure_filters``):

- **a key that reads** prints what the page prints for it, through its filters (``render`` of ``{key|filters}``), and
  holds the value ``resolve`` reads, on every fixture input (r1 to r5, the misfits, Kairo round 2, s1 to s3), for
  every arm and every figure the grammar's examples name there; it names its family's form and the filters its figure
  takes, each of which spells it;
- **a key that reads nothing** says why: unstated where the key is well formed and the snapshot states no such figure
  (a monthly figure with no declared volume), refused where the path names nothing (the resolver's continuations), the
  grammar names no such filter, or the figure does not suit the filter (the remedy names the filters it takes);
- **the Kairo editor's keys**: the four keys its editor could not verify through its cut read on the snapshot;
- **the call**: a key repeated is read once, keys are paged, too many or none are refused; both the analyst's tool and
  the editor's cut carry it; through the action every call answers alike twice within the budget, its text writing
  each figure as ``key|filters = reads``.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from packages.evals.tests.analyst_support import (
    KAIRO,
    TEXT_BUDGET,
    action_calls,
    budget_inputs,
    fixture_inputs,
    fixture_snapshots,
)
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture
from threetears.evals.actions import MountedTool, analyst_actions, analyst_tools, eval_catalogue
from threetears.evals.analysis.analyst.keys import KEYS_MOST, figure_keys_check
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.errors import UnresolvableReference, UnstatedFigure
from threetears.evals.analysis.report.figure_keys import (
    FILTER_SPELLINGS,
    figure_filters,
    format_figure,
    key_form,
    parse_key,
    render,
    resolve,
)
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.errors import RefusedInput


def _inputs() -> dict[str, ReportInput]:
    held = dict(fixture_inputs())
    held[KAIRO] = frozen_input(fixture_snapshots("s")[KAIRO])
    held |= {name: budget_inputs()[name] for name in ("s1", "s2", "s3")}
    return held


#: Every input the check is held to, by name.
INPUTS = _inputs()


def _keys(held: ReportInput) -> list[str]:
    """Keys read off the input itself: each arm; each arm's cost, pass rate and every statistic of every measure and
    judged dimension it holds, where the input states it; each criterion's threshold; the candidates' count."""
    keys = [f"arm:{arm.id}" for arm in held.arms] + ["n_candidates"]
    for key, measure in held.measures.items():
        keys += [f"measure.{key}.{arm}.{name}" for arm in measure.per_arm for name in ("mean", "half_width", "n")]
    for key, judged in held.judged.items():
        keys += [f"judged.{key}.{arm}.{name}" for arm in judged.per_arm for name in ("mean", "n")]
    keys += [f"cost_per_1k.{arm.id}" for arm in held.arms]
    keys += [f"criteria.{criterion.id}.threshold" for criterion in held.criteria]
    return keys


def _expected(key: str, held: ReportInput) -> tuple[str, Any] | None:
    """What the page prints for a key and the value it reads; None where the resolver refuses it."""
    try:
        figure = resolve(key, held)
    except UnresolvableReference:
        return None
    value = list(figure.value) if isinstance(figure.value, tuple) else figure.value
    return render("{" + key + "}", held, {}), value


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_each_key_reads_what_the_page_prints_or_says_why_not(name: str) -> None:
    held = INPUTS[name]
    keys = list(dict.fromkeys(_keys(held)))
    rows = []
    for start in range(0, len(keys), KEYS_MOST):
        chunk = keys[start : start + KEYS_MOST]
        rows += figure_keys_check(held, chunk, limit=len(chunk)).keys
    assert [row.asked for row in rows] == keys
    for row in rows:
        expected = _expected(row.asked, held)
        if expected is None:
            assert row.status in ("unstated", "refused") and row.figure is None, row.asked
            assert row.refusal and row.remedy, row.asked
            with pytest.raises(UnstatedFigure if row.status == "unstated" else UnresolvableReference):
                resolve(row.asked, held)
            continue
        assert row.status == "reads" and row.figure is not None, row
        reads, value = expected
        assert (row.figure.key, row.figure.reads, row.figure.model_dump(mode="json")["value"]) == (
            row.asked,
            reads,
            value,
        )
        assert row.form == key_form(row.asked, held).form


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_a_key_written_through_a_filter_reads_as_that_filter_spells_it_and_names_every_filter_it_takes(
    name: str,
) -> None:
    held = INPUTS[name]
    numbers = [key for key in _keys(held) if (found := _expected(key, held)) and not isinstance(found[1], str)]
    for key in numbers[:12]:
        figure = resolve(key, held)
        taken = figure_filters(figure)
        assert taken, key
        written = [f"{{{key}|{one}}}" for one in taken]
        checked = figure_keys_check(held, written, limit=len(written))
        for row, one in zip(checked.keys, taken, strict=True):
            assert row.status == "reads" and row.figure is not None, row
            assert row.figure.reads == format_figure(figure, (one,)) == render(row.asked, held, {})
            assert row.figure.filters == [one] and row.filters_taken == list(taken)


def test_a_key_reads_nothing_with_why_and_what_to_write_instead() -> None:
    held = fixture_inputs()["r5"]
    assert held.meta.volume_per_month is None, "the control: no declared volume"
    arm = held.arms[0].id
    (unstated, misspelt, unsuited, unknown_filter, text_filtered) = figure_keys_check(
        held,
        [
            f"cost_per_month.{arm}",
            f"cost_per_1k.{arm}.mean",
            f"{{cost_per_1k.{arm}|pct}}",
            f"cost_per_1k.{arm}|bogus",
            "meta.campaign|pct",
        ],
    ).keys
    assert unstated.status == "unstated" and unstated.form == "cost_per_month.<arm>"
    assert unstated.remedy is not None and "figure_keys.cost" in unstated.remedy
    with pytest.raises(UnstatedFigure):
        resolve(f"cost_per_month.{arm}", held)
    assert misspelt.status == "refused" and misspelt.form is None
    assert misspelt.refusal is not None and "names no figure" in misspelt.refusal
    taken = figure_filters(resolve(f"cost_per_1k.{arm}", held))
    assert unsuited.status == "refused" and unsuited.remedy is not None
    assert all(one in unsuited.remedy for one in taken) and "pct" not in taken
    assert unknown_filter.status == "refused" and unknown_filter.remedy is not None
    assert all(one in unknown_filter.remedy for one in FILTER_SPELLINGS)
    assert text_filtered.status == "refused" and text_filtered.remedy is not None
    assert "no filter" in text_filtered.remedy


def test_the_keys_kairos_editor_could_not_verify_read_on_the_snapshot() -> None:
    """The W2 prototype's Kairo editor could not read these through its cut; they are keys the grammar holds."""
    held = INPUTS[KAIRO]
    keys = [f"reasoning_ratio.{arm.id}" for arm in held.arms if arm.reasoning_ratio is not None]
    keys += ["judge.model", "judge.tier", "cost.uplift_range"]
    assert len(keys) > 3, "the control: Kairo's arms state their reasoning share"
    checked = figure_keys_check(held, keys)
    assert [row.status for row in checked.keys] == ["reads"] * len(keys)
    for row in checked.keys:
        assert row.figure is not None and row.figure.reads == render("{" + row.asked + "}", held, {})


def test_a_repeated_key_is_read_once_and_the_keys_are_paged() -> None:
    held = fixture_inputs()["r4"]
    keys = [f"arm:{arm.id}" for arm in held.arms]
    checked = figure_keys_check(held, [*keys, keys[0], f" {keys[1]} "], limit=2)
    assert checked.total == len(keys) and [row.asked for row in checked.keys] == keys[:2]
    assert checked.next_offset == 2
    rest = figure_keys_check(held, keys, limit=len(keys), offset=2)
    assert [row.asked for row in rest.keys] == keys[2:] and rest.next_offset is None


def test_no_key_and_too_many_keys_are_refused() -> None:
    held = fixture_inputs()["r4"]
    with pytest.raises(RefusedInput, match="no key"):
        figure_keys_check(held, [])
    with pytest.raises(RefusedInput, match="more than one call checks"):
        figure_keys_check(held, [f"meta.campaign|{index}" for index in range(KEYS_MOST + 1)])


def test_a_braced_key_and_a_bare_one_read_alike() -> None:
    held = fixture_inputs()["r4"]
    arm = held.arms[1].id
    braced, bare = figure_keys_check(held, [f"{{cost_per_1k.{arm}|usd}}", f"cost_per_1k.{arm}|usd"]).keys
    assert braced.figure is not None and bare.figure is not None
    assert braced.figure == bare.figure and parse_key(f"cost_per_1k.{arm}|usd").filters == ("usd",)


# =============================================================================
# Through the action, on both roles
# =============================================================================


def _seeded() -> tuple[OpsFixture, dict[str, Any]]:
    fixture = ops_fixture()
    snapshots = fixture_snapshots(TOYHOST_SCOPE)
    for snapshot in snapshots.values():
        fixture.host.eval_host.storage.save_report_snapshot(snapshot)
    return fixture, snapshots


def _tool(role: str) -> MountedTool:
    (spec,) = analyst_tools(role=role)  # type: ignore[arg-type]
    return eval_catalogue().mount(spec)


def test_both_the_analyst_and_its_editor_carry_the_check() -> None:
    assert "figure_keys_check" in {action.name for action in analyst_actions()}
    for role in ("analyst", "editor"):
        assert "figure_keys_check" in {action.name for action in _tool(role).actions}, role


@pytest.mark.parametrize("role", ["analyst", "editor"])
async def test_every_call_answers_alike_twice_within_its_budget_writing_each_figure_by_its_key(role: str) -> None:
    fixture, snapshots = off_loop(_seeded)
    tool = _tool(role)
    (action,) = (action for action in analyst_actions() if action.name == "figure_keys_check")
    for name, snapshot in snapshots.items():
        held = frozen_input(snapshot)
        for arguments in action_calls(action, held, snapshot.id):
            first = await tool.call(arguments, host=fixture.host, caller=CALLER)
            second = await tool.call(arguments, host=fixture.host, caller=CALLER)
            where = f"figure_keys_check {arguments} on {name}"
            assert first.ending == "ok" and first.structured is not None, f"{where}: {first.text}"
            assert (first.text, first.structured) == (second.text, second.structured), where
            assert len(first.text) <= TEXT_BUDGET, where
            for row in first.structured["keys"]:
                figure = row["figure"]
                if figure is None:
                    continue
                cited = "|".join([figure["key"], *figure["filters"]])
                line = re.escape(f"{cited} = {figure['reads']}")
                assert re.search(rf"^\s*{line}$", first.text, re.MULTILINE), (where, cited)


async def test_too_many_keys_through_the_action_are_refused_naming_the_most() -> None:
    fixture, snapshots = off_loop(_seeded)
    arguments = {
        "action": "figure_keys_check",
        "snapshot_id": snapshots["r4"].id,
        "keys": [f"meta.campaign|{index}" for index in range(KEYS_MOST + 1)],
    }
    outcome = await _tool("editor").call(arguments, host=fixture.host, caller=CALLER)
    assert outcome.is_error and str(KEYS_MOST) in outcome.text
