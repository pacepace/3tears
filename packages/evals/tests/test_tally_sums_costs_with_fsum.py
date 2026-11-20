"""The writer's tally sums turn costs the way the ledger does: exactly, by ``math.fsum``.

Ten turns of a tenth of a dollar add to ``0.9999999999999999`` by a running ``+=`` and to ``1.0`` exactly; the stored
report's by-role, by-round and total rollups all state the exact sum, the same figure the tally's own spend reads.
"""

from __future__ import annotations

import math

from packages.evals.tests.report_writer_support import reply
from threetears.evals.ops.report_writer.tally import Tally, TurnCost

_COSTS = [0.1] * 10


def _tally(costs: list[float | None]) -> Tally:
    tally = Tally()
    for index, cost in enumerate(costs):
        tally.record_turn(
            TurnCost.of(reply(cost=cost), session="analyst", role="analyst", round=1, turn=index, wall_s=1.0)
        )
    return tally


def test_the_inputs_make_a_running_sum_differ_from_the_exact_one() -> None:
    running = 0.0
    for cost in _COSTS:
        running += cost
    assert running != math.fsum(_COSTS) == 1.0


def test_every_rollup_states_the_exact_sum() -> None:
    stored = _tally(list(_COSTS)).to_json()
    assert stored["total"]["cost_usd"] == 1.0
    assert stored["by_role"]["analyst"]["cost_usd"] == 1.0
    assert stored["by_round"]["analyst:1"]["cost_usd"] == 1.0


def test_the_rollup_and_the_tally_spend_are_one_figure() -> None:
    tally = _tally(list(_COSTS))
    assert tally.to_json()["total"]["cost_usd"] == tally.spent_usd()


def test_an_unpriced_turn_leaves_the_rollup_without_a_cost() -> None:
    assert _tally([0.1, None]).to_json()["total"]["cost_usd"] is None
