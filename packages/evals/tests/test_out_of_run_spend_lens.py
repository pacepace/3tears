"""The out-of-run ledger is read back: a lens, its catalogue action and the CLI's ``spend``, one read.

Every call the engine makes outside a run — a launch's case generation, a rubric proposal — is ledgered as an
``eval_out_of_run_spend`` row, and until this lens nothing read one back: the spend existed only in the store.
These pin:

- **The lens lists the scope's calls and sums them**, overall, per purpose and per launch, narrowed by
  purpose, launch or template.
- **Missing is not zero**: a call that reported no cost and one that raised are counted unpriced and left out
  of the sum, and the text says the sum is then a floor.
- **Every surface reads it**: the ``scope_out_of_run_spend`` action and the CLI's ``spend``.
"""

from __future__ import annotations

import pytest

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.contracts import OutOfRunSpend
from threetears.evals.ops import OutOfRunSpendReport, out_of_run_spend_text, scope_out_of_run_spend
from threetears.evals.quick import run_cli
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE
from packages.evals.tests.ops_support import CALLER, ops_fixture

_GROUP = "launch-1"


def _row(**fields: object) -> OutOfRunSpend:
    defaults: dict[str, object] = {
        "scope_id": TOYHOST_SCOPE,
        "purpose": "variation",
        "model": "writer/model",
        "outcome": "completed",
        "stop_reason": "end_turn",
        "cost_usd": 0.01,
        "priced_ceiling_usd": 0.02,
        "template_id": "tpl-a",
        "launch_group_id": _GROUP,
    }
    return OutOfRunSpend.model_validate({**defaults, **fields})


def _ledgered() -> tuple[object, list[OutOfRunSpend]]:
    fixture = ops_fixture()
    rows = [
        _row(created_at="2026-10-05T10:00:00+00:00"),
        _row(created_at="2026-10-05T10:00:01+00:00", cost_usd=None, priced_ceiling_usd=None),
        _row(
            created_at="2026-10-05T10:00:02+00:00",
            outcome="raised",
            failure="TimeoutError",
            stop_reason=None,
            cost_usd=None,
        ),
        _row(
            created_at="2026-10-05T10:00:03+00:00",
            purpose="proposer",
            template_id=None,
            launch_group_id=None,
            cost_usd=0.05,
            priced_ceiling_usd=0.1,
        ),
        _row(created_at="2026-10-05T10:00:04+00:00", scope_id="another-scope", cost_usd=9.0),
    ]
    for row in rows:
        fixture.host.eval_host.storage.save_out_of_run_spend(row)
    return fixture, rows


def test_the_lens_lists_a_scopes_calls_and_sums_them_with_the_unpriced_counted_beside():
    fixture, rows = _ledgered()

    report = scope_out_of_run_spend(fixture.host.eval_host, TOYHOST_SCOPE)

    assert [row.id for row in report.rows] == [row.id for row in rows[:4]], "the scope's calls, oldest first"
    totals = report.totals
    assert (totals.n_calls, totals.n_raised, totals.n_unpriced, totals.n_unbounded) == (4, 1, 2, 1)
    assert totals.priced_usd == pytest.approx(0.06), "only the priced calls are summed"
    assert totals.ceiling_usd == pytest.approx(0.14)
    assert (report.by_purpose["variation"].n_calls, report.by_purpose["proposer"].priced_usd) == (3, 0.05)
    assert list(report.by_launch) == [_GROUP] and report.by_launch[_GROUP].n_calls == 3, "a proposal is in no launch"
    text = out_of_run_spend_text(report)
    assert "total: 4 call(s): $0.0600 reported, 2 unpriced (so at least), 1 raised" in text
    assert "TimeoutError" in text


@pytest.mark.parametrize(
    ("narrowed", "expected"),
    [({"purpose": "proposer"}, 1), ({"launch_group_id": _GROUP}, 3), ({"template_id": "tpl-a"}, 3)],
    ids=["purpose", "launch", "template"],
)
def test_the_lens_narrows_by_purpose_launch_or_template(narrowed: dict[str, str], expected: int):
    fixture, _rows = _ledgered()

    report = scope_out_of_run_spend(fixture.host.eval_host, TOYHOST_SCOPE, **narrowed)  # type: ignore[arg-type]

    assert report.totals.n_calls == len(report.rows) == expected
    assert next(iter(narrowed.values())) in out_of_run_spend_text(report)


def test_an_empty_ledger_says_so():
    fixture = ops_fixture()

    report = scope_out_of_run_spend(fixture.host.eval_host, TOYHOST_SCOPE)

    assert report.rows == [] and report.totals.n_calls == 0
    assert "no out-of-run call is ledgered here" in out_of_run_spend_text(report)


async def test_the_action_reads_the_ledger_through_the_lens():
    fixture, _rows = _ledgered()
    evals = eval_catalogue().mount_all(standard_tools())[0]

    outcome = await evals.call(
        {"action": "scope_out_of_run_spend", "purpose_filter": "variation"}, host=fixture.host, caller=CALLER
    )

    assert not outcome.is_error, outcome.text
    report = OutOfRunSpendReport.model_validate(outcome.structured)
    assert report.totals.n_calls == 3 and report.purpose == "variation"
    assert "unpriced (so at least)" in outcome.text


def test_the_clis_spend_prints_the_ledger(capsys: pytest.CaptureFixture[str]):
    fixture, _rows = _ledgered()

    code = run_cli(
        ["spend", "--scope", TOYHOST_SCOPE, "--launch-group", _GROUP], host_factory=lambda: fixture.host.eval_host
    )

    assert code == 0
    printed = capsys.readouterr().out
    assert f"(launch {_GROUP})" in printed and "total: 3 call(s)" in printed
