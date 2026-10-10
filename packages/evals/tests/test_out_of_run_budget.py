"""The out-of-run budget's own guards, each driven to fire: what it admits, what it makes, and what it ledgers.

The budget is reached elsewhere only through ``propose_draft`` and ``generate_variations``, which hand it
well-formed admissions on the client they were priced on — so nothing there can show that it refuses the
rest. These do, one refusal per forbidden shape, beside the accepted one:

- **A call is made only as it was priced**: the very admission ``admit`` minted, on the very client it was
  priced on, once. A copy (altered or not), another budget's admission, a second making and another client
  of the same model are refused.
- **A ceiling and a cap are amounts**: a client pricing a call at a negative or non-finite figure, and a cap
  that is not positive and finite, are refused.
- **A paid call is always ledgered**: a completion reporting what the row cannot hold is recorded with those
  attributes unreadable, not dropped; a ledger write that fails while the call itself raised leaves the
  call's own failure to propagate.
- **Usage is read by the completion protocol's names**: ``served_model`` is the model the response named,
  and a result offering only ``model`` records none.
- **A ledger row leaves the loop**: it is written on the executor the budget names, for a completed call and a
  raised one alike.
"""

from __future__ import annotations

import dataclasses
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import pytest

from threetears.evals.contracts import (
    OutOfRunBudget,
    OutOfRunSpend,
    PlannedCall,
    StorageError,
    ValidationFailedError,
)

_MODEL = "writer/model"
_CALL = PlannedCall(system="write values", user="axis: invoice id")


@dataclass(frozen=True)
class _Reported:
    """A completion as a client reports it: whatever attributes it carries, read by name."""

    content: str = "{}"
    served_model: str | None = None
    stop_reason: Any = "end_turn"
    input_tokens: Any = 10
    output_tokens: Any = 5
    reasoning_tokens: Any = None
    cost_usd: Any = 0.002
    price_source: str | None = "rate card"


@dataclass(frozen=True)
class _SimulatorShaped:
    """A completion carrying only the simulator-port's attribution name, ``model``."""

    content: str = "{}"
    model: str = "served/elsewhere"


# parity-with: threetears.evals.contracts.completion.PricedCompletion
@dataclass(eq=False)
class _FakeWriter:
    """A priced client: prices every call at ``ceiling`` and answers ``reported``, or raises ``raises``."""

    model_name: str = _MODEL
    ceiling: float | None = 0.01
    reported: Any = field(default_factory=_Reported)
    raises: BaseException | None = None
    calls: int = 0

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        return self.ceiling

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self.reported


# parity-with: threetears.evals.contracts.out_of_run_spend.OutOfRunSpendStore
@dataclass
class _FakeLedger:
    """Keeps every row written, or refuses every write when ``failing``."""

    failing: bool = False
    rows: list[OutOfRunSpend] = field(default_factory=list)
    attempts: int = 0

    def save_out_of_run_spend(self, spend: OutOfRunSpend, /) -> None:
        self.attempts += 1
        if self.failing:
            raise StorageError("the ledger store is down")
        self.rows.append(spend)


def _budget(ledger: _FakeLedger | None = None, *, cap_usd: float | None = 1.0) -> OutOfRunBudget:
    return OutOfRunBudget(
        ledger if ledger is not None else _FakeLedger(), scope_id="s", cap_usd=cap_usd, blocking_executor=None
    )


# =============================================================================
# A call is made only as it was priced
# =============================================================================


async def test_an_admitted_call_is_made_on_its_client_and_ledgered_once():
    ledger = _FakeLedger()
    budget = _budget(ledger)
    writer = _FakeWriter()
    [admitted] = budget.admit(writer, "variation", [_CALL])

    recorded = await budget.generate(writer, admitted)

    assert writer.calls == 1 and ledger.rows == [recorded.spend]
    assert (recorded.spend.outcome, recorded.spend.priced_ceiling_usd, recorded.spend.unreadable) == (
        "completed",
        0.01,
        [],
    )


async def test_a_call_already_made_is_refused():
    budget = _budget()
    writer = _FakeWriter()
    [admitted] = budget.admit(writer, "variation", [_CALL])
    await budget.generate(writer, admitted)

    with pytest.raises(ValueError, match="not one this budget admitted and has not yet made"):
        await budget.generate(writer, admitted)
    assert writer.calls == 1


async def test_another_budgets_admission_is_refused():
    writer = _FakeWriter()
    [theirs] = _budget().admit(writer, "variation", [_CALL])
    ledger = _FakeLedger()

    with pytest.raises(ValueError, match="not one this budget admitted and has not yet made"):
        await _budget(ledger).generate(writer, theirs)
    assert writer.calls == 0 and ledger.attempts == 0


@pytest.mark.parametrize(
    "altered",
    [{"call": PlannedCall(system="write values", user="axis: invoice id" + " and much more" * 500)}, {}],
    ids=["its-call-enlarged-after-pricing", "an-unaltered-copy"],
)
async def test_a_copy_of_an_admission_is_refused(altered: dict[str, Any]):
    """``dataclasses.replace`` keeps the token, so a check on the token alone made an enlarged prompt look priced."""
    ledger = _FakeLedger()
    budget = _budget(ledger)
    writer = _FakeWriter()
    [admitted] = budget.admit(writer, "variation", [_CALL])

    with pytest.raises(ValueError, match="a copy of an admission"):
        await budget.generate(writer, dataclasses.replace(admitted, **altered))
    assert writer.calls == 0 and ledger.attempts == 0
    await budget.generate(writer, admitted)
    assert writer.calls == 1, "the admission itself is still made"


async def test_a_call_made_on_another_client_of_the_same_model_is_refused():
    """Same model name, another client: its output cap — and so the price of the call — can differ."""
    budget = _budget()
    priced_on = _FakeWriter()
    larger_cap = _FakeWriter()
    [admitted] = budget.admit(priced_on, "variation", [_CALL])

    with pytest.raises(ValueError, match="is being made on another"):
        await budget.generate(larger_cap, admitted)
    assert (priced_on.calls, larger_cap.calls) == (0, 0)


# =============================================================================
# A ceiling and a cap are amounts
# =============================================================================


@pytest.mark.parametrize("ceiling", [-0.01, math.nan, math.inf], ids=["negative", "nan", "infinite"])
def test_a_client_pricing_a_call_at_no_amount_is_refused(ceiling: float):
    budget = _budget()

    with pytest.raises(ValueError, match="a ceiling is a finite amount, 0 or more"):
        budget.admit(_FakeWriter(ceiling=ceiling), "variation", [_CALL])
    assert budget.committed_usd == 0.0


@pytest.mark.parametrize("cap", [0.0, -1.0, math.nan, math.inf], ids=["zero", "negative", "nan", "infinite"])
def test_a_cap_that_is_not_a_positive_amount_is_refused(cap: float):
    with pytest.raises(ValueError, match="cap_usd must be a positive amount, or None"):
        _budget(cap_usd=cap)


def test_no_cap_and_a_positive_cap_are_accepted():
    assert _budget(cap_usd=None).cap_usd is None
    assert _budget(cap_usd=0.5).cap_usd == 0.5


def test_an_admission_past_what_is_left_of_the_cap_is_refused_and_commits_nothing():
    budget = _budget(cap_usd=0.015)
    writer = _FakeWriter()
    budget.admit(writer, "variation", [_CALL])

    with pytest.raises(ValidationFailedError, match="on top of \\$0.0100 already admitted"):
        budget.admit(writer, "variation", [_CALL])
    assert budget.committed_usd == pytest.approx(0.01)


# =============================================================================
# A paid call is always ledgered
# =============================================================================


async def test_a_completed_call_reporting_what_the_ledger_cannot_hold_is_still_ledgered():
    """A raw provider stop reason and a negative count used to fail the row's validation and drop the row."""
    ledger = _FakeLedger()
    budget = _budget(ledger)
    writer = _FakeWriter(reported=_Reported(stop_reason="stop", input_tokens=-3, cost_usd=0.004))
    [admitted] = budget.admit(writer, "variation", [_CALL])

    recorded = await budget.generate(writer, admitted)

    [row] = ledger.rows
    assert row is recorded.spend
    assert (row.stop_reason, row.prompt_tokens) == (None, None)
    assert row.unreadable == ["stop_reason", "prompt_tokens"]
    assert (row.cost_usd, row.completion_tokens, row.price_source) == (0.004, 5, "rate card"), "the rest as reported"


def test_a_raised_call_carries_nothing_unreadable():
    with pytest.raises(ValueError, match="a raised call reported nothing"):
        OutOfRunSpend(scope_id="s", purpose="variation", model=_MODEL, outcome="raised", failure="E", unreadable=["x"])


async def test_a_ledger_write_failing_while_the_call_raised_leaves_the_calls_own_failure():
    ledger = _FakeLedger(failing=True)
    budget = _budget(ledger)
    writer = _FakeWriter(raises=TimeoutError("the provider timed out"))
    [admitted] = budget.admit(writer, "variation", [_CALL])

    with pytest.raises(TimeoutError, match="the provider timed out"):
        await budget.generate(writer, admitted)
    assert ledger.attempts == 1, "the row was attempted"


async def test_a_ledger_write_failing_for_a_completed_call_raises():
    budget = _budget(_FakeLedger(failing=True))
    writer = _FakeWriter()
    [admitted] = budget.admit(writer, "variation", [_CALL])

    with pytest.raises(StorageError, match="the ledger store is down"):
        await budget.generate(writer, admitted)


# =============================================================================
# Usage is read by the completion protocol's names
# =============================================================================


async def test_the_served_model_is_the_one_the_response_named():
    ledger = _FakeLedger()
    budget = _budget(ledger)
    named = _FakeWriter(reported=_Reported(served_model="writer/model-2026-09"))
    simulator_shaped = _FakeWriter(reported=_SimulatorShaped())
    [first] = budget.admit(named, "variation", [_CALL])
    [second] = budget.admit(simulator_shaped, "variation", [_CALL])

    await budget.generate(named, first)
    await budget.generate(simulator_shaped, second)

    assert [row.served_model for row in ledger.rows] == ["writer/model-2026-09", None], (
        "``model`` is attribution, which a client may fill from the request; it is never read as who answered"
    )


# =============================================================================
# A ledger row is written off the loop, on the executor the budget names
# =============================================================================


# parity-with: threetears.evals.contracts.out_of_run_spend.OutOfRunSpendStore
@dataclass
class _FakeThreadRecordingLedger(_FakeLedger):
    """Records the thread each write ran on."""

    threads: list[int] = field(default_factory=list)

    def save_out_of_run_spend(self, spend: OutOfRunSpend, /) -> None:
        self.threads.append(threading.get_ident())
        super().save_out_of_run_spend(spend)


@pytest.mark.parametrize("raises", [None, RuntimeError("the provider refused")], ids=["completed", "raised"])
async def test_the_ledger_row_is_written_on_the_budget_s_executor_and_never_on_the_loop(raises: BaseException | None):
    """Both outcomes' rows — the completed call's and the raised one's — leave the loop for the named pool."""
    ledger = _FakeThreadRecordingLedger()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool_thread = pool.submit(threading.get_ident).result()
        budget = OutOfRunBudget(ledger, scope_id="s", cap_usd=1.0, blocking_executor=pool)
        writer = _FakeWriter(raises=raises)
        [admitted] = budget.admit(writer, "variation", [_CALL])
        if raises is None:
            await budget.generate(writer, admitted)
        else:
            with pytest.raises(RuntimeError):
                await budget.generate(writer, admitted)

    assert ledger.threads == [pool_thread] != [threading.get_ident()]
    assert budget.recorded == tuple(ledger.rows) and len(ledger.rows) == 1
