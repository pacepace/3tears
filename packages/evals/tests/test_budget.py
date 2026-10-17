"""Tests for the per-run eval cost cap.

Covers :class:`~threetears.evals.run.budget.EvalRunCostCap` (the in-memory per-run
accumulator that stops a run once its own spend exceeds its cap) and
:class:`~threetears.evals.run.budget.BudgetStoppedError`.

The cap's contract under test:

- **Pure arithmetic** — it accumulates ``cost_usd`` in memory and trips once the
  running total *exceeds* the ceiling. No BudgetService, no shared pool, no
  database I/O, so no fail-open surface.
- **Enforcement toggle** — disabled (``enabled=False``) makes it never trip
  (uncapped), while still accumulating for observability.
- **Configuration arrives as values** — the module reads no configuration of
  its own. The caller supplies the default ceiling and the enforcement flag, and
  an explicit ``max_cost_usd`` overrides the default.
- **Unpriced cost** — a ``None`` cost is spend nobody priced, never zero. An enforcing cap
  stops the run on it (it cannot say the run is inside a ceiling it cannot count against);
  an uncapped one counts it and carries on. Either way it is logged at WARNING with the run id.
"""

from __future__ import annotations

import inspect
import logging

import pytest

from threetears.evals.run.budget import BudgetStoppedError, CapBreach, EvalRunCostCap


def _cap(*, run_id="run-1", max_cost_usd=15.0, enabled=True) -> EvalRunCostCap:
    return EvalRunCostCap(run_id, max_cost_usd, enabled=enabled)


# ---------------------------------------------------------------------------
# BudgetStoppedError
# ---------------------------------------------------------------------------


def test_budget_stopped_error_carries_progress():
    err = BudgetStoppedError(3, 12, CapBreach(max_cost_usd=15.0, accumulated_usd=15.5, unpriced_results=0))
    assert err.completed == 3
    assert err.total == 12
    assert "3/12" in str(err)


# ---------------------------------------------------------------------------
# Accumulation + trip semantics
# ---------------------------------------------------------------------------


def test_new_cap_is_not_exceeded_and_has_zero_spend():
    cap = _cap()
    assert cap.accumulated_usd == 0.0
    assert cap.exceeded is False


def test_records_accumulate_in_memory():
    cap = _cap(max_cost_usd=10.0)
    cap.record(1.5)
    cap.record(2.25)
    assert cap.accumulated_usd == pytest.approx(3.75)
    assert cap.exceeded is False


def test_at_the_cap_is_not_exceeded_strictly_over_is():
    """The cap trips on strictly-greater, so a run at exactly the ceiling proceeds."""
    cap = _cap(max_cost_usd=5.0)
    cap.record(5.0)
    assert cap.exceeded is False  # 5.0 > 5.0 is False
    cap.record(0.01)
    assert cap.exceeded is True  # 5.01 > 5.0


def test_max_cost_usd_is_exposed():
    assert _cap(max_cost_usd=15.0).max_cost_usd == 15.0


# ---------------------------------------------------------------------------
# Enforcement toggle — disabled never trips but still accumulates
# ---------------------------------------------------------------------------


def test_disabled_cap_never_exceeds_but_still_accumulates():
    cap = _cap(max_cost_usd=1.0, enabled=False)
    cap.record(100.0)
    assert cap.accumulated_usd == pytest.approx(100.0)  # observability preserved
    assert cap.exceeded is False  # uncapped


# ---------------------------------------------------------------------------
# The cascade — default ceiling + enforcement toggle, with per-run override,
# all of it supplied by the caller
# ---------------------------------------------------------------------------


def test_an_omitted_override_inherits_the_default_the_caller_supplied():
    cap = EvalRunCostCap.for_run("run-x", None, configured_max_cost_usd=12.0, enforcement_enabled=True)
    assert cap.max_cost_usd == 12.0


def test_an_override_replaces_the_default_and_is_what_enforcement_reads():
    cap = EvalRunCostCap.for_run("run-x", 3.0, configured_max_cost_usd=12.0, enforcement_enabled=True)
    assert cap.max_cost_usd == 3.0
    cap.record(3.5)
    assert cap.exceeded is True


def test_the_enforcement_flag_the_caller_passed_is_what_decides():
    cap = EvalRunCostCap("run-x", 1.0, enabled=False)
    cap.record(50.0)
    assert cap.exceeded is False


def test_an_unenforced_run_still_gets_a_cap_that_knows_its_ceiling():
    """The cap keeps its number and switches enforcement off — the ledger does the opposite.

    :meth:`~threetears.evals.run.metering.MeteredCallLedger.for_run` reads enforcement-off as
    ``ceiling=None``; this one keeps the ceiling, because the missing-cost WARNING quotes
    it and a ``None`` there would print as a limit nobody set. A host that guessed the
    ledger's shape here would build a cap with no ceiling at all, which is what makes the
    two factories worth having rather than two constructor calls.
    """
    cap = EvalRunCostCap.for_run("run-x", None, configured_max_cost_usd=12.0, enforcement_enabled=False)
    assert cap.max_cost_usd == 12.0
    cap.record(50.0)
    assert cap.exceeded is False


def test_enforcement_must_be_stated_rather_than_defaulted():
    """A cap that silently stops enforcing is worse than one that fails to construct."""
    with pytest.raises(TypeError):
        EvalRunCostCap("run-x", 1.0)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Unpriced cost — never zero: an enforcing cap stops on it, an uncapped run counts it
# ---------------------------------------------------------------------------


def test_unpriced_spend_stops_an_enforcing_cap_however_far_below_its_ceiling(caplog):
    """Counted as zero, an unpriced model let a capped run spend without bound inside a cap that read it as free."""
    cap = _cap(run_id="run-42", max_cost_usd=10.0)
    cap.record(0.5)
    with caplog.at_level(logging.WARNING, logger="threetears.evals.run.budget"):
        cap.record(None)
    assert cap.accumulated_usd == pytest.approx(0.5), (
        "the priced spend is kept as it was, nothing added for the unpriced"
    )
    assert cap.unpriced_results == 1
    assert cap.exceeded is True
    breach = cap.check()
    assert breach == CapBreach(max_cost_usd=10.0, accumulated_usd=0.5, unpriced_results=1)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "run-42" in warnings[0].getMessage()


def test_priced_spend_alone_does_not_stop_a_cap_below_its_ceiling():
    """The accepting side of the refusal above, on the same cap."""
    cap = _cap(max_cost_usd=10.0)
    cap.record(0.5)
    cap.record(0.0)
    assert cap.unpriced_results == 0
    assert cap.check() is None


def test_an_uncapped_run_counts_unpriced_spend_and_carries_on():
    cap = _cap(max_cost_usd=1.0, enabled=False)
    cap.record(None)
    cap.record(0.25)
    assert cap.unpriced_results == 1
    assert cap.accumulated_usd == pytest.approx(0.25)
    assert cap.check() is None


def test_the_stop_on_unpriced_spend_says_so_rather_than_claiming_the_cap_was_exceeded():
    err = BudgetStoppedError(3, 8, CapBreach(max_cost_usd=2.0, accumulated_usd=0.4, unpriced_results=1))
    message = str(err)
    assert "1 result(s) spent what could not be priced" in message
    assert "$0.4000" in message and "$2.0000" in message and "3/8" in message
    assert "exceeded" not in message


# ---------------------------------------------------------------------------
# Structural: the cap path has no BudgetService / DB dependency
# ---------------------------------------------------------------------------


def test_cap_takes_no_budget_service_dependency():
    """The gate path is pure arithmetic — no BudgetService injected anywhere."""
    init_params = set(inspect.signature(EvalRunCostCap.__init__).parameters) - {"self"}
    assert init_params == {"run_id", "max_cost_usd", "enabled"}

    resolve_params = set(inspect.signature(EvalRunCostCap.resolve_effective_ceiling).parameters) - {"cls"}
    assert resolve_params == {"max_cost_usd", "configured_max_cost_usd", "enforcement_enabled"}

    # The production entry point, pinned for the same reason: a service or storage handle
    # would arrive here first, and everything below it takes only what this one passes on.
    for_run_params = set(inspect.signature(EvalRunCostCap.for_run).parameters) - {"cls"}
    assert for_run_params == {"run_id", "max_cost_usd", "configured_max_cost_usd", "enforcement_enabled"}

    # Construction stores only pure-arithmetic state — no service / storage handle.
    cap = _cap()
    assert set(vars(cap)) == {"_run_id", "_max_cost_usd", "_enabled", "_accumulated_usd", "_unpriced_results"}


class TestEffectiveCeilingMatchesTheEnforcingCap:
    """The number a run records must be the number that bounded it.

    These read through one shared cascade; if they ever diverged, a run's stored
    ceiling would misreport what actually limited its spend.
    """

    def test_the_recorded_ceiling_equals_the_caps_ceiling_when_enforcing(self):
        cap = EvalRunCostCap.for_run("run-1", 7.5, configured_max_cost_usd=15.0, enforcement_enabled=True)
        assert (
            EvalRunCostCap.resolve_effective_ceiling(7.5, configured_max_cost_usd=15.0, enforcement_enabled=True)
            == cap.max_cost_usd
        )

    def test_the_default_ceiling_also_agrees(self):
        cap = EvalRunCostCap.for_run("run-1", None, configured_max_cost_usd=15.0, enforcement_enabled=True)
        assert (
            EvalRunCostCap.resolve_effective_ceiling(None, configured_max_cost_usd=15.0, enforcement_enabled=True)
            == cap.max_cost_usd
        )

    def test_an_unenforced_run_records_no_ceiling(self):
        """Enforcement off means nothing bounded the run — recording a number would lie."""
        assert (
            EvalRunCostCap.resolve_effective_ceiling(7.5, configured_max_cost_usd=15.0, enforcement_enabled=False)
            is None
        )


class TestTheStopSaysWhatItMeasured:
    """A budget stop that names neither number reads as an assertion, not a measurement.

    The stop message used to say only how far the run got — "cost cap exceeded
    after 2/4 results". An operator reading it could not tell what ceiling was in
    force or what the run had spent to cross it, and neither number was on any
    other surface either, so the only way to judge whether the stop was reasonable
    was to go and reconstruct it. Both now ride the breach the cap reports.
    """

    def test_the_gate_returns_nothing_while_the_run_is_within_its_cap(self):
        cap = _cap(max_cost_usd=5.0)
        cap.record(4.99)
        assert cap.check() is None

    def test_the_gate_reports_the_ceiling_and_the_spend_that_crossed_it(self):
        cap = _cap(max_cost_usd=5.0)
        cap.record(5.25)
        breach = cap.check()
        assert breach is not None
        assert breach.max_cost_usd == 5.0
        assert breach.accumulated_usd == pytest.approx(5.25)

    def test_an_unenforced_cap_never_reports_a_breach(self):
        """Enforcement off is uncapped, so there is no ceiling to have crossed."""
        cap = _cap(max_cost_usd=1.0, enabled=False)
        cap.record(100.0)
        assert cap.check() is None

    def test_the_error_names_the_cap_and_the_spend_alongside_the_progress(self):
        err = BudgetStoppedError(2, 4, CapBreach(max_cost_usd=0.02, accumulated_usd=0.0299, unpriced_results=0))
        assert err.breach.max_cost_usd == 0.02
        assert err.breach.accumulated_usd == pytest.approx(0.0299)
        message = str(err)
        assert "2/4" in message
        assert "$0.0299" in message, "the spend that crossed the cap must be in the message"
        assert "$0.0200" in message, "the cap that was crossed must be in the message"

    def test_the_error_cannot_be_raised_without_its_measurement(self):
        """The breach is required: an unattributed stop is the defect, not a variant of it."""
        with pytest.raises(TypeError):
            BudgetStoppedError(2, 4)  # type: ignore[call-arg]


class TestTheCeilingsOriginIsResolvedBesideIt:
    """A resolved ceiling cannot say which tier supplied it, and only one tier moves.

    ``max_cost_usd`` is stored after the override-or-config cascade, so $15 on a run
    that named it and $15 on a run that inherited the configured default are the same
    number recording two different facts: re-launching the second under a changed
    setting runs under a different ceiling, while the first does not.
    """

    def test_a_launch_supplied_ceiling_is_chosen(self):
        assert EvalRunCostCap.resolve_ceiling_origin(0.02, enforcement_enabled=True) == "chosen"

    def test_an_omitted_ceiling_is_inherited(self):
        assert EvalRunCostCap.resolve_ceiling_origin(None, enforcement_enabled=True) == "inherited"

    def test_enforcement_off_is_uncapped_whatever_the_launch_asked_for(self):
        """Uncapped is a recording, not a blank: it is what distinguishes an unbounded
        run from a run whose writer recorded no ceiling, which store the same null ceiling.
        """
        assert EvalRunCostCap.resolve_ceiling_origin(0.02, enforcement_enabled=False) == "uncapped"
        assert (
            EvalRunCostCap.resolve_effective_ceiling(0.02, configured_max_cost_usd=15.0, enforcement_enabled=False)
            is None
        )

    def test_the_origin_agrees_with_the_ceiling_about_whether_one_was_in_force(self):
        """The two are written from one launch; disagreeing would leave a run claiming
        a cap it did not have, or an origin for a cap that is not there.

        Swept over enforcement too, because that is the input the caller now owns:
        the pair has to agree whichever way the host has it set.
        """
        for enforcing in (True, False):
            for override in (0.02, None):
                ceiling = EvalRunCostCap.resolve_effective_ceiling(
                    override, configured_max_cost_usd=15.0, enforcement_enabled=enforcing
                )
                origin = EvalRunCostCap.resolve_ceiling_origin(override, enforcement_enabled=enforcing)
                assert (ceiling is None) == (origin == "uncapped")
