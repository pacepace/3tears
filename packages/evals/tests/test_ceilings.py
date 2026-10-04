"""Tests for the shared ceiling cascade.

:mod:`threetears.evals.run.ceilings` holds the three steps both of a run's per-run bounds resolve
through — override-or-default, the enforcement-off short circuit, and which tier answered —
and this file covers the two things that make the de-duplication worth having:

- **The cascade answers in the currency it was asked in.** Dollars come back as ``float`` and
  calls as ``int``, from one implementation, because :data:`~threetears.evals.run.ceilings.Ceiling`
  is a constrained ``TypeVar`` rather than a second copy.
- **Both classes actually delegate.** Each of the three functions is replaced in turn and both
  :class:`~threetears.evals.run.budget.EvalRunCostCap` and
  :class:`~threetears.evals.run.metering.MeteredCallLedger` are asked the same question again. A class
  that re-inlined the cascade — the state this issue existed to end — answers from its own copy
  and the assertion fails. This is what stands behind the acceptance criterion that a change to
  the origin vocabulary cannot land in one class only.

The constructor asymmetry the two classes keep is deliberately NOT covered here. It is asserted
in both directions where it lives — ``test_budget.py`` and ``test_metered_calls.py`` — and a
third copy in this file would be the same duplication one layer up.
"""

from __future__ import annotations

import pytest

from threetears.evals.run import ceilings
from threetears.evals.run.budget import EvalRunCostCap
from threetears.evals.run.metering import MeteredCallLedger

# Deliberately far apart, and different between the two currencies. A transposed override and
# default, or a cost value that reached the call ledger, changes the answer rather than
# returning a number that still looks reasonable.
_OVERRIDE_USD = 3.5
_CONFIGURED_USD = 91.0
_OVERRIDE_CALLS = 7
_CONFIGURED_CALLS = 250


class TestTheCascadeAnswersInTheCurrencyItWasAsked:
    """One implementation over two currencies, without widening either caller's type."""

    def test_dollars_resolve_to_dollars(self):
        assert ceilings.resolve_ceiling(_OVERRIDE_USD, configured=_CONFIGURED_USD) == _OVERRIDE_USD
        assert ceilings.resolve_ceiling(None, configured=_CONFIGURED_USD) == _CONFIGURED_USD

    def test_calls_resolve_to_calls_and_stay_integers(self):
        """``int`` in, ``int`` out — a float ceiling would compare fine and then be unprintable
        as a call count in the ledger's refusal message.
        """
        resolved = ceilings.resolve_ceiling(None, configured=_CONFIGURED_CALLS)
        assert resolved == _CONFIGURED_CALLS
        assert type(resolved) is int

    def test_enforcement_off_drops_the_ceiling_in_either_currency(self):
        assert (
            ceilings.resolve_effective_ceiling(_OVERRIDE_USD, configured=_CONFIGURED_USD, enforcement_enabled=False)
            is None
        )
        assert (
            ceilings.resolve_effective_ceiling(_OVERRIDE_CALLS, configured=_CONFIGURED_CALLS, enforcement_enabled=False)
            is None
        )

    def test_the_origin_names_the_tier_without_being_told_the_currency(self):
        assert ceilings.resolve_ceiling_origin(_OVERRIDE_USD, enforcement_enabled=True) == "chosen"
        assert ceilings.resolve_ceiling_origin(_OVERRIDE_CALLS, enforcement_enabled=True) == "chosen"
        assert ceilings.resolve_ceiling_origin(None, enforcement_enabled=True) == "inherited"
        assert ceilings.resolve_ceiling_origin(None, enforcement_enabled=False) == "uncapped"


def test_the_override_and_the_default_cannot_be_transposed():
    """The argument-swap detector, made structural rather than reviewed.

    ``float`` and ``int`` are mutually assignable and this repo runs no static type gate, so a
    caller that passed the default where the override goes would resolve a plausible number
    with nothing going red — the failure mode removing the ``from_config`` factories was
    written up for. Every argument after the override is keyword-only, so the transposition is
    a ``TypeError`` at the call site instead of a wrong ceiling in a stored run.
    """
    with pytest.raises(TypeError):
        ceilings.resolve_ceiling(_OVERRIDE_USD, _CONFIGURED_USD)  # type: ignore[misc]
    with pytest.raises(TypeError):
        ceilings.resolve_effective_ceiling(_OVERRIDE_CALLS, _CONFIGURED_CALLS, True)  # type: ignore[misc]


class TestBothClassesReadTheOneCascade:
    """Replace a step of the cascade and ask both classes again.

    The de-duplication's whole payoff: a run document carries both origins side by side, so a
    later change to the origin vocabulary — or to what enforcement-off means — landing in one
    copy would leave a run recording two answers computed under two rules with nothing failing.
    These substitutions are what make that structurally impossible, and they work only because
    both modules call through the module object (``ceilings.resolve_*``) rather than binding
    the functions by name at import.
    """

    def test_a_new_origin_value_reaches_both_classes(self, monkeypatch):
        monkeypatch.setattr(ceilings, "resolve_ceiling_origin", lambda override, *, enforcement_enabled: "renamed")

        assert EvalRunCostCap.resolve_ceiling_origin(_OVERRIDE_USD, enforcement_enabled=True) == "renamed"
        assert MeteredCallLedger.resolve_ceiling_origin(_OVERRIDE_CALLS, enforcement_enabled=True) == "renamed"

    def test_a_changed_enforcement_off_rule_reaches_both_recorded_ceilings(self, monkeypatch):
        monkeypatch.setattr(
            ceilings, "resolve_effective_ceiling", lambda override, *, configured, enforcement_enabled: -1
        )

        assert (
            EvalRunCostCap.resolve_effective_ceiling(
                None, configured_max_cost_usd=_CONFIGURED_USD, enforcement_enabled=False
            )
            == -1
        )
        assert (
            MeteredCallLedger.resolve_effective_ceiling(
                None, configured_max_metered_calls=_CONFIGURED_CALLS, enforcement_enabled=False
            )
            == -1
        )

    def test_a_changed_override_rule_reaches_the_cap_the_run_is_bounded_by(self, monkeypatch):
        """The cost cap is the one that reads the UNFILTERED ceiling, so it is the class this
        substitution has to reach — the ledger's ``for_run`` goes through the effective one
        covered above, which is the asymmetry the two constructors keep.
        """
        monkeypatch.setattr(ceilings, "resolve_ceiling", lambda override, *, configured: 42.0)

        cap = EvalRunCostCap.for_run(
            "run-1", _OVERRIDE_USD, configured_max_cost_usd=_CONFIGURED_USD, enforcement_enabled=True
        )
        assert cap.max_cost_usd == 42.0


class TestTheDelegationCarriesTheRightArgumentsInBothCurrencies:
    """Exercise the trio from both classes with values a mix-up would change.

    Delegation that compiles is not delegation that passes the right things: the override and
    the default are the same type, so a class threading them the wrong way round produces a
    number rather than an error. Every value below differs from every other, so a transposed
    pair or a cross-currency leak lands on a figure no assertion here accepts.
    """

    def test_the_cost_cap_bounds_the_run_at_its_override_not_its_default(self):
        cap = EvalRunCostCap.for_run(
            "run-1", _OVERRIDE_USD, configured_max_cost_usd=_CONFIGURED_USD, enforcement_enabled=True
        )
        assert cap.max_cost_usd == _OVERRIDE_USD

    def test_the_cost_cap_inherits_the_default_when_the_launch_named_nothing(self):
        cap = EvalRunCostCap.for_run("run-1", None, configured_max_cost_usd=_CONFIGURED_USD, enforcement_enabled=True)
        assert cap.max_cost_usd == _CONFIGURED_USD

    def test_the_ledger_bounds_the_run_at_its_override_not_its_default(self):
        ledger = MeteredCallLedger.for_run(
            "run-1", _OVERRIDE_CALLS, configured_max_metered_calls=_CONFIGURED_CALLS, enforcement_enabled=True
        )
        assert ledger.ceiling == _OVERRIDE_CALLS

    def test_the_ledger_inherits_the_default_when_the_launch_named_nothing(self):
        ledger = MeteredCallLedger.for_run(
            "run-1", None, configured_max_metered_calls=_CONFIGURED_CALLS, enforcement_enabled=True
        )
        assert ledger.ceiling == _CONFIGURED_CALLS

    def test_the_recorded_ceiling_is_the_one_that_bounds_each_class(self):
        """Both directions of the pairing the cascade exists to keep, in both currencies."""
        assert (
            EvalRunCostCap.resolve_effective_ceiling(
                _OVERRIDE_USD, configured_max_cost_usd=_CONFIGURED_USD, enforcement_enabled=True
            )
            == EvalRunCostCap.for_run(
                "run-1", _OVERRIDE_USD, configured_max_cost_usd=_CONFIGURED_USD, enforcement_enabled=True
            ).max_cost_usd
        )
        assert (
            MeteredCallLedger.resolve_effective_ceiling(
                _OVERRIDE_CALLS, configured_max_metered_calls=_CONFIGURED_CALLS, enforcement_enabled=True
            )
            == MeteredCallLedger.for_run(
                "run-1", _OVERRIDE_CALLS, configured_max_metered_calls=_CONFIGURED_CALLS, enforcement_enabled=True
            ).ceiling
        )
