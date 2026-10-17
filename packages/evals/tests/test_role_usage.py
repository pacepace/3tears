"""Unit tests for per-role usage capture — one token/cost row per (role, model).

Covers three contracts:

- the ``RoleUsage`` schema, including *missing != zero* (``int | None`` token
  fields, ``float | None`` cost) so a cost-only external row or an unreported
  reasoning split stays ``None`` rather than a fabricated 0;
- ``EvalResult.usage`` is always an observation — ``[]`` (captured, no roles) or a
  populated list, distinct through the storage ``to_dict``/``from_dict`` path (the
  persistence seam) — and a result without one is refused;
- the candidate accumulator that derives one candidate ``RoleUsage`` from the
  turn records a runner pass produces;
- the external role's rate card, which turns counted search calls into the
  provider's own metered unit (credits) and then into dollars — the second step
  only when the account's rate was declared, so "unpriced" stays a reportable
  state rather than becoming a zero.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.evals.schema.external_spend import ExternalSpend
from threetears.evals.schema.models import EvalResult, RoleUsage
from threetears.evals.kernel.spend import ExternalRateTable
from threetears.evals.kernel.usage_capture import RoleUsageLedger, blended_cost_roles
from packages.evals.tests.factories import result_capture_defaults

#: The rate a priced run in this module is measured against, and the volume the search
#: sub-tool reports at ``advanced``. Named because several tests below assert the product.
_SEARCH_RATE = ExternalRateTable(rates={("search_api", "credits"): 0.008})


def _search(*, calls: int, credits: int | None) -> ExternalSpend:
    """One search-API contribution, as the tool that made the calls reports it."""
    return ExternalSpend(
        provider="search_api",
        calls=calls,
        provider_units=credits,
        provider_unit="credits" if credits is not None else None,
    )


def _eval_result(**overrides) -> EvalResult:
    base = dict(
        result_capture_defaults(),
        scope_id="u",
        eval_run_id="run-1",
        test_case_id="tc-1",
        model="anthropic/claude-haiku-4-5",
        k_iteration=1,
    )
    base.update(overrides)
    return EvalResult(**base)


class TestRoleUsageModel:
    def test_candidate_row_round_trips_through_model(self):
        row = RoleUsage(
            role="candidate",
            model="m",
            prompt_tokens=100,
            completion_tokens=40,
            reasoning_tokens=12,
            cost_usd=0.002,
            price_source="gateway",
        )
        again = RoleUsage.model_validate(row.model_dump(mode="json"))
        assert again == row

    def test_external_cost_only_row_keeps_none_tokens(self):
        # missing != zero: an external (e.g. a search API) row has no token data.
        row = RoleUsage(role="external", cost_usd=0.01, price_source="search_api")
        assert row.prompt_tokens is None
        assert row.completion_tokens is None
        assert row.reasoning_tokens is None
        again = RoleUsage.model_validate(row.model_dump(mode="json"))
        assert again.prompt_tokens is None
        assert again.reasoning_tokens is None
        assert again.cost_usd == pytest.approx(0.01)

    def test_reasoning_none_is_distinct_from_zero(self):
        assert RoleUsage(role="candidate").reasoning_tokens is None
        assert RoleUsage(role="candidate", reasoning_tokens=0).reasoning_tokens == 0

    def test_role_is_constrained_to_the_five_values(self):
        for role in ("candidate", "judge", "simulator", "inner_agent", "external"):
            assert RoleUsage(role=role).role == role
        with pytest.raises(Exception):
            RoleUsage(role="not-a-role")

    def test_negative_tokens_rejected(self):
        with pytest.raises(Exception):
            RoleUsage(role="candidate", prompt_tokens=-1)


class TestUsageIsAlwaysAnObservation:
    """``usage`` is a list on every result — ``[]`` (captured, no roles) or populated — never absent."""

    def test_a_result_with_no_usage_is_refused(self):
        """The runner writes a list at every exit, so a result without one is not a shape anything produces."""
        with pytest.raises(ValidationError, match="usage"):
            _eval_result(usage=None)
        base = _eval_result().model_dump()
        del base["usage"]
        with pytest.raises(ValidationError, match="usage"):
            EvalResult(**base)

    def test_empty_and_populated_are_distinct_through_storage_to_dict_from_dict(self):
        # to_dict()/from_dict() is the persistence path (storage.save_eval_result → load_eval_result).
        populated = [
            RoleUsage(
                role="candidate",
                prompt_tokens=5,
                completion_tokens=2,
                reasoning_tokens=1,
                cost_usd=0.001,
                price_source="gateway",
            )
        ]
        for usage in ([], populated):
            assert EvalResult.from_dict(_eval_result(usage=usage).to_dict()).usage == usage


class TestExternalRatePricing:
    """Reported external volume becomes dollars, or honestly stays unpriced.

    Two halves fail independently — the provider's own unit count (knowable only to the
    caller that made the call) and the operator's declared rate — so the tests below pin
    all three reachable states: both known, units-only, and neither.
    """

    def test_a_declared_rate_puts_dollars_on_the_reported_units(self):
        led = RoleUsageLedger.for_external(_SEARCH_RATE)
        led.add_external(_search(calls=3, credits=6))

        row = led.rows()[0]
        assert row.call_count == 3
        assert row.provider == "search_api"
        assert row.provider_unit == "credits"
        assert row.provider_units == 6
        assert row.cost_usd == pytest.approx(0.048)
        assert row.price_source == "search_api:configured_rate", (
            "an operator-declared rate is not the provider's own reported charge, and a reader "
            "deciding whether to re-derive at current rates has to be able to tell them apart"
        )

    def test_the_reported_volume_is_what_sets_the_total_not_the_call_count(self):
        """The defect this shape closes: eval no longer re-derives units from a parameter.

        Two ledgers with identical call counts and different reported units must price
        differently — which is what a ``basic`` search costing half an ``advanced`` one
        looks like once the caller is the thing that says so.
        """
        basic = RoleUsageLedger.for_external(_SEARCH_RATE)
        basic.add_external(_search(calls=3, credits=3))

        assert basic.rows()[0].provider_units == 3
        assert basic.rows()[0].cost_usd == pytest.approx(0.024)

    def test_units_survive_an_undeclared_dollar_rate(self):
        """Volume is knowable without a price, and saying so is the point of the split.

        An operator who never declared a rate still gets the quantity their provider bills
        in — the one number the spend can be reconstructed from by hand — while the dollars
        stay absent rather than becoming a zero that reads as free.
        """
        led = RoleUsageLedger.for_external(ExternalRateTable(rates={}))
        led.add_external(_search(calls=4, credits=8))

        row = led.rows()[0]
        assert row.provider_units == 8
        assert row.cost_usd is None
        assert row.price_source is None, "no dollars were produced, so nothing has a provenance to name"

    def test_an_uncountable_call_prices_nothing_rather_than_assuming_a_volume(self):
        """A call whose unit the caller could not state must not fall through to a default.

        Assuming one would price a run at whatever the tool default happened to be that
        month, which is the guessed constant the whole rate table exists to avoid.
        """
        led = RoleUsageLedger.for_external(_SEARCH_RATE)
        led.add_external(_search(calls=2, credits=None))

        row = led.rows()[0]
        assert row.call_count == 2, "the volume is still an observation and still reported"
        assert row.provider == "search_api", "and it is still attributable, which is what lets it be priced by hand"
        assert row.provider_units is None
        assert row.cost_usd is None

    def test_the_composition_names_external_only_when_dollars_could_be_folded_in(self):
        """``cost_roles`` is a claim about ``cost_usd``, so it tracks the dollars, not the calls.

        A run that counted units but priced none of them has a total no external call
        contributed to — naming ``external`` there would make the total look complete.
        """
        assert blended_cost_roles(_SEARCH_RATE)[-1] == "external"
        assert "external" not in blended_cost_roles(ExternalRateTable(rates={}))
        assert "external" not in blended_cost_roles(None)

    def test_the_composition_is_a_convention_not_an_observation(self):
        """A priced run that made no external call still names the role.

        Same rule the judge role carries: the marker states what this run's totals sum
        over, so deriving it from whether a given cell happened to search would make every
        quiet cell its own epoch.
        """
        assert "external" in blended_cost_roles(_SEARCH_RATE)
