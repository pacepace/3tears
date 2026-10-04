"""External spend is reported by the caller, in provider-agnostic units.

Eval holds no provider-specific parameter-to-cost table and interprets no provider-specific
parameter. The thing that made the call reports what it consumed — calls, the provider's own
weighted units *and the name of that unit*, money where the provider reports a charge — and
eval prices those units at an operator-declared rate per ``(provider, unit)``.

Three properties, one test class each:

1. A run's external calls price from a **supplied rate table**, with eval importing no tool
   module to do it.
2. Two providers metering in different units are priced at **their own** rates, and their
   units are **never summed**.
3. A zero-money provider and a provider with **no declared rate** are **distinguishable** —
   one is free, the other is a real cost nobody declared a rate for, and a single "unpriced"
   bucket cannot say which.
"""

from __future__ import annotations

import pytest

from threetears.evals.contracts.host.spend import ExternalSpend
from threetears.evals.contracts.spend import ExternalRateTable
from threetears.evals.contracts.usage_capture import RoleUsageLedger


class TestPricingComesFromASuppliedTable:
    """Eval multiplies a reported volume by a declared rate, and does nothing else."""

    def test_reported_units_price_at_the_declared_rate(self):
        ledger = RoleUsageLedger.for_external(ExternalRateTable(rates={("search_api", "credits"): 0.008}))
        ledger.add_external(ExternalSpend(provider="search_api", calls=3, provider_units=6, provider_unit="credits"))

        (row,) = ledger.rows()
        assert row.provider == "search_api"
        assert row.provider_unit == "credits"
        assert row.provider_units == 6
        assert row.call_count == 3
        assert row.cost_usd == pytest.approx(0.048)
        assert row.price_source == "search_api:configured_rate"

    def test_eval_never_re_derives_the_volume(self):
        """Doubling the reported units doubles the cost; the CALL count does not drive it.

        The defect this closes: eval used to compute a call's cost from a provider parameter
        it had to understand, so a ``basic`` search was billed at the calling tool's
        ``advanced`` rate. Volume now comes from the caller, and this fails if anything here
        starts deriving it from ``calls`` again.
        """
        table = ExternalRateTable(rates={("search_api", "credits"): 0.01})
        cheap = RoleUsageLedger.for_external(table)
        cheap.add_external(ExternalSpend(provider="search_api", calls=2, provider_units=2, provider_unit="credits"))
        dear = RoleUsageLedger.for_external(table)
        dear.add_external(ExternalSpend(provider="search_api", calls=2, provider_units=4, provider_unit="credits"))

        assert cheap.rows()[0].cost_usd == pytest.approx(0.02)
        assert dear.rows()[0].cost_usd == pytest.approx(0.04)

    def test_the_contracts_modules_import_no_tool_module(self):
        """The structural half of the property, checked at the source.

        ``test_extraction_import_boundary.py`` polices the whole package; this names the specific
        reach that must stay absent -- a host's tool layer, for a provider's cost table -- so a regression says
        *why* it matters here rather than only that a boundary moved. Whatever ``usage_capture``
        names is the standard library, pydantic, the logging package, or the engine itself.

        Each import is resolved to the module it names before it is judged, so a relative import
        walking out of the package is judged where it lands rather than read off the node.
        """
        import ast
        import sys
        from pathlib import Path

        from packages.evals.tests.import_resolution import absolute_module

        root = Path(__file__).resolve().parents[1] / "src"
        path = root / "threetears" / "evals" / "contracts" / "usage_capture.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            absolute_module(path, node, root=root) for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        assert any(m.startswith("threetears.evals.contracts") for m in imported), "the walk read no import"
        foreign = sorted(
            m
            for m in imported
            if m.split(".")[0] not in sys.stdlib_module_names
            and not any(
                m == root_ or m.startswith(root_ + ".")
                for root_ in ("threetears.evals", "pydantic", "threetears.observe")
            )
        )
        assert not foreign, (
            f"usage_capture must not reach outside the engine for a provider's cost table -- found: {foreign}"
        )


class TestTwoProvidersAreNeverBlended:
    """Different providers, different units: priced apart and summed never."""

    def test_each_provider_prices_at_its_own_rate_and_units_do_not_sum(self):
        ledger = RoleUsageLedger.for_external(
            ExternalRateTable(rates={("search_api", "credits"): 0.01, ("video_api", "quota_units"): 0.0001})
        )
        ledger.add_external(ExternalSpend(provider="search_api", calls=2, provider_units=4, provider_unit="credits"))
        ledger.add_external(
            ExternalSpend(provider="video_api", calls=1, provider_units=100, provider_unit="quota_units")
        )

        rows = {r.provider: r for r in ledger.rows()}
        assert set(rows) == {"search_api", "video_api"}
        assert rows["search_api"].provider_units == 4
        assert rows["search_api"].cost_usd == pytest.approx(0.04)
        assert rows["video_api"].provider_units == 100
        assert rows["video_api"].cost_usd == pytest.approx(0.01)

        # The fabricated quantity this exists to prevent: 4 credits + 100 quota units = 104
        # of nothing. No row carries it, because the two never share one.
        assert not any(r.provider_units == 104 for r in ledger.rows())

    def test_one_providers_units_are_not_priced_at_anothers_rate(self):
        """A provider absent from the table is unpriced — not priced at a neighbour's rate."""
        ledger = RoleUsageLedger.for_external(ExternalRateTable(rates={("search_api", "credits"): 0.01}))
        ledger.add_external(ExternalSpend(provider="calc_api", calls=1, provider_units=1, provider_unit="queries"))

        (row,) = ledger.rows()
        assert row.provider == "calc_api"
        assert row.provider_units == 1
        assert row.cost_usd is None, "an undeclared provider must not inherit another's rate"
        assert row.price_source is None

    def test_same_provider_different_units_stay_apart(self):
        """Qualification is by (provider, unit), not provider alone."""
        ledger = RoleUsageLedger.for_external(ExternalRateTable(rates={}))
        ledger.add_external(ExternalSpend(provider="p", calls=1, provider_units=2, provider_unit="credits"))
        ledger.add_external(ExternalSpend(provider="p", calls=1, provider_units=3, provider_unit="requests"))

        by_unit = {r.provider_unit: r.provider_units for r in ledger.rows()}
        assert by_unit == {"credits": 2, "requests": 3}

    def test_a_caller_that_cannot_name_its_provider_stays_visibly_unattributed(self):
        """The absent-provider arm, which is a state and not a spelling.

        A delivering tool that has not been taught to report WHO it called still made the
        calls, and they must still be counted. What must not happen is folding them into
        whichever provider happened to be present — so they land in their own row with a
        null provider, where a reader can see that nothing was attributable.
        """
        ledger = RoleUsageLedger.for_external(ExternalRateTable(rates={("search_api", "credits"): 0.01}))
        ledger.add_external(ExternalSpend(provider="search_api", calls=1, provider_units=2, provider_unit="credits"))
        ledger.add_external(ExternalSpend(provider=None, calls=4))

        rows = {r.provider: r for r in ledger.rows()}
        assert set(rows) == {"search_api", None}
        assert rows[None].call_count == 4
        assert rows[None].provider_units is None
        assert rows[None].cost_usd is None, "an unattributed call cannot be priced from a per-provider table"
        assert rows["search_api"].call_count == 1, "the attributed row must not absorb the unattributed calls"


class TestTheQualifiedFormHasOnePlaceItIsComposed:
    """The ``"<provider>:<unit>"`` spelling, and the case that has none.

    A second spelling appearing somewhere else would compare unequal to this one, and two
    spends from the same provider would then refuse to combine — the failure the
    qualification exists to prevent, reintroduced by the fix for it. Every surface that
    renders the form goes through here.
    """

    def test_a_named_provider_and_unit_compose_one_way(self):
        assert (
            ExternalSpend(provider="search_api", provider_units=2, provider_unit="credits").qualified_unit
            == "search_api:credits"
        )

    def test_an_unnamed_provider_has_no_qualified_form_rather_than_a_stand_in(self):
        """A unit nobody can attribute is comparable to nothing.

        Composing ``"unattributed:credits"`` would sort it beside real providers and let it
        combine with them, which is exactly what the qualification bars.
        """
        assert ExternalSpend(provider=None, provider_units=2, provider_unit="credits").qualified_unit is None

    def test_a_spend_that_meters_no_unit_has_no_qualified_form(self):
        assert ExternalSpend(provider="search_api", calls=3).qualified_unit is None


class TestFreeIsNotTheSameAsUnpriced:
    """A self-hosted zero and an undeclared rate must not read alike."""

    def test_a_zero_money_provider_and_an_unpriced_one_are_distinguishable(self):
        """The case the single-bucket design could not express.

        A self-hosted backend costs an observed zero. A paid provider nobody declared a rate
        for is a real cost that is simply unknown. Reporting both as "no dollars" tells an
        operator the same thing about two opposite situations.
        """
        ledger = RoleUsageLedger.for_external(ExternalRateTable(rates={}))
        # Self-hosted: reports its own charge, and that charge is zero.
        ledger.add_external(ExternalSpend(provider="self_hosted_search", calls=5, money=0.0))
        # Paid, metered, and nobody declared what a unit costs.
        ledger.add_external(ExternalSpend(provider="search_api", calls=5, provider_units=10, provider_unit="credits"))

        rows = {r.provider: r for r in ledger.rows()}
        free, unpriced = rows["self_hosted_search"], rows["search_api"]

        assert free.cost_usd == 0.0, "an observed zero is a measurement and must survive as one"
        assert free.price_source == "self_hosted_search:reported"
        assert unpriced.cost_usd is None, "an undeclared rate is unknown, not zero"
        assert unpriced.price_source is None
        # The distinction, stated the way a cost surface would have to read it.
        assert (free.cost_usd is None) != (unpriced.cost_usd is None)

    def test_a_reported_charge_beats_a_declared_rate(self):
        """An observation is not replaced by an estimate.

        If the provider says what it charged, that figure stands even when the operator also
        declared a rate — re-deriving would swap a measurement for a guess, and the price
        source has to say which one a reader is looking at.
        """
        ledger = RoleUsageLedger.for_external(ExternalRateTable(rates={("p", "credits"): 100.0}))
        ledger.add_external(ExternalSpend(provider="p", calls=1, provider_units=1, provider_unit="credits", money=0.25))

        (row,) = ledger.rows()
        assert row.cost_usd == pytest.approx(0.25)
        assert row.price_source == "p:reported"
