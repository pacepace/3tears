"""Witnessed provenance has a real source: the run records it, and the analysis reads it from there.

Whether a run's apparatus was set before the fact (``commissioned``) or found after it
(``witnessed``) is the difference between an experiment and a log. The bundle used to stamp every
observation with one hardcoded value, so a captured session and a launched arm of the same variant
pooled into one cell. Now:

- every run states its provenance, and a run that does not is refused;
- the launch path stamps ``commissioned`` on every run it starts;
- the bundle reads the provenance off each run, and it enters the apparatus class id — so a
  witnessed and a commissioned observation of one variant are two cells on every per-cell surface,
  not only in the cell algebra;
- what the campaign declared held fixed is compared with what the runs recorded.

Mutations that turn this file red: restoring ``provenance="commissioned"`` in the bundle's
``_apparatus_classes`` (the hardcoded value, renamed); dropping ``provenance`` from the class digest
in ``apparatus_class_of``; giving ``EvalRun.apparatus_provenance`` a default; deleting the launch
path's stamp; inverting the contradiction test in ``_held_fixed_reading``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle
from threetears.evals.contracts import EvalResult, EvalRun
from threetears.evals.contracts.campaign import EvalCampaign
from threetears.evals.contracts.declaration import ControlDeclaration
from threetears.evals.quick import callable_host, run_eval
from threetears.evals.run import list_runs
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

#: The id of the commissioned twin of the toy host's narrow batch.
_COMMISSIONED = "commissioned-narrow"


def _with_a_commissioned_twin(
    *, held_fixed: ControlDeclaration | None = None
) -> tuple[EvalCampaign, list[EvalRun], dict[str, list[EvalResult]]]:
    """The toy campaign (two witnessed batches) plus a commissioned batch of the narrow variant.

    The twin carries the narrow batch's every value and every observation — the same variant under
    the same recorded rig — and differs only in having been commissioned.

    Args:
        held_fixed: What the campaign declares held fixed; ``None`` keeps the toy host's own.

    Returns:
        The campaign, its runs (narrow, wide, twin) and each run's results.
    """
    campaign, storage = toyhost_campaign()
    narrow_id, wide_id = campaign.run_ids
    narrow, wide = (storage.load_eval_run(run_id, TOYHOST_SCOPE) for run_id in (narrow_id, wide_id))
    assert narrow is not None and wide is not None
    assert (narrow.apparatus_provenance, wide.apparatus_provenance) == ("witnessed", "witnessed")
    twin = narrow.model_copy(update={"id": _COMMISSIONED, "apparatus_provenance": "commissioned"})
    results = {run_id: storage.query_eval_results_by_run(run_id, TOYHOST_SCOPE) for run_id in (narrow_id, wide_id)}
    results[twin.id] = [
        result.model_copy(update={"id": f"twin-{result.id}", "eval_run_id": twin.id}) for result in results[narrow_id]
    ]
    design = campaign.declared_design
    assert design is not None
    if held_fixed is not None:
        design = design.model_copy(update={"held_fixed": held_fixed})
    campaign = campaign.model_copy(update={"run_ids": [narrow_id, wide_id, twin.id], "declared_design": design})
    return campaign, [narrow, wide, twin], results


def _assemble(
    campaign: EvalCampaign, runs: Sequence[EvalRun], results: Mapping[str, list[EvalResult]]
) -> AnalysisContextBundle:
    return assemble_context_bundle(campaign, storage=ToyhostStorage(runs, results), profile=toyhost_profile())


def _cell_of_run(bundle: AnalysisContextBundle, run_id: str) -> Any:
    (cell,) = [cell for cell in bundle.cell_measures if run_id in cell.run_ids]
    return cell


class TestARunStatesItsProvenance:
    """Only the writer of a run knows how its apparatus came to be, so nothing may assume it."""

    def test_a_run_without_one_is_refused(self) -> None:
        fields = make_eval_run().model_dump()
        del fields["apparatus_provenance"]

        with pytest.raises(ValidationError, match="apparatus_provenance"):
            EvalRun(**fields)

    def test_the_bundles_old_word_is_not_a_provenance(self) -> None:
        """`declared` was the hardcoded stamp; the declaration's own words are the only two."""
        with pytest.raises(ValidationError, match="apparatus_provenance"):
            make_eval_run(apparatus_provenance="declared")

    @pytest.mark.parametrize("provenance", ["commissioned", "witnessed"])
    def test_both_words_are_accepted_and_survive_a_round_trip(self, provenance: str) -> None:
        run = make_eval_run(apparatus_provenance=provenance)

        assert EvalRun.from_dict(run.to_dict()).apparatus_provenance == provenance


async def _echo(case: Mapping[str, Any]) -> Any:
    return case["n"]


def answered(case: Mapping[str, Any], answer: Any) -> float:
    return 1.0


async def test_the_launch_path_stamps_every_run_it_starts_commissioned() -> None:
    """A launch is the act of fixing a rig and measuring against it."""
    host = callable_host([answered])

    await run_eval([{"n": 1}], _echo, [answered], scope_id="provenance", host=host)

    (run,) = list_runs(host, "provenance")
    assert run.apparatus_provenance == "commissioned"


class TestTheBundleReadsProvenanceOffTheRun:
    """A witnessed and a commissioned observation of one variant land in separate cells — everywhere."""

    def test_they_are_two_cells_of_one_variant_and_the_refusal_says_why(self) -> None:
        campaign, runs, results = _with_a_commissioned_twin()
        bundle = _assemble(campaign, runs, results)

        narrow_variant = _cell_of_run(bundle, runs[0].id).variant_key
        cells = [cell for cell in bundle.cells if cell.variant_key == narrow_variant]
        assert sorted(cell.provenance for cell in cells) == ["commissioned", "witnessed"], (
            "the same variant under the same recorded rig, set once and found once, is two cells"
        )
        assert len({cell.apparatus_class_id for cell in cells}) == 2
        assert [(r.variant_key, r.reason) for r in bundle.refused_merges] == [(narrow_variant, "provenance_differs")]

    def test_every_per_cell_surface_keeps_both_cells(self) -> None:
        """The surfaces key a cell by (variant, class); a shared class would keep one and drop the other."""
        campaign, runs, results = _with_a_commissioned_twin()
        bundle = _assemble(campaign, runs, results)

        witnessed, commissioned = _cell_of_run(bundle, runs[0].id), _cell_of_run(bundle, _COMMISSIONED)
        assert witnessed.variant_key == commissioned.variant_key
        assert witnessed.apparatus_class_id != commissioned.apparatus_class_id
        assert len(bundle.cell_measures) == 3
        assert witnessed.n_observations == commissioned.n_observations == len(results[_COMMISSIONED])
        adjudicated = [bar for bar in bundle.bar_adjudications if bar.state == "adjudicated"]
        assert adjudicated, "the toy campaign adjudicates bars, so the verdict count below is not vacuous"
        for bar in adjudicated:
            assert len(bar.verdicts) == 3, f"{bar.measure_id} gave a verdict per cell, both twins included"

    def test_a_campaign_of_one_provenance_pools_as_before(self) -> None:
        """The positive control: nothing here splits observations that share a provenance."""
        campaign, storage = toyhost_campaign()
        bundle = assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile())

        assert {cell.provenance for cell in bundle.cells} == {"witnessed"}
        assert bundle.refused_merges == []
        assert len(bundle.cells) == len(campaign.run_ids)


class TestTheDeclaredControlsAreReadAgainstTheRuns:
    """The declaration and the runs use the same two words, so they are compared value for value."""

    def test_a_run_contradicting_the_declared_apparatus_is_named(self) -> None:
        campaign, runs, results = _with_a_commissioned_twin()
        reading = _assemble(campaign, runs, results).held_fixed_reading

        assert reading.declared_apparatus == "witnessed"
        assert reading.run_provenance[_COMMISSIONED] == "commissioned"
        assert reading.contradicting_run_ids == [_COMMISSIONED]
        assert reading.disclosure is not None
        assert "declares its apparatus witnessed, but 1 of its 3 resolved runs recorded otherwise" in reading.disclosure

    def test_runs_that_agree_with_the_declaration_contradict_nothing(self) -> None:
        campaign, storage = toyhost_campaign()
        reading = assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile()).held_fixed_reading

        assert reading.contradicting_run_ids == []
        assert reading.disclosure is not None and "recorded otherwise" not in reading.disclosure

    def test_a_mix_with_nothing_declared_is_disclosed(self) -> None:
        campaign, runs, results = _with_a_commissioned_twin()
        reading = _assemble(campaign.model_copy(update={"declared_design": None}), runs, results).held_fixed_reading

        assert reading.declared_apparatus is None
        assert reading.contradicting_run_ids == [], "nothing was declared, so nothing is contradicted"
        assert reading.disclosure is not None and "declares nothing held fixed" in reading.disclosure

    def test_an_uncontrolled_stimulus_is_said_with_its_reason(self) -> None:
        campaign, storage = toyhost_campaign()
        reading = assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile()).held_fixed_reading

        assert reading.declared_stimulus == "uncontrolled"
        assert reading.disclosure == (
            "The stimulus was not held fixed: documents arrive from live intake; the vendor mix drifts week to week"
        )

    def test_a_controlled_agreeing_campaign_has_nothing_to_disclose(self) -> None:
        controlled = ControlDeclaration(stimulus="controlled", apparatus="witnessed")
        campaign, storage = toyhost_campaign()
        design = campaign.declared_design
        assert design is not None
        campaign = campaign.model_copy(update={"declared_design": design.model_copy(update={"held_fixed": controlled})})

        reading = assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile()).held_fixed_reading

        assert reading.disclosure is None
