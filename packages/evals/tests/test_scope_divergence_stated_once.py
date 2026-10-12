"""A whole-run swing is stated once per lever pair, however many parts it diverged from (Discodon #2716).

The symptom, on a model sweep: each model pair's "``total_ms`` moved −133184, carried by ``llm_ms`` (100% of it)"
reached the reader four times, and "contained by ``total_ms``" twice. Two causes could produce it, and this module
holds both to account on campaigns built to provoke them.

- **A surface that folds into the lever.** A resolved surface whose movement is its knob's is skipped by the
  divergence lens (``_SurfaceFolds.folds_away``); compared as a lever of its own it would restate every divergence
  the knob reports. ``TestASurfaceThatFoldsIntoTheLeverAddsNoDivergence`` sweeps a model knob together with the
  surface it is written into, at several runs a level and at three levels, and finds one divergence per lever pair
  and whole-and-part pair. That skip already held: this block is its regression test.
- **One whole diverging from several parts.** A divergence is a whole-and-part pair, and ``total_ms`` has a part
  for every subsystem millisecond measure: ``tool_ms`` and ``orchestration_ms`` (declared components of it),
  ``async_wait_ms`` and ``judge_ms`` (disjoint from it). With the movement all in ``llm_ms`` the four remainders
  against the others all move, so one model pair publishes four divergences — two of them contained by
  ``total_ms``, which is the four-and-two the symptom counted. Where the whole's movement went is a fact about
  the whole and the lever pair, not about any part, and it was copied onto each of the four.
  ``TestWhereAWholeRunSwingWentIsStatedOnce`` pins that it is now stated once, in ``whole_swings``, and
  ``TestAReporterCaseFrozenBeforeTheMoveStillRebuilds`` that a case frozen over the older shape keeps every value.
"""

from __future__ import annotations

import json
import random
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle
from threetears.evals.analysis.generator import build_user_message
from threetears.evals.analysis.reporter_kind import ReporterCase, rebuild_bundle
from threetears.evals.kernel import EvalCampaign
from threetears.evals.kernel.host import HostProfile, KindContract, ResolvesInto, Sweepable, freeze
from threetears.evals.schema import EvalRun, LatencyMetrics, SweepableValue
from threetears.evals.schema.hashing import canonical_digest
from packages.evals.tests.fixtures.toyhost.contract import TOY_EXTRACTOR_CONTRACT
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    ToyhostStorage,
    toyhost_batch,
    toyhost_measurements,
)
from packages.evals.tests.fixtures.toyhost.kind import TOY_EXTRACTOR_KIND
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.sweepables import TOYHOST_SWEEPABLE_REGISTRY
from packages.evals.tests.fixtures.toyhost.variant import variant_levers

if TYPE_CHECKING:
    from threetears.evals.schema import EvalResult

#: The extractor's model knob, and the host surface it is resolved into.
_MODEL = "extractor.model"
_PARAMS = "llm_parameters"

#: Where the toy host keeps its own keys on a run.
_NAMESPACE = "toyhost"

#: A fixed namespace for the ids this module derives, so a run's id is a function of what it is.
_IDS = uuid.UUID("5d1e7c2a-9b34-4f80-a6c1-3e2b7d9f0a48")

#: Time in model calls at each model, in ms: far enough apart that every pair's whole-run swing separates.
_LLM_MS = {"m-a": 140000.0, "m-b": 6800.0, "m-c": 70000.0}

#: The parts ``total_ms`` is compared against, other than the one carrying the swing.
_STILL_PARTS = {"tool_ms", "orchestration_ms", "async_wait_ms", "judge_ms"}


def _recorded_params(run: EvalRun, _results: Sequence[EvalResult]) -> str | None:
    """The content hash of the model parameters a run resolved, ``None`` where it recorded none."""
    value = (run.host_payload or {}).get(_NAMESPACE, {}).get(_PARAMS)
    return None if value is None else SweepableValue.of(value, display=_PARAMS).content_hash


class _Overlays(BaseModel):
    """The extractor's model, written into the parameters the host records resolved."""

    model: Annotated[Literal["m-a", "m-b", "m-c"], ResolvesInto(_PARAMS)] = Field(
        "m-a", description="which model the extractor calls"
    )


def _contract() -> KindContract:
    return KindContract(TOY_EXTRACTOR_KIND, overlays=_Overlays, prefix="extractor", seats=TOY_EXTRACTOR_CONTRACT.seats)


def _variant_levers(run: EvalRun) -> dict[str, SweepableValue]:
    """The toy host's own levels, plus the resolved parameters — the coordinates the host writes."""
    payload = (run.host_payload or {}).get(_NAMESPACE, {})
    return {**variant_levers(run), _PARAMS: SweepableValue.of(payload.get(_PARAMS), display=_PARAMS)}


def _profile() -> HostProfile:
    """The toy host, with the model knob on its extractor and the surface it resolves into registered."""
    surface = Sweepable(
        name=_PARAMS,
        role="lever",
        read=_recorded_params,
        reader_prose="the model parameters the extractor resolved, after any overlaid knob",
    )
    return replace(
        toyhost_profile(),
        host_sweepables=TOYHOST_SWEEPABLE_REGISTRY.extend((surface,)),
        variant_levers=_variant_levers,
        kinds=(_contract(),),
    )


def _batch(model: str, repeat: int) -> EvalRun:
    """One toy batch at ``model``, the ``repeat``-th of its arm, recording the parameters it resolved."""
    batch = toyhost_batch(
        chunk_tokens=512,
        retriever_top_k=3,
        extraction_schema="v1",
        ocr_engine_version="tess-5.3.1",
        reviewer_pool="pool-a",
        **{_PARAMS: {"model": model, "max_output_tokens": 4096}},
    )
    return batch.model_copy(
        update={
            "id": str(uuid.uuid5(_IDS, f"{model}|{repeat}")),
            "overlays": freeze(_contract().validate_overlays({"model": model})),
        }
    )


def _timed(result: EvalResult, model: str, rng: random.Random) -> EvalResult:
    """``result`` with its latency partitioned: the model's time in model calls, every other part held still."""
    llm = round(_LLM_MS[model] + rng.gauss(0.0, 400.0), 3)
    tool = round(3000.0 + rng.gauss(0.0, 150.0), 3)
    orchestration = round(900.0 + rng.gauss(0.0, 60.0), 3)
    latency = LatencyMetrics(
        total_ms=round(llm + tool + orchestration, 3),
        llm_ms=llm,
        tool_ms=tool,
        async_wait_ms=round(400.0 + rng.gauss(0.0, 30.0), 3),
        judge_ms=round(2500.0 + rng.gauss(0.0, 120.0), 3),
    )
    return result.model_copy(update={"latency": latency})


def _bundle(models: Sequence[str], *, runs_per_level: int, seed: str) -> AnalysisContextBundle:
    """A model sweep at ``runs_per_level`` runs a model, every one resolving the surface the model is written into."""
    profile = _profile()
    rng = random.Random(f"divergence-stated-once-{seed}")
    batches = [_batch(model, repeat) for model in models for repeat in range(runs_per_level)]
    results = {
        batch.id: [
            _timed(result, batch.overlays["model"], rng)
            for result in toyhost_measurements(
                batch, profile=profile, cost_usd=0.02, total_ms=900.0, field_accuracy=0.8
            )
        ]
        for batch in batches
    }
    campaign = EvalCampaign(
        id="2f6c8a1e-4d3b-4e97-b5a2-7c9d0e1f3a64",
        scope_id=TOYHOST_SCOPE,
        name="model",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id="",
        run_ids=[batch.id for batch in batches],
        created_by="test:fixture",
    )
    return assemble_context_bundle(campaign, storage=ToyhostStorage(batches, results), profile=profile)


def _pairs(models: Sequence[str]) -> set[tuple[str, str]]:
    """Every pair of levels, in the name order a divergence states its two levels in."""
    ordered = sorted(models)
    return {(a, b) for index, a in enumerate(ordered) for b in ordered[index + 1 :]}


#: Two levels at one run each; two levels at several runs each; three levels at several runs each.
_SWEEPS = pytest.mark.parametrize(
    ("models", "runs_per_level"),
    [(("m-a", "m-b"), 1), (("m-a", "m-b"), 3), (("m-a", "m-b", "m-c"), 2)],
    ids=["two-levels", "several-runs-a-level", "three-levels"],
)


class TestASurfaceThatFoldsIntoTheLeverAddsNoDivergence:
    @_SWEEPS
    def test_the_surface_is_compared_as_no_lever_of_its_own(self, models: Sequence[str], runs_per_level: int) -> None:
        bundle = _bundle(models, runs_per_level=runs_per_level, seed="fold")
        assert {divergence.lever for divergence in bundle.scope_divergences} == {_MODEL}

    @_SWEEPS
    def test_each_lever_pair_diverges_once_from_each_part(self, models: Sequence[str], runs_per_level: int) -> None:
        bundle = _bundle(models, runs_per_level=runs_per_level, seed="fold")
        keys = Counter(
            (
                divergence.lever,
                divergence.level_a,
                divergence.level_b,
                divergence.end_to_end.name,
                divergence.subsystem.name,
            )
            for divergence in bundle.scope_divergences
        )
        assert set(keys.values()) == {1}, keys
        # Every pair's whole moved by its model's time in model calls, away from each part held still: one
        # divergence per pair and part, published or counted as dropped by the cap (three levels reach it).
        expected = {(_MODEL, a, b, "total_ms", part) for a, b in _pairs(models) for part in _STILL_PARTS}
        assert set(keys) <= expected
        assert len(keys) + bundle.divergences_omitted == len(expected)


class TestWhereAWholeRunSwingWentIsStatedOnce:
    @_SWEEPS
    def test_one_swing_per_lever_pair_and_whole(self, models: Sequence[str], runs_per_level: int) -> None:
        bundle = _bundle(models, runs_per_level=runs_per_level, seed="swing")
        keys = [(swing.lever, swing.level_a, swing.level_b, swing.whole) for swing in bundle.whole_swings]
        published = {(d.lever, d.level_a, d.level_b, d.end_to_end.name) for d in bundle.scope_divergences}
        assert sorted(keys) == sorted(published)
        assert published <= {(_MODEL, a, b, "total_ms") for a, b in _pairs(models)}
        assert len(bundle.scope_divergences) > len(published), "each pair's whole diverged from several parts"

    @_SWEEPS
    def test_the_swing_names_the_part_that_carried_it(self, models: Sequence[str], runs_per_level: int) -> None:
        bundle = _bundle(models, runs_per_level=runs_per_level, seed="swing")
        for swing in bundle.whole_swings:
            assert swing.carried_by == "llm_ms"
            assert swing.carried_share == pytest.approx(1.0, abs=0.02)
            assert [component.name for component in swing.components] == ["llm_ms", "orchestration_ms", "tool_ms"]

    @_SWEEPS
    def test_every_divergence_s_whole_has_its_swing(self, models: Sequence[str], runs_per_level: int) -> None:
        bundle = _bundle(models, runs_per_level=runs_per_level, seed="swing")
        swings = {(swing.lever, swing.level_a, swing.level_b, swing.whole) for swing in bundle.whole_swings}
        for divergence in bundle.scope_divergences:
            assert (divergence.lever, divergence.level_a, divergence.level_b, divergence.end_to_end.name) in swings

    @_SWEEPS
    def test_the_writer_reads_each_carrier_once(self, models: Sequence[str], runs_per_level: int) -> None:
        payload = json.loads(
            build_user_message(_bundle(models, runs_per_level=runs_per_level, seed="swing")).removeprefix(
                "CONTEXT BUNDLE (JSON):\n"
            )
        )
        assert all("carried_by" not in divergence for divergence in payload["scope_divergences"])
        carriers = Counter(
            (swing["lever"], swing["level_a"], swing["level_b"], swing["whole"], swing["carried_by"])
            for swing in payload["whole_swings"]
        )
        published = {(d["lever"], d["level_a"], d["level_b"]) for d in payload["scope_divergences"]}
        assert {key[:3] for key in carriers} == published
        assert set(carriers.values()) == {1}

    @_SWEEPS
    def test_each_withheld_or_contained_statement_names_its_own_part(
        self, models: Sequence[str], runs_per_level: int
    ) -> None:
        """What the "Withheld / refused" section read: per pair, two parts contained by the whole, each its own fact.

        The two lines read alike only where a render left the part's name out; in the bundle each is keyed by it.
        """
        bundle = _bundle(models, runs_per_level=runs_per_level, seed="withheld")
        for a, b in {(d.level_a, d.level_b) for d in bundle.scope_divergences}:
            at_pair = [d for d in bundle.scope_divergences if (d.level_a, d.level_b) == (a, b)]
            contained = [d.subsystem.name for d in at_pair if d.contained_by == "total_ms"]
            assert set(contained) <= {"orchestration_ms", "tool_ms"}
            assert len(set(contained)) == len(contained)
            statements = [(d.subsystem.name, d.unattributed_withheld) for d in at_pair]
            assert len(set(statements)) == len(statements)
            assert all(d.unattributed_withheld and d.subsystem.name in d.unattributed_withheld for d in at_pair)


def _frozen_before_49(bundle: AnalysisContextBundle) -> dict[str, Any]:
    """``bundle`` as a shape-48 assembly wrote it: each divergence carrying its whole's swing, and no ``whole_swings``."""
    frozen = bundle.to_dict()
    swings = {(s["lever"], s["level_a"], s["level_b"], s["whole"]): s for s in frozen.pop("whole_swings")}
    for divergence in frozen["scope_divergences"]:
        swing = swings[
            (divergence["lever"], divergence["level_a"], divergence["level_b"], divergence["end_to_end"]["name"])
        ]
        divergence |= {
            "whole_components": swing["components"],
            "carried_by": swing["carried_by"],
            "carried_share": swing["carried_share"],
        }
    frozen["schema_version"] = 48
    return frozen


def _case(frozen: dict[str, Any]) -> ReporterCase:
    return ReporterCase(
        bundle=frozen, bundle_fingerprint=canonical_digest(frozen), bundle_assembled_at="2026-01-01T00:00:00+00:00"
    )


class TestAReporterCaseFrozenBeforeTheMoveStillRebuilds:
    """A frozen bundle is the evidence a case's labels were written against: the move loses none of its values."""

    def test_each_swing_is_read_off_its_divergences_once(self) -> None:
        bundle = _bundle(("m-a", "m-b"), runs_per_level=1, seed="frozen")
        rebuilt = rebuild_bundle(_case(_frozen_before_49(bundle)))
        assert rebuilt.whole_swings == bundle.whole_swings
        assert rebuilt.scope_divergences == bundle.scope_divergences
        assert rebuilt.schema_version == 48, "the frozen document's own shape version is a frozen value too"

    def test_two_different_copies_of_one_swing_are_refused(self) -> None:
        frozen = _frozen_before_49(_bundle(("m-a", "m-b"), runs_per_level=1, seed="frozen"))
        frozen["scope_divergences"][0]["carried_by"] = "tool_ms"
        with pytest.raises(ValueError, match="two different swings of total_ms"):
            rebuild_bundle(_case(frozen))
