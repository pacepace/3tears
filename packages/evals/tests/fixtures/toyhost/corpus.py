"""The toy host's observation corpus — fixture rows, no runner, no battery, no test cases.

A host that ran its candidate elsewhere — observational data, not a commissioned run — has no
template and no case set to show the engine. These observations are constructed directly, to show
what the engine accepts from such a host.

**The corpus is shaped to force three answers**, not to look realistic:

* two observations differing only in ``chunk_tokens`` — the clean sweep, where the confound scan
  must report nothing;
* two differing in ``reviewer_pool`` as well — a rival explanation the engine must surface with
  the host's own reason attached;
* two where ``ocr_engine_version`` was never recorded on one side — ``unknown``, which is neither
  agreement nor difference: a designed state, not a legacy accident.

Beside those three sits a **measured** corpus — :func:`toyhost_batch` /
:func:`toyhost_measurements` and the storage double they feed. The three above carry settings and
no numbers, which is all a cell needs; a bundle needs numbers, and an analysis generated over one
needs enough of them to compare two arms. Kept in one module because they are one host's corpus,
and a second module would let the two drift into describing different products.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from threetears.evals.schema import (
    CalibrationRating,
    EvalCaseStratum,
    EvalResult,
    EvalRun,
    GoalStateOutcome,
    LatencyMetrics,
    RoleUsage,
    RubricScore,
    omit_paths,
)
from threetears.evals.kernel import EvalAnalysis, EvalInsight, resolve_variant_identity
from threetears.evals.kernel.host import HostProfile
from threetears.evals.schema import SubjectSnapshot, SweepableValue
from threetears.evals.schema import WorldPlacement
from packages.evals.tests.fixtures.toyhost.kind import FIELD_ACCURACY, TOY_EXTRACTOR_KIND
from packages.evals.tests.fixtures.toyhost.world import toyhost_world

#: The toy host's extractor. One contestant across the whole corpus, so the shared-core ``models``
#: lever is held fixed and every difference the corpus shows belongs to a toy-host lever.
TOYHOST_EXTRACTOR = "extractor-v2"

#: The toy host's subject: an extractor configuration. The components are the extractor's prompt
#: and schema — content-addressed, named in the toy host's own vocabulary, and carrying no payload
#: at all, which is the runless case the engine must accept.
TOYHOST_SUBJECT = SubjectSnapshot(
    subject_id="extractor-config-7",
    subject_label="Invoice extractor, config 7",
    components={
        "extraction_prompt": SweepableValue.of("pull every field you can identify", display="extraction prompt, rev 3"),
    },
    labels={"deployment": "eu-west"},
    # ``None`` rather than ``{}``, the state a host arrives in before it wires a state reader: this
    # host wired none, so it cannot say what its subject carried in — which is a
    # different fact from saying the subject carried nothing. The engine reads it as an absence,
    # drops the ``subject_state`` component and reports the context as partial, exactly as it
    # reports a variant partial for a host with no lever reader.
    state=None,
    captured_at="2026-08-01T00:00:00+00:00",
)


#: The scope every toy-host observation lives in. The engine partitions on it and never reads its
#: value; a host names its own.
TOYHOST_SCOPE = "toyhost-scope"

#: The documents every measured observation is taken over. Not test cases from a battery — the toy
#: host has neither — just the sampling unit its numbers are means across.
#:
#: **Twelve, not two.** A generator handed two shared cases per arm rightly declines to rank the
#: arms on so few; twelve is enough cases for a comparison to be worth making.
TOYHOST_DOCUMENTS = (
    "doc-01",
    "doc-02",
    "doc-03",
    "doc-04",
    "doc-05",
    "doc-06",
    "doc-07",
    "doc-08",
    "doc-09",
    "doc-10",
    "doc-11",
    "doc-12",
)

#: Every wall-clock value in this fixture, pinned to one instant.
#:
#: `EvalRun.created_at`, `CampaignDesign.declared_at` and a question's `asked_at`
#: all default to `utc_now_iso`, so a fixture that let them default carried the
#: clock into the bundle — and `assemble_context_bundle` derives the measurement
#: windows and the window disclosure from them, which is how ONE unpinned default
#: reached eleven leaves.
#:
#: This is what makes the fixture usable for a fixed-bundle prompt A/B: hold the
#: bundle still, vary the prompt, compare. It could not be held still while it
#: was built from `now()`.
#:
#: A date in the past, not a round number, so nothing reads as a sentinel and a
#: window computed from it is a plausible span rather than an epoch artefact.
TOYHOST_INSTANT = "2026-03-14T09:30:00+00:00"


def _offset(instant: str, *, seconds: int) -> str:
    """``instant`` moved forward, so a fixture span is a span.

    Args:
        instant: ISO-8601 base.
        seconds: How far forward.

    Returns:
        ISO-8601, same offset as the input.
    """
    return (datetime.fromisoformat(instant) + timedelta(seconds=seconds)).isoformat()


#: Namespace for the fixture's derived ids. A fixed UUID rather than a name, so
#: the derivation cannot move if the module is renamed.
_TOYHOST_ID_NAMESPACE = uuid.UUID("6f3a1c9e-2b74-4d51-9a0e-5c8f7b12d430")


#: The spend ceiling in force for every toy-host batch, in USD.
#:
#: Recorded and seated rather than left out of the kind's seats, and the distinction is the point. The extractor calls
#: a model, so it plausibly HAS a ceiling — a blank here was a real fixture gap, and a real gap
#: left unseated would be a detector switched off rather than a fact recorded. The
#: same value on every batch, so the dimension reads as observed-and-agreeing: `reviewer_pool`
#: already carries the fixture's differing-apparatus case, and a second one would not add a shape.
TOYHOST_COST_CEILING_USD = 4.0

#: The grader version every toy-host observation records unless it says otherwise.
#:
#: Defaulted here rather than repeated at each construction for the reason the cost ceiling is: a
#: declaration carrying ``indeterminate_when_blank`` reports an undecidable the moment an
#: observation leaves it out, so a fixture that forgot it on one arm would fabricate the exact
#: confound this host exists to show the engine NOT fabricating. An arm that means to move the
#: grader names its own.
TOYHOST_GRADER_VERSION = "grade-2.1.0"


#: What an observational batch did with every dimension the toy host's world declares, derived by
#: the engine's own run-time algebra over the batch's FACTS: an observation is a record of cells the
#: toy host ran elsewhere, so this batch seeded nothing and attached no carrier for a subject, and
#: every declared dimension is ``out_of_play``. Recorded rather than left out, because a batch with
#: no record reads every world dimension as an undecided confound -- an observation nobody made.
#: The run path (``run.toyhost_run``) places its own batches through the same call over what its
#: kind actually seeds and attaches.
TOYHOST_OBSERVATIONAL_PLACEMENTS: dict[str, WorldPlacement] = toyhost_world()[0].place(seeded=(), carriers=())


def toyhost_observation(**toyhost_values: Any) -> EvalRun:
    """One toy-host observation, carrying only toy-host vocabulary.

    The carrier is the engine's own :class:`~threetears.evals.schema.EvalRun`, and the toy host's
    vocabulary rides in its opaque ``host_payload`` slot, where the host's readers find it.

    Args:
        **toyhost_values: The toy host's keys for this observation. A key omitted is a key this
            observation never recorded, which is a different fact from a key recorded as empty.

    Returns:
        A constructed observation. Nothing is mocked.
    """
    return EvalRun(
        scope_id=TOYHOST_SCOPE,
        # Observational: no template ran it, and the run says so rather than naming one.
        template_id="",
        # And no rig was set for it: the host recorded the apparatus it found on work it did not
        # control, which is what `witnessed` means.
        apparatus_provenance="witnessed",
        test_case_ids=list(TOYHOST_DOCUMENTS),
        k_runs=1,
        candidate_kind=TOY_EXTRACTOR_KIND,
        candidate_model=TOYHOST_EXTRACTOR,
        subject_snapshot=TOYHOST_SUBJECT,
        rubric_scales={},
        # An ENGINE field, on the terms the shared core declares it: every LLM product has a spend
        # ceiling, and this host's blank was a gap rather than a state. See
        # ``TOYHOST_COST_CEILING_USD`` for why it is recorded and seated.
        max_cost_usd=TOYHOST_COST_CEILING_USD,
        # The toy host's own vocabulary lives in the engine-owned opaque slot.
        # ``grader_version`` first, so a caller that names it wins: the default is what makes
        # every arm record a grader, and an arm that means to move it must be able to.
        host_payload={"toyhost": {"grader_version": TOYHOST_GRADER_VERSION, **toyhost_values}},
        world_placements=dict(TOYHOST_OBSERVATIONAL_PLACEMENTS),
    )


#: Two observations differing in exactly one lever. The confound scan must report nothing: this
#: is the clean sweep, and an engine that invents a confound here is unusable.
CLEAN_SWEEP: tuple[EvalRun, ...] = (
    toyhost_observation(chunk_tokens=256, retriever_top_k=3, ocr_engine_version="tess-5.3.1", reviewer_pool="pool-a"),
    toyhost_observation(chunk_tokens=1024, retriever_top_k=3, ocr_engine_version="tess-5.3.1", reviewer_pool="pool-a"),
)

#: The same sweep with the apparatus moved underneath it. The scan must surface ``reviewer_pool``
#: carrying the host's own reason — a confound no generic engine could have guessed.
CONFOUNDED_SWEEP: tuple[EvalRun, ...] = (
    toyhost_observation(chunk_tokens=256, retriever_top_k=3, ocr_engine_version="tess-5.3.1", reviewer_pool="pool-a"),
    toyhost_observation(chunk_tokens=1024, retriever_top_k=3, ocr_engine_version="tess-5.3.1", reviewer_pool="pool-b"),
)

#: One side never recorded the OCR build. Neither agreement nor difference — the state that must
#: not collapse into either, because both readings assert an observation nobody made.
UNRECORDED_APPARATUS: tuple[EvalRun, ...] = (
    toyhost_observation(chunk_tokens=256, ocr_engine_version="tess-5.3.1", reviewer_pool="pool-a"),
    toyhost_observation(chunk_tokens=1024, reviewer_pool="pool-a"),
)


#: What each sampled document does to a batch's numbers, as
#: ``{document: (accuracy offset, cost-and-latency multiplier)}``.
#:
#: **Widening the corpus without this would have made it worse than the two documents it
#: replaced.** Every observation carried the batch's mean plus a per-repeat ±0.01, so twelve
#: identical documents would have multiplied n while holding dispersion at the repeat spread — a
#: confidence interval that tightens because the fixture repeated itself, which is a fabricated
#: precision and a worse answer than an honest "not enough cases". Documents differ in how legible
#: they are and in how much text they carry, so the numbers differ per document and the comparison
#: has something to be uncertain about.
#:
#: The offsets sum to zero and the multipliers average 1.0, so each batch's declared mean is still
#: the mean in ``_MEASUREMENTS``: the widening added dispersion without quietly restating the
#: campaign's own numbers. Fixed values rather than a seeded generator, because this fixture's
#: whole premise is a bundle that fingerprints identically across processes.
_DOCUMENT_PROFILE: dict[str, tuple[float, float]] = {
    "doc-01": (-0.06, 1.20),
    "doc-02": (-0.05, 1.14),
    "doc-03": (-0.04, 1.08),
    "doc-04": (-0.02, 1.05),
    "doc-05": (-0.01, 1.02),
    "doc-06": (-0.01, 1.00),
    "doc-07": (0.01, 1.00),
    "doc-08": (0.01, 0.98),
    "doc-09": (0.02, 0.95),
    "doc-10": (0.04, 0.92),
    "doc-11": (0.05, 0.86),
    "doc-12": (0.06, 0.80),
}


#: The one dimension the toy host's reviewer pool scores, in the host's own word — namespaced by its
#: scoring context, as every stored dim name is (``DimName``).
TOYHOST_JUDGED_DIMENSION = "extraction.layout_fidelity"


class ToyhostStorage:
    """The reads the bundle assembler makes, over an in-memory toy-host corpus.

    The reads :func:`~threetears.evals.analysis.assemble_context_bundle` makes, and nothing
    else — the narrow port a read-only consumer implements over data it already holds. It honours
    the scope partition on runs and results, so a wrong-scope read misses exactly as a real store's
    does.
    """

    def __init__(
        self,
        runs: Sequence[EvalRun],
        results_by_run: Mapping[str, list[EvalResult]],
        insights: Sequence[EvalInsight] = (),
        analyses: Sequence[EvalAnalysis] = (),
    ) -> None:
        """Hold one campaign's observations.

        Args:
            runs: The batches.
            results_by_run: Each batch's observations, keyed by batch id.
            insights: The prior ledger. Empty by default — the toy host has generated nothing
                before — and supplied by a test that needs a ledger to read as of an instant.
            analyses: The stored analyses those insights were minted by, for a test that needs
                one archived. Empty by default, so an insight's source resolves to nothing.
        """
        self._runs = list(runs)
        self._results_by_run = dict(results_by_run)
        self._insights = list(insights)
        self.analyses = {analysis.id: analysis for analysis in analyses}
        self._ratings: dict[str, CalibrationRating] = {}

    def load_eval_run(self, run_id: str, scope_id: str) -> EvalRun | None:
        """The batch with this id in this scope, or None."""
        return next((r for r in self._runs if r.id == run_id and r.scope_id == scope_id), None)

    def load_eval_runs(
        self, run_ids: Sequence[str], scope_id: str, *, elide_payload: frozenset[str] = frozenset()
    ) -> list[EvalRun]:
        """The batches with these ids in this scope, each without the payload paths the listing elides."""
        runs = [run for run_id in run_ids if (run := self.load_eval_run(run_id, scope_id)) is not None]
        if not elide_payload:
            return runs
        listed = []
        for run in runs:
            copy = run.model_copy(update={"host_payload": omit_paths(run.host_payload, sorted(elide_payload))})
            copy.note_elided_payload(elide_payload)
            listed.append(copy)
        return listed

    def query_eval_results_by_run(self, run_id: str, scope_id: str) -> list[EvalResult]:
        """Every observation of this batch in this scope."""
        return [r for r in self._results_by_run.get(run_id, []) if r.scope_id == scope_id]

    def load_case_strata(self, test_case_ids: Sequence[str], scope_id: str) -> list[EvalCaseStratum]:
        """The stratum each named document declares in this scope: none, since the toy host sorts its documents into no kinds."""
        if scope_id != TOYHOST_SCOPE:
            return []
        return [EvalCaseStratum(id=document) for document in test_case_ids if document in TOYHOST_DOCUMENTS]

    def load_eval_result(self, result_id: str, scope_id: str) -> EvalResult | None:
        """The observation with this id in this scope, or None — what a calibration rating reads first."""
        return next(
            (
                r
                for results in self._results_by_run.values()
                for r in results
                if r.id == result_id and r.scope_id == scope_id
            ),
            None,
        )

    def save_calibration_rating(self, rating: CalibrationRating) -> None:
        """Hold a reviewer's rating, replacing that reviewer's earlier rating of the same thing, as the real store does."""
        self._ratings[rating.id] = rating

    def query_calibration_ratings(self, scope_id: str, *, run_id: str | None = None) -> list[CalibrationRating]:
        """The ratings in one scope, narrowed by run, oldest first."""
        return sorted(
            (
                rating
                for rating in self._ratings.values()
                if rating.scope_id == scope_id and (run_id is None or rating.run_id == run_id)
            ),
            key=lambda rating: rating.rated_at,
        )

    def query_insights(
        self, scope_id: str, *, subject_id: str | None = None, source_campaign_id: str | None = None
    ) -> list[EvalInsight]:
        """The prior ledger in one scope, filtered the way the real store filters it — by exact id."""
        return [
            insight
            for insight in self._insights
            if insight.scope_id == scope_id
            and (subject_id is None or insight.subject_id == subject_id)
            and (source_campaign_id is None or insight.source_campaign_id == source_campaign_id)
        ]

    def load_analysis(self, analysis_id: str, scope_id: str) -> EvalAnalysis | None:
        """The stored analysis with this id in this scope, or None — held in a dict a test may mutate to archive one."""
        analysis = self.analyses.get(analysis_id)
        return analysis if analysis is not None and analysis.scope_id == scope_id else None

    def analysis_archived(self, analysis_id: str, scope_id: str) -> bool | None:
        """Whether the stored analysis with this id in this scope is archived, or None when there is none."""
        analysis = self.load_analysis(analysis_id, scope_id)
        return None if analysis is None else analysis.archived


def toyhost_batch(**toyhost_values: Any) -> EvalRun:
    """One measured toy-host batch: three repeats over both documents, in the toy host's scope.

    :func:`toyhost_observation` builds the minimal carrier the cell work needs. This adds what a
    bundle needs on top — a scope, the documents, and the repeat count the coverage floor reads —
    without which every measure arrives with n=1 and nothing can be compared.

    Args:
        **toyhost_values: The toy host's keys for this batch.

    Returns:
        The batch.
    """
    return toyhost_observation(**toyhost_values).model_copy(
        update={
            # DERIVED from the batch's own values, not minted: `EvalRun.id`
            # defaults to `uuid4`, so two calls with identical values would
            # produce two different batches and every bundle over them would
            # fingerprint differently.
            #
            # uuid5 over the sorted values, so distinct batches stay distinct and
            # an identical one is identical across processes. A real UUID shape,
            # because readers downstream treat these as ids rather than opaque
            # strings.
            "id": str(uuid.uuid5(_TOYHOST_ID_NAMESPACE, json.dumps(toyhost_values, sort_keys=True))),
            "created_at": TOYHOST_INSTANT,
            "k_runs": 3,
            # A batch that produced its observations ENDED; `status` defaults to "pending",
            # which says the opposite, and a generator reading a pending run treats its
            # numbers as provisional.
            "status": "completed",
        }
    )


#: The accuracy the toy host's bar holds an extraction to, and the check the kind reports against it.
_ACCURACY_BAR = 0.92
_ACCURACY_CHECK = f"{FIELD_ACCURACY} >= {_ACCURACY_BAR}"


def _observed_accuracy(field_accuracy: float, document: str, repeat: int) -> float:
    """One observation's field accuracy: the batch's setting, the document's offset and the repeat's spread."""
    return round(field_accuracy + _DOCUMENT_PROFILE[document][0] + 0.01 * (repeat - 2), 6)


def _document_cost(cost_usd: float, document: str) -> float:
    """What one document cost at a batch's per-document spend, scaled by how hard the document is."""
    return round(cost_usd * _DOCUMENT_PROFILE[document][1], 6)


def toyhost_measurements(
    batch: EvalRun,
    *,
    profile: HostProfile,
    cost_usd: float,
    total_ms: float,
    field_accuracy: float,
    layout_fidelity: int | None = None,
    covariates: Mapping[str, float] | None = None,
) -> list[EvalResult]:
    """The observations one toy-host batch produced — three repeats over each document.

    Two of the values are the engine's shared-core ones (spend and wall-clock). The third,
    ``field_accuracy``, is the toy host's OWN — declared on its profile and carried on
    ``EvalResult.host_measures``.

    **It is the quality half of the campaign's declared question.** "Is the wider chunk worth
    what it costs to retrieve?" is a trade, and a corpus carrying only spend and wall-clock
    supplies one side of it.

    Each observation carries the variant key the runner would have stamped on it — resolved
    through ``profile`` from the batch, exactly as the run loop resolves it — because a result
    is only ever written stamped, and the bundle reads the stamp rather than deriving one.

    Args:
        batch: The batch these belong to.
        profile: The host the batch ran under, whose variant-lever reader keys its observations.
        cost_usd: Spend per document at this batch's setting.
        total_ms: Wall-clock per document at this batch's setting, before the per-repeat spread.
        field_accuracy: Share of fields extracted exactly right at this setting, before the
            per-repeat spread. The host declares this one; the engine has never heard of it.
        layout_fidelity: The reviewer pool's 1-5 score for how faithfully the extracted record
            keeps the invoice's line-item structure, before a third repeat's one-point lift —
            a JUDGED dimension, scored by people rather than computed, which the engine carries
            off the ranking surface. ``None`` means the pool scored nothing, which is the state
            a batch nobody reviewed is in.
        covariates: The measurement-condition covariates every observation of this batch recorded,
            by the engine's covariate names (``context_tokens_in``, ``reasoning_ratio``). ``None``
            records none, which says nothing was measured rather than that anything was zero.

    Returns:
        Thirty-six observations — three repeats × twelve documents.
    """
    variant = resolve_variant_identity(run=batch, profile=profile)
    return [
        EvalResult(
            # Derived, like the batch's own id and for the same reason: these
            # reach the bundle as `observation_ids`, so minted ones made every
            # assembled bundle unique and the fixed-bundle A/B unrunnable.
            id=str(uuid.uuid5(_TOYHOST_ID_NAMESPACE, f"{batch.id}|{document}|{repeat}")),
            # `scored_at` is the result's only timestamp, and the measurement window
            # reads it: pinning it keeps the bundle's fingerprint still. Spread one
            # second per repeat so the window is a real span rather than a
            # zero-length point, which the engine treats differently.
            scored_at=_offset(TOYHOST_INSTANT, seconds=repeat),
            eval_run_id=batch.id,
            scope_id=TOYHOST_SCOPE,
            test_case_id=document,
            # The model the batch declared: one batch is one arm, and an arm runs one model.
            model=batch.candidate_model,
            k_iteration=repeat,
            subject_id=TOYHOST_SUBJECT.subject_id,
            # The toy host's own dimension, scored by its reviewer pool; one point higher on the
            # third repeat so a cell's scores are a distribution rather than one number restated.
            rubric_scores=(
                [
                    RubricScore(
                        dim=TOYHOST_JUDGED_DIMENSION, score=layout_fidelity + (1 if repeat == 3 else 0), scale="ordinal"
                    )
                ]
                if layout_fidelity is not None
                else []
            ),
            # The kind's own mechanical fact, the one `kind.py` reports for the same
            # observation: whether this extraction cleared the accuracy the host's bar names. Carried so the
            # bundle's per-check pass rate — the mechanical tier a generator reads with or without a bar —
            # has a toy-host instance.
            goal_state_outcomes=[
                GoalStateOutcome(
                    expression=_ACCURACY_CHECK,
                    passed=_observed_accuracy(field_accuracy, document, repeat) >= _ACCURACY_BAR,
                )
            ],
            # Per DOCUMENT as well as per repeat. A longer, less legible document costs more and
            # takes longer, which is what makes twelve documents twelve observations rather than
            # one restated twelve times — see ``_DOCUMENT_PROFILE``.
            cost_usd=_document_cost(cost_usd, document),
            latency=LatencyMetrics(total_ms=round(total_ms * _DOCUMENT_PROFILE[document][1] + 10 * repeat, 3)),
            # Spread ±0.01 around the mean so three repeats are a real distribution rather
            # than one number restated — a zero-dispersion measure reads as a constant, and
            # the coverage floor exists to stop exactly that being compared. `repeat` is
            # 1..3, so the offsets are -0.01, 0, +0.01 and the mean is unchanged. The document
            # offset is the second axis, and the one that makes the arms' spread real.
            host_measures={FIELD_ACCURACY: _observed_accuracy(field_accuracy, document, repeat)},
            candidate_kind=TOY_EXTRACTOR_KIND,
            # What a completed cell's capture states: it ran to the end, its spend is the
            # candidate's alone (no judge, no simulator, no background work) — one priced
            # candidate row, which is what the result's `cost_usd` is derived from and what makes
            # it an observed spend rather than the sum of nothing — and it carried no phase timings
            # and only the covariates the caller names. The row names no model: one it named would be an observed candidate
            # model, a lever of its own on the coverage map, which the corpus holds fixed by `model`. It does name the
            # model the toy provider's response said answered — the run's own, since the toy provider resolves no
            # alias — as a conforming host's completions do; a row naming none reads as not recorded.
            termination="completed",
            cost_roles=["candidate"],
            usage=[
                RoleUsage(
                    role="candidate",
                    model=None,
                    served_model=batch.candidate_model,
                    cost_usd=_document_cost(cost_usd, document),
                    call_count=1,
                )
            ],
            covariates=dict(covariates or {}),
            phase_timings={},
            variant_key=variant.variant_key,
            identity_version=variant.identity_version,
        )
        for document in TOYHOST_DOCUMENTS
        for repeat in (1, 2, 3)
    ]


__all__ = [
    "CLEAN_SWEEP",
    "CONFOUNDED_SWEEP",
    "TOYHOST_DOCUMENTS",
    "TOYHOST_EXTRACTOR",
    "TOYHOST_GRADER_VERSION",
    "TOYHOST_JUDGED_DIMENSION",
    "TOYHOST_OBSERVATIONAL_PLACEMENTS",
    "TOYHOST_SCOPE",
    "TOYHOST_SUBJECT",
    "UNRECORDED_APPARATUS",
    "ToyhostStorage",
    "toyhost_batch",
    "toyhost_measurements",
    "toyhost_observation",
]
