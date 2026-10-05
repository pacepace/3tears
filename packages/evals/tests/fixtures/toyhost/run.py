"""The toy host's run path — its template, its cases, its store, and the real runner driving them.

A template naming :data:`~packages.evals.tests.fixtures.toyhost.kind.TOY_EXTRACTOR_KIND`, three cases
over three invoices, two extractor models, ``k=2``, and :func:`~threetears.evals.run.execute_run` —
the engine's runner, not a re-implementation of it — producing twelve results that the analysis then
assembles into a bundle.

**This is the low-level path.** :func:`~threetears.evals.run.execute_run` runs one assembled run's
matrix and nothing else: it changes no run status, launches no job and stamps nothing a launch
stamps. A product adopts the engine through :func:`~threetears.evals.run.start_run`, which resolves
a launch request into a run, admits it, and drives this same runner as a job (``launch.py`` is the
toy host's wiring for it). This module drives the runner directly because what it pins is the
runner's own output, cell by cell.

**The hand-built corpus stays.** It forces designed states a run cannot: a never-recorded
apparatus dimension, a confounded sweep, a pair whose grader moved. This adds the path beside it,
and the two answer different questions — one about what the engine accepts, one about what the
engine produces.

Two runs, one model each
------------------------

The matrix is two models × three cases × ``k=2``, and a run carries exactly one arm, the candidate
model among its settings. So the matrix is two runs, as any launch naming two models starts; a run
names one model, so no run can name both.

What the toy host supplies, and nothing more
--------------------------------------------

Its :class:`~threetears.evals.contracts.host.EvalHost` (``host.py``): the profile, the engine's own
storage over an in-memory document store, and the three services every host names. The runner writes
each cell through that storage and the analysis reads the same storage back, so runner output feeds
analysis input through the engine's own store rather than a hand-off the test arranges.

What the runner does not ask of a host:

* A kind reaches the runner only as a factory on ``RunnerOptions.candidate_kinds``, which is
  handed the cell; the toy kind holds no state between cells, so its factory hands back the one
  instance. ``execute_run`` takes no subject factory and no simulator of its own, so this host
  passes nothing it does not have.
* Nothing of a host's subject arrives through the engine: ``prepare`` is handed the engine's
  subject snapshot alone, and the runner recovers no subject of its own. What the toy kind needs it
  holds from construction, and its run's own settings ride on ``EvalRun.host_payload``, read by the
  toy host's registries and never by the runner.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle
from threetears.evals.contracts import (
    CampaignDesign,
    EvalCampaign,
    EvalResult,
    EvalRun,
    EvalTemplate,
    EvalTestCase,
    EvalTrace,
    WorldSeed,
)
from threetears.evals.contracts.host import CANDIDATE_MODEL_LEVER, EvalHost, WorldRegistry
from threetears.evals.run import JudgeService, RunnerOptions, execute_run
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_COST_CEILING_USD,
    TOYHOST_GRADER_VERSION,
    TOYHOST_INSTANT,
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    toyhost_observation,
)
from packages.evals.tests.fixtures.toyhost.kind import (
    DOCUMENT_PARAM,
    TOY_DOCUMENTS,
    TOY_EXTRACTOR_KIND,
    TOY_SCRIPTS,
    ScriptedExtractionClient,
    ToyExtractorKind,
)

#: The run path's own id namespace, distinct from the corpus's so a run-produced batch and a
#: hand-built one can never collide on a derived id.
_RUN_ID_NAMESPACE = uuid.UUID("b1f4c2d7-8e39-45a6-9c11-3f7d2a840e55")

#: The chunk width and retriever width every run-path batch sits at. FIXED across both arms: this
#: campaign sweeps the extractor model, and an arm that also moved a retrieval lever would make
#: every difference ambiguous between the two — the same discipline ``campaign.py`` holds for the
#: corpus campaign.
RUN_CHUNK_TOKENS = 512
RUN_RETRIEVER_TOP_K = 4

#: The documents the run's cases are drawn over, in declaration order.
RUN_DOCUMENTS: tuple[str, ...] = tuple(document.document_id for document in TOY_DOCUMENTS)

#: The two arms, narrow-capability first. Read off the scripts so the client and the run cannot
#: disagree about which models exist.
RUN_MODELS: tuple[str, ...] = tuple(script.model for script in TOY_SCRIPTS)

#: Repeats per test case, per extractor model — each run's own ``k_runs``. Two, so the matrix is
#: twelve; see ``kind.py`` on why a scripted client makes the two repeats of one case exact
#: duplicates and what that does to an arm's dispersion.
RUN_K = 2

#: The question this campaign declares, by id.
RUN_QUESTION_ID = "q-extractor-model"


def toyhost_template() -> EvalTemplate:
    """The toy host's template: a kind name, three world dimensions, and no rubric.

    Fixed ids and timestamps for the reason every other toy-host fixture pins its own — the
    defaults are ``uuid4`` and the clock, and either makes two builds of this fixture
    incomparable.

    **What it declares nothing of is the point.** No rubric (nothing here is model-scored), no
    simulated actors, no goal-state checks (the mechanical tier is the kind's, reported through
    ``CandidateOutput.mechanical_facts`` rather than evaluated from a DSL against a world the
    engine derived), and no preconditions.

    Returns:
        The template.
    """
    return EvalTemplate(
        id="toyhost-template-extract-invoice-fields",
        scope_id=TOYHOST_SCOPE,
        name="Invoice field extraction",
        intent="Extract every declared field from a scanned invoice and be graded against the adjudicated key.",
        candidate_kind=TOY_EXTRACTOR_KIND,
        created_at=TOYHOST_INSTANT,
        updated_at=TOYHOST_INSTANT,
        # Keyed by CARRIER, which is the shape the seed's own docstring describes ("keys are tool
        # namespaces and each sub-key names one state dimension") read in the toy host's
        # vocabulary: a carrier is what supplies a dimension, so a seed the subject's carriers
        # cannot reach is one no write could have landed. `ToyExtractorKind.CARRIERS` is what
        # says which those are, and the kind refuses a namespace outside it.
        world_seed=WorldSeed(
            namespaces={
                "page_reader": {
                    "document_language": "de",
                    "scan_quality": "faint",
                    "vendor_template": "acme-2019",
                },
                "console": {"operator_corrections": ["reprice"]},
            }
        ),
    )


def toyhost_test_cases(template: EvalTemplate) -> list[EvalTestCase]:
    """One case per invoice — the sampling unit this campaign's numbers are means across.

    The document rides on ``variation_params`` because that is the only per-case channel the
    engine has: there is no per-case host payload, so a stimulus a host names in its own
    vocabulary has to be expressible as strings.

    Args:
        template: The template these cases were generated from.

    Returns:
        Three cases, in document order.
    """
    return [
        EvalTestCase(
            id=f"{template.id}:{document}",
            scope_id=template.scope_id,
            template_id=template.id,
            created_at=TOYHOST_INSTANT,
            variation_params={DOCUMENT_PARAM: document},
        )
        for document in RUN_DOCUMENTS
    ]


@dataclass(frozen=True)
class ToyhostArm:
    """One batch of a run-path campaign: which extractor it binds, and how its retrieval was tuned.

    The standard matrix is one arm per model and tunes nothing, which is what
    :func:`execute_toyhost_run` builds when handed no arms. A retrieval sweep holds the model and
    moves a knob instead, and records the RESOLVED configuration the launch merged it into — the
    shape in which one overlaid knob moves two levers.
    """

    model: str
    #: Names the arm in its run id, so two arms on one model are two batches. ``None`` for the
    #: standard matrix, whose arms are named by their model alone.
    label: str | None = None
    #: The retrieval configuration this batch actually ran, after any overlay. ``None`` records
    #: nothing — the standard matrix's batches carry no retrieval tuning at all.
    retrieval_config: dict[str, Any] | None = None
    #: The knobs the launch overlaid onto that configuration. ``None`` overlaid nothing.
    retrieval_overrides: dict[str, Any] | None = None


def toyhost_run(
    *, model: str, template: EvalTemplate, kind: ToyExtractorKind, world: WorldRegistry, arm: ToyhostArm | None = None
) -> EvalRun:
    """One arm of the run-path campaign: one extractor model over every case, ``k=2``.

    Args:
        model: The extractor this arm binds the candidate to.
        template: The template being run.
        kind: The kind that will drive the cells — asked which carriers it attaches and which
            dimensions the seed sets, because the run-time algebra is a report of what a run
            could actually have done rather than of what a template asked for.
        world: The world the placements are computed against. The SAME registry the kind seeds
            through, so the record and the run cannot disagree about what was declared.
        arm: The arm's retrieval tuning and label. ``None`` is the standard matrix's arm for
            ``model``, which records no retrieval tuning.

    Returns:
        The run, with its world placements already recorded.
    """
    # A key is recorded only when the arm states it: a key omitted is a key this batch never
    # recorded, which is a different fact from one recorded as empty.
    retrieval = {
        key: value
        for key, value in (
            ("retrieval_config", arm.retrieval_config if arm else None),
            ("retrieval_overrides", arm.retrieval_overrides if arm else None),
        )
        if value is not None
    }
    label = f"{model}|{arm.label}" if arm is not None and arm.label else model
    run = toyhost_observation(
        chunk_tokens=RUN_CHUNK_TOKENS,
        retriever_top_k=RUN_RETRIEVER_TOP_K,
        extraction_schema="v1",
        ocr_engine_version="tess-5.3.1",
        reviewer_pool="pool-a",
        grader_version=TOYHOST_GRADER_VERSION,
        batch_label=f"run-path/{label}",
        **retrieval,
    )
    return run.model_copy(
        update={
            # Derived from the arm, so two arms are two batches and a rebuild is the same two.
            "id": str(uuid.uuid5(_RUN_ID_NAMESPACE, f"run|{label}")),
            "created_at": TOYHOST_INSTANT,
            "scope_id": TOYHOST_SCOPE,
            # NAMED, unlike the corpus's batches. The toy host commissions nothing in its
            # observational shape, but a RUN is a commission by construction — `execute_run`
            # takes a template — so this arm has one to name and says so.
            "template_id": template.id,
            "candidate_model": model,
            "k_runs": RUN_K,
            "test_case_ids": list(RUN_DOCUMENTS),
            "max_cost_usd": TOYHOST_COST_CEILING_USD,
            "resolved_world_seed": dict(template.world_seed.namespaces),
            # The algebra over the FACTS this run produced: what the kind seeds, through the
            # carriers the kind attaches. `witnessed` for `ingest_backlog` (perceived, unseeded)
            # and `judge_only` for the two the extractor is never shown falls out of that.
            "world_placements": {
                name: placement
                for name, placement in world.place(
                    seeded=kind.seedable_dimensions(template.world_seed),
                    carriers=ToyExtractorKind.CARRIERS,
                ).items()
            },
            "status": "pending",
        }
    )


@dataclass
class ToyhostRunPath:
    """Everything one drive of the toy host's run path produced."""

    #: The host the drive ran under — its storage holds the runs, the results and their traces.
    host: EvalHost
    template: EvalTemplate
    test_cases: list[EvalTestCase]
    runs: list[EvalRun]
    kind: ToyExtractorKind
    client: ScriptedExtractionClient

    @property
    def results(self) -> list[EvalResult]:
        """Every cell the drive produced, run by run.

        Returns:
            Twelve observations for the standard matrix.
        """
        return [
            result for run in self.runs for result in self.host.storage.query_eval_results_by_run(run.id, run.scope_id)
        ]

    def trace(self, result: EvalResult) -> EvalTrace | None:
        """The payload document the runner wrote beside one cell, or ``None`` when it wrote none.

        Args:
            result: The cell.

        Returns:
            Its trace.
        """
        return self.host.storage.load_eval_trace(result.id, result.scope_id)


async def execute_toyhost_run(
    *,
    host: EvalHost,
    template: EvalTemplate | None = None,
    arms: Sequence[ToyhostArm] | None = None,
    judge_service: JudgeService | None = None,
    judge_model: str | None = None,
) -> ToyhostRunPath:
    """Drive the whole matrix through the real runner, for the toy host.

    The standard matrix passes ``judge_service=None``: nothing it produces is model-scored, so the
    judge phase is suppressed and every score axis stays ``None``, and the extractor's rig seats —
    which leave the model-judge axes out — keep that from reading as undecided confounds in the bundle.
    A judged drive (``packages.evals.tests.fixtures.toyhost.judge``) passes a service, and the kind then renders
    the evidence that service scores.

    Args:
        host: The toy host (:func:`~packages.evals.tests.fixtures.toyhost.host.toyhost_host`). Passed
            in rather than built here because the world's read handles are bound to the object its
            profile carries, so a caller that wants to read the world back afterwards needs the same
            one — and its trace sink is the caller's decision, which a default here would make for it.
        template: The template to run. ``None`` is :func:`toyhost_template`. A caller supplies its
            own to reach a refusal the standard template cannot — a seed naming an undeclared
            dimension, say, which the kind refuses and the runner records as one excluded cell.
        arms: The batches to run. ``None`` is the standard matrix — one arm per extractor model,
            no retrieval tuning.
        judge_service: The engine's judge, or ``None`` for a drive nothing scores. Given one, the
            kind hands each cell's evidence to it; the template must carry the rubric it scores.
        judge_model: The model ``judge_service`` scores with. The launch that resolves it is the
            host's, so a host driving the runner directly records it on each arm's run — the one
            carrier the runner stamps every cell from — and the runner refuses a judge without one.

    Returns:
        What the drive produced.
    """
    world = host.profile.world
    assert world is not None, "the run path seeds a world; this profile registers none"
    client = ScriptedExtractionClient()
    kind = ToyExtractorKind(client=client, world=world, judged=judge_service is not None)
    template = template if template is not None else toyhost_template()
    test_cases = toyhost_test_cases(template)
    arms = arms if arms is not None else [ToyhostArm(model=model) for model in RUN_MODELS]
    runs = [
        toyhost_run(model=arm.model, template=template, kind=kind, world=world, arm=arm).model_copy(
            update={"judge_model": judge_model}
        )
        for arm in arms
    ]
    for run in runs:
        host.storage.save_eval_run(run)
        await execute_run(
            host,
            run=run,
            template=template,
            test_cases=test_cases,
            judge_service=judge_service,
            options=RunnerOptions(candidate_kinds={TOY_EXTRACTOR_KIND: lambda _cell: kind}),
        )
    # `EvalRun.status` defaults to `pending` and the job manager owns the transition, which the
    # runner deliberately does not make — so a drive of `execute_run` directly makes it. A generator
    # reading a pending run treats its numbers as provisional.
    runs = [run.model_copy(update={"status": "completed"}) for run in runs]
    for run in runs:
        host.storage.save_eval_run(run)
    return ToyhostRunPath(
        host=host,
        template=template,
        test_cases=test_cases,
        runs=runs,
        kind=kind,
        client=client,
    )


def toyhost_run_design() -> CampaignDesign:
    """What the run-path campaign declared before any of it ran.

    One nominal axis at two levels — the extractor model, which is the shared core's own lever
    and the only one this campaign moves — one live question, and controls recording that the
    stimulus WAS held: the three invoices are a fixed corpus every arm saw, which is the opposite
    of the corpus campaign's observational shape and is the reason both exist.

    Returns:
        The declaration.
    """
    return CampaignDesign.model_validate(
        {
            "axes": [
                {
                    "axis_id": CANDIDATE_MODEL_LEVER,
                    # Nominal, with no scale: one extractor build is not a measured distance from
                    # another, and annotating a spacing it does not have is what an interval scale
                    # on a model name would claim.
                    # A display on every level, because a nominal one has no natural rendering
                    # from its content hash and an analysis that fell back to one would read as
                    # "component 3 moved" — the declaration refuses it rather than rendering that.
                    "values": [{"content": model, "display": model} for model in RUN_MODELS],
                    "rationale": "does the dearer extractor earn its price on the fields it is graded on",
                }
            ],
            "questions": [
                {
                    "id": RUN_QUESTION_ID,
                    "text": "is the dearer extractor worth what it costs per document?",
                    "asked_at": TOYHOST_INSTANT,
                }
            ],
            "declared_at": TOYHOST_INSTANT,
            "intended_repetitions": RUN_K,
            # The opposite of the corpus campaign's pair on BOTH axes, which is why both
            # fixtures exist: three fixed invoices every arm saw is a controlled stimulus, and a
            # matrix the runner executed is a commissioned apparatus. The corpus campaign is
            # observational on both, and a host that could only produce one of the two shapes
            # would leave the other's branches unexercised.
            "controls": {
                "stimulus": "controlled",
                "apparatus": "commissioned",
            },
        }
    )


def toyhost_run_campaign(path: ToyhostRunPath) -> EvalCampaign:
    """The campaign the two arms belong to.

    Args:
        path: What the drive produced.

    Returns:
        The campaign, naming both arms in declaration order.
    """
    return EvalCampaign(
        id="7c1e2f05-4b83-42d7-9a6e-8d0f35b91c47",
        scope_id=TOYHOST_SCOPE,
        name="extractor model bake-off",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id=path.template.id,
        run_ids=[run.id for run in path.runs],
        declared_design=toyhost_run_design(),
        created_by="test:fixture",
    )


def toyhost_run_bundle(path: ToyhostRunPath) -> AnalysisContextBundle:
    """The assembled context one memo over the run-path campaign is generated from.

    Assembled in the vocabulary of the host the drive ran under.

    Args:
        path: What the drive produced.

    Returns:
        The bundle, over runner-produced observations.
    """
    return assemble_context_bundle(toyhost_run_campaign(path), storage=path.host.storage, profile=path.host.profile)


#: The retrieval configuration every retrieval-sweep batch starts from.
RETRIEVAL_BASE: dict[str, Any] = {"rerank_depth": 8, "dedupe_threshold": 0.9}

#: The knob the retrieval sweep moves, as a lever name, and the level its arms move it to.
RETRIEVAL_AXIS = "retrieval.rerank_depth"
RETRIEVAL_SWEPT_DEPTH = 16


def toyhost_retrieval_arms() -> tuple[ToyhostArm, ToyhostArm, ToyhostArm]:
    """A control and two arms that each overlay ONE knob, on one extractor model.

    ``deeper`` overlays ``rerank_depth`` and nothing else changed: its resolved configuration is
    the control's with that key moved, so the resolved-configuration lever moved only because the
    knob did. ``deeper_and_drifted`` overlays the same knob, but its base configuration had ALSO
    drifted on a knob nobody overlaid — the resolved configuration moved by more than the swept
    knob accounts for.

    Returns:
        ``(control, deeper, deeper_and_drifted)``.
    """
    model = RUN_MODELS[0]
    overlay = {"rerank_depth": RETRIEVAL_SWEPT_DEPTH}
    drifted_base = {**RETRIEVAL_BASE, "dedupe_threshold": 0.75}
    return (
        ToyhostArm(model=model, label="control", retrieval_config=dict(RETRIEVAL_BASE)),
        ToyhostArm(
            model=model, label="deeper", retrieval_config={**RETRIEVAL_BASE, **overlay}, retrieval_overrides=overlay
        ),
        ToyhostArm(
            model=model,
            label="deeper-and-drifted",
            retrieval_config={**drifted_base, **overlay},
            retrieval_overrides=overlay,
        ),
    )


def toyhost_retrieval_campaign(path: ToyhostRunPath, members: Sequence[EvalRun], *, control: EvalRun) -> EvalCampaign:
    """A retrieval-sweep campaign over some of a drive's batches, controlled on one of them.

    Args:
        path: What the drive produced — where the control's observations are read.
        members: The batches the campaign names.
        control: The batch whose variant is the declared control.

    Returns:
        The campaign.
    """
    from threetears.evals.analysis import variant_key_of_run

    control_key = variant_key_of_run(path.host.storage.query_eval_results_by_run(control.id, control.scope_id))
    assert control_key is not None, "the control batch resolves no variant"
    design = CampaignDesign.model_validate(
        {
            "axes": [
                {
                    "axis_id": RETRIEVAL_AXIS,
                    "values": [{"content": RETRIEVAL_SWEPT_DEPTH, "display": str(RETRIEVAL_SWEPT_DEPTH)}],
                    "rationale": "does reranking deeper earn its latency",
                }
            ],
            "declared_at": TOYHOST_INSTANT,
            "control": control_key,
            "controls": {"stimulus": "controlled", "apparatus": "commissioned"},
        }
    )
    return EvalCampaign(
        id=str(uuid.uuid5(_RUN_ID_NAMESPACE, "campaign|" + "|".join(run.id for run in members))),
        scope_id=TOYHOST_SCOPE,
        name="retrieval depth sweep",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id=path.template.id,
        run_ids=[run.id for run in members],
        declared_design=design,
        created_by="test:fixture",
    )


__all__ = [
    "RETRIEVAL_AXIS",
    "RETRIEVAL_BASE",
    "RETRIEVAL_SWEPT_DEPTH",
    "RUN_CHUNK_TOKENS",
    "RUN_DOCUMENTS",
    "RUN_K",
    "RUN_MODELS",
    "RUN_QUESTION_ID",
    "RUN_RETRIEVER_TOP_K",
    "ToyhostArm",
    "ToyhostRunPath",
    "execute_toyhost_run",
    "toyhost_retrieval_arms",
    "toyhost_retrieval_campaign",
    "toyhost_run",
    "toyhost_run_bundle",
    "toyhost_run_campaign",
    "toyhost_run_design",
    "toyhost_template",
    "toyhost_test_cases",
]
