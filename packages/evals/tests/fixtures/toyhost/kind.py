"""The toy host's ``CandidateKind`` — an invoice extractor driven by a scripted client.

The candidate-kind seam is the one thing that varies between evaluable subjects, and every
``CandidateKind`` lives on the host side: the engine's runner prepares one per cell and invokes it,
and never learns what it is driving.

**Everything it is not.** No conversation, so no simulated other side. No subject minted from a
snapshot — the extractor's configuration IS the subject, and it arrives as the batch's own
recorded levers. No model grader: ``field_accuracy`` is computed here, against an adjudicated
key, which is why the profile declares the core's three model-grader apparatus dimensions
inapplicable. The standard matrix wires no judge service; ``judged=True`` is the variant that does.

**The scripted client is the only thing swapped out, and it reports usage.** A cell that spent
nothing observable would leave the run with no usage rows and no cost, and every claim about the
cost-vs-latency axis would then be a claim about zeros. The script is deterministic per
``(model, document)``, which has one consequence worth stating rather than hiding: **the two
``k`` repeats of one cell are exact duplicates**, so within-cell dispersion is zero and the
spread in an arm's numbers is the three documents' alone. That is the honest shape of a scripted
(or replayed) run, and it is what makes the twelve cells' values assertable to the digit.

The three spans and where each one opens
----------------------------------------

``prepare`` is handed this cell's :class:`~threetears.evals.contracts.CellSpanWindow` and carries it
on the instance, because the windows are the CELL's and one kind instance drives every cell of a
run. :meth:`ToyExtractorKind.invoke` opens both, and what falls inside each is a measurement
decision this kind owns:

* **identity** covers the whole of ``invoke`` — the document load, the world read, the grading.
  All of it is this cell's work even though none of it is the candidate's.
* **collecting** covers the extraction call and nothing else, so the buckets it fills are the
  candidate's own latency. Grading inside it would charge the candidate for the grader.

Inside the collection window the extractor emits its own turn-root span, so **a toy cell reports a
real ``total_ms`` and appears on the cost-vs-latency axis.**
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from threetears.evals.contracts import (
    CallLedger,
    CandidateOutput,
    CandidatePreparationFailed,
    CandidateTelemetry,
    CellCassettes,
    CellSink,
    CellSpanWindow,
    EvalTestCase,
    GoalStateOutcome,
    JudgedArtifact,
    JudgeEvidence,
    RoleUsage,
    WorldSeed,
    VariantConfig,
)
from threetears.evals.contracts.host import ApparatusError, SeedRefused, SubjectSnapshot, WorldRegistry, check_seed
from threetears.evals.run import GoalCheckUnevaluable, grade_goal_checks
from packages.evals.tests.fixtures.toyhost.product import ExtractionRequest, extraction_request
from packages.evals.tests.fixtures.toyhost.tracing import OP_HOST_GRADE, OP_MODEL_CALL, OP_TURN_ROOT, toy_span

#: The name a toy-host template carries on ``EvalTemplate.candidate_kind``, and the key the run
#: wires this kind under on ``RunnerOptions.candidate_kinds``. Opaque to the engine, which
#: dispatches on it at one site and never branches on its value.
TOY_EXTRACTOR_KIND = "toy-extractor"

#: The extractor's one action, as its call ledger records it: one call per field it emits.
EXTRACTOR_TOOL = "extractor"
EMIT_FIELD_ACTION = "emit_field"

#: The measure this kind takes, spelled as the toy host's profile declares it.
FIELD_ACCURACY = "field_accuracy"

#: Which case a cell ran, as the stimulus names it. A test case's ``variation_params`` is the
#: only per-case channel the engine has — there is no per-case host payload — so the document is
#: named here and the batch's own settings ride on the run's ``host_payload``.
DOCUMENT_PARAM = "document_id"

#: The fields every invoice in the toy corpus is graded on. Four, so an accuracy is a quarter at
#: a time and the bar at 0.92 is not reachable by a cell that missed one — the bar has to be
#: crossable and missable by the fixture or it is decoration.
InvoiceField = Literal["invoice_number", "invoice_date", "total_amount", "vendor_name"]
INVOICE_FIELDS: tuple[InvoiceField, ...] = get_args(InvoiceField)


@dataclass(frozen=True)
class ToyDocument:
    """One invoice the extractor is run over, with the key its fields are graded against."""

    document_id: str
    #: The adjudicated key — what a human pool decided each field actually is.
    key: dict[str, str]
    #: How much text the page carries, which is what makes one document dearer than another.
    prompt_tokens: int


#: The three documents this run's cases are drawn over. Three rather than one, because an arm's
#: dispersion has to come from somewhere and a scripted client gives it none across repeats.
TOY_DOCUMENTS: tuple[ToyDocument, ...] = (
    ToyDocument(
        document_id="doc-01",
        key={
            "invoice_number": "INV-1001",
            "invoice_date": "2026-01-14",
            "total_amount": "1420.00",
            "vendor_name": "Acme Paper",
        },
        prompt_tokens=900,
    ),
    ToyDocument(
        document_id="doc-02",
        key={
            "invoice_number": "INV-1002",
            "invoice_date": "2026-01-21",
            "total_amount": "88.50",
            "vendor_name": "Northwind Ink",
        },
        prompt_tokens=600,
    ),
    ToyDocument(
        document_id="doc-03",
        key={
            "invoice_number": "INV-1003",
            "invoice_date": "2026-02-02",
            "total_amount": "10450.75",
            "vendor_name": "Baltic Freight",
        },
        prompt_tokens=1500,
    ),
)


@dataclass(frozen=True)
class ExtractorScript:
    """What one extractor model does to the toy corpus, and what it costs to do it.

    A script rather than a model, and stated per document rather than as a rate: an accuracy
    that came out of a random draw would make the twelve cells' values unassertable, and a
    fixture whose own numbers cannot be quoted cannot exercise a gate that adjudicates numbers.
    """

    model: str
    #: Fields this model gets WRONG, per document. Everything not named here it gets right.
    misses: dict[str, tuple[str, ...]]
    #: Dollars per thousand prompt tokens — the better extractor is the dearer one, so the
    #: campaign's declared question is a trade rather than a foregone conclusion.
    usd_per_kilotoken: float
    #: How long one extraction takes, in seconds. Real wall-clock, deliberately: the span
    #: buckets are a MEASUREMENT, and a fixture that reported a scripted duration would be
    #: asserting its own arithmetic rather than the port's.
    latency_s: float


#: The two contestants. ``extractor-v3`` is better on every document and dearer per token, which
#: is the shape a reader has to weigh; an arm that won on every axis would leave a memo with
#: nothing to say that the table does not.
TOY_SCRIPTS: tuple[ExtractorScript, ...] = (
    ExtractorScript(
        model="extractor-v2",
        misses={"doc-01": ("total_amount",), "doc-02": (), "doc-03": ("total_amount", "vendor_name")},
        usd_per_kilotoken=0.004,
        latency_s=0.004,
    ),
    ExtractorScript(
        model="extractor-v3",
        misses={"doc-01": (), "doc-02": (), "doc-03": ("total_amount",)},
        usd_per_kilotoken=0.010,
        latency_s=0.012,
    ),
)


@dataclass
class ExtractionResult:
    """What one extraction call returned, and what the call reported about itself."""

    fields: dict[str, str]
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    model: str


class ScriptedExtractionClient:
    """The toy host's completion client: one scripted extraction per ``(model, document)``.

    A client rather than a patched function, because the kind's collaborators arrive as explicit
    typed arguments and a kind that reached a registry from inside ``invoke`` would be the shape
    the protocol forbids. It reports usage on every call, which is what puts rows on the cell.
    """

    def __init__(self, scripts: tuple[ExtractorScript, ...] = TOY_SCRIPTS) -> None:
        """Bind the client to its scripts.

        Args:
            scripts: One per extractor model this client can serve.
        """
        self._scripts = {script.model: script for script in scripts}
        #: Every call made, in order, as ``(model, document_id)``. The fixture's own record; the
        #: engine never sees it.
        self.calls: list[tuple[str, str]] = []
        #: Every request received, in order — what a boundary-equality test compares.
        self.requests: list[ExtractionRequest] = []

    @property
    def models(self) -> tuple[str, ...]:
        """The models this client is scripted for.

        Returns:
            The model names, in declaration order.
        """
        return tuple(self._scripts)

    async def extract(self, request: ExtractionRequest) -> ExtractionResult:
        """Run one extraction, reporting what it cost.

        Args:
            request: The request, as the product's one constructor
                (:func:`~packages.evals.tests.fixtures.toyhost.product.extraction_request`) built it.

        Returns:
            The extracted fields and the call's own usage.

        Raises:
            KeyError: No script covers ``model``. Deliberately unguarded — a run whose models and
                whose client disagree is misconfigured, and every cell of it would fail the same
                way, which is the condition the engine's own ``UnknownCandidateKind`` refuses to
                degrade into N identical excluded cells.
        """
        model, document = request.model, request.document
        script = self._scripts[model]
        self.calls.append((model, document.document_id))
        self.requests.append(request)
        # Real wall-clock, so the span buckets measure something. The whole matrix costs twelve
        # cells' worth of these — tens of milliseconds.
        await asyncio.sleep(script.latency_s)
        missed = set(script.misses.get(document.document_id, ()))
        fields = {name: ("" if name in missed else value) for name, value in document.key.items()}
        completion_tokens = 20 * len(INVOICE_FIELDS)
        return ExtractionResult(
            fields=fields,
            prompt_tokens=document.prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=round(script.usd_per_kilotoken * (document.prompt_tokens + completion_tokens) / 1000.0, 6),
            model=model,
        )


@dataclass
class ToyExtractorInstance:
    """One prepared extractor, ready to be run over one document.

    The opaque instance :meth:`ToyExtractorKind.prepare` hands back. The engine passes it
    straight to :meth:`ToyExtractorKind.invoke` and reads no field of it.
    """

    #: The model this cell bound the candidate to, off the variant config.
    model: str
    #: This cell's tracing windows, handed to ``prepare`` and opened in ``invoke``. Held per cell
    #: rather than on the kind because one kind instance drives every cell of a run — the
    #: property the protocol states and the reason ``span_window`` is an argument at all.
    span_window: CellSpanWindow
    #: What this cell's world was seeded to, read back off the world's own read handles at the
    #: end of seeding. The extractor is shown some of it and never the rest.
    world: dict[str, Any] = field(default_factory=dict)
    #: Which declared dimensions this cell actually set, for the run-time algebra.
    seeded: tuple[str, ...] = ()


class ToyExtractorKind:
    """An invoice extractor, as the engine's candidate-kind seam sees one.

    Its collaborators — the scripted client and the world it seeds through — arrive as explicit
    typed keyword arguments. One instance serves a whole run, so it carries no state from one
    ``prepare``/``invoke`` pair to the next: the cell's own state lives on
    :class:`ToyExtractorInstance`.
    """

    #: The carriers this kind attaches, and therefore the ones a seed can reach through. **The
    #: kind declaring itself the carrier is what makes the run-time algebra answerable**: a seed
    #: writes through the thing that supplies a dimension, so a dimension whose carrier this kind
    #: never attaches is ``out_of_play`` for the run however the template names it, and recording
    #: it otherwise would assert an instantiation that never happened.
    CARRIERS: tuple[str, ...] = ("page_reader", "console")

    def __init__(
        self,
        *,
        client: ScriptedExtractionClient,
        world: WorldRegistry,
        documents: tuple[ToyDocument, ...] = TOY_DOCUMENTS,
        judged: bool = False,
        graded_fields: tuple[str, ...] = INVOICE_FIELDS,
        goal_checks: tuple[str, ...] = (),
    ) -> None:
        """Wire the extractor's collaborators.

        Args:
            client: The scripted extraction client.
            world: The world this kind seeds and reads through. The registry rather than the
                state object: every write goes through a declared ``seed`` handle and every read
                through a declared ``read`` handle, so the path a run takes is the path the
                conformance kit proves.
            documents: The corpus cases are drawn from.
            judged: Whether each cell also hands a judge its evidence: the invoice as the material
                the extraction was produced from, and the extraction as the artifact. A host renders
                that evidence because only it knows its document's shape; the engine carries the two
                strings to the judge without reading either. ``False`` is the standard matrix, which
                wires no judge: the kind then declares itself unjudged, and the runner refuses a
                judge wired to it.
            graded_fields: The invoice fields ``field_accuracy`` is computed over — what a template's
                ``kind_spec`` states (``contract.py``'s ``ExtractorSpec``). Every field, by default.
            goal_checks: The template's goal checks, graded at each cell's end against the call ledger
                this kind fills and the world it read back — through the engine's
                :func:`~threetears.evals.run.grade_goal_checks`, the one evaluation every kind uses.
        """
        self._client = client
        #: The extraction is a document, judged against the invoice it was read from — or, unjudged,
        #: graded by code alone. The one fact the evidence below is rendered from, so the kind
        #: cannot declare one thing and hand the runner another.
        self.judged_artifact = JudgedArtifact.DOCUMENT if judged else JudgedArtifact.UNJUDGED
        self._world = world
        self._documents = {document.document_id: document for document in documents}
        self.graded_fields = graded_fields
        self.goal_checks = goal_checks
        #: What each cell measured, keyed by ``(model, document_id)``. Kept for the host's own
        #: assertions about what it graded — the number itself reaches the result through
        #: ``CandidateOutput.host_measures``, so nothing downstream depends on this. Keyed on the
        #: cell's coordinates rather than appended, so the two ``k`` repeats of one cell collapse
        #: onto one entry and this ledger cannot be mistaken for a per-observation record.
        self.measures: dict[tuple[str, str], dict[str, float]] = {}

    def seedable_dimensions(self, seed: WorldSeed) -> tuple[str, ...]:
        """Which declared dimensions a seed asks this kind to set.

        The run records this on ``EvalRun.world_placements`` through
        :meth:`~threetears.evals.contracts.host.world.WorldRegistry.place`, which needs the facts a run
        actually produced rather than a mode anybody declared.

        Args:
            seed: The template's world seed.

        Returns:
            The dimension names, sorted, covering every namespace the seed names.
        """
        return tuple(sorted(name for namespace in seed.namespaces.values() for name in namespace))

    async def prepare(
        self,
        *,
        subject_snapshot: SubjectSnapshot | None,
        variant_config: VariantConfig,
        world_seed: WorldSeed,
        span_window: CellSpanWindow,
        cassettes: CellCassettes | None,
    ) -> ToyExtractorInstance:
        """Seed this cell's world through the registry and hand back the prepared extractor.

        The refusals are the engine's, not this kind's: :func:`~threetears.evals.contracts.host.check_seed`
        is the one seed walk every host calls — a carrier this kind does not attach, a key no
        dimension declares, a value under the wrong carrier, a dimension no run can set, and a value
        its dimension's schema refuses. Any of them would leave the grader reading state the
        extractor never received, which is the candidate/judge divergence the seeding step exists to
        prevent. What is this kind's is what a refusal costs here — one excluded cell under
        ``seed_failed`` — and the carriers it attaches. Everything is checked before anything is
        written, so a refused seed leaves the world as it was.

        Args:
            subject_snapshot: The engine's view of the subject. Read only for the labels a
                diagnostic names; the extractor's substance is its batch's recorded levers.
            variant_config: The A/B'd stack this cell runs; its candidate model is the extractor.
            world_seed: The world state this scenario presumes, keyed by carrier.
            span_window: This cell's tracing windows, carried out on the instance.
            cassettes: Unwired — the extractor calls no tool, so a cassette run of it is refused.

        Returns:
            The prepared extractor.

        Raises:
            CandidatePreparationFailed: The seed named something this world cannot set, or set it
                to a value its schema refuses. Recorded by the runner as one cleanly-excluded cell
                under ``seed_failed``, which is the arm that sends an operator to the apparatus
                rather than to the subject factory.
        """
        try:
            writes = check_seed(self._world, world_seed.namespaces, attached=self.CARRIERS)
        except SeedRefused as refused:
            raise CandidatePreparationFailed(f"apparatus: {refused}", termination="seed_failed") from refused
        for write in writes:
            await self._world.call(write.handle, write.value)

        # Read back through the declared READ handles, not off the object the seeds wrote: a
        # seeder wired to nothing is the founding defect, and only a round trip catches it. Every
        # declared dimension, not just the seeded ones — the witnessed one is state this cell
        # observed and did not choose, and that is a fact about the cell.
        world = {
            declared.name: await self._world.call(declared.read)
            for declared in self._world.declarations
            if declared.read is not None
        }
        return ToyExtractorInstance(
            model=variant_config.candidate_model,
            span_window=span_window,
            world=world,
            seeded=tuple(sorted(write.name for write in writes)),
        )

    async def invoke(self, instance: ToyExtractorInstance, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Extract one document's fields, grade them against the key, and report both.

        Args:
            instance: Whatever :meth:`prepare` returned.
            test_case: The cell's stimulus. Its ``variation_params`` names the document.
            sink: This cell's sink. No progress reading is registered: the cell makes one call,
                and a deadline that cuts it off cuts off the only spend there is to report.

        Returns:
            The extracted record as the judged artifact, the mechanical tier, and the cell's
            telemetry.

        Raises:
            ApparatusError: The case names a document the corpus does not carry — the case bank
                is broken, not the extractor. The engine records it as this one cell excluded under
                ``apparatus_failed`` and goes on; any other exception out of ``invoke`` ends the run.
        """
        scopes = instance.span_window
        with scopes.identity():
            document_id = test_case.variation_params.get(DOCUMENT_PARAM, "")
            document = self._documents.get(document_id)
            if document is None:
                # The rig's fault, raised as one: ``ApparatusError`` is the engine's cell-level arm
                # for exactly this, and the cell is excluded rather than scored on a broken case.
                raise ApparatusError(f"no document {document_id!r} in the toy corpus for test case {test_case.id}")

            # COLLECTING covers the extraction and nothing else. The grading below is the host's
            # own grader, and time spent in it is not time the candidate spent.
            with scopes.collecting():
                with toy_span(f"extract:{document.document_id}", operation=OP_TURN_ROOT, model=instance.model):
                    with toy_span("extract.call", operation=OP_MODEL_CALL, model=instance.model):
                        # The product's own constructor, not a copy of it: the fidelity contract
                        # (``fidelity.py``) holds this call to it.
                        request = extraction_request(model=instance.model, document=document)
                        extraction = await self._client.extract(request)

            # What the candidate did, recorded through the engine's helper from what the call
            # actually returned: one call per field the extraction emitted, empty or not.
            ledger = CallLedger()
            for name in extraction.fields:
                ledger.record(EXTRACTOR_TOOL, EMIT_FIELD_ACTION, {"field": name})

            # Spanned, and OUTSIDE the collection window on purpose: grading is the host's own
            # grader, not the candidate's work, and the port's two extents differ for exactly
            # this reason. Recorded rather than argued — a test asserts no ``grade`` span reaches
            # any cell's collected record, which is what makes the difference between the two
            # windows a fact rather than a comment.
            with toy_span(f"grade:{document.document_id}", operation=OP_HOST_GRADE, model=instance.model):
                graded = self.graded_fields
                correct = [name for name in graded if extraction.fields.get(name) == document.key[name]]
                accuracy = len(correct) / len(graded)
                # One entry per cell coordinate, overwritten identically by the second k repeat.
                self.measures[instance.model, document.document_id] = {FIELD_ACCURACY: accuracy}
                # The template's goal checks, graded by the engine against the ledger and the world
                # this cell read back. A check that cannot be evaluated is the rig's fault, not a
                # verdict on the extractor, so it excludes the cell rather than failing it.
                try:
                    goal_outcomes = grade_goal_checks(
                        self.goal_checks,
                        ledger=ledger,
                        end_state=instance.world,
                        variation=test_case.variation_params,
                        world=self._world,
                    )
                except GoalCheckUnevaluable as unevaluable:
                    raise ApparatusError(str(unevaluable)) from unevaluable

        return CandidateOutput(
            # One JSON document: the extracted record. The engine never looks inside it.
            output=[
                {
                    "document_id": document.document_id,
                    "fields": extraction.fields,
                    "language": instance.world.get("document_language"),
                    "scan_quality": instance.world.get("scan_quality"),
                }
            ],
            mechanical_facts=[
                GoalStateOutcome(
                    expression=f"{FIELD_ACCURACY} >= 0.92",
                    passed=accuracy >= 0.92,
                    detail=f"{FIELD_ACCURACY}={accuracy:.2f} — {len(correct)} of {len(graded)} graded fields exact",
                ),
                *goal_outcomes,
            ],
            # Stored on the cell's trace, so a re-check re-grades the goal checks from exactly this.
            call_ledger=ledger,
            telemetry=CandidateTelemetry(
                usage=[
                    RoleUsage(
                        role="candidate",
                        model=extraction.model,
                        prompt_tokens=extraction.prompt_tokens,
                        completion_tokens=extraction.completion_tokens,
                        cost_usd=extraction.cost_usd,
                        price_source="toyhost_script",
                    )
                ],
            ),
            # The mechanical grade, across the seam in the open: a registered measure name and
            # a float are engine vocabulary, so this needs no opaque carrier and gets none. The
            # engine lands it on ``EvalResult.host_measures`` and interprets nothing.
            host_measures={FIELD_ACCURACY: accuracy, "fields_correct": float(len(correct))},
            # Which fields were wrong is this kind's fact, not a measure: the engine has no word for
            # an invoice field, so it travels opaque, stored verbatim on the result for a reader of
            # this host to unpack. The count beside it is the measure; the names are the payload.
            kind_payload={"missed_fields": [name for name in graded if name not in correct]},
            judge_evidence=(
                JudgeEvidence(
                    case_material=render_invoice(document), artifact=json.dumps(extraction.fields, sort_keys=True)
                )
                if self.judged_artifact is JudgedArtifact.DOCUMENT
                else None
            ),
        )


def render_invoice(document: ToyDocument) -> str:
    """The invoice as a judge reads it: the page's fields, one per line, in declared order.

    The toy corpus carries no page image, so its adjudicated key stands in for the page — which
    makes a judge able to check an extraction against it, the one thing the evidence is for.

    Args:
        document: The invoice.

    Returns:
        The page, as text.
    """
    return "\n".join([f"Invoice {document.document_id}", *(f"{name}: {document.key[name]}" for name in INVOICE_FIELDS)])


__all__ = [
    "DOCUMENT_PARAM",
    "EMIT_FIELD_ACTION",
    "EXTRACTOR_TOOL",
    "FIELD_ACCURACY",
    "INVOICE_FIELDS",
    "TOY_DOCUMENTS",
    "TOY_EXTRACTOR_KIND",
    "TOY_SCRIPTS",
    "ExtractionResult",
    "ExtractorScript",
    "ScriptedExtractionClient",
    "ToyDocument",
    "ToyExtractorInstance",
    "ToyExtractorKind",
    "render_invoice",
]
