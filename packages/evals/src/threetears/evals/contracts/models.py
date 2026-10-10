"""Eval system data models.

Four managed shapes carry an evaluation:

- ``EvalTemplate`` — abstract blueprint: intent, applicability, variation axes,
  conversation, world seed, goal-state checks, rubric dimensions, kind spec.
- ``EvalTestCase`` — immutable concrete inputs: template_id + variation_params.
- ``EvalRun`` — execution record: subject snapshot, candidate model, k_runs, frozen
  ``test_case_ids``, status, progress.
- ``EvalResult`` — one test case x one model x one k-iteration: goal-state
  outcomes, rubric scores, cost, and what the candidate's kind reported.

Schema rule: every stored entity carries ``schema_version`` (:data:`EVAL_SCHEMA_VERSION`), and
a read is strict — an unknown field, a missing required one, or a document written under another
schema version is refused, never coerced. Stored eval documents are disposable: across a schema
change they are dropped and regenerated, not migrated.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any, Literal, NamedTuple, Self, get_args

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from threetears.evals.contracts.base import EvalBaseModel, EvalDocumentModel, VerbatimJsonObject, VerbatimObject
from threetears.evals.contracts.call_ledger import CallLedger, RecordedCall
from threetears.evals.contracts.hashing import canonical_digest
from threetears.evals.contracts.dsl import DSLError, extract_paths, parse, referenced_fires
from threetears.evals.contracts.host.spend import ExternalSpend
from threetears.evals.contracts.host.subject import SubjectSnapshot
from threetears.evals.contracts.host.values import SweepableValue
from threetears.evals.contracts.host.world import WorldPlacement, WorldRegistry
from threetears.evals.contracts.judge_attribution import (
    JudgeAttributionSource,
    JudgeAttributionState,
    attribution_state,
)
from threetears.evals.contracts.prose import ModelProse
from threetears.evals.contracts.world_events import WorldEvent
from threetears.observe import get_logger

log = get_logger(__name__)


EVAL_SCHEMA_VERSION: int = 8
"""The schema version every stored eval document is written under, and the only one a read accepts.

Bump it when a stored shape changes so that a document written before the change would not mean
what it says after it — a field renamed, retyped, removed or made required. A bump is a drop:
documents written under the old version are refused on read (:data:`SchemaVersion`) and deleted,
never migrated, because a read path that coerces old evidence outlives the evidence by years.

**v6** is the scope-and-kind contract: one opaque ``scope_id`` on every stored document, kind-owned launch
overlays and template ``kind_spec``, kind-rendered judge evidence, kind-owned result payloads
(``async_deliveries``, ``kind_payload``, ``candidate_instance_id``), and every field its writers
set required — no field is read as "absent because older" — and a run naming one
``candidate_model`` (with its one ``variant_levers`` map) where it carried a list. Nothing written
before it loads. (The required fields and the one-model run joined v6 before it was first released,
so they share its number; the one stored value the required fields change is
``GenerationProvenance.cell_model_version``, which the generator had never written. So did the
cassette corpus — ``EvalCassette`` keyed by corpus and occurrence, ``EvalRun.cassette_corpus_id`` in
place of ``cassette_version`` on the run and the result — and the background-work spend
``AsyncDelivery`` carries; and ``WorldEvent.event``, the identity of the event a firing names, required
on every firing so a firing's ``armed`` is the event's provenance rather than the dimension's.)

**v7**: a calibration rating records who KIND of rater wrote it (``CalibrationRating.rater_kind``, a person or
an agent), required, so an agent's rating is never read as a person's. A rating written before it says
nothing about which it was, so nothing written under v6 loads.

**v8**: judged readings carry a code-decided evidence tier (PD-13). A stored analysis's judged evidence rows
(``EvidenceRow.judged_tier``) and its decision surface's judged readings (``JudgedReading.evidence_tier``)
carry a tier, required; the finding tier ``directional`` is gone from ``EvidenceTier``; a repeated judge score
records the config that asked for the score it repeats (``RepeatedScore.first_judge_config_id``, required),
so a repeat under one judge prompt never measures another. A v7 analysis holding a judged reading cannot say
what tier it stood on, so nothing written under v7 loads.

**Within v8, not a bump**: ``RubricScore.axis`` joined as an OPTIONAL field — the rubric axis the judge
stamped from the dimension's definition, so a boundary (guardrail) score stays out of the composite and
pass^k. A score judged before it carries None, and is read as capability, which is how it was read then:
its result's composite does not move, and the bundle names the dimensions read that way
(``GuardrailReadings.unstamped_dimensions``) rather than presenting them as known capability.

**Within v8, not a bump**: ``EvalResult.turns_delivered`` joined as an OPTIONAL field — how many turns the
candidate delivered, which decides whether a model failure's time and spend are a turn's. A result written
before it carries none and still means what it says; it reads as None, "nothing counted", and every reader
falls back to the failure's cause alone (``delivered_a_turn``). It, and the decision surface's
``CellFacts.n_candidate_failed`` and ``n_no_turn`` (and their ``StratumFacts`` twins), None on an analysis
frozen before them, are deliberate exceptions to v6's "no field is read as absent because older": requiring
them would drop every stored document to learn counts the old ones never had, and their honest reading is
"unknown", which None states. ``EvalAnalysis.judged_tier_rule`` joined the same way: the rule its judged tiers
were decided by, None on an analysis stored before tiers were decided on the agreement's interval — whose tiers
were the point estimate against the bar, and are rendered as that, never as the interval rule's claim. And
``EvalRun.goal_check_proofs``: whether each goal check was shown, at launch, to beat doing nothing; None on a run
launched before it, read as unproven. And ``GenerationProvenance.bundle_schema_version`` and
``.host_declarations_digest``: the bundle shape and the host's declarations a generation ran over, None on an
analysis stored before them, read as "cannot say" — never as the current version.

**Within v8, not a bump: fields retired** (``__retired_fields__``, read only by a stored read — see
:mod:`threetears.evals.contracts.base`). ``LeverCoverage.confidence`` is removed: it was a fixed lookup on the
lever's ``status``, so a stored analysis loses nothing when the key is discarded on read. ``EvalCampaign.status``
(open / closed) is removed: nothing could change it after creation and nothing enforced it, so a stored
campaign's ``closed`` froze nothing and discarding it changes no membership and no analysis; the one thing it
fed, ``list_campaigns``'s ``status`` filter, is gone with it. ``CampaignDesign.controls`` is renamed ``held_fixed``
(one letter from ``control``, it named a different thing), and the bundle's ``controls_reading`` with it
(``held_fixed_reading``): a stored campaign, an analysis's ``design_snapshot`` and a reporter case's frozen bundle
read the old key under the new name, value unchanged.

**Within v8, not a bump**: ``RubricDimTombstone`` joined as a new stored type — the record a rubric dim delete
leaves so the definition seed does not write the key back. A store written before it holds none, which reads as
"no key was deleted since": a dim deleted before then is still written back at the next seed, as it was then.

**Within v8, not a bump**: ``RoleUsage.served_model`` joined as an OPTIONAL field — the model the provider's
response named as having answered the row's calls, which for a candidate launched on a floating alias is the
only record of which model produced its numbers. A row stored before it carries None and reads as "not
recorded", never as the alias in ``model``: the analysis names such an arm's served model unknown rather than
the one requested.

**Within v8, not a bump**: the judge's temperature joined as OPTIONAL fields (#633) — ``RubricScore.judge_temperature``
(what the call was sent at), ``EvalRun.judge_temperature`` (what a dimension with no config was requested at) and
``RepeatedScore.first_judge_temperature``. A document stored before them carries None and reads as not recorded:
its unconfigured dimensions were requested at the provider's default, which is not today's 0, so such a run's
roles component is not composable, its scores' judge reads unknown, and nothing pools it with a run judged at 0.

**Within v8, not a bump**: ``EvalRun.measure_latency`` and ``EvalRun.cell_concurrency`` joined as OPTIONAL fields
(#701) — whether the launch declared latency under test, and how many of the run's cells executed at once. A run
stored before them carries None in both: its cells executed one at a time (the runner of that build had no other
way), so its ``cell_concurrency`` reads as 1, and whether it declared latency reads as not recorded — never as
declared. Whether another RUN executed beside it is what its results' ``execution_mode`` says, as it always was.
``CampaignDesign.measure_latency`` joined the same way, defaulting to False: a campaign (or an analysis's design
snapshot) stored before it reads as not declaring latency under test, which is what it declared — a stored
design asking about latency still loads, and is refused only when it is declared again.

``JudgeConfigTombstone`` joined the same way, for a judge config's slot; a config deleted before it is written
back at the next seed. ``EvalRun`` gained ``goal_check_proof_rules`` (None on a run stored before it, read as rules
1, so its ``proven`` checks read unproven) and ``refused_goal_checks`` (None, not recorded), and ``EvalResult``
gained ``judge_cannot_tell_boundary`` (empty, its can't-tells read as capability) — all optional within v8.

**Within v8, not a bump**: ``ClientRequestSettings.strict_output`` joined as a defaulted field (#686) — whether a
role's requests were to be routed only to providers honouring every parameter sent. A stamp stored before it
carries none and reads False, "no such requirement was stated", which is what the engine sent then. Its apparatus
level (``judge_request_settings`` / ``simulator_request_settings``) leaves the flag out while it is False, so a
stored run's level is unchanged; a judge stamp carrying True reads as a different level from one stored before.

**Within v8, not a bump**: ``CampaignDesign.guardrail_margins`` joined as an OPTIONAL field (#697) — the margin
each judged guardrail (a boundary rubric dimension) is held to. A campaign, or an analysis's design snapshot,
stored before it carries none and reads as declaring none: its judged guardrails are held at zero change, exactly
as they were decided then, so no stored decision moves.

**Within v8, not a bump**: ``EvalRun.declared_margins`` joined as an OPTIONAL field (#698) — the margins a launch
declared on core rate measures (accuracy). A run stored before it carries none and reads as declaring none, so no
comparison over it reads a margin it never declared.

**Within v8, not a bump**: ``EvalRun.declared_measures`` joined as an OPTIONAL field — how the launching host
declared each of its own measures to be read (direction, merit axis, guardrail, margin, range). A run stored before
it carries none, and a campaign of such runs is read on the reading host's declarations, as every campaign was.
"""


def _current_schema_only(version: int) -> int:
    """Refuse a document written under any schema version but this build's.

    Args:
        version: The document's ``schema_version``.

    Returns:
        The version, unchanged.

    Raises:
        ValueError: ``version`` is not :data:`EVAL_SCHEMA_VERSION`.
    """
    if version != EVAL_SCHEMA_VERSION:
        raise ValueError(
            f"this document was written under eval schema v{version} and this build reads v{EVAL_SCHEMA_VERSION} "
            "only; stored eval documents are dropped across a schema change, never migrated"
        )
    return version


#: The ``schema_version`` field type of every stored eval entity: defaulted to the current version on
#: write, and refusing any other on read.
SchemaVersion = Annotated[int, AfterValidator(_current_schema_only)]


def _finite_setting(value: str | bool | int | float) -> str | bool | int | float:
    """Refuse a non-finite number as an apparatus setting — it has no JSON form and no level to compare."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"an apparatus setting is a finite number, a string or a bool; got {value!r}")
    return value


#: One host-declared apparatus value a launch sets (:attr:`EvalRun.apparatus_settings`): a string, a
#: bool, or a finite number — a level two runs can be compared on, and hashed into the measurement
#: context. Strict, so ``"1"`` and ``1`` stay two levels rather than one coerced into the other.
ApparatusSettingValue = Annotated[StrictStr | StrictBool | StrictInt | StrictFloat, AfterValidator(_finite_setting)]


def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


# =============================================================================
# Embedded shapes — variation generation, simulator, world, rubric
# =============================================================================


class VariationAxis(EvalDocumentModel):
    """One axis along which test cases vary for a template.

    Three generator types:

    - ``enum`` — deterministic; yields each value in ``values`` exactly once.
    - ``sample`` — non-deterministic; samples from ``values`` (with or without
      replacement is the variation generator's choice).
    - ``llm`` — LLM-driven; generates novel values, deduplicating against
      existing persisted test cases for the template.

    ``values`` is required for ``enum`` and ``sample``; ignored for ``llm``
    where the prompt and template intent drive generation.
    """

    name: str = Field(
        min_length=1, description="Axis identifier; referenced by goal-state expressions as variation.<name>"
    )
    generator: Literal["enum", "sample", "llm"]
    values: list[str] = Field(default_factory=list)
    description: str = Field(default="")
    stratum: bool = Field(
        default=False,
        description=(
            "Whether this axis's value is the stratum of every case generated from it — the kind of case the "
            "analysis reads results by (`EvalTestCase.stratum`). The generator copies the value into the case's "
            "stratum when it writes the case; nothing reads this flag afterwards, since the case carries its own "
            "stratum. At most one axis of a template is nominated, and only an `enum` or `sample` axis, whose "
            "values are a closed set: an `llm` axis writes a new value for every case, so each case would be a "
            "stratum of one."
        ),
    )


class ActorPolicy(EvalDocumentModel):
    """One simulated actor in a template's conversation.

    A conversation may have several: a party of players, a panel, a room. Each speaks under its
    own ``policy`` and pursues its own ``intent``; :class:`ConversationSpec` says who speaks next.
    """

    id: str = Field(min_length=1, description="Stable identifier within the template")
    name: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The name this actor goes by in the candidate's world — what a message from it is attributed to "
            "where the candidate reads it. Distinct from `id`, which is the template's handle for the actor and "
            "never shown to the candidate. Optional because a host may have no notion of who is speaking; the "
            "host decides how an unnamed actor's message reads."
        ),
    )
    policy: str = Field(min_length=1, description="How this actor speaks and behaves: tone, manner, temperament")
    intent: str = Field(min_length=1, description="What this actor wants from the conversation")
    initial_utterance_template: str | None = Field(
        default=None,
        description="Optional first message; supports {variation.<field>} placeholders.",
    )


#: The scheduler's answer that the current speaker round is over and the candidate answers next. An
#: ``llm_decided`` scheduler chooses between it and the actors' ids, so no actor may carry it as an id.
ROUND_DONE = "round_done"

#: The speaker label the candidate's own turns carry in a simulated transcript. Reserved for the same
#: reason as :data:`ROUND_DONE`: an actor named for it would read as the candidate to every other actor.
CANDIDATE_SPEAKER = "__candidate__"


class ConversationSpec(EvalDocumentModel):
    """The simulated side of a conversing candidate: who talks to it, in what order, for how long.

    One optional block on :class:`EvalTemplate` rather than loose fields, because they mean something
    only together and only to a kind whose candidate converses. A template for a document or a
    classifier carries none, which says so; a conversing kind reads it and drives the engine's
    :class:`~threetears.evals.run.simulator.TurnDriver` over it, through
    :func:`~threetears.evals.run.conversation.drive_conversation`.

    **A conversation is a sequence of speaker rounds.** In each round one or more actors speak, then
    the candidate answers the round once. ``max_turns`` counts the candidate's answers, so it counts
    rounds that were answered. ``max_speakers_per_round`` bounds how many simulated utterances a round
    may hold; ``turn_scheduler`` decides who fills them.

    **Sessions split the conversation.** ``sessions`` is how many sittings the ``max_turns`` answers
    are spread over, as evenly as integer division allows. Between two sessions the driver marks a
    break on the next utterance it delivers, and the kind decides what a break means for its candidate
    (a fresh conversation history, a new game session) while its world persists.
    """

    actors: list[ActorPolicy] = Field(
        min_length=1,
        description=(
            "The simulated actors. Their order is the round-robin order, and the first actor's "
            "initial_utterance_template, when set, opens the conversation under either scheduler. At least "
            "one: a conversation with nobody on the other side is not a conversation, and a template with "
            "nothing to simulate declares no block at all. Ids are unique, and neither 'round_done' nor "
            "'__candidate__', which the scheduler and the transcript reserve."
        ),
    )
    turn_scheduler: Literal["round_robin", "llm_decided"] = Field(
        default="round_robin",
        description=(
            "Who speaks next within a round. round_robin: the next actor in list order, until the round "
            "holds max_speakers_per_round utterances. llm_decided: a simulator-role model call chooses the "
            "next actor, or that the round is done once it holds at least one utterance."
        ),
    )
    max_speakers_per_round: int = Field(
        default=1,
        ge=1,
        le=20,
        description=(
            "The most simulated utterances one round may hold before the candidate answers. An actor may "
            "speak more than once in a round under llm_decided. 1 keeps the one-utterance, one-answer shape."
        ),
    )
    max_turns: int = Field(
        default=10, ge=1, le=100, description="The candidate turns after which the conversation stops."
    )
    sessions: int = Field(
        default=1,
        ge=1,
        le=100,
        description=(
            "How many sittings the conversation's candidate turns are spread over; each boundary is a "
            "session break the driver marks. At most max_turns, since a session holds at least one turn."
        ),
    )

    @model_validator(mode="after")
    def _actors_and_sessions_are_addressable(self) -> ConversationSpec:
        """Refuse a block a driver could not run as written.

        Returns:
            The block, unchanged.

        Raises:
            ValueError: Two actors share an id, an actor's id is one the scheduler or the transcript
                reserves, or ``sessions`` exceeds ``max_turns`` and so names a session with no turn in it.
        """
        seen: set[str] = set()
        for actor in self.actors:
            if actor.id in (ROUND_DONE, CANDIDATE_SPEAKER):
                raise ValueError(f"actor id {actor.id!r} is reserved; choose another id")
            if actor.id in seen:
                raise ValueError(f"actor id {actor.id!r} appears twice; a scheduler could not tell the two apart")
            seen.add(actor.id)
        if self.sessions > self.max_turns:
            raise ValueError(
                f"sessions={self.sessions} exceeds max_turns={self.max_turns}; every session holds at least one turn"
            )
        return self


class WorldSeed(EvalDocumentModel):
    """Initial state for the eval's stateful world.

    ``namespaces`` keys are carriers and each sub-key names one state dimension, through the host
    registry's addressing (``WorldRegistry.address``). Each value is the literal initial state of its
    namespace — data the template states, never a reference resolved against anything at run start.

    **What may be seeded is the host's world registry's answer, not this model's.** A key
    addressing no declared dimension — or one declared and unseedable, which a subject
    perceives and no run controls — is refused before anything is applied, and the refusal names
    what a run can seed. Read the registry rather than a list here: this docstring cannot hold
    the vocabulary without going stale under the carriers that add to it, and a stale list here
    is one an author writes a template against and pays a run to discover.
    """

    namespaces: dict[str, Any] = Field(default_factory=dict)

    ambient_perturbation_turns: list[Annotated[int, Field(ge=1)]] = Field(
        default_factory=list,
        description=(
            "The candidate turns, counted from 1, before which the host's ambient-perturbation handle moves "
            "state no dimension declares — so a run, and not only the conformance kit, exposes a candidate that "
            "perceives undeclared state. Empty perturbs nothing. Applied by the cell's world session when the "
            "kind reaches each turn (``WorldSession.at_turn``) and recorded on the result's ``world_events``; a "
            "turn the cell never reached is never applied, and a cell whose kind announced no turn (or skipped "
            "one) is refused rather than recorded as perturbed. Refused at authoring on a host whose world has no "
            "ambient-perturbation handle, and frozen onto the run beside the namespaces "
            "(``EvalRun.resolved_ambient_perturbation_turns``)."
        ),
    )

    @field_validator("ambient_perturbation_turns")
    @classmethod
    def _each_turn_once(cls, value: list[int]) -> list[int]:
        """Refuse a turn named twice: perturbation is applied once per named turn, so a repeat states nothing.

        Args:
            value: The turns.

        Returns:
            They, unchanged.

        Raises:
            ValueError: A turn appears more than once.
        """
        if len(set(value)) != len(value):
            raise ValueError(f"ambient_perturbation_turns names a turn twice: {value!r}")
        return value


class Precondition(EvalDocumentModel):
    """One thing a template presumes about the world before the subject's first turn.

    A scenario that only makes sense against a particular starting state has always presumed
    one; what changes here is that it *says so*. Nothing parsed a presumption before this
    field existed, so a probe written for a queue with three jobs in it ran against an empty
    one and scored the subject on a situation it was never placed in.

    **An expression plus its reason, never a bare expression.** The same rule a world
    declaration lives under: a bare name is a label, and a bare expression is one too. The
    reader that pays for it is the excluded-run record — a precondition that did not hold makes
    an observation invalid rather than low-scoring, and *"expression 2 was false"* does not tell
    the person reading that exclusion whether the template or the world was wrong.

    The postcondition half — ``EvalTemplate.goal_state_checks`` — stays a bare expression list,
    and the asymmetry is deliberate: a failed goal check is the measurement, read beside a
    rubric that already says what was being scored, while a failed precondition is the
    apparatus refusing to run and has no such neighbour.
    """

    expression: str = Field(
        min_length=1,
        description=(
            "A goal-state DSL expression over the world at t=0 — e.g. 'state.orders.pending.length "
            ">= 3'. Parsed when the template is written, and resolved against the host's declared "
            "world so a presumption naming state nobody can set is refused before a run. "
            "Evaluated once per cell the moment the world is seeded: a cell whose precondition "
            "did not hold is recorded with its "
            "``PreconditionOutcome``s and EXCLUDED from the aggregates, never scored."
        ),
    )
    presumes: str = Field(
        min_length=1,
        description=(
            "What this precondition presumes, in prose — the sentence an excluded run is read "
            "with. Required: a bare expression names no reason, and the exclusion record is "
            "unreadable without one."
        ),
    )

    @field_validator("expression")
    @classmethod
    def check_expression_parses(cls, value: str) -> str:
        """Refuse an expression the language cannot read, where it is written.

        Parse time is the whole point of a closed grammar: an unparseable precondition left to a
        run is an apparatus failure discovered after the spend, attributed to whatever the run
        was doing.

        Args:
            value: The expression text.

        Returns:
            It unchanged.

        Raises:
            ValueError: The expression is malformed or uses something outside the grammar, or reads
                ``fired()``.
        """
        try:
            parse(value)
            triggered_dimensions = referenced_fires(value)
        except DSLError as malformed:
            raise ValueError(f"precondition expression does not parse: {malformed}") from malformed
        if triggered_dimensions:
            raise ValueError(
                f"precondition reads fired({triggered_dimensions[0]!r}), and a precondition reads the world at t=0, before any "
                "trigger could fire — a check on what fired is a goal check"
            )
        return value

    @property
    def presumed_paths(self) -> tuple[str, ...]:
        """The world paths this precondition reads, statically, without evaluating it.

        Returns:
            Dotted paths for a world registry to resolve. Case parameters are deliberately not
            here: ``variation.tone`` is an input the case carries, not state anything seeds.
        """
        return extract_paths(self.expression).world


#: What a goal check says about the behaviour it grades, and so which verdict "the candidate did
#: nothing" must get. An ``act`` check grades something the candidate must DO, so doing nothing
#: fails it; a ``hold`` check grades something the candidate must NOT do (or must leave alone), so
#: doing nothing passes it. A check whose verdict on doing nothing is the opposite of its intent
#: grades something other than what its author meant.
GoalCheckIntent = Literal["act", "hold"]

#: Whether a goal check was shown, when its run launched, to tell its outcomes apart
#: (:func:`threetears.evals.run.check_controls.goal_check_proofs`). ``proven``: the template names a control and
#: the check gives the verdicts its intent requires on it and on the do-nothing control. ``unproven``: the
#: template names no control for it — a template written past authoring, or a quick run's — so nothing shows
#: its pass rate is not what a candidate that did nothing would score. ``refuted``: a control is named and the
#: check does not tell it from doing nothing, or cannot be evaluated against it. Only ``proven`` reads as a
#: measurement of the behaviour; the other two are marked wherever the check's pass rate is shown.
GoalCheckProof = Literal["proven", "unproven", "refuted"]

#: The rules a run's goal-check proofs are derived under, stamped on the run beside them
#: (``EvalRun.goal_check_proof_rules``). ``1``: a control's case parameters were read under the types it stated,
#: so a check reading a parameter as a list could pass its control and be stamped proven, then fail every case
#: (#665). ``2``: a control's parameters are read as a case stores them, one string each, and a control stating
#: another type is refuted. A ``proven`` recorded under an older rule is read as ``unproven``
#: (:func:`goal_check_proofs_as_read`): the controls are editable and were not frozen with the run, so the proof
#: cannot be re-derived for the template the run actually graded, and a proof earned under a rule since found
#: wrong is not one.
GOAL_CHECK_PROOF_RULES = 2

#: The words every surface uses for a goal check the current grammar refuses, frozen on the run that excluded it.
CHECK_REFUSED_UNDER_CURRENT_GRAMMAR = "refused under the current grammar"


class ControlEndState(EvalDocumentModel):
    """An end state a template's author states, to prove its goal checks can tell outcomes apart.

    Authoring data, and nothing else: it is read when the template is written, to evaluate the
    template's goal checks against, and by nothing that runs a cell — the candidate, the simulated
    user and the judge never see it. A control that reached the candidate would be a hint about the
    answer.

    It is the seed with what the candidate changed laid over it, plus the calls it made. The overlay
    is per state key: a key named here replaces the seed's value whole, and a key not named keeps
    the seed's, so a control states only what the behaviour changed.
    """

    describes: str = Field(
        min_length=1,
        description=(
            "What the candidate did to leave the world this way, in prose — the sentence a reviewer "
            "checks the rest against. Required: a bare state is a fixture, and a reviewer cannot tell "
            "whether it shows the behaviour the template means."
        ),
    )
    world: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "The world as the candidate left it, where it differs from the seed: carrier -> key -> "
            "value, keyed exactly as the seed is, so each key names the dimension a goal check reads as "
            "state.<dimension>. A key named replaces the seed's value whole; a key left out keeps the seed's."
        ),
    )
    calls: list[RecordedCall] = Field(
        default_factory=list,
        description=(
            "Every call the candidate made, in order — the whole ledger, not an overlay: an empty list "
            "states a candidate that made no calls."
        ),
    )
    variation: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The case parameters the checks are evaluated under, for a check that reads variation.*. "
            "The do-nothing control is evaluated under the same parameters, so only the behaviour differs. "
            "Each value is a string, as a case stores it: a control stating another type is refused, since "
            "a check proven under it would grade differently on every case."
        ),
    )
    fired: list[str] = Field(
        default_factory=list,
        description=(
            'The triggered dimensions the candidate made fire, by name — what a check reads with fired("<dimension>"). '
            "Laid over the do-nothing control's firings, which are what fires with no candidate action at all: "
            "every clock-driven (turn-triggered) dimension, since the world's own clock moves whatever the "
            "candidate does. Each name must be a triggered dimension the host declares."
        ),
    )
    fired_armed: list[str] = Field(
        default_factory=list,
        description=(
            "The triggered dimensions on which the event the template's seed armed fired, by name — what a check "
            'reads with fired_armed("<dimension>"). Each is also a firing, so it need not be repeated in `fired`. '
            "Laid over the do-nothing control's, which are the clock-driven dimensions the seed arms. Each name "
            "must be a triggered dimension, and the template's seed must arm at least one event: a firing is armed "
            "when its event is one the seed armed, on the dimension it was armed on or another that event moves "
            "(WorldSession.observe), so what no run could leave is an armed firing under a seed that armed none."
        ),
    )


class GoalCheckControl(EvalDocumentModel):
    """One goal check's intent, and the end state that proves it discriminates."""

    check: str = Field(min_length=1, description="The goal check, verbatim as it appears in goal_state_checks.")
    intent: GoalCheckIntent = Field(
        description=(
            "'act' when the check grades something the candidate must do (doing nothing fails it); "
            "'hold' when it grades something the candidate must not do (doing nothing passes it)."
        ),
    )
    control: str = Field(
        min_length=1,
        description=(
            "The name of the end state, in end_states, that this check must give the other verdict on: "
            "for an 'act' check, one where the behaviour happened (it must pass); for a 'hold' check, "
            "one where the forbidden thing happened (it must fail)."
        ),
    )


class GoalCheckControls(EvalDocumentModel):
    """Proof, at authoring, that each of a template's goal checks can tell its outcomes apart.

    Each check is evaluated twice: against the untouched seed with no calls — the candidate did
    nothing — and against the end state its entry names. Its intent says which verdict each must
    get, and a check that gives the same verdict on both is refused, because it grades something
    that does not depend on what the candidate did. The do-nothing control needs no data: it is
    derived from the template's own seed.

    Validated where a template is written (``threetears.evals.run.check_controls``), and evaluated again
    at launch, whose verdict per check the run freezes (``EvalRun.goal_check_proofs``); the run summary, the
    analysis bundle and the report mark every check not proven. Read by nothing that runs a cell.
    """

    checks: list[GoalCheckControl] = Field(
        min_length=1,
        description="One entry per goal check, naming its intent and its control end state.",
    )
    end_states: dict[str, ControlEndState] = Field(
        min_length=1,
        description="The control end states the checks name, by name. One end state may serve several checks.",
    )

    @model_validator(mode="after")
    def _references_resolve(self) -> GoalCheckControls:
        """Refuse a block whose own references do not close: a duplicate check, a missing or an unused end state.

        Returns:
            The block, unchanged.

        Raises:
            ValueError: A check is listed twice, names an end state the block does not hold, or an
                end state is named by no check — data nothing would evaluate, which reads as proof
                and is not.
        """
        seen: set[str] = set()
        for entry in self.checks:
            if entry.check in seen:
                raise ValueError(f"goal check {entry.check!r} is listed twice in goal_check_controls")
            seen.add(entry.check)
            if entry.control not in self.end_states:
                raise ValueError(
                    f"goal check {entry.check!r} names control {entry.control!r}, which is not in end_states "
                    f"(have: {sorted(self.end_states)})"
                )
        unused = sorted(set(self.end_states) - {entry.control for entry in self.checks})
        if unused:
            raise ValueError(f"end states named by no check: {unused} — remove them or name them from a check")
        return self


#: How a criterion is answered: an integer from 1 to 5, or pass/fail. Every criterion states its
#: scale and every score the scale it was judged on — none is assumed. New criteria are
#: pass/fail; an existing 1–5 criterion keeps its scale until what it measures changes.
RubricScale = Literal["ordinal", "pass_fail"]

#: The two rubric axes a dimension sits on: what a subject should DO (``capability``) and what it
#: should refuse or withstand (``boundary``). One alias, because a dimension's stored axis and the
#: axis a draft was proposed on are the same vocabulary — the proposer stamps the axis it ran on into
#: the very field this validates.
#:
#: **A boundary dimension is a guardrail, and the two axes are never added together.** The composite
#: and pass^k are read over capability dimensions alone, a boundary dimension joins no comparison
#: family, and the analysis bundle decides each one on its own against the control (``guardrails``):
#: averaged in, a capability gain could pay for a guardrail loss and the sum would read as progress.
RubricAxis = Literal["capability", "boundary"]

#: The stored score a pass/fail answer becomes. 1 and 0, so a dimension's mean is its pass rate.
PASS_FAIL_SCORES: dict[str, int] = {"pass": 1, "fail": 0}


class ScaleSpec(NamedTuple):
    """What one rubric scale means, stated once.

    Every reading of a score's arithmetic — its bounds, its 0-1 value, its bar, how it reads and how a
    judge's answer becomes it — goes through this, so a third scale is one entry here plus a KeyError
    at each keyed table that has not learned it, never a silent fall into another scale's arithmetic.
    Rules defined on one scale's values rather than on scales in general (the reporter's 1-5 label
    bands) name that scale and refuse the rest themselves.
    """

    levels: tuple[str, ...]
    """The scoring-guide keys the scale admits, in the order a judge reads them."""
    scores: tuple[int, int]
    """The stored score's inclusive bounds."""
    labels: Mapping[int, str]
    """How a stored score reads, where it is not its own number (pass/fail); empty means the number."""
    fixed_bar: int | None
    """The score that clears the bar on this scale whatever the run's threshold (a pass), or None where
    the bar is the run's threshold (a 1-5 level)."""
    reads_as: str
    """How a descriptor states the scale, completing "Judged rubric dimension X, …"."""

    @property
    def value_range(self) -> tuple[float, float]:
        """The range a mean of stored scores lives in."""
        return (float(self.scores[0]), float(self.scores[1]))

    def normalized(self, score: int) -> float:
        """The score on 0-1."""
        low, high = self.scores
        return (score - low) / (high - low)


#: The scales, by name. `RubricScale`'s members and this table's keys are asserted equal by the tests.
#: Read-only, because it is public (a host renders a dimension's scale from it) and one process-wide
#: table a host could write into would change every other host's arithmetic in that process.
SCALES: Mapping[str, ScaleSpec] = MappingProxyType(
    {
        "ordinal": ScaleSpec(
            levels=("1", "2", "3", "4", "5"),
            scores=(1, 5),
            labels=MappingProxyType({}),
            fixed_bar=None,
            reads_as="scored 1-5 against the template's scoring guide",
        ),
        "pass_fail": ScaleSpec(
            levels=("pass", "fail"),
            scores=(0, 1),
            labels=MappingProxyType({v: k for k, v in PASS_FAIL_SCORES.items()}),
            fixed_bar=PASS_FAIL_SCORES["pass"],
            reads_as="answered pass (1) or fail (0); its mean is the pass rate",
        ),
    }
)

#: The scoring-guide keys each scale admits (a projection of :data:`SCALES`, kept for its readers).
SCALE_LEVELS: dict[str, tuple[str, ...]] = {name: spec.levels for name, spec in SCALES.items()}


#: Reserved ``rubric_dim_id`` for the dual-score transcript axis. The
#: ``__`` prefix avoids collision with a template dim literally named
#: ``transcript``. Used as :attr:`RubricScore.dim` on
#: :attr:`EvalResult.transcript_score` and as the ``rubric_dim_id`` a
#: transcript-axis :class:`JudgeConfig` binds to.
TRANSCRIPT_DIM_ID = "__transcript__"

#: Reserved ``rubric_dim_id`` for the dual-score outcome axis.
OUTCOME_DIM_ID = "__outcome__"

#: The reserved dim ids, which are deliberately NOT namespaced: they identify
#: the two dual-score axes rather than a rubric dimension scored in some context,
#: so there is no context to name. :func:`require_namespaced_dim_name` exempts
#: them — a judge service is built for these ids on every run.
RESERVED_DIM_IDS: frozenset[str] = frozenset({TRANSCRIPT_DIM_ID, OUTCOME_DIM_ID})

#: Separator between a dim's scoring context and its bare name.
DIM_NAMESPACE_SEPARATOR = "."


def require_namespaced_dim_name(name: str) -> None:
    """Reject a rubric-dim name that does not carry its scoring context.

    A judge config binds to a dim by *name*, globally:
    ``threetears.evals.run.launch.build_judge_service`` resolves one config per ``dim.name``
    across every template, so two templates that both name a dim ``character``
    share one config — and a support template would be scored by a prompt that
    opens "in a TRIAGE context". Namespacing the name is what makes the binding
    specific. The guard is on the model (:data:`DimName`), so a bare name is unrepresentable:
    a template rubric, a catalog dim, a judge config's binding and a proposer's draft alike
    refuse one wherever it is constructed or read.

    The required form is ``<context>.<dim>`` — exactly one separator, both sides
    non-empty. The context is the context a dimension is scored *against*, not
    the tool that produced the output: a tool's name says what made the output and
    nothing about what it was judged for, so a tool-shaped axis would be incoherent to
    persist.

    Args:
        name: The dim name or ``rubric_dim_id`` being checked.

    Raises:
        ValueError: ``name`` is not namespaced. Reserved dual-score axis ids
            (:data:`RESERVED_DIM_IDS`) are exempt and raise nothing.
    """
    if name in RESERVED_DIM_IDS:
        return
    context, separator, bare = name.partition(DIM_NAMESPACE_SEPARATOR)
    if not separator or not context or not bare or DIM_NAMESPACE_SEPARATOR in bare:
        # Build the worked example from the LAST segment, never by prefixing onto
        # the whole input: `.character` would otherwise suggest `triage..character`
        # and `triage.sub.character` would suggest `triage.triage.sub.character`
        # — an example the guard emitting it would itself reject. `example` is
        # asserted valid by a test that feeds every rejected shape back through here.
        suggestion = name.rsplit(DIM_NAMESPACE_SEPARATOR, 1)[-1].strip() or "character"
        example = f"triage{DIM_NAMESPACE_SEPARATOR}{suggestion}"
        raise ValueError(
            f"rubric dim name {name!r} must be namespaced by its scoring context as "
            f"'<context>{DIM_NAMESPACE_SEPARATOR}<dim>' — exactly one {DIM_NAMESPACE_SEPARATOR!r}, "
            f"with both sides non-empty (for example {example!r}). A judge config binds to a dim by "
            f"name across every template, so a name that does not carry its context would bind every "
            f"context at once"
        )


def _namespaced_dim_name(name: str) -> str:
    """Pass a namespaced dim name (or a reserved axis id) through; refuse any other.

    Args:
        name: The name being validated.

    Returns:
        ``name``, unchanged.

    Raises:
        ValueError: ``name`` is not namespaced — see :func:`require_namespaced_dim_name`.
    """
    require_namespaced_dim_name(name)
    return name


#: A rubric dimension's name as every model holds it: ``<context>.<dim>``, or a reserved
#: dual-score axis id. A bare name is unrepresentable, so no stored rubric, judge config or
#: draft can carry one into the global name binding.
DimName = Annotated[str, AfterValidator(_namespaced_dim_name)]


class RubricDim(EvalDocumentModel):
    """One judge-scored rubric dimension.

    Reserved for genuinely subjective qualities (tone, style) —
    goal-state checks handle the objective portion of scoring. ``scoring_guide`` maps the
    scale's levels (:data:`SCALE_LEVELS`) to behavioral descriptors; empty means the judge
    prompt relies on ``description`` only.
    """

    name: DimName
    description: str = Field(min_length=1)
    scale: RubricScale = Field(description="Integer 1–5 ('ordinal'), or 'pass_fail'. Stated, never assumed.")
    scoring_guide: dict[str, str] = Field(default_factory=dict)
    axis: RubricAxis = Field(
        default="capability",
        description=(
            "'capability' = something the subject should do well, read into the composite and pass^k; "
            "'boundary' = something it must not do (leak, comply with an unsafe ask, break policy), a guardrail "
            "that is decided on its own and never averaged with capability. The judge stamps it onto each score."
        ),
    )

    @model_validator(mode="after")
    def _guide_matches_the_scale(self) -> RubricDim:
        """Refuse a guide describing levels the scale does not have.

        A pass/fail criterion carrying a "3" descriptor, or a 1–5 one carrying "pass", would put
        a level in front of the judge that its answer is then refused for giving.
        """
        if stray := sorted(set(self.scoring_guide) - set(SCALE_LEVELS[self.scale])):
            raise ValueError(f"rubric dim {self.name!r} is {self.scale} but its scoring guide has levels {stray}")
        return self


class GoalStateOutcome(EvalDocumentModel):
    """One judge-free fact a candidate's execution established, and whether it held.

    Named for the check it was built for — a goal-state expression evaluated against the
    final world state — and still that for a conversational turn. Since the
    candidate-kind seam it is also the *general* mechanical tier every kind reports
    (``CandidateOutput.mechanical_facts``), so a classifier's ``label == expected`` and an
    artifact generator's schema check land here too, and neither has a world. What the
    three share is the shape the tier needs: a stated claim, a verdict, and enough prose to
    tell a reader which one failed without re-running anything.
    """

    expression: str = Field(min_length=1)
    passed: bool
    detail: str = Field(default="", description="Brief explanation; e.g. 'queue.length=2 satisfies >= 1'")


class PreconditionOutcome(EvalDocumentModel):
    """Outcome of one precondition asserted against the world at t=0.

    The reader :class:`Precondition`'s required ``presumes`` prose exists for. A precondition
    that did not hold makes an observation invalid rather than low-scoring, and the person
    reading that exclusion needs to know whether the template or the world was wrong —
    *"expression 2 was false"* tells them neither, so the prose is carried onto the record
    beside the expression rather than left in the template for them to go and find.

    **A sibling of :class:`GoalStateOutcome`, not a reuse of it.** The two look alike and mean
    opposite things: a failed goal check is the measurement, read beside a rubric that already
    says what was being scored; a failed precondition is the apparatus refusing to run, and the
    cell it belongs to has no scores at all.
    """

    expression: str = Field(min_length=1, description="The DSL expression, as the template wrote it.")
    presumes: str = Field(
        min_length=1,
        description="What the template said this precondition presumes — the sentence the exclusion is read with.",
    )
    held: bool = Field(description="Whether the world satisfied it at t=0.")
    detail: str = Field(default="", description="What the world actually held; e.g. 'queue.length=0 fails >= 3'.")


#: The temperature every judge call is requested at unless a :class:`JudgeConfig` for its dimension says
#: otherwise, and that config's own default (#633). A judge sampled at a provider's default (around 1.0 on
#: some) and one at 0 are two judges: before this, a dimension with a config was judged at its 0.0 and one
#: without at the provider default, in one run, because nobody chose otherwise.
DEFAULT_JUDGE_TEMPERATURE: float = 0.0

#: A judge call SENT with no temperature, because its model refuses one (some reasoning models do): the
#: model's own default applied. Recorded as this word rather than as a number nobody sent.
MODEL_DEFAULT_TEMPERATURE: Literal["model_default"] = "model_default"

#: The temperature a judge call was actually sent at: a number, or :data:`MODEL_DEFAULT_TEMPERATURE`.
JudgeTemperature = float | Literal["model_default"]


class RubricScore(EvalDocumentModel):
    """Outcome of one rubric judge dimension.

    ``dim`` carries the dimension's identifier — a template ``RubricDim`` name
    for per-template dims, or a reserved axis id (:data:`TRANSCRIPT_DIM_ID` /
    :data:`OUTCOME_DIM_ID`) for the dual-score axes. The field stays
    named ``dim`` (not ``dim_id``) until the shared rubric-dim catalog
    gives dims stable ids worth renaming an embedded field for.
    """

    dim: DimName = Field(min_length=1)
    scale: RubricScale = Field(description="The dimension's scale when it was judged.")
    axis: RubricAxis | None = Field(
        default=None,
        description=(
            "The dimension's axis when it was judged, stamped by the judge from the dimension's definition. "
            "'boundary' scores are guardrail readings and enter neither the composite nor pass^k. None = judged "
            "before the axis was stamped, so which axis it served is not recorded; it is read as capability, "
            "which is how every score was read then, and re-judging stamps it."
        ),
    )
    score: int = Field(description="1–5 on the ordinal scale; 1 (pass) or 0 (fail) on pass/fail.")
    reasoning: ModelProse = Field(default="")
    served_model: str | None = Field(
        default=None,
        description=(
            "The model that produced this score, as the provider's response named it — the concrete "
            "id a floating alias resolved to, never the id that was requested. It is the judge's "
            "identity for every apparatus comparison (the ``judge_model`` sweepable reads it), because "
            "the run's ``judge_model`` pin records what was ASKED for and a provider-side alias such as "
            "``~vendor/model-latest`` names a different model from one month to the next. None = the "
            "response named no model, so nobody observed which model scored, and comparisons read it "
            "as unknown, never as a match."
        ),
    )
    judge_temperature: JudgeTemperature | None = Field(
        default=None,
        description=(
            "The sampling temperature the call that produced this score was actually SENT at, as the completion "
            "reported it: a number, or 'model_default' when the model refuses a temperature and was sent none. "
            "Part of the judge's identity beside served_model: a different temperature is a different judge, and "
            "never pools with this one. None = not recorded (a client that reports no temperature, or a score "
            "judged before temperatures were recorded, when a dimension without a JudgeConfig was requested at the "
            "provider's default); compared as unknown, never as a match."
        ),
    )

    @model_validator(mode="after")
    def _score_is_on_the_scale(self) -> RubricScore:
        """Refuse a score its own scale cannot hold."""
        low, high = SCALES[self.scale].scores
        if isinstance(self.score, bool) or not low <= self.score <= high:
            raise ValueError(f"score {self.score!r} is not on the {self.scale} scale of {self.dim!r}")
        return self

    @property
    def normalized(self) -> float:
        """The score on 0–1: ``(score - 1) / 4`` for 1–5, the score itself for pass/fail."""
        return SCALES[self.scale].normalized(self.score)

    @property
    def label(self) -> str:
        """The score as a reader should see it: ``pass``/``fail``, or the 1–5 integer."""
        return SCALES[self.scale].labels.get(self.score, str(self.score))

    def clears(self, rubric_threshold: int) -> bool:
        """Whether this score meets the bar: at or above the threshold on 1–5, a pass on pass/fail.

        The threshold is an ordinal level, so it has nothing to say about a pass/fail criterion,
        whose bar is the pass itself.
        """
        fixed = SCALES[self.scale].fixed_bar
        return self.score >= (rubric_threshold if fixed is None else fixed)


# =============================================================================
# Rubric proposal — the validated DRAFT the rubric proposer returns
# =============================================================================
#
# These are NOT managed/persisted documents. The proposer renders the live
# subject's self-description + tool cognitive hints (Feed 1) plus the reusable
# catalog dims (Feed 2) into one LLM call; the model's JSON is validated INTO
# this shape and returned for operator review. Validation failures must
# surface (the operator needs to know the LLM produced garbage), so these use
# ``EvalBaseModel`` (``extra="forbid"``): an unexpected or missing field is an error, not a
# silently-dropped one.


class ProposedDimSuggestion(EvalBaseModel):
    """A novel rubric dim the proposer invented (not a catalog reuse).

    Shaped like the operator-authored portion of
    :class:`CatalogRubricDim` minus every server-owned field (``id`` /
    ``doc_type`` / ``schema_version`` / ``scope_id`` / ``archived`` /
    timestamps): the operator copies an accepted suggestion into
    ``eval(action='rubric_dim_create')``, which mints those fields itself.
    """

    key: str = Field(min_length=1, description="Stable version-group slug the operator would persist this dim under.")
    dim: RubricDim = Field(description="The judge-readable definition (name / description / scoring_guide).")
    axis: RubricAxis = Field(
        default="capability",
        description="Which rubric axis this dim belongs to ('boundary' is the boundary proposer's).",
    )
    universal: bool = Field(default=False, description="True → the operator would mark it applicable to every subject.")


class ProposedTemplate(EvalBaseModel):
    """The drafted template fields for one subject (capability or boundary).

    A subset of :class:`EvalTemplate` — the fields the proposer drafts. The
    capability proposer leaves ``conversation`` unset and
    ``universal`` False; the boundary proposer drafts a
    ``conversation`` (the adversarial askers) and sets ``universal=True`` so
    the saved template joins the battery. The operator copies an accepted draft
    into ``eval(action='create_template')``, which assigns the rest (id,
    lifecycle, world seed, goal-state checks).
    """

    name: str = Field(min_length=1, description="Proposed template name.")
    intent: str = Field(min_length=1, description="What this template tests.")
    rubric: list[RubricDim] = Field(
        default_factory=list,
        description=(
            "The dims the judge would score (reused catalog dims + novel ones), each on the axis the proposer "
            "ran on: capability dims from the capability proposer, boundary dims from the boundary proposer."
        ),
    )
    variation_axes: list[VariationAxis] = Field(
        default_factory=list,
        description="Scenario variation axes derived from the subject's self-description + tool hints.",
    )
    conversation: ConversationSpec | None = Field(
        default=None,
        description=(
            "The adversarial simulated actors the boundary proposer drafts: an out-of-domain "
            "ask plus injection / jailbreak / unsafe-request / character-break pressure. Null for the "
            "capability proposer, whose drafts simulate nobody."
        ),
    )
    universal: bool = Field(
        default=False,
        description=(
            "True → the boundary proposer drafts this as a universal battery template applicable to "
            "every subject. False for capability drafts."
        ),
    )


class RubricProposal(EvalBaseModel):
    """The validated DRAFT the rubric proposer returns for operator review.

    Not persisted: the operator edits the draft, then calls
    ``create_template`` / ``rubric_dim_create`` to commit the parts they
    accept. ``reused_dim_keys`` records which Feed-2 catalog dims the proposer
    chose to reuse (so the operator can see reuse vs. invention at a glance);
    ``new_dim_suggestions`` carries the novel dims worth promoting into the
    shared catalog.
    """

    template: ProposedTemplate = Field(description="The drafted capability rubric + scenario axes.")
    reused_dim_keys: list[str] = Field(
        default_factory=list,
        description="Catalog dim keys (Feed 2) the proposer reused verbatim rather than inventing near-duplicates.",
    )
    new_dim_suggestions: list[ProposedDimSuggestion] = Field(
        default_factory=list,
        description="Novel rubric dims the proposer invented, shaped for promotion into the shared catalog.",
    )


def _calibration_rating_id(result_id: str, rubric_dim: str, rater: str, rater_kind: str) -> str:
    """The id of one rater's rating of one dimension of one result, as a person or as an agent.

    Derived rather than minted, because "this rater's score for this dimension of this result" is
    one fact: a rater who rates the same dimension again is correcting themselves, and the write
    replaces the earlier rating instead of standing beside it as a second, disagreeing human. Two
    raters of one dimension are two ratings, which is what agreement pools.

    The rater's kind is part of the fact. A host may let an agent act under a person's identity
    (an MCP tool calling as the account it serves), so one ``rater`` can write a person's rating and
    an agent's of the same dimension; they are different facts — only the person's calibrates the
    judge — and an id without the kind would let either silently overwrite the other.

    Args:
        result_id: The rated result.
        rubric_dim: The rated dimension.
        rater: Who rated it.
        rater_kind: Whether a ``person`` or an ``agent`` wrote it.

    Returns:
        ``rating:`` and the digest of the four, so it can never collide with the result's own id or
        its trace sibling's in a shared partition.
    """
    return "rating:" + canonical_digest([result_id, rubric_dim, rater, rater_kind])


def _derived_rating_id(data: dict[str, Any]) -> str:
    """The default ``id`` of a rating, from the fields validated before it.

    Blank when one of them failed validation, which is then the error the construction reports.
    """
    try:
        return _calibration_rating_id(data["result_id"], data["rubric_dim"], data["rater"], data["rater_kind"])
    except KeyError:
        return ""


#: Who wrote a calibration rating: a ``person``, whose rating is the human side of judge calibration, or an
#: ``agent`` (a model acting through a tool), whose rating is not.
RaterKind = Literal["person", "agent"]


class CalibrationRating(EvalDocumentModel):
    """A rater's score for one judged dimension of one result — a person's, or an agent's.

    A person's is the human side of judge calibration. Only a person's rating is agreement with people; an agent's (``rater_kind="agent"``) is stored and listed,
    never paired with the judge (:func:`threetears.evals.analysis.judge_agreement`).

    A standalone document, never embedded on the result: a rating is written after the run, by
    someone who is not the run, and a result is a measurement the engine does not rewrite to add an
    opinion about it. The judge's side is never copied here. Agreement pairs this score with the
    score the result carries for the same dimension when it is read, so a re-judged result is
    compared with the judge that now scores it, and a rating cannot drift from the judge it claims
    to be read against.

    Written through :func:`threetears.evals.run.rate_result`, which reads the result first: the
    run, the scale and the existence of a judge score to calibrate are taken from it, never from
    the caller. Read by the bundle (``AnalysisContextBundle.judge_agreement``) and by a reporter
    run's calibration read, both through :func:`threetears.evals.analysis.judge_agreement`.

    One per ``(result, dimension, rater, rater_kind)``: the ``id`` is derived from the four, and a stored
    id that disagrees with its own fields is refused.
    """

    doc_type: Literal["calibration_rating"] = "calibration_rating"
    schema_version: SchemaVersion = EVAL_SCHEMA_VERSION
    scope_id: str = Field(min_length=1, description="The scope of the rated result.")
    run_id: str = Field(min_length=1, description="The run the rated result belongs to, read off the result.")
    result_id: str = Field(min_length=1, description="The rated result.")
    rubric_dim: DimName = Field(description="The judged dimension rated, spelled as the result's score spells it.")
    rater: str = Field(
        min_length=1,
        description=(
            "Who rated, as the host names them: a person's account or seat, or an agent's identity. Agreement pools "
            "a person's ratings with other people's and lists them; one rater's second rating of the same dimension "
            "of the same result, as the same kind of rater, replaces the first."
        ),
    )
    rater_kind: RaterKind = Field(
        description=(
            "Whether a person or an agent wrote the rating. Only a person's rating calibrates the judge against "
            "people: an agent's is another model's opinion, read apart and never pooled into judge-versus-human "
            "agreement."
        ),
    )
    id: str = Field(
        default_factory=lambda data: _derived_rating_id(data),
        description=(
            "Derived from (result_id, rubric_dim, rater, rater_kind): one rating per rater, per kind of rater, per "
            "dimension per result."
        ),
    )
    scale: RubricScale = Field(description="The dimension's scale on the rated result, read off the judge's score.")
    score: int = Field(
        strict=True, description="1-5 on the ordinal scale; 1 (pass) or 0 (fail) on pass/fail. A bool is not a score."
    )
    reason: str = Field(min_length=1, description="The rater's own words for the score — the evidence for it.")
    rated_at: str = Field(default_factory=utc_now_iso)

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "calibration_rating":
            raise ValueError(f"doc_type must be 'calibration_rating', got '{v}'")
        return v

    @model_validator(mode="after")
    def _score_on_scale_and_id_derived(self) -> CalibrationRating:
        """Refuse a score its scale cannot hold, and an id that is not the one its fields derive."""
        low, high = SCALES[self.scale].scores
        if not low <= self.score <= high:
            raise ValueError(f"score {self.score!r} is not on the {self.scale} scale of {self.rubric_dim!r}")
        derived = _calibration_rating_id(self.result_id, self.rubric_dim, self.rater, self.rater_kind)
        if self.id != derived:
            raise ValueError(
                f"id {self.id!r} is not the one (result_id, rubric_dim, rater, rater_kind) derive ({derived!r}): a rating "
                "under another id would stand beside the rater's own rating of the same thing instead of replacing it"
            )
        return self


# =============================================================================
# EvalTemplate
# =============================================================================


class EvalTemplate(EvalDocumentModel):
    """Abstract scenario blueprint — subject-agnostic, domain-level.

    Templates declare WHAT to test (``intent``, ``tools_required``) and HOW
    to score it (``goal_state_checks``, ``rubric``).  Concrete inputs are
    generated as ``EvalTestCase`` documents via the template's
    ``variation_axes``. What only the template's kind reads — a router's label
    set, a game master's table — is ``kind_spec``, which the kind's own model
    validates.

    Applicability is scoped by ``tools_required``, or — for the operator-curated
    boundary battery — by ``universal=True``, which marks the template as
    applying to every subject; ``start_universal_battery`` launches one run per
    active universal template.

    ``tools_allowed`` is the one tool field that is NOT about applicability: it
    bounds what the candidate may call, and so what the run can spend on third
    parties. See the field for the full contrast with ``tools_required``.

    Lives in one ``scope_id``, which the author names and storage never stamps.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    doc_type: Literal["eval_template"] = "eval_template"
    schema_version: SchemaVersion = EVAL_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    # Identity
    name: str = Field(min_length=1)
    description: str = Field(default="")
    intent: str = Field(min_length=1)

    # Which kind of candidate this template evaluates — the seam the runner dispatches on
    # (:mod:`threetears.evals.contracts.candidate_kind`). A plain string rather than an enum or a Literal, deliberately:
    # the set of kinds a host wires is the HOST's, so pinning it in the stored model would make
    # adding one a schema change. Required and un-defaulted for the same reason: any default
    # would be one host's kind name that every other host inherited without choosing it.
    candidate_kind: str = Field(
        min_length=1,
        description=(
            "Which candidate kind the runner builds for this template, by the name the host registers "
            "it under. The runner dispatches on it once, at the top of a cell, and everything below "
            "that sees only the candidate's output and its mechanical facts. Required: no kind is "
            "assumed."
        ),
    )

    # Applicability
    tools_required: list[str] = Field(default_factory=list)
    universal: bool = Field(
        default=False,
        description=(
            "When True the template applies to every subject (the boundary battery), and "
            "start_universal_battery includes it."
        ),
    )

    # Spend bound — NOT applicability. ``tools_required`` above answers "which
    # subjects does this template apply to" and gates nothing at run time; this
    # answers "which of that subject's tools may the candidate call".
    # ``max_cost_usd`` caps the run's LLM cost and explicitly does NOT cover
    # third-party provider quota (search credits, video-API requests), so before this
    # field a template written to be tool-free still registered every one of the
    # subject's tools and spent on both.
    #
    # It bounds WHICH tools, not HOW MANY calls, and it was for a while the only
    # third-party bound there was. ``EvalRun.max_metered_calls`` is the other
    # half: this field decides what a candidate may reach for, that one decides
    # how often. Neither substitutes for the other — an unbounded tool set with a
    # tight ceiling and a single search tool with none are different runs.
    tools_allowed: list[str] | None = Field(
        default=None,
        description=(
            "Upper bound on the subject's tools registered onto the eval candidate — a SPEND "
            "bound, not an applicability filter (that is tools_required, which selects which "
            "subjects the template applies to and restricts nothing at run time). None (the "
            "default) registers every tool the subject carries; a list registers only the named "
            "ones; [] registers none at all, which is how a template is authored to spend no "
            "third-party quota. [] and None are different values and are never collapsed. "
            "Names are checked against the tool catalog when the template is authored "
            "(create_template / update_template) and against the candidate subject at run "
            "launch (start_run) — the split exists because a template is authored with no "
            "subject in hand, so 'this subject has no such tool' is unknowable until launch."
        ),
    )

    # Variation generation
    variation_axes: list[VariationAxis] = Field(default_factory=list)

    # The simulated side, for a kind whose candidate converses. None for every other kind.
    conversation: ConversationSpec | None = Field(
        default=None,
        description=(
            "The simulated actors, their turn order and the turn budget, for a candidate that holds a "
            "conversation. Null for a template whose candidate does not converse — a document or a classifier."
        ),
    )

    # World seed — initial state
    world_seed: WorldSeed = Field(default_factory=WorldSeed)

    # What only this template's kind reads. Validated against the kind's spec model where the
    # template is authored and again where it launches, so the stored value is the model's
    # resolved form — every field, defaults included.
    kind_spec: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "What a template of this kind states beyond the engine's own fields — a router's label set, a "
            "game master's table, the facts only a judge may see — as the kind's spec model "
            "(``HostProfile.kinds``) validated it at authoring: every field, defaults included. The engine "
            "never reads inside it. A launch validates it again and freezes it onto each run, and the run's "
            "measurement context hashes that frozen copy. Empty for a kind that declares no spec."
        ),
    )

    # What this template presumes the world holds before the subject's first turn. Empty means
    # it presumes nothing — a weaker claim than it looks, since a scenario with an unstated
    # presumption still has one.
    preconditions: list[Precondition] = Field(default_factory=list)

    # Scoring
    goal_state_checks: list[str] = Field(default_factory=list)
    # Authoring-time proof that each goal check discriminates (``GoalCheckControls``). Authoring
    # requires it for every check a template declares; a template written past authoring (saved
    # straight to a store, or seeded before the seeder admitted templates through authoring) can lack it.
    # Its checks are then recorded unproven when it is launched (``EvalRun.goal_check_proofs``), and the run
    # summary, the analysis bundle and the report mark each one beside its pass rate, never taking it as proven.
    goal_check_controls: GoalCheckControls | None = Field(
        default=None,
        description=(
            "Each goal check's intent ('act' or 'hold') and an end state that proves the check tells "
            "outcomes apart: an act check must fail when the candidate did nothing and pass on its "
            "control; a hold check must pass when the candidate did nothing and fail on its control. "
            "Checked where the template is written; required for every goal check a create or an "
            "update authors. Never shown to the candidate, the simulated user or the judge. Null on a "
            "template written past authoring (saved straight to the store) or by the quick path — its checks are "
            "recorded unproven at launch (`EvalRun.goal_check_proofs`) and marked so wherever their pass rates show."
        ),
    )
    rubric: list[RubricDim] = Field(default_factory=list)

    # Lifecycle
    archived: bool = Field(default=False)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "eval_template":
            raise ValueError(f"doc_type must be 'eval_template', got '{v}'")
        return v

    @field_validator("variation_axes")
    @classmethod
    def _one_closed_stratum_axis(cls, axes: list[VariationAxis]) -> list[VariationAxis]:
        """Refuse more than one stratum axis, or a stratum axis whose values are not a closed set.

        Two nominated axes would give a generated case two strata, and the case holds one. An ``llm``
        axis writes a new value per case, so every case it wrote would be a stratum of one.

        Raises:
            ValueError: Two or more axes are nominated, or a nominated axis is ``llm``.
        """
        nominated = [axis for axis in axes if axis.stratum]
        if len(nominated) > 1:
            names = ", ".join(repr(axis.name) for axis in nominated)
            raise ValueError(f"at most one variation axis is a template's stratum; {names} are all nominated")
        if nominated and nominated[0].generator == "llm":
            raise ValueError(
                f"variation axis {nominated[0].name!r} is nominated as the stratum but is written by a model, which "
                "writes a new value for every case; nominate an `enum` or `sample` axis"
            )
        return axes

    def resolve_preconditions(self, world: WorldRegistry | None) -> list[Precondition]:
        """Refuse a presumption naming a dimension this host's world does not declare.

        Called where a stored template is USED — read by id, or launched — so a template
        presuming a dimension somebody removed says so on the surface an author is looking at
        rather than mid-run. Resolving it to nothing instead would turn a removed dimension into
        a template that quietly presumes less than it says, which is missing-value semantics one
        layer up and the defect this whole contract exists to catch.

        **Not on enumeration**, deliberately. A listing is a catalogue rather than a use, it is
        how an operator finds the offending template, and one of its callers seeds definitions at
        web startup — so a refusal reaching every enumeration would take the recovery down along
        with the problem, and the web process's boot with it.

        **Names are the compatibility surface**, so this is what a world-registry rename costs: a
        stored template referencing the old name stops loading, loudly, naming the path. That is
        the intended price rather than an accident of the implementation — the alternative is a
        corpus of scenarios silently presuming nothing. The recovery survives the refusal, which
        is the property that makes the price payable: an update reads the stored shape through
        storage rather than through the use path, so a refused template is still editable into
        one that resolves — and still listable, and still deletable.

        **Preconditions only, and ``goal_state_checks`` deliberately not.** A goal check reading
        an unregistered path is a real defect — it scores the subject down for a state nobody
        wrote — but it is caught statically, by the conformance kit's vocabulary check and by the
        authoring gate, where it can be reported without making a stored template unreadable. A
        host part-way through registering its carriers has goal checks over paths no dimension
        speaks for yet, and refusing to load them would refuse the corpus rather than the defect.

        Args:
            world: The host's world registry, or None when the host declares no world at
                all. None is not an empty registry: it says this host instantiates nothing, so
                "the registry no longer has it" is a question with no subject. Whether a template
                may presume a world on such a host is the authoring gate's to answer, through
                ``HostProfile.representable``, which calls it inapplicable rather than uncovered.

        Returns:
            The preconditions in declaration order — what a run asserts before the first turn.

        Raises:
            ValueError: A precondition reads a path no declared dimension covers. Its callers
                translate it — an untranslated ``ValueError`` reaches an operator as a 500 with
                no message, which is the opposite of failing loudly.
        """
        if world is None or not self.preconditions:
            return list(self.preconditions)
        unresolved = [
            f"{path!r} (presuming {precondition.presumes!r})"
            for precondition in self.preconditions
            for path in precondition.presumed_paths
            if world.resolve_path(path) is None
        ]
        if unresolved:
            raise ValueError(
                f"template {self.name!r} presumes world state this host does not declare: "
                + "; ".join(unresolved)
                + " — a dimension was removed or renamed, or the precondition names it wrongly"
            )
        return list(self.preconditions)


# =============================================================================
# JudgeConfig — versioned judge prompt + model + decoding params
# =============================================================================


class JudgeConfig(EvalDocumentModel):
    """Versioned judge configuration for one rubric dimension.

    A single-dimension judge's prompt, model, and decoding params, captured as
    a versioned record so operators can iterate on judge prompts safely: a new
    config supersedes the prior one for live runs, but the old config stays in
    storage so runs scored under it remain interpretable (and calibration can
    validate a new version before it takes over).

    Lives in one ``scope_id``, like :class:`EvalTemplate` (judge configs are
    subject-agnostic definitions, stable across runs). ``rubric_dim_id`` binds the config to a
    template :class:`RubricDim` **by name** (per-template dims have no stable id
    outside the shared catalog) or to a reserved dual-score axis id
    (:data:`TRANSCRIPT_DIM_ID` / :data:`OUTCOME_DIM_ID`).

    "Active" config for a dim = the non-archived record with the latest
    ``created_at`` (see :meth:`EvalStorage.load_active_judge_config`). When no
    config exists for a dim, the judge service falls back to its built-in
    default prompt.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    doc_type: Literal["judge_config"] = "judge_config"
    schema_version: SchemaVersion = EVAL_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    name: str = Field(min_length=1)
    description: str = Field(default="")

    rubric_dim_id: DimName = Field(
        description="Which dim this config scores — a template RubricDim name or a reserved axis id.",
    )
    prompt_template: str = Field(min_length=1, description="System-prompt body for the single-dim judge call.")
    model: str = Field(
        default="",
        description=(
            "Judge model alias/literal. Setting it is the most specific statement in the "
            "judge-model cascade and overrides the run-level pin for this dim alone. Blank "
            "means no opinion — the dim inherits the run's pin, NOT the EVAL_JUDGE role "
            "default, so configuring a dim's prompt cannot silently change which model scores it."
        ),
    )
    temperature: float = Field(
        default=DEFAULT_JUDGE_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description=(
            "The temperature this dim's judge calls are requested at — the same default a dim with no config is "
            "judged at, so configuring a dim's prompt never changes how it is sampled. A model that refuses a "
            "temperature is sent none; each score records what was actually sent (RubricScore.judge_temperature)."
        ),
    )

    archived: bool = Field(default=False)
    created_at: str = Field(default_factory=utc_now_iso)

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "judge_config":
            raise ValueError(f"doc_type must be 'judge_config', got '{v}'")
        return v


# =============================================================================
# CatalogRubricDim — shared, versioned, reusable rubric-dimension catalog
# =============================================================================


class CatalogRubricDim(EvalDocumentModel):
    """A reusable rubric dimension in the shared catalog.

    Distinct from the embedded :class:`RubricDim` value object that
    ``EvalTemplate.rubric`` carries: that one is per-template, name-keyed, and
    has no identity. This catalog record is a doc-enveloped, versioned, env-
    partitioned *source* of reusable dim definitions — the proposer
    draws from it, and calibration anchors pooled ratings to its stable
    ``id``.

    The judge-readable definition (``name`` / ``description`` /
    ``scoring_guide``) is composed verbatim as :attr:`dim` — the same
    :class:`RubricDim` shape the judge already reads — rather than duplicated,
    so the catalog and the embedded value object can never drift in field set.

    Lives in one ``scope_id``, like :class:`EvalTemplate` and :class:`JudgeConfig`. ``key`` is
    the stable version-group slug: re-authoring a dim writes a new record with
    a new ``id`` but the same ``key``, and "active" = the non-archived record
    with the latest ``created_at`` for that key (see
    :meth:`EvalStorage.load_active_rubric_dim`). Calibration follows ``id`` for
    a single version and ``key`` to pool history across versions.

    **DECISION (recorded — lock-in):** this catalog does NOT rebind
    the judge's name-keying to catalog ids. Templates still embed name-keyed
    :class:`RubricDim` value objects and the judge path is untouched; the
    catalog is purely a reusable-definition source plus a stable anchor for
    calibration.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    doc_type: Literal["rubric_dim"] = "rubric_dim"
    schema_version: SchemaVersion = EVAL_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    key: str = Field(min_length=1, description="Stable version-group slug; re-authoring keeps the key, mints a new id.")
    dim: RubricDim = Field(description="The judge-readable definition (name / description / scoring_guide).")

    axis: RubricAxis = Field(
        default="capability",
        description=(
            "Which rubric axis this dim belongs to ('boundary' is the boundary proposer's). Always equal to "
            "dim.axis, so copying dim into a template keeps a guardrail a guardrail: where the two disagree, "
            "both read 'boundary'."
        ),
    )
    universal: bool = Field(default=False, description="True → applies to every subject.")

    archived: bool = Field(default=False)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    @model_validator(mode="before")
    @classmethod
    def _one_axis_on_the_record_and_its_dim(cls, data: Any) -> Any:
        """Make the record's axis and the embedded dim's one axis, ``boundary`` when either says so.

        The embedded :attr:`dim` is what a template copies, and the judge stamps ITS axis onto each score.
        A record declaring ``axis="boundary"`` over a dim left at its ``capability`` default was copied into
        a template as capability, and its scores then entered the composite and pass^k: a guardrail leaking
        into the capability pillar. So the two are made one here, on every construction and every read of a
        stored record. A disagreement resolves to ``boundary`` rather than being refused: a stored record
        carries both fields (serialization emits defaults), so refusing would make every such record
        unreadable, and the direction that never lets a guardrail be averaged with capability is the safe one.

        Args:
            data: The raw input.

        Returns:
            The input with both axes set alike.
        """
        if not isinstance(data, dict) or "dim" not in data:
            return data
        dim = data["dim"]
        dim_axis = dim.get("axis") if isinstance(dim, dict) else getattr(dim, "axis", None)
        axes = {data.get("axis"), dim_axis} - {None}
        if len(axes) < 2 and data.get("axis") == dim_axis:
            return data
        axis = "boundary" if "boundary" in axes else (axes.pop() if axes else "capability")
        if isinstance(dim, RubricDim):
            dim = dim.model_copy(update={"axis": axis})
        elif isinstance(dim, dict):
            dim = {**dim, "axis": axis}
        return {**data, "axis": axis, "dim": dim}

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "rubric_dim":
            raise ValueError(f"doc_type must be 'rubric_dim', got '{v}'")
        return v


def stored_variation(params: Mapping[str, Any]) -> dict[str, str]:
    """Case parameters as a case stores them: each a string, a non-string one as its sorted-key JSON.

    ``EvalTestCase.variation_params`` is a flat string map, so whatever a parameter was, a goal check reads
    it as one string. Whatever else evaluates a check under parameters (a control end state's) goes through
    here, so it reads the types a real case holds and cannot pass on one no case could have.

    Args:
        params: The parameters, by name.

    Returns:
        The same names, each value a string: verbatim when it already is one.
    """
    return {
        name: value if isinstance(value, str) else json.dumps(value, sort_keys=True) for name, value in params.items()
    }


class RubricDimTombstone(EvalDocumentModel):
    """The record that a rubric dim key was deleted, so a seed never writes it back.

    Seeding fills empty slots only (:func:`~threetears.evals.run.definition_seed.seed_eval_definitions`),
    and a delete empties a slot. Without this, deleting a seeded dim undid itself at the next boot while
    archiving one was permanent — the opposite of what an operator choosing the destructive path meant.
    :func:`~threetears.evals.run.authoring.delete_rubric_dim` writes one for the key it deletes, and the
    seeder treats a tombstoned key as decided: it is not written, and is reported as deleted.

    It holds no prose: the dim's definition and scoring guide are what the delete destroys. Authoring a dim
    under the key again is unaffected (``create_rubric_dim`` does not read tombstones); the key is then
    occupied, which the seed respects in any case.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    doc_type: Literal["rubric_dim_tombstone"] = "rubric_dim_tombstone"
    schema_version: SchemaVersion = EVAL_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    key: str = Field(min_length=1, description="The deleted dim's version-group key — the seed's slot.")
    deleted_dim_id: str = Field(min_length=1, description="The id of the record whose delete wrote this.")
    deleted_at: str = Field(default_factory=utc_now_iso)

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "rubric_dim_tombstone":
            raise ValueError(f"doc_type must be 'rubric_dim_tombstone', got '{v}'")
        return v


class JudgeConfigTombstone(EvalDocumentModel):
    """The record that a judge config slot was deleted, so a seed never writes it back.

    :class:`RubricDimTombstone`'s mechanism for the seed's judge-config slot, ``(rubric_dim_id, name)``: a
    delete empties the slot when it takes the last record under it, and without this the next boot wrote the
    seeded config again while archiving one kept it retired.
    :func:`~threetears.evals.run.authoring.delete_judge_config` writes one for the slot it deletes from, and
    the seeder treats a tombstoned slot as decided: it is not written, and is reported as deleted. Authoring a
    config into the slot again is unaffected (``create_judge_config`` does not read tombstones).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    doc_type: Literal["judge_config_tombstone"] = "judge_config_tombstone"
    schema_version: SchemaVersion = EVAL_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    rubric_dim_id: str = Field(min_length=1, description="The dim the deleted config scored — half the seed's slot.")
    name: str = Field(min_length=1, description="The deleted config's name — the other half of the seed's slot.")
    deleted_config_id: str = Field(min_length=1, description="The id of the record whose delete wrote this.")
    deleted_at: str = Field(default_factory=utc_now_iso)

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "judge_config_tombstone":
            raise ValueError(f"doc_type must be 'judge_config_tombstone', got '{v}'")
        return v


# =============================================================================
# EvalTestCase — immutable concrete inputs
# =============================================================================


class EvalTestCase(EvalDocumentModel):
    """Concrete, immutable inputs generated from an ``EvalTemplate``.

    Once persisted, a test case's content never mutates.  ``variation_params`` freezes
    the template's axis values for this case.  Re-running a template against
    the same subject reuses existing test cases — durability is the rule that
    makes run-to-run comparison interpretable. ``archived`` is curation state
    beside that content, on the terms ``EvalRun.archived`` and
    ``EvalAnalysis.archived`` are: an operator's decision, not a fact about the
    inputs.

    Lives in the runtime tier partitioned by ``scope_id``
    (same as ``EvalRun`` / ``EvalResult``).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    doc_type: Literal["eval_test_case"] = "eval_test_case"
    schema_version: SchemaVersion = EVAL_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    # Lineage
    template_id: str | None = Field(
        min_length=1,
        description=(
            "The template this case was generated from, or None for a WITNESSED case — the stimulus of a session "
            "a host observed rather than one a template set, recorded through `record_witnessed_cell` under a "
            "witnessed run (whose own `template_id` is None for the same reason, unless the run is judged, when it "
            "names the template whose intent and rubric the judge reads). Required with no default, so "
            "every writer states which it is: a None is a case no template produced, never a template id "
            "forgotten. A launch refuses a case whose template is not the one it runs, so a witnessed case "
            "can never be launched."
        ),
    )

    # Variation — frozen at generation
    variation_params: dict[str, str] = Field(default_factory=dict)

    host_payload: VerbatimObject = Field(
        default_factory=dict,
        description=(
            "The case's own rich stimulus, in the HOST's vocabulary, and the engine must never read it. "
            "The stimulus half of the subject pair the candidate-kind seam rules for the SUBJECT, one "
            "level down: `variation_params` is the engine's view of what varies (flat strings it can group, "
            "hash and render), and a stimulus that is not a flat string map travels here instead — the "
            "classifier's N pending messages with the activity they arrive against, an artifact "
            "generator's input set. Opaque on exactly the terms `EvalRun.host_payload` is: nothing under "
            "the engine outside the host's own adapter may reach into it, and a gate says so. Empty for "
            "every case whose whole stimulus IS its variation parameters, which is every "
            "conversational-turn case, so no stored case changes meaning by acquiring the field."
        ),
    )

    stratum: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The kind of case this is — `plain`, `boundary`, `lookalike`, in the author's own words — which the "
            "analysis reads results by: every measure of a cell is summarised again over the cases of each "
            "stratum, beside the figure pooled over all of them. Metadata about the case, never stimulus: the "
            "engine puts it in no prompt — not the candidate's variation, the simulated user's or the judge's — "
            "and a kind, which is handed the whole case, must leave it out of what it renders, so naming a case a "
            "lookalike never tells the candidate what to look out for. Not part of `content_hash` or of any identity key, because "
            "the case's id already pins it (a stored case never changes) and two arms over the same cases share "
            "their strata. None when the case declares none; a run none of whose cases declares one reads "
            "exactly as an unstratified run. A generated case takes it from the template's nominated axis "
            "(`VariationAxis.stratum`)."
        ),
    )

    content_hash: str | None = Field(
        default=None,
        description=(
            "Digest of this case's generated content (variation_params), pinning the identity of an "
            "LLM-generated artifact so 'the same cell' cannot silently differ between runs. None when the "
            "case has no generated content to pin, and its identity is then its template plus its id. Stored rather than computed on read so it is groupable "
            "in a query."
        ),
    )

    archived: bool = Field(
        default=False,
        description=(
            "Curation state, orthogonal to the case's content: an archived (retired) case is not launched "
            "again, while it stays readable for every run already measured against it. Written only by "
            "`EvalService.set_reporter_case_archived`, which refuses a case carrying no reporter case — the "
            "reporter case bank is the one launch path that reads it, so setting it on any other kind's case "
            "would record a retirement no launch honours."
        ),
    )
    archived_reason: str | None = Field(
        default=None,
        description=(
            "Why the case was retired, in the operator's words — None while it is live, and None again the "
            "moment it is restored, since a reason outliving the retirement describes a state it is no longer in."
        ),
    )

    created_at: str = Field(default_factory=utc_now_iso)

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "eval_test_case":
            raise ValueError(f"doc_type must be 'eval_test_case', got '{v}'")
        return v


class EvalCaseStratum(EvalDocumentModel):
    """What the analysis reads off a test case: which case, and the stratum it declares.

    Read in place of the whole case wherever the stratum is all a reader needs — assembling a
    campaign's bundle reads one per case its results name, and a case document carries the host's
    whole stimulus. Built from a stored row reduced to these two fields, so it is on the document base
    and hydrated like every other stored eval model, as :class:`EvalRunStamp` is for a run.
    """

    id: str
    stratum: str | None = EvalTestCase.model_fields["stratum"].default


# =============================================================================
# Measurement-context identity
# =============================================================================


class ContextComponents(EvalDocumentModel):
    """The separately-recorded pieces a run's ``context_key`` is composed from.

    ``context_key`` alone answers "are these two runs comparable?"; it cannot
    answer "comparable except for WHAT?".  Recording each component beside the
    composite lets a comparison surface badge the single dimension that differs
    — the subject's carried state drifted, the frozen world changed, a different
    judge was pinned — instead of reporting an opaque key mismatch the operator
    has to bisect by hand.

    Hashed components are digests rather than raw values because their inputs
    (everything a subject accumulated, a whole frozen world) are large and are
    already persisted on the run. Two fields are not digests, for the same
    reason inverted: ``scope`` stays raw because it is small and a badge naming
    the scope beats one reading "scope hash differs", and ``subject_state``
    stays a MAP because collapsing the host's names into one hash would answer
    "the subject's state moved" where the map answers which of them did, in the
    host's own vocabulary.

    A component is ``None`` when it could not be computed rather than when it
    was empty: an empty carried state hashes to a real digest of "nothing
    carried", which is a meaningful and *stable* context. ``None`` appears only on a read-time
    derivation for a run whose inputs were never persisted, and such a
    derivation is flagged partial.
    """

    subject_state: dict[str, str] | None = Field(
        default=None,
        description=(
            "The host's carried-state map for the subject, name -> content hash — what the subject brought "
            "into the run, which no campaign chose. Kept as the map rather than digested into one string so a "
            "badge can name WHICH of the host's state names moved. None whenever the snapshot recorded no "
            "state map at all, on the same all-or-none rule roles follows; an empty MAP is a recording (this "
            "subject carries nothing outside its variant components) and hashes."
        ),
    )
    seeded_world: str | None = Field(
        default=None,
        description=(
            "Digest of the mapping of what this run froze before the subject's first turn — the world it "
            "seeded, and the spec its kind validated. Its own component rather than part of another: a run "
            "that froze nothing passes an "
            "empty value per carrier, so this is always composable, while the components it used to ride in "
            "were not."
        ),
    )
    case_basis: str | None = Field(
        default=None,
        description=(
            "Digest of the subject_id, the template_id, and the frozen, sorted test_case_ids — "
            "who was measured, and the denominator they scored against. The subject joined the basis because "
            "without it two different subjects carrying nothing produced an identical key and pooled as one "
            "condition."
        ),
    )
    roles: str | None = Field(
        default=None,
        description=(
            "Digest of the pinned non-candidate roles: the resolved judge and simulator models, the "
            "per-dim judge attribution, and the judge configuration set the run committed to. None "
            "whenever any of those was not recorded — the component is all of its inputs or none of "
            "them, so a run that could not say is left absent rather than hashed as a narrower "
            "component wearing the same name."
        ),
    )
    cassette: str | None = Field(
        default=None,
        description=(
            "Digest of cassette_mode plus the corpus a replay serves. Separates replayed tool output from live, "
            "and one replayed corpus from another, as distinct measurement contexts: two runs that differ here "
            "were not measured against the same world. Capture runs hash no corpus, since their tools ran live."
        ),
    )
    # ``k`` (repetitions per cell) stood here until IDENTITY_VERSION v7 and is deliberately
    # gone rather than kept-but-unhashed. It is not a condition — it is how many repetitions
    # were taken under one — so two runs at k=1 and k=3 were measured identically and differ
    # only in precision, and badging that as a context difference was a caveat about nothing.
    # A component this class does not compose has no business being on it: every field here
    # is read as "a dimension the key holds fixed". Depth is disclosed where it belongs, by
    # the surfaces that own it (``Iters/case``, the per-point case counts on the pass^k curve,
    # the completeness sentence).
    tool_permissions: str | None = Field(
        default=None,
        description=(
            "Digest of the tool allowlist frozen onto the run at launch. Its own component because the "
            "template is mutable, so template_id in case_basis cannot stand for it: two runs on one template "
            "spanning an edit to its allowlist would share a key while one candidate could reach for "
            "capabilities the other could not. Always composable — no allowlist means UNBOUNDED, which is a "
            "recorded level rather than an absence, so two unbounded runs hash equal."
        ),
    )
    world: str | None = Field(
        default=None,
        description=(
            "Digest of the run's derived world placements — which declared dimensions this run seeded, "
            "and which its subject merely witnessed. The world a subject is placed in is stimulus, so it "
            "belongs to the conditions a cell holds constant: two runs that seeded different parts of one "
            "world are not repetitions of one condition, and pooling them reports an experimental k over "
            "evidence that varied the stimulus. The PLACEMENTS rather than the values, because the values "
            "are already composed into seeded_world and hashing them twice would split nothing further. "
            "None whenever the run did not record placements, on the same all-or-none rule roles follows."
        ),
    )
    apparatus_settings: str | None = Field(
        default=None,
        description=(
            "Digest of the host-declared apparatus values the run's launch set (EvalRun.apparatus_settings) — how "
            "the measuring rig was set up, such as who sat in an adjudicator's seat. Its own component because one "
            "template is run at two such values to compare them, and those runs were measured on two rigs, not "
            "repeated on one. Always composable: no settings is a recorded level, so two runs that set none hash "
            "equal."
        ),
    )
    scope: str | None = Field(
        default=None,
        description="Storage scope the run executed in, kept raw so a mismatch badges as a value.",
    )


# =============================================================================
# EvalRun — execution record
# =============================================================================


#: ``budget_stopped`` is a budget the run was launched under binding — its cost cap or its wall-clock
#: budget, ``EvalRun.budget_stop_reason`` says which: a designed stop that keeps what the run delivered,
#: never ``failed``.
#:
#: ``exhausted`` is the account-side twin of ``budget_stopped``: the provider account paying for the
#: run refused a candidate call (out of credit, or the key refused), so every later cell would be
#: refused the same way. The run stops, keeps what it delivered, and names the account — not the
#: harness and not the candidate — as the reason it is short.
EvalRunStatus = Literal["pending", "running", "completed", "failed", "cancelled", "budget_stopped", "exhausted"]

#: Statuses a run can still leave under its own power — it is either queued or
#: executing, and something in-process is expected to write its terminal status.
#: Derived by subtraction in :data:`TERMINAL_RUN_STATUSES` so a new status added
#: to the Literal above lands on the terminal side by default: forgetting to
#: classify a new *terminal* status is harmless, while forgetting to classify a
#: new *non-terminal* one would make reclaim skip it silently.
NON_TERMINAL_RUN_STATUSES: frozenset[str] = frozenset({"pending", "running"})

#: Every status a run can carry, derived from the Literal above rather than
#: restated. Named because the read surfaces validate a caller's ``status``
#: filter against it (:func:`~threetears.evals.contracts.status_filter.validate_status_filter`,
#: and :func:`~threetears.evals.contracts.status_filter.normalize_status_filter` through it) and a
#: hand-copied tuple over there would drift the moment a status is added —
#: silently, and in the direction that refuses a real status.
RUN_STATUSES: frozenset[str] = frozenset(get_args(EvalRunStatus))

#: Statuses from which a run never moves again.
TERMINAL_RUN_STATUSES: frozenset[str] = RUN_STATUSES - NON_TERMINAL_RUN_STATUSES

#: Whether a run's apparatus was set before the fact or found after it.
#:
#: * ``commissioned`` — a rig was chosen and the observations were gathered under it: the launch
#:   path stamps this on every run it starts, because a launch is exactly the act of fixing a rig and
#:   measuring against it.
#: * ``witnessed`` — nothing set the apparatus; the host observed traffic it did not control (a real
#:   user's session, captured afterwards) and recorded the apparatus it found.
#:
#: The difference between an experiment and a log. Cells never pool across it
#: (:func:`~threetears.evals.analysis.cells.pool_observations`), and the same two words are what a
#: campaign declares it held (:attr:`~threetears.evals.contracts.declaration.ControlDeclaration.apparatus`),
#: so a declaration and the runs it governs are compared value for value rather than through a
#: translation table.
ApparatusProvenance = Literal["commissioned", "witnessed"]

#: How a run arrived at one of its resolved role models. ``chosen`` means the launch
#: named it and re-running with the same arguments pins the same model; ``inherited``
#: means the role default supplied it, so the same launch arguments would pick up
#: whatever that default has since become. Once the model is stored resolved, this is
#: the only thing that still distinguishes the two.
ModelRoleOrigin = Literal["chosen", "inherited"]

#: How a run arrived at one of its resolved role MODELS — :data:`ModelRoleOrigin`'s two values plus
#: ``alternate``: the launch named no judge, and the host's alternate judge
#: (:attr:`~threetears.evals.run.LaunchSettings.judge_alternate_model`) scored, because the judge role's
#: default was one of the launch's candidates (:func:`~threetears.evals.run.resolve_judge_pin`). Its own
#: literal because only a role model can be stepped off: a judge config's origin
#: (``judge_config_provenance``) is chosen or inherited and nothing else. Re-running the same launch
#: arguments picks the alternate setting as it then stands, and only while the role default is still a
#: candidate. Recorded from what the launch can see — no pin named, and the judge equal to the alternate
#: setting — so a host whose alternate is set to the role default itself records ``alternate`` for that
#: model too: the two settings named one model, and either reproduces it.
RoleModelOrigin = Literal["chosen", "inherited", "alternate"]

#: How a run arrived at the spend ceiling it ran under. Deliberately NOT
#: :data:`ModelRoleOrigin`: that literal's two values cannot express the third state this
#: cascade really has — a run can be bounded by nothing at all.
#:
#: * ``chosen`` — the launch passed ``max_cost_usd``, at or below the host's ceiling (a launch may
#:   only lower it, :func:`~threetears.evals.run.ceilings.refuse_raised_ceiling`), so re-launching
#:   with the same arguments runs under the same ceiling while the host's stays at or above it.
#: * ``inherited`` — the launch passed nothing and the host's configured default
#:   (``LaunchSettings.max_cost_usd``) supplied it, so the same launch arguments run under
#:   whatever that setting has since become. This is the one that moves under a run's feet, and
#:   it is the common case.
#: * ``uncapped`` — the host's ceiling enforcement is off (``LaunchSettings.enforcement_enabled``),
#:   so nothing bounded this run whatever the cascade would otherwise have resolved. Recorded
#:   rather than left blank because ``max_cost_usd`` is null both for an uncapped run and for a run
#:   whose writer recorded no ceiling, and those are not the same fact.
#:
#: ``None`` = the run's writer recorded no ceiling (a run assembled without the launch, which is
#: where the cascade resolves).
#:
#: **The bullets above are written in cost, and the type is not cost-only.** It also annotates
#: :attr:`EvalRun.max_metered_calls_origin` and is the single return of
#: :func:`~threetears.evals.run.ceilings.resolve_ceiling_origin`, whose module says none of that
#: resolution is currency-specific. Read each bullet with the currency swapped and it stays
#: true: the launch argument becomes ``max_metered_calls`` and the setting
#: ``LaunchSettings.max_metered_calls``, and ``uncapped`` comes off the same enforcement flag
#: both ceilings share. The name is cost's because cost came first.
CostCapOrigin = Literal["chosen", "inherited", "uncapped"]

#: Which tier supplied a run's metered-call ceiling (:attr:`EvalRun.max_metered_calls_origin`): the
#: three :data:`CostCapOrigin` tiers, read with the currency swapped, and one more —
#: ``none_declared``, a host declaring it has no metered tools (``LaunchSettings.max_metered_calls``
#: of ``None``). That run records a ceiling of ``0``: no metered call may happen, and one that does
#: contradicts the host's declaration and is refused and counted. Distinct from ``uncapped``, which
#: is a host with metered tools whose enforcement is off.
MeteredCallOrigin = Literal["chosen", "inherited", "uncapped", "none_declared"]


def scored_dim_ids(rubric_dim_names: list[str], judged_artifact: JudgedArtifact) -> list[str]:
    """Every dim a judged cell of this kind is scored on, in the order the judge phase calls them.

    A conversation is scored on the two reserved dual-score axes and then the rubric. A document has
    no turns for the transcript axis to read and no goal for the outcome axis to have reached, so it
    is scored on the rubric alone — and a launch that attributed those axes a judge would record a
    judge for a question nobody asked. One rule, here, for the launch that attributes judges, the
    judge phase that asks them and the re-judge that asks again.

    Args:
        rubric_dim_names: The template's rubric dim names, in template order.
        judged_artifact: What the kind's judge reads (:class:`JudgedArtifact`).

    Returns:
        The dim ids, reserved axes first for a conversation; empty for an unjudged kind.
    """
    if judged_artifact is JudgedArtifact.UNJUDGED:
        return []
    axes = [TRANSCRIPT_DIM_ID, OUTCOME_DIM_ID] if judged_artifact is JudgedArtifact.TRANSCRIPT else []
    return [*axes, *rubric_dim_names]


def resolve_effective_judges(
    dim_ids: list[str],
    judge_config_for_dim: Callable[[str], JudgeConfig | None],
    judge_pin: str,
) -> dict[str, str]:
    """Apply the judge-model cascade to name the model that scores each dim.

    The cascade is ``EVAL_JUDGE`` role default < run-level pin < per-dim
    ``JudgeConfig.model``. The role default is already collapsed into ``judge_pin``
    by the time a run records anything, so only the last two steps appear here.

    Complete rather than sparse — every dim gets an entry, not only the ones that
    overrode the pin — so a reader can answer "which models scored this run" from
    the result alone without re-deriving the cascade against ``judge_model``.

    Args:
        dim_ids: The dims to attribute, from :func:`scored_dim_ids`.
        judge_config_for_dim: Resolves a dim id to the governing config, or
            ``None``. Injected so the launch path can pass the same frozen config
            load it built the judge service from, without reaching for storage
            here.
        judge_pin: The run's resolved run-level judge, used wherever no config
            names a model.

    Returns:
        ``{dim_id: resolved_model}`` over exactly ``dim_ids``.
    """
    effective = {}
    for dim_id in dim_ids:
        config = judge_config_for_dim(dim_id)
        effective[dim_id] = config.model if config is not None and config.model else judge_pin
    return effective


#: Which observation a :class:`RunCompleteness` was counted from. ``run_loop`` is the
#: loop's own tally, the only one that can see a cell that ran and failed to persist;
#: ``stored_results`` is what a later process can reconstruct after the one executing
#: the run died with its tally. They are not interchangeable and the record says which,
#: because a reconstruction's *attribution* of the shortfall is weaker than its total.
class VariationCounts(EvalDocumentModel):
    """How many test cases a launch asked generation for, and how many it froze.

    Generation may legitimately return fewer cases than asked — a ``sample`` or ``enum``
    axis has only so many values, and the Cartesian product of the axes may be smaller
    than the request — so a short case set is not a fault. It is a fact about the run's
    denominator that nothing else on the document records: ``test_case_ids`` says how many
    cases there are, never how many were wanted. ``None`` on a run that reused the
    template's stored cases instead of generating (``n_variations=0``), where nothing was
    requested.
    """

    requested: int = Field(ge=1, description="The n_variations the launch asked generation for.")
    kept: int = Field(ge=0, description="Cases frozen onto the run — reused plus newly minted.")
    reused: int = Field(
        ge=0,
        description="Of the kept cases, how many were already stored for this template with the same variation params.",
    )
    variation_model: str | None = Field(
        default=None,
        description=(
            "RESOLVED model that wrote the values of the template's llm-generated variation axes, as the client "
            "that made the calls named it; None when no axis is llm-generated, so no model wrote any case. "
            "Provenance of the stimulus, not a measurement condition: the cases it wrote are what the candidate "
            "faced, and they are already hashed into the run's context key through `test_case_ids`, so it "
            "enters no identity. Its calls run before the run starts and are outside the run's cost cap and "
            "metered-call ceiling."
        ),
    )

    @property
    def short(self) -> bool:
        """Whether the run holds fewer cases than the launch asked for."""
        return self.kept < self.requested


#: Where a run's completeness counts were taken: the run loop's own tally, or the results in storage
#: (:attr:`RunCompleteness.counted_from`).
CompletenessSource = Literal["run_loop", "stored_results"]


class RunCompleteness(EvalDocumentModel):
    """How much of its matrix a run actually delivered.

    A run whose status says ``completed`` while three of its twelve cells never
    reached storage presents its pass^k on a denominator its siblings do not
    share, and nothing on the record says so — the numbers stay comparable-looking
    while having stopped being comparable. This is the field that says so.

    **Counts only.** ``degraded`` and :attr:`measured_cells` are computed from
    them rather than stored, for the reason
    :class:`~threetears.evals.contracts.result_condition.ResultCondition` gives: a derivation
    written into a document becomes indistinguishable from an observation the next
    time the document is saved, and then a change to the rule silently disagrees
    with every row already written under the old one.

    **Written on every terminal run**, whatever stopped it — a completed matrix,
    an operator's cancel, the per-run cost cap, the job timeout, a harness failure,
    a cancel that landed while the job was still queued for a slot, or a hard kill
    reclaimed by a later process. That is the point: a run that produced fewer
    cells than intended is badged on its *denominator*, never on the cause, because
    a badge keyed on cause has to enumerate causes and misses the next one — which
    is why the write is derived from whether the run's work function ever started
    rather than requested at each terminal branch.

    Absence on a terminal run therefore means every attempt to write the record was
    refused, which is rare and loud in the log (``eval.completeness … NOT recorded``).
    Stating that case rather than claiming an absolute is deliberate: the write is
    retried and can still fail, and a guarantee the code keeps only nearly always is
    worse than an honest one, because it is the absent case that gets believed.

    Where the counts came from is not uniform, and :attr:`counted_from` says
    which — the loop's own tally, or a reconstruction from what reached storage,
    which sees strictly less.
    """

    expected_cells: int = Field(
        ge=0,
        description=(
            "The matrix this run committed to at launch: ``len(test_case_ids) × k_runs`` at its one "
            "candidate model, computed from the RUN document rather than from the test-case list the "
            "loop was handed. Those two can disagree — the loop counts what it was given, the "
            "run states what it promised — and taking the denominator from the run is what makes "
            "such a disagreement visible instead of internally consistent."
        ),
    )
    produced_cells: int = Field(
        ge=0,
        description=(
            "Cells the loop actually executed and produced a result for. Below ``expected_cells`` "
            "means the run was launched against fewer cases than it recorded; equal to it is the "
            "normal case, since the loop is total over whatever matrix it was handed. **Above it "
            "is the mirror disagreement and is not folded into the shortfall**: the run delivered "
            "everything it promised, so it is not degraded, but the surplus cells were never part "
            "of what it committed to and no rate here is computed over them. Only the "
            "``produced <= expected`` side sums cleanly into the disclosure's causes; the surplus "
            "case is visible as the gap between these two counts and nowhere else."
        ),
    )
    persisted_cells: int = Field(
        ge=0,
        description=(
            "Cells whose write landed. The storage layer reports a failed write by returning "
            "False rather than raising, so a cell can run, produce a result, and leave no row — "
            "which is the one shortfall no count of rows in storage can ever detect, because the "
            "row is not there to count. That is why this record is stored rather than derived at "
            "read time."
        ),
    )
    infra_excluded_cells: int = Field(
        ge=0,
        description=(
            "Persisted cells a harness failure removed from the aggregates (factory, simulator, "
            "delivery, judge, a cell timeout charged to the rig). Counted apart from candidate failures on purpose: a "
            "candidate that failed is a real measurement scored as a hard fail and belongs in the "
            "denominator, while an infra exclusion shortens it. Both look like 'a cell that did "
            "not go well' and only one of them costs the run comparability."
        ),
    )

    counted_from: CompletenessSource = Field(
        description=(
            "Where the counts above were taken. ``run_loop`` is the run's own in-process tally — the "
            "cells its loop executed, which is the only observation that can see a cell whose write was "
            "refused. It covers a run whose loop never ran at all (a cancel that landed while the job was "
            "still queued for a slot): zero executed cells is that tally, not a reconstruction of one, and "
            "the distinction this field draws is loop-side truth versus storage-side inference rather than "
            "how far the loop got. "
            "``stored_results`` is a reconstruction, made when a process died mid-run and a later one "
            "reclaimed the document: the loop's tally died with it, so the counts are read back from "
            "the results that reached storage and a cell that ran without persisting is indistinguishable "
            "from one that never ran — it lands in ``expected - produced`` rather than in "
            "``produced - persisted``. The total shortfall is right either way; its attribution is not. "
            "Required: each writer states which observation it counted from."
        ),
    )

    @property
    def measured_cells(self) -> int:
        """Cells that contributed a real observation — persisted, and not infra-excluded."""
        return self.persisted_cells - self.infra_excluded_cells

    @property
    def degraded(self) -> bool:
        """Whether this run delivered fewer measurements than the matrix it promised."""
        return self.measured_cells < self.expected_cells


#: A reasoning effort level, as the router's ``reasoning.effort`` takes it. ``none`` disables
#: reasoning, and a model whose reasoning is mandatory refuses it; which of the others a model
#: accepts is its own (the router lists each model's ``supported_efforts``).
ReasoningEffort = Literal["max", "xhigh", "high", "medium", "low", "minimal", "none"]


class ClientRequestSettings(EvalDocumentModel):
    """The request parameters a host applied to one apparatus role's LLM client, for one run.

    Recorded because the same model can be asked two different ways, and the asking is part of
    the instrument: a reasoning judge given a private-reasoning budget thinks before it scores,
    one sent none may not, and an output cap below what a reasoning model spends cuts its answer
    off. Two runs carrying the same ``judge_model`` were therefore NOT judged alike if these
    differ, and nothing else on the run can say so — the values are applied process-wide by the
    host's client builder, and moving them regrades every later run without touching a model id.

    **Reasoning is asked for one way or the other, never both.** ``reasoning_max_tokens`` is a
    budget in tokens, which Anthropic- and Gemini-style models take as one; ``reasoning_effort`` is
    a level, which is all an effort-only model (OpenAI's reasoning series) understands — the router
    maps a token budget onto one of those by its share of the cap, so a budget sent to such a model
    bounds nothing in tokens. A request carrying both would leave the provider to pick, and the
    record could not say which it did, so a settings value naming both is refused rather than
    given a precedence.

    The host builds this from the same value its client builder applies, so the stamp and the
    request cannot disagree (see :data:`threetears.evals.run.judge.JUDGE_REQUEST_SETTINGS`).
    """

    max_tokens: int = Field(gt=0, description="The output cap every request of this role was sent with.")
    reasoning_max_tokens: int | None = Field(
        default=None,
        gt=0,
        description=(
            "The private-reasoning budget sent with every request (the provider's `reasoning.max_tokens`), a share "
            "of `max_tokens`. None, with `reasoning_effort` also None, = no reasoning parameter was sent, so the "
            "provider's default applied — a recorded level, not an absence. Never set beside `reasoning_effort`."
        ),
    )
    reasoning_effort: ReasoningEffort | None = Field(
        default=None,
        description=(
            "The reasoning effort level sent with every request (the provider's `reasoning.effort`): a level, not a "
            "token bound, so `max_tokens` stays the only hard stop. None = no effort was sent. Never set beside "
            "`reasoning_max_tokens`."
        ),
    )

    strict_output: bool = Field(
        default=False,
        description=(
            "Whether every request of this role must be routed only to a provider that honours every parameter "
            "sent (an OpenRouter-style `provider.require_parameters`), so a strict `response_format` or a reasoning "
            "bound is enforced rather than silently dropped. The host's client builder applies it. False = no such "
            "requirement was stated, which is also how a stamp stored before this field reads: none was."
        ),
    )

    @model_validator(mode="after")
    def _one_way_to_ask_for_reasoning(self) -> Self:
        """Refuse a settings value that asks for reasoning both by a token budget and by an effort level.

        Returns:
            The settings, unchanged.

        Raises:
            ValueError: Both ``reasoning_max_tokens`` and ``reasoning_effort`` are set.
        """
        if self.reasoning_max_tokens is not None and self.reasoning_effort is not None:
            raise ValueError(
                "reasoning is asked for by a token budget (reasoning_max_tokens) or by an effort level "
                "(reasoning_effort), not both: the provider would pick one and the record could not say which"
            )
        return self


#: Repeats per (case, model) a launch uses when the caller names none: one observation per case cannot
#: tell a setting from the model's own variance. Every launch entrypoint
#: defaults to this; ``EvalRun.k_runs`` has no default of its own, so whoever writes a run states it.
DEFAULT_LAUNCH_K_RUNS = 3


#: The merit axes a declaration may name: :data:`~threetears.evals.contracts.metrics.MeritAxis`, restated here
#: because that module imports this one; ``metrics`` checks the two agree at import.
DeclaredMeritAxis = Literal["quality", "cost", "latency", "reliability"]


class MeasureDeclaration(EvalBaseModel):
    """How the launching host declared one of its own measures to be read, frozen onto the run it launched.

    The reading-side half of a :class:`~threetears.evals.contracts.metrics.MetricDescriptor`: which way is better,
    the merit axis it serves, whether it is a guardrail, its margin and its range. A host builds these in code, and
    the quick path builds them from ``compare(margins=, ranges=, guardrails=)``, so a later reader's host may declare
    them differently, or not at all. A comparison of stored runs is read on what the runs were launched under
    (:attr:`EvalRun.declared_measures`), never on whoever reads it.
    """

    higher_is_better: bool | None = None
    merit_axis: DeclaredMeritAxis | None = None
    guardrail: bool = False
    materiality_threshold: float | None = None
    value_range: tuple[float, float] | None = None


class EvalRun(EvalDocumentModel):
    """One execution of a template (or explicit test case set) against one candidate model.

    **A run is one arm.** It names exactly one ``candidate_model``, so a run read as "one arm at a
    set of models" is not a state any document can be in; a launch naming several models starts
    one run per model.

    ``test_case_ids`` is frozen at run start — even if new test cases are
    generated for the template after the run begins, this run scores against
    the exact set captured here.  That's the idempotency property that lets
    run-vs-run comparison subtract the same denominator.

    ``budget_stopped`` is a run a budget it was launched under stopped GRACEFULLY
    mid-flight, with its already-delivered results preserved: the per-run cost cap, once
    its accumulated spend exceeded the cap, or the job's wall-clock budget (sized to the
    matrix), once it ran out. ``budget_stop_reason`` says which. That is an honest terminal
    outcome, not an infra failure, so it is its own status rather than ``failed``.
    """

    @property
    def elided_payload_paths(self) -> frozenset[str]:
        """Paths the read that produced this copy left out of ``host_payload``; empty when whole.

        A listing drops what its host declared in ``HostProfile.listing_elisions``. A reader that
        needs a dropped value refuses on a non-empty answer here rather than reading its absence
        as "never recorded" — which is what the missing key would otherwise look like.
        """
        return self._elided_payload_paths

    def note_elided_payload(self, paths: frozenset[str]) -> None:
        """Record that ``paths`` were left out of this copy's ``host_payload`` by the read.

        Args:
            paths: Dotted paths relative to ``host_payload``.
        """
        self._elided_payload_paths = frozenset(paths)

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    doc_type: Literal["eval_run"] = "eval_run"
    schema_version: SchemaVersion = EVAL_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    # What runs
    template_id: str | None = Field(
        default=None,
        description=(
            "The template the run launched; for a witnessed run, the template its cells are judged against "
            "(`stamp_witnessed_judge`), None when it is unjudged. None on a commissioned run = ad-hoc, from explicit "
            "test_case_ids."
        ),
    )
    subject_snapshot: SubjectSnapshot = Field(
        description=(
            "Who was measured, in the engine's vocabulary: a required non-empty key, a separate reader-facing "
            "label, and the content-addressed levels of whatever components the host registered. The variant key "
            "is computed over these. Holds hashes, never component text."
        ),
    )
    host_payload: VerbatimObject = Field(
        default_factory=dict,
        description=(
            "The rich object the HOST needs and the engine must never read. Opaque in the same sense `host_id` is: "
            "nothing in the engine outside the host's own adapter may reach into it, and a gate says so. It "
            "exists because `subject_snapshot` structurally cannot carry a payload — it holds content hashes, so a "
            "run's retention stays bounded by its matrix rather than by what the candidate produced — while three "
            "host consumers need the object itself: the runner instantiates a live subject from it, the judge "
            "renders it into the scoring prompt, and the A/B comparison shows it to an operator."
        ),
    )
    # Private, so it is never serialized or stored: it describes how THIS copy was read, not the run.
    _elided_payload_paths: frozenset[str] = PrivateAttr(default_factory=frozenset)
    candidate_model: str = Field(
        min_length=1,
        description=(
            "The one model the candidate ran on. A run is one arm, so it names one model; a launch naming several "
            "starts one run per model."
        ),
    )
    k_runs: int = Field(ge=1, le=20)
    test_case_ids: list[str] = Field(min_length=1)
    apparatus_provenance: ApparatusProvenance = Field(
        description=(
            "How this run's apparatus came to be: `commissioned` = a rig was set and the observations were gathered "
            "under it (the launch path stamps it on every run it starts); `witnessed` = nothing set it, and the host "
            "recorded the apparatus it found on traffic it did not control (a captured session). Required, with no "
            "default: whoever writes a run is the only party that knows which it is, and a default would let a "
            "captured log read as an experiment. The analysis reads it per observation, and the two never share a cell."
        ),
    )
    variation_counts: VariationCounts | None = Field(
        default=None,
        description=(
            "Requested vs. frozen case counts when the launch generated its cases; None when it reused the "
            "template's stored cases. kept < requested means generation ran short of the request."
        ),
    )
    candidate_kind: str = Field(
        min_length=1,
        description=(
            "Which candidate kind this run launched — stamped at launch from the template's "
            "`candidate_kind`, the same field the launch dispatched on. Recorded on the run rather "
            "than read off its template because templates are store-mastered and editable, so a "
            "template lookup answers with today's kind rather than the one that ran. It is what "
            "tells a surface which apparatus the run HAD: a single-shot kind runs no simulated user, "
            "so its blank `simulator_model` means 'this kind has none'. Required: a run that cannot "
            "say which kind it measured cannot be read against any kind's contract."
        ),
    )

    # Optional
    judge_model: str | None = Field(
        default=None,
        description=(
            "RESOLVED model the rubric judge falls back to, captured at launch — the run-level pin "
            "a scored dimension uses when no JudgeConfig names its own. Resolved rather than "
            "as-passed for the reason given on ``simulator_model``: the launch parameter's None means "
            "'role default', so two runs that both stored None compared equal across a change to that "
            "default AND could not be reproduced once it moved. Whether the value was named or "
            "inherited is recorded separately, on ``model_role_provenance``. This is the model "
            "REQUESTED, not the scorer's identity: a provider-side floating alias "
            "(``~vendor/model-latest``) is resolved after the request leaves, so the model that "
            "answered is recorded per score as ``RubricScore.served_model``, and that is what "
            "apparatus comparison reads. None = the run is not judged: the runner refuses to execute a "
            "judged run that names no judge, or an unjudged run that names one."
        ),
    )
    model_role_provenance: dict[str, RoleModelOrigin] | None = Field(
        default=None,
        description=(
            "How each pinned role model was arrived at, keyed by role (``candidate`` / ``judge`` / ``simulator``): "
            "``chosen`` = the launch named it, ``inherited`` = the role default supplied it, "
            "``alternate`` = the launch named no judge and the host's alternate judge scored in place of a "
            "role default that was one of the launch's candidates (see ``RoleModelOrigin``). Kept "
            "beside the resolved pins because resolution makes the two indistinguishable on the "
            "value alone, and they answer different questions — the resolved model says what ran, "
            "the origin says whether re-running today would pick the same one. Deliberately NOT "
            "hashed into any identity key: a run that named the default and a run that inherited it "
            "were measured under identical conditions, so splitting them would assert a difference "
            "that does not exist. ``candidate`` is the launch's naming of the candidate model (``chosen``) or its "
            "running at the kind's own default (``inherited``), which is what says whether a production-replicating "
            "cost was measured off the subject's model; a run stored before it carries no ``candidate`` key, read as "
            "not recorded. None = the run's writer recorded no origins; a missing role key = that "
            "role was not pinned on this run (or, for ``candidate``, not recorded)."
        ),
    )
    effective_judges: dict[DimName, str] | None = Field(
        default=None,
        description=(
            "The model each dim was REQUESTED from, ``{dim_id: resolved_model}``, captured at "
            "launch for EVERY dim the kind's judge scores (``scored_dim_ids``: the two reserved dual-score "
            "axes for a conversation, then each of the template's rubric dims) — not only the ones that "
            "overrode the pin. Complete rather "
            "than sparse so a reader can answer 'which models scored this run' from this field "
            "alone, without re-deriving the cascade against ``judge_model``. Exists because "
            "``judge_model`` names the run's PIN, and under the judge-model cascade (role default "
            "< run pin < per-dim ``JudgeConfig.model``) a dim whose config names a model is scored "
            "by that instead — so a run stamping one judge could have had its rubric scored by "
            "another, and two such runs differing only in that were reading as comparable. "
            "Knowable at launch because the launch's ``build_judge_service`` freezes each dim's config there, "
            "so the REQUEST drifts nothing mid-run — the model that answered can, when the value is "
            "a provider-side floating alias, and that is recorded per score as "
            "``RubricScore.served_model``. None = no per-dim attribution was recorded: the run pinned "
            "no judge, or its writer recorded none, and whether its dims diverged is then unknowable "
            "from the run alone."
        ),
    )
    rubric_scales: dict[DimName, RubricScale] = Field(
        description=(
            "How each template rubric dim was answered, ``{dim: 'ordinal' | 'pass_fail'}``, captured at "
            "launch from the template. A stored score carries its own scale; this is what names the "
            "scale of a dim no trial produced a score for (every answer 'can't tell'). Required, and "
            "``{}`` for a template with no rubric: a scale is never assumed."
        ),
    )

    def rubric_scale(self, dim: str) -> RubricScale:
        """The scale ``dim`` was judged on in this run.

        Raises:
            ValueError: ``dim`` is not among the scales the run recorded.
        """
        if dim not in self.rubric_scales:
            raise ValueError(f"run {self.id} recorded no scale for rubric dim {dim!r}")
        return self.rubric_scales[dim]

    effective_judges_source: JudgeAttributionSource | None = Field(
        default=None,
        description=(
            "Whether ``effective_judges`` was captured at launch (``recorded``) or reconstructed "
            "afterwards from stored ``JudgeConfig`` records (``derived``). Kept beside the map "
            "rather than inferred from whether it is set, because the two are not interchangeable: only "
            "``recorded`` attribution is hashed into the context key. A reconstruction infers each "
            "dim's config from stored records as of the run's start — a strong inference, since "
            "configs supersede rather than replace, but not a record of what scored it: it cannot "
            "see a config authored and archived inside the run's own window, it resolves at the "
            "start instant rather than as the run progressed, and it is blind to a config "
            "hard-deleted rather than archived. A ``derived`` run therefore still reports its "
            "context as partial while displaying what the reconstruction found. None = no "
            "attribution of either kind."
        ),
    )
    judge_config_ids: dict[DimName, str] | None = Field(
        default=None,
        description=(
            "The ``JudgeConfig`` SET this run committed to, ``{dim_id: config_id}``, frozen at "
            "launch from the same config load that built the judge service. Sparse — a dim with no "
            "config has no key — which is safe because ``effective_judges`` records every scored "
            "dim beside it, so the dim set is never inferred from this map. Exists because the pin "
            "was previously in-memory and implicit: the set was reconstructible only after the fact "
            "from each result's ``judge_config_ids``, so a run whose results were lost, or which "
            "delivered none, could not say what judged it. ``None`` means no set was recorded: the "
            "run pinned no judge, or its writer recorded none. ``{}`` is NOT the same answer: it says no scored dim carried a config, so every "
            "dim ran on the judge's built-in default prompt — a real and stable measurement "
            "condition. **Never test this field for truthiness**; the two collapse and runs that "
            "genuinely agree stop pooling together."
        ),
    )
    judge_config_provenance: dict[DimName, ModelRoleOrigin] | None = Field(
        default=None,
        description=(
            "How each entry of ``judge_config_ids`` was arrived at, keyed by the same dim ids: "
            "``chosen`` = the launch named this config for this dim, ``inherited`` = the active-config "
            "lookup supplied it. Per-dim rather than one value for the run because a launch may name "
            "configs for some dims and leave the rest to the lookup, and a single ``chosen`` would "
            "then claim more than the launch said. Kept beside the resolved set for the reason "
            "``model_role_provenance`` is, and hashed into nothing for the same one: a dim whose "
            "config was named and a dim that inherited the identical config were measured under "
            "identical conditions. ``None`` = the run's writer recorded no origins."
        ),
    )
    overlays: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The knobs this run's launch turned, as the kind's overlay model validated them — every field the "
            "model declares, defaults included, so a launch naming a default and one naming nothing record the "
            "same thing. Frozen at launch: the model is the kind's code and moves, so this is what the run "
            "actually ran at. The engine reads it only through the levers the kind's contract derives "
            "(``HostProfile.kinds``). Empty for a kind that declares no overlays."
        ),
    )
    apparatus_settings: dict[str, ApparatusSettingValue] = Field(
        default_factory=dict,
        description=(
            "The host-declared apparatus values this run's launch set — a setup value of the measuring rig the "
            "kind's launcher reads (an adjudicator's seat, a rules version), keyed by the apparatus dimension "
            "the host declares, so one template can be run at two of them and compared. Each is a launch "
            "argument the kind declares it honours (``LaunchableKind.apparatus_settings``). Hashed into the "
            "measurement context: two runs whose rig was set up differently are not repetitions of one "
            "condition. Empty when the launch set none, which is a level."
        ),
    )
    declared_margins: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Margins the launch declared on core rate measures (``accuracy``), by measure, in the measure's units: "
            "the most two arms may differ on it and still be alike. A core measure's descriptor is the engine's and "
            "declares no margin, so a run-scoped one is how a comparison of runs can read `equivalent` on it; the "
            "analysis reads it only when every member run of the campaign declares the same margin, and each "
            "contrast tested against it names it (``FamilyComparison.margin_source`` `run`). Declared at launch, "
            "before any result, so it is chosen before the data is seen. Not part of the measurement context: a "
            "margin changes how a difference is read, not what was measured. Empty when the launch declared none, "
            "and on every run stored before run-scoped margins existed, which then read as declaring none."
        ),
    )
    declared_measures: dict[str, MeasureDeclaration] = Field(
        default_factory=dict,
        description=(
            "How the launching host declared each of its own measures to be read — direction, merit axis, "
            "guardrail, margin (``materiality_threshold``) and range — by measure, frozen at launch. A campaign's "
            "analysis reads a measure on the declaration its member runs agree on, whatever the reading host now "
            "declares, and names any difference; so a stored comparison's verdicts do not depend on who reads it. "
            "Not part of the measurement context: a declaration changes how a difference is read, not what was "
            "measured. Empty on a run stored before it, which is then read on the reading host's declarations."
        ),
    )
    resolved_world_seed: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The template's ``world_seed`` as THIS run froze it at ``start_run`` — the immutable "
            "provenance of the world the candidate actually faced, and the seed the goal judge and the eval context panel read. Maps each "
            "carrier to its keyed values, exactly as ``WorldSeed.namespaces`` holds them. Persisted on the "
            "RUN, not the template: the template seed is mutable/versioned, "
            "but the run needs the frozen world of this run (context-panel render, "
            "provenance/repro, world diff in bisect). Empty ``{}`` when the template declared no "
            "world seed."
        ),
    )
    resolved_ambient_perturbation_turns: list[int] = Field(
        default_factory=list,
        description=(
            "The template's ``world_seed.ambient_perturbation_turns`` as THIS run froze it at launch — the turns "
            "before which the host's ambient perturbation moved undeclared state. Persisted beside "
            "``resolved_world_seed`` for its reason (the template is mutable and versioned) and hashed into the "
            "measurement context with it, so a run that perturbed and one that did not are never pooled as "
            "repetitions of one condition. Empty when the template scheduled none."
        ),
    )

    kind_spec: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The template's ``kind_spec`` as THIS run froze it at launch, re-validated against the kind's spec "
            "model then — every field, defaults included. Persisted on the RUN for the reason "
            "``resolved_world_seed`` is: the template is mutable and versioned, so ``template_id`` alone "
            "cannot say what the candidate was handed, and this is part of what it was handed. Hashed into "
            "the measurement context. Empty for a kind that declares no spec."
        ),
    )

    world_placements: dict[str, WorldPlacement] | None = Field(
        default=None,
        description=(
            "What this run actually did with every dimension the host's world declares, derived at "
            "launch from the run's own seed record and the carriers its subject holds — never declared. "
            "Dimension name -> ``representable`` (this run seeded it and the subject perceived it) / "
            "``judge_only`` (seeded, unperceived) / ``witnessed`` (perceived, unseeded — a confound this "
            "run did not control) / ``out_of_play`` (neither, so not a precondition this run could "
            "presume). Persisted rather than re-derived on read for the reason ``resolved_world_seed`` "
            "is: the registry is code and moves, so a later derivation would report today's world for a "
            "run measured under an older one. ``None`` means NOT RECORDED — a run whose writer placed "
            "nothing (a host assembling a run without its launch, which is where placements are "
            "derived), which is a different fact from ``{}``, the recording that this run placed no "
            "dimensions at all (a host that declares no world, or an empty registry)."
        ),
    )

    goal_check_proofs: dict[str, GoalCheckProof] | None = Field(
        default=None,
        description=(
            "Each goal check the run grades -> whether it was shown to tell its outcomes apart when the run "
            "launched (`GoalCheckProof`), frozen for the reason `resolved_world_seed` is: the template's controls "
            "are editable. Only `proven` reads as measuring the behaviour; every surface showing an `unproven` or "
            "`refuted` check's pass rate marks it. None = NOT RECORDED: a run launched before proofs were frozen, "
            "or assembled without a launch — read as unproven, never as proven. Optional within v8 for that reason."
        ),
    )
    goal_check_proof_rules: int | None = Field(
        default=None,
        description=(
            "The proof rules `goal_check_proofs` were derived under (`GOAL_CHECK_PROOF_RULES`). None on a run "
            "launched before the rules were stamped, which are rules 1. A `proven` recorded under rules older than "
            "the current ones reads as unproven (`goal_check_proofs_as_read`), and is counted as needing re-proof."
        ),
    )
    refused_goal_checks: dict[str, str] | None = Field(
        default=None,
        description=(
            "Each of the template's goal checks the grammar refused when the run launched -> the refusal's reason. "
            "A template stored before a grammar rule can carry a check the rule now refuses; the run grades none "
            "of its cells on it, so the cells are not rig faults, and every surface names the check as refused "
            "under the current grammar. None = not recorded (a run launched before this was frozen); {} = none."
        ),
    )

    resolved_tools_allowed: list[str] | None = Field(
        default=None,
        description=(
            "The template's ``tools_allowed`` frozen onto THIS run at ``start_run`` — the bound on which "
            "tools the candidate could call, and so on what it could spend at third parties. Persisted on "
            "the RUN for the same reason ``resolved_world_seed`` and ``kind_spec`` are: the "
            "template is mutable and versioned, so ``template_id`` cannot say what the candidate was "
            "handed. Without it a bounded run and a later unbounded run of the same template are "
            "indistinguishable — the variant key hashes the subject's tool configs and the context key "
            "hashes ``template_id``, so neither moves when this does, and the two would pool as "
            "repetitions of one condition while the apparatus scan reported the rig as having held still. "
            "``None`` means unbounded — the template allowed every tool the subject carried — which is a "
            "level the context key records, not an absence."
        ),
    )

    # Cassette layer. ``'off'`` is the default; ``'capture'`` runs tools live and records what
    # came back into this run's own corpus (named by this run's ``id``), ``'replay'`` re-serves the
    # corpus a capture run recorded and spends no third-party quota.
    cassette_mode: Literal["capture", "replay", "off"] = Field(default="off")
    cassette_corpus_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The capture run whose recorded corpus a replay run serves, set exactly when cassette_mode is "
            "'replay'. A capture run writes the corpus its own id names, so two runs capturing at once never "
            "write into each other's, and a replay binds to one whole recording rather than to whatever the "
            "latest capture of each case happened to leave."
        ),
    )
    measure_latency: bool | None = Field(
        default=None,
        description=(
            "Whether the launch declared latency under test (`start_run(measure_latency=True)`). True: the run "
            "executed its cells one at a time, with no other run executing beside it, so the latency it recorded "
            "is read clean. False: it was not under test, and the run executed its cells concurrently "
            "(`cell_concurrency`). None: launched before 3tears-evals recorded the declaration (or recorded by a "
            "writer that is no launch) — such a run executed its cells one at a time, and whether it ran beside "
            "another run is what its results' `execution_mode` says."
        ),
    )
    cell_concurrency: int | None = Field(
        default=None,
        ge=1,
        description=(
            "How many of the run's cells executed at once: 1 is serial. Above 1, every result stamps "
            "`execution_mode` `concurrent` and its latency is kept out of every comparison, bar and ranking. "
            "None: a run stored before the width was recorded, whose cells executed one at a time — read as 1, "
            "never as unknown, because the runner of that build could not run them otherwise."
        ),
    )

    # Pinned roles + spend ceiling: inputs that can move a run's numbers, recorded on the run so
    # a diff of two runs can see them.
    simulator_model: str | None = Field(
        default=None,
        description=(
            "RESOLVED model that drove the simulated user, captured at launch. Resolved rather than "
            "as-passed because the launch parameter's None means 'role default', and two runs that both "
            "stored None would compare equal across a change to that default. None = no simulated user "
            "was pinned: the kind runs none, or the run's writer recorded none."
        ),
    )
    judge_request_settings: ClientRequestSettings | None = Field(
        default=None,
        description=(
            "The output cap and reasoning budget every judge request of this run was sent with, stamped at "
            "launch from the value the host's client builder applies to the judge role. Part of the judge "
            "apparatus beside ``judge_model``: the same judge model at a different reasoning budget grades "
            "differently. None = no judge was pinned (a code-graded kind), or the run's writer recorded no "
            "settings — then a comparison reads them as unrecorded, never as equal to today's values."
        ),
    )
    judge_temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description=(
            "The temperature this run's judge calls were requested at for every dimension with no JudgeConfig "
            "(a config states its own, and its id is already part of the judge's identity). Stamped at launch, and "
            "part of the measurement context's roles: a run judged at another temperature is judged by another "
            "judge and never pools with this one. What each call was actually sent at — none, for a model that "
            "refuses a temperature — is on each score. None = no judge was pinned, or the run was launched before "
            "this was recorded, when such dimensions were requested at the provider's default; its roles component "
            "is then not composable, never equal to today's."
        ),
    )
    simulator_request_settings: ClientRequestSettings | None = Field(
        default=None,
        description=(
            "The output cap and reasoning parameter every simulated-user request of this run was sent with, "
            "stamped at launch from the value the host's client builder applies to the simulator role — "
            "recorded for the reason ``judge_request_settings`` is. None = no simulated user was pinned, or "
            "the run's writer recorded no settings."
        ),
    )
    max_cost_usd: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Effective per-run spend ceiling in force, after the override-or-config cascade. None means the "
            "run was uncapped (cost enforcement disabled, origin ``uncapped``) or its writer recorded no "
            "ceiling (origin None); a truncated run is visible either way through the budget_stopped status."
        ),
    )
    max_cost_usd_origin: CostCapOrigin | None = Field(
        default=None,
        description=(
            "Which tier of the cascade supplied max_cost_usd — see CostCapOrigin. Kept beside the ceiling for "
            "the reason model_role_provenance is kept beside the judge and simulator pins: the ceiling is "
            "stored resolved, and a resolved number cannot say whether the launch named it or a config default "
            "did — and only the second kind changes under a run's feet when nobody touched the launch. Not "
            "hashed into any identity key, on the same reasoning: two runs bounded by the same ceiling were "
            "measured under the same condition however each arrived at it. None = the run's writer recorded "
            "no ceiling (a run assembled without its launch, which is where the cascade resolves)."
        ),
    )

    max_metered_calls: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Effective per-run ceiling on METERED THIRD-PARTY CALLS in force, after the "
            "override-or-config cascade — the quota max_cost_usd cannot see, since that counts "
            "LLM dollars and a search credit or a billed image generation is neither. None "
            "means the run was unbounded (eval enforcement disabled) or its writer recorded no "
            "ceiling; max_metered_calls_origin tells those apart. 0 means the host declared it has no "
            "metered tools (origin none_declared), so any metered call is refused. Reaching it REFUSES further metered "
            "calls and records the count on metered_calls_refused — it never stops the run, "
            "because a hard stop would discard a partly-measured matrix."
        ),
    )
    max_metered_calls_origin: MeteredCallOrigin | None = Field(
        default=None,
        description=(
            "Which tier of the cascade supplied max_metered_calls — see MeteredCallOrigin. Kept "
            "beside the ceiling for the reason max_cost_usd_origin is: the ceiling is stored "
            "resolved, and a resolved number cannot say whether the launch named it or "
            "the host's configured default did — and only the second kind moves "
            "under a run's feet when nobody touched the launch. None = the run's writer recorded "
            "no ceiling."
        ),
    )
    metered_calls_refused: int | None = Field(
        default=None,
        ge=0,
        description=(
            "How many metered third-party calls this run's ceiling turned away, across every "
            "cell. Non-zero means the run's measurement was BOUNDED — the candidate asked for "
            "provider calls it did not get, so what it could not call it could not be measured "
            "on, and a truncated run must not read as a complete one. Written once, when the "
            "run loop finishes, beside completeness and for the same reason: it is the "
            "disclosure that cannot be reconstructed from storage afterwards. Zero is the "
            "honest record of a run that stayed inside its ceiling; None means no ledger "
            "counted: the run's loop never started, or it ran with no ledger at all."
        ),
    )
    turn_budget_s: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "The host's per-turn budget every candidate turn of this run ran under, in seconds, resolved once "
            "at launch from the bound the host runs its turns under in production. A turn that outlives it is "
            "ended, which the result counts as a candidate failure (covariate turns_ended_by_budget). None "
            "means no budget bounded the turns — a candidate kind or a host with none. Not hashed into any identity key, like max_cost_usd: it is a condition stated beside "
            "the run, and a comparison across two budgets reads it here."
        ),
    )

    # Measurement-context identity, stamped at launch.
    context_key: str | None = Field(
        default=None,
        description=(
            "Digest composing context_components — the pinned conditions a comparison must hold fixed. "
            "Stamped by the launch; None on a run its host assembled without it, whose key is derived "
            "at read time and labelled as derived."
        ),
    )
    context_components: ContextComponents | None = Field(
        default=None,
        description="The separately-recorded pieces context_key composes, so a mismatch badges which one differs.",
    )
    identity_version: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Which key-derivation predicate produced context_key. Stored beside the key rather than only "
            "hashed into it so keys from different predicates are queryably distinct instead of silently "
            "regrouped. None exactly when context_key is."
        ),
    )

    # Contestant identity's PRE-IMAGE, stamped at launch beside the context key's.
    variant_levers: dict[str, SweepableValue] | None = Field(
        default=None,
        description=(
            "The resolved lever map this run's variant_key was digested from, at its candidate model — the "
            "variant predicate's pre-image, kept for the reason context_components is kept beside "
            "context_key: a digest says two observations are the same contestant stack and refuses to say "
            "what that stack WAS. Recorded rather than re-derived on read, and "
            "that is the whole point: re-derivation replays TODAY's predicate, which is designed to move, so "
            "every arm of a campaign analysed across an IDENTITY_VERSION bump became undescribable at once — "
            "the keys stayed authoritative for pooling and nothing could say what any of them ran. A "
            "recorded map needs no predicate to read it, so a bump costs a description nothing. None means "
            "NOT RECORDED — a run its host assembled without the launch, which stays on the read-time "
            "derivation. The launch always records one: the engine resolves every run's model, kind and "
            "kind-contract levels itself."
        ),
    )

    launch_group_id: str | None = Field(
        default=None,
        description=(
            "The campaign launch this run was started in, shared by every arm that launch started "
            "together in one concurrency slot, so their measurement windows overlap by construction. "
            "None = launched on its own, which puts this arm's timing beside its siblings' only by the "
            "scheduler's chance — the analysis says so when a campaign's arms do not share one group."
        ),
    )

    # Lifecycle
    status: EvalRunStatus = Field(default="pending")
    completeness: RunCompleteness | None = Field(
        default=None,
        description=(
            "How much of its matrix this run delivered. ``status`` says how the run ENDED; this "
            "says whether what it ended with is the whole of what it promised — a run can end "
            "``completed`` while short, which is the case it exists for, and it can end "
            "``cancelled`` after one cell of five, which is the case that used to report a "
            "perfect rate unqualified. Written on every terminal run whatever stopped it — including "
            "a harness failure, and a cancel that lands while the job is still queued for a slot, whose "
            "honest record is 0 of N. On a TERMINAL run None therefore means every attempt to write the "
            "record was refused (rare, and loud in the log). A run "
            "that has not gone terminal yet simply has none — this field is readable at any status, so "
            "a surface reading across runs of mixed status cannot treat absence as either. Absence is "
            "never evidence that nothing was delivered: read it as unknown."
        ),
    )
    progress: dict[str, Any] = Field(default_factory=dict)
    cancellation_reason: str | None = Field(
        default=None,
        description=(
            "Why a human stopped this run, when one did. Its own channel rather than an "
            "``error_details`` entry, because a cancel is a distinct terminal outcome — neither "
            "failure nor success — and filing it beside genuine harness errors made a controlled "
            "stop read as breakage in every later triage. Says nothing about whether the run's "
            "data is usable: that is ``completeness``'s answer. None on a run nobody cancelled, and "
            "on one cancelled without a reason given."
        ),
    )
    budget_stop_reason: str | None = Field(
        default=None,
        description=(
            "Why a budget the run was launched under stopped it, when one did: the per-run cost cap "
            "(the reason names the dollars spent against the cap) or the job's wall-clock budget (the "
            "reason opens with ``wall-clock budget``). Its own channel rather than an "
            "``error_details`` entry, for the same reason ``cancellation_reason`` is: a bound the "
            "operator configured doing exactly what it was configured to do is a designed terminal "
            "outcome, not a fault, and filing it beside genuine harness errors made every capped run "
            "that bound contribute a phantom to the error count operators scan for real breakage. "
            "Every run has a cap (an omitted ``max_cost_usd`` inherits "
            "the host's configured default), so that was not a rare miscount. Says nothing "
            "about whether the run's data is usable — that is ``completeness``'s answer. None on a "
            "run no budget stopped. A run whose wall-clock budget fired under 3tears-evals 0.66.0 or "
            "earlier is stored ``failed`` with a ``Job timed out after Ns`` entry in ``error_details``; "
            "it is not rewritten, so that entry is how such a run reads."
        ),
    )
    error_details: list[str] = Field(
        default_factory=list,
        description=(
            "Things that BROKE — the run-terminal failure channel, written only by the job "
            "manager's failure branches (a harness exception, an exhausted provider account). The "
            "designed stops each have their own channel (``cancellation_reason`` for a cancel, "
            "``budget_stop_reason`` for the cost cap and the wall-clock budget), so a non-empty list "
            "means a fault, and its length is a count an operator can triage on."
        ),
    )
    created_at: str = Field(default_factory=utc_now_iso)
    completed_at: str | None = Field(default=None)
    archived: bool = Field(
        default=False,
        description=(
            "Curation state, orthogonal to ``status``: an archived run is excluded from every "
            "cohort a reporting or analysis surface aggregates over, while staying individually "
            "readable. It is how an observation known to be junk — a cancelled run, a run whose "
            "apparatus was broken — stops contaminating the analysis unit it sits in without "
            "being destroyed. ``status`` says how the run ENDED and is written by the runner; "
            "this says whether an operator still wants it counted, and only an operator writes it."
        ),
    )

    @property
    def expected_cells(self) -> int:
        """The matrix this run committed to at launch — ``cases × k_runs``, at its one candidate model.

        The one definition of a run's promised cell count, read by the completeness
        counters and by the read surfaces that need a denominator before the loop has
        reported one. It is computed from the RUN document, so it states what the run
        promised rather than what the loop was handed; the two can disagree, and
        :class:`RunCompleteness` exists to make that disagreement visible.

        Returns:
            The promised cell count. Never zero — ``test_case_ids`` is ``min_length=1`` and
            ``k_runs`` is ``ge=1``.
        """
        return len(self.test_case_ids) * self.k_runs

    @property
    def attribution_state(self) -> JudgeAttributionState:
        """What this run can say about which model scored each dim.

        See :func:`attribution_state` for why the rule lives in one place.
        """
        return attribution_state(self.effective_judges, self.effective_judges_source)

    @property
    def hashable_effective_judges(self) -> dict[str, str] | None:
        """Per-dim attribution when it may enter an identity key, else ``None``.

        The gate is the SOURCE, not whether the map is set. A reconstruction infers what
        scored each dim from stored records rather than recording it, so folding one
        into a key would let a backfill buy a run into the namespace where runs pool
        as repetitions of a single condition — an inferred condition made
        indistinguishable from a measured one. Both the context key and the
        comparison badge read this, so the two cannot come apart about what a
        reconstruction is allowed to assert.
        """
        return self.effective_judges if self.attribution_state == "recorded" else None

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "eval_run":
            raise ValueError(f"doc_type must be 'eval_run', got '{v}'")
        return v

    @model_validator(mode="after")
    def _replay_names_its_corpus(self) -> Self:
        """Require a replay to name the corpus it serves, and refuse one on any other run.

        Raises:
            ValueError: A replay run names no corpus, or a capture or cassettes-off run names one.
        """
        if (self.cassette_corpus_id is not None) != (self.cassette_mode == "replay"):
            raise ValueError(
                f"cassette_corpus_id is set exactly when cassette_mode is 'replay' (got mode {self.cassette_mode!r}, "
                f"corpus {self.cassette_corpus_id!r}): a replay serves the corpus of the capture run it names, a "
                "capture writes the corpus its own id names, and a run with cassettes off reads none"
            )
        return self


def goal_check_proofs_as_read(run: EvalRun) -> dict[str, GoalCheckProof] | None:
    """A run's goal-check proofs as every surface reads them: a ``proven`` from an older proof rule is ``unproven``.

    ``proven`` is the one proof that reads as measuring the behaviour, so it must have been earned under the rules
    in force (:data:`GOAL_CHECK_PROOF_RULES`). A run stamped before them may hold a ``proven`` its control earned
    under a parameter type no case can carry (#665). It cannot be re-derived when read — the controls are editable
    and were not frozen with the run, so a re-derivation would prove the template as it is now, not the one the
    run graded — so it is read as ``unproven`` until the template is launched again.
    ``refuted`` and ``unproven`` stand: an older rule never made a check look worse than it is.

    Args:
        run: The run.

    Returns:
        The proofs as read, or None when the run recorded none.
    """
    if run.goal_check_proofs is None:
        return None
    if (run.goal_check_proof_rules or 1) >= GOAL_CHECK_PROOF_RULES:
        return dict(run.goal_check_proofs)
    return {check: "unproven" if proof == "proven" else proof for check, proof in run.goal_check_proofs.items()}


def stale_goal_check_proofs(run: EvalRun) -> list[str]:
    """The checks whose ``proven`` the run recorded under an older proof rule — read as unproven, needing re-proof.

    Args:
        run: The run.

    Returns:
        The checks, sorted; empty when the run's proofs are current or it recorded none.
    """
    if run.goal_check_proofs is None or (run.goal_check_proof_rules or 1) >= GOAL_CHECK_PROOF_RULES:
        return []
    return sorted(check for check, proof in run.goal_check_proofs.items() if proof == "proven")


def refused_goal_checks(checks: Sequence[str]) -> dict[str, str]:
    """The goal checks the current grammar refuses, each with the refusal's reason.

    The grammar refuses at authoring, but a template stored before a rule keeps the check it now refuses, and
    grading it raises in every cell. Read at launch, so the run grades its cells without the check and names it.

    Args:
        checks: The template's goal checks.

    Returns:
        Each refused check -> why; empty when the grammar reads every one.
    """
    refused: dict[str, str] = {}
    for check in checks:
        try:
            parse(check)
        except DSLError as refusal:
            refused[check] = str(refusal)
    return refused


class EvalRunStamp(EvalDocumentModel):
    """What a reader of a run's place in a campaign reads off it: which run, curated out or not, and when.

    Read in place of a whole run where those three are all a surface needs — the campaign view
    derives its window from member runs' start times and names the archived ones, and a run
    document is mostly the host's frozen payload. Built from a stored row, so it is on the
    document base and hydrated through ``from_dict`` like every other stored eval model. A
    contract rather than storage's own type because the campaign view's store port promises it,
    and that port is declared in a package that may not import the run package.
    """

    id: str
    archived: bool = EvalRun.model_fields["archived"].default
    created_at: str


# =============================================================================
# LatencyMetrics — per-result wall-clock decomposition
# =============================================================================


class LatencyMetrics(EvalDocumentModel):
    """Per-result latency decomposition: harvested OTel spans, plus what they miss.

    The price/performance verdict needs a performance-time axis alongside
    ``cost_usd``. The span-derived components are wall-clock milliseconds summed
    across the test case's spans, partitioned by ``gen_ai.operation.name``:

    - ``total_ms`` — Σ of the ``agent.invoke`` turn-root spans (one per
      simulator turn, top-level and non-overlapping → total candidate
      processing time). Judging runs after the harvest closes, so it is
      excluded here and carried by ``judge_ms`` below.
    - ``llm_ms`` — Σ of the ``llm.call`` round spans. The
      **model-attributable** latency: tool time is model-independent, so
      this is the axis to compare when picking a model.
    - ``tool_ms`` — Σ of the ``tool.execute`` spans.

    ``llm_ms`` and ``tool_ms`` nest under the turn spans, and both the round
    loop and the action loop are serial, so the two are a genuine sub-partition
    of ``total_ms`` rather than overlapping stretches — see the exact split at
    the bottom of this docstring. Computed by the host's
    :class:`~threetears.evals.contracts.host.traces.TraceSink` from the same spans it renders
    into its own trace document. The runner reads the
    three off the record it is handed and never sees a raw span, so a cell run
    with no sink wired carries all three as ``None``.

    - ``async_wait_ms`` — the odd one out, and deliberately so: wall-clock the
      runner spent WAITING on in-flight background tool work between turns,
      measured off a monotonic clock rather than harvested from a span. Async
      tool work runs on its own detached trace root, precisely so a ~100s
      background run cannot inflate turn latency, and the runner's wait for it
      sits outside every ``agent.invoke`` span — so nothing timed it at all
      until this field. It is **not** part of ``total_ms`` and must never be
      subtracted from it; the measure registry records that by leaving its
      ``contained_by`` unset while ``llm_ms`` and ``tool_ms`` declare
      ``total_ms``.
    - ``judge_ms`` — clock-measured like the drain wait above, and disjoint from
      ``total_ms`` for the same reason it is: the judge scores a
      finished transcript after the harvest closes, so it falls outside every
      turn-root span rather than inside one. It is timed rather than harvested
      because the judge emits no spans at all — ``llm.call`` spans come from the
      candidate's agent loop, so a wider harvest window would collect nothing from
      it, and instrumenting the judge would instead blend judge-model time into
      ``llm_ms``, which exists to rank *candidate* models. Never subtract it from
      ``total_ms``; ``contained_by`` is left unset to say so.

    **Each span-derived component is ``None`` when no span contributed to it**, which is
    not the same fact as a measured zero and must not be averaged, ranked, or
    rendered as one. A cell can harvest an ``llm.call`` span while producing
    no ``agent.invoke`` turn-root — the candidate failing outside the turn
    wrapper — leaving a known ``llm_ms`` beside an unknown ``total_ms``. On
    axes where lower is better, a zero standing in for "unmeasured" wins every
    comparison it should have been excluded from.

    Components are independently nullable because they are independently
    sourced. A caller wanting "did this result measure anything at all" should
    check ``EvalResult.latency is None``, which stays the marker for a cell that
    produced no timing whatever — no span landing in a latency bucket, no drain window,
    and no judge phase. Any one measured component alone is enough to build this record,
    and it is the measurement that builds it, not the harvest: spans that all fall outside
    these buckets — a background tool's enqueue, its background task's own root — leave the
    marker intact rather than persisting a carrier whose every field is unmeasured. The check
    reads both ways, which is what makes it usable as the check.

    **The relevance classifier is deliberately NOT in that list, though it reads like it
    belongs.** Its span carries ``agent.invoke``, so it lands squarely in the ``total_ms``
    bucket rather than outside every one, and harvesting it would count a standalone
    classification as a candidate turn. What keeps it out is not its operation name but
    its identity: the sink narrows the harvest to spans stamped with this **cell
    execution's** own nonce — not with its run and test case, which every sibling cell of the
    same run shares — and the classifier runs on the host's own path where no eval
    context is set at all.

    ``total_ms`` partitions exactly into ``llm_ms + tool_ms`` plus a named remainder —
    ``threetears.evals.analysis.reporting.decompose_total_ms``, which derives it rather than reading a
    fourth stored field, since these three already settle it.
    """

    total_ms: Annotated[float, Field(ge=0.0)] | None = None
    llm_ms: Annotated[float, Field(ge=0.0)] | None = None
    tool_ms: Annotated[float, Field(ge=0.0)] | None = None
    #: Clock-measured, not span-derived, and disjoint from ``total_ms`` — see the
    #: class docstring. A real ``0.0`` when a drain ran with nothing in flight;
    #: ``None`` when the cell never reached one.
    async_wait_ms: Annotated[float, Field(ge=0.0)] | None = None
    #: Wall-clock of the judge phase, clock-measured and likewise disjoint from
    #: ``total_ms``. Present whenever the phase ran — **including one that
    #: errored**, since that time was really spent and ``EvalResult.judge_error``
    #: is what says the scoring failed. ``None`` when no judge phase was timed:
    #: the run had no judge service, or the cell had no transcript to judge. A cell that died before judging (candidate
    #: factory failure, cell timeout) carries no ``LatencyMetrics`` at all. The
    #: phase's calls run concurrently (bounded by ``eval.judge_concurrency``), so
    #: this is the phase's elapsed time, not the sum of its calls — two runs scored
    #: at different widths compare on their scores, never on this figure.
    judge_ms: Annotated[float, Field(ge=0.0)] | None = None


#: Where one piece of background work stood when the cell ended. ``delivered`` — its payload
#: reached the conversation; ``failed`` — the work ended in an error and delivered nothing;
#: ``undelivered`` — it was acknowledged and the cell ended before it arrived.
AsyncDeliveryStatus = Literal["delivered", "failed", "undelivered"]


class AsyncExternalSpend(EvalDocumentModel):
    """Paid non-LLM calls one piece of background work made at one provider, as the work reports them.

    The caller reports and the engine prices: these are the calls, the provider's own units, and —
    only where the provider itself billed a figure — what it charged. A charge the provider reported
    (``money``) is an observation and wins over any rate; otherwise the runner prices the units at the
    run's declared rate for exactly this ``(provider, unit)``
    (:meth:`~threetears.evals.contracts.spend.ExternalRateTable.money_for`), and leaves them counted and
    unpriced — unknown, never ``0`` — where neither exists. :meth:`as_external_spend` is the one
    conversion every reader takes, so no reader can rebuild the report and drop a field of it.
    """

    provider: str | None = Field(description="Who was called; None only when the work could not say")
    calls: int = Field(ge=1, description="How many calls reached the provider")
    provider_units: int | None = Field(
        default=None, ge=0, description="The provider's own metered units, where the work could state them"
    )
    provider_unit: str | None = Field(
        default=None, description="The bare name of that unit ('credits'); set exactly when provider_units is"
    )
    money: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "What the provider charged for these calls, in USD, as the provider itself reported it. None when it "
            "reported no charge — unpriced, never 0; a declared rate may still price the units. 0.0 is a real, "
            "reported zero."
        ),
    )

    def as_external_spend(self) -> ExternalSpend:
        """This report in the metering seam's vocabulary — every field, the provider's charge included.

        The single conversion the readers of a delivery's spend take (the usage fold and the cell's cost),
        so the two cannot disagree about what the work reported.

        Returns:
            The same report as an :class:`~threetears.evals.contracts.host.spend.ExternalSpend`.
        """
        return ExternalSpend(
            provider=self.provider,
            calls=self.calls,
            provider_units=self.provider_units,
            provider_unit=self.provider_unit,
            money=self.money,
        )

    @model_validator(mode="after")
    def _unit_named_with_its_count(self) -> Self:
        """Refuse a unit count without its unit's name, or a name without a count.

        Raises:
            ValueError: ``provider_units`` and ``provider_unit`` are not set together.
        """
        if (self.provider_units is None) != (self.provider_unit is None):
            raise ValueError("provider_units and provider_unit are reported together or not at all")
        return self


class AsyncDelivery(EvalDocumentModel):
    """One piece of background work the candidate started: acknowledged at once, delivered later.

    A candidate can hand work to a tool that answers immediately ("on it") and reports back turns
    later, from a background task running on a model of its own — a scout sent ahead while the
    conversation carries on. Nothing that work spends or takes happens inside the turn that asked
    for it, so the kind records each one here and the engine stores the list on
    :attr:`EvalResult.async_deliveries`. Every field is engine vocabulary: who asked, when it was
    acknowledged and delivered, what it ran on, how long it took, what it delivered, and whether a
    harness supplied the payload instead. What the tool's own work IS — its stop reasons, its
    internal phases — is the kind's, and travels on :attr:`EvalResult.kind_payload`.

    **The turn fields are positions, and the measures are the gap and the clock.** ``acknowledged_turn``
    and ``delivered_turn`` place the delivery in the conversation, 0-based over its turns, and are
    ``None`` for a kind that does not converse. ``elapsed_ms`` is wall-clock from acknowledgement to
    delivery (or to failure).

    **``substituted`` is required, because the two arms are different evidence.** A delivery whose
    payload a harness supplied — a seed the case carries, or a replayed capture — describes no run
    of the tool at all: no model chose anything and nothing was spent. A kind must say which arm
    each delivery took; defaulting to "live" would let a forgotten flag report a seeded run's cost
    as production's. A replayed one is held to it by the engine, which knows what it replayed.

    **The work's spend travels on the entry, whatever its status.** The tokens, dollars and calls the
    background work spent on its own model, and the paid calls it made elsewhere, are reported here
    and folded by the runner into the ``inner_agent`` and ``external`` roles — for work that was still
    in flight when the cell ended as much as for work that delivered, since it spent either way. Each
    is ``None`` (or empty) when unreported, never ``0``. A substituted entry spent nothing and may
    report no spend at all.
    """

    tool: str = Field(min_length=1, description="The tool that did the work, as the kind names it")
    requested_by: str | None = Field(
        default=None,
        description=(
            "Who asked for the work — the candidate, or the participant it acted for — in the kind's own "
            "names. None when the kind does not tell requesters apart."
        ),
    )
    status: AsyncDeliveryStatus = Field(description="Where the work stood when the cell ended")
    model: str | None = Field(
        default=None,
        description=(
            "The model the background work ran on, when it ran on one. None for a substituted delivery, and "
            "for work no model did."
        ),
    )
    acknowledged_turn: int | None = Field(
        default=None,
        ge=0,
        description="The turn whose call the tool acknowledged; None for a kind that does not converse",
    )
    delivered_turn: int | None = Field(
        default=None,
        ge=0,
        description="The turn the payload reached the conversation on; set only on a delivered entry of a conversing kind",
    )
    elapsed_ms: float | None = Field(
        default=None, ge=0.0, description="Wall-clock from acknowledgement to delivery or failure; None when unmeasured"
    )
    delivered_items: int | None = Field(
        default=None,
        ge=0,
        description=(
            "How many items the payload carried, for a tool whose payload counts. Set only on a delivered "
            "entry; 0 is a real delivery of nothing."
        ),
    )
    summary: str | None = Field(
        default=None, description="What was delivered, in a line a reader can scan; set only on a delivered entry"
    )
    error: str | None = Field(default=None, description="Why the work failed; set exactly when status is failed")
    substituted: bool = Field(
        description="True when a harness supplied the payload (a seed or a replayed capture) instead of a live run"
    )
    input_tokens: int | None = Field(
        default=None, ge=0, description="Prompt tokens the work spent on its own model; None when unreported"
    )
    output_tokens: int | None = Field(
        default=None, ge=0, description="Generated tokens the work spent on its own model; None when unreported"
    )
    reasoning_tokens: int | None = Field(
        default=None, ge=0, description="The reasoning subset of output_tokens, where the provider split it"
    )
    llm_calls: int | None = Field(
        default=None, ge=0, description="How many model calls the work made; None when unknown, never one by default"
    )
    cost_usd: float | None = Field(
        default=None,
        ge=0.0,
        description="What the work's model calls cost, as its client reported; None when unreported",
    )
    price_source: str | None = Field(
        default=None, description="Where cost_usd came from, as the work's client named it; set only with cost_usd"
    )
    external_spend: list[AsyncExternalSpend] = Field(
        default_factory=list, description="Paid non-LLM calls the work made, one entry per provider and unit"
    )

    @model_validator(mode="after")
    def _one_story(self) -> Self:
        """Refuse an entry whose fields tell two different stories about the same work.

        Raises:
            ValueError: An error without a failure or a failure without one; a delivery's own
                fields on an entry that delivered nothing; a delivery before its acknowledgement;
                a substituted entry naming the model it supposedly ran on or reporting spend; or a
                price source for a cost nobody reported.
        """
        defects: list[str] = []
        if (self.error is not None) != (self.status == "failed"):
            defects.append("error is set exactly when status is 'failed'")
        if self.status != "delivered":
            delivered_only = {
                "delivered_turn": self.delivered_turn,
                "delivered_items": self.delivered_items,
                "summary": self.summary,
            }
            defects.extend(
                f"{name} is set on a {self.status!r} entry, which delivered nothing"
                for name, value in delivered_only.items()
                if value is not None
            )
        if (
            self.acknowledged_turn is not None
            and self.delivered_turn is not None
            and self.delivered_turn < self.acknowledged_turn
        ):
            defects.append(f"delivered_turn {self.delivered_turn} precedes acknowledged_turn {self.acknowledged_turn}")
        if self.substituted and self.model is not None:
            defects.append(f"a substituted delivery ran on no model, yet names {self.model!r}")
        if self.substituted and (spent := self.reported_spend):
            defects.append(f"a substituted delivery spent nothing, yet reports {', '.join(spent)}")
        if self.price_source is not None and self.cost_usd is None:
            defects.append("price_source names where cost_usd came from, and cost_usd is unreported")
        if defects:
            raise ValueError(f"async delivery from {self.tool!r}: " + "; ".join(defects))
        return self

    @property
    def reported_spend(self) -> list[str]:
        """The spend fields this entry reports, by name — empty when it reports none."""
        named = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "llm_calls": self.llm_calls,
            "cost_usd": self.cost_usd,
        }
        spent = [name for name, value in named.items() if value is not None]
        if self.external_spend:
            spent.append("external_spend")
        return spent


# =============================================================================
# RoleUsage — per-role token + cost observation
# =============================================================================


#: The five roles an eval cell can spend on. Named once because two fields carry the
#: vocabulary — the role a usage row observed, and the roles a blended total was summed
#: over — and a second copy of a closed set is a copy that can go stale in one place.
UsageRole = Literal["candidate", "judge", "simulator", "inner_agent", "external"]


#: What one simulator-role call was for: an actor's line, or the ``llm_decided`` scheduler's pick of
#: who speaks next. Carried on the ``simulator`` role's :class:`RoleUsage` rows.
SimulatorPurpose = Literal["utterance", "schedule"]


class RoleUsage(EvalDocumentModel):
    """Per-role token + cost observation for one :class:`EvalResult`.

    One row per (role, model) that spent tokens or dollars producing a result.
    All five roles are captured: candidate and simulator from their own LLM
    results, judge from every judge call including the failed ones, and
    inner_agent / external from the async delivery a background task carries
    back. Tokens are the durable observation; ``cost_usd`` is the
    price-at-observation figure and ``price_source`` records where it came from
    so dollars stay re-derivable at current rates. The source is what the
    caller reported — a completion client names its own — and the engine supplies none.

    The list on ``EvalResult.usage`` decomposes into **production-replicating**
    (candidate + inner_agent + external) vs **program** (all roles) cost by role
    MEMBERSHIP — there is deliberately no stored scope field, and the
    name ``attribution_scope`` is reserved for the measure registry's own axis.

    Every token field is ``int | None`` and ``cost_usd`` is ``float | None`` on
    purpose: **missing is not zero.** A role that reported no reasoning split
    carries ``None`` reasoning — coercing it to 0 would fabricate an observation.
    A genuine zero (a non-reasoning model reporting 0 reasoning tokens) stays 0.

    Rows are keyed by **(role, model, served model, price source)**, not role alone: a role that spent
    tokens on more than one model — a run whose per-dim judge configs pin different
    models, say — contributes one row per model, because blending them would have
    to drop ``model`` and with it the ability to re-derive the dollars. Dollars priced
    two ways stay in two rows for the same reason, and so do calls one requested alias had
    answered by two different models: ``served_model`` is the evidence of which model produced
    the numbers, and a blended row could name neither.

    The ``external`` role (paid non-LLM APIs, e.g. web search) has no token
    concept at all: it reports ``call_count`` and — where the caller could count
    them — ``provider_units`` with ``None`` tokens. Those providers mostly return
    no per-call dollar figure, so dollars are reachable only through an
    operator-declared rate per ``(provider, unit)``: an input absent by default.
    Declared, ``cost_usd`` is filled and ``price_source`` says the rate was
    configured rather than provider-reported; undeclared, the volume is still
    observable and the spend stays honestly unknown. What must never happen is a
    guessed constant standing in for the rate: that would put an estimate nobody
    made behind ``price_source``, which exists to name a real provenance.

    **An external row is keyed by ``(role, model, provider, provider_unit, price source)``, and
    that is a guarantee rather than a detail.** Two providers' weighted units are
    not one quantity — search credits and video-API quota units added together are
    a fabricated number, not a coarser true one — so they land in two rows and
    there is nowhere for the addition to happen. Downstream code cannot forget
    the rule because it never gets the chance.
    """

    role: UsageRole
    model: str | None = Field(
        default=None,
        description=(
            "Model slug this role's tokens were attributed to, for spend; None when not model-attributable (e.g. an "
            "external API). A client may fill it from the REQUEST, so for a floating alias it names the alias, not "
            "the model that answered — that is served_model."
        ),
    )
    served_model: str | None = Field(
        default=None,
        description=(
            "The model the provider's RESPONSE named as having answered this row's calls, never the id requested. "
            "None = not recorded: the responses named no model, or the row was stored before this was recorded. "
            "Never read the alias in `model` in its place."
        ),
    )
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Reasoning portion of completion_tokens; None when the role reported no split (missing != zero).",
    )
    cost_usd: float | None = Field(
        default=None, ge=0.0, description="Price-at-observation cost for this role; None when no cost was observed."
    )
    price_source: str | None = Field(
        default=None,
        description="Provenance of cost_usd so dollars are re-derivable (e.g. a provider-reported price). None when no cost was observed.",
    )
    call_count: int | None = Field(
        default=None,
        ge=0,
        description="How many calls this role made; None when not counted.",
    )
    provider: str | None = Field(
        default=None,
        description=(
            "WHO was called — a search API, a video API, as the host names it. A different field from price_source, "
            "which is the provenance of a PRICE. None on every token-metered role, and on an "
            "external contribution whose caller could not say."
        ),
    )
    provider_unit: str | None = Field(
        default=None,
        description=(
            "The bare name of the unit provider_units counts, as the provider calls it "
            "('credits', 'quota_units'). None whenever provider_units is."
        ),
    )
    provider_units: int | None = Field(
        default=None,
        ge=0,
        description=(
            "The provider's own weighted metering this role consumed, summed from what each "
            "caller REPORTED and never re-derived from a rate. Comparable ONLY within one "
            "(provider, provider_unit). None on token-metered roles, and wherever the count was "
            "not knowable — never 0, which would claim a free call."
        ),
    )
    actor_id: str | None = Field(
        default=None,
        description=(
            "The simulated actor this simulator row's calls spoke for, or — for scheduling picks — chose to "
            "speak next. None on every other role, and on scheduling picks that chose nobody (the round "
            "ending, a refused reply)."
        ),
    )
    purpose: SimulatorPurpose | None = Field(
        default=None,
        description=(
            "Which simulator calls this row counts: an actor's 'utterance', or a 'schedule' pick by the "
            "llm_decided turn scheduler. None on every other role."
        ),
    )

    @model_validator(mode="after")
    def _simulator_attribution_is_the_simulators(self) -> RoleUsage:
        """Refuse an actor or a purpose on a row of any role but ``simulator``, which alone has them."""
        if (self.actor_id is not None or self.purpose is not None) and self.role != "simulator":
            raise ValueError(f"actor_id and purpose attribute simulator calls, not the {self.role!r} role's")
        return self


# =============================================================================
# EvalResult — one test case × one model × one k-iteration
# =============================================================================


#: How a cell's execution ended — the branch the runner took, recorded because the record
#: cannot recover it.
#:
#: This is the one axis of a result's condition that is an OBSERVATION rather than a
#: derivation, which is why it is stored while the judging and scoring axes are resolved at
#: read (:func:`threetears.evals.contracts.result_condition.resolve_result_condition`). Every stored
#: alternative is a *consequence* of the branch rather than the branch: ``usage == []``
#: separates a factory failure from a completed cell only while no completed cell attributes
#: zero roles, and a timed-out cell's rows look like any other cell's. A label derived from a
#: correlate is right until the correlation breaks and silently wrong after.
#:
#:   ``completed``       the cell ran to the end of :func:`~threetears.evals.run.runner.run_one_result`
#:                       — including one that errored, since a candidate whose LLM failed still
#:                       completed the cell. What it says is that nothing was cancelled and no
#:                       measurement was lost.
#:   ``factory_failed``  the candidate factory raised before any candidate turn. Nothing spent, so
#:                       the zero total is a real sum over the run's roles.
#:   ``cell_timeout``    the cell was cancelled from outside on its deadline. Any kind's billed
#:                       spend it had reported through the cell's sink is kept, with the judge
#:                       calls that returned — but the call in flight at the
#:                       deadline never reported, so the total and the rows stop short of it.
#:                       Whose failure it is follows what was pending, on the error fields.
#:   ``seed_failed``     the eval apparatus failed seeding the world — after the factory
#:                       returned a candidate, before any candidate turn. Shares
#:                       ``factory_failed``'s properties (nothing spent, so the zero total is a
#:                       real sum) but not its cause, and the disclosure names the cause: an
#:                       operator told "the candidate factory failed" about a malformed world seed
#:                       seed goes looking in the wrong place. That is the whole reason this is a
#:                       fourth arm rather than a reuse of the third.
#:   ``precondition_failed``
#:                       a precondition the template declared did not hold when the world was
#:                       seeded. Shares the other two pre-turn arms' cost properties and is
#:                       distinct from both in what it says about the apparatus: nothing is
#:                       broken. The rig worked and the world was not the one the probe presumes,
#:                       which makes the observation invalid rather than the harness faulty — and
#:                       an operator sent looking for a bug would not find one. The result's
#:                       ``precondition_outcomes`` name which presumption and what it presumed.
#:   ``apparatus_failed``
#:                       an :class:`~threetears.evals.contracts.host.apparatus.ApparatusError` — a
#:                       replay miss, a corrupt recording, a harness fault — reached the engine
#:                       out of the kind's ``prepare`` or ``invoke``. The cell is excluded (an
#:                       ``apparatus:`` infra error) and the run goes on to the next one. Like
#:                       ``cell_timeout`` it keeps what the kind had reported through the cell's
#:                       sink, and like it the total stops short of whatever was in flight when the
#:                       fault unwound the kind.
#:   ``cancelled``       the run was cancelled while this cell ran. Recorded rather than dropped
#:                       because its spend was real: what the kind had reported, and the judge calls
#:                       that returned, with the same "at least this much" caveat as a deadline.
#:                       Excluded — an infra error names the cancel, or, when the cancel struck the
#:                       judge phase, the dims it cut off are the cell's judge errors.
#:
#: **Storing this is a deliberate departure from the project's derive-don't-store preference**
#: (a `@property` over a persisted field, so the two can never disagree), and the reason is the
#: paragraph above rather than convenience: a derived value would have to read a correlate, and
#: the whole defect this field closes is a read surface reasoning from correlates. Where a
#: derivation IS honest — the judging and scoring axes, which read the judge's own outputs and
#: the error taxonomy — nothing is stored, and that asymmetry is the point.
#:
#: Required on every result: the runner states one at every exit, so it is never inferred from
#: the correlates above.
CellTermination = Literal[
    "completed",
    "factory_failed",
    "cell_timeout",
    "seed_failed",
    "precondition_failed",
    "apparatus_failed",
    "cancelled",
]


class ConversationStopCause(StrEnum):
    """Why a conversation trial's turn loop stopped — one structural signal, never a reading of prose.

    A different fact from :data:`CellTermination`. Termination says how the CELL ended as work
    (ran to the end, was cancelled, was excluded before any turn); this says why the CONVERSATION
    inside a cell that ran stopped taking turns. A conversation that stopped on a rig fault still
    has ``termination == "completed"``, because the cell ran to its end and recorded the fault.

    Three members end a conversation on its own terms, and they are the only ones:

      ``max_turns``           the template's turn budget was spent.
      ``user_done``           every simulated actor said, in its structured reply, that it had nothing
                              more to say. Each one leaves when it says so, and a closing reply is
                              never delivered to the candidate.
      ``participants_ended``  the session's REAL participants ended it — a witnessed session, which a
                              host observed rather than an engine ran, stopped because the people in
                              it stopped. Its own member rather than ``user_done``, which says a
                              simulator decided, and a witnessed session has no simulator; only
                              :func:`~threetears.evals.run.record_witnessed_cell` records a cell
                              that has no simulator, and it refuses the simulator's two causes.

    Three name a rig fault, and each also puts an infra error on the result, which excludes it:

      ``simulator_error``  the simulated user's call failed or its reply broke the schema.
      ``apparatus_error``  the eval apparatus failed during a turn or a delivery drain.
      ``candidate_error``  the candidate's turn raised out of the harness boundary.

    One is the run's own ceiling, and it too excludes the cell:

      ``budget_stopped``   the run's cost cap was reached mid-conversation, counting the simulator's spend
                           so far (:func:`~threetears.evals.run.conversation.drive_conversation`), so no
                           further paid call was made. The conversation is cut short rather than finished, so
                           the runner puts an infra error on the result, which excludes it, and the run stops
                           ``budget_stopped`` after the cell is saved. Never a judgement of the candidate.

    There is deliberately no member for a refusal. Whether the candidate refused well is a
    judged property of the transcript, and stopping on the words of a refusal truncated exactly
    the conversations whose refusals the rubric exists to grade. Nor is there one for goal checks
    holding: they grade the end state, and one that holds early ("never called X") would stop a
    pressure conversation at its first refusal, the same truncation by another route.
    """

    MAX_TURNS = "max_turns"
    USER_DONE = "user_done"
    PARTICIPANTS_ENDED = "participants_ended"
    SIMULATOR_ERROR = "simulator_error"
    APPARATUS_ERROR = "apparatus_error"
    CANDIDATE_ERROR = "candidate_error"
    BUDGET_STOPPED = "budget_stopped"


class JudgedArtifact(StrEnum):
    """What a judge reads of a kind's output — the kind's declaration, which picks the judged axes.

    Declared by the kind rather than inferred from what one cell happened to hand back, because
    the axes differ by artifact and a guess fails silently: a document scored on the transcript
    and outcome axes is scored against turns it does not have. With the declaration, a cell that
    contradicts it is a :class:`~threetears.evals.contracts.candidate_kind.CandidateKindDefect`
    and nothing is judged.
    """

    TRANSCRIPT = "transcript"
    """A conversation: the transcript, outcome and rubric axes, each read off the kind's :class:`JudgeEvidence`."""

    DOCUMENT = "document"
    """One document judged against its case material: the rubric axes alone, off the kind's :class:`JudgeEvidence`."""

    UNJUDGED = "unjudged"
    """Nothing a judge reads — the kind's grade is code (``host_measures``); a judged run refuses it."""


class JudgeEvidence(BaseModel):
    """Everything a judge reads about one cell's candidate, rendered by the kind that ran it.

    The engine renders none of it and reads none of it. A judge needs to know what it is
    scoring, what the candidate was working from, and what the candidate produced — and each of
    those is shaped by the kind alone: who the candidate is, which of the facts in play the
    judge should see (a game master's hidden trap, a whisper only one player heard), and how a
    turn, a tool call or a page reads as text. So a judged kind hands back all three as strings,
    and this one object is both what the judge is sent and what a re-judge later re-sends,
    because it is stored on the cell's :class:`EvalTrace`.

    Every judged kind — :attr:`JudgedArtifact.TRANSCRIPT` and :attr:`JudgedArtifact.DOCUMENT`
    alike — renders one for every non-empty output, and an unjudged kind renders none. The
    declaration, not this object, decides which axes are scored.

    Plain ``BaseModel`` rather than the eval base: every field is rendered text, whose
    whitespace is part of what the judge reads.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    subject: str | None = None
    """Who or what is under test, as the judge should know it — the candidate's brief, its style,
    its standing instructions. ``None`` when the artifact speaks for itself, as a document
    judged against its source does."""

    case_material: str = Field(min_length=1)
    """What the candidate's output is judged against: the source a document was produced from,
    or the facts of a scenario the judge must know to score a conversation — including those
    the candidate's interlocutors never saw."""

    artifact: str = Field(min_length=1)
    """The candidate's output as the judge should read it: a document, or a conversation's
    transcript in whatever form the kind renders turns, tool calls and their results."""


def eval_trace_doc_id(result_id: str) -> str:
    """Doc id for a result's :class:`EvalTrace` sibling.

    Derived from the result id so a detail read can point-read the payload instead of
    querying for it. Suffixed rather than reusing the result id because both documents
    share a container and a partition, where an identical id IS the same row.

    **This does not make a re-run idempotent.** ``EvalResult.id`` is a fresh ``uuid7`` per
    cell, so re-running the same (test case, model, k) mints a new result and a new payload
    beside it rather than replacing the old pair — which is correct, since the two runs are
    two observations. Idempotency here is only per result id: writing the same result twice
    overwrites rather than duplicates.
    """
    return f"{result_id}:trace"


class EvalTrace(EvalDocumentModel):
    """The candidate's output, what its judge read, and the OTel spans — stored beside a result, not inside it.

    ``trace`` carries the candidate's output exactly as its kind handed it back (for a
    conversation, its turns; for a document, the document); ``otel_trace`` carries the harvested
    spans. Both are lists of dicts rather than typed shapes on purpose — the engine never looks
    inside one, and pinning a shape here would make every kind's output a schema change.

    ``judge_evidence`` is what the judge was sent, as the kind rendered it, and
    ``judged_artifact`` the kind's declaration that picked the axes. They are stored because a
    re-judge must ask the same question the first judge was asked: rebuilding the evidence from
    ``trace`` would need the kind's renderer, and a renderer that has changed since — or a kind
    no longer wired — would score a different input under the old result. Both are set together
    or not at all: a cell whose kind is unjudged, or whose candidate produced nothing, has none.

    **They live in their own document because of what they weigh.** The two dominate a
    result's size — in an illustrative 120 kB ``eval_result``, a 90 kB ``trace`` and a
    21 kB ``otel_trace`` are 92% of the row — and no list, aggregate or cost path
    reads them. Splitting them out is what makes a scope-wide
    aggregate read tens of MB instead of GB.

    ``id`` is :func:`eval_trace_doc_id` of the result's id — derived, not random, so
    a detail read is a point read on the result's own partition and the write is
    idempotent **per result id**: writing the same result twice overwrites its payload
    instead of duplicating it. That is not re-run idempotency, and the difference
    matters to anyone reasoning about row growth here — re-running the same (test case,
    model, k) mints a fresh ``EvalResult.id``, so it writes a new payload beside the old
    pair rather than replacing it, which is correct, since the two runs are two
    observations. It is **not** the bare result id: both documents live in the same
    container, so reusing it would collide. ``result_id`` carries the link explicitly
    rather than leaving it to be parsed back out of the key.
    """

    id: str = Field(min_length=1)
    doc_type: Literal["eval_trace"] = "eval_trace"
    schema_version: SchemaVersion = EVAL_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)
    #: The result this belongs to. Stored rather than derived from ``id`` so the
    #: document answers "whose is this" on its own.
    result_id: str = Field(min_length=1)
    #: Read by the run-scoped delete sweeps (a host's orphan cleanup), which
    #: reclaim payloads BY RUN. That is deliberately not derived per result id: a payload
    #: whose result was already deleted is unreachable any other way, and it is the largest
    #: row in the container.
    eval_run_id: str = Field(min_length=1)

    trace: list[dict[str, Any]] = Field(default_factory=list)
    otel_trace: list[dict[str, Any]] = Field(default_factory=list)
    #: What the judge was sent for this cell, as its kind rendered it. ``None`` when nothing was
    #: rendered for a judge: an unjudged kind, or a candidate that produced nothing.
    judge_evidence: JudgeEvidence | None = None
    #: The kind's declaration that picked the judged axes, stored with the evidence it applies to.
    judged_artifact: JudgedArtifact | None = None
    #: The calls the candidate made that succeeded, as its kind recorded them and graded its goal
    #: checks against. Stored so a re-check re-grades from what the candidate did rather than from
    #: the verdicts alone. ``None`` when the kind keeps no ledger.
    call_ledger: CallLedger | None = None
    #: The world the cell left behind, keyed by declared dimension name: every dimension of every carrier
    #: the cell attached, read back through its ``read`` handle once the kind's ``invoke`` returned (or
    #: earlier, when the kind read it to grade against — one read either way). Stored so a re-check
    #: re-grades a ``state.*`` check from what the world held rather than from the verdict alone.
    #: ``None`` when nothing was read: a host with no world, a kind that never seeded through its world
    #: session, or a cell cut off before its end state was read.
    end_state: VerbatimJsonObject | None = None

    @model_validator(mode="after")
    def _evidence_and_its_declaration_travel_together(self) -> Self:
        """Refuse evidence without the declaration that picks its axes, or the reverse.

        A re-judge reads both: evidence alone cannot say whether the transcript and outcome axes
        apply, and a declaration alone has nothing to send. An unjudged declaration has no
        evidence by definition, so it is never stored.
        """
        if (self.judge_evidence is None) != (self.judged_artifact is None):
            raise ValueError("judge_evidence and judged_artifact are stored together or not at all")
        if self.judged_artifact is JudgedArtifact.UNJUDGED:
            raise ValueError("an unjudged kind renders no judge evidence, so none is stored for it")
        return self


class JudgeRescore(EvalDocumentModel):
    """One re-judge of a result's failed judge dimensions, recorded on the result it changed.

    A judge call can fail after the trial it scores has finished — a reply cut off at its
    output cap, an unparseable score — and the cell stores that dimension as missing. A
    re-judge asks the same judge the same question again from the stored transcript, under
    the apparatus the run recorded, and writes any score it gets where the judge phase would
    have put it. This entry is what keeps that visible: without it a re-scored result is
    indistinguishable from one scored in a single pass.

    **Its spend is recorded here and nowhere else.** ``EvalResult.cost_usd`` and ``usage``
    measure the cell as it ran; a re-judge is apparatus spend paid later, and folding it in
    would make one cell's cost depend on whether an operator re-scored it.
    """

    rejudged_at: str = Field(default_factory=utc_now_iso)
    dims: list[DimName] = Field(min_length=1, description="The failed dims re-asked, in dimension order.")
    prior_judge_error: str = Field(
        min_length=1, description="The result's ``judge_error`` before this re-judge, verbatim."
    )
    scores: dict[DimName, int] = Field(
        default_factory=dict,
        description="dim -> the score the re-judge produced (1-5, or 1/0 on pass/fail), for each dim that scored.",
    )
    errors: dict[DimName, str] = Field(
        default_factory=dict, description="dim -> why it failed again, for each dim that did not score."
    )
    cannot_tell: dict[DimName, ModelProse] = Field(
        default_factory=dict,
        description="dim -> the judge's reason, for each dim it answered it could not score from the evidence.",
    )
    judge_model: str = Field(
        min_length=1, description="The run's judge pin, which scored every dim whose config names no model."
    )
    judge_config_ids: dict[DimName, str] = Field(
        default_factory=dict,
        description=(
            "dim -> the versioned JudgeConfig that answered. Absent = the built-in prompt scored it, or the "
            "call raised before the service could name a config (that dim then has an entry in ``errors``)."
        ),
    )
    usage: list[RoleUsage] = Field(default_factory=list, description="The re-judge's own judge-role usage rows.")
    cost_usd: float | None = Field(
        ge=0.0,
        description=(
            "What the re-judge cost. Not part of ``EvalResult.cost_usd``. None when any of its judge calls "
            "went unpriced: unknown, never zero."
        ),
    )


class RepeatedScore(EvalDocumentModel):
    """One dimension's stored judge score, asked again of the same judge from the same evidence.

    The pair a judge's self-agreement is read from: ``first_score`` is the score the result held
    when the repeat was asked, and the repeat's answer sits beside it. Both are carried here rather
    than the first read back off the result later, because a re-judge may rewrite the result's
    score afterwards, and a pair whose halves were read at different moments is not one pair.
    """

    dim: DimName = Field(min_length=1, description="The dimension asked again.")
    scale: RubricScale = Field(description="The scale the first score was on, and the repeat was asked on.")
    first_score: int = Field(description="The score the result held on the dimension when the repeat was asked.")
    first_served_model: str | None = Field(
        description=(
            "The model that served the first score, as its response named it; None when it named none. "
            "The judge whose self-agreement this pair measures."
        ),
    )
    first_judge_config_id: str | None = Field(
        description=(
            "The versioned JudgeConfig that asked for the first score, as the result recorded it; None = the "
            "built-in prompt. The rest of the judge's identity: a repeat answered under another config measures "
            "a different judge, and is not paired."
        ),
    )
    first_judge_temperature: JudgeTemperature | None = Field(
        default=None,
        description=(
            "The temperature the first score was sent at, as it recorded it; None when it recorded none. A repeat "
            "sent at another temperature — or beside a first score that recorded none — measures a different (or "
            "an unknown) judge, and is not paired."
        ),
    )
    repeat: RubricScore | None = Field(
        default=None, description="The repeat's score, when the judge scored the dimension again."
    )
    error: str | None = Field(default=None, description="Why the repeat call failed, when it did.")
    cannot_tell: ModelProse | None = Field(
        default=None, description="The judge's reason, when the repeat answered it could not score the dimension."
    )

    @model_validator(mode="after")
    def _one_answer_on_the_scale(self) -> Self:
        """Refuse a repeat with no answer or two, a first score off its scale, or a repeat on another dim or scale.

        Raises:
            ValueError: Not exactly one of ``repeat``, ``error`` and ``cannot_tell`` is set, ``first_score``
                is off ``scale``, or ``repeat`` scores another dimension or scale.
        """
        answers = [name for name in ("repeat", "error", "cannot_tell") if getattr(self, name) is not None]
        if len(answers) != 1:
            raise ValueError(f"a repeated score carries exactly one of repeat, error and cannot_tell; got {answers}")
        low, high = SCALES[self.scale].scores
        if not low <= self.first_score <= high:
            raise ValueError(f"first_score {self.first_score} is off the {self.scale} scale [{low}, {high}]")
        if self.repeat is not None and (self.repeat.dim != self.dim or self.repeat.scale != self.scale):
            raise ValueError(
                f"the repeat scored {self.repeat.dim!r} on {self.repeat.scale!r}, not {self.dim!r} on {self.scale!r}"
            )
        return self


class JudgeRepeat(EvalDocumentModel):
    """One repeat of a result's judge scores: the same judge asked the same question again, recorded beside them.

    A judge's agreement with ITSELF — re-scoring evidence it already scored, under the apparatus the run
    recorded — is what the ``separation`` evidence tier reads
    (:func:`threetears.evals.analysis.judge_self_agreement`). A repeat never changes the result's scores:
    it is a measurement OF the judge, so the scores the cell was judged with stay the ones every lens
    reads, and this entry is the only place the repeat's answers live.

    **Its spend is not here.** Each call a repeat makes is priced before it is made and written to the
    out-of-run ledger (:class:`~threetears.evals.contracts.out_of_run.OutOfRunSpend`, purpose ``judge``,
    stamped with the run), which is the one record of what it cost.
    """

    repeated_at: str = Field(default_factory=utc_now_iso)
    judge_model: str = Field(
        min_length=1, description="The run's judge pin, which scored every dim whose config names no model."
    )
    scores: list[RepeatedScore] = Field(
        min_length=1, description="One entry per dimension asked again, in dimension order."
    )
    judge_config_ids: dict[DimName, str] = Field(
        default_factory=dict,
        description="dim -> the versioned JudgeConfig that answered the repeat. Absent = the built-in prompt.",
    )

    @field_validator("scores")
    @classmethod
    def _each_dim_once(cls, scores: list[RepeatedScore]) -> list[RepeatedScore]:
        """Refuse a repeat asking one dimension twice — two answers to one question in one repeat."""
        dims = [score.dim for score in scores]
        if repeated := sorted({dim for dim in dims if dims.count(dim) > 1}):
            raise ValueError(f"a repeat asks each dimension once; repeated: {repeated}")
        return scores


class EvalResult(EvalDocumentModel):
    """One test case x one model x one k-iteration.

    **This is the analysis-shaped record and it carries no debug payload.** The
    turn-by-turn ``trace`` and the OTel spans live in a sibling
    :class:`EvalTrace` document keyed by this result's id; :attr:`has_trace` says
    whether one was written. The split is not an optimization detail a reader can
    ignore — it is why a query over thousands of results is affordable.

    There is deliberately **no sometimes-loaded trace field here.** An earlier
    design kept one, populated on the detail path and empty elsewhere, which would
    have made ``result.trace == []`` mean either "no trace" or "not loaded" —
    the absent-vs-empty ambiguity that ``usage``, ``cost_roles``, ``covariates``
    and ``termination`` each cost a defect to get right. Absence of the field
    cannot be misread: hold an ``EvalResult`` and there is no trace, full stop.
    A host's detail surface composes the two when it shows one.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    doc_type: Literal["eval_result"] = "eval_result"
    schema_version: SchemaVersion = EVAL_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    # Lineage
    eval_run_id: str = Field(min_length=1)
    test_case_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    k_iteration: int = Field(ge=1)

    candidate_kind: str | None = Field(
        default=None,
        description=(
            "Which candidate kind produced this observation — stamped by the runner's single dispatch "
            "site from the name it read there, so it is the kind that actually ran rather than a "
            "reference to something that could later say otherwise. It is here rather than left to a "
            "template lookup because templates are store-mastered and editable: a reader following "
            "`test_case_id` to its template would get today's `candidate_kind`, so an operator "
            "re-pointing a template would silently rewrite the provenance of every cell already "
            "stored under it. Every cell carries one, including the ones excluded before their "
            "candidate was built — except a cell whose deadline struck before the dispatch read its "
            "kind, the one exit that cannot say, which carries None."
        ),
    )

    candidate_instance_id: str | None = Field(
        default=None,
        description=(
            "The host identity this candidate ran under (``CandidateOutput.candidate_instance_id``) — the key "
            "that joins this result to whatever the host persisted while the cell ran. None for a kind that mints "
            "no identity of its own, and for a cell that ended before its candidate was built."
        ),
    )
    subject_id: str | None = Field(default=None)

    # Whether a sibling ``EvalTrace`` document was WRITTEN for this result — a fact about
    # storage, not a loaded-ness flag, so it reads the same on every path. Set by
    # ``EvalStorage.save_eval_result`` from the outcome of that write rather than from the
    # intent to make it, so a trace write that failed leaves this False (true, and the
    # result still persists) instead of promising detail no fetch can find.
    #
    # The readers, and only one of them is display:
    #   * a host's result renderer — a ``full=false`` footer, advertising omitted detail
    #     without fetching a whole trace to decide whether to print one line.
    #   * a host's result surfaces — the guard that skips the trace read entirely.
    #   * :mod:`threetears.evals.run.reads` ``get_result_trace`` — the marker-vs-reality check that warns
    #     when a result claims a trace no document backs. This is the reader that shapes the
    #     API: that function takes the whole result rather than its ids precisely so the
    #     marker can be checked in one place.
    #
    # This inventory has been wrong twice. It first claimed exactly one caller — true when
    # written, left standing as the fetch guards were added. Correcting it to "three" then
    # amended the number without re-deriving it, and missed the service.py reader in the very
    # commit that fixed a different miscounted inventory. The list is enumerated now rather
    # than counted, because a bare number is what kept going stale.
    has_trace: bool = Field(default=False)

    # What the world was asserted to hold before the candidate's first turn, one entry per
    # declared precondition. **Written only when the assertion FAILED** — a cell that ran carries
    # an empty list, because recording every satisfied presumption on every result would put the
    # template's own text on millions of rows to say nothing happened. Non-empty therefore means
    # this cell was excluded before any turn, and reads beside ``termination
    # == "precondition_failed"``; the entries that DID hold are carried alongside the ones that
    # did not, since an exclusion is read by asking what was presumed and which part of it the
    # world failed.
    precondition_outcomes: list[PreconditionOutcome] = Field(default_factory=list)

    # Outcomes
    goal_state_outcomes: list[GoalStateOutcome] = Field(default_factory=list)
    rubric_scores: list[RubricScore] = Field(default_factory=list)
    judge_reasoning: ModelProse = Field(default="")
    judge_model: str | None = Field(
        default=None,
        description=(
            "The run-level judge pin in force for this result — resolved at launch, so it is present "
            "whether the launch named it or inherited the role default. It is the model a scored "
            "dimension falls back to, NOT necessarily the model that scored any given one: a versioned "
            "JudgeConfig can pin its own, and which config scored which dim is on ``judge_config_ids``. "
            "It is also the model REQUESTED, so a provider-side floating alias here names no fixed "
            "model; the model the provider said answered is on each score's ``served_model``, which is "
            "what apparatus comparison reads. None = the result's run pinned no judge."
        ),
    )

    # Dual-score axes. Every non-factory-failure result carries both,
    # scored by purpose-built single-dim judges: ``transcript_score`` grades the
    # candidate's reasoning + tool-param choices *given the context it had*
    # (stable when externals drift); ``outcome_score`` grades whether the final
    # state / response satisfied the intent (drifts when externals drift). The
    # divergence between them decomposes regression source. ``None`` when
    # judging was skipped (factory failure / no judge service / empty trace).
    transcript_score: RubricScore | None = Field(default=None)
    outcome_score: RubricScore | None = Field(default=None)

    # Which versioned JudgeConfig scored each axis/dim — keyed by rubric_dim_id
    # (the reserved axis ids or a template dim name), value is the config id.
    # Absent key = the built-in default prompt was used. A map (not a scalar
    # ``judge_config_id``) because per-dim configs
    # make a single id insufficient for honest score bisection.
    #
    # Same name as ``EvalRun.judge_config_ids`` and NOT the same fact. This is what
    # OBSERVABLY scored this result; the run's is the set it committed to at launch.
    # The run's is the upper bound: this map is written per judge call, so a dim
    # that was never judged (a cell that died before scoring, a judge call that
    # failed) leaves no key here while the run still declares the config it had
    # pinned for it.
    judge_config_ids: dict[DimName, str] = Field(default_factory=dict)

    # Re-judges of this result's failed judge dimensions, oldest first — see
    # :class:`JudgeRescore`. Empty for a result scored in one pass.
    judge_rescores: list[JudgeRescore] = Field(default_factory=list)

    # Repeats of this result's judge scores, oldest first — see :class:`JudgeRepeat`. A measurement
    # of the judge, never a change to the scores above. Empty for a result nobody repeated.
    judge_repeats: list[JudgeRepeat] = Field(default_factory=list)

    # Cost. The blended spend over ``cost_roles``, derived from ``usage`` by
    # :func:`threetears.evals.contracts.usage_capture.blended_cost` at every exit of a cell.
    # **None means UNPRICED, never free**: a model call in those roles whose client reported no
    # price makes the total unknown, and a sum of the priced rows alone would be read as the
    # whole. The rows say which role went unpriced. Every aggregate leaves an unpriced result out
    # of its dollars and counts it beside them; the cost cap stops a capped run on the first one.
    # Required, with no default: a default of 0.0 is how an unpriced cell used to read as a free one.
    cost_usd: float | None = Field(ge=0.0)

    # Which roles' spend the blended ``cost_usd`` above was summed over — the marker
    # that makes that number placeable.
    #
    # The composition is not fixed: ``external`` joins CONDITIONALLY, per run — metered calls
    # become dollars only for a run whose operator declared the account's credit rate, so two
    # results written the same minute can carry different compositions. A cost comparison
    # across two compositions is comparing two different quantities, and without this field
    # nothing on the record says which one a result carries.
    #
    # **This states the CONVENTION the run applied, not the roles that happened to
    # spend.** A cell whose judge never ran still names ``judge`` here: the claim is
    # about what the total would have counted, which is what makes two results
    # comparable. Deriving it from the observed rows instead would make every cheap cell
    # look like its own epoch.
    #
    # Required: every exit of a cell writes it
    # (:func:`threetears.evals.contracts.usage_capture.blended_cost_roles`).
    cost_roles: list[UsageRole]

    # Per-role usage & cost. Required: every exit of a cell sets a list,
    # ``[]`` when capture ran and attributed no roles (e.g. a candidate-factory failure before
    # any candidate turn). ``cost_usd`` above is derived FROM these rows, over the roles
    # ``cost_roles`` names, so the two cannot disagree; spend that reaches no row reaches no
    # total. Background work's spend arrives here folded from its ``AsyncDelivery`` entries. The
    # external role carries dollars only on a run that resolved a credit rate for it.
    # Production-replicating vs program cost is derived from role membership, never stored.
    usage: list[RoleUsage]

    metered_calls_refused: int | None = Field(
        default=None,
        description=(
            "How many metered third-party calls this cell asked for and the RUN's ceiling "
            "refused (the host's configured default, or the launch override). A "
            "non-zero value means the candidate was measured on a world it could not fully "
            "reach — it asked to search, or to generate, and was told no — so a score here is "
            "not comparable with a score from a cell that ran unbounded. Zero says the ceiling "
            "was in force and never bound this cell; None says nothing counted: a cell run with no "
            "run ledger, or one that ended before its dispatch read the ledger back. The refusals are "
            "NOT errors and are deliberately not in error_details: a ceiling doing its "
            "configured job is a designed outcome, the same reasoning that gave a cancel and a "
            "budget stop their own channels on EvalRun."
        ),
    )

    # Latency — wall-clock decomposition from harvested OTel spans.
    # ``None`` when no spans were harvested (factory failure / empty trace).
    # Persisted as null by ``to_dict()`` like the other optional fields here.
    latency: LatencyMetrics | None = Field(default=None)

    async_deliveries: list[AsyncDelivery] | None = Field(
        default=None,
        description=(
            "One entry per piece of background work the candidate started, as its kind reported it "
            "(:class:`AsyncDelivery`). ``[]`` means the kind watched and no work was started — a cell excluded "
            "before its first turn records ``[]`` too, like its other capture fields; None means nothing "
            "watched, which is a kind that starts no background work. The two are different facts, so a "
            "consumer tests ``is None``."
        ),
    )
    kind_payload: VerbatimJsonObject | None = Field(
        default=None,
        description=(
            "What this cell's candidate kind reported that only the kind can name, stored verbatim and read by "
            "nothing in the engine (``CandidateOutput.kind_payload``). The host that owns the kind is the one "
            "reader. None for a kind that reports none."
        ),
    )
    world_events: list[WorldEvent] | None = Field(
        default=None,
        description=(
            "What moved this cell's world after it was seeded, in order: each triggered dimension that fired "
            "(by the rig through the host's fire handle, or in the world and recorded by the kind) and each "
            "ambient perturbation the template's seed scheduled (:class:`WorldEvent`). Recorded by the cell's "
            "world session, which the runner owns, so a cell cut off at its deadline keeps what fired before "
            "it. ``[]`` means the cell's kind opened the world and nothing moved it; None means no world was "
            "opened — a host that declares none, or a kind that never seeded through its session. "
            "``fired(...)`` reads it, and a re-check re-grades a fired check from it."
        ),
    )

    # Measurement-condition covariates — the conditions this
    # observation was made UNDER, so pooled latency is stratifiable instead of
    # unexplainable variance. Required: every exit derives it; ``{}`` means nothing was
    # measurable. Within the dict, a covariate nothing measured has NO KEY —
    # never a fabricated 0 — the same missing-is-not-zero rule ``RoleUsage``
    # holds with None. Keys are open by design (marginals iterate them);
    # see :mod:`threetears.evals.contracts.covariates` for what each one means and how it is
    # derived.
    covariates: dict[str, str | float]

    # Per-phase wall-clock inside this cell, in milliseconds and
    # keyed ``<tool>_<phase>_ms``. Latency attribution finer-grained than every
    # component of ``latency``: "model X synthesizes N% slower than Y" is a phase
    # question, and the phase spans do not exist to harvest — an async tool's
    # inner agent runs on a detached trace root deliberately outside the eval
    # latency buckets, so the timings ride out on the delivery instead. Required,
    # like ``covariates``: {} = nothing reported, absent key = that phase was never measured.
    phase_timings: dict[str, float]

    # Measures the HOST declared and this observation took. Before this field, a host
    # declared measures on `HostProfile.measures` and nothing could ever observe one: the
    # registry validated a catalogue, answered `is_observable` and gated bars over a vocabulary
    # no result had a slot for, so a second host's memo could only ever be about the shared
    # core, spend and wall-clock. `EvalRun.host_payload` is the same idea on the lever side.
    #
    # An open map rather than a sub-model because a host catalogue IS open: a host declares
    # whatever it measures and carries it without a schema change here. Resolution is NOT open,
    # though — `describe_measure` consults the host's measure registry, so a name the host never
    # declared stays undescribed and reports as a lost measurement rather than pooling
    # silently under a descriptor nobody wrote.
    #
    # Required, like its two neighbours: {} = the host reported nothing, absent key = that
    # measure was not taken for this observation.
    #
    # A value is typed by its descriptor's data type: a number for a numeric measure, a bool for
    # a boolean one, a string for a categorical or text one. The bundle drops (and reports) a value
    # whose type contradicts its descriptor rather than coercing it.
    host_measures: dict[str, bool | float | str]

    # **There is no ``graph_version_hash`` here, and there is no subject-side
    # ``system_graph_version_hash`` either.** Both were declared apparatus that nothing ever
    # wrote, and the reason nothing wrote them is not that a producer was never got round to:
    # the first host's prompt graph was an authoring and visualisation surface, and no turn's
    # prompt was rendered by it, so there was no graph version for a result to carry. The only
    # value available to stamp mirrored the DEFAULT prompt rather than the resolved one a
    # candidate ran under, so writing it would fabricate an apparatus condition instead of
    # recording one, which is worse than the blank it replaced.
    #
    # Retired rather than left declared because the blank was not free: every analysis over this
    # corpus hedged its findings on an apparatus condition that no campaign could ever confirm by
    # running. What an eval result IS produced under is its resolved prompt TEXT, which the
    # variant predicate already hashes. Wiring a real graph producer means first making prompt
    # graphs the production assembler, and that is a product decision rather than a field.

    # Contestant identity, stamped at result construction. Per-RESULT so a result read alone
    # carries its own coordinate, and so two runs' results pool on the key rather than on the
    # documents they came from.
    variant_key: str = Field(
        min_length=1,
        description=(
            "Digest of the resolved contestant stack — everything that would ship if this cell wins. "
            "Computed over RESOLVED config, so an override restating the system default is the same variant. "
            "Stamped on every cell: the engine resolves every run's candidate model and kind into the map it "
            "digests, so there is no unkeyed result."
        ),
    )
    identity_version: int = Field(
        ge=1,
        description=(
            "Which key-derivation predicate produced variant_key. Stored beside the key so a predicate "
            "change leaves old keys queryably distinct rather than silently regrouped."
        ),
    )

    # How the cell's execution ended — see :data:`CellTermination` for the arms and for
    # why this is stored rather than derived. Required: the runner writes it at every exit it
    # can return from. Read through
    # :func:`~threetears.evals.contracts.result_condition.resolve_result_condition`, which is the one
    # place a surface asks what condition a result is in — this field is the answer to a
    # third of that question, not a field for a surface to branch on directly.
    termination: CellTermination

    stop_cause: ConversationStopCause | None = Field(
        default=None,
        description=(
            "Why this trial's conversation stopped taking turns: the turn budget spent, the simulated "
            "user declaring itself done in its structured reply, a witnessed session's real participants "
            "ending it, or a rig "
            "fault (simulator, apparatus, candidate) — see ConversationStopCause. Present on every "
            "conversation trial whose turn loop ran. None when there was no conversation to stop: a "
            "candidate kind that does not converse (a classifier, a document generator), a cell "
            "excluded before its first turn, or a cell cancelled on its deadline, whose termination "
            "says so. Distinct from termination, which is how the cell ended as work."
        ),
    )

    turns_delivered: int | None = Field(
        default=None,
        ge=0,
        description=(
            "How many turns the candidate delivered in this cell before it ended — counted, not inferred: what the "
            "kind reported (CandidateTelemetry.turns_delivered), else the candidate turn records its trace stamps. "
            "What tells a model failure on the sixth turn of a conversation, after five delivered turns' time and "
            "spend, from a call refused straight away: the first took turns, and its cost and latency are the "
            "arm's (`delivered_a_turn`). None when nothing counted them: a kind that reports no count and stamps "
            "no turn records, a cell cut off before its kind reported anything, or a result stored before the "
            "field existed. A reader treats None as unknown — never as 0 — and falls back to the cause alone."
        ),
    )

    # Errors. ``runner_error`` is the combined human-display string (every
    # error_detail joined, as before). Scoring does NOT read it — it reads the
    # categorized fields below, so an infra hiccup is classified structurally,
    # not by re-parsing this string (fragile-parse rule).
    runner_error: str | None = Field(default=None)
    judge_error: str | None = Field(default=None)
    judge_cannot_tell: dict[DimName, ModelProse] = Field(
        default_factory=dict,
        description=(
            "dim -> the judge's reason, for each dim the judge answered it could not score because "
            "the evidence does not decide it. Such a dim carries no score and is not a judge error: "
            "the result stays in every other dim's measure and is excluded from the measures that "
            "need every dim — pass^k and the composite (result_condition.judged_on_every_dim). Empty when "
            "the judge scored or failed on every dim it was asked."
        ),
    )
    judge_cannot_tell_boundary: list[DimName] = Field(
        default_factory=list,
        description=(
            "The dims in judge_cannot_tell that are boundary (guardrail) dims, stamped from each dim's "
            "definition when it was judged, as a score's axis is. A boundary dim is in neither pass^k nor the "
            "composite, so a can't-tell on one leaves the trial in both; it is out of that guardrail's own "
            "reading only. Empty on a result stored before the field existed: its can't-tells are read as "
            "capability, which is how a score with no recorded axis is read."
        ),
    )

    # Error taxonomy. A result's error is one of two kinds,
    # and scoring treats them oppositely (an infra failure
    # must never score as candidate quality):
    #   * ``candidate_error`` — a model the candidate's configuration runs failed its
    #     call (a 400, a timeout, …): its OWN turn model, or the model its configuration
    #     sets for background work, whose failure the delivery flags — or a per-cell
    #     timeout charged to the candidate (``runner._DEADLINE_CHARGE``). A refusal of the
    #     calling account (auth, payment) is not this — it is the rig's. A broken
    #     candidate must FAIL, not vanish — it lowers pass^k / composite.
    #   * ``infra_error`` — a harness/infra failure (candidate factory, simulator,
    #     an async-delivery error that is not the background model's own, the drain's
    #     timeout/cap, a per-cell timeout charged to the rig). It EXCLUDES the
    #     result from the pass^k denominator and the composite mean — excluded ≠
    #     pass (no inflation), excluded ≠ fail (no flooring). ``judge_error`` is
    #     always infra and is treated identically at scoring time.
    # Both are additive (default None); a result may carry both, and candidate
    # takes precedence in scoring (a broken candidate fails even if infra also
    # hiccuped). Scoring reads ONLY these two fields plus ``judge_error`` — it
    # does not fall back to ``runner_error``: the runner writes all three from one ledger,
    # so a categorized field is never missing beside a set ``runner_error``.
    candidate_error: str | None = Field(default=None)
    infra_error: str | None = Field(default=None)

    scored_at: str = Field(default_factory=utc_now_iso)

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "eval_result":
            raise ValueError(f"doc_type must be 'eval_result', got '{v}'")
        return v

    def judge_score(self, dim: str) -> RubricScore | None:
        """The judge's score for one dimension, wherever this result carries it.

        A template dimension lives in ``rubric_scores``; the two dual-score axes live in
        ``transcript_score`` / ``outcome_score`` under their reserved ids. One lookup, so a rating's
        write and every agreement read find the same score.

        Args:
            dim: The dimension, as its score spells it.

        Returns:
            The score, or ``None`` when this result carries none on that dimension.
        """
        return next((score for score in self.judge_scores() if score.dim == dim), None)

    def judge_scores(self) -> list[RubricScore]:
        """Every score the judge gave this result: the template's dimensions, then the two reserved axes when scored.

        The assembly :meth:`judge_score` looks a dimension up in and the result actions list, so a reserved
        axis is never on one of those surfaces and missing from another.

        Returns:
            The scores as the judge gave them, in that order; empty for a result no judge scored.
        """
        return [
            score for score in (*self.rubric_scores, self.transcript_score, self.outcome_score) if score is not None
        ]


# =============================================================================
# EvalCassette — one recorded answer in a capture run's corpus
# =============================================================================


#: Which interception point a cassette was recorded at. See :class:`EvalCassette`
#: for what each seam implies about ``response``'s shape.
CassetteSeam = Literal["action", "delivery"]

#: How a recorded piece of background work ended when its capture cell did.
CassetteDeliveryOutcome = Literal["delivered", "failed", "undelivered"]


@dataclass(frozen=True)
class CassetteKey:
    """Everything that names one recorded answer — and the one place its document id is composed.

    A recording is the answer to one ask: in this corpus, for this template and case, this tool and
    action asked with these parameters, for the ``occurrence``-th time in the cell (0-based). The
    occurrence is what keeps two identical asks apart — a session that rolls ``1d20`` twice records
    two answers, not one overwritten by the other — and it is counted per cell in the order the
    candidate ASKED, so a replay pairs each ask with its own recording whatever order the work
    finished in.

    Attributes:
        corpus_id: The capture run that recorded it. A corpus is one run's, so two runs capturing at
            once never write into each other's.
        template_id: The template the capture ran.
        test_case_id: The case the recording belongs to.
        tool: The tool asked.
        action: The action asked, or the engine's delivery action for background work.
        params_hash: The digest of the parameters (an action) or the request (background work).
        occurrence: Which time, in the cell, this exact ask was made — 0 for the first.
    """

    corpus_id: str
    template_id: str
    test_case_id: str
    tool: str
    action: str
    params_hash: str
    occurrence: int

    @property
    def doc_id(self) -> str:
        """The deterministic document id: re-recording the same ask in the same corpus replaces it."""
        return (
            f"cassette:{self.corpus_id}:{self.template_id}:{self.test_case_id}:"
            f"{self.tool}:{self.action}:{self.params_hash}:{self.occurrence}"
        )


class EvalCassette(EvalDocumentModel):
    """One recorded answer — an action's result, or how a piece of background work ended.

    Cassettes are written by a run with ``cassette_mode='capture'`` while its tools run live, into
    that run's own corpus (``corpus_id`` is the capture run's id). A later ``'replay'`` run names
    the corpus it replays (``EvalRun.cassette_corpus_id``) and is served from it instead of running
    the tools, so a regression lane for a new prompt, model or judge costs no third-party quota.

    Keyed by :class:`CassetteKey` — corpus, template, case, tool, action, parameter digest and
    occurrence — whose fields are denormalized onto the row so an inspection can filter without
    parsing the id.

    ``seam`` says what ``response`` holds:

    * ``seam='action'`` — recorded at a synchronous tool's ``act()``. ``response`` is its result as
      the tool's declared type dumps it.
    * ``seam='delivery'`` — a piece of background work an asynchronous tool started, keyed by the
      request that started it. ``outcome`` says how it ended in the capture: ``response`` is the
      delivered payload for ``'delivered'``, ``error`` says why for ``'failed'``, and both are absent
      for ``'undelivered'`` — the capture's cell ended with it still in flight.
    """

    id: str = Field(min_length=1)
    doc_type: Literal["eval_cassette"] = "eval_cassette"
    schema_version: SchemaVersion = EVAL_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    corpus_id: str = Field(min_length=1)
    template_id: str = Field(min_length=1)
    test_case_id: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    action: str = Field(min_length=1)
    params_hash: str = Field(min_length=1)
    occurrence: int = Field(ge=0)

    seam: CassetteSeam
    #: Delivery seam only — how the recorded work ended.
    outcome: CassetteDeliveryOutcome | None = None
    #: The recorded answer: an action's result, or a delivered payload. ``None`` exactly for a
    #: delivery that delivered nothing.
    response: dict[str, Any] | None = None
    #: Delivery seam only — why the recorded work failed.
    error: str | None = None

    #: The model the capture run's candidate ran on — provenance; the corpus is what binds.
    captured_model: str = Field(min_length=1)
    captured_at: str = Field(default_factory=utc_now_iso)

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "eval_cassette":
            raise ValueError(f"doc_type must be 'eval_cassette', got '{v}'")
        return v

    @model_validator(mode="after")
    def check_one_story(self) -> EvalCassette:
        """Refuse a recording whose fields tell two stories, or whose id is not its key's.

        Raises:
            ValueError: The id is not the one its key composes; an action recording without a
                response or with a delivery's fields; a delivery recording without an outcome; or a
                delivery whose response or error disagrees with its outcome.
        """
        if self.id != self.key.doc_id:
            raise ValueError(f"cassette id {self.id!r} is not its key's ({self.key.doc_id!r})")
        if self.seam == "action":
            if self.response is None:
                raise ValueError("an action-seam cassette records the action's result in response")
            if self.outcome is not None or self.error is not None:
                raise ValueError("an action-seam cassette carries no delivery outcome or error")
            return self
        if self.outcome is None:
            raise ValueError("a delivery-seam cassette records how the work ended in outcome")
        if (self.response is not None) != (self.outcome == "delivered"):
            raise ValueError("a delivery-seam cassette carries a response exactly when its outcome is 'delivered'")
        if (self.error is not None) != (self.outcome == "failed"):
            raise ValueError("a delivery-seam cassette carries an error exactly when its outcome is 'failed'")
        return self

    @property
    def key(self) -> CassetteKey:
        """The key this recording answers."""
        return CassetteKey(
            corpus_id=self.corpus_id,
            template_id=self.template_id,
            test_case_id=self.test_case_id,
            tool=self.tool,
            action=self.action,
            params_hash=self.params_hash,
            occurrence=self.occurrence,
        )

    @classmethod
    def build(
        cls,
        key: CassetteKey,
        *,
        scope_id: str,
        seam: CassetteSeam,
        captured_model: str,
        response: dict[str, Any] | None = None,
        outcome: CassetteDeliveryOutcome | None = None,
        error: str | None = None,
    ) -> EvalCassette:
        """Construct a cassette with its key's document id.

        Args:
            key: The ask this recording answers.
            scope_id: The scope the corpus lives in.
            seam: Which seam recorded it.
            captured_model: The capture run's candidate model.
            response: The action's result, or the delivered payload.
            outcome: How the background work ended (delivery seam only).
            error: Why it failed (delivery seam only).

        Returns:
            The unsaved cassette.
        """
        return cls(
            id=key.doc_id,
            scope_id=scope_id,
            corpus_id=key.corpus_id,
            template_id=key.template_id,
            test_case_id=key.test_case_id,
            tool=key.tool,
            action=key.action,
            params_hash=key.params_hash,
            occurrence=key.occurrence,
            seam=seam,
            outcome=outcome,
            response=response,
            error=error,
            captured_model=captured_model,
        )


__all__ = [
    "stored_variation",
    "CHECK_REFUSED_UNDER_CURRENT_GRAMMAR",
    "GOAL_CHECK_PROOF_RULES",
    "goal_check_proofs_as_read",
    "refused_goal_checks",
    "stale_goal_check_proofs",
    "ApparatusSettingValue",
    "MeteredCallOrigin",
    "CANDIDATE_SPEAKER",
    "DEFAULT_JUDGE_TEMPERATURE",
    "EVAL_SCHEMA_VERSION",
    "MODEL_DEFAULT_TEMPERATURE",
    "JudgeTemperature",
    "NON_TERMINAL_RUN_STATUSES",
    "OUTCOME_DIM_ID",
    "ROUND_DONE",
    "RaterKind",
    "ReasoningEffort",
    "TERMINAL_RUN_STATUSES",
    "TRANSCRIPT_DIM_ID",
    "ActorPolicy",
    "ApparatusProvenance",
    "AsyncDelivery",
    "AsyncDeliveryStatus",
    "AsyncExternalSpend",
    "CalibrationRating",
    "CassetteDeliveryOutcome",
    "CassetteKey",
    "CassetteSeam",
    "CatalogRubricDim",
    "CellTermination",
    "ClientRequestSettings",
    "CompletenessSource",
    "ControlEndState",
    "ConversationSpec",
    "ConversationStopCause",
    "EvalCassette",
    "EvalResult",
    "EvalRun",
    "MeasureDeclaration",
    "EvalRunStatus",
    "EvalTemplate",
    "EvalTestCase",
    "GoalCheckControl",
    "GoalCheckControls",
    "GoalCheckIntent",
    "GoalStateOutcome",
    "JudgeConfig",
    "JudgeEvidence",
    "JudgeRepeat",
    "JudgeRescore",
    "JudgedArtifact",
    "Precondition",
    "PreconditionOutcome",
    "ProposedDimSuggestion",
    "ProposedTemplate",
    "RubricDim",
    "RubricDimTombstone",
    "JudgeConfigTombstone",
    "RubricProposal",
    "RepeatedScore",
    "RubricScore",
    "RunCompleteness",
    "SchemaVersion",
    "WorldSeed",
    "VariationAxis",
    "resolve_effective_judges",
    "scored_dim_ids",
]
