"""The report document: a campaign report as the analyst writes it, and as it is stored (``report-doc/1``).

An analyst (a model with the ``evals_analyst`` tool and the analyst skill) reads a frozen report snapshot and writes
one :class:`ReportDoc`: the story it tells, its headline and lede, the call (``recommends``), how sure the call is and
on which caveats, the decision block, the claims in page order with the evidence each rests on, what a reader can
explore, what it left out and why, the next steps and the method. ``docs/eval-report-kit/ANALYST.md`` §5 is the
program design, and ``docs/eval-report-kit/schema/report-doc.schema.json`` is :func:`report_doc_json_schema` written
out, which a test holds byte-identical to this model.

**Code computes and verifies; the analyst chooses.** Nothing here reads ``story``, a slot's word, a claim's weight or
a block's kind to decide anything: each is a choice the renderer draws as written. ``story`` is read by no code at all,
so a page and its checks are byte-identical when only it changes. What the renderer adds (the masthead, the arms key,
the options table with ``recommends`` marked, the form words, each block's level, unit and axis, the full-results
appendix, the omitted list and the made line) is computed, never written here. ``recommends`` holds only the arms the
report recommends: a tradeoff's lean stays in its prose, and an empty list (no call) is valid.

**Where the prose is.** :data:`REPORT_DOC_PROSE_FIELDS` names every field a reader reads as prose, by path. Three of
them (``explore[].outcome``, ``next[].settles``, ``confidence.caveats[].label``) are not typed :data:`ModelProse`,
because the prose scan reads by attribute name and other models read a structured ``outcome``, ``settles`` and
``label`` by value; so a check over a document's prose (the invariants' typed digits and internal ids) walks this
tuple, never the type alone.

**Three layers of refusal, each with its own owner:**

- **The model** (:class:`ReportDoc`) refuses a document out of shape: an unknown field, slot, block kind or reference
  line; a count outside its bounds (1 to 8 decision entries, 1 to 12 claims, 1 or 2 evidence blocks a claim, up to 3
  explore rows, up to 6 next steps, 1 to 12 method items); a caveat that is both checked and conditional, or neither;
  an ``analyst:`` caveat with no label, or one a claim or method item names that the confidence does not declare; two
  claims of one id; an arm recommended twice.
- **Validity** (:func:`report_doc_validity`) refuses a well-shaped document that names what its snapshot does not
  hold: a recommended arm that is no arm, a caveat id that is neither one code assigns on the snapshot nor
  ``analyst:<slug>``. It is a precondition of the invariants, not one of them.
- **The invariants** (``analysis/report_doc_checks.py``) refuse what code can prove wrong about a valid document on its
  snapshot: a number not given by a key, a block that does not render, a caveat landing short, a confidence form its
  caveats contradict, an internal id above the appendix.

**Caveats** are named by the id code assigns
(:func:`~threetears.evals.analysis.report_decision.report_caveats` and
:func:`~threetears.evals.analysis.report_decision.implied_caveats`: ``late:margin``, ``judge_uncalibrated``,
``point_estimate:<criterion>``, ...), or by ``analyst:<slug>`` for one the analyst found that code does not assign.
Code has a label for each kind it assigns, so a code caveat's ``label`` is optional; an analyst caveat has no code
label, so it carries its own and is declared once in ``confidence.caveats``, where a claim or method item may name it
by id. A caveat is open, with the condition that would make it material (``material_if``), or ``checked:
"immaterial"``, which the invariants allow only where code proves it.

**Stored** as :class:`EvalReportDoc` (``doc_type`` ``eval_report_doc``): the document as JSON, checked against
:class:`ReportDoc` on every construction and read, beside the snapshot it was written from, how the writing loop
ended, the editor's unresolved findings and the loop's tally. Derived and regenerable like its snapshot: writing the
report again makes a new one, ``EvalStorage.drop_derived`` sweeps it, and a regenerable schema bump refuses it on read.

**Reads are strict, so every change to the document's shape ships with a regenerable bump.** A stored document is
never half-read: one this build's :class:`ReportDoc` does not accept is refused whole. That is safe only because a
change to :class:`ReportDoc` (or to anything it holds) always moves
:data:`~threetears.evals.schema.versioning.REGENERABLE_SCHEMA_VERSION`, so a stale document is refused by its version
first, before its body is read, with the remedy to drop it and write the report again; and a document at the current
version always reads. The degraded read of plan §2.3 ("a doc that no longer validates") therefore arises only as that
version refusal.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Collection
from typing import Annotated, Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from threetears.evals.kernel.report_plan import ChartQuery, PlanChartKind, ReferenceKind
from threetears.evals.schema.base import EvalDocumentModel, VerbatimJsonObject
from threetears.evals.schema.models import SchemaVersion, utc_now_iso
from threetears.evals.schema.prose import ModelProse
from threetears.evals.schema.versioning import REGENERABLE_SCHEMA_VERSION

#: The schema id every report document carries.
REPORT_DOC_SCHEMA: Final = "report-doc/1"

#: The prefix of a caveat the analyst names itself, which code does not assign.
ANALYST_CAVEAT_PREFIX: Final = "analyst:"

#: An analyst caveat's slug: lower-case words of letters and digits joined by ``_`` or ``-``.
_ANALYST_SLUG = re.compile(r"[a-z0-9]+(?:[_-][a-z0-9]+)*")

#: ANALYST §3: the story a report tells. Read by no code; the analyst's own label for its story.
Story = Literal["decision", "tradeoff", "diagnosis", "status", "exploratory"]

#: How sure the call is: ``high``, no open caveat; ``high_if_immaterial``, open caveats each with what would make it
#: material; ``low``, the evidence is weak whatever the caveats.
ConfidenceForm = Literal["high", "high_if_immaterial", "low"]

#: ANALYST §5: the decision block's shared vocabulary. Each entry is titled by its word; the analyst picks the words
#: that fit its story, in its own order, and may repeat one.
DecisionSlotWord = Literal[
    "recommendation",
    "why_not",
    "watch",
    "next",
    "the_trade",
    "where_it_flips",
    "our_lean",
    "what_settles_it",
    "what_is_tangled",
    "known_despite_it",
]

#: How much a claim carries, which sets its type size and nothing else.
ClaimWeight = Literal["lead", "support", "detail"]

#: What a caveat code proved immaterial is checked as.
_Checked = Literal["immaterial"]

#: An id the document names something by: never empty.
_DocId = Annotated[str, StringConstraints(min_length=1)]


class _Doc(BaseModel):
    """A closed object: an unknown field is refused, on a submission and on a stored read alike."""

    model_config = ConfigDict(extra="forbid")


#: Every field of a report document a reader reads as prose, by path (``[]`` marks a list's items), in document
#: order. The module docstring says why a check over prose walks this rather than the :data:`ModelProse` type; a test
#: holds it to every text field of :class:`ReportDoc` that is not an id.
REPORT_DOC_PROSE_FIELDS: Final[tuple[str, ...]] = (
    "headline",
    "lede",
    "flag",
    "confidence.caveats[].label",
    "confidence.caveats[].material_if",
    "decision[].text",
    "claims[].text",
    "claims[].evidence[].caption",
    "explore[].control",
    "explore[].binds",
    "explore[].outcome",
    "omitted[].what",
    "omitted[].why",
    "next[].text",
    "next[].settles",
    "method[].term",
    "method[].text",
)


def is_analyst_caveat(caveat_id: str) -> bool:
    """Whether a caveat id names one the analyst found, rather than one code assigns.

    :param caveat_id: A caveat id as a report document names it.
    :ptype caveat_id: str
    :return: Whether it starts ``analyst:``.
    :rtype: bool
    """
    return caveat_id.startswith(ANALYST_CAVEAT_PREFIX)


class DocCaveat(_Doc):
    """One caveat the call's confidence carries: open, with what would make it material, or checked immaterial."""

    id: _DocId = Field(
        description="The caveat's id: one code assigns on the snapshot (caveats_list lists them), or `analyst:<slug>` "
        "for one code does not assign."
    )
    label: str | None = Field(
        default=None,
        min_length=1,
        description="The caveat's name as the page prints it. Optional for a code caveat, which code labels by its "
        "kind; required for an `analyst:` caveat.",
    )
    material_if: ModelProse | None = Field(
        default=None,
        min_length=1,
        description="For an open caveat: what would make it material, figures by key. Exactly one of this and "
        "`checked`.",
    )
    checked: _Checked | None = Field(
        default=None,
        description="`immaterial` for a caveat code proves does not move the call; the page prints code's reason. "
        "Exactly one of this and `material_if`.",
    )

    @model_validator(mode="after")
    def _open_or_checked(self) -> Self:
        """Refuse a caveat that is both open and checked, or neither, and an analyst caveat out of form.

        :raises ValueError: Both or neither of ``material_if`` and ``checked`` are set; an ``analyst:`` id's slug is
            empty or not lower-case words; an ``analyst:`` caveat has no label.
        """
        if (self.material_if is None) == (self.checked is None):
            raise ValueError(
                f'caveat {self.id!r} is either open, with `material_if`, or `checked: "immaterial"`; exactly one'
            )
        if is_analyst_caveat(self.id):
            slug = self.id.removeprefix(ANALYST_CAVEAT_PREFIX)
            if not _ANALYST_SLUG.fullmatch(slug):
                raise ValueError(
                    f"caveat {self.id!r}: an analyst caveat's id is `analyst:` and a slug of lower-case letters and "
                    "digits joined by `_` or `-`"
                )
            if self.label is None:
                raise ValueError(f"caveat {self.id!r}: an analyst caveat carries its label, since code has none for it")
        return self


class DocConfidence(_Doc):
    """How sure the call is, in one of three forms, and the caveats it carries, each once."""

    form: ConfidenceForm
    caveats: list[DocCaveat] = Field(description="The caveats the confidence carries, open or checked, each once.")


class DecisionEntry(_Doc):
    """One entry of the decision block: the slot word that titles it, and what it says."""

    slot: DecisionSlotWord
    text: ModelProse = Field(min_length=1, description="What the entry says, figures by key.")


class Evidence(_Doc):
    """One evidence block under a claim: a kind of block over a query of the snapshot, as ``block_render`` draws it."""

    block: PlanChartKind = Field(description="The kind of block, as block_catalog lists it.")
    query: ChartQuery = Field(
        description="What the block reads, every field filled, as block_render returns it: an empty list or string "
        "is no filter, `limit` 0 no limit."
    )
    reference: ReferenceKind | None = Field(
        default=None, description="The reference line to draw, which code keys from the snapshot; null for none."
    )
    emphasis: str = Field(
        default="", description="The one arm, stratum or task the block stresses, by id; empty for none."
    )
    caption: ModelProse = Field(min_length=1, description="The line under the block, figures by key.")


class Claim(_Doc):
    """One claim of the story, in page order, with the evidence that shows it and the caveats said beside it."""

    id: _DocId = Field(description="The claim's id, unique in the document; the editor's findings name it.")
    text: ModelProse = Field(min_length=1, description="The claim, figures by key.")
    weight: ClaimWeight = Field(description="How much it carries; it sets the claim's type size and nothing else.")
    evidence: list[Evidence] = Field(min_length=1, max_length=2, description="The one or two blocks that show it.")
    caveats: list[_DocId] = Field(
        default_factory=list, description="The ids of the caveats stated beside this claim; may be empty."
    )
    durable: bool = Field(
        default=False, description="Whether the claim should outlive this campaign, as an insight later ones read."
    )


class ExploreRow(_Doc):
    """One thing a reader could change and what it would do, in prose.

    ``outcome`` here, :class:`NextStep`'s ``settles`` and :class:`DocCaveat`'s ``label`` are model prose not typed
    :data:`ModelProse`: the prose scan reads by attribute name, and other models read a structured ``outcome``,
    ``settles`` and ``label`` by value.
    """

    control: ModelProse = Field(min_length=1, description="What a reader could change.")
    binds: ModelProse = Field(min_length=1, description="What it decides.")
    outcome: str = Field(min_length=1, description="What happens as it moves, figures by key.")


class Omitted(_Doc):
    """One thing the analyst saw and left out of the body, and why; the appendix still holds it."""

    what: ModelProse = Field(min_length=1)
    why: ModelProse = Field(min_length=1)


class NextStep(_Doc):
    """One experiment or declaration to make next, and what it would settle."""

    text: ModelProse = Field(min_length=1, description="The step, figures by key.")
    settles: str = Field(min_length=1, description="What its result would settle.")


class MethodItem(_Doc):
    """One term of the method appendix, and the caveats whose reach is the method."""

    term: ModelProse = Field(min_length=1)
    text: ModelProse = Field(min_length=1, description="What it means here, figures by key.")
    caveats: list[_DocId] = Field(
        default_factory=list, description="The ids of the caveats stated in this item; may be empty."
    )


class ReportDoc(_Doc):
    """A campaign report as the analyst writes it, report-doc/1: the story, the call and how sure it is, the decision
    block, the claims in page order with their evidence, and the appendix's parts. Its prose names every figure by key.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
        json_schema_extra={"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": REPORT_DOC_SCHEMA},
    )

    schema_id: Literal["report-doc/1"] = Field(alias="schema")
    story: Story = Field(description="The story the report tells; no code reads it.")
    headline: ModelProse = Field(min_length=1, max_length=240, description="The finding, figures by key.")
    lede: ModelProse = Field(min_length=1, max_length=420, description="What the headline rests on, figures by key.")
    flag: ModelProse | None = Field(
        default=None,
        min_length=1,
        max_length=300,
        description="One line above the decision for what a reader must know first; null for none.",
    )
    recommends: list[_DocId] = Field(
        description="The arms the report recommends, by id, each once; empty for no call. The options table marks them."
    )
    confidence: DocConfidence
    decision: list[DecisionEntry] = Field(
        min_length=1, max_length=8, description="The decision block's entries, in page order; a slot may repeat."
    )
    claims: list[Claim] = Field(min_length=1, max_length=12, description="The story's claims, in page order.")
    explore: list[ExploreRow] = Field(max_length=3, description="What a reader could change; may be empty.")
    omitted: list[Omitted] = Field(description="What was left out of the body, and why; may be empty.")
    next: list[NextStep] = Field(max_length=6, description="The next steps, after the claims; may be empty.")
    method: list[MethodItem] = Field(min_length=1, max_length=12, description="The method appendix's terms.")

    @model_validator(mode="after")
    def _ids_hold_together(self) -> Self:
        """Refuse two claims of one id, an arm recommended twice, and an analyst caveat named but not declared.

        :raises ValueError: A claim id repeats; an arm is recommended twice; a claim or method item names an
            ``analyst:`` caveat the confidence does not carry.
        """
        claims = [claim.id for claim in self.claims]
        repeated = sorted({claim for claim in claims if claims.count(claim) > 1})
        if repeated:
            raise ValueError(f"each claim has its own id; repeated: {repeated}")
        twice = sorted({arm for arm in self.recommends if self.recommends.count(arm) > 1})
        if twice:
            raise ValueError(f"recommends names each arm once; repeated: {twice}")
        declared = {caveat.id for caveat in self.confidence.caveats}
        for where, named in _caveats_named_beside(self):
            if is_analyst_caveat(named) and named not in declared:
                raise ValueError(
                    f"{where} names {named!r}: an analyst caveat is declared once in confidence.caveats, with its "
                    "label, and named elsewhere by id"
                )
        return self


def _caveats_named_beside(doc: ReportDoc) -> list[tuple[str, str]]:
    """Every caveat id a claim or a method item names, with the field path it sits at, in document order."""
    named: list[tuple[str, str]] = []
    for i, claim in enumerate(doc.claims):
        named += [(f"claims[{i}].caveats[{j}]", caveat) for j, caveat in enumerate(claim.caveats)]
    for i, item in enumerate(doc.method):
        named += [(f"method[{i}].caveats[{j}]", caveat) for j, caveat in enumerate(item.caveats)]
    return named


def _caveat_mentions(doc: ReportDoc) -> list[tuple[str, str]]:
    """Every caveat id the document names, the confidence's first, with the field path it sits at."""
    confidence = [(f"confidence.caveats[{i}].id", caveat.id) for i, caveat in enumerate(doc.confidence.caveats)]
    return confidence + _caveats_named_beside(doc)


#: Where a document naming an arm its snapshot does not hold sends its writer.
_ARM_REMEDY = "campaign_overview lists the snapshot's arm ids"

#: Where a document naming a caveat its snapshot does not assign sends its writer.
_CAVEAT_REMEDY = (
    "caveats_list lists the caveat ids code assigns on this snapshot; name one of them, or a caveat code does not "
    "assign as `analyst:<slug>`, declared in confidence.caveats with its label"
)


def report_doc_validity(doc: ReportDoc, *, arms: Collection[str], caveats: Collection[str]) -> list[tuple[str, str]]:
    """What a well-shaped document names that its snapshot does not hold (this module's account of validity).

    The kernel does not read the report input, so its caller passes the snapshot's arm ids and the caveat ids code
    assigns on it, stated and implied: the ids of
    :func:`~threetears.evals.analysis.report_decision.every_caveat`, the one set ``caveats_list`` pages too. Emphasis and every query are the invariants' to check, since a block names
    strata and tasks too and its query's names are read where it is drawn.

    :param doc: The document.
    :ptype doc: ReportDoc
    :param arms: The snapshot's arm ids.
    :ptype arms: Collection[str]
    :param caveats: The caveat ids code assigns on the snapshot.
    :ptype caveats: Collection[str]
    :return: Each fault as ``(field path, why)``, in document order; empty for a valid document.
    :rtype: list[tuple[str, str]]
    """
    faults = [
        (f"recommends[{i}]", f"names arm {arm!r}, which the snapshot does not hold; {_ARM_REMEDY}")
        for i, arm in enumerate(doc.recommends)
        if arm not in arms
    ]
    faults += [
        (where, f"names caveat {named!r}, which code does not assign on the snapshot; {_CAVEAT_REMEDY}")
        for where, named in _caveat_mentions(doc)
        if not is_analyst_caveat(named) and named not in caveats
    ]
    return faults


def report_doc_json_schema() -> dict[str, Any]:
    """The ``report-doc/1`` JSON schema an analyst writes against, as ``report-doc.schema.json`` holds it.

    :return: :class:`ReportDoc`'s validation schema, by alias.
    :rtype: dict[str, Any]
    """
    return ReportDoc.model_json_schema(by_alias=True)


# --- the stored document -------------------------------------------------------------------------------------------


class ReportFinding(_Doc):
    """One review finding the editor left unresolved, which the page prints in Method.

    ``severity`` ``must_fix`` fails a review; ``should_fix`` does not.
    """

    claim: str = Field(description="The claim it is about, by id; empty for the page as a whole.")
    severity: Literal["must_fix", "should_fix"]
    finding: ModelProse = Field(min_length=1, description="What the editor found.")


class EvalReportDoc(EvalDocumentModel):
    """One report document as stored: the document, the snapshot it reads, how its writing ended, and what it cost.

    Immutable once stored, and never written over: writing a report again stores a new one beside it, so the earlier
    stays readable until a regenerable sweep. ``doc`` is held as its ``report-doc/1`` JSON, checked against
    :class:`ReportDoc` on every construction and read and kept in the model's canonical JSON, so a stored document is
    always one the model accepts, and each of ``findings`` likewise against :class:`ReportFinding`; a reader validates
    them into those models. Held as JSON, the document's own types stay out of this stored model's public signature,
    as a report plan's do. Its figure keys resolve against the snapshot's report input, which is why the snapshot is
    named here and not in the document.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    doc_type: Literal["eval_report_doc"] = "eval_report_doc"
    schema_version: SchemaVersion = REGENERABLE_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    snapshot_id: str = Field(
        min_length=1, description="Reference → EvalReportSnapshot.id: the snapshot the document was written from."
    )
    campaign_id: str | None = Field(
        description="Reference → EvalCampaign.id: the snapshot's campaign. None for a snapshot of a bare report input."
    )
    doc: VerbatimJsonObject = Field(
        description="The document, as `report-doc/1` JSON (its prose names figures by key)."
    )
    ending: Literal["stored", "stored_with_findings", "over_budget"] = Field(
        description="How the writing loop ended: `stored`, a review passed with no must_fix finding; "
        "`stored_with_findings`, the revisions ran out with findings open; `over_budget`, the budget ran out and the "
        "last draft the invariants passed, reviewed at least once, was kept."
    )
    findings: list[VerbatimJsonObject] = Field(
        default_factory=list,
        description="The editor's findings left unresolved, which Method prints, each `{claim, severity, finding}`.",
    )
    generation: VerbatimJsonObject = Field(
        description="The writing loop's tally: per turn and rolled up by role and round, tokens, cost, tool calls, "
        "bounces and wall time, as the loop records it."
    )
    created_by: str = Field(min_length=1, description="Who asked for the report, as the host named the caller.")
    created_at: str = Field(default_factory=utc_now_iso)

    @model_validator(mode="before")
    @classmethod
    def check_version_first(cls, data: Any) -> Any:
        """Refuse a document written under another regenerable version by its version alone, before its body is read.

        A document from before a shape change would otherwise be refused for its body too, field by field, which reads
        as a fault to fix rather than a document to make again (the module docstring's strict-read decision).

        :param data: The input under validation; only a mapping is read.
        :ptype data: Any
        :return: The input, unchanged.
        :rtype: Any
        :raises ValueError: Its ``schema_version`` is not this build's regenerable version.
        """
        if isinstance(data, dict) and "schema_version" in data and data["schema_version"] != REGENERABLE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version: this report document was written under eval schema v{data['schema_version']} and "
                f"this build reads v{REGENERABLE_SCHEMA_VERSION} only; a report document is regenerated across a schema "
                "change, never migrated or half-read. An operator clears every old one in a scope with "
                "EvalStorage.drop_derived(scopes), then writes each report again from a fresh snapshot"
            )
        return data

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "eval_report_doc":
            raise ValueError(f"doc_type must be 'eval_report_doc', got '{v}'")
        return v

    @field_validator("doc")
    @classmethod
    def check_doc(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Refuse a document the model does not accept, and keep it in the model's canonical JSON.

        :raises ValueError: The document does not validate as a :class:`ReportDoc`.
        """
        canonical: dict[str, Any] = ReportDoc.model_validate(v).model_dump(mode="json", by_alias=True)
        return canonical

    @field_validator("findings")
    @classmethod
    def check_finding_shape(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Refuse a finding out of shape, and keep each in the model's canonical JSON.

        :raises ValueError: A finding does not validate as a :class:`ReportFinding`.
        """
        return [ReportFinding.model_validate(finding).model_dump(mode="json") for finding in v]

    @model_validator(mode="after")
    def check_findings(self) -> Self:
        """Refuse a finding on a claim the document does not hold, and an ending its findings contradict.

        :return: This document, unchanged.
        :rtype: EvalReportDoc
        :raises ValueError: A finding names a claim id the document does not hold; a ``stored`` document carries a
            ``must_fix`` finding. A ``stored_with_findings`` one may carry none, since a review fails on its scores
            alone too.
        """
        claims = {claim.id for claim in ReportDoc.model_validate(self.doc).claims}
        findings = [ReportFinding.model_validate(finding) for finding in self.findings]
        for finding in findings:
            if finding.claim and finding.claim not in claims:
                raise ValueError(f"a finding names claim {finding.claim!r}, which the document does not hold")
        if self.ending == "stored" and any(finding.severity == "must_fix" for finding in findings):
            raise ValueError("a report stored after a passing review carries no must_fix finding")
        return self


#: The names with a caller outside this module. The document's own models and functions are this build's first
#: report-doc/1 surface; each joins this list with its first production caller (the invariants, the claims renderer,
#: the writing loop), never for a test alone.
__all__ = ["EvalReportDoc"]
