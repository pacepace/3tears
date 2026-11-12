"""A capability gap the analyst reasoned around, or a notice code wrote about a report's writing: ``eval_analyst_gap``.

The analyst and its editor call ``capability_gap`` when no tool gave what they needed (kind ``missing``) or a result
looked wrong (kind ``suspect``, carrying the figure keys and the tool calls that reproduce it). Code writes two notices
of its own into the same store, from what a report's writing already knows: a figure key that did not resolve on its
snapshot and sent a draft back (``unresolved_key``, with the key and the bounce's verdicts at its field), and a rubric
item the editor scored 0 or 1 (``editor_score``, with the item, its score and reason, and whether the review held a
``must_fix`` finding). Each is one :class:`EvalAnalystGap`, what was said and where, deciding nothing.

**Durable.** Nothing makes a gap again: it is what a model said in a paid session, or what code saw during one, and a
session a write did not store is gone once the write ends. So the type is ``durable`` (a durable minor), outliving every
core baseline reset and never swept by ``EvalStorage.drop_derived``.

**One record per call.** A record's id is the call's own (:func:`analyst_gap_id`): a model's gap is named by its
snapshot, kind, need and why, so a replay or a retried write over the same snapshot names the gap it named before, and
the store keeps the first (``EvalStorage.save_analyst_gap``, create-only). A ``missing`` gap's id is the one
``capability_gap`` returned before kinds existed. A notice is named by its snapshot, kind, session and that session's
transcript digest, and the submission and key, or the review round and item, it records: a byte-for-byte replay of a
write names the notices it named before, and a second paid write on the same snapshot, whose transcripts differ,
names its own.

**Provenance.** The snapshot always; inside a report write, the report the write stored (``None`` when it stored
none), the session's role and name, the model, and the session's transcript digest as the write's tally holds it;
outside a write (a caller over MCP, the dev CLI), the caller alone. ``created_by`` is who asked, as the host names
them.

**The digest's shape.** :class:`GapDigest` and :class:`GapGroup` are what ``analyst_gaps_list`` returns: the records
grouped, ranked and written as issue text (``analysis/analyst_gaps.py`` says how).
"""

from __future__ import annotations

from typing import Annotated, Final, Literal, get_args

from pydantic import Field, field_validator

from threetears.evals.schema.base import (
    CoreShapeModel,
    DurableDocumentModel,
    EvalBaseModel,
    EvalDocumentModel,
    VerbatimJsonObject,
    known_unless_newer_minor,
)
from threetears.evals.schema.hashing import bytes_digest
from threetears.evals.schema.models import DurableSchemaVersion, utc_now_iso
from threetears.evals.schema.versioning import DURABLE_SCHEMA_VERSION

#: What a stored gap records: ``missing``, a capability the analyst needed and no tool gave; ``suspect``, a result
#: that looked wrong, with what reproduces it; ``unresolved_key``, code's notice of a figure key that did not resolve
#: and sent a draft back; ``editor_score``, code's notice of a rubric item the editor scored below 2. An open vocabulary
#: (:mod:`~threetears.evals.schema.versioning`): a kind a newer minor adds is read from that minor's document, and the
#: gap digest groups and ranks it by its own name, beside the kinds it knows.
GapKind = str

#: A :data:`GapKind` this build knows, as a type: what a caller names and what this build writes.
KnownGapKind = Literal["missing", "suspect", "unresolved_key", "editor_score"]

#: The values of :data:`GapKind` this build knows, in the order the digest names them.
GAP_KINDS: Final[tuple[str, ...]] = get_args(KnownGapKind)

#: The kinds a model's ``capability_gap`` call names; the other two are code's notices.
CallGapKind = Literal["missing", "suspect"]

#: The type of a stored gap's ``kind``: one of :data:`GAP_KINDS`, or a newer minor's.
StoredGapKind = Annotated[GapKind, known_unless_newer_minor("gap kind", GAP_KINDS)]

#: The most characters a gap call's need, why or workaround holds, and each of its calls as JSON: a note of one
#: capability, not a transcript, so a durable record stays small.
GAP_TEXT_MAX_CHARS: Final = 2_000

#: The most figure keys, calls or claim ids one gap call names.
GAP_LIST_MAX: Final = 20

#: The most characters one figure key or claim id a gap call names holds.
GAP_NAME_MAX_CHARS: Final = 500

#: How many hex digits of the sha256 a gap id carries.
_GAP_DIGITS: Final = 16


def analyst_gap_id(snapshot_id: str, *parts: str) -> str:
    """The id one gap is stored under: its snapshot and the parts that make it the gap it is.

    A model's call is named by ``(need, why)`` for ``missing``, the id ``capability_gap`` has always returned, and by
    ``(need, why, kind)`` for any other kind; a notice by its kind and what it records.

    :param snapshot_id: The snapshot the gap was noted against.
    :ptype snapshot_id: str
    :param parts: What else names it, in order.
    :ptype parts: str
    :return: ``gap:`` and 16 hex digits of the sha256 of the snapshot and the parts, one per line.
    :rtype: str
    """
    return "gap:" + bytes_digest("\n".join((snapshot_id, *parts)).encode())[:_GAP_DIGITS]


def call_gap_id(snapshot_id: str, kind: str, need: str, why: str) -> str:
    """The id of a model's ``capability_gap`` call (:func:`analyst_gap_id`), the one its acknowledgement names.

    :param snapshot_id: The snapshot the analyst was reading.
    :ptype snapshot_id: str
    :param kind: ``missing`` or ``suspect``.
    :ptype kind: str
    :param need: What it needed, or what looked wrong.
    :ptype need: str
    :param why: Why.
    :ptype why: str
    :return: The id.
    :rtype: str
    """
    return analyst_gap_id(snapshot_id, need, why) if kind == "missing" else analyst_gap_id(snapshot_id, need, why, kind)


class GapToolCall(EvalDocumentModel, CoreShapeModel):
    """One tool call that reproduces a ``suspect`` result: the action and its arguments, as the model would send them."""

    action: str = Field(min_length=1, description="The analyst tool's action, as `action` names it.")
    arguments: VerbatimJsonObject = Field(
        default_factory=dict, description="The call's other arguments, exactly as given; empty for none."
    )


class GapVerdict(EvalDocumentModel, CoreShapeModel):
    """One verdict a bounce carried at a key's field: the check, its field path, and the fault in the check's words."""

    invariant: int = Field(
        ge=0, description="The check: 0 validity, 1 to 6 the invariants (a strict build's under its field's)."
    )
    path: str = Field(description="The field path it sits at.")
    fault: str = Field(min_length=1, description="What the check found there, as it said it.")


class GapBounce(EvalDocumentModel, CoreShapeModel):
    """An ``unresolved_key`` notice's evidence: the key, where the draft wrote it, and what the bounce said there."""

    submission: int = Field(
        ge=1,
        description="Which of the write's drafts that reached its checker it was, counted from 1 across its rounds.",
    )
    path: str = Field(min_length=1, description="The field the draft wrote the key in.")
    key: str = Field(min_length=1, description="The key as the draft wrote it, braces and filters included.")
    verdicts: list[GapVerdict] = Field(description="Every verdict the submission drew at that field.")


class GapEditorItem(EvalDocumentModel, CoreShapeModel):
    """An ``editor_score`` notice's evidence: the rubric item as the editor scored it, and its review's ``must_fix``."""

    round: int = Field(ge=1, description="The review round.")
    n: int = Field(ge=1, description="The item's number in the rubric.")
    name: str = Field(min_length=1, description="The item's name, as the rubric names it.")
    score: int = Field(ge=0, description="Its score, 0 or 1 of 2.")
    reason: str = Field(description="Why, in the editor's words.")
    must_fix: bool = Field(description="Whether the review that scored it held a must_fix finding.")


class EvalAnalystGap(DurableDocumentModel):
    """One capability gap or notice, with where it came from (the module docstring)."""

    id: str = Field(min_length=1, description="The gap's id (analyst_gap_id): the same call names the same id.")
    doc_type: Literal["eval_analyst_gap"] = "eval_analyst_gap"
    schema_version: DurableSchemaVersion = DURABLE_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    kind: StoredGapKind = Field(description="What it records (GapKind).")
    need: str = Field(
        min_length=1,
        description="What was needed and no tool gave, or what looked wrong; a notice's is code's words, naming its key or item.",
    )
    why: str = Field(
        min_length=1, description="What it was needed for; a notice's is the verdict or the editor's reason."
    )
    workaround: str | None = Field(default=None, description="What the caller did instead; None for a notice.")
    claim_ids: list[str] = Field(default_factory=list, description="The claims of the report it touches.")
    figure_keys: list[str] = Field(
        default_factory=list, description="The figure keys that show it: a suspect result's, or the unresolved key."
    )
    calls: list[GapToolCall] = Field(
        default_factory=list, description="The tool calls that reproduce a suspect result."
    )
    bounce: GapBounce | None = Field(default=None, description="An unresolved_key notice's evidence; None otherwise.")
    editor_item: GapEditorItem | None = Field(
        default=None, description="An editor_score notice's evidence; None otherwise."
    )
    must_fix: bool = Field(
        default=False,
        description=(
            "Whether a must_fix finding is about it: a review's must_fix finding names a claim a model's gap touches. "
            "A finding names no rubric item, so an editor_score notice is never linked; its item says whether its "
            "review held one."
        ),
    )

    snapshot_id: str = Field(min_length=1, description="The snapshot it was noted against.")
    report_doc_id: str | None = Field(
        default=None, description="The report the write stored; None outside a write, or for a write that stored none."
    )
    role: str | None = Field(
        default=None, description="The writer role of the session it came from; None outside a write."
    )
    session: str | None = Field(
        default=None, description="The session's name (`analyst`, `editor-<round>`); None outside a write."
    )
    model: str | None = Field(default=None, description="The model the session ran on; None outside a write.")
    transcript_digest: str | None = Field(
        default=None,
        description="The sha256 of the session's messages at its last reply, as the write's tally holds it; None outside a write.",
    )
    created_by: str = Field(min_length=1, description="Who asked, as the host names the caller.")
    created_at: str = Field(default_factory=utc_now_iso, description="When it was stored, ISO 8601 in UTC.")

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "eval_analyst_gap":
            raise ValueError(f"doc_type must be 'eval_analyst_gap', got '{v}'")
        return v


class GapGroup(EvalBaseModel):
    """One group of the digest: the records of one kind whose needs normalize alike, and their issue text."""

    kind: str = Field(description="The records' kind (missing, suspect, unresolved_key, editor_score).")
    need: str = Field(description="The need as its first record wrote it.")
    normalized: str = Field(
        description="The need as the digest normalizes it (``analysis/analyst_gaps.py``), which the group is keyed by."
    )
    count: int = Field(ge=1, description="How many records the group holds within the digest's window.")
    first_seen: str = Field(description="The first record's time.")
    last_seen: str = Field(description="The last record's time.")
    must_fix: bool = Field(description="Whether a must_fix finding is about any of its records.")
    snapshots: list[str] = Field(description="The snapshots its records were noted against, in first-seen order.")
    reports: list[str] = Field(description="The stored reports its records came from, in first-seen order.")
    gap_ids: list[str] = Field(description="Its records' ids, in first-seen order.")
    title: str = Field(description="An issue title for the group.")
    body: str = Field(description="An issue body for the group: what, why, and each record's evidence, in Markdown.")


class GapDigest(EvalBaseModel):
    """The scope's analyst gaps and notices, grouped and ranked, each group with its issue text."""

    since: str | None = Field(description="The time the records were read from, as given; None for every record.")
    records: int = Field(ge=0, description="How many records the digest read.")
    groups: list[GapGroup] = Field(
        description="The groups, in rank order: suspect, then must_fix-linked, then the rest."
    )

    def render(self) -> str:
        """The digest as text: how many records it read, then each group's title and body, in rank order.

        :return: The text.
        :rtype: str
        """
        window = f"since {self.since}" if self.since is not None else "every record"
        head = f"analyst gaps ({window}): {self.records} records in {len(self.groups)} groups"
        return "\n\n".join([head, *(f"## {group.title}\n\n{group.body}" for group in self.groups)])


__all__ = [
    "GAP_LIST_MAX",
    "GAP_NAME_MAX_CHARS",
    "GAP_TEXT_MAX_CHARS",
    "CallGapKind",
    "EvalAnalystGap",
    "GapBounce",
    "GapDigest",
    "GapEditorItem",
    "GapGroup",
    "GapToolCall",
    "GapVerdict",
    "analyst_gap_id",
    "call_gap_id",
]
