"""What each stored document type is: where it comes from, whether it is kept, and what it suppresses.

The engine stores twenty document types through one :class:`~threetears.evals.schema.store_port.DocumentStore`,
and three questions about each one decide how it may be kept: **where it comes from** (an input a run reads as given,
a record of what happened, or a document computed from other stored documents), **whether a later release reads it**
(kept, and upgraded across core versions; or regenerable, refused after a regenerable bump and made again), and, for a
tombstone, **which type it suppresses**. :data:`DOC_TYPE_FACTS` answers them, one entry per type, and every list of
types the engine keeps is derived from it: :data:`~threetears.evals.schema.models.EVAL_DOC_TYPES` (what the operator
wipe sweeps) is its keys, and :data:`~threetears.evals.schema.versioning.CORE_DOC_TYPES` (what is kept) its ``kept``
entries. Two hand-kept lists could disagree; one registry and two derivations cannot. ``tests/test_doc_type_facts.py``
ties it to the model graph, so a stored model whose ``doc_type`` is not registered exactly once goes red.

**The facts are what a host partitions its storage by.** A host that keeps its evidence in one database and what it
can remake in a cheaper one routes by ``lifecycle``; a host that versions what people author apart from what runs
record routes by ``origin``. :class:`~threetears.evals.storage.RoutedDocumentStore` takes such a rule over these facts
and never over a type's name, so a type added later is placed by what it is, without the host naming it.

**Origin is about the document, not its author.** An ``input`` is what a run reads as given — a template, a judge
config, a case set — whether a person typed it or a proposer drafted it; a campaign's declared design is an input
because it states what the campaign set out to learn before anything ran. A ``record`` says what happened: a run, its
results and traces, a rating someone gave, what a sweep or an analysis attempt came to, what a tool answered while a
capture ran. A ``derived`` document is computed from other stored documents, and computing it again gives the same
reading.

**Lifecycle follows from whether the document can be made again**, not from its origin. A ``derived`` document is
always regenerable. An input or a record is kept unless the engine can make it again without loss: two records are
regenerable: the cassette, whose replacement is a fresh capture under a new corpus, and a campaign's generation
slot, a claim the next start makes again (its holder's attempt, which is kept, says whether the claim still stands) (see
:mod:`threetears.evals.schema.versioning`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, get_args

#: Where a stored document comes from: ``input`` — what runs read as given, whoever wrote it; ``record`` — what
#: happened; ``derived`` — computed from other stored documents.
DocOrigin = Literal["input", "record", "derived"]

#: Whether a later release reads a stored document: ``kept`` — in the core, read from the core baseline on and
#: upgraded across core versions; ``regenerable`` — refused on read after a regenerable schema bump, and made again.
DocLifecycle = Literal["kept", "regenerable"]


@dataclass(frozen=True)
class DocTypeFacts:
    """The facts of one stored document type.

    Attributes:
        origin: Where a document of this type comes from.
        lifecycle: Whether a later release reads it (``kept``) or it is made again (``regenerable``).
        suppresses: For a tombstone, the ``doc_type`` it suppresses: the definition whose slot it holds so a seed
            does not write a deleted key back. ``None`` for every other type. A tombstone and the type it
            suppresses are read together, so they share their origin and lifecycle (``tests/test_doc_type_facts.py``)
            and whatever separates documents by facts must keep them in one place.
    """

    origin: DocOrigin
    lifecycle: DocLifecycle
    suppresses: str | None = None

    def __post_init__(self) -> None:
        """Refuse a value outside the vocabulary, and a derived document that claims to be kept.

        Raises:
            ValueError: ``origin`` or ``lifecycle`` is not one of its values, or a ``derived`` type is ``kept``:
                what is computed from stored documents is computed again, never migrated.
        """
        if self.origin not in get_args(DocOrigin):
            raise ValueError(f"origin is one of {get_args(DocOrigin)}, not {self.origin!r}")
        if self.lifecycle not in get_args(DocLifecycle):
            raise ValueError(f"lifecycle is one of {get_args(DocLifecycle)}, not {self.lifecycle!r}")
        if self.origin == "derived" and self.lifecycle == "kept":
            raise ValueError("a derived document is computed again from what it was derived from, so it is regenerable")


#: Every ``doc_type`` the engine stores, and its facts. The one list: :data:`EVAL_DOC_TYPES` and
#: :data:`CORE_DOC_TYPES` are derived from it, and a stored model whose type is missing here fails
#: ``tests/test_doc_type_facts.py``. The order is the order :data:`EVAL_DOC_TYPES` sweeps in.
DOC_TYPE_FACTS: Final[Mapping[str, DocTypeFacts]] = MappingProxyType(
    {
        # What runs read as given.
        "eval_template": DocTypeFacts(origin="input", lifecycle="kept"),
        "judge_config": DocTypeFacts(origin="input", lifecycle="kept"),
        "rubric_dim": DocTypeFacts(origin="input", lifecycle="kept"),
        "rubric_dim_tombstone": DocTypeFacts(origin="input", lifecycle="kept", suppresses="rubric_dim"),
        "judge_config_tombstone": DocTypeFacts(origin="input", lifecycle="kept", suppresses="judge_config"),
        "case_set": DocTypeFacts(origin="input", lifecycle="kept"),
        "eval_test_case": DocTypeFacts(origin="input", lifecycle="kept"),
        "eval_campaign": DocTypeFacts(origin="input", lifecycle="kept"),
        # What happened.
        "eval_run": DocTypeFacts(origin="record", lifecycle="kept"),
        "eval_result": DocTypeFacts(origin="record", lifecycle="kept"),
        "eval_trace": DocTypeFacts(origin="record", lifecycle="kept"),
        "calibration_rating": DocTypeFacts(origin="record", lifecycle="kept"),
        "eval_out_of_run_spend": DocTypeFacts(origin="record", lifecycle="kept"),
        "eval_sweep": DocTypeFacts(origin="record", lifecycle="kept"),
        "eval_analysis_attempt": DocTypeFacts(origin="record", lifecycle="kept"),
        "eval_cassette": DocTypeFacts(origin="record", lifecycle="regenerable"),
        "eval_generation_slot": DocTypeFacts(origin="record", lifecycle="regenerable"),
        # Computed from the documents above.
        "eval_analysis": DocTypeFacts(origin="derived", lifecycle="regenerable"),
        "eval_insight": DocTypeFacts(origin="derived", lifecycle="regenerable"),
        "eval_judge_profile": DocTypeFacts(origin="derived", lifecycle="regenerable"),
    }
)


__all__ = ["DOC_TYPE_FACTS", "DocLifecycle", "DocOrigin", "DocTypeFacts"]
