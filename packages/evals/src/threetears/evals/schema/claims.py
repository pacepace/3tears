"""The text a candidate produced, as claims grading reads it: an ordered list of segments, some of them units."""

from __future__ import annotations

from pydantic import ConfigDict, Field

from threetears.evals.schema.base import CoreShapeModel, EvalDocumentModel

__all__ = ["Segment"]


class Segment(EvalDocumentModel, CoreShapeModel):
    """One piece of a candidate's text, in the order the host cut it: a unit with its index, or any other piece.

    A unit is what the host's per-unit readings count by: an answer, an entry, a section, one item of a sequence. The
    claims in every segment are graded; only units make the per-unit denominator.
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(pattern=r"\S", description="The text of this piece, as the candidate's output holds it.")
    unit: int | None = Field(
        default=None,
        ge=0,
        description="The index of the unit this piece is, when it is one: a cell's units are numbered 0..n-1, each "
        "once. None for any other piece (a preamble, an aside, a message between units), whose claims count but "
        "which is no unit.",
    )
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="The host's own labels for this piece (an entity id, a section name), opaque to the engine and "
        "never shown to a model. A host's derived readings read them.",
    )
