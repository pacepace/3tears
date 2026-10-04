"""The toy extractor's kind contract: what a launch may turn, and what a template states — the whole declaration.

This is the path a product follows, and it is short on purpose. Write the knobs a launch may turn as
one plain Pydantic model, a field each with a default and a description, and what a template states
for the kind as another; wrap both in a :class:`~threetears.evals.contracts.host.KindContract` named
for the kind, and name the contract on the profile's ``kinds`` (``profile.py``) — once, and nowhere
else. Everything else is the engine's: the profile adds the contract's levers to the registry every
lens reads, a launch's overlays are validated against the overlay model before any run exists and
refused by field, a template's ``kind_spec`` against the spec model where it is authored and again
where it launches, both are frozen onto each run whole (defaults included), every overlay field
reaches every lens as the lever ``extractor.<field>`` and the engine resolves each run's level of it
into the variant key, and the frozen spec is part of the run's measurement context.

The overlays:

Each field is one shape the engine reads differently:

| Field | Shape | Read as |
|---|---|---|
| ``prompt_style`` | a ``Literal`` marked ``Ordinal()`` | ranked levels, in declaration order |
| ``page_limit`` | an ``int`` marked ``Interval(unit=...)`` | a number with real spacing, and its unit |
| ``instructions`` | a ``str`` | one level per distinct text, joined by content |
| ``field_aliases`` | a ``dict[str, str]`` | an open family: one lever per key a launch sets |

The scripted extractor answers per ``(model, document)`` and reads none of these, so they are a
batch's recorded settings exactly as ``chunk_tokens`` is — what the fixture pins is the declaration
path, not an effect on the script.

The spec is which invoice fields a template grades, and the kind honours it: a launched run's
extractor scores only those, so a template grading two fields and one grading four are two
measurement conditions, and their runs' context keys say so.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from threetears.evals.contracts.host import Interval, KindContract, Ordinal
from packages.evals.tests.fixtures.toyhost.kind import INVOICE_FIELDS, TOY_EXTRACTOR_KIND, InvoiceField


class ExtractorOverlays(BaseModel):
    """The knobs a launch of the toy extractor may turn."""

    prompt_style: Annotated[Literal["terse", "standard", "verbose"], Ordinal()] = Field(
        "standard", description="how much instruction the extraction prompt carries"
    )
    page_limit: Annotated[int, Interval(unit="pages")] = Field(
        10, ge=1, le=50, description="how many pages of a document the extractor reads"
    )
    instructions: str = Field("", description="the free-text instructions prepended to the extraction prompt")
    field_aliases: dict[str, str] = Field(
        default_factory=dict, description="the other labels the extractor is told a field may appear under"
    )


class ExtractorSpec(BaseModel):
    """What a toy-extractor template states for its kind: the invoice fields it grades."""

    graded_fields: list[InvoiceField] = Field(default_factory=lambda: list(INVOICE_FIELDS), min_length=1)


#: The toy extractor's contract. ``prefix`` names its levers ``extractor.<field>`` rather than after
#: the kind's own hyphenated name — a dotted, readable name is also what a pivot reads as an axis.
TOY_EXTRACTOR_CONTRACT = KindContract(
    TOY_EXTRACTOR_KIND, overlays=ExtractorOverlays, spec=ExtractorSpec, prefix="extractor"
)


__all__ = ["TOY_EXTRACTOR_CONTRACT", "ExtractorOverlays", "ExtractorSpec", "InvoiceField"]
