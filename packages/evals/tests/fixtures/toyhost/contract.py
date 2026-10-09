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
| ``page_limit`` | an ``int`` marked ``Interval(unit=...)`` and ``ActsOn(...)`` | a number with real spacing, its unit, and the covariate it should move |
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

from threetears.evals.contracts.host import ActsOn, Interval, KindContract, Ordinal
from packages.evals.tests.fixtures.toyhost.kind import INVOICE_FIELDS, TOY_EXTRACTOR_KIND, InvoiceField


class ExtractorOverlays(BaseModel):
    """The knobs a launch of the toy extractor may turn."""

    prompt_style: Annotated[Literal["terse", "standard", "verbose"], Ordinal()] = Field(
        "standard", description="how much instruction the extraction prompt carries"
    )
    # More pages read is more context carried into extraction, so the knob names the engine covariate it
    # should move; the bundle checks it moved, exactly as for a lever the host declares itself.
    page_limit: Annotated[int, Interval(unit="pages"), ActsOn("context_tokens_in")] = Field(
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
#:
#: **Its seats are the rig it has, and nothing else.** The extractor is graded against an adjudicated key by
#: a comparison rule (``grader_version``), which pool of humans adjudicated the key is ``reviewer_pool`` (the
#: host's ``adjudicator`` role), its pages are read by an OCR engine (``ocr_engine_version``), and it calls a
#: model under a spend ceiling (``max_cost_usd``). Its layout dimension is scored by that reviewer pool, with
#: no versioned judge configuration, and the core records that as a level (``judge_config_ids``), so it is
#: seated too. It is scored by no model and talks to nobody, so the core's model-judge axes and simulator axes
#: are not blanks on its runs but things they do not have — and so is any dimension added to the rig later,
#: until this contract claims it.
TOY_EXTRACTOR_CONTRACT = KindContract(
    TOY_EXTRACTOR_KIND,
    overlays=ExtractorOverlays,
    spec=ExtractorSpec,
    prefix="extractor",
    seats=frozenset({"grader_version", "judge_config_ids", "adjudicator", "ocr_engine_version", "max_cost_usd"}),
)


__all__ = ["TOY_EXTRACTOR_CONTRACT", "ExtractorOverlays", "ExtractorSpec", "InvoiceField"]
