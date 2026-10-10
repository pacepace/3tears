"""Typed viz payloads — the contract a generated chart is validated against.

``Viz.payload`` stays an OPEN dict on the stored model, because a stored analysis
must keep loading when the type set moves ahead of it. These models are the
*generation-time* gate over that dict: the payload shape is declared once, in
code, and a generation that does not match it is refused while it can still be
retried.

That timing is the whole point. Payload conformance validated at render is
conformance discovered by a reader — and a generation is billed per run, so the
correction costs another paid call. Structurally malformed payloads reach a
reader when the contract lives only in a prompt and a browser-side mirror,
and each degrades to an empty box rather than an error.

**Only types with a model here are validated.** :data:`PAYLOAD_MODELS` is the
registry, and a type absent from it is deliberately unchecked rather than
silently accepted — :func:`parse_payload` says which case it is, so an
unvalidated type is a visible gap and not a passing test.
"""

from __future__ import annotations

import math
from datetime import date
from collections.abc import Iterable
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from threetears.evals.analysis.reporting import NULL_LEVEL
from threetears.evals.kernel.host.style import SERIES_SLOTS, VALIDATED_SLOTS
from threetears.evals.schema.prose import ModelProse
from threetears.evals.kernel.metrics import Materiality, MeasureScale


#: How far a breakdown's parts may miss the `total` they claim to make up, as a
#: fraction of the larger quantity involved.
#:
#: Not zero: a payload's figures can arrive rounded (an analysis from before references carries
#: the generator's typed figures), and eight shares rounded to one
#: decimal will not re-sum to exactly 100. Wide enough to absorb that, far too
#: narrow to absorb a dropped part — the defect these checks exist to catch.
#:
#: **One reconciliation, one constant, and the split is the point.** This was a single
#: `_TOTAL_TOLERANCE` serving all three of the reconciliations below, so widening it to
#: absorb rounded shares — the only reason anyone would touch it — silently widened two
#: attribution guards that have nothing to do with rounded percentages. The three now
#: hold the same value and move independently; that they agree today is a coincidence
#: of calibration rather than a shared meaning.
_BREAKDOWN_TOTAL_TOLERANCE = 0.005

#: How far a movement's two levels may miss the `delta` between them, as a fraction of
#: the movement — the proportional term of two.
#:
#: :data:`_LEVEL_ROUNDING` adds the second, scaled off the LEVELS, because there the
#: disagreement and the rounding have different bases. Widening this one widens what a
#: movement may misstate proportionally, and nothing else.
_MOVEMENT_DELTA_TOLERANCE = 0.005

#: How far an attribution's quantified remainder may miss the arithmetic it claims,
#: as a fraction of the larger movement it sits between.
#:
#: The whole tolerance for that check — no rounding term, because a remainder is stated
#: rather than derived from two rounded levels.
_REMAINDER_TOLERANCE = 0.005

#: Rounding allowance for a movement's level/delta reconciliation, as a fraction of
#: the LEVELS rather than of the movement between them.
#:
#: A purely proportional tolerance demands ever-finer agreement as a movement shrinks,
#: until a delta the generator quoted at ordinary precision cannot satisfy it. The
#: allowance has to come from the levels because that is where the rounding happens —
#: a delta derived from two numbers written to four significant figures inherits their
#: last place, not its own.
#:
#: **Deliberately not an absolute constant.** The payload's `unit` is free text, so a
#: fixed slack is a different claim in every unit: 0.5 is invisible in `ms` and larger
#: than the whole quantity in `%` or `usd`, which turned this guard off for exactly
#: the payloads whose numbers are small. A fraction of the levels means the same thing
#: whatever they measure.
#:
#: The magnitude is a **judgment about the precision a generator quotes at** — roughly
#: four significant figures — and nothing in this package enforces that, so it is
#: stated as the estimate it is rather than derived from a rule. In particular it is
#: NOT pinned to `format_number`: that runs at display time, after this validation,
#: and its integer branch renders exactly, so it constrains what a reader sees and
#: never what the generator wrote.
#:
#: Known residual, accepted: this reinstates a level-scaled term, which is the shape
#: that made an earlier version of the movement guard inert. **It is not bounded by
#: the proportional term, and the ratio runs the other way in the ordinary case.**
#: Term for term, level/proportional is ``(max(|a|,|b|) / max(|delta|,|b-a|)) / 50``,
#: so the level term is the smaller of the two wherever the levels are under ~50x the
#: movement, and it dominates past that. For the worked example below the levels exceed
#: the movement ~8800x — 176x the proportional term — and it is then essentially the
#: whole tolerance: a 5 ms move between two ~44 s levels tolerates +/-4.4 ms.
#: Left alone because this type charts swings three orders of magnitude larger, so the
#: unguarded band sits far under anything it draws; tighten it if a payload ever needs
#: to state a movement that small. (An earlier note here claimed the 1/50 bound as a
#: general fact — it contradicted the very example beneath it, which is the tell.)
_LEVEL_ROUNDING = 1e-4


class PayloadError(ValueError):
    """A viz payload does not match the contract declared for its type.

    Carries the offending field path in the message: the reader of this error is
    either an operator deciding whether to regenerate or a developer fixing the
    compiler, and neither can act on "invalid payload".
    """


class _VizPayload(BaseModel):
    """Base for every typed payload — strict, because this IS the gate.

    ``extra="forbid"`` is the deliberate opposite of the stored model's open
    dict. A key the contract does not name is either a typo for one it does (the
    observed failure — ``cells`` for ``points``, ``pair`` for ``rows``,
    ``stop_causes`` for ``groups``) or a field nothing renders. Both are worth an
    error at generation time; neither is worth an empty box at read time.

    ``caption`` lives HERE rather than on each type, and that is the point: it is
    the same field with the same meaning on all of them, so the contract widens
    once and a type added later inherits it without anyone remembering to.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    caption: ModelProse | None = Field(
        default=None,
        description=(
            "The editorial line beside the chart: a conclusion, a limit on one, or a caveat the reader would not "
            "otherwise reach. Written by whoever has the analytical context, because a compiler can only narrate "
            "how it drew. It is shown as written and never extended: the compiler's own disclosures — what this "
            "compilation truncated, filtered or restated, which is not knowable when this payload is written — "
            "travel as separate lines after it. None where the chart needs no editorial line; an empty string reads the same "
            "way, since a caption nobody wrote and a caption of no words are the same absence."
        ),
    )


class BreakdownPart(BaseModel):
    """One named part of a whole."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    label: str = Field(
        min_length=1, description="The part's name — this is what identifies it in the chart, not its colour."
    )
    value: float = Field(description="The part's magnitude, in the payload's `unit`.")
    n: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Test cases behind this part, when the payload counts them; a case judged k times is one."
            " A payload compiled before 0.66 counted observations here."
        ),
    )

    @field_validator("value")
    @classmethod
    def _finite_and_non_negative(cls, value: float) -> float:
        """Reject a part that cannot be drawn as a share of a whole."""
        if not math.isfinite(value):
            raise ValueError("must be a finite number")
        if value < 0:
            # A negative part is not a small part — it is a different chart. Drawn as a
            # bar it either inverts across the baseline or vanishes, and either way the
            # reader is shown a part-to-whole that the values do not describe.
            raise ValueError("must not be negative — a negative share is not a part of a whole")
        return value


class BreakdownPayload(_VizPayload):
    """A part-to-whole finding: named components of one measured quantity.

    The shape that recurs most and had no type to hold it — stop causes,
    synthesis paths, any "what did the time/the failures/the runs divide into".

    Drawn as a sorted horizontal bar rather than a stacked bar or a donut, and
    that choice is a constraint on this payload rather than a rendering
    preference: the categorical palette validates to four slots, and a stacked bar
    of nine causes would draw the ninth in the first one's hue. A sorted bar puts
    identity on the axis labels, which has no such ceiling and keeps identity off
    colour entirely.
    """

    parts: list[BreakdownPart] = Field(description="The components, in any order — the compiler sorts them.")
    unit: str = Field(
        min_length=1, description="The unit every `value` is in, carried as data so no reader infers it from a name."
    )
    measure: str | None = Field(
        default=None, description="What was broken down, when it has a name in the measure catalog."
    )
    total: float | None = Field(
        default=None, description="The whole the parts are of, when the generator knows it independently."
    )
    total_n: int | None = Field(default=None, ge=0, description="Observations behind the whole.")

    @field_validator("parts")
    @classmethod
    def _at_least_two_distinct_parts(cls, parts: list[BreakdownPart]) -> list[BreakdownPart]:
        """Reject a breakdown that does not break anything down."""
        if len(parts) < 2:
            # One part is the whole restated; drawn, it is a single bar at 100%, which
            # the form heuristic names as a non-chart. The finding wanted a number.
            raise ValueError(f"needs at least 2 parts to be a breakdown, got {len(parts)}")
        labels = [part.label for part in parts]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            # Two parts sharing a name draw as two bars a reader cannot tell apart, and
            # the sort order then decides which is which — so the chart's meaning would
            # depend on the order the generator happened to emit.
            raise ValueError(f"part labels must be unique; duplicated: {', '.join(duplicates)}")
        return parts

    @model_validator(mode="after")
    def _parts_account_for_the_total(self) -> BreakdownPayload:
        """Reject parts that do not add up to the total they claim to divide.

        A shortfall means a part was dropped, and the chart would then present a
        partial division as a complete one — the reader sees shares that look
        exhaustive and are not. Where the remainder is genuinely unplaceable it
        belongs in a type that can say so, not in a total this silently absorbs.
        """
        if self.total is None:
            return self
        if not math.isfinite(self.total):
            raise ValueError("total must be a finite number")
        summed = math.fsum(part.value for part in self.parts)
        slack = abs(self.total) * _BREAKDOWN_TOTAL_TOLERANCE
        if abs(summed - self.total) > slack:
            raise ValueError(
                f"total {self.total:g} does not match the parts, which sum to {summed:g} "
                f"(tolerance ±{slack:g}) — a part is missing, or the total is not the whole these divide"
            )
        return self


class ConfidenceInterval(BaseModel):
    """A value's uncertainty band — a point estimate is never drawn without one.

    ``variability`` is carried HERE rather than on the payload, because an
    interval and the thing it varies over travel together or the source drifts:
    two arms of one comparison can legitimately span different things, and that
    difference is exactly when they must not be read as comparable. Required, like
    the coverage level: code fills every interval from the decision surface
    (:mod:`threetears.evals.analysis.viz_refs`) and states both, so an interval that
    cannot say what it covers or spans is not one the engine draws.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    low: float = Field(description="The interval's lower bound.")
    high: float = Field(description="The interval's upper bound.")
    mean: float = Field(description="The point estimate the interval is around.")
    level: float = Field(
        gt=0.0,
        lt=1.0,
        description=(
            "Coverage level, e.g. 0.95 — a fraction, never a percentage. The compiled caption names it beside what "
            "the interval spans, because a 50% band and a 95% band over the same values are different widths."
        ),
    )
    variability: str = Field(
        min_length=1,
        description="What the interval's variability is over — 'across 5 runs', 'across the 12 cases'.",
    )

    @model_validator(mode="after")
    def _bounds_are_finite_and_ordered(self) -> ConfidenceInterval:
        """Reject an interval that cannot be drawn as a span around its estimate."""
        for name, value in (("low", self.low), ("high", self.high), ("mean", self.mean)):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.low > self.high:
            raise ValueError(f"low {self.low:g} exceeds high {self.high:g} — the bounds are the wrong way round")
        if not self.low <= self.mean <= self.high:
            # A mean outside its own interval is not a wide interval, it is two
            # different quantities reported as one. Drawn, the estimate marker
            # lands off the band and the reader reconciles it as a rendering bug.
            raise ValueError(f"mean {self.mean:g} lies outside its interval [{self.low:g}, {self.high:g}]")
        return self


class HistogramBucket(BaseModel):
    """One pre-binned count — a shape recorded coarsely, when raw values were not kept.

    **A bin with edges can be placed; a bin with only a label cannot.** ``range`` is prose and is never
    parsed: reading ``45–50k`` as the numbers 45000 and 50000 would mean guessing a unit, a separator
    and a suffix the payload never stated. So a bin reaches the shared value axis only when its producer
    states ``low`` and ``high`` in the payload's ``unit``, and a group whose bins carry only labels is
    named as unplaceable and kept, exact, in the values table.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    range: str = Field(
        min_length=1,
        description=(
            "The bin's human range label in the payload's `unit`, e.g. '0.6–0.7'. A label, not a number pair: "
            "it is never parsed, so a bin is placed on the value axis only by `low` and `high`."
        ),
    )
    count: int = Field(ge=0, description="Observations that fell in the bin.")
    low: float | None = Field(
        default=None,
        description=(
            "The bin's lower edge, in the payload's `unit`. Given together with `high` or not at all; a bin "
            "with edges is drawn on the shared value axis, one with only its label is not."
        ),
    )
    high: float | None = Field(
        default=None, description="The bin's upper edge, in the payload's `unit`, above `low`. Given with `low`."
    )

    @model_validator(mode="after")
    def _edges_are_a_bin(self) -> HistogramBucket:
        """Reject half a bin, an edge that is not a number, and a bin of no width.

        One edge alone does not locate a bin, and the missing one would have to be invented from the
        label — the inference this field exists to make unnecessary.
        """
        if (self.low is None) != (self.high is None):
            raise ValueError(f"bin {self.range!r} states one edge — give both `low` and `high`, or neither")
        if self.low is not None and self.high is not None:
            if not (math.isfinite(self.low) and math.isfinite(self.high)):
                raise ValueError(f"bin {self.range!r} has an edge that is not a finite number")
            if not self.low < self.high:
                raise ValueError(f"bin {self.range!r} has `low` {self.low} not below `high` {self.high}")
        return self

    @property
    def edged(self) -> bool:
        """Whether the bin states numeric edges, and so can be placed on a value axis."""
        return self.low is not None


class DistributionGroup(BaseModel):
    """One group's spread — raw samples, pre-binned buckets, or an interval alone."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    label: str = Field(min_length=1, description="The group's name — the chart's identity channel.")
    samples: list[float] | None = Field(default=None, description="Raw values, drawn as a countable strip of dots.")
    buckets: list[HistogramBucket] | None = Field(
        default=None, description="Pre-binned counts, drawn as a density strip."
    )
    ci: ConfidenceInterval | None = Field(default=None, description="The group's interval — never a spread-free point.")
    n: int | None = Field(
        default=None,
        ge=0,
        description="Test cases behind the group; a case judged k times is one. A payload compiled before 0.66 counted observations here.",
    )

    @field_validator("samples")
    @classmethod
    def _samples_are_finite(cls, samples: list[float] | None) -> list[float] | None:
        """Reject samples that cannot be placed on a value axis."""
        if samples is None:
            return None
        if any(not math.isfinite(value) for value in samples):
            raise ValueError("every sample must be a finite number")
        return samples

    @model_validator(mode="after")
    def _has_something_to_draw(self) -> DistributionGroup:
        """Reject a group carrying no spread at all.

        Samples, buckets or an interval — any one of the three is drawable, and a
        group with none of them contributes a label and nothing else. That is not
        a thin group, it is a row of empty axis, and the reader cannot tell it
        apart from a group whose values were all zero.

        **A lone sample satisfies that literally and carries no spread**, which is
        the gap this second clause closes. Two groups each reporting ``n=4`` with one
        sample and no interval draw a figure that is a title, two labels and two 1px
        ticks clipped at opposite edges of a cropped axis — indistinguishable from a
        failed render. The generator's own prompt
        already says to *"always carry the ``ci`` — never a spread-free point"*, so
        this is guidance the validator was not enforcing rather than a new rule; the
        argument for enforcing it in the bundle rather than in the prompt is the same
        one the scope rules were built on.

        Two samples at the same value are still admitted. Zero variance is a real
        finding and a rug of two coincident ticks states it; refusing it would be
        this validator inventing a claim about the data.
        """
        if not self.samples and not self.buckets and self.ci is None:
            raise ValueError(f"group {self.label!r} carries no samples, buckets or ci — there is no spread to draw")
        buckets = self.buckets or []
        if any(bucket.edged for bucket in buckets) and not all(bucket.edged for bucket in buckets):
            # A group is placed whole or not at all: drawing the edged bins and dropping the rest would show a
            # shape with a hole where the label-only bins' counts went.
            raise ValueError(
                f"group {self.label!r} gives edges for some bins and not others — give every bin `low` and `high`, "
                "or none"
            )
        edged = sorted(
            (bucket.low, bucket.high, bucket.range)
            for bucket in buckets
            if bucket.low is not None and bucket.high is not None
        )
        for (_, high, name), (low, _, following) in zip(edged, edged[1:], strict=False):
            if low < high:
                raise ValueError(
                    f"group {self.label!r} has bins {name!r} and {following!r} that overlap — an observation "
                    "would be counted in both"
                )
        if self.ci is None and not self.buckets and len(self.samples or []) < 2:
            raise ValueError(
                f"group {self.label!r} carries a single sample and no ci or buckets — a distribution is a spread, "
                "and one point drawn on a shared axis is a tick the reader cannot distinguish from a failed render; "
                "carry the ci, or the samples the estimate was computed from"
            )
        return self


class DistributionPayload(_VizPayload):
    """Per-group spreads over one shared value axis.

    One ``unit`` for the whole payload rather than one per group, and that is the
    one-unit-per-quantity rule expressed as a shape: the groups share a value
    axis, so quantities in two units drawn on it would be two rulers read as one.
    """

    groups: list[DistributionGroup] = Field(description="The groups compared, in the order they should be read.")
    unit: str | None = Field(
        default=None,
        description="The unit every plotted value is in. None when the quantity is unitless.",
    )
    x_label: str | None = Field(default=None, description="What the value axis measures, when it has a name.")

    @field_validator("groups")
    @classmethod
    def _at_least_one_group(cls, groups: list[DistributionGroup]) -> list[DistributionGroup]:
        """Reject a distribution with nothing in it, or with two groups a reader cannot tell apart."""
        if not groups:
            raise ValueError("needs at least 1 group; an empty distribution has nothing to draw")
        labels = [group.label for group in groups]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(f"group labels must be unique; duplicated: {', '.join(duplicates)}")
        return groups


class NullResultArm(BaseModel):
    """One arm of a null result. Its interval is what the chart draws, so it is required."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    label: str = Field(min_length=1, description="The arm's name.")
    ci: ConfidenceInterval = Field(description="The arm's interval — this type has nothing to draw without it.")
    n: int | None = Field(
        default=None,
        ge=0,
        description="Test cases behind the arm; a case judged k times is one. A payload compiled before 0.66 counted observations here.",
    )


class NullResultPayload(_VizPayload):
    """The arms of an established null, drawn as intervals with their overlap shown.

    Whether the intervals overlap is GEOMETRY, never the verdict: two marginal
    intervals can overlap while the difference between the means is real, and a
    chart that read overlap as "the lever did not move the metric" published a
    false null over arms 33% apart. The compiler states what the intervals do and
    stops; what settles the comparison is the campaign's own difference test,
    which reaches the reader through the finding's verdict and mechanism.
    """

    groups: list[NullResultArm] = Field(description="The arms compared — two, per the type's own contract.")
    metric: str | None = Field(default=None, description="The measure the arm comparison was about.")
    unit: str | None = Field(
        default=None, description="The unit every bound is in. None when the quantity is unitless."
    )
    mechanism: ModelProse | None = Field(
        default=None, description="Why the lever cannot act — what makes this an established null."
    )

    @field_validator("groups")
    @classmethod
    def _at_least_two_distinct_arms(cls, groups: list[NullResultArm]) -> list[NullResultArm]:
        """Reject a null result that does not compare anything.

        A code-compiled null result names exactly two cells by construction; what is
        refused here is the shape a stored payload can still carry that cannot be
        drawn at all: fewer than two arms, or two the reader cannot tell apart.
        """
        if len(groups) < 2:
            raise ValueError(f"needs at least 2 arms to be a comparison, got {len(groups)}")
        labels = [group.label for group in groups]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(f"arm labels must be unique; duplicated: {', '.join(duplicates)}")
        return groups


class DeltaRow(BaseModel):
    """One metric's A-vs-B comparison.

    ``unit`` is per-row rather than per-payload, unlike every other type here: the
    rows are DIFFERENT metrics, so a shared unit would be a claim about them that
    is false as soon as a latency and a cost sit in one table. The chart is drawn
    on relative change for exactly the same reason — see the compiler.

    ``paired`` is per-row for the same kind of reason — pairing is decided by
    whether the two sides' samples line up case for case, which can differ from
    one metric to the next — and it defaults to the WEAKER claim. It has to: this
    payload is stored and shipped as an open dict (``Viz.payload``), and the same
    dict is rendered twice, server-side by :mod:`threetears.evals.analysis.viz.intent` and
    browser-side by a host's eval kit, so an ABSENT
    field is read by both and no default that only one side holds can fill it in.
    Unstated therefore means unpaired on both surfaces — the same reading the
    compare endpoint's per-model rows take. The reverse default would let a row
    that says nothing about its test assert the more powerful within-case
    comparison, and print an unpaired Cohen's d under the name d_z.
    ``tests/test_significance_rule.py`` pins this side of the pair.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    metric: str = Field(min_length=1, description="The measure compared.")
    data_type: Literal["numeric", "categorical", "boolean"] = Field(
        default="numeric", description="How `a`/`b` are encoded. Only numeric rows can be drawn as a magnitude."
    )
    a: float | str | bool | None = Field(default=None, description="Variant A's value, typed by `data_type`.")
    b: float | str | bool | None = Field(default=None, description="Variant B's value, typed by `data_type`.")
    unit: str | None = Field(
        default=None, description="The unit `a`/`b`/`delta` are in. None when the quantity is unitless."
    )
    delta: float | None = Field(default=None, description="B − A, for a numeric metric.")
    materiality: Materiality = Field(
        default="material",
        description=(
            "`immaterial` when the delta is below the measure's declared materiality threshold — too small to act "
            "on. Unstated reads as `material` on every surface, the weaker claim: a row that says nothing about "
            "its threshold must not be read as one too small to matter."
        ),
    )
    scale: MeasureScale | None = Field(
        default=None,
        description=(
            "`interval` when the measure's zero is arbitrary (a 1-5 judged score): the row states its change in "
            "points and is not drawn on the relative axis. `ratio`, or None — unstated, as on a payload compiled "
            "before the field existed — draws it as relative change."
        ),
    )
    d_z: float | None = Field(
        default=None,
        description=(
            "A Cohen's d the comparison reported, printed as `d_z` when `paired` and `d` when not — `paired` says "
            "which, never this field's name, which is historical. The engine's own tests now report Hedges' g "
            "(`hedges_g`), a different number at eval sizes (0.5 against d's 0.88 at three pairs), and compile no "
            "effect into this row; a g placed here would print under d's name. A stored row's value is a Cohen's d."
        ),
    )
    p: float | None = Field(default=None, description="p-value of the significance test, when one was run.")
    n: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Test cases behind `delta`/`p` (the smaller arm's), never observations: a case judged k times is one."
            " A payload compiled before 0.66 counted observations here."
        ),
    )
    paired: bool = Field(
        default=False,
        description="Whether the test behind `p`/`d_z` paired its samples. False — the default — names the effect size Cohen's d, the weaker claim an unstated pairing may make.",
    )
    significant: bool | None = Field(
        default=None, description="Whether the difference cleared the bar. None — the default — means NO test was run."
    )

    @model_validator(mode="after")
    def _drop_a_significance_verdict_with_no_statistic(self) -> DeltaRow:
        """Demote a significance verdict that carries no statistic to "not tested".

        A statistical claim requires the statistic behind it:
        "not significant" and "not tested" are different facts, and a flag without
        the ``p`` or effect size it came from cannot tell them apart. ``False`` is the dangerous half — it asserts that a test
        was run and came back negative, which a reader takes as a measured null.

        **Demoted here rather than refused, and the layering is the point.** This
        model runs on BOTH paths: over a payload being generated, where a retry is
        cheap, and over one already in storage, where nothing can be retried at
        all. Refusing here would throw away a whole stored chart — every other row
        of it correct — over one field the reader can simply be told nothing
        about. A code-compiled table never states significance at all — the bundle
        carries no paired statistic — so this is the stored-payload half, and it is
        the only half left.
        """
        if self.significant is not None and self.p is None and self.d_z is None:
            object.__setattr__(self, "significant", None)
        return self

    @model_validator(mode="after")
    def _numeric_rows_hold_numbers(self) -> DeltaRow:
        """Reject a numeric row whose sides are not finite numbers.

        A numeric row is the only kind that can be POSITIONED, so a string in one
        of its sides is not a tolerable oddity — it is a row the chart would
        silently drop while the table above it still counted it.
        """
        if self.data_type != "numeric":
            return self
        for name, value in (("a", self.a), ("b", self.b)):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"metric {self.metric!r} is numeric but {name} is {value!r}")
            if not math.isfinite(value):
                raise ValueError(f"metric {self.metric!r} has a non-finite {name}")
        return self


class DeltaTablePayload(_VizPayload):
    """A two-arm, many-metric comparison.

    Two arms and no more, deliberately: ``a_label``/``b_label`` are the only
    columns the shape has, so a three-level sweep squeezed into it drops a level
    with nothing on screen admitting the loss. Three or more levels are a
    ``distribution``, one group per level.
    """

    rows: list[DeltaRow] = Field(description="One row per metric compared.")
    a_label: str | None = Field(default=None, description="Variant A's name — the baseline.")
    b_label: str | None = Field(default=None, description="Variant B's name.")

    @field_validator("rows")
    @classmethod
    def _rows_are_present_and_distinct(cls, rows: list[DeltaRow]) -> list[DeltaRow]:
        """Reject an empty comparison, or one naming the same metric twice."""
        if not rows:
            raise ValueError("needs at least 1 row; a comparison of nothing has nothing to draw")
        metrics = [row.metric for row in rows]
        duplicates = sorted({metric for metric in metrics if metrics.count(metric) > 1})
        if duplicates:
            raise ValueError(f"metrics must be unique; duplicated: {', '.join(duplicates)}")
        return rows

    @model_validator(mode="after")
    def _something_can_be_positioned(self) -> DeltaTablePayload:
        """Reject a comparison in which no row can be drawn as a magnitude.

        Two ways a row fails to place, and both have to be checked here or the
        chart compiles to an axis with no marks on it — a frame the reader has to
        interpret, which is the empty box this contract exists to remove.
        Categorical and boolean rows carry a comparison a reader can follow but
        not a length. And a numeric row baselined at zero has no *relative*
        change — every non-zero B is infinitely far from it — which is the
        quantity this chart's shared axis is in, because absolute deltas in
        different units cannot share one.
        """
        if not any(row.data_type == "numeric" and isinstance(row.a, int | float) and row.a != 0 for row in self.rows):
            raise ValueError(
                "no row can be positioned on a relative-change axis — every row is either non-numeric "
                "or baselined at zero, so state this comparison in the finding rather than charting it"
            )
        return self


class AttributionMovement(BaseModel):
    """How one measure moved between the two levels being compared.

    ``a``/``b`` are optional because the movement is the subject: a divergence is
    a statement about ``delta``, and a stored analysis that recorded only the
    change is still fully drawable. Where both levels ARE given they must agree
    with the delta, since a table stating three numbers that do not reconcile
    sends the reader hunting for the rounding that explains it.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    measure: str = Field(min_length=1, description="The measure's name, as the measure catalog spells it.")
    delta: float = Field(description="The signed movement from level A to level B, in the payload's `unit`.")
    a: float | None = Field(default=None, description="The measure's value at level A, when the payload states levels.")
    b: float | None = Field(default=None, description="The measure's value at level B, when the payload states levels.")
    n: int | None = Field(
        default=None,
        ge=0,
        description="Test cases behind this movement (the smaller arm's). A payload compiled before 0.66 counted observations here.",
    )

    @model_validator(mode="after")
    def _the_movement_is_drawable_and_reconciles(self) -> AttributionMovement:
        """Reject a movement that cannot be placed, or whose levels contradict it."""
        for name, value in (("delta", self.delta), ("a", self.a), ("b", self.b)):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.a is None or self.b is None:
            return self
        # The disagreement tolerance scales off the MOVEMENT, not the levels it runs
        # between: scaling it off the levels makes the guard inert exactly where it is
        # needed, since a 50 ms delta between two ~44 s levels would then tolerate a
        # stated delta anywhere in ±220 ms — over four times the movement itself. The
        # compiler draws the bar from `delta` while the table prints `a`/`b` beside it,
        # so the picture would contradict its own numbers.
        #
        # The rounding allowance scales off the LEVELS, because that is where the
        # rounding happens — a delta between two numbers written to four significant
        # figures inherits their last place, not its own. Two different quantities,
        # two different bases; collapsing either into the other is what made an
        # earlier version of this guard wrong in one direction or the other.
        slack = max(abs(self.delta), abs(self.b - self.a)) * _MOVEMENT_DELTA_TOLERANCE
        slack += max(abs(self.a), abs(self.b)) * _LEVEL_ROUNDING
        if abs((self.b - self.a) - self.delta) > slack:
            raise ValueError(
                f"measure {self.measure!r} states delta {self.delta:g} but its levels move "
                f"{self.b - self.a:g} ({self.a:g} → {self.b:g}, tolerance ±{slack:g})"
            )
        return self


class AttributionPayload(_VizPayload):
    """A whole-run movement the part under test does not account for.

    The "movement we cannot place" shape: end-to-end latency moves by far more
    than the subsystem being tuned, or the subsystem swings by ~100s that never
    reaches the whole run. Both statements are individually true — each is a real
    measurement over its own population — and the question the chart has to
    answer honestly is what, if anything, may be said about the difference.

    **The remainder is stated or its absence is explained, never neither and
    never both.** ``unattributed_delta`` is the whole's movement the part does not
    account for; ``unattributed_withheld`` is the sentence a reader gets in its
    place. Exactly one is set, because a chart with neither leaves the reader to
    infer a subtraction from two bars, and a chart with both states a number it
    simultaneously disowns.

    **A quantified remainder is earned by declared containment, never by a shared
    unit.** ``contained_by`` carries :attr:`~threetears.evals.kernel.metrics.MetricDescriptor.contained_by`
    for the part — the measure catalog's declaration of which whole it is a
    component of — as DATA, so a stored analysis keeps rendering when the catalog
    moves under it. Milliseconds of detached background work are disjoint from the
    milliseconds a turn span covers even though both are milliseconds, and
    subtracting one from the other once produced a ~95-second remainder describing
    no stretch of wall-clock at all. That is the arithmetic this type exists to
    refuse, so the number is admitted only where the catalog declares the nesting.

    **One part, not many, and that is a constraint rather than a simplification.**
    Containment is declared per measure; mutual disjointness is not declared
    anywhere, so two parts that each nest inside the whole may still overlap each
    other and their sum would double-count. Where several measures moved, the
    finding names the strongest pair and the rest belong in its prose.
    """

    end_to_end: AttributionMovement = Field(description="How the whole-run measure moved.")
    subsystem: AttributionMovement = Field(description="How the isolating measure — the part under test — moved.")
    unit: str = Field(
        min_length=1, description="The unit both movements are in; a shared unit is what makes them comparable."
    )
    contained_by: str | None = Field(
        default=None,
        description=(
            "What the measure catalog declares `subsystem.measure` to be a component OF, verbatim. None means "
            "'not known to be contained', which is where every measure starts and is what keeps an unvetted "
            "pair from being differenced."
        ),
    )
    unattributed_delta: float | None = Field(
        default=None,
        description="The whole's movement the part does not account for, in `unit`. Set only when the subtraction is earned.",
    )
    unattributed_withheld: ModelProse | None = Field(
        default=None,
        description="Why no remainder is stated, as a sentence — the reader is owed the reason, not a null to interpret.",
    )
    lever: str | None = Field(default=None, description="The lever whose two levels are being compared.")
    a_label: str | None = Field(default=None, description="Level A's name — the baseline the movement runs from.")
    b_label: str | None = Field(default=None, description="Level B's name.")

    @model_validator(mode="after")
    def _the_two_scopes_are_different_measures(self) -> AttributionPayload:
        """Reject a divergence between a measure and itself.

        One measure compared to itself has no remainder to place — the two
        movements are the same movement, so the chart would draw a difference of
        zero and present it as a finding about attribution.
        """
        if self.end_to_end.measure == self.subsystem.measure:
            raise ValueError(
                f"end_to_end and subsystem are both {self.end_to_end.measure!r} — a measure cannot diverge from itself"
            )
        return self

    @model_validator(mode="after")
    def _the_remainder_is_stated_or_explained(self) -> AttributionPayload:
        """Reject a payload that neither states a remainder nor says why it cannot."""
        stated = self.unattributed_delta is not None
        explained = bool(self.unattributed_withheld)
        if stated and explained:
            raise ValueError(
                "unattributed_delta and unattributed_withheld are both set — a remainder is either "
                "quotable or it is not, and stating a number beside the reason it cannot be stated is neither"
            )
        if not stated and not explained:
            raise ValueError(
                "neither unattributed_delta nor unattributed_withheld is set — two movements drawn with no "
                "word about their difference invite the subtraction this type exists to refuse"
            )
        return self

    @model_validator(mode="after")
    def _a_quantified_remainder_is_earned_by_containment(self) -> AttributionPayload:
        """Reject a remainder subtracted across populations that are not nested.

        This is the rule the whole type turns on. A part and a whole in the same
        unit are COMPARABLE; only a declared component is SUBTRACTABLE. Refused
        here rather than demoted, unlike a significance flag with no statistic:
        the withheld sentence is generator-authored prose naming which of the
        reasons applies, so there is nothing this layer could substitute for it
        that would not be invented.
        """
        if self.unattributed_delta is None:
            return self
        if self.contained_by != self.end_to_end.measure:
            declared = (
                f"declared a component of {self.contained_by!r}"
                if self.contained_by
                else "not declared a component of anything"
            )
            raise ValueError(
                f"unattributed_delta subtracts {self.subsystem.measure!r} from {self.end_to_end.measure!r}, but "
                f"{self.subsystem.measure!r} is {declared} — they share a unit without being nested, so the "
                "remainder may be two disjoint stretches of it; state unattributed_withheld instead"
            )
        return self

    @model_validator(mode="after")
    def _a_quantified_remainder_is_the_arithmetic_it_claims(self) -> AttributionPayload:
        """Reject a remainder that is not the difference of the two movements it sits between.

        The chart draws all three, so a remainder that does not reconcile is a bar
        whose length contradicts the two beside it — and the reader's only way out
        is to assume the picture is wrong.
        """
        if self.unattributed_delta is None:
            return self
        if not math.isfinite(self.unattributed_delta):
            # NaN specifically slips the comparison below — every comparison against
            # it is False, so `abs(nan - expected) > slack` reports the arithmetic as
            # sound. Infinity is caught there, which is exactly why the gap was quiet:
            # the check looks like it covers both. Downstream a NaN reaches
            # `spec.data.values`, and the charts route serialises with `allow_nan=False`,
            # so ONE bad payload would 500 the whole response — the in-band-failure
            # contract says a payload that cannot be drawn costs its own entry and
            # nothing else.
            raise ValueError("unattributed_delta must be a finite number")
        expected = self.end_to_end.delta - self.subsystem.delta
        slack = max(abs(self.end_to_end.delta), abs(self.subsystem.delta)) * _REMAINDER_TOLERANCE
        if abs(self.unattributed_delta - expected) > slack:
            raise ValueError(
                f"unattributed_delta {self.unattributed_delta:g} is not {self.end_to_end.measure} minus "
                f"{self.subsystem.measure}, which is {expected:g} (tolerance ±{slack:g})"
            )
        return self


class FrontierVizPoint(BaseModel):
    """One contestant's position on cost against quality.

    Named for the chart rather than for the lens: :class:`threetears.evals.analysis.lenses.frontier.FrontierPoint`
    is the computed contestant, carrying every denominator its axes were drawn over.
    This is the handful of numbers a picture places, and the two must not be confused
    when one is being read for the other.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    label: str = Field(
        min_length=1, description="The contestant — a model or variant name. Identity rides here, never on hue."
    )
    cost: float | None = Field(
        default=None,
        description="Production-replicating cost only. `None` — never 0 — where no result at this point priced one.",
    )
    quality: float = Field(description="The headline quality and the domination axis, e.g. pass^k. Higher is better.")
    latency_ms: float | None = Field(default=None, description="The secondary read, in ms. Never a second y-axis.")
    dominated: bool = Field(
        default=False,
        description=(
            "Shown dominated by the frontier lens's test — another point shown better on every axis it measured. "
            "Kept and flagged, never dropped. False is not a claim that nothing beats it."
        ),
    )
    dominance: Literal["dominated", "not_separated", "untested"] | None = Field(
        default=None,
        description=(
            "The frontier lens's verdict on this point, copied rather than decided from the drawn means: "
            "`dominated`; `not_separated` — tested and not shown dominated, which says nothing about whether it is; "
            "`untested` — no test could decide (too few cases, or no spread over too few cases for an exact test to "
            "reach α). None where the producer recorded none, which the chart states as not tested."
        ),
    )
    disqualified: bool = Field(
        default=False, description="Failed a two-pillar / safety bar, so it is out of contention on any axis."
    )
    disqualified_reason: ModelProse | None = Field(default=None, description="Why it was disqualified.")

    @field_validator("cost", "quality", "latency_ms")
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        """Reject a coordinate that cannot be placed.

        A NaN reaches ``spec.data.values`` and the charts route serialises with
        ``allow_nan=False``, so one unplaceable number would fail the whole
        response rather than costing its own entry.
        """
        if value is not None and not math.isfinite(value):
            raise ValueError("must be a finite number")
        return value

    @field_validator("latency_ms")
    @classmethod
    def _latency_is_not_negative(cls, value: float | None) -> float | None:
        """Reject a duration that ran backwards."""
        if value is not None and value < 0:
            raise ValueError("must not be negative — a latency is a duration")
        return value

    @model_validator(mode="after")
    def _the_flag_is_the_verdict(self) -> FrontierVizPoint:
        """Reject a flag that disagrees with the verdict it is read from.

        The shape is drawn from the verdict and the flag is what older readers key on; a point carrying
        both must say one thing, or the chart and the table it stands beside say two.
        """
        if self.dominance is not None and self.dominated != (self.dominance == "dominated"):
            raise ValueError(f"dominated is {self.dominated} but dominance is {self.dominance!r}")
        return self

    @model_validator(mode="after")
    def _a_reason_needs_the_flag(self) -> FrontierVizPoint:
        """Reject a stated disqualification the flag does not carry.

        The reason is what a reader is shown; the flag is what the mark is drawn
        from. A reason without the flag draws the point as an ordinary contestant
        and prints why it was excluded beside it — the two halves disagreeing in
        public, which is worse than either alone.
        """
        if self.disqualified_reason and not self.disqualified:
            raise ValueError(
                "disqualified_reason is set but disqualified is false — the reason would print beside a point drawn as eligible"
            )
        return self


class FrontierPayload(_VizPayload):
    """A cost-against-quality trade-off: which contestant to ship, and what it costs.

    Drawn as a point plot, which is the one type here whose identity axis is
    quantitative on both sides. Dominance reaches the reader through **shape and
    weight, never hue** — a point not shown dominated is a circle, one never tested a
    square, a dominated one a diamond, a disqualified one a cross — because a compiled spec carries no
    colour, and because a distinction drawn only in opacity is one a reader with
    low contrast vision does not receive at all.
    """

    points: list[FrontierVizPoint] = Field(
        description="The contestants, in any order — position is the whole encoding."
    )
    bar: float | None = Field(default=None, description="The quality bar the verdict was made against, echoed back.")
    cost_label: str | None = Field(
        default=None, description="The x caption. Defaults to `Cost` where the generator names none."
    )
    quality_label: str | None = Field(default=None, description="The y caption. Defaults to `Quality`.")

    @field_validator("points")
    @classmethod
    def _at_least_two_distinct_points(cls, points: list[FrontierVizPoint]) -> list[FrontierVizPoint]:
        """Reject a frontier that ranks nothing against anything."""
        if len(points) < 2:
            # A trade-off needs something to trade off against. One point drawn on
            # two axes is a coordinate, and the finding wanted a number.
            raise ValueError(f"needs at least 2 points to be a frontier, got {len(points)}")
        labels = [point.label for point in points]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            # Two contestants sharing a name draw as two marks a reader cannot tell
            # apart, and a frontier's whole output is which NAME to ship.
            raise ValueError(f"point labels must be unique; duplicated: {', '.join(duplicates)}")
        return points

    @model_validator(mode="after")
    def _at_least_two_points_can_be_placed(self) -> FrontierPayload:
        """Reject a figure that cannot draw a trade-off, however many points it lists.

        ``cost`` is nullable per point because a contestant may have observed no
        production-role cost, and those are disclosed rather than dropped — but they are not
        drawn, so the count that matters here is the PLACEABLE one. Requiring two
        points and then admitting one with a cost lets the same coordinate through
        that :meth:`_at_least_two_distinct_points` exists to refuse: a lone mark on
        two axes, each cropped to a padded band around that one value, which reads
        as a resolved comparison and is a scale spanning nothing.
        """
        placeable = sum(point.cost is not None for point in self.points)
        if placeable < 2:
            raise ValueError(
                f"needs at least 2 points carrying a cost to draw a trade-off, got {placeable} — "
                "a contestant with no cost is disclosed in the values, not placed on the axis"
            )
        return self

    @model_validator(mode="after")
    def _the_bar_is_finite(self) -> FrontierPayload:
        """Reject a bar that cannot be drawn as a rule."""
        if self.bar is not None and not math.isfinite(self.bar):
            raise ValueError("bar must be a finite number")
        return self


#: The level a configuration shows for a lever it never set.
#:
#: The bundle writes this em dash for an absent override (`_lever_level` in
#: `threetears.evals.analysis.bundle`), so it arrives here as a level like any other and
#: is emphatically not one. It is excluded before orderedness is inferred, because a
#: single absence would otherwise make a genuinely numeric knob unparsable and flip it
#: to categorical — measured on three levers of one stored campaign, where
#: `call_timeout_s`, `fetch_concurrency` and `search_depth` would each have drawn as
#: hues rather than as a ramp.
ABSENT_LEVEL = "—"

#: The levels a numeric lever may carry that are not points on its scale: the absence
#: sentinel, and :data:`~threetears.evals.analysis.reporting.NULL_LEVEL`, the level a lever
#: overlaid to ``null`` resolves to (#574). Neither parses as a number, so either one would
#: flip a genuinely numeric knob (``null / 6 / 12``) to categorical and draw every level as
#: a hue. Both are set apart before orderedness is inferred; the absence draws in the
#: neutral and ``null`` as its own marked cell (#694).
OFF_SCALE_LEVELS: frozenset[str] = frozenset({ABSENT_LEVEL, NULL_LEVEL})


class ResolvedDimension(NamedTuple):
    """A swept lever, and the single settled answer about how its levels draw.

    Produced only by :meth:`SweepRankingPayload.dimensions_resolved`, which is the
    one place the question is answered. The fields separate what was decided from
    who decided it, because the caption owes a reader both.
    """

    #: The lever's name, matching a key of every row's ``config``.
    name: str
    #: Whether it draws as a light-to-dark ramp. False means hues, and this is the
    #: field every consumer should branch on — nothing re-derives it.
    ordered: bool
    #: Whether the payload stated its orderedness rather than the inference
    #: guessing. A guess is admitted in the caption; a ramp otherwise asserts an
    #: order the payload never claimed.
    declared: bool
    #: Declared ordered, drawn as hues anyway, because the levels state no order to
    #: sort them into and this payload has no field that could supply one. Also
    #: admitted in the caption: a declaration the compiler could not honour is a
    #: fact about the figure, not an implementation detail.
    demoted: bool


def infer_ordered(levels: Iterable[str]) -> bool:
    """Whether a dimension's levels have an order, inferred from their text.

    Nothing in the campaign bundle declares this, and the two answers draw
    differently — an ordered dimension takes a lightness ramp, a categorical one
    takes hues — so it has to be decided somewhere. Numeric parsability is the
    inference: ``2``/``4``/``8`` is a knob, ``gpt-5``/``glm-5.2`` is a set.

    Deliberately NOT a version-string heuristic. ``v2`` before ``v10`` is only
    obvious to a reader who already knows the scheme, and a wrong guess here draws
    a ramp asserting an order the data does not have — which is the one failure
    this whole chart type is built to avoid, since the block pattern in a column IS
    the finding.

    Args:
        levels: Every level the dimension takes, the absence sentinel included.

    Returns:
        Whether every level that is actually a level parses as a number. The off-scale
        levels (:data:`OFF_SCALE_LEVELS`) are set apart first, so a ``null`` beside numeric
        levels leaves the knob ordered, while a lever whose only stated levels are
        ``null`` (or words) has no order to draw.
    """
    stated = [level for level in levels if level not in OFF_SCALE_LEVELS]
    if not stated:
        # Every configuration left this lever alone, so there is no order to have.
        return False
    for level in stated:
        try:
            float(level)
        except ValueError:
            return False
    return True


class SweepMeasure(BaseModel):
    """One of the two measures a ranked sweep reads — the ranked one or its companion."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    measure: str = Field(min_length=1, description="The measure's name, as the measure catalog spells it.")
    unit: str | None = Field(
        default=None,
        description="The unit its values are in. None when the quantity is unitless — never inferred from the name.",
    )


class SweepDimension(BaseModel):
    """One swept lever, and whether its levels have an order.

    ``ordered`` is carried rather than derived at render time so a reader can see
    what was assumed. Where the payload states nothing,
    :meth:`SweepRankingPayload.dimensions_resolved` infers it and says so — the
    inference is the same either way, but "the generator decided this" and "we
    guessed" are different facts and the caption states which.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, description="The lever's name, matching a key of every row's `config`.")
    ordered: bool = Field(
        description="Whether its levels run from low to high, which is what earns it a ramp instead of hues."
    )


class SweepHeldFixed(BaseModel):
    """The constraint that turns a full ranking into a slice of one.

    A predicate on which rows qualify, never part of the form: the same layout
    answers "rank every configuration" and "rank the ones that cost about the
    same", and the tolerance is what separates them.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    value: float = Field(description="The secondary measure's held value — the centre of the slice.")
    tolerance: float = Field(gt=0, description="How far either side of `value` still counts as held. Never zero.")

    @field_validator("value")
    @classmethod
    def _is_finite(cls, value: float) -> float:
        """Reject a slice centred on a number that cannot be placed."""
        if not math.isfinite(value):
            raise ValueError("must be a finite number")
        return value

    @field_validator("tolerance")
    @classmethod
    def _tolerance_is_finite(cls, value: float) -> float:
        """Reject a tolerance that admits everything or cannot be drawn."""
        if not math.isfinite(value):
            raise ValueError(
                "must be a finite number — an infinite tolerance is the unconstrained mode, so omit `held_fixed`"
            )
        return value


class SweepOmission(BaseModel):
    """Configurations the producer left out, and the band they fell in.

    Stated rather than silently dropped: a ranking that shows ten of thirty-six
    rows reads as the whole sweep, and the reader's conclusion about which lever
    drives the result is drawn from a sample nobody told them was one.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    count: int = Field(
        gt=0, description="How many configurations were dropped. Zero is not an omission — omit the block."
    )
    low: float = Field(description="The lowest ranked value among them.")
    high: float = Field(description="The highest ranked value among them.")

    @model_validator(mode="after")
    def _the_band_is_drawable(self) -> SweepOmission:
        """Reject a band that cannot be stated as a range."""
        for name, value in (("low", self.low), ("high", self.high)):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.low > self.high:
            raise ValueError(f"low {self.low:g} exceeds high {self.high:g} — the bounds are the wrong way round")
        return self


class SweepRow(BaseModel):
    """One configuration: which levels it ran at, and what the two measures read."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    config: dict[str, str] = Field(description="Dimension name → the level this configuration ran at, as text.")
    ranked_value: float = Field(
        description="The measure this sweep is ranked on. Higher is better; the sort relies on it."
    )
    secondary_value: float = Field(
        description="The companion measure — held within a tolerance, or free and its range stated. Never absent: the "
        "invariant is that its spread is always stated, and a missing value makes the spread unstatable."
    )
    n: int | None = Field(
        default=None,
        ge=0,
        description="Test cases behind this configuration. A payload compiled before 0.66 counted observations here.",
    )
    label: str | None = Field(
        default=None, description="A name for the configuration, where it has one beyond its levels."
    )

    @field_validator("config")
    @classmethod
    def _levels_are_present_and_named(cls, config: dict[str, str]) -> dict[str, str]:
        """Reject a configuration that identifies nothing.

        A row IS its levels — that is what the barcode draws and what makes one row
        different from another — so an empty config is a row a reader cannot tell
        from its neighbour, drawn as though it were a distinct configuration.
        """
        if not config:
            raise ValueError("config carries no dimensions — a configuration with no levels is not one")
        blank = sorted(name for name, level in config.items() if not name.strip() or not level.strip())
        if blank:
            raise ValueError(f"config has an empty dimension name or level: {', '.join(blank)}")
        return config

    @model_validator(mode="after")
    def _both_measures_are_placeable(self) -> SweepRow:
        """Reject a row whose numbers cannot be positioned.

        A NaN reaches ``spec.data.values`` and the charts route serialises with
        ``allow_nan=False``, so one unplaceable number would fail the whole response
        rather than costing its own entry.
        """
        for name, value in (("ranked_value", self.ranked_value), ("secondary_value", self.secondary_value)):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        return self


class SweepRankingPayload(_VizPayload):
    """Configurations ranked on one measure, with the other shown alongside.

    **A combination is not a series; it is a row.** A campaign that sweeps two or
    more levers produces combinations, and drawing them as coloured series is what
    the categorical palette cannot survive — thirty-six cells is four times the
    slots. So the configuration goes on the row, drawn as a fused barcode of its
    levels, and the ranked measure takes position.

    **The secondary measure's spread is always stated.** Held within a tolerance in
    constrained mode, stated as a range in unconstrained mode, and in the second
    case the caption also says the ranking is not controlled for it. Without that,
    a block in the prompt column could equally mean "v3 configurations cost more",
    and the finding the barcode exists to carry is confounded by spend.

    **Rows are sorted by the ranked measure, never by a dimension** — the block
    pattern in a column IS the finding, and sorting by that column manufactures it.
    The sort is descending because rank 1 sits at the top and the axis puts high
    values at the right, so the marks descend as a monotone staircase and best is
    top-right. That makes "higher is better" a property the ranked measure must
    have; a measure where smaller is better belongs in ``secondary``, or is
    inverted by the producer before it gets here.
    """

    ranked: SweepMeasure = Field(description="The measure the configurations are ranked on.")
    secondary: SweepMeasure = Field(description="The companion measure — held, or free and its range stated.")
    rows: list[SweepRow] = Field(description="One entry per configuration, in any order — the compiler sorts them.")
    held_fixed: SweepHeldFixed | None = Field(
        default=None, description="The constraint, in constrained mode. Absent means the full sweep is ranked."
    )
    dimensions: list[SweepDimension] | None = Field(
        default=None,
        description="The swept levers and whether each is ordered. Absent means the compiler infers it and says so.",
    )
    omitted: SweepOmission | None = Field(
        default=None, description="Configurations the producer dropped before sending, and the band they fell in."
    )

    @field_validator("rows")
    @classmethod
    def _at_least_two_rows_over_one_dimension_set(cls, rows: list[SweepRow]) -> list[SweepRow]:
        """Reject a sweep that ranks nothing, or one whose rows are not comparable.

        Two configurations is the minimum a ranking can be about. And every row has
        to carry the same dimensions: a barcode whose columns mean different things
        on different rows is not a barcode, and a column present on some rows would
        draw as a gap that reads as a level.
        """
        if len(rows) < 2:
            raise ValueError(f"needs at least 2 configurations to be a ranking, got {len(rows)}")
        shapes = {frozenset(row.config) for row in rows}
        if len(shapes) > 1:
            spread = sorted(", ".join(sorted(shape)) or "(none)" for shape in shapes)
            raise ValueError(
                f"rows sweep different dimension sets ({' | '.join(spread)}) — a barcode needs one column set"
            )
        return rows

    @model_validator(mode="after")
    def _no_configuration_appears_twice(self) -> SweepRankingPayload:
        """Reject two rows that ran at the same levels.

        The barcode IS the row's identity — the glyph is what tells one
        configuration from the next — so two rows at identical levels draw as two
        indistinguishable stripes at different ranks, and the reader's only reading
        is that the same configuration scored two different values. Where a
        configuration really was measured twice, that is a `distribution` over its
        measure; where the rows differ on a lever this payload did not carry, the
        missing lever is the fix.
        """
        seen: dict[tuple[tuple[str, str], ...], int] = {}
        for row in self.rows:
            seen[tuple(sorted(row.config.items()))] = seen.get(tuple(sorted(row.config.items())), 0) + 1
        repeated = sorted(
            ", ".join(f"{name}={level}" for name, level in config) for config, count in seen.items() if count > 1
        )
        if repeated:
            raise ValueError(
                f"configurations must be unique; {'; '.join(repeated)} appears more than once — two rows at the same "
                "levels draw as identical glyphs at different ranks, which reads as one configuration scoring twice. "
                "If the same configuration really was measured more than once, that is a `distribution` over its "
                "measure; if the rows differ on a lever this payload does not carry, the missing lever is the fix"
            )
        return self

    @model_validator(mode="after")
    def _no_two_rows_answer_to_the_same_name(self) -> SweepRankingPayload:
        """Reject two rows carrying the same label.

        The sibling of the configuration check above, and it is reachable past it:
        that one compares ``config``, while the row's drawn identity prefers
        ``label`` where one is given. So two rows at genuinely different levels
        but under one label collapse into a single band on the shared y scale —
        the same "identical glyphs at different ranks" the configuration rule
        refuses, arrived at through the field that overrides it. A label is a
        name for a configuration; two configurations under one name is a naming
        mistake, and the fix is a distinguishing label or none at all.
        """
        labels = [row.label for row in self.rows if row.label]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(
                f"row labels must be unique; duplicated: {', '.join(duplicates)} — the label is what names the row "
                "wherever it is given, so two rows sharing one draw as a single configuration holding two ranks"
            )
        return self

    @model_validator(mode="after")
    def _declared_dimensions_match_the_rows(self) -> SweepRankingPayload:
        """Reject a dimension list that describes different levers than the rows carry.

        The list decides which columns take a ramp and which take hues, so one that
        names a lever no row has — or misses one every row has — silently draws a
        column by whichever default the lookup falls through to.
        """
        if self.dimensions is None:
            return self
        declared = [dimension.name for dimension in self.dimensions]
        duplicates = sorted({name for name in declared if declared.count(name) > 1})
        if duplicates:
            raise ValueError(f"dimension names must be unique; duplicated: {', '.join(duplicates)}")
        swept = set(self.rows[0].config)
        if set(declared) != swept:
            missing = sorted(swept - set(declared))
            unknown = sorted(set(declared) - swept)
            detail = "; ".join(
                part
                for part in (
                    f"rows carry {', '.join(missing)} with nothing declared" if missing else "",
                    f"declares {', '.join(unknown)} which no row sweeps" if unknown else "",
                )
                if part
            )
            raise ValueError(f"`dimensions` does not describe the swept levers — {detail}")
        return self

    @model_validator(mode="after")
    def _the_categorical_levels_fit_the_validated_hues(self) -> SweepRankingPayload:
        """Reject a sweep whose categorical levels outrun the palette.

        This is the ceiling that makes the barcode's use of hue legitimate rather
        than merely bounded by good intentions. A categorical ramp RECYCLES — past
        the palette the ninth level takes the first's hue and two different levels
        become indistinguishable — and the safeguard that normally makes recycling
        survivable is a direct label on every mark, which a barcode cell is far too
        narrow to carry. So the two protections are both unavailable at once, and
        the honest answer is a different chart.

        **What counts is what draws in HUES, which is not the same as what was
        declared categorical.** A lever that draws as a ramp is sampled from a path
        rather than assigned to slots, so its levels close up rather than colliding
        and it has no ceiling at all — which is why a thirty-six-configuration sweep
        of numeric knobs is exactly what this type is for. But a lever *declared*
        ordered whose levels cannot be placed in an order draws in hues too, and
        hues are what this rations, so it counts. Exempting it on the strength of
        its declaration would let a barcode draw more hues than the palette
        validates, which is the outcome this rule exists to prevent.

        That makes the producer's instructions part of this rule rather than an
        adjacent nicety: the generator is told "only numeric levels ramp" so it can
        predict the scope, because a refusal here discards the whole paid analysis
        rather than merely its chart. Changing what counts without changing that
        sentence bills the producer for a bound it was never given.
        """
        # The absence sentinel is not a level and consumes no hue — it draws in the
        # neutral that carries no identity — so counting it would refuse a sweep that
        # fits the palette perfectly well.
        categorical = sorted(
            {
                f"{name}={row.config[name]}"
                for name in (lever.name for lever in self.dimensions_resolved() if not lever.ordered)
                for row in self.rows
                if row.config[name] != ABSENT_LEVEL
            }
        )
        if len(categorical) > VALIDATED_SLOTS:
            raise ValueError(
                f"{len(categorical)} categorical levels across the swept levers ({', '.join(categorical)}) exceeds the "
                f"{VALIDATED_SLOTS} validated hues — past them the palette recycles and two levels draw alike, and a "
                "barcode cell cannot carry the direct label that normally makes that survivable; use a heatmap for a "
                "crossing this wide, or sweep the extra lever separately"
            )
        return self

    @model_validator(mode="after")
    def _an_omission_is_outside_what_is_drawn(self) -> SweepRankingPayload:
        """Reject an omission band that overlaps the rows it claims to sit beside.

        The annotation says "and N more, between X and Y". If that band covers rows
        that ARE drawn, the reader counts the same configurations twice — once as
        marks and once as a number.
        """
        if self.omitted is None:
            return self
        drawn_low = min(row.ranked_value for row in self.rows)
        if self.omitted.high > drawn_low:
            raise ValueError(
                f"omitted band [{self.omitted.low:g}, {self.omitted.high:g}] reaches above the lowest drawn "
                f"configuration ({drawn_low:g}) — the dropped rows are the tail, so their band sits below what is shown"
            )
        return self

    def dimensions_resolved(self) -> list[ResolvedDimension]:
        """The swept levers in a stable order, each with its orderedness and its source.

        **The one place orderedness is decided.** Every consumer reads this — the
        caption, the categorical ceiling, and the compiler's ramp — because the
        question has exactly one right answer per lever and two answers drawn from
        two rules is how a cell ends up matching no layer at all.

        Two decisions, not one, and they belong to different parties. The
        DECLARATION says whether the levels mean an order; that is the generator's
        to state, and it wins over the inference, which exists only to answer for a
        payload that stated nothing. The DATA says whether an order can be drawn: a
        ramp is a position along a sorted range, so levels that cannot be placed on
        one cannot take it however emphatically they were declared ordered. A lever
        declared ordered over ``small``/``medium``/``large`` means something true
        that this payload has no field to express — the declaration carries no level
        sequence — so it draws as hues and the caption says the declaration was not
        honoured. Silently overriding a declaration would be the worse of the two,
        and inventing an order from row sequence the worst of the three.

        Returns:
            One :class:`ResolvedDimension` per lever, sorted by name so the
            barcode's columns are in the same order on every figure of a report.
        """
        stated = {dimension.name: dimension.ordered for dimension in self.dimensions or []}
        resolved: list[ResolvedDimension] = []
        for name in sorted(self.rows[0].config):
            placeable = infer_ordered(row.config[name] for row in self.rows)
            declared = name in stated
            means_an_order = stated[name] if declared else placeable
            resolved.append(
                ResolvedDimension(
                    name=name,
                    ordered=means_an_order and placeable,
                    declared=declared,
                    demoted=declared and means_an_order and not placeable,
                )
            )
        return resolved

    def qualifying(self) -> list[SweepRow]:
        """The rows the constraint admits, or every row where there is none.

        Returns:
            The rows to rank. Possibly empty in constrained mode, which is a
            RESULT — the slice found nothing — and is drawn as one rather than
            refused: a tolerance admitting nothing is a measurement outcome, and
            refusing legitimate data is never the right answer. The empty *chart* the reporting rules forbid
            is what the compiler avoids by
            drawing the sweep the slice was taken from.
        """
        if self.held_fixed is None:
            return list(self.rows)
        held = self.held_fixed
        return [row for row in self.rows if abs(row.secondary_value - held.value) <= held.tolerance]


class TimeseriesPoint(BaseModel):
    """One series' estimate at one time position. Its interval is what the chart draws, so it is required."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    position: str = Field(min_length=1, description="The time position this point sits at — one of `positions`.")
    ci: ConfidenceInterval = Field(description="The estimate and its interval at that position.")
    n: int | None = Field(
        default=None,
        ge=0,
        description="Test cases behind the point. A payload compiled before 0.66 counted observations here.",
    )


class TimeseriesSeries(BaseModel):
    """One line: one cell's reading followed across the time axis."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    label: str = Field(min_length=1, description="The series' name — the line's identity, drawn at its end.")
    points: list[TimeseriesPoint] = Field(min_length=1, description="Its points, in the axis's order.")


class TimeseriesGap(BaseModel):
    """A position a series has no point at, and why — a gap the line is broken across, never bridged."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    series: str = Field(min_length=1, description="The series' name, drawn or not.")
    position: str = Field(min_length=1, description="The position it has no point at — one of `positions`.")
    reason: str = Field(min_length=1, description="Why there is no point there.")


class TimeseriesPayload(_VizPayload):
    """One reading across the campaign's time axis, a line per series, every point with its interval.

    **Time is an axis here, and identity rides on a label at each line's end.** Every other type puts its
    categories on an axis; this one cannot, because both axes are taken — time and the value — so the line is
    named where it ends, and the hue it draws in is never the only thing saying which line is which.

    **A gap is stated, never bridged.** A line drawn straight across a position it has no point at is a value
    the campaign did not measure, so a series' line breaks there and the payload names the gap.
    """

    metric: str = Field(min_length=1, description="The measure (or judged dimension) drawn.")
    unit: str | None = Field(default=None, description="The unit every value is in. None when unitless.")
    basis: Literal["release", "date"] = Field(
        description="What the positions are: the builds the host labels, or the UTC days the runs started on."
    )
    release_label: str | None = Field(
        default=None, description="The host label the positions name, on a `release` axis; None on a `date` axis."
    )
    positions: list[str] = Field(description="The time positions, earliest first — the axis, in its drawn order.")
    series: list[TimeseriesSeries] = Field(description="The lines, in the order they should be read.")
    gaps: list[TimeseriesGap] = Field(
        default_factory=list, description="Each position a series has no point at, with why — disclosed, never drawn."
    )
    interleaved: list[str] = Field(
        default_factory=list,
        description=(
            "On a `release` axis, each build whose runs went on being made after the NEXT build's first run, so its "
            "cells pool runs from both sides of that step — disclosed, since the line reads as a clean before/after. "
            "Empty when every build's runs ended before the next began, and always on a `date` axis."
        ),
    )

    @field_validator("positions")
    @classmethod
    def _at_least_two_distinct_positions(cls, positions: list[str]) -> list[str]:
        """Refuse an axis that is not a span of time, or one whose positions a reader cannot tell apart."""
        if len(positions) < 2:
            raise ValueError(f"needs at least 2 positions to be a time series, got {len(positions)}")
        if any(not position.strip() for position in positions):
            raise ValueError("every position needs a name")
        if repeated := sorted({p for p in positions if positions.count(p) > 1}):
            raise ValueError(f"positions must be unique; duplicated: {', '.join(repeated)}")
        return positions

    @model_validator(mode="after")
    def _series_sit_on_the_axis(self) -> TimeseriesPayload:
        """Refuse a series that is not a line along this axis, and a gap that contradicts a point.

        Each point's position is on the axis, once, in the axis's order — a line connects its points in the
        order they are listed, so a point out of order draws a trend backwards in time. At least one series
        has two points, or there is no line at all. A gap names a position on the axis and never one where
        its series has a point: a point and a gap at one position are two answers to one question.
        """
        if (self.basis == "release") != (self.release_label is not None):
            raise ValueError("a `release` axis names its release_label, and a `date` axis names none")
        if not self.series:
            raise ValueError("needs at least 1 series; an empty time series has nothing to draw")
        labels = [line.label for line in self.series]
        if repeated := sorted({label for label in labels if labels.count(label) > 1}):
            raise ValueError(f"series labels must be unique; duplicated: {', '.join(repeated)}")
        order = {position: index for index, position in enumerate(self.positions)}
        for line in self.series:
            placed = [point.position for point in line.points]
            if off := sorted({position for position in placed if position not in order}):
                raise ValueError(f"series {line.label!r} has points at {', '.join(off)}, which the axis does not hold")
            if repeated := sorted({position for position in placed if placed.count(position) > 1}):
                raise ValueError(f"series {line.label!r} has more than one point at {', '.join(repeated)}")
            if [order[position] for position in placed] != sorted(order[position] for position in placed):
                raise ValueError(f"series {line.label!r} lists its points out of the axis's order")
        if not any(len(line.points) >= 2 for line in self.series):
            raise ValueError("no series has points at two positions — there is no line to draw")
        drawn = {(line.label, point.position) for line in self.series for point in line.points}
        stated = {(gap.series, gap.position) for gap in self.gaps}
        for gap in self.gaps:
            if gap.position not in order:
                raise ValueError(f"a gap names position {gap.position!r}, which the axis does not hold")
            if gap.series not in labels:
                raise ValueError(f"a gap names series {gap.series!r}, which the payload does not draw")
            if (gap.series, gap.position) in drawn:
                raise ValueError(f"series {gap.series!r} has both a point and a gap at {gap.position!r}")
        if self.interleaved and self.basis != "release":
            raise ValueError("only builds interleave: a `date` axis's days are disjoint by construction")
        if off := sorted(set(self.interleaved) - set(order)):
            raise ValueError(f"interleaved names {', '.join(off)}, which the axis does not hold")
        if self.positions[-1] in self.interleaved:
            raise ValueError("the last build has no next build to interleave with")
        # The other half of "a gap is stated, never bridged": a position a series has no point at is a gap,
        # and an unstated one is a line broken with nothing saying why.
        for line in self.series:
            if unstated := [
                p for p in self.positions if (line.label, p) not in drawn and (line.label, p) not in stated
            ]:
                raise ValueError(
                    f"series {line.label!r} has no point at {', '.join(unstated)} and no gap saying why; a "
                    "position a series was not measured at is a stated gap"
                )
        return self

    @model_validator(mode="after")
    def _days_are_days_in_calendar_order(self) -> TimeseriesPayload:
        """Refuse a ``date`` axis whose positions are not ISO calendar days, earliest first.

        The chart's disclosure says a ``date`` axis is UTC days in calendar order, so the payload is held to
        it rather than the sentence being true only of the payloads the builder happens to make.
        """
        if self.basis != "date":
            return self
        for position in self.positions:
            try:
                parsed = date.fromisoformat(position)
            except ValueError:
                parsed = None
            if parsed is None or parsed.isoformat() != position:
                raise ValueError(f"a `date` axis's positions are days as YYYY-MM-DD; {position!r} is not one")
        if self.positions != sorted(self.positions):
            raise ValueError("a `date` axis lists its days in calendar order, earliest first")
        return self


#: Viz type → the model validating its payload.
#:
#: A type absent from this map is UNVALIDATED, not valid. Keeping the registry
#: explicit is what lets :func:`parse_payload` distinguish "conforms" from "nothing
#: checked it", so an unmigrated type reads as a gap rather than a pass.
PAYLOAD_MODELS: dict[str, type[_VizPayload]] = {
    "attribution": AttributionPayload,
    "breakdown": BreakdownPayload,
    "delta_table": DeltaTablePayload,
    "distribution": DistributionPayload,
    "frontier": FrontierPayload,
    "null_result": NullResultPayload,
    "sweep_ranking": SweepRankingPayload,
    "timeseries": TimeseriesPayload,
}


def parse_payload(viz_type: str, payload: dict[str, Any]) -> _VizPayload | None:
    """Validate one viz payload against its declared type.

    Args:
        viz_type: The ``Viz.type`` discriminator.
        payload: The open payload dict, exactly as stored or generated.

    Returns:
        The parsed payload model, or ``None`` when no model is registered for
        ``viz_type`` — the caller must treat that as unchecked, never as valid.

    Raises:
        PayloadError: The payload does not match the type's declared contract.
    """
    model = PAYLOAD_MODELS.get(viz_type)
    if model is None:
        return None
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise PayloadError(f"{viz_type} payload is malformed: {describe_validation(exc)}") from exc


def describe_validation(exc: ValidationError) -> str:
    """Render a pydantic failure as ``field: reason``, naming every offending field.

    The default ``ValidationError`` rendering is multi-line and repeats the model
    name on every entry; this surface is a one-line error an operator reads next
    to a generation that just cost money, so it names the fields and stops.
    """
    parts = []
    for error in exc.errors():
        location = ".".join(str(segment) for segment in error.get("loc", ())) or "(payload)"
        parts.append(f"{location}: {error.get('msg', 'invalid')}")
    return "; ".join(parts) or str(exc)


__all__ = [
    "SERIES_SLOTS",
    "VALIDATED_SLOTS",
    "ABSENT_LEVEL",
    "OFF_SCALE_LEVELS",
    "PAYLOAD_MODELS",
    "AttributionMovement",
    "AttributionPayload",
    "BreakdownPart",
    "BreakdownPayload",
    "ConfidenceInterval",
    "DeltaRow",
    "DeltaTablePayload",
    "DistributionGroup",
    "DistributionPayload",
    "FrontierPayload",
    "FrontierVizPoint",
    "HistogramBucket",
    "NullResultArm",
    "NullResultPayload",
    "PayloadError",
    "ResolvedDimension",
    "SweepDimension",
    "SweepHeldFixed",
    "SweepMeasure",
    "SweepOmission",
    "SweepRankingPayload",
    "SweepRow",
    "TimeseriesGap",
    "TimeseriesPayload",
    "TimeseriesPoint",
    "TimeseriesSeries",
    "describe_validation",
    "infer_ordered",
    "parse_payload",
]
