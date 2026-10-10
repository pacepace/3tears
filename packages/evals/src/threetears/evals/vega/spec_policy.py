"""The Vega-Lite renderer's own gate: the presentation rules, checked against the spec it renders.

**The intent is gated first, and this is the second gate.** What a chart says is held to the rules by
:mod:`threetears.evals.analysis.viz.policy`, which reads the
:class:`~threetears.evals.analysis.viz.intent.ChartIntent` every renderer is handed. This module checks
that the Vega-Lite spec drawn from it still satisfies them as a picture — and checks what only a
rendered figure can break (upright text, an undashed grid, one scale across layers and panels, a colour
the rasteriser can parse). It is the Vega-Lite adapter's, and nothing in the core reads it.

**This gate sits OUTSIDE the compiler, and that placement is the point.** The
rules below are properties of a *Vega-Lite spec*, not of any particular thing
that produced one — so they are written against a spec and applied to whatever
arrives. A check living inside the compiler could only ever vouch for specs the
compiler wrote, and the direction of travel for this subsystem is that a chart
shape becomes expressible without a code change, which means a spec this module
has never seen. One gate, every producer.

**Which side a new rule belongs on.** Anything a *generator-emitted* spec could
violate is a rule and lives here — whether the chart is honest. Layout arithmetic
and mark construction live in the compiler — *how* a chart is drawn, which a
hand-written spec is free to do differently. The tell that a rule was put on the
wrong side: the compiler enforces it by never emitting the bad shape, and nothing
would catch the same shape arriving from somewhere else. Stated here rather than
only in the plan that introduced it, because a build plan is deleted when its work
ships and this criterion outlives it.

Each rule is a report standard that a chart can break while drawing perfectly
cleanly — which is why they are mechanical here rather than left to review:

1. **Shape only from values.** A length-encoding mark on a truncated baseline
   draws a ratio the numbers do not contain.
2. **One unit per quantity.** A quantity drawn without its unit stated, or one
   axis carrying two of them, makes the reader supply the missing half.
3. **Fixed-order palette that never refuses to draw.** Slots 1-4 are validated, 5-8
   are the second tier, and a domain wider than the palette recycles with a
   warning rather than being rejected — refusing to render legitimate data is the one
   outcome the palette rules out. What IS refused is a palette that cannot say
   which slot a category took: an absent domain, or an interpolated scheme.
4. **A shared axis across layers and facet cells.** Independent scales let two
   series be drawn to different rulers inside one frame. Concatenated panels are
   separate frames, and Vega-Lite draws them on independent *positional* scales
   by default — so ``x`` and ``y`` are exempt there. ``color`` is not: Vega-Lite
   shares it across concatenated panels, and an independent one paints the same
   category two different hues from panel to panel.
5. **A named variability span.** An interval is uninterpretable without saying
   what it varies over — the same band means different things across runs than
   across cases.
6. *(Not a rule here.)* What rendered text CLAIMS is never checked — code checks
   structure, never prose. A STRUCTURED significance
   verdict with no statistic behind it is dropped where the payload is parsed
   (``DeltaRow``); whether prose overclaims is the reporter eval's question.
7. **Identity never rides on colour alone.** Past the validated hues a direct label
   on every mark is mandatory — the safeguard that lets the palette draw past
   validated separation — and below them a legend survives only for a faceted
   chart, where one key genuinely serves several panels.
8. **A length mark starts at zero.** A bar or an area states a quantity by how far
   it reaches, so a cropped axis under one multiplies every ratio a reader takes off
   it — that is refused outright rather than disclosed. A point or an interval states
   a position its labelled ticks already locate, and is left alone; there is no
   footnote rule here, and the comment beside ``_check_baseline`` records why. Which
   is why a cropped position axis may not suppress those labels.
9. **Upright text, undashed grid.** Rotated words are read by turning the page, and
   a dashed line is a mark property the reader has learned to read as data.
10. **A ranking is ordered by a measure it draws.** A block in one column of a sweep is the finding; ordering the rows by that column
    manufactures it, and the two are the same picture in a different sequence.
11. **A line through categories states their order.** A line connects its points in the order its
    scale puts them, and a categorical scale left to sort itself sorts by name — so builds ``0.9`` and
    ``0.10`` draw as ``0.10`` then ``0.9``, and the slope between them is a trend the data never had.

**What "a chart" means to these rules.** A composed spec is read as a set of
FRAMES (:class:`_View`): a ``layer`` puts its marks in one frame sharing one set
of axes, while ``concat``/``facet``/``repeat`` open a frame with axes of its own.
Rules 1-4 are read per frame, and the scoping cuts both ways — a violation
nested inside a concatenated panel is caught where a root-only check would pass
it, and a figure whose two panels legitimately measure two different quantities
is allowed where a spec-wide pool would refuse it. Rule 5 is raised per frame but
*satisfied* by any frame that CONTAINS it, because a ``description`` written on
the figure names the spans drawn in its panels — one written on a sibling panel
does not, and that panel's own interval stays unnamed. One check escapes the
frame loop entirely: :func:`check_spec` runs :func:`_check_renderable_colour`
over the whole spec before it walks the frames, because one unparseable colour
blackens whatever draws it wherever it sits.

**Prose is not read here at all.** A caption or disclosure line shown beside a chart
is served as text (a ``<figcaption>``, or lines of MCP output) and never reaches the
rasteriser, so no rendering rule applies to it; and what it says is the reporter
eval's question, never a check's. Inside the spec the
colour rule reads only a string that IS a colour value, never one that merely
mentions a colour function inside a title or a label.
"""

from __future__ import annotations

import dataclasses
import itertools
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from threetears.evals.vega.palette import VALUE_ON_FILL_STYLE, series_slots, validated_slots
from threetears.observe import get_logger

log = get_logger(__name__)

#: Marks that encode magnitude by LENGTH from a baseline, so a non-zero baseline
#: rescales the comparison the reader makes. A line or point encodes position, not
#: length, and legitimately omits zero.
_LENGTH_MARKS = frozenset({"bar", "area"})

#: Marks that draw an uncertainty interval by name, and so owe a statement of what
#: it spans. Not the only way to draw one — see :func:`_draws_interval`.
_INTERVAL_MARKS = frozenset({"errorbar", "errorband"})

#: Encoding channel pairs that place a mark BETWEEN two values rather than at one.
#: A `rule` from `x` to `x2` is an interval however it is named, and naming is
#: exactly what a producer gets to choose.
_SPAN_CHANNELS = (("x", "x2"), ("y", "y2"))

#: Why an independently resolved scale misleads, per channel — the failure a colour
#: scale produces is not the one a positional scale produces, and a message naming
#: the wrong one sends the producer to the wrong fix. This mapping is also the
#: REGISTER of gated channels (see :data:`_SHARED_SCALE_CHANNELS`), so widening the
#: gate means writing the harm, which is the sentence a producer needs anyway.
_SHARED_SCALE_HARM = {
    "x": "two series drawn to different rulers read as one comparison",
    "y": "two series drawn to different rulers read as one comparison",
    "color": "one category takes a different colour from one panel or layer to the next",
}

#: Positional and identity channels a shared scale matters for. Derived from
#: :data:`_SHARED_SCALE_HARM` rather than listed beside it: a channel gated with no
#: harm written for it raised ``KeyError`` from the message builder, so the two could
#: not be allowed to drift apart.
_SHARED_SCALE_CHANNELS = tuple(_SHARED_SCALE_HARM)


#: A colour VALUE in a notation the server-side rasteriser does not parse. It renders
#: one BLACK and raises nothing, so a spec carrying one produces a valid image in which
#: every series has silently lost its colour.
#:
#: **Matched against the whole string, never searched inside one.** A colour the
#: renderer parses is a value on its own — a ``color``, a ``fill``, a scale range
#: entry — so the rule's population is strings SHAPED like a colour value. A title or
#: a label that merely mentions ``lab(`` is text the rasteriser draws as glyphs, not a
#: colour it parses, and refusing a chart over it would be a check reading prose
#: (which is also why no caption-side twin of this rule
#: exists). The bracket stays attached, matching CSS's own
#: function grammar.
#:
#: The argument list admits one level of nested brackets (a ``color-mix`` over an ``oklch``) and
#: no more, so ``oklch(…) (observations)`` — a title that merely starts with one — is not a value.
_UNRENDERABLE_COLOUR = re.compile(r"\s*(?:oklch|oklab|lch|lab|color-mix)\((?:[^()]|\([^()]*\))*\)\s*", re.IGNORECASE)

#: Marks that CONNECT their points along a positional scale, so the scale's order is the order of the line.
_CONNECTING_MARKS = frozenset({"line", "trail", "area"})

#: Sentinel telling an ABSENT ``legend`` key from one explicitly set to ``None``.
#: The two are opposite instructions — absent draws Vega-Lite's default legend,
#: ``None`` suppresses it — and ``dict.get`` collapses them onto the same answer.
_LEGEND_UNSET = object()

#: The Vega-Lite ``name`` by which a spec declares itself a ranking.
#:
#: A ranking is the one shape here whose correctness depends on a claim rather than
#: on a structure: "these rows are in the order the measure put them" is not
#: readable off marks that would be drawn identically in any order. So the producer
#: states it, and :func:`_check_ranked_by_its_measure` holds the producer to it —
#: the same division as ``description`` for an interval's variability and the
#: title's subtitle for a crop, where the spec asserts a fact about itself and the
#: gate refuses the assertion it cannot support.
#:
#: A producer that omits the name is not exempted from anything; it has declined to
#: make the claim, and a figure making no ranking claim is an ordinary ordered list.
RANKING_SPEC_NAME = "sweep_ranking"


def _palette_slots() -> int:
    """How many categorical slots the palette holds before Vega recycles from slot 1.

    Read from the artifact, never written here. Distinct from
    :func:`~threetears.evals.vega.palette.validated_slots`, which marks where *validated
    separation* ends: 1-4 are the validated hues and 5-8 the second tier.
    Passing this width is a warning, not a refusal — a palette that will not draw
    legitimate data has answered the wrong question: the palette never refuses to draw.

    A literal here would be a third copy of a number the contract and the palette
    module both already bind, and the drift it would produce is silent: a palette
    widened later would leave the warning firing at the wrong count and naming the
    wrong series, with every suite green.

    It reads the artifact's ``series_slots`` rather than measuring
    ``len(series_colors("dark"))``: the two give the same answer, which is precisely
    why both should not be in the codebase.
    """
    return series_slots()


#: Composition keys that keep their children in the SAME frame — layered marks are
#: drawn on one set of axes, which is why pooling a rule across them is the point.
_LAYER_KEYS = ("layer",)

#: Composition keys that open a NEW frame per child. Vega-Lite resolves their
#: POSITIONAL scales independently by default, so two panels are two coordinate
#: systems the reader reads separately, each with its own axes. Their other scales,
#: ``color`` included, it shares — see :data:`_CONCAT_INDEPENDENT_CHANNELS`.
_CONCAT_KEYS = ("concat", "hconcat", "vconcat")

#: The key under which ``facet`` and ``repeat`` wrap the single view they repeat.
#: One frame per cell, so the inner view is scoped — for the FRAME-level rules —
#: like a concatenated panel. Scale resolution is a separate question with a
#: separate answer per composition: :func:`_concat_normalised`.
_NESTED_VIEW_KEYS = ("spec",)

#: The channels Vega-Lite's ``defaultScaleResolve`` returns ``independent`` for when
#: the model is a concat — ``isXorY(channel) || channel === 'theta' || channel ===
#: 'radius'``, read from ``vega-lite/src/compile/resolve.ts``. Every other channel,
#: ``color`` among them, defaults to ``shared`` there, so declaring one independent
#: is the same violation at a concat root as it is inside a layer. ``theta`` and
#: ``radius`` are listed because the exemption is a statement about Vega-Lite rather
#: than about :data:`_SHARED_SCALE_CHANNELS`, which does not yet reach them.
_CONCAT_INDEPENDENT_CHANNELS = frozenset({"x", "y", "theta", "radius"})


class SpecPolicyError(ValueError):
    """A chart's spec breaks one or more of the report's presentation rules."""

    def __init__(self, violations: list[str]) -> None:
        """Store the full violation list and render it as one message.

        Args:
            violations: Every rule the chart broke, not merely the first — a
                producer fixing a chart wants the whole list in one pass.
        """
        self.violations = violations
        super().__init__("chart breaks the report's presentation rules: " + "; ".join(violations))


def enforce_spec(spec: dict[str, Any]) -> None:
    """Raise unless ``spec`` satisfies every rule.

    All are checked in one pass so a producer fixing a chart gets the whole list
    rather than one violation per attempt. Prose shown beside the chart (a caption,
    a disclosure line) is not an argument: it never reaches the rasteriser, and what
    it says is never a check's question.

    Args:
        spec: A Vega-Lite spec.

    Raises:
        SpecPolicyError: The spec breaks at least one rule.
    """
    violations = check_spec(spec)
    if violations:
        raise SpecPolicyError(violations)


def check_spec(spec: dict[str, Any]) -> list[str]:
    """Check a Vega-Lite spec against the presentation rules.

    Args:
        spec: A Vega-Lite spec, from any producer.

    Returns:
        One message per violation, empty when the spec conforms.
    """
    violations = _check_renderable_colour(spec) + _check_resolved_size(spec)
    for view in _views(spec):
        violations.extend(_check_view(view))
    return violations


@dataclass(frozen=True)
class _View:
    """One frame of a composed spec — the scope the frame-level rules are read in.

    A ``layer`` composes marks that share one set of axes, so its layers are part
    of the SAME frame; ``concat``/``facet``/``repeat`` open a new frame with axes
    of its own. Pooling a rule across the first is the point of the rule; pooling
    it across the second refuses charts that are correct, because two panels
    measuring two quantities is a legitimate figure rather than one axis carrying
    two units.
    """

    label: str
    """Path to this frame's root, e.g. ``vconcat[1]``; empty at the figure root."""

    members: tuple[tuple[str, dict[str, Any]], ...]
    """Every ``(label, node)`` inside this frame: its root and its layers."""

    units: tuple[dict[str, Any], ...]
    """The mark-bearing members, in document order."""

    described: bool
    """Whether this frame or a frame containing it states a ``description``."""

    titled: bool
    """Whether this frame's own title names what it draws.

    Deliberately NOT inherited across a concatenation. A figure's title belongs to
    the figure, and letting it stand in for a panel's own would license every
    suppressed axis title in every panel under any heading at all. It IS inherited
    into a facet's repeated cell, because a facet node and the view it repeats are
    one panel: the heading sits over the cells it titles.
    """

    faceted: bool
    """Whether this frame is a small-multiple cell.

    The one place a legend still earns its keep — one key serves every panel — and
    so the one place the check that replaces legends with direct labels stands down.
    """


@dataclass(frozen=True)
class _Inherited:
    """What a frame passes to the frames it opens.

    A record rather than five positional arguments, because the parameters differ
    in *which* compositions carry them and getting one wrong is invisible: captions
    flow through every child, a title only into a facet's cell.
    """

    described: bool = False
    titled: bool = False
    faceted: bool = False


def _views(spec: Any) -> list[_View]:
    """Split a composed spec into its frames, outermost first.

    Replaces a bare walk over mark-bearing leaves: the rules need to know not only
    *that* a leaf exists but which frame it draws in, because a check that only
    inspected the top level would pass a violation nested one layer down, and one
    that pooled every leaf would refuse a legitimate multi-panel figure.
    """
    views: list[_View] = []
    _walk_views(spec, "", _Inherited(), views)
    return views


def _walk_views(node: Any, label: str, inherited: _Inherited, views: list[_View]) -> None:
    """Record the frame rooted at ``node``, then recurse into the frames it opens."""
    if not isinstance(node, dict):
        return
    members: list[tuple[str, dict[str, Any]]] = []
    nested: list[tuple[str, dict[str, Any]]] = []
    repeated: list[tuple[str, dict[str, Any]]] = []
    _collect_frame(node, label, members, nested, repeated)
    # A `description` on an enclosing frame names the spans drawn inside it: the
    # reader gets one caption per figure, not one per panel.
    described = inherited.described or any(str(member.get("description") or "").strip() for _, member in members)
    titled = inherited.titled or any(_title_text(member) for _, member in members)
    views.append(
        _View(
            label=label,
            members=tuple(members),
            units=tuple(member for _, member in members if "mark" in member),
            described=described,
            titled=titled,
            faceted=inherited.faceted,
        )
    )
    passed_down = _Inherited(described=described, faceted=inherited.faceted)
    for child_label, child in nested:
        _walk_views(child, child_label, passed_down, views)
    # A repeated view is a CELL of the frame above it rather than a picture beside
    # it, so it inherits that frame's heading — and, where the repetition is a
    # facet, the fact that it is one panel of several sharing a key.
    cell = dataclasses.replace(passed_down, titled=titled, faceted=inherited.faceted or _is_small_multiple(node))
    for child_label, child in repeated:
        _walk_views(child, child_label, cell, views)


def _is_small_multiple(node: dict[str, Any]) -> bool:
    """Whether this node repeats one view across the values of a field.

    A repeat over ``layer`` alone is deliberately not one: it stacks its
    repetitions in a single frame rather than laying them out as panels, so there
    is no "one key serves every panel" for a legend to earn its keep by.
    """
    if isinstance(node.get("facet"), dict):
        return True
    repeat = node.get("repeat")
    if isinstance(repeat, dict):
        return bool(repeat.get("row") or repeat.get("column"))
    return isinstance(repeat, list)


def _title_text(node: dict[str, Any]) -> str:
    """The heading a node draws, whether written as a string, lines, or an object."""
    title = node.get("title")
    if isinstance(title, dict):
        title = title.get("text")
    if isinstance(title, list):
        return " ".join(part for part in title if isinstance(part, str)).strip()
    return title.strip() if isinstance(title, str) else ""


def _collect_frame(
    node: dict[str, Any],
    label: str,
    members: list[tuple[str, dict[str, Any]]],
    nested: list[tuple[str, dict[str, Any]]],
    repeated: list[tuple[str, dict[str, Any]]],
) -> None:
    """Gather one frame's nodes, deferring the frames it opens.

    ``facet`` itself is not walked as a composition: it holds field definitions,
    and the view it repeats is under ``spec``.

    The two kinds of opened frame are kept apart because they inherit differently:
    a concatenated panel is its own picture, while a repeated view is a cell of
    this one.
    """
    members.append((label, node))
    for key in _LAYER_KEYS:
        for index, child in enumerate(node.get(key) or []):
            if isinstance(child, dict):
                _collect_frame(child, _path(label, f"{key}[{index}]"), members, nested, repeated)
    for key in _CONCAT_KEYS:
        for index, child in enumerate(node.get(key) or []):
            if isinstance(child, dict):
                nested.append((_path(label, f"{key}[{index}]"), child))
    for key in _NESTED_VIEW_KEYS:
        child = node.get(key)
        if isinstance(child, dict):
            repeated.append((_path(label, key), child))


def _path(parent: str, part: str) -> str:
    """The label of a node reached by ``part`` from the node labelled ``parent``."""
    return part if not parent else f"{parent}/{part}"


def _at(label: str) -> str:
    """Where a violation sits, as a message fragment — empty at the figure root."""
    return "" if not label else f" in view {label}"


def _subject(label: str) -> str:
    """What a violation is about, as a message subject — the spec, or one view."""
    return "spec" if not label else f"view {label}"


def _concat_normalised(node: dict[str, Any]) -> bool:
    """Whether Vega-Lite compiles this node as a CONCAT model.

    Asked of the compile-time model rather than of the spec key, because the
    default scale resolution the exemption rests on is a property of the model and
    two different keys reach it. ``repeat`` is one of them: ``mapNonLayerRepeat``
    in ``vega-lite/src/normalize/core.ts`` returns ``{..., concat}``, so a repeat
    over an array of fields, or over ``row``/``column``, is a concat by the time
    resolution is decided. The one repeat that is not is a repeat over ``layer``
    alone — ``mapLayerRepeat`` returns a ``layer`` when neither ``row`` nor
    ``column`` is present, and a layer shares every scale by default, which is
    precisely the case this rule exists to catch.

    ``facet`` is never a concat: ``defaultScaleResolve`` shares every channel but
    ``theta`` for a facet model, so a facet gets no exemption at all.
    """
    if any(isinstance(node.get(key), list) for key in _CONCAT_KEYS):
        return True
    repeat = node.get("repeat")
    if isinstance(repeat, list):
        return True
    if isinstance(repeat, dict):
        return bool(repeat.get("row") or repeat.get("column"))
    return False


def _check_view(view: _View) -> list[str]:
    """Apply every frame-scoped rule to one frame."""
    violations: list[str] = []
    axis_titles: dict[str, set[str]] = {}
    # Computed over the WHOLE frame, because a value label is always its own layer: the
    # text mark's encoding carries y/x/text and nothing else, while the colour encoding
    # that would put a fill past slot 1 sits on the SIBLING mark layer. A per-unit test
    # could therefore only fire on a text mark carrying its own categorical colour
    # scale, which is not a shape this compiler can produce — so the case the knockout's
    # bound exists for would have passed silently.
    frame_is_coloured = any(
        isinstance(unit.get("encoding"), dict) and "color" in unit["encoding"] for unit in view.units
    )
    for unit in view.units:
        violations.extend(_check_baseline(unit))
        violations.extend(_check_palette(unit))
        violations.extend(_check_knockout_ink(unit, coloured=frame_is_coloured))
        _collect_axis_titles(unit, axis_titles, violations, titled=view.titled)
    violations.extend(_check_one_unit_per_axis(view, axis_titles))
    violations.extend(_check_shared_scales(view))
    violations.extend(_check_named_variability(view))
    violations.extend(_check_direct_labels(view))
    violations.extend(_check_upright_text(view))
    violations.extend(_check_solid_grid(view))
    violations.extend(_check_cropped_axis_labels(view))
    violations.extend(_check_ranked_by_its_measure(view))
    violations.extend(_check_line_order(view))
    return violations


def _mark_type(unit: dict[str, Any]) -> str:
    """The mark's type, whether written as a string or an object."""
    mark = unit.get("mark")
    if isinstance(mark, str):
        return mark
    if isinstance(mark, dict):
        return str(mark.get("type") or "")
    return ""


def _draws_interval(unit: dict[str, Any]) -> bool:
    """Whether a leaf spec draws a band between two values rather than at one.

    Checked STRUCTURALLY rather than by mark name, because the mark name is the
    producer's choice and the reader's problem is the same either way: a `rule`
    from ``x`` to ``x2`` is a confidence interval drawn without ever using the
    word. Restricting this to the two named errorbar marks would have let every
    interval this package actually draws past the interval rule.
    """
    if _mark_type(unit) in _INTERVAL_MARKS:
        return True
    encoding = unit.get("encoding") or {}
    return any(start in encoding and end in encoding for start, end in _SPAN_CHANNELS)


def _check_baseline(unit: dict[str, Any]) -> list[str]:
    """A length-encoding mark must have a baseline to measure its length from.

    A bar measures along ONE axis and is placed by the other. Which is which is not
    the producer's to declare — Vega-Lite decides it, and the channel it measures
    along is the one whose scale asserts a zero. So the rule is stated as the mark's
    own requirement rather than per channel: **at least one positional channel has
    zero on it.** A bar with none measures its length from wherever its domain
    happens to start, which is the ratio the values do not hold.

    Checking every channel instead would refuse a correct chart, and the case is not
    hypothetical: a histogram bin rises from zero on ``y`` while sitting on a value
    axis cropped to the data on ``x``. Both are right — the height is the count and
    the position is the value — and a rule that demanded zero on both would demand
    that a marginal be drawn on an axis starting at zero, which is the very
    rescaling a position axis crops to avoid.
    """
    if _mark_type(unit) not in _LENGTH_MARKS:
        return []
    measured: list[str] = []
    baselined = False
    for channel, definition in (unit.get("encoding") or {}).items():
        if channel not in ("x", "y") or not isinstance(definition, dict):
            continue
        if definition.get("type") != "quantitative":
            continue
        scale = definition.get("scale") or {}
        domain = scale.get("domain")
        if isinstance(domain, list) and len(domain) == 2 and all(isinstance(edge, int | float) for edge in domain):
            # A stated domain is the ANSWER and outranks the `zero` flag it
            # overrides: `zero: false` beside a domain containing zero crops nothing.
            if domain[0] <= 0 <= domain[1]:
                baselined = True
            else:
                measured.append(
                    f"{_mark_type(unit)} mark's {channel} domain {domain} excludes zero — length would encode a ratio the values do not hold"
                )
        elif scale.get("zero") is False:
            measured.append(
                f"{_mark_type(unit)} mark truncates the {channel} baseline (scale.zero is false) — length would encode a ratio the values do not hold"
            )
        else:
            baselined = True
    return [] if baselined else measured


def _check_palette(unit: dict[str, Any]) -> list[str]:
    """A colour encoding uses a fixed, non-cycling order — and is never refused.

    The palette does not refuse to draw past the validated slots: the number of series is a
    property of the data, and refusing to render legitimate data is the one outcome that is never acceptable. Slots 1-4 are the
    validated categorical hues, 5-8 are the second tier, and a domain wider than the
    palette is warned about rather than rejected.

    What remains a violation is a palette that cannot state which slot a category took —
    an absent domain, or an interpolated scheme. Those make the assignment follow the data,
    so the same category takes a different colour from one chart to the next, which no
    amount of drawing can recover.

    **A QUANTITATIVE colour scale is checked too, and by the same rules for the same
    reasons.** The chart-substrate decision admits a hue for a
    level within a swept dimension, drawn as a `linear` scale over the level's rank
    against a named range — and it makes this widening a stated precondition of that
    permission, because a gate reading only `nominal|ordinal` would let a producer reach
    the hue channel by declaring a type this function never looked at. Two of the three
    rules carry over unchanged: an absent domain still lets the ramp's endpoints follow
    the data, so the same level draws at a different lightness from one chart to the
    next, and an interpolated scheme is still not the validated set.

    The slot COUNT is the one rule that does not carry over, and its absence is the
    point rather than an omission: a sequential ramp is a path Vega samples rather than a
    set of slots, so a seventh level closes up against a sixth instead of recycling onto
    the first. Nothing collides, so there is nothing to warn about.
    """
    colour = (unit.get("encoding") or {}).get("color")
    if not isinstance(colour, dict) or colour.get("type") not in ("nominal", "ordinal", "quantitative"):
        return []
    continuous = colour.get("type") == "quantitative"
    kind = "ordered" if continuous else "categorical"
    scale = colour.get("scale")
    if not isinstance(scale, dict) or not isinstance(scale.get("domain"), list):
        # Without an explicit domain the scale follows the DATA — a category takes a
        # different slot, and a level a different lightness, from one chart to the next.
        return [
            (
                f"{kind} colour encoding has no explicit scale.domain — "
                f"{'the ramp endpoints' if continuous else 'slot assignment'} would follow the data, not the "
                f"{'levels' if continuous else 'category'}"
            )
        ]
    violations = []
    if "scheme" in scale or isinstance(scale.get("range"), dict):
        violations.append(
            f"{kind} colour encoding uses a scheme — the palette is a fixed validated set, not an interpolated one"
        )
    if isinstance(scale.get("range"), list):
        # A range given as a list IS a colour literal, and a literal in a spec draws one theme's ink into the other's render.
        violations.append(
            f"{kind} colour encoding states its own range — a spec carries no colour value, so name a config range instead"
        )
    if continuous:
        # A ramp has no slot ceiling to warn about; see the docstring.
        return violations
    domain = scale["domain"]
    slots = _palette_slots()
    if len(domain) > slots:
        # A warning, never an error. Past the palette Vega recycles from slot 1, which makes
        # the ninth series look like the first — real, and admitted here rather than
        # prevented by refusing to draw the data at all. The re-encodings named are the
        # answers that remove the need rather than tolerate it.
        #
        # What keeps it survivable is not this line: it is the direct-label mandate
        # (:func:`_check_direct_labels`), which has already taken identity off the hue
        # channel by the time the hue channel weakens.
        #
        # Three re-encodings named, and only two of them can currently be drawn: the
        # ordered ramp lives in the sweep arm and the facet in the distribution one,
        # while highlight-plus-context has its token authored and no arm that draws
        # it. Named anyway, because the advice is the palette rule's own, and the honest
        # reading of the gap is that a required mechanism is missing rather than that the
        # advice is wrong. Narrowing the advice here would answer that question by
        # deleting it.
        log.warning(
            "chart draws %d categories against %d palette slots — the palette recycles from slot 1, so series %d "
            "repeats series 1's hue. Prefer an ordered ramp if the dimension is ordered, a facet if the sweep is "
            "crossed, or highlight-plus-context if the ranking is flat.",
            len(domain),
            slots,
            slots + 1,
        )
    return violations


def _check_knockout_ink(unit: dict[str, Any], *, coloured: bool) -> list[str]:
    """The on-fill knockout may only be asked for by a text mark, and never beside a colour encoding.

    **The bound was prose, and prose is not a gate.** A palette carries a third text ink
    for a value drawn on a mark's fill, bounded to slot 1 — the single-series mark colour
    it is measured against. Over other slots the same ink can fall short: in the packaged
    palette it reaches 3.52:1 over slot 2 in dark and 2.45:1 over slot 7 in light, under
    the 4.5:1 it exists to clear. So a chart that colours its marks from the categorical
    range and writes a value on one of them would be granted an ink measured for a fill
    it is not drawn on.

    Two refusals, both structural:

    * a **non-text** mark asking for the style, which is a mark asking to be PAINTED in
      a text ink — in the packaged dark palette, the chart surface itself, a hole in the
      figure;
    * a text mark asking for it in a frame that carries a **colour encoding**, which is
      the only way a mark's fill reaches past slot 1 today.

    Nothing in the repo trips either. That is the point of adding it while nothing
    does: a future arm writing a value onto a coloured fill reopens the measurement, and this is what makes that reopening a failure
    rather than a silent contrast regression.

    Args:
        unit: One leaf spec.
        coloured: Whether ANY unit in this frame carries a colour encoding. Passed in
            rather than read off ``unit``, because the value label and the coloured mark
            are always different layers of the same frame.

    Returns:
        Violation messages, or an empty list.
    """
    mark = unit.get("mark")
    style = mark.get("style") if isinstance(mark, dict) else None
    styles = [style] if isinstance(style, str) else list(style or []) if isinstance(style, list) else []
    if VALUE_ON_FILL_STYLE not in styles:
        return []
    if _mark_type(unit) != "text":
        return [
            (
                f"a {_mark_type(unit) or 'non-text'} mark asks for the `{VALUE_ON_FILL_STYLE}` style — that is a text ink "
                "chosen against slot 1's fill (in the packaged dark palette, the chart surface itself), so painting a mark "
                "with it draws a hole; it is for a VALUE written on a fill"
            )
        ]
    if coloured:
        return [
            (
                f"a value label asks for `{VALUE_ON_FILL_STYLE}` in a frame that carries a colour encoding — the on-fill "
                "ink is measured against slot 1 only, and over other slots it can fall under 4.5:1. Re-measure the ink "
                "per slot before drawing a value on a mark whose fill came from the categorical range"
            )
        ]
    return []


def _check_direct_labels(view: _View) -> list[str]:
    """Identity never rides on colour alone, and past the validated hues not at all.

    The palette's never-refuse rule permits it — past four
    hues a chart takes a second tier, past eight it recycles — and the
    thing that makes that honest is not a colour. It is that **a direct label on
    every mark becomes mandatory at five series**, one slot before validated
    separation ends: identity has left the hue channel before the hue channel
    weakens, so weaker separation costs comprehension rather than correctness.
    That safeguard is what this function is.

    Two rules, and they answer two different questions:

    * **The mandate.** Past :func:`~threetears.evals.vega.palette.validated_slots` a
      categorical colour encoding must be accompanied by a per-mark label. There
      is no exemption for a faceted chart: a key at the top of a figure is still a
      colour the reader has to carry, and at five series the colours are no longer
      ones validation says they can carry.
    * **Legends are a fallback**, at every width rather than only below the
      mandate. A legend asks the reader to hold a colour in memory and walk back to
      it, which a label does not. So a legend survives only where one key genuinely
      serves several panels — a faceted chart — and anywhere else the encoding
      suppresses it. The two rules are checked independently: a chart that breaks
      both is told about both, because being told one at a time is the second
      attempt :class:`SpecPolicyError` exists to spare a producer.

    **What counts as a direct label.** A ``text`` mark drawn from the same field
    the colour is, or that field also placed on a positional axis. The second is
    how every chart this compiler emits identifies its categories, and it is a
    direct label in the sense that matters: the name is beside the mark and the
    reader never consults a key.

    **A QUANTITATIVE colour scale answers to the legend rule and not to the
    mandate**, widened alongside :func:`_check_palette` and for the same
    reason — a gate reading only ``nominal|ordinal`` lets a producer reach the hue
    channel by declaring a type it never looked at. The split is not a convenience:
    the mandate exists because a categorical palette RECYCLES past its slots, and a
    ramp does not recycle, so there is no slot count past which a label becomes the
    thing holding identity together. The legend rule has no such dependency — a key
    asks the reader to hold a colour in memory and walk back to it whether that
    colour came from a slot or from a ramp.
    """
    violations: list[str] = []
    for unit in view.units:
        colour = (unit.get("encoding") or {}).get("color")
        if not isinstance(colour, dict) or colour.get("type") not in ("nominal", "ordinal", "quantitative"):
            continue
        domain = (colour.get("scale") or {}).get("domain")
        field = colour.get("field")
        labelled = isinstance(field, str) and _directly_labelled(view, field)
        # The mandate is scoped to a palette that can recycle; a ramp cannot. See the
        # docstring — this is the one of the two rules that does not carry over.
        if colour.get("type") == "quantitative":
            domain = None
        if isinstance(domain, list) and len(domain) > validated_slots() and not labelled:
            violations.append(
                f"{_subject(view.label)} colours {len(domain)} categories — past the {validated_slots()} validated hues — "
                "with no label on the marks; the second tier is only legitimate behind a direct label on every "
                "mark, which is the safeguard that lets the palette draw past validated separation at all"
            )
        # Independent, not `elif`. A five-series chart with neither labels nor a
        # suppressed legend breaks both, and reporting one sends the producer round
        # twice — which is the loop `SpecPolicyError` says in its own docstring that
        # it exists to prevent ("every rule the chart broke, not merely the first").
        #
        # `None` is the suppression; absent is the default, which DRAWS one.
        if not view.faceted and colour.get("legend", _LEGEND_UNSET) is not None:
            violations.append(
                f"{_subject(view.label)} identifies its categories with a legend — a legend asks the reader to hold a "
                "colour in memory and walk back to it, so it survives only for a faceted chart where one key serves "
                "every panel; label the marks directly and set `legend: null`"
            )
    return violations


def _directly_labelled(view: _View, field: str) -> bool:
    """Whether ``field``'s categories are named beside their marks in this frame."""
    for unit in view.units:
        encoding = unit.get("encoding") or {}
        text = encoding.get("text")
        if _mark_type(unit) == "text" and isinstance(text, dict) and text.get("field") == field:
            return True
        for channel in ("x", "y"):
            definition = encoding.get(channel)
            if (
                isinstance(definition, dict)
                and definition.get("field") == field
                and definition.get("type") in ("nominal", "ordinal")
            ):
                return True
    return False


def _check_upright_text(view: _View) -> list[str]:
    """No text in a chart is rotated.

    Rotated text is text the reader turns their head to read, and it is always the
    symptom of a layout decision taken somewhere else — labels too long for the
    space they were given, or a quantity named on an axis that has no room for its
    name. Both have answers that keep the words upright.

    **Two Vega-Lite defaults rotate text without the spec ever saying so**, which
    is why each is flagged whenever it appears rather than on an angle:

    * a ``y`` axis title, whose default ``titleAngle`` is 270. The rule wants the
      quantity named in a heading above the panel instead, so the axis states
      nothing and the title says it once, horizontally.
    * a small-multiple **row header**, whose default ``labelAngle`` is also 270 —
      the case that drew one figure's category names flat in its value panel and
      sideways in the marginal below it, from one payload, with nothing in either
      spec asking for a rotation.
    """
    violations: list[str] = []
    for label, node in view.members:
        for channel, definition in (node.get("encoding") or {}).items():
            axis = definition.get("axis") if isinstance(definition, dict) else None
            if not isinstance(axis, dict):
                continue
            for angled in ("labelAngle", "titleAngle"):
                if _turned(axis.get(angled)):
                    violations.append(
                        f"axis {channel}{_at(label)} sets {angled}={axis[angled]!r} — rotated text is read by turning "
                        "the page; shorten the label, move it above the panel, or give it its own line"
                    )
            if channel == "y" and str(axis.get("title") or "").strip() and "titleAngle" not in axis:
                violations.append(
                    f"axis y{_at(label)} carries a title ({axis['title']!r}) that Vega draws rotated a quarter turn by "
                    "default — name the quantity in a heading above the panel and leave the axis title unset"
                )
        for channel, header in _headers(node):
            if _turned(header.get("labelAngle")):
                violations.append(
                    f"the {channel} header{_at(label)} sets labelAngle={header['labelAngle']!r} — rotated text is read "
                    "by turning the page"
                )
            elif channel == "row" and "labelAngle" not in header:
                # Only `row`. A column header sits above its panel and Vega draws it
                # flat; a row header sits beside one and Vega turns it, so silence
                # means two different things depending on which side it is on.
                violations.append(
                    f"the row header{_at(label)} leaves labelAngle unset, and Vega draws a row header rotated a "
                    "quarter turn by default — state `labelAngle: 0`"
                )
    for unit in view.units:
        mark = unit.get("mark")
        angle = mark.get("angle") if isinstance(mark, dict) else None
        if _turned(angle):
            violations.append(
                f"a {_mark_type(unit)} mark{_at(view.label)} is drawn at angle={angle!r} — chart text is upright"
            )
    return violations


def _turned(angle: Any) -> bool:
    """Whether a stated angle is anything other than upright."""
    return isinstance(angle, int | float) and not isinstance(angle, bool) and angle % 360 != 0


def _headers(node: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(channel, header)`` for every small-multiple this node repeats over.

    Two spellings reach the same drawing: ``facet: {row: {…}}`` on a facet operator
    and ``encoding: {row: {…}}`` on the shorthand, plus a single-field ``facet``
    with no row/column at all, which Vega-Lite wraps as a wrapped-row facet. A rule
    that read only one spelling would hold for whichever this compiler happens to
    use today and not for the producer it exists to police.

    **An ABSENT header is yielded as an empty one**, because absent is not silence:
    it is Vega-Lite's default, and for a row header that default is a quarter turn.
    A rule that only inspected headers a spec had bothered to write would have
    passed the exact defect this check was added for.
    """
    facet = node.get("facet")
    sources: list[tuple[str, Any]] = []
    if isinstance(facet, dict):
        keyed = {channel: facet[channel] for channel in ("row", "column") if channel in facet}
        sources.extend(keyed.items() if keyed else [("row", facet)])
    sources.extend(
        (channel, (node.get("encoding") or {}).get(channel))
        for channel in ("row", "column")
        if isinstance((node.get("encoding") or {}).get(channel), dict)
    )
    for channel, definition in sources:
        if not isinstance(definition, dict):
            continue
        header = definition.get("header")
        yield channel, header if isinstance(header, dict) else {}


def _check_solid_grid(view: _View) -> list[str]:
    """No gridline is dashed.

    A dash is a mark property the reader has been taught to read as data — a
    forecast, a target, a series drawn differently on purpose — and spending it on
    furniture makes the grid compete with what is plotted on it. A grid that needs
    to recede recedes in weight and colour, which the config already gives it.
    """
    violations: list[str] = []
    for label, node in view.members:
        for channel, definition in (node.get("encoding") or {}).items():
            axis = definition.get("axis") if isinstance(definition, dict) else None
            if isinstance(axis, dict) and axis.get("gridDash"):
                violations.append(
                    f"axis {channel}{_at(label)} draws a dashed grid ({axis['gridDash']!r}) — a dash reads as data; "
                    "let the grid recede in colour instead"
                )
    return violations


# There is no footnoted-crop check. The crop footnote was first scoped to span marks and
# then withdrawn altogether: a figure whose axis
# carries labelled numeric ticks has already said where it starts, so the footnote
# repeated a fact the picture states and trained readers to skip the subtitle —
# where the disclosures that DO carry information live.
#
# The protection a reader actually needs is `_check_baseline`, immediately above,
# which refuses a cropped axis under a LENGTH mark outright rather than accepting a
# disclosure for it. That distinction is the whole of it: a bar misstates a ratio,
# while a point or an interval states a position the ticks label. Recorded here
# rather than deleted silently, because "this gate used to exist" is the question a
# reader of `_check_baseline` will have.


def _cropped(definition: dict[str, Any]) -> bool:
    """Whether a quantitative position encoding's scale leaves zero off the axis.

    Read the way :func:`_check_baseline` reads it: a stated two-number domain is the
    answer and outranks the ``zero`` flag, and without one only ``zero: false`` crops
    (Vega-Lite includes zero by default on an unbinned quantitative position).
    """
    scale = definition.get("scale") or {}
    domain = scale.get("domain")
    if isinstance(domain, list) and len(domain) == 2 and all(isinstance(edge, int | float) for edge in domain):
        return not domain[0] <= 0 <= domain[1]
    return scale.get("zero") is False


def _check_cropped_axis_labels(view: _View) -> list[str]:
    """A cropped position axis draws its tick labels, which are the crop's only disclosure.

    The comment above records why there is no crop footnote: labelled ticks already say
    where the axis starts. That argument holds only while the ticks ARE labelled. A
    point or interval mark on a cropped axis with ``labels: false`` (or no axis at all) states positions on
    a ruler whose origin the reader cannot find, so it is refused. A zero-based axis
    may suppress its labels (the distribution marginal's rise axis is ``[0, cell]``,
    and states shape, not counts).

    Read per frame and per channel, because a layer frame shares its axes: a channel
    is unlabelled when no unit in the frame draws its labels, so one layer suppressing
    a duplicate of an axis another layer labels is not a crop left undisclosed.
    """
    cropped: dict[str, str] = {}
    labelled: set[str] = set()
    for label, node in view.members:
        for channel, definition in (node.get("encoding") or {}).items():
            if channel not in ("x", "y") or not isinstance(definition, dict):
                continue
            if definition.get("type") != "quantitative":
                continue
            axis = definition.get("axis", _LEGEND_UNSET)
            # `axis: null` draws no axis at all, so it suppresses the labels as surely
            # as `labels: false` does.
            if not (axis is None or isinstance(axis, dict) and axis.get("labels") is False):
                labelled.add(channel)
            elif _cropped(definition):
                cropped.setdefault(channel, label)
    return [
        (
            f"axis {channel}{_at(label)} is cropped (its domain leaves out zero) and suppresses its tick labels — "
            "the labelled ticks are the only disclosure a crop has, so draw them or start the axis at zero"
        )
        for channel, label in cropped.items()
        if channel not in labelled
    ]


def _collect_axis_titles(
    unit: dict[str, Any], titles: dict[str, set[str]], violations: list[str], *, titled: bool
) -> None:
    """Record each positional quantitative axis's stated unit, flagging any that state none.

    **A quantity may be named above the panel instead of on the axis**, which is
    the only way to name a ``y`` quantity without rotating the words. The two forms
    are distinguished rather than pooled: an axis that sets ``title: null`` has
    *suppressed* its title, and a frame with a heading of its own has somewhere for
    the name to have gone. An axis that simply omits the key has not — Vega-Lite
    fills it with the field name, which is machinery rather than a quantity, and
    that stays a violation.

    A suppressed title contributes nothing to the per-channel pool: it states no
    unit, so it cannot disagree with one. Two layers where one states a unit and
    the other defers to the heading are read as one stated unit, which is what
    they are.

    Args:
        unit: One mark-bearing node.
        titles: Per-channel accumulator of the units stated in this frame.
        violations: Accumulator, appended to in place.
        titled: Whether this frame carries a heading of its own.
    """
    for channel, definition in (unit.get("encoding") or {}).items():
        if channel not in ("x", "y") or not isinstance(definition, dict):
            continue
        if definition.get("type") != "quantitative":
            continue
        axis = definition.get("axis")
        title = None if axis is None else (axis or {}).get("title")
        if not str(title or "").strip():
            if titled and isinstance(axis, dict) and "title" in axis and axis["title"] is None:
                continue
            violations.append(
                f"quantitative {channel} axis states no title — a drawn quantity carries its unit or the reader supplies one"
            )
            continue
        titles.setdefault(channel, set()).add(str(title))


def _check_one_unit_per_axis(view: _View, axis_titles: dict[str, set[str]]) -> list[str]:
    """One frame's axis states one quantity.

    Pooled per frame rather than per spec: two layers on one axis are one ruler
    and must agree, while two concatenated panels are two rulers with two axis
    labels, and a figure that puts latency beside cost is a report doing its job.
    """
    return [
        f"axis {channel}{_at(view.label)} carries more than one unit ({', '.join(sorted(titles))}) — "
        "one axis states one quantity, or the reader reads two rulers as one"
        for channel, titles in sorted(axis_titles.items())
        if len(titles) > 1
    ]


def _check_shared_scales(view: _View) -> list[str]:
    """Layers and facet cells share one ruler per channel.

    Checked on every node of the frame, not only the spec's root, because a
    ``resolve`` one level down does exactly what a ``resolve`` at the root does.

    A concatenation is exempt for the channels Vega-Lite itself resolves
    independently there (:data:`_CONCAT_INDEPENDENT_CHANNELS`) and for no others:
    its panels are separate frames with separate axes, so refusing an independent
    ``x`` would refuse the two-quantity figure :func:`_check_one_unit_per_axis`
    deliberately allows.

    **``color`` is checked against its own harm rather than wherever it appears.** The harm this
    rule names for a colour scale is exact — *"one
    category takes a different colour from one panel or layer to the next"* — and
    it is a statement about categories appearing TWICE. Where the children colour
    disjoint category sets it cannot happen: a barcode whose ordered levels take a
    ramp and whose categorical levels take hues is two scales over two vocabularies
    that share no member, and forcing them onto one ruler would be the real defect,
    since it would put a concurrency level and a model name in one domain. So the
    check reads the domains and refuses the overlap, which is what the sentence
    always said. Positional channels get no such escape: their harm is about
    RULERS, and two rulers mislead whether or not they measure the same values.
    """
    violations = []
    for label, node in view.members:
        exempt = _CONCAT_INDEPENDENT_CHANNELS if _concat_normalised(node) else frozenset()
        resolve = (node.get("resolve") or {}).get("scale") or {}
        for channel in _SHARED_SCALE_CHANNELS:
            if channel in exempt or resolve.get(channel) != "independent":
                continue
            if channel == "color" and _colours_disjoint_vocabularies(node):
                continue
            violations.append(
                f"scale for {channel} is resolved independently{_at(label)} — {_SHARED_SCALE_HARM[channel]}"
            )
    return violations


def _colours_disjoint_vocabularies(node: dict[str, Any]) -> bool:
    """Whether ``node``'s children colour category sets that share no member.

    The evidence :func:`_check_shared_scales` needs to tell an independent colour
    scale that misleads from one that cannot. Three conditions, and every one of
    them has to hold — the default is that an independent colour scale is the
    defect the rule names, and this only lifts it where the named harm is
    impossible rather than merely unlikely.

    1. **The children are distinct views.** Only ``layer`` and the concat keys
       qualify. ``facet`` and ``repeat`` draw the SAME encoding once per panel, so
       every panel's domain is the same domain — the collision is total, and the
       fact that there is one child rather than several is exactly why it looks
       like none.
    2. **There are at least two of them.** One child cannot disagree with a sibling
       it does not have, and declaring independence there says nothing about what a
       second one would do.
    3. **Every domain is stated, and no category appears twice.** An unstated
       domain follows the DATA, so nothing can rule out that it names a category a
       sibling also draws.

    Args:
        node: A composition node.

    Returns:
        Whether an independently resolved colour scale is harmless here.
    """
    children = [child for key in _LAYER_KEYS + _CONCAT_KEYS for child in node.get(key) or []]
    if len(children) < 2:
        return False
    seen: set[str] = set()
    for child in children:
        domains: set[str] = set()
        for unit in _collect_units(child):
            colour = (unit.get("encoding") or {}).get("color")
            if not isinstance(colour, dict):
                continue
            domain = (colour.get("scale") or {}).get("domain")
            if not isinstance(domain, list):
                return False
            # Qualified by the scale's TYPE, because the vocabularies are only
            # comparable within one. A ramp's domain is `[0, n]` and a categorical
            # lever may legitimately have a level spelled `"0"`; compared as bare
            # strings those read as the same category and refuse a conforming chart.
            # Two NOMINAL scales sharing `"0"` is still the collision this is for.
            domains.update(f"{colour.get('type')}:{entry}" for entry in domain)
        if domains & seen:
            return False
        seen |= domains
    return bool(seen)


def _collect_units(node: Any) -> list[dict[str, Any]]:
    """Every mark-bearing node at or under ``node``, at any composition depth."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if "mark" in node:
            found.append(node)
        for value in node.values():
            found.extend(_collect_units(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_collect_units(item))
    return found


def _check_ranked_by_its_measure(view: _View) -> list[str]:
    """A frame calling itself a ranking is ordered by a quantity it plots, descending.

    The rule the sweep barcode rests on. A contiguous block in one dimension's
    column is the finding — that lever is what drove the ranking — and ordering the
    rows by that column manufactures the block from nothing. The two orders are
    indistinguishable in the picture: the marks are identical, only their sequence
    differs, and a reader has no way to tell a discovered block from a sorted one.

    **What a rendered spec can show, and how this reads it.** The row order is in
    the spec — a categorical positional encoding states its ``sort`` as an explicit
    list, because a sort resolved inside Vega would exist only in the picture — and
    the plotted quantities are in the frame's inline rows. So the check joins the
    two: walk the stated order, read each row's value for each quantitative field
    the frame places, and require at least one of them to be non-increasing. A
    ranking ordered by its measure satisfies that by construction; one ordered by a
    dimension agrees with no quantity the chart draws, which is the whole of what
    "sorted by a dimension" means from outside.

    Three things are refused, and they are the three ways deliverable 3's "best
    top-right" is lost:

    * **An implicit order.** A ranking whose categories carry no stated sort is
      ordered by whatever the renderer decides, which is not a ranking.
    * **An order matching no drawn quantity**, which includes ascending: the
      staircase then climbs, and rank 1 is at the bottom.
    * **A reversed value scale**, which turns a descending order back into an
      ascending picture without touching the data.

    Frames that make no ranking claim are untouched. ``breakdown`` sorts by value
    and would pass; ``delta_table`` orders by the payload's own sequence and would
    not, which is correct for it — a comparison of metrics is not a ranking, and
    only a spec that says it is one is held to this.
    """
    if not any(node.get("name") == RANKING_SPEC_NAME for _, node in view.members):
        return []
    violations: list[str] = []
    identity = _identity_placement(view)
    if identity is None:
        return [
            (
                f"{_subject(view.label)} declares itself a ranking but states no explicit row order — the order would be "
                "the renderer's, and a ranking whose sequence is decided downstream ranks nothing"
            )
        ]
    field, order = identity
    violations.extend(
        f"{_subject(view.label)} declares itself a ranking and reverses its value scale — a reversed axis turns a "
        "descending order back into an ascending picture, so the best configuration no longer sits where the form puts it"
        for encoding in _quantitative_placements(view)
        if (encoding.get("scale") or {}).get("reverse") is True
    )
    plotted = _ranked_quantities(view, field, order)
    if not plotted:
        # Nothing quantitative is placed against the categories — the shape a
        # constrained slice that admitted nothing compiles to, where the figure is
        # the sentence. There is no order to be wrong about.
        return violations
    if not any(_non_increasing(values) for values in plotted.values()):
        violations.append(
            f"{_subject(view.label)} declares itself a ranking but its row order matches none of the quantities it "
            f"draws ({', '.join(sorted(plotted))}) descending — a sweep ordered by one of its dimensions manufactures "
            "the block pattern that was supposed to be the finding; sort by the ranked measure, best first"
        )
    return violations


def _identity_placement(view: _View) -> tuple[str, list[str]] | None:
    """The frame's stated category order: ``(field, order)``, or ``None``.

    A ``sort`` given as an explicit list is what separates a compiled order from a
    renderer's — the same distinction the compiler's own identity axis is built on.
    """
    for _, node in view.members:
        for channel, definition in (node.get("encoding") or {}).items():
            if channel not in ("x", "y") or not isinstance(definition, dict):
                continue
            if definition.get("type") not in ("nominal", "ordinal"):
                continue
            order, field = definition.get("sort"), definition.get("field")
            if isinstance(order, list) and isinstance(field, str) and order:
                return field, [str(entry) for entry in order]
    return None


def _quantitative_placements(view: _View) -> Iterator[dict[str, Any]]:
    """Every quantitative positional encoding this frame draws."""
    for _, node in view.members:
        for channel, definition in (node.get("encoding") or {}).items():
            if channel in ("x", "y") and isinstance(definition, dict) and definition.get("type") == "quantitative":
                yield definition


def _ranked_quantities(view: _View, identity: str, order: list[str]) -> dict[str, list[float]]:
    """Each drawn quantity's values, read out in the frame's stated row order.

    Args:
        view: The frame.
        identity: The row key carrying each row's category.
        order: The stated category order.

    Returns:
        Field name → its values along ``order``, for every quantitative field the
        frame both places and holds rows for. Categories the field has no row for
        are skipped rather than zero-filled: an absent value is not a small one.
    """
    placed = {
        definition["field"] for definition in _quantitative_placements(view) if isinstance(definition.get("field"), str)
    }
    ranked: dict[str, list[float]] = {}
    for field in sorted(placed):
        by_category: dict[str, float] = {}
        for _, node in view.members:
            for row in _inline_rows(node):
                value, category = row.get(field), row.get(identity)
                if isinstance(value, int | float) and not isinstance(value, bool) and isinstance(category, str):
                    by_category.setdefault(category, float(value))
        values = [by_category[category] for category in order if category in by_category]
        if values:
            ranked[field] = values
    return ranked


def _non_increasing(values: list[float]) -> bool:
    """Whether a run of values never rises — the shape a descending ranking has."""
    return all(earlier >= later for earlier, later in itertools.pairwise(values))


def _check_line_order(view: _View) -> list[str]:
    """A connecting mark over a categorical position states the order of its categories.

    A line, a trail or an area joins its points in the order of the scale they sit on. For a quantitative
    or temporal scale that order is the values' own; for a nominal or ordinal one it is whatever the spec
    states — and a spec that states nothing gets Vega-Lite's default, which sorts the categories by name.
    Over builds that draws ``0.10`` before ``0.9``; over anything else it draws the alphabet. The picture
    is a clean line either way, so nothing on it tells the reader the order is wrong.

    **Stated means a list**: an explicit ``sort`` list, or an explicit ``scale.domain``, either of which
    fixes the sequence the line walks. The channel is read off the mark's own encoding over the frame
    root's, which is where a layered spec shares an encoding among its marks.
    """
    shared = (view.members[0][1].get("encoding") or {}) if view.members else {}
    violations: list[str] = []
    for unit in view.units:
        mark = _mark_type(unit)
        if mark not in _CONNECTING_MARKS:
            continue
        encoding = shared | (unit.get("encoding") or {})
        for channel in ("x", "y"):
            definition = encoding.get(channel)
            if not isinstance(definition, dict) or definition.get("type") not in ("nominal", "ordinal"):
                continue
            stated = isinstance(definition.get("sort"), list) or isinstance(
                (definition.get("scale") or {}).get("domain"), list
            )
            if not stated:
                violations.append(
                    f"a {mark} mark{_at(view.label)} connects its points along a categorical {channel} with no stated "
                    "order — Vega sorts the categories by name, so the line draws a trend in an order the data never "
                    "had; state the order as a `sort` list or a `scale.domain`"
                )
    return violations


def _inline_rows(node: dict[str, Any]) -> list[dict[str, Any]]:
    """The inline data rows one node declares, if it declares any."""
    data = node.get("data")
    values = data.get("values") if isinstance(data, dict) else None
    return [row for row in values or [] if isinstance(row, dict)]


def _check_named_variability(view: _View) -> list[str]:
    """A frame that draws a span says what the span varies over."""
    if view.described or not any(_draws_interval(unit) for unit in view.units):
        return []
    return [
        (
            f"{_subject(view.label)} draws a mark between two values but has no `description` naming what that span is — "
            "the same band means different things across runs than across cases"
        )
    ]


def _check_renderable_colour(spec: dict[str, Any]) -> list[str]:
    """No colour in the spec is one the server-side rasteriser cannot parse.

    The substrate's most expensive discovered hazard, and until now it was guarded
    only where the palette is loaded and by an assertion over this repo's own
    compiler. That leaves the rule absent from the layer built to be the universal
    gate — and a generator-emitted spec is the stated destination, so the producer
    this most needs to catch is the one that does not exist yet.

    Unreachable from the current compiler, which provably emits no colour at all.
    That is the point: a rule that only fires for a producer nobody has written is
    exactly the rule that has to exist before they write it.

    A string counts only when the whole of it is a colour value — see
    :data:`_UNRENDERABLE_COLOUR` for why a title mentioning one does not.
    """
    offenders = sorted({text for text in _rendered_strings(spec) if _UNRENDERABLE_COLOUR.fullmatch(text)})
    if not offenders:
        return []
    return [
        (
            f"spec carries a colour notation the server-side rasteriser cannot parse ({'; '.join(offenders)}) — "
            "it renders BLACK with no warning, so the failure is invisible in the output; use resolved sRGB hex"
        )
    ]


def _check_resolved_size(spec: dict[str, Any]) -> list[str]:
    """No view asks a renderer to decide how wide it is.

    ``"width": "container"`` is Vega-Lite's "fill your parent", and it is the same
    class of hazard as an unparseable colour: it fails by drawing something rather
    than by refusing. Vega-Lite does not support it under ``facet``, ``concat`` or
    ``repeat`` at all, so a composed view carrying it falls back to Vega's own
    default — measured at 362px against a plot area of 832 — and rasterises a valid
    PNG with every mark drawn, no warning either way. The browser, meanwhile,
    honours it. One spec then draws at two sizes on two surfaces, which is the
    disagreement the compiled substrate exists to remove.

    Each renderer used to substitute a width of its own, and the substitution was
    where the rule lived. The compiler now resolves every size itself, so the
    substitution is gone — and this is what stops that removal from being a rule
    that holds only because of who happens to be emitting specs today.

    Unreachable from the current compiler, which emits numbers at every depth and
    is asserted to. That is the point, exactly as it is for the colour rule: a
    generator-emitted spec is the stated destination, so the producer this most
    needs to catch is the one nobody has written yet.

    **A size of its own is not required, only resolution.** A layer inside a sized parent
    legitimately carries no size of its own, and several shapes here compile that
    way; demanding a size on every node would refuse correct charts.

    **Nor is a bare number the only resolved form.** ``{"step": 20}`` sizes a
    discrete axis per band, and it IS resolved — the renderer computes it from the
    data rather than from a container it has to measure. Refusing it would refuse
    legitimate Vega-Lite from exactly the generator-emitted producer this rule
    exists to police, which is how a gate ends up worse than no gate.
    """
    offenders = sorted({f"{path}={value!r}" for path, value in _declared_sizes(spec) if not _is_resolved_size(value)})
    if not offenders:
        return []
    return [
        (
            f"spec leaves a size for a renderer to resolve ({'; '.join(offenders)}) — "
            "unsupported under facet/concat, where it silently rasterises at the renderer's own default "
            "while the browser honours it; compile a number"
        )
    ]


def _is_resolved_size(value: Any) -> bool:
    """Whether a declared dimension is one the renderer can draw without measuring.

    A number is. So is ``{"step": N}`` — a per-band size the renderer derives from
    the data. ``"container"`` is not, and neither is any other string.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return True
    return (
        isinstance(value, dict)
        and isinstance(value.get("step"), int | float)
        and not isinstance(value.get("step"), bool)
    )


def _declared_sizes(node: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Yield ``(path, value)`` for every VIEW dimension a spec declares.

    Two subtrees are skipped, because a ``width`` inside either is not a view's:

    * ``mark`` — a dumbbell's connector states a fraction of its band, which is a
      legitimate mark height and not a view that failed to resolve one.
    * ``data`` — inline row values are the CHART'S SUBJECT. A payload plotting a
      column named ``width`` is ordinary data, and reading it as an unresolved
      dimension would refuse a chart for drawing the thing it was asked to draw.
    """
    if isinstance(node, list):
        for index, item in enumerate(node):
            yield from _declared_sizes(item, f"{path}[{index}]")
    elif isinstance(node, dict):
        for key, value in node.items():
            if key in ("mark", "data"):
                continue
            where = f"{path}.{key}" if path else key
            if key in ("width", "height"):
                yield where, value
            else:
                yield from _declared_sizes(value, where)


def _rendered_strings(node: Any) -> Iterator[str]:
    """Yield every string value a spec carries, each on its own.

    Field names and encoding types are excluded — they are machinery, never a
    colour. Each string is yielded as the value it is, because the colour rule reads
    VALUES: a multi-line title's lines are glyphs, not a colour, so joining them
    would only manufacture text the rule has no business reading.
    """
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from _rendered_strings(item)
    elif isinstance(node, dict):
        for key, value in node.items():
            if key in ("field", "type", "$schema", "format", "as", "op", "sort"):
                continue
            yield from _rendered_strings(value)


__all__ = [
    "RANKING_SPEC_NAME",
    "SpecPolicyError",
    "check_spec",
    "enforce_spec",
]
