"""How a host's output looks — a bounded contract, deliberately with no prose in it.

**The hard rule: no free-text style prose ever reaches the generator prompt.** This repo has
proven that prose becomes instruction — a "tone" sentence handed to a model is an instruction it
follows, and a host that can write one can steer the analysis rather than style it.

That is enforced structurally rather than promised:

1. :attr:`StyleProfile.tone_register` is an **enum**, and the words it maps to are engine-owned
   prompt fragments. The host picks from a list; it never writes the instruction.
2. **The one non-enum field cannot carry an instruction to a model.** ``chart_palette`` holds
   colours and nothing else: every value is checked to be resolved sRGB hex
   (:func:`require_resolved_colour`) when the palette is built, so it has no room for a word. And it
   is structural besides: :func:`prompt_fragment` is the only function in this module that returns
   prompt text, it takes no argument but the register, and :func:`assert_no_style_text` proves no
   value from the palette reaches a given prompt.
3. **No field here is read by the prompt builder.** :func:`prompt_fragment` is the single
   function that turns any of this into prompt text, and it reads exactly one enum.
4. :func:`assert_no_style_text` asserts a prompt carries no token traceable to a host free-text
   field, with **descriptor prose and case text** the deliberate exceptions — those are
   *substance*, the domain words riding in the data, not style. ``tests/test_host_contract.py``
   holds the narrow half (the one function turning style into prompt text reads an engine-owned
   table); no test in this repository yet runs it over an assembled generator prompt. It is a
   TEST and not a production gate on purpose: point 3 is what production actually rests on, and a
   substring scan gating a billed generation would discard a real analysis for a colour that
   happens to appear in a case's own text.

**Every field changes what a report shows, and nothing else is declared.** A locale, units, a date
format and a length budget are not slots here: no renderer and no number formatter reads one, and a
field that claims to change formatting while changing nothing is worse than no field — a host that
declared ``de-DE`` would get reports formatted exactly as ``en-US`` with nothing telling it so. Each
comes back only with the code that honours it.

**The palette is renderer-neutral.** The core names chart INTENT — which colour slot a series takes
(:data:`VALIDATED_SLOTS`, :data:`SERIES_SLOTS`), which ink a label is drawn in — and a renderer is an
adapter that turns a :class:`ChartPalette` into its own theme. Nothing here is any renderer's config
shape: a Vega-Lite config is built from a palette inside the ``vega`` adapter, and a host never writes
one.

**Style never touches substance.** The memo shape, ordering, gating semantics, truth gates,
posture model, caveat attachment and the materiality mechanism are engine-owned and identical for
every consumer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

#: The registers a host may pick from. Engine-owned and closed — the point of the enum is that a
#: host cannot write its own.
ToneRegister = Literal["neutral", "executive", "technical"]


class StyleError(ValueError):
    """A style profile contradicts the bounded contract this module promises."""


#: How many categorical colour slots a chart may assign with validated separation.
#:
#: Part of eval's chart vocabulary, not of any palette: a chart intent names colour SLOTS, never colours,
#: and a palette supplies the hues. Slots 1-4 are the ones a palette must separate for colourblind
#: readers and against its background; 5-8 are a derived second tier that need not; past
#: :data:`SERIES_SLOTS` a renderer recycles from slot 1. The core decides what a slot promises and the
#: palette keeps the promise.
VALIDATED_SLOTS = 4

#: How many categorical colour slots a palette supplies before it recycles — the width of the vocabulary,
#: and so exactly how many :attr:`ChartPalette.series` colours a palette declares.
SERIES_SLOTS = 8

#: Resolved sRGB hex, the one colour notation every renderer this package knows can draw. A notation a
#: rasteriser cannot parse (``oklch(...)`` through resvg) does not raise; it draws BLACK, with a valid
#: image and no warning — so the notation is held here, where a palette is built, rather than found in a
#: picture.
_RESOLVED_COLOUR = re.compile(r"^#[0-9A-Fa-f]{6}$")


def require_resolved_colour(where: str, value: object) -> str:
    """The palette's one colour check: ``value`` is resolved sRGB hex (``#rrggbb``), or a refusal.

    Every palette — a host's :class:`ChartPalette` and the ``vega`` adapter's packaged one — is checked
    by this function, so "what a palette may hold" has one answer.

    Args:
        where: What the value is, for the refusal (``chart_palette.series[2]``).
        value: The colour.

    Returns:
        The colour, unchanged.

    Raises:
        StyleError: ``value`` is not a ``#rrggbb`` string — another notation (``oklch(...)``, a named
            colour, three-digit hex) included, since a renderer may draw it black without raising.
    """
    if not isinstance(value, str) or not _RESOLVED_COLOUR.match(value):
        raise StyleError(
            f"{where} is {value!r}, not resolved sRGB hex (#rrggbb) — a palette carries resolved colours only, "
            "because a rasteriser handed another notation (oklch(...) through resvg) draws it black without raising"
        )
    return value


@dataclass(frozen=True)
class ChartPalette:
    """A host's chart colours, by role — what a renderer themes every chart it draws with.

    Renderer-neutral: each field is a role a chart intent already speaks in (a numbered series slot, the
    ink a label is drawn in, the neutral a receded mark takes), never a renderer's config key. Every
    colour is resolved sRGB hex (:func:`require_resolved_colour`). The renderer's own type scale, font
    and geometry are not here: they are the renderer's, not the host's palette.

    What is checked is the SHAPE — the notation of every colour, and that :attr:`series` is exactly the
    :data:`SERIES_SLOTS` the chart vocabulary assigns. Whether slots 1-:data:`VALIDATED_SLOTS` actually
    separate for colourblind readers against :attr:`background` is the palette author's measurement to
    make; nothing here measures colour.

    Attributes:
        series: The categorical slots, slot 1 first — exactly :data:`SERIES_SLOTS`. Slot 1 is also the
            single-series mark colour.
        sequential: An ordered ramp's stops, lightest first — at least two. A renderer samples it as a
            path, so the stop count is no ceiling on how many levels a dimension may have.
        background: The chart's surface.
        ink: Every label, value and title.
        muted: A subtitle, and a legend's neutral symbols.
        grid: Gridlines.
        rule: Axis domains, ticks and a zero rule.
        context: A mark drawn as context rather than as the answer — a dominated contestant, a recycled row.
        on_fill: A value drawn ON slot 1's fill (the knockout), where :attr:`ink` would not clear contrast.
    """

    series: tuple[str, ...]
    sequential: tuple[str, ...]
    background: str
    ink: str
    muted: str
    grid: str
    rule: str
    context: str
    on_fill: str

    def __post_init__(self) -> None:
        """Refuse a palette a renderer could not draw every chart with.

        Raises:
            StyleError: A colour is not resolved sRGB hex, :attr:`series` is not exactly
                :data:`SERIES_SLOTS` colours, or :attr:`sequential` has fewer than two stops.
        """
        object.__setattr__(self, "series", tuple(self.series))
        object.__setattr__(self, "sequential", tuple(self.sequential))
        if len(self.series) != SERIES_SLOTS:
            raise StyleError(
                f"chart_palette.series holds {len(self.series)} colour(s); a palette supplies exactly the "
                f"{SERIES_SLOTS} slots a chart assigns (1-{VALIDATED_SLOTS} validated, the rest a second tier), so "
                "a renderer never recycles before the vocabulary says it does"
            )
        if len(self.sequential) < 2:
            raise StyleError(
                f"chart_palette.sequential holds {len(self.sequential)} stop(s); an ordered ramp is a path between "
                "at least two"
            )
        for at, colour in enumerate(self.series):
            require_resolved_colour(f"chart_palette.series[{at}]", colour)
        for at, colour in enumerate(self.sequential):
            require_resolved_colour(f"chart_palette.sequential[{at}]", colour)
        for role in ("background", "ink", "muted", "grid", "rule", "context", "on_fill"):
            require_resolved_colour(f"chart_palette.{role}", getattr(self, role))

    def colours(self) -> list[str]:
        """Every colour the palette holds, slots first then each role — what a purity check walks."""
        return [
            *self.series,
            *self.sequential,
            self.background,
            self.ink,
            self.muted,
            self.grid,
            self.rule,
            self.context,
            self.on_fill,
        ]


#: The engine-owned prompt fragment each register maps to. **This mapping is the reason
#: ``tone_register`` can be safe:** the host picks a key and the engine supplies the words, so no
#: host sentence ever reaches a model. Editing these is an engine change, reviewed as one.
_TONE_FRAGMENTS: dict[ToneRegister, str] = {
    "neutral": "Write plainly. State what the evidence supports and no more.",
    "executive": "Lead with the decision. Keep supporting detail to what changes that decision.",
    "technical": "Prefer precise mechanism over summary. Name the measure behind every claim.",
}


@dataclass(frozen=True)
class StyleProfile:
    """One host's bounded presentation contract.

    Every field is either an engine-owned enum or a value the renderer consumes. There is
    deliberately **no** string field a host can fill with instructions, and no field that nothing reads.
    """

    tone_register: ToneRegister = "neutral"
    """Which engine-owned register to write in. The host picks; the engine supplies the words."""

    chart_palette: ChartPalette | None = None
    """The host's chart colours, which every renderer it builds draws with; never serialised into a prompt.

    ``None`` declares no palette: the host has chosen to draw in a renderer's packaged one, and a renderer
    built for this style (``VegaRenderer.for_style``) says so by taking it. That is the host's stated
    choice, not a substitute for something it declared — a declared palette is always the one drawn.
    """


def prompt_fragment(style: StyleProfile) -> str:
    """The only text this module contributes to a generator prompt.

    One function, reading one enum, so "does host style reach the prompt" has a single call site
    to inspect rather than being a property of how carefully each caller behaved.

    Args:
        style: The host's style profile.

    Returns:
        The engine-owned fragment for the profile's register. Never host-supplied text.
    """
    return _TONE_FRAGMENTS[style.tone_register]


def assert_no_style_text(prompt: str, style: StyleProfile) -> None:
    """Raise when a host-supplied style **string** appears verbatim in ``prompt``.

    The structural half of the no-style-text-in-prompts promise, as a callable rather than a habit: it
    walks whatever the host actually supplied and proves none of it is there.

    **Scoped to the values a host writes**, which is what makes it safe to point at a whole assembled
    prompt rather than only at :func:`prompt_fragment`'s output: every colour of ``chart_palette``, each
    held to ``#rrggbb``. The closed enums are out — see
    :func:`_host_supplied_strings` for why matching them reports the engine's own words as a leak.

    Args:
        prompt: The assembled prompt text to check.
        style: The profile whose values must not appear in it.

    Raises:
        StyleError: A host-supplied style value is present in the prompt, naming the value.
    """
    leaked = [value for value in _host_supplied_strings(style) if value and value in prompt]
    if leaked:
        raise StyleError(f"host style values reached the prompt: {sorted(leaked)} — this contract carries no free text")


def _host_supplied_strings(style: StyleProfile) -> list[str]:
    """Every string a host WROTE into ``style``.

    Args:
        style: The profile to walk.

    Returns:
        Every colour of ``chart_palette`` (none when it declares no palette).

        ``tone_register`` is deliberately absent: it is a CLOSED engine-owned enum, so a host picks
        a key from a list and cannot hold a sentence in it, and a value that cannot carry an
        instruction is not what this check is for. The engine-owned tone fragment is absent for
        the opposite reason — it is the one thing that is *supposed* to reach a prompt.
    """
    return style.chart_palette.colours() if style.chart_palette is not None else []


__all__ = [
    "SERIES_SLOTS",
    "VALIDATED_SLOTS",
    "ChartPalette",
    "StyleError",
    "StyleProfile",
    "ToneRegister",
    "assert_no_style_text",
    "prompt_fragment",
    "require_resolved_colour",
]
