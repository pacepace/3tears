"""How wide a string draws, so the compiler can decide layout instead of guessing.

Series are full model IDs with no aliases and the label gutter is a fixed 176px,
so *whether a label fits* has to be computed. The alternative is what the charts
did before: hand Vega a ``labelLimit`` and let it truncate, which turns a model ID
into ``anthropic/claude-opus-4-2025…`` and loses exactly the characters that
distinguish one build from another.

**The width comes from a measured table, not from the font file.** The advances in
``font_metrics.json`` were measured through vl-convert's own layout and written by
:func:`write_font_metrics`, so they describe what Vega will do rather than
what the TTF declares — a variable font's declared advances depend on axis values
this stack never sets. The artifact is committed because a measurement taken at
compile time would cost a full Vega layout pass on a request path. The tool that
takes it is ``packages/evals/scripts/measure_font_metrics.py`` — dev tooling, not
part of the installed package.

**A table belongs to one face, and travels with it.** The table is a
:class:`~threetears.evals.contracts.host.ChartFont` — the family list a renderer
emits and the advances measured for its first family, in one value — so a layout
can never be computed from one face's widths and drawn in another. The packaged one
(:func:`packaged_font`) is **Liberation Sans**, chosen because it is the one face this
stack can promise everywhere: vl-convert embeds it, so a server-side raster draws it
on a host with no fonts installed at all, and it is metric-compatible with Arial, so
a browser that lacks it and falls back to Arial lays the same strings out at the same
widths. A host that draws in its own face declares it, measured, on its style profile
and every layout function here takes it as ``font``; ``None`` means the packaged face.

**The sum is an estimate, and its error has a direction that matters.** A
character's advance depends on its neighbours — in the packaged face a run of ``1``
draws 0.489 of the font size per digit against 0.556 for one between two ``H``
glyphs, because the face kerns that pair — so no per-character table can be exact
for every string. The table holds each character's *widest* measured neighbourhood,
and its widest across the type scale's weights, which makes the sum lean long:
measured against a corpus of real model IDs and measure names it never
underestimates past the probe's rounding (0.02%), and it overestimates a
deliberately kern-heavy string like ``AVAVAV…`` by 14%. Folding the weights in costs
regular-weight text 5-9% of headroom on that corpus, since the packaged face's bold
is wider; a name that close to its gutter moves to its own line, which is the cheap
error.

Leaning long is the deliberate choice, because the two errors do not cost the
same. Too narrow says a label fits when it does not, and the reader gets the
truncation the label rules forbid. Too wide moves the label to its own line, which the
same rules call cheap. :data:`SAFETY_MARGIN` is the remaining allowance, applied
by :func:`fits` and never by :func:`text_width` — a caller that wants the raw
estimate gets the raw estimate, and a caller asking the layout question gets the
conservative answer.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from threetears.evals.contracts.host import ChartFont, StyleError

#: The measured artifact, beside the module that reads it.
#:
#: This is the pattern the chart palette was later moved onto, and the two are now
#: siblings here for the same reason: a package that must be installable reads only
#: what ships with it. They are still produced by different tools — nothing in the
#: token pipeline produces this one, and the token build does not regenerate it.
#: Each artifact names its own generator in its own ``$comment``.
METRICS_PATH = Path(__file__).resolve().parent / "font_metrics.json"

#: How far over its estimate a string is allowed to draw before the estimate is
#: not to be trusted, as a fraction.
#:
#: What this covers is the neighbourhoods the measurement did not visit. Each
#: character was measured isolated and repeated and the table kept the wider, so
#: the two contexts a probe can reach are already accounted for; what remains is a
#: character sequence that draws wider than either — reachable in principle, and
#: not observed on any corpus string, where the sum's worst case is to overestimate.
#:
#: Four percent is chosen against that residual rather than measured from it,
#: which is why the number sits here with its reasoning instead of in the artifact
#: with the measurements. No corpus string on the packaged face draws wider than its
#: estimate at all, and the asymmetry is what makes erring large correct:
#: too tight truncates a model ID, too loose spends vertical space the label rules call
#: cheap. :func:`write_font_metrics` refuses to write a table whose corpus
#: underestimates past this, so a font or rasteriser change that invalidates the
#: choice fails at the point of regeneration rather than in a report.
SAFETY_MARGIN = 0.04


class TextMetricsError(RuntimeError):
    """The font metrics artifact is missing or unusable."""


@lru_cache(maxsize=1)
def load_metrics() -> dict[str, Any]:
    """Load the measured advance table.

    Returns:
        The parsed artifact.

    Raises:
        TextMetricsError: The artifact is absent or holds no advances.
    """
    try:
        artifact: dict[str, Any] = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TextMetricsError(
            f"chart font metrics missing at {METRICS_PATH} — regenerate it by measuring through vl-convert "
            "and writing the result with write_font_metrics(), then commit it"
        ) from exc
    if not artifact.get("advances"):
        raise TextMetricsError(
            f"chart font metrics at {METRICS_PATH} hold no advances — the artifact is truncated or stale"
        )
    return artifact


def write_font_metrics(
    advances: dict[str, float],
    *,
    fallback_advance: float,
    worst_label: str,
    worst_ratio: float,
    font: str,
    measured_with: str,
    probe_size: int,
    weights: list[int],
    path: Path = METRICS_PATH,
) -> Path:
    """Commit a fresh measurement as a table :func:`text_width` reads.

    The one entry point for a measuring tool, which measures and checks
    while this module owns the artifact's shape and the margin a measurement has to
    clear before it may be written. Keeping the write here is what lets the tool hold
    none of those as copies.

    Args:
        advances: Character to advance width, as a fraction of the font size.
        fallback_advance: Advance for a character the table does not hold.
        worst_label: The corpus string the per-character sum underestimates most.
        worst_ratio: That string's ``drawn / estimated`` width; above 1 is an underestimate.
        font: The CSS family list a renderer emits for this table; its first family is
            the face the measurement was drawn in.
        measured_with: The rasteriser and version that drew it.
        probe_size: The font size the probes were drawn at, in px.
        weights: The font weights measured. Each advance is the widest across them, so
            one table serves every step of the type scale.
        path: Where to write it — the packaged table by default; a host measuring its own
            face writes it anywhere and reads it back with :func:`load_chart_font`.

    Returns:
        The path written.

    Raises:
        TextMetricsError: The corpus underestimates past :data:`SAFETY_MARGIN`, so a
            label would be truncated rather than moved to its own line, or the table is
            not a usable :class:`~threetears.evals.contracts.host.ChartFont`; nothing is
            written.
    """
    if worst_ratio - 1.0 > SAFETY_MARGIN:
        raise TextMetricsError(
            f"the per-character sum underestimates {worst_label!r} by {(worst_ratio - 1) * 100:.2f}%, past the "
            f"{SAFETY_MARGIN * 100:.1f}% margin the compiler applies — a label would be truncated rather than moved "
            "to its own line, which the label rules forbid. Raise SAFETY_MARGIN with the measurement that justifies it, "
            "or find what moved in the font or the rasteriser."
        )
    artifact = {
        "$comment": (
            "GENERATED by packages/evals/scripts/measure_font_metrics.py through text_metrics.write_font_metrics "
            "— do not hand-edit. Advances are a fraction of the font size, measured through vl-convert's own layout "
            "so they describe what Vega will do rather than what the font file says, in the first family of `font`. "
            "Linear in size, measured. Each advance is the widest across the measured weights, and across a "
            "character drawn between two H glyphs and in a run of itself. Kerning is not included: the "
            "per-character sum is an estimate carrying SAFETY_MARGIN, and the generator refuses to write a table "
            "whose corpus underestimates past it."
        ),
        "font": font,
        "measured_with": measured_with,
        "probe_size": probe_size,
        "weights": weights,
        "worst_corpus_underestimate": round(max(worst_ratio - 1.0, 0.0), 6),
        "worst_corpus_label": worst_label,
        "fallback_advance": fallback_advance,
        "advances": advances,
    }
    chart_font_from_artifact(artifact, where=str(path))
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if path == METRICS_PATH:
        load_metrics.cache_clear()
        packaged_font.cache_clear()
    return path


def chart_font_from_artifact(artifact: dict[str, Any], *, where: str) -> ChartFont:
    """The :class:`~threetears.evals.contracts.host.ChartFont` a metrics artifact describes.

    Args:
        artifact: A parsed artifact, in the shape :func:`write_font_metrics` writes.
        where: Where it came from, for the refusal.

    Returns:
        The font: the artifact's family list with its measured advances.

    Raises:
        TextMetricsError: The artifact names no family, or its table is not one a
            renderer can lay out against (the contract's refusal, with its reason).
    """
    family = artifact.get("font")
    if not isinstance(family, str):
        raise TextMetricsError(f"chart font metrics at {where} name no font family")
    try:
        return ChartFont(
            family=family,
            advances=dict(artifact.get("advances") or {}),
            fallback_advance=float(artifact.get("fallback_advance", 0.0)),
        )
    except StyleError as refused:
        raise TextMetricsError(f"chart font metrics at {where} are not usable: {refused}") from refused


def load_chart_font(path: Path) -> ChartFont:
    """Read a metrics artifact a host measured for its own face, as the font to declare.

    What a host puts on ``StyleProfile.chart_font`` after measuring its face with
    ``packages/evals/scripts/measure_font_metrics.py --out <path>``.

    Args:
        path: The artifact.

    Returns:
        The font.

    Raises:
        TextMetricsError: The file is absent, or its table is not usable.
    """
    try:
        artifact: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TextMetricsError(f"chart font metrics missing at {path}") from exc
    return chart_font_from_artifact(artifact, where=str(path))


@lru_cache(maxsize=1)
def packaged_font() -> ChartFont:
    """The face this renderer draws in when its host declares none, with the table measured for it.

    Returns:
        The packaged font.

    Raises:
        TextMetricsError: The packaged artifact is missing or unusable.
    """
    return chart_font_from_artifact(load_metrics(), where=str(METRICS_PATH))


def text_width(text: str, font_size: float, font: ChartFont | None = None) -> float:
    """The width ``text`` draws at, in px — the raw estimate, with no margin.

    A character the table does not hold takes the widest advance measured, which
    errs the same way the margin does. That covers two cases at once — a title is
    generator prose and may carry anything, and the table is deliberately ASCII-only
    because a glyph the font does not provide is drawn by whatever the *host* falls
    back to, which is not a property of this app and does not travel between a
    developer's machine and the container that renders in production.

    Args:
        text: The string to measure.
        font_size: The size it is drawn at, in px. Advance is linear in size —
            measured, not assumed — so one table serves every size.
        font: The face it is drawn in; ``None`` for the packaged one.

    Returns:
        The estimated advance width in px.
    """
    face = font if font is not None else packaged_font()
    advances = face.advances
    fallback = face.fallback_advance
    return sum(advances.get(character, fallback) for character in text) * font_size


def fits(text: str, font_size: float, limit: float, font: ChartFont | None = None) -> bool:
    """Whether ``text`` can be trusted to draw inside ``limit`` px.

    The conservative form of :func:`text_width`: the estimate has to clear the
    limit by :data:`SAFETY_MARGIN` before the answer is yes, because a wrong yes
    truncates and a wrong no costs a line.

    Args:
        text: The string to place.
        font_size: The size it is drawn at, in px.
        limit: The space available, in px.
        font: The face it is drawn in; ``None`` for the packaged one.

    Returns:
        Whether it fits.
    """
    return text_width(text, font_size, font) * (1 + SAFETY_MARGIN) <= limit


def wrap_text(text: str, font_size: float, limit: float, font: ChartFont | None = None) -> list[str]:
    """Break ``text`` into lines that each draw inside ``limit`` px.

    Greedy, on whitespace only. **A word wider than the limit is never split and
    never truncated** — it takes a line of its own and overruns, which is the one
    outcome the label rules permit when it cannot be avoided, and the figure card's
    horizontal scroll is what catches it. Splitting mid-token would break exactly
    the identifiers this whole rule exists to keep whole.

    Args:
        text: The string to lay out.
        font_size: The size it is drawn at, in px.
        limit: The width available, in px.
        font: The face it is drawn in; ``None`` for the packaged one.

    Returns:
        One entry per line, in order. A blank input gives no lines, so a caller
        can hand the result straight to Vega-Lite's multi-line title without
        emitting an empty one.
    """
    words = text.split()
    if not words:
        return []
    lines = [words[0]]
    for word in words[1:]:
        candidate = f"{lines[-1]} {word}"
        if fits(candidate, font_size, limit, font):
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines


__all__ = [
    "METRICS_PATH",
    "SAFETY_MARGIN",
    "TextMetricsError",
    "chart_font_from_artifact",
    "fits",
    "load_chart_font",
    "load_metrics",
    "packaged_font",
    "text_width",
    "wrap_text",
    "write_font_metrics",
]
