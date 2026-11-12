"""Every figure an analyst tool returns comes from its figure key.

A tool never builds a number itself: it builds the key that names the figure (``reading.qwen3.8-flash.change``) and
:func:`cite` resolves it, so every figure a tool hands back is, by construction, one
:func:`~threetears.evals.analysis.report.figure_keys.resolve` accepts, with the value it reads. The analyst's prose
cites the same key, and the renderer prints the same number.

A :class:`CitedFigure` carries the key, the value, what kind of value it is and its scale, and ``reads``: the figure as
the page writes it (``{key}`` as :func:`~threetears.evals.analysis.report.figure_keys.render` writes it), so the text a
tool returns reads as the page will: an analyst tool reads off any page, so it spells each figure in the forms its
input's page would print it in (:meth:`~threetears.evals.analysis.report.forms.PageForms.of`). It is spelled from the
figure the key resolved to, never by parsing ``{key}`` back
out of text, so a key whose ids carry a brace (a floor's key, a cell's row) reads as surely as any other. What is not a figure is not cited: an arm is named ``arm:<id>``, a caveat by its
id, an example by its trial id, and reasons, excerpts and notes are text.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis.errors import UnstatedFigure
from threetears.evals.analysis.report.figure_keys import (
    Figure,
    FigureKind,
    Scale,
    format_figure,
    page_arm_names,
    resolve,
    written_arm,
)
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.words import arm_tags
from threetears.evals.analysis.report_input import Contrast, ReportInput


class CitedFigure(BaseModel):
    """One figure, as its key names it and as the page writes it.

    :ivar key: The figure key's path, without braces (``reading.fast.v2.change``).
    :ivar reads: The figure as the page writes it: the key rendered with no filter.
    :ivar value: The value the key resolves to: a number (a count a whole one), an interval as its two ends, or text
        (an arm's id for an arm figure).
    :ivar kind: What the value is: a number, an interval, text, or an arm.
    :ivar scale: The scale a number is on (a share, a percent, points, dollars, a count, or plain).
    :ivar unit: The unit a plain number is in, when it has one.
    :ivar derived: Whether code derived the figure from the input's figures (a flip margin, a share of a loss, a
        change against the control in a slice), which is descriptive: no interval and no test.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1, description="The figure key's path, without braces.")
    reads: str = Field(description="The figure as the page writes it: the key rendered with no filter.")
    value: int | float | tuple[float, float] | str = Field(
        description="The value the key resolves to: a count as a whole number."
    )
    kind: FigureKind = Field(description="What the value is: number, interval, text or arm.")
    scale: Scale = Field(description="The scale a number is on.")
    unit: str = Field(description="The unit a plain number is in; empty when it has none.")
    derived: bool = Field(description="Whether code derived the figure from the input's figures; descriptive only.")

    def cited_as(self) -> str:
        """The key as a report cites it, as a result's text writes it before `` = ``: the path.

        :return: The path.
        :rtype: str
        """
        return self.key


class WrittenFigure(CitedFigure):
    """One figure as a key with filters writes it: the figure, its filters, and ``reads`` written through them.

    :ivar filters: The key's filters, in the order the key names them; empty for none.
    """

    filters: list[str] = Field(description="The key's filters, as the key names them; empty for none.")

    def cited_as(self) -> str:
        """The key as a report cites it: the path and its filters, ``path|filter``.

        :return: The path, then each filter after a bar.
        :rtype: str
        """
        return "|".join([self.key, *self.filters])


def _whole(figure: Figure) -> int | float | tuple[float, float] | str:
    """A figure's value, a count as the whole number it is (``170``, never ``170.0``)."""
    value = figure.value
    if figure.scale == "count" and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _arm_written(report_input: ReportInput, arm_id: str) -> str:
    """An arm as the page writes ``{arm:<id>}`` with no computed short names: its name on the page
    (:func:`~threetears.evals.analysis.report.figure_keys.page_arm_names`), then its letter tag, once."""
    arms = report_input.arms
    control = next((held.id for held in arms if held.role == "control"), None)
    tag = arm_tags([held.id for held in arms], control)[arm_id]
    return written_arm(page_arm_names(report_input)[arm_id], tag)


def _forms_of(report_input: ReportInput, forms: PageForms | None) -> PageForms:
    """The forms a cited figure is written in: the caller's, checked to be ``report_input``'s, or, for a tool that
    reads the input off any page and holds none, forms of the figure's own, built from its input as its page builds
    them."""
    return forms.checked(report_input) if forms is not None else PageForms.of(report_input)


def cite(key: str, report_input: ReportInput, *, forms: PageForms | None = None) -> CitedFigure:
    """The figure a key names on the report input, with the text the page writes it as.

    :param key: The path, without braces or filters (``reading.fast.v2.change``, ``arm:base``).
    :ptype key: str
    :param report_input: The frozen report input the tool reads.
    :ptype report_input: ReportInput
    :param forms: The forms the caller prints in, built from ``report_input`` once for all it prints; None for the
        input's own, built for this figure alone.
    :ptype forms: PageForms | None
    :return: The figure.
    :rtype: CitedFigure
    :raises UnresolvableReference: The key names nothing the input holds, reads two ways, or names a figure the input
        leaves unstated.
    :raises ValueError: ``forms`` were built from another input.
    """
    figure = resolve(key, report_input)
    return CitedFigure(
        key=key,
        reads=(
            _arm_written(report_input, str(figure.value))
            if figure.kind == "arm"
            else format_figure(figure, forms=_forms_of(report_input, forms))
        ),
        value=_whole(figure),
        kind=figure.kind,
        scale=figure.scale,
        unit=figure.unit,
        derived=figure.derived,
    )


def cite_if(key: str, report_input: ReportInput, *, forms: PageForms | None = None) -> CitedFigure | None:
    """The figure a key names, or ``None`` when the input leaves that figure unstated.

    For a tool listing what an input holds, where a figure one arm or aspect lacks (an interval the pipeline did not
    compute) is absent rather than an error: the key resolver raises :class:`UnstatedFigure` for it, and this reads
    that type alone as absent. Every other refusal is the tool's own key being wrong, and is raised.

    :param key: The path, without braces or filters.
    :ptype key: str
    :param report_input: The frozen report input the tool reads.
    :ptype report_input: ReportInput
    :param forms: The forms the caller prints in, built from ``report_input`` once for all it prints (:func:`cite`);
        None for the input's own, built for this figure alone.
    :ptype forms: PageForms | None
    :return: The figure, or ``None`` when the input does not state it.
    :rtype: CitedFigure | None
    :raises UnresolvableReference: The key names nothing the input holds, or reads two ways.
    """
    try:
        return cite(key, report_input, forms=forms)
    except UnstatedFigure:
        # NOSILENT: an unstated figure is an expected condition, the input's shape and not a fault: a listing shows the
        # figure as absent, which is what None says to its caller. Every other refusal is raised.
        return None


def cite_written(
    key: str, filters: tuple[str, ...], report_input: ReportInput, *, forms: PageForms | None = None
) -> WrittenFigure:
    """The figure a key names, written through its filters as the page writes ``{key|filter}``.

    :param key: The path, without braces or filters; never an arm, which takes no filter.
    :ptype key: str
    :param filters: The key's filters, as :func:`~threetears.evals.analysis.report.figure_keys.parse_key` checked them.
    :ptype filters: tuple[str, ...]
    :param report_input: The frozen report input the tool reads.
    :ptype report_input: ReportInput
    :param forms: The forms the caller prints in, built from ``report_input`` once for all it prints; None for the
        input's own, built for this figure alone.
    :ptype forms: PageForms | None
    :return: The figure, ``reads`` written through the filters.
    :rtype: WrittenFigure
    :raises UnresolvableReference: The key does not resolve, or a filter does not suit its figure.
    :raises ValueError: ``forms`` were built from another input.
    """
    figure = resolve(key, report_input)
    reads = (
        _arm_written(report_input, str(figure.value))
        if figure.kind == "arm"
        else format_figure(figure, filters, forms=_forms_of(report_input, forms))
    )
    return WrittenFigure(
        key=key,
        reads=reads,
        value=_whole(figure),
        kind=figure.kind,
        scale=figure.scale,
        unit=figure.unit,
        derived=figure.derived,
        filters=list(filters),
    )


def contrast_key(report_input: ReportInput, contrast: Contrast) -> str:
    """The key a contrast's figures are read under: unqualified, or by its aspect where the input holds its arm and
    measure more than once. ``contrasts_list`` cites a contrast's figures under it, ``arms_compare`` its cases, and the
    editor's ``caveat_not_carried`` fact (``analysis/report_doc_facts.py``) names them alike.

    :param report_input: The snapshot's frozen report input.
    :ptype report_input: ReportInput
    :param contrast: One of its contrasts.
    :ptype contrast: Contrast
    :return: The key's path, before its statistic (``contrast.<arm>.<measure>``).
    :rtype: str
    """
    held = [c for c in report_input.contrasts if (c.arm, c.measure) == (contrast.arm, contrast.measure)]
    if len(held) > 1 and contrast.aspect is not None:
        return f"aspects.{contrast.aspect}.contrast.{contrast.arm}.{contrast.measure}"
    return f"contrast.{contrast.arm}.{contrast.measure}"


__all__ = ["CitedFigure", "WrittenFigure", "cite", "cite_if", "cite_written", "contrast_key"]
