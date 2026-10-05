"""The renderer seam, and the Vega-Lite adapter held to it.

Pinned here, each in both directions:

- **The Vega-Lite renderer conforms.** Every chart type — and the shapes a type draws differently (a
  binned distribution, a withheld and an earned remainder, names drawn with a shared prefix stripped) —
  is drawn, read back out of the spec, and compared with the intent's values. The comparison is by value
  under an identity, so it is the drawing that is checked, not the renderer's input.
- **The check fires.** For every type, a drawing whose values were altered after the arm drew them
  disagrees; so does one that drops a row, and one that draws an identity the intent does not hold. A
  conformance run over no intents is refused, since it would pass every renderer.
- **The read-back refuses what it cannot resolve.** A drawn name that is no spelling of an identity raises
  rather than being read as a row of its own.
- **The adapter is complete and keeps its theme's promise.** An arm per stored type; the palette supplies
  the slots the intent vocabulary names; the theme a renderer is built with is the one it configures and
  rasterises in.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, get_args

import pytest

from threetears.evals.analysis.viz import (
    SERIES_SLOTS,
    VALIDATED_SLOTS,
    ChartIntent,
    ChartRenderer,
    assert_renderer_conforms,
    chart_intent,
    renderer_disagreements,
)
from threetears.evals.contracts.campaign import VizType
from threetears.evals.vega import CompiledChart, VegaRenderer, vega_config
from threetears.evals.vega.arms import ARMS
from threetears.evals.vega.palette import series_slots, validated_slots
from packages.evals.tests.chart_examples import EVERY_TYPE
from packages.evals.tests.test_vega_compiler import ATTRIBUTION_EARNED, ATTRIBUTION_WITHHELD, BINNED

#: Every intent the renderer is held to: one per type, plus the shapes a type draws by another path.
CASES: dict[str, tuple[str, dict[str, Any]]] = {
    **{viz_type: (viz_type, payload) for viz_type, payload in EVERY_TYPE.items()},
    "distribution-binned": ("distribution", BINNED),
    "attribution-withheld": ("attribution", ATTRIBUTION_WITHHELD),
    "attribution-earned": ("attribution", ATTRIBUTION_EARNED),
    # Every name shares `p/`, so the axis draws them stripped — and the stripped `p/c` is ALSO an
    # identity, spelled in full: the read-back has to resolve each drawn name through the right spelling.
    "breakdown-stripped": (
        "breakdown",
        {
            "parts": [{"label": "p/p/c", "value": 60.0, "n": 6}, {"label": "p/c", "value": 40.0, "n": 4}],
            "unit": "%",
            "measure": "share",
            "total": 100.0,
            "total_n": 10,
        },
    ),
}


def _intent(case: str) -> ChartIntent:
    viz_type, payload = CASES[case]
    return chart_intent(viz_type, payload)


def _numbers(node: Any) -> int:
    """How many numbers the inline data of a spec holds."""
    if isinstance(node, dict):
        return sum(_numbers(value) for value in node.values())
    if isinstance(node, list):
        return sum(_numbers(value) for value in node)
    return int(isinstance(node, int | float) and not isinstance(node, bool))


@dataclass(frozen=True)
class _Tampered(VegaRenderer):
    """The real renderer, its spec's inline data edited after the arm drew it."""

    edit: Any = None

    def draw(self, intent: ChartIntent) -> CompiledChart:
        chart = super().draw(intent)
        spec = copy.deepcopy(chart.spec)
        _edit_data(spec, self.edit)
        return CompiledChart(intent=chart.intent, spec=spec)


def _edit_data(node: Any, edit: Any) -> None:
    if isinstance(node, dict):
        source = node.get("data")
        if isinstance(source, dict) and isinstance(source.get("values"), list):
            source["values"] = edit(source["values"])
        for value in node.values():
            _edit_data(value, edit)
    elif isinstance(node, list):
        for value in node:
            _edit_data(value, edit)


def _shifted(values: list[Any]) -> list[Any]:
    """Every number moved, as a renderer that rescaled its data after the intent was decided would."""
    return [
        {
            key: value + 1 if isinstance(value, int | float) and not isinstance(value, bool) else value
            for key, value in datum.items()
        }
        if isinstance(datum, dict)
        else datum
        for datum in values
    ]


# =============================================================================
# The Vega-Lite renderer conforms
# =============================================================================


def test_the_vega_renderer_is_a_chart_renderer() -> None:
    renderer: ChartRenderer[CompiledChart] = VegaRenderer()
    assert renderer.drawn_data(renderer.draw(_intent("breakdown")))


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_vega_renderer_draws_what_the_intent_holds(case: str) -> None:
    assert renderer_disagreements(VegaRenderer(), _intent(case)) == []


def test_the_vega_renderer_passes_the_conformance_run() -> None:
    assert_renderer_conforms(VegaRenderer(theme="light"), [_intent(case) for case in sorted(CASES)])


def test_the_cases_cover_every_type() -> None:
    assert {viz_type for viz_type, _ in CASES.values()} == set(get_args(VizType))


def test_a_stripped_name_is_read_back_as_the_identity_it_stands_for() -> None:
    renderer = VegaRenderer()
    drawn = renderer.drawn_data(renderer.draw(_intent("breakdown-stripped")))
    by_name = {datum["display"]: datum["label"] for datum in drawn if "display" in datum}
    assert by_name == {"p/c": "p/p/c", "c": "p/c"}


# =============================================================================
# The check fires
# =============================================================================


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_drawing_whose_values_moved_disagrees(case: str) -> None:
    """Every case: the comparison reaches its values, so moving them is seen."""
    assert _numbers(VegaRenderer().draw(_intent(case)).spec) > 0
    disagreements = renderer_disagreements(_Tampered(edit=_shifted), _intent(case))
    assert any("without the intent's value" in line for line in disagreements), disagreements


def test_a_drawing_that_drops_a_row_disagrees() -> None:
    def drop_a_row(values: list[Any]) -> list[Any]:
        # Every layer's datum for the row: the bar (named by its label) and the name and value text
        # beside it (named by the drawn name alone).
        return [
            datum
            for datum in values
            if not (isinstance(datum, dict) and "budget_exhausted" in (datum.get("label"), datum.get("display")))
        ]

    disagreements = renderer_disagreements(_Tampered(edit=drop_a_row), _intent("breakdown"))
    assert "did not draw 'budget_exhausted', which the values table holds" in disagreements


def test_a_drawing_of_an_identity_the_intent_does_not_hold_disagrees() -> None:
    def add_a_row(values: list[Any]) -> list[Any]:
        return [*values, {"label": "invented", "value": 1.0}]

    disagreements = renderer_disagreements(_Tampered(edit=add_a_row), _intent("breakdown"))
    assert "drew 'invented', which the intent does not hold" in disagreements


def test_the_conformance_run_raises_on_a_disagreement() -> None:
    with pytest.raises(AssertionError, match="breakdown"):
        assert_renderer_conforms(_Tampered(edit=_shifted), [_intent("breakdown")])


def test_a_conformance_run_over_no_intents_is_refused() -> None:
    with pytest.raises(ValueError, match="no intents"):
        assert_renderer_conforms(VegaRenderer(), [])


# =============================================================================
# The read-back refuses what it cannot resolve
# =============================================================================


def test_a_drawn_name_that_is_no_spelling_of_an_identity_is_refused() -> None:
    def rename(values: list[Any]) -> list[Any]:
        return [
            {key: value for key, value in datum.items() if key != "label"} | {"display": "nobody"}
            if isinstance(datum, dict) and "display" in datum
            else datum
            for datum in values
        ]

    renderer = _Tampered(edit=rename)
    with pytest.raises(ValueError, match="not one spelling of the intent's identities"):
        renderer.drawn_data(renderer.draw(_intent("delta_table")))


# =============================================================================
# The adapter is complete and keeps its theme's promise
# =============================================================================


def test_every_stored_type_has_an_arm() -> None:
    assert set(ARMS) == set(get_args(VizType))


def test_the_vega_themes_palette_supplies_the_slots_the_vocabulary_names() -> None:
    """The vocabulary decides what a slot promises; the Vega theme's artifact has to keep it."""
    assert validated_slots() == VALIDATED_SLOTS
    assert series_slots() == SERIES_SLOTS


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_a_renderer_configures_and_draws_in_the_theme_it_was_built_with(theme: Any) -> None:
    renderer = VegaRenderer(theme=theme)
    chart = renderer.draw(_intent("breakdown"))

    assert renderer.config() == vega_config(theme)
    assert vega_config(theme)["background"] in renderer.svg(chart)
