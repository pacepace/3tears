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
    check_intent,
    renderer_disagreements,
    table_disagreements,
)
from threetears.evals.contracts.campaign import VizType
from threetears.evals.contracts.host import StyleProfile
from threetears.evals.vega import CompiledChart, VegaRenderer, packaged_palette, vega_config
from threetears.evals.vega.arms import ARMS
from threetears.evals.vega.palette import series_slots, validated_slots
from packages.evals.tests.chart_examples import EVERY_TYPE
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_PALETTE, toyhost_profile
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
    renderer: ChartRenderer[CompiledChart] = VegaRenderer.packaged()
    assert renderer.drawn_data(renderer.draw(_intent("breakdown")))


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_vega_renderer_draws_what_the_intent_holds(case: str) -> None:
    assert renderer_disagreements(VegaRenderer.packaged(), _intent(case)) == []


def test_the_vega_renderer_passes_the_conformance_run() -> None:
    assert_renderer_conforms(VegaRenderer.packaged("light"), [_intent(case) for case in sorted(CASES)])


def test_the_cases_cover_every_type() -> None:
    assert {viz_type for viz_type, _ in CASES.values()} == set(get_args(VizType))


def test_a_stripped_name_is_read_back_as_the_identity_it_stands_for() -> None:
    renderer = VegaRenderer.packaged()
    drawn = renderer.drawn_data(renderer.draw(_intent("breakdown-stripped")))
    by_name = {datum["display"]: datum["label"] for datum in drawn if "display" in datum}
    assert by_name == {"p/c": "p/p/c", "c": "p/c"}


# =============================================================================
# The check fires
# =============================================================================


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_drawing_whose_values_moved_disagrees(case: str) -> None:
    """Every case: the comparison reaches its values, so moving them is seen."""
    assert _numbers(VegaRenderer.packaged().draw(_intent(case)).spec) > 0
    disagreements = renderer_disagreements(_Tampered(palette=packaged_palette("dark"), edit=_shifted), _intent(case))
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

    disagreements = renderer_disagreements(
        _Tampered(palette=packaged_palette("dark"), edit=drop_a_row), _intent("breakdown")
    )
    assert "did not draw 'budget_exhausted', which the values table holds" in disagreements


def test_a_drawing_of_an_identity_the_intent_does_not_hold_disagrees() -> None:
    def add_a_row(values: list[Any]) -> list[Any]:
        return [*values, {"label": "invented", "value": 1.0}]

    disagreements = renderer_disagreements(
        _Tampered(palette=packaged_palette("dark"), edit=add_a_row), _intent("breakdown")
    )
    assert "drew 'invented', which the intent does not hold" in disagreements


def _table_spelled(intent: ChartIntent, value: Any) -> ChartIntent:
    """``intent`` with every non-identity cell of its values table replaced by ``value`` — the drawing untouched."""
    field = intent.identity.field if intent.identity is not None else None
    rows = [{key: (cell if key == field else value) for key, cell in row.items()} for row in intent.rows]
    return intent.model_copy(update={"rows": rows})


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_values_table_that_disagrees_with_the_marks_is_refused_by_the_policy_and_the_conformance_run(
    case: str,
) -> None:
    """A builder that spelled a row from the wrong value: the drawing is faithful, the table beside it is not."""
    intent = _intent(case)
    assert check_intent(intent) == [] and renderer_disagreements(VegaRenderer.packaged("dark"), intent) == []

    tampered = _table_spelled(intent, "999999")

    assert any("values table" in violation for violation in check_intent(tampered))
    assert any("values table" in d for d in renderer_disagreements(VegaRenderer.packaged("dark"), tampered))


def test_one_number_moved_in_one_row_is_named() -> None:
    intent = _intent("timeseries")
    rows = copy.deepcopy(intent.rows)
    rows[1]["mean"] = rows[1]["mean"] + 1.0
    tampered = intent.model_copy(update={"rows": rows})

    violations = table_disagreements(tampered)
    assert any(f"row for {rows[1]['series']!r} states mean={rows[1]['mean']!r}" in v for v in violations), violations
    assert any(f"the chart draws {rows[1]['series']!r} at mean=" in v for v in violations), "the mark it left unstated"


def test_a_sweep_whose_drawn_configuration_the_table_misstates_is_refused() -> None:
    """A sweep's rows show the levels, not the configuration's name: the drawn mark finds no row stating it."""
    intent = _intent("sweep_ranking")
    rows = copy.deepcopy(intent.rows)
    rows[0]["ranked"] = rows[0]["ranked"] + 0.5
    (violation,) = table_disagreements(intent.model_copy(update={"rows": rows}))
    assert violation.startswith(f"the chart draws {intent.data[0]['config']!r} at ranked=")


def test_a_withheld_row_with_no_mark_is_not_a_disagreement() -> None:
    """A row the table states and the chart deliberately does not draw contradicts no mark."""
    intent = _intent("attribution-withheld")
    field = intent.identity.field if intent.identity is not None else ""
    drawn = {datum.get(field) for datum in intent.data}
    assert any(row.get(field) not in drawn for row in intent.rows), "the fixture withholds a row"
    assert table_disagreements(intent) == []


def test_the_conformance_run_raises_on_a_disagreement() -> None:
    with pytest.raises(AssertionError, match="breakdown"):
        assert_renderer_conforms(_Tampered(palette=packaged_palette("dark"), edit=_shifted), [_intent("breakdown")])


def test_a_conformance_run_over_no_intents_is_refused() -> None:
    with pytest.raises(ValueError, match="no intents"):
        assert_renderer_conforms(VegaRenderer.packaged(), [])


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

    renderer = _Tampered(palette=packaged_palette("dark"), edit=rename)
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
    renderer = VegaRenderer.packaged(theme)
    chart = renderer.draw(_intent("breakdown"))

    assert renderer.config() == vega_config(packaged_palette(theme))
    assert vega_config(packaged_palette(theme))["background"] in renderer.svg(chart)


# =============================================================================
# The host's palette reaches the renderer
# =============================================================================


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_a_host_that_declares_a_palette_is_drawn_in_it_whatever_the_theme(theme: Any) -> None:
    """A declared palette is never replaced: ``theme`` names the packaged variant and is not read here."""
    renderer = VegaRenderer.for_style(toyhost_profile().style, theme=theme)
    config = renderer.config()

    assert renderer.palette == TOYHOST_PALETTE
    assert config["background"] == TOYHOST_PALETTE.background
    assert config["range"]["category"] == list(TOYHOST_PALETTE.series)
    assert config["range"]["chart-seq"] == list(TOYHOST_PALETTE.sequential)
    assert config["bar"]["color"] == TOYHOST_PALETTE.series[0]
    assert config["axis"]["labelColor"] == TOYHOST_PALETTE.ink
    assert config["style"]["chart-context"]["color"] == TOYHOST_PALETTE.context
    assert config["style"]["chart-value-on-fill"]["color"] == TOYHOST_PALETTE.on_fill
    assert TOYHOST_PALETTE.background in renderer.svg(renderer.draw(_intent("breakdown")))


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_a_host_that_declares_no_palette_is_drawn_in_the_packaged_one(theme: Any) -> None:
    """The stated default: no declared palette draws in the packaged variant ``theme`` names."""
    renderer = VegaRenderer.for_style(StyleProfile(), theme=theme)

    assert renderer.palette == packaged_palette(theme)
    assert renderer.config() == vega_config(packaged_palette(theme))


def test_a_host_palette_and_the_packaged_one_build_the_same_config_keys() -> None:
    """One builder for both palettes: a host theme changes colours, never the config's shape."""

    def keys(node: Any, at: str = "") -> set[str]:
        if not isinstance(node, dict):
            return set()
        return {f"{at}.{key}" for key in node} | {
            path for key, child in node.items() for path in keys(child, f"{at}.{key}")
        }

    assert keys(vega_config(TOYHOST_PALETTE)) == keys(vega_config(packaged_palette("dark")))
