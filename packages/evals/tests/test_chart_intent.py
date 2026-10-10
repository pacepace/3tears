"""Chart intent: eval's own chart vocabulary, the rules it is held to, and the seam a renderer sits behind.

Three things are pinned, each in both directions:

- **The vocabulary is closed and complete.** Every stored chart type has a payload model and an intent
  builder, keyed alike; every type's intent round-trips through its JSON. (That the Vega-Lite renderer has
  an arm for each is the renderer's, in ``test_renderer_conformance.py``.)
- **The policy reads the intent.** Each presentation rule refuses the shape it names (one test per
  refusal, each against an intent that otherwise passes), and :func:`chart_intent` enforces them, so no
  intent that breaks one leaves the entry point.
- **The core never reaches the renderer.** No core module imports ``threetears.evals.vega``, the Vega-Lite
  renderer adapter, at all.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, get_args

import pytest

from threetears.evals.analysis.viz import (
    SERIES_SLOTS,
    VALIDATED_SLOTS,
    ChartAxis,
    ChartColours,
    ChartColumn,
    ChartEncoding,
    ChartIdentity,
    ChartIntent,
    ChartReference,
    IntentPolicyError,
    PayloadError,
    chart_intent,
    check_intent,
)
from threetears.evals.analysis.viz.intents import INTENTS
from threetears.evals.analysis.viz.payloads import PAYLOAD_MODELS
from threetears.evals.kernel.campaign import VizType
from packages.evals.tests.import_resolution import absolute_module
from packages.evals.tests.package_placement import eval_modules
from packages.evals.tests.chart_examples import EVERY_TYPE


def _intent(**update: Any) -> ChartIntent:
    """A breakdown intent that passes every rule, with ``update`` applied over it."""
    return chart_intent("breakdown", EVERY_TYPE["breakdown"]).model_copy(update=update)


# =============================================================================
# The vocabulary
# =============================================================================


class TestTheVocabularyIsClosedAndComplete:
    def test_every_stored_type_has_a_payload_an_intent_and_an_arm(self) -> None:
        stored = set(get_args(VizType))
        assert set(PAYLOAD_MODELS) == stored
        assert set(INTENTS) == stored

    def test_the_shared_fixtures_cover_every_type(self) -> None:
        assert set(EVERY_TYPE) == set(get_args(VizType))

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_every_types_intent_passes_the_rules_and_round_trips_through_its_json(self, viz_type: str) -> None:
        intent = chart_intent(viz_type, EVERY_TYPE[viz_type])

        assert check_intent(intent) == []
        assert ChartIntent.model_validate_json(intent.model_dump_json()) == intent
        assert intent.parsed() == PAYLOAD_MODELS[viz_type].model_validate(EVERY_TYPE[viz_type])

    @pytest.mark.parametrize("viz_type", sorted(EVERY_TYPE), ids=sorted(EVERY_TYPE))
    def test_every_intent_names_what_its_rows_are_and_how_it_measures(self, viz_type: str) -> None:
        """Every type declares its identity and at least one measured field on a declared axis."""
        intent = chart_intent(viz_type, EVERY_TYPE[viz_type])

        assert intent.identity is not None and intent.identity.order
        assert any(encoding.axis and intent.axis(encoding.axis) for encoding in intent.encodings)

    def test_a_type_with_no_builder_is_refused_as_a_payload_problem(self) -> None:
        with pytest.raises(PayloadError, match="no chart intent for viz type 'scatter'"):
            chart_intent("scatter", {})


# =============================================================================
# The policy, rule by rule
# =============================================================================


def _breakdown_value_axis(**update: Any) -> list[ChartAxis]:
    (axis,) = _intent().axes
    return [axis.model_copy(update=update)]


@pytest.mark.parametrize(
    ("update", "refusal"),
    [
        pytest.param(
            {"axes": _breakdown_value_axis(zero_baseline=False)},
            "value is a length drawn against axis 'value', which does not start at zero",
            id="rule1-length-off-zero",
        ),
        pytest.param(
            {"axes": []},
            "value is a length measured against no declared axis ('value')",
            id="rule2-measured-on-no-axis",
        ),
        pytest.param(
            {"axes": _breakdown_value_axis(unit="ms", quantity="Latency")},
            "axis 'value' is drawn in ms but tells the reader 'Latency', which does not name it",
            id="rule2-unit-unstated",
        ),
        pytest.param(
            {"references": [ChartReference(axis="quality", value=0.5, label="quality bar")]},
            "the quality bar is drawn across no declared axis ('quality')",
            id="rule2-reference-on-no-axis",
        ),
        pytest.param(
            {"colours": [ChartColours(field="value", scheme="categorical", domain=[])]},
            "the categorical scheme on value has no domain",
            id="rule3-no-domain",
        ),
        pytest.param(
            {"colours": [ChartColours(field="n", scheme="categorical", domain=["9", "9"])]},
            "lists 9 twice — one value, two slots",
            id="rule3-repeated-slot",
        ),
    ],
)
def test_a_rule_refuses_the_shape_it_names(update: dict[str, Any], refusal: str) -> None:
    assert refusal in " | ".join(check_intent(_intent(**update)))


def test_rule3_a_coloured_value_with_no_slot_is_refused() -> None:
    drawn = sorted({str(row["n"]) for row in _intent().data})
    violations = check_intent(_intent(colours=[ChartColours(field="n", scheme="categorical", domain=drawn[:1])]))

    assert f"n draws {', '.join(drawn[1:])}, which the categorical scheme gives no slot" in violations


def test_rule3_a_value_named_uncoloured_needs_no_slot() -> None:
    drawn = sorted({str(row["n"]) for row in _intent().data})
    colours = ChartColours(field="n", scheme="categorical", domain=drawn[:1], uncoloured=drawn[1:])

    assert check_intent(_intent(colours=[colours], direct_labels=True)) == []


def test_rule3_a_domain_past_the_palette_warns_and_is_not_refused(caplog: pytest.LogCaptureFixture) -> None:
    """The palette never refuses to draw: past its width it recycles, and says so."""
    wide = [str(index) for index in range(SERIES_SLOTS + 1)]
    data = [{"label": "a", "value": 1.0, "level": level} for level in wide]
    intent = _intent(
        data=data,
        rows=[],
        colours=[ChartColours(field="level", scheme="categorical", domain=wide)],
        direct_labels=True,
    )

    with caplog.at_level(logging.WARNING):
        violations = check_intent(intent)

    assert not [line for line in violations if "slot" in line], violations
    assert f"past the {SERIES_SLOTS} slots a theme supplies" in caplog.text


def test_rule5_an_interval_end_that_does_not_say_what_it_spans_is_refused() -> None:
    intent = chart_intent("null_result", EVERY_TYPE["null_result"])
    unnamed = [
        encoding.model_copy(update={"varies_over": ""}) if encoding.role == "interval_low" else encoding
        for encoding in intent.encodings
    ]

    assert "low is an interval end that does not say what the interval varies over" in check_intent(
        intent.model_copy(update={"encodings": unnamed})
    )


def test_rule5_every_interval_end_the_builders_write_names_its_span() -> None:
    for viz_type, payload in EVERY_TYPE.items():
        intent = chart_intent(viz_type, payload)
        for encoding in intent.encodings:
            if encoding.role in ("interval_low", "interval_high"):
                assert encoding.varies_over, (viz_type, encoding.field)


def test_rule7_the_field_that_names_the_rows_is_never_coloured() -> None:
    labels = [str(row["label"]) for row in _intent().data]
    colours = ChartColours(field="label", scheme="categorical", domain=labels)

    assert any(
        line.startswith("label names the rows and is coloured") for line in check_intent(_intent(colours=[colours]))
    )


def test_rule7_past_the_validated_slots_every_mark_needs_its_label() -> None:
    levels = [str(index) for index in range(VALIDATED_SLOTS + 1)]
    data = [{"label": "a", "value": 1.0, "level": level} for level in levels]
    colours = [ChartColours(field="level", scheme="categorical", domain=levels)]
    refusal = f"level takes {len(levels)} categorical slots, past the {VALIDATED_SLOTS} validated ones"

    assert any(refusal in line for line in check_intent(_intent(data=data, colours=colours, direct_labels=False)))
    assert not any(refusal in line for line in check_intent(_intent(data=data, colours=colours, direct_labels=True)))


def test_rule10_a_ranking_by_a_field_the_chart_does_not_measure_is_refused() -> None:
    identity = ChartIdentity(field="label", order=["a"], ordered_by="n", ranked_by="n")

    assert "the rows are ranked by 'n', which the chart does not draw as a measure" in check_intent(
        _intent(identity=identity)
    )


def test_rule10_a_ranking_whose_rows_do_not_descend_is_refused() -> None:
    intent = chart_intent("sweep_ranking", EVERY_TYPE["sweep_ranking"])
    assert check_intent(intent) == [], "the fixture is a ranking the rule accepts"

    reversed_data = list(reversed(intent.data))

    assert "the rows claim a ranking by 'ranked' and do not descend on it" in check_intent(
        intent.model_copy(update={"data": reversed_data})
    )


def test_rule11_an_ordinal_position_on_an_axis_that_states_no_order_is_refused() -> None:
    intent = chart_intent("timeseries", EVERY_TYPE["timeseries"])
    unordered = [axis.model_copy(update={"order": []}) if axis.name == "time" else axis for axis in intent.axes]

    assert any(
        line.startswith("position is an ordinal position on an axis that states no order")
        for line in check_intent(intent.model_copy(update={"axes": unordered}))
    )


def test_rule11_an_ordinal_position_its_axis_does_not_hold_is_refused() -> None:
    intent = chart_intent("timeseries", EVERY_TYPE["timeseries"])
    order = next(axis.order for axis in intent.axes if axis.name == "time")
    short = [axis.model_copy(update={"order": order[:-1]}) if axis.name == "time" else axis for axis in intent.axes]

    assert f"position draws {order[-1]}, which its axis's stated order does not hold" in check_intent(
        intent.model_copy(update={"axes": short})
    )


def test_rule12_a_chart_with_no_values_table_is_refused() -> None:
    assert any(line.startswith("the chart has no values table") for line in check_intent(_intent(columns=[])))


def test_rule12_a_row_stating_a_value_no_column_shows_is_refused() -> None:
    rows = [dict(row) | {"hidden": 3} for row in _intent().rows]

    assert "the values table's rows state hidden, which no column shows" in check_intent(_intent(rows=rows))


def test_rule12_a_key_absent_in_every_row_needs_no_column() -> None:
    rows = [dict(row) | {"hidden": None} for row in _intent().rows]

    assert check_intent(_intent(rows=rows)) == []


def test_rule12_a_chart_placing_marks_with_an_empty_table_is_refused() -> None:
    assert "the chart places marks and its values table has no rows" in check_intent(_intent(rows=[]))


def test_the_entry_point_enforces_the_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    """A builder that produced a rule-breaking intent would be caught where every surface reads one."""
    build = INTENTS["breakdown"]

    def cropped(payload: Any) -> ChartIntent:
        intent = build(payload)
        return intent.model_copy(
            update={"axes": [axis.model_copy(update={"zero_baseline": False}) for axis in intent.axes]}
        )

    monkeypatch.setitem(INTENTS, "breakdown", cropped)

    with pytest.raises(IntentPolicyError, match="does not start at zero"):
        chart_intent("breakdown", EVERY_TYPE["breakdown"])


def test_an_intent_builder_never_writes_the_caption_the_entry_point_carries_it() -> None:
    caption = "the budget path dominates"
    assert chart_intent("breakdown", {**EVERY_TYPE["breakdown"], "caption": f"  {caption}  "}).caption == caption


class TestTheIntentModel:
    def test_an_encoding_role_outside_the_vocabulary_is_refused(self) -> None:
        with pytest.raises(ValueError, match="role"):
            ChartEncoding(field="x", role="hue")  # type: ignore[arg-type]

    def test_a_chart_type_outside_the_vocabulary_is_refused(self) -> None:
        with pytest.raises(ValueError, match="type"):
            _intent().model_validate({**_intent().model_dump(), "type": "scatter"})

    def test_values_as_drawn_render_the_table(self) -> None:
        intent = _intent(columns=[ChartColumn(key="label", header="Part")], rows=[{"label": "a"}, {"label": None}])
        assert intent.values_as_drawn() == ["Part", "a", "—"]


# =============================================================================
# The core never reaches the renderer
# =============================================================================

#: The renderer adapter, relative to ``threetears.evals``. Nothing outside it may import it; the
#: package matrix's ``vega`` row holds the same edge (and fires on a planted one), and this walk says it
#: in the chart's own terms.
RENDERER: str = "vega"

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


def _in_the_renderer(relative: str) -> bool:
    return relative == RENDERER or relative.startswith(RENDERER + ".")


def _renderer_edges(root: Path) -> dict[str, set[str]]:
    """Every module under ``root`` outside the renderer that imports it, and what it imports."""
    edges: dict[str, set[str]] = {}
    for relative, path in eval_modules(root).items():
        if _in_the_renderer(relative):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = absolute_module(path, node, root=root)
                targets = [f"{base}.{alias.name}" for alias in node.names] + [base]
            elif isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            else:
                continue
            for target in targets:
                if target.startswith("threetears.evals.") and _in_the_renderer(
                    target.removeprefix("threetears.evals.")
                ):
                    edges.setdefault(relative, set()).add(target)
    return edges


def test_no_core_module_imports_the_renderer() -> None:
    assert _renderer_edges(_SOURCE_ROOT) == {}


def test_the_walk_sees_an_edge_where_one_exists(tmp_path: Path) -> None:
    """A walk that saw nothing would be broken, not clean — so plant the edge the service used to hold."""
    service = tmp_path / "threetears" / "evals" / "analysis" / "service.py"
    service.parent.mkdir(parents=True)
    service.write_text("from threetears.evals.vega.compiler import compile_chart\n", encoding="utf-8")
    (tmp_path / "threetears" / "evals" / "vega").mkdir()
    (tmp_path / "threetears" / "evals" / "vega" / "compiler.py").write_text("", encoding="utf-8")

    assert _renderer_edges(tmp_path) == {
        "analysis.service": {"threetears.evals.vega.compiler.compile_chart", "threetears.evals.vega.compiler"}
    }
