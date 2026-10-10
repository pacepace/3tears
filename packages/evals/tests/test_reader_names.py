"""Every rendered surface names a measure, a question and an arm level in a reader's words (#627, #581).

A measure's key (``candidate_output_tokens_per_s``) is what the store files a number under; a reader needs
"Output speed". A declared question's id is a uuid; a reader needs the question. A prompt-sweep level's host
display is a fingerprint; a reader needs the name the campaign gave it. Each test here fails when its fix is
reverted: the key, the id or the host's display comes back.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from threetears.evals.analysis import (
    ChartBlock,
    TableBlock,
    campaign_report,
)
from threetears.evals.analysis.references import ReadingRef
from threetears.evals.analysis.surface_table import build_surface_table
from threetears.evals.analysis.viz_refs import FrontierRef, build_viz_payload
from threetears.evals.contracts.campaign import EvalAnalysis
from threetears.evals.contracts.host import EvalHost
from threetears.evals.contracts.host.measures import MeasureRegistrationError, MeasureRegistry
from threetears.evals.contracts.metrics import METRIC_DESCRIPTORS, MetricDescriptor, goal_check_of
from threetears.evals.vega.compiler import compile_chart, draw_intent
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES
from packages.evals.tests.report_support import toy_report


@pytest.fixture
async def toy() -> tuple[EvalHost, EvalAnalysis, Any]:
    return await toy_report()


# --- #627: a measure has a reader-facing name, and every surface heads it by that -------------------------------


def _spec_titles(node: Any) -> list[str]:
    """Every title a compiled Vega-Lite spec carries, at any depth."""
    if isinstance(node, dict):
        found = [str(value) for key, value in node.items() if key == "title" and isinstance(value, str)]
        return found + [title for value in node.values() for title in _spec_titles(value)]
    if isinstance(node, list):
        return [title for value in node for title in _spec_titles(value)]
    return []


def _chart_texts(intent: Any) -> list[str]:
    """What a chart heads anything by: its title, its axes, its value table's identities, and the compiled spec."""
    compiled = draw_intent(intent)
    identities = [str(row.get("metric")) for row in intent.data if "metric" in row]
    return [intent.title, *(axis.quantity for axis in intent.axes), *identities, *_spec_titles(compiled.spec)]


def _raw_keys(analysis: EvalAnalysis) -> set[str]:
    """The measure keys that read as keys — the ones with a separator in them; ``n`` and ``k`` are also words."""
    return {name for name in analysis.decision_surface.measures if any(sep in name for sep in "_:.")}


def _without_checks(text: str, keys: set[str]) -> str:
    """The text, less any goal check's own expression — the check's words, which may spell a host key inside them."""
    for key in keys:
        if (expression := goal_check_of(key)) is not None:
            text = text.replace(expression, "")
    return text


async def test_no_surface_table_or_compiled_chart_of_a_toy_campaign_heads_anything_by_a_raw_measure_key(toy) -> None:
    host, analysis, report = toy
    keys = _raw_keys(analysis)
    assert {"total_ms", "field_accuracy", "cost_usd"} <= keys

    texts: list[str] = []
    surface = build_surface_table(analysis)
    texts += [column.header for column in surface.columns]
    # The key a reader cites stays on the column; the header is what they read.
    assert {"total_ms", "field_accuracy"} & {column.measure_id for column in surface.columns}

    code_only = campaign_report(host, analysis.campaign_id, analysis.scope_id)
    charts = 0
    for each in (report, code_only):
        for block in each.blocks:
            if isinstance(block, TableBlock):
                texts += [column.header for column in block.columns]
                texts += [
                    str(row[key]) for row in block.rows for key in ("measure", "reading", "guardrail") if key in row
                ]
            if isinstance(block, ChartBlock) and block.intent is not None:
                charts += 1
                texts += _chart_texts(block.intent)
    frontier = compile_chart(
        "frontier",
        build_viz_payload(
            FrontierRef(quality=ReadingRef(measure_id="field_accuracy", reading="measure")),
            analysis.decision_surface,
            analysis.variant_index,
            measures=host.profile.measures,
        ),
    )
    texts += _chart_texts(frontier.intent)
    assert charts >= 2, "the toy report draws its delta table and the code-only report its distributions"

    offending = {text for text in texts for key in keys if key in _without_checks(text, keys)}
    assert offending == set()
    assert "Turn time" in " ".join(texts) and "Field accuracy (" not in frontier.intent.title


def test_a_host_measure_with_no_reader_name_is_refused_at_registration() -> None:
    unnamed = next(d for d in TOYHOST_MEASURES if d.name == "field_accuracy").model_copy(update={"reader_name": None})
    with pytest.raises(MeasureRegistrationError, match="field_accuracy has no reader_name"):
        MeasureRegistry([unnamed])


def test_two_measures_a_reader_could_not_tell_apart_are_refused() -> None:
    accuracy = next(d for d in TOYHOST_MEASURES if d.name == "field_accuracy")
    twin = accuracy.model_copy(update={"name": "field_accuracy_v2", "reader_name": "FIELD ACCURACY"})
    with pytest.raises(MeasureRegistrationError, match="share the reader_name"):
        MeasureRegistry([accuracy, twin])
    # Nor may a host measure take a core measure's words: its column would read as the core's.
    taken = accuracy.model_copy(update={"reader_name": METRIC_DESCRIPTORS["total_ms"].reader_name})
    with pytest.raises(MeasureRegistrationError, match="total_ms"):
        MeasureRegistry([taken])


def test_every_core_measure_and_every_one_described_by_construction_has_a_reader_name() -> None:
    assert all(d.reader_name for d in METRIC_DESCRIPTORS.values())
    names = [d.reader_name.casefold() for d in METRIC_DESCRIPTORS.values() if d.reader_name]
    assert len(names) == len(set(names))
    assert METRIC_DESCRIPTORS["candidate_output_tokens_per_s"].reader_name == "Output speed"


def test_a_descriptor_stored_before_the_field_reads_with_none_and_is_headed_by_its_key() -> None:
    stored = {k: v for k, v in METRIC_DESCRIPTORS["total_ms"].model_dump(mode="json").items() if k != "reader_name"}
    assert MetricDescriptor.model_validate_json(json.dumps(stored)).reader_name is None
