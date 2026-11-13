"""Every id a host may name reads one name on every surface, by one rule each, and two ids never read as one.

- **A host's name, else the id.** An aspect, a stratum and a task print the name their input declares, else their id
  with its underscores and dots as spaces (:func:`~threetears.evals.analysis.report.words.given_name`): a figure key
  (``aspects.<id>.name``, ``task.<id>.name``), :func:`~threetears.evals.analysis.report.figure_keys.reader_name` and a
  chart's slice all read that one rule.
- **A judged dimension's declared label wins everywhere.** A label the input declares for a judged dimension
  (``Judged.label``) is its name on a chart and a written table (``measure_words``), in a figure key
  (``judged.<id>.label``) and among the names the analyst is shown (``reader_names``); undeclared, every one of them
  names it as :func:`~threetears.evals.analysis.report.words.judged_name` does.
- **A stratum is named on its own aspect.** A stratum id held on two aspects, each declaring its own name, is a stratum
  on each: :func:`reader_name` names each on its aspect, and ``reader_names`` keeps both names, each keyed by the stem
  its figure keys take there (``aspects.<aspect>.strata.<id>``), where it used to keep the first aspect's alone.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.analysis.analyst.blocks import block_render
from threetears.evals.analysis.report.block_spec import block_look
from threetears.evals.analysis.report.figure_keys import reader_name, reader_names, resolve
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.parts.full_results import every_table
from threetears.evals.analysis.report.parts.tables import arms_table
from threetears.evals.analysis.report.parts.charting import measure_words
from threetears.evals.analysis.report.words import given_name, judged_name, label_case
from threetears.evals.analysis.report_arms import fact_arms
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.chart_query import ChartQuery

FIGURES = {"base": {"m": {"mean": 0.8}}, "a": {"m": {"mean": 0.7}}}


def _input(**parts: Any) -> ReportInput:
    """A control and one candidate on two aspects, a judged dimension and a stratum held on each aspect."""
    base: dict[str, Any] = {
        "schema": "report-input/1",
        "meta": {"campaign": "Names", "kind": "change"},
        "arms": [{"id": "base", "role": "control", "settings": {}}, {"id": "a", "role": "challenger", "settings": {}}],
        "aspects": [{"id": "reply_tone", "asked": True, "readings": []}, {"id": "billing.v2", "readings": []}],
        "scoreboard": {},
        "measures": {"m": {"grade": "pass_fail", "higher_is_better": True, "per_arm": {}}},
        "judged": {"stays_polite": {"scale": "pass_fail", "per_arm": {"base": {"mean": 0.9}, "a": {"mean": 0.8}}}},
        "contrasts": [],
        "tasks": [{"id": "refund_flow", "n_cases": 4, "per_arm": FIGURES}],
        "strata": [
            {"id": "off_lane", "n_cases": 10, "aspect": "reply_tone", "per_arm": FIGURES},
            {"id": "off_lane", "n_cases": 10, "aspect": "billing.v2", "per_arm": FIGURES},
        ],
        "caveats": {},
    }
    return ReportInput.model_validate({**base, **parts})


class TestAHostsNameElseTheId:
    def test_an_undeclared_aspect_and_task_read_their_id_humanised_on_every_surface(self) -> None:
        held = _input()
        for aspect in held.aspects:
            named = given_name(aspect.id)
            assert resolve(f"aspects.{aspect.id}.name", held).value == named == reader_name("aspect", aspect.id, held)
        assert given_name("billing.v2") == "billing v2"
        assert resolve("task.refund_flow.name", held).value == "refund flow" == reader_name("task", "refund_flow", held)

    def test_a_declared_aspect_and_task_read_their_name_on_every_surface(self) -> None:
        held = _input(
            aspects=[
                {"id": "reply_tone", "name": "Tone", "asked": True, "readings": []},
                {"id": "billing.v2", "readings": []},
            ],
            tasks=[{"id": "refund_flow", "name": "Refunds", "n_cases": 4, "per_arm": FIGURES}],
        )
        assert resolve("aspects.reply_tone.name", held).value == "Tone" == reader_name("aspect", "reply_tone", held)
        assert resolve("task.refund_flow.name", held).value == "Refunds" == reader_name("task", "refund_flow", held)


class TestADeclaredJudgedLabelWinsOnEverySurface:
    def test_a_declared_label_names_the_dimension_on_a_chart_a_table_a_key_and_the_analysts_names(self) -> None:
        held = _input(
            judged={
                "stays_polite": {
                    "label": "Keeps it civil",
                    "scale": "pass_fail",
                    "per_arm": {"base": {"mean": 0.9}, "a": {"mean": 0.8}},
                }
            }
        )
        assert measure_words(held, "stays_polite") == "Keeps it civil"
        assert measure_words(held, "stays_polite", kind="judged", prose=True) == "Keeps it civil"
        assert resolve("judged.stays_polite.label", held).value == "Keeps it civil"
        assert reader_name("judged", "stays_polite", held) == "Keeps it civil"
        assert reader_names(held)["judged"] == {"stays_polite": "Keeps it civil"}

    def test_undeclared_every_surface_reads_the_one_rule(self) -> None:
        held = _input()
        named = judged_name("stays_polite")
        assert named == "stays polite"
        assert measure_words(held, "stays_polite") == label_case(named)
        assert measure_words(held, "stays_polite", prose=True) == named
        assert resolve("judged.stays_polite.label", held).value == named == reader_name("judged", "stays_polite", held)


class TestAStratumIsNamedOnItsOwnAspect:
    def test_one_id_two_aspects_two_declared_names_each_named_and_both_kept(self) -> None:
        held = _input(
            strata=[
                {"id": "off_lane", "name": "Off-lane asks", "n_cases": 10, "aspect": "reply_tone", "per_arm": FIGURES},
                {"id": "off_lane", "name": "Wrong desk", "n_cases": 10, "aspect": "billing.v2", "per_arm": FIGURES},
            ]
        )
        assert reader_name("stratum", "off_lane", held, aspect="reply_tone") == "Off-lane asks"
        assert reader_name("stratum", "off_lane", held, aspect="billing.v2") == "Wrong desk"
        assert reader_names(held)["stratum"] == {
            "aspects.reply_tone.strata.off_lane": "Off-lane asks",
            "aspects.billing.v2.strata.off_lane": "Wrong desk",
        }

    def test_one_id_every_aspect_prints_alike_is_keyed_by_its_id(self) -> None:
        held = _input()
        assert reader_names(held)["stratum"] == {"off_lane": "off lane"}
        assert reader_name("stratum", "off_lane", held, aspect="billing.v2") == "off lane"

    def test_two_ids_one_aspect_would_print_alike_each_print_as_its_id_on_that_aspect_alone(self) -> None:
        """``edge_case`` and ``edge.case`` on one aspect would both read ``edge case`` there, so each reads as its
        id; ``edge_case`` alone on the other aspect reads its humanised name, and the two names are both kept."""
        held = _input(
            strata=[
                {"id": "edge_case", "n_cases": 10, "aspect": "reply_tone", "per_arm": FIGURES},
                {"id": "edge.case", "n_cases": 10, "aspect": "reply_tone", "per_arm": FIGURES},
                {"id": "edge_case", "n_cases": 10, "aspect": "billing.v2", "per_arm": FIGURES},
            ]
        )
        assert reader_name("stratum", "edge_case", held, aspect="reply_tone") == "edge_case"
        assert reader_name("stratum", "edge_case", held, aspect="billing.v2") == "edge case"
        assert reader_names(held)["stratum"] == {
            "aspects.reply_tone.strata.edge_case": "edge_case",
            "aspects.billing.v2.strata.edge_case": "edge case",
            "edge.case": "edge.case",
        }


#: One stratum id on two aspects, each declaring its own name, with a failed example on each.
_NAMED_APART = {
    "strata": [
        {"id": "off_lane", "name": "Off-lane asks", "n_cases": 10, "aspect": "reply_tone", "per_arm": FIGURES},
        {"id": "off_lane", "name": "Wrong desk", "n_cases": 10, "aspect": "billing.v2", "per_arm": FIGURES},
    ],
    "examples": [
        {"trial": "T1", "case": "c1", "stratum": "off_lane", "aspect": "reply_tone", "arm": "a", "outcome": "fail"},
        {"trial": "T2", "case": "c2", "stratum": "off_lane", "aspect": "billing.v2", "arm": "a", "outcome": "fail"},
    ],
}

#: Every query field, empty: no filter, no limit.
_QUERY: dict[str, Any] = {
    "aspect": "",
    "measures": [],
    "arms": [],
    "strata": [],
    "limit": 0,
    "transform": "none",
    "small_multiples_by": "none",
}


class TestEveryTableNamesAStratumOnItsOwnAspect:
    """Each table a written page lays a stratum in reads it by its own aspect's name: both declared names are printed,
    and the bare id, which no aspect names it by, never is."""

    def test_the_full_results_strata_table(self) -> None:
        held = _input(**_NAMED_APART)
        (table,) = [
            table
            for table in every_table(held, block_look(held, fact_arms(held)), forms=PageForms.of(held))
            if table.name == "strata"
        ]
        printed = [row["strata"].text for row in table.rows]
        assert "Off-lane asks" in printed and "Wrong desk" in printed and "off lane" not in printed

    def test_the_examples_table_and_the_example_cards(self) -> None:
        held = _input(**_NAMED_APART)
        for kind in ("table", "examples"):
            rendered = block_render(
                held, block_kind=kind, query=ChartQuery.model_validate({**_QUERY, "source": "examples"})
            ).model_dump_json()
            assert "Off-lane asks" in rendered and "Wrong desk" in rendered, kind
            assert "off lane" not in rendered, kind


class TestAnAspectsColumnIsNamedByTheOneRule:
    def test_an_undeclared_aspect_reads_its_id_humanised_in_the_arms_by_measure_table(self) -> None:
        judged = {
            "stays_polite": {
                "scale": "pass_fail",
                "per_arm": {"base": {"mean": 0.9}, "a": {"mean": 0.8}},
                "by_aspect": {"billing.v2": {"base": {"mean": 0.7}, "a": {"mean": 0.6}}},
            }
        }
        held = _input(judged=judged)
        (table,) = [
            table
            for table in every_table(held, block_look(held, fact_arms(held)), forms=PageForms.of(held))
            if table.name == "arms_by_measure"
        ]
        assert {row["aspect"].text for row in table.rows} == {"reply tone", "billing v2"}

    @pytest.mark.parametrize(
        ("names", "heads"),
        [
            pytest.param({}, ("Reply tone", "Billing v2"), id="undeclared"),
            pytest.param({"reply_tone": "Tone", "billing.v2": "Billing"}, ("Tone", "Billing"), id="declared"),
        ],
    )
    def test_the_arms_table_heads_each_aspects_columns_by_its_name(
        self, names: dict[str, str], heads: tuple[str, str]
    ) -> None:
        board = {"base": {"ran": 10, "usable": 10, "passed": 8}, "a": {"ran": 10, "usable": 10, "passed": 7}}
        held = _input(
            aspects=[
                {"id": aspect, "asked": aspect == "reply_tone", "readings": [], "scoreboard": board}
                | ({"name": names[aspect]} if aspect in names else {})
                for aspect in ("reply_tone", "billing.v2")
            ]
        )
        table = arms_table(held, block_look(held, fact_arms(held)), (), forms=PageForms.of(held))
        headers = [column.header for column in table.columns]
        for head in heads:
            assert any(header.startswith(f"{head} ") for header in headers), (head, headers)
        assert not any("billing.v2" in header or "reply_tone" in header for header in headers), headers
