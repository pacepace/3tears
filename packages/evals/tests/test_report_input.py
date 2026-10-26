"""The report input: its model and its documentation schema agree, and the five report-plan fixtures are valid inputs.

``ReportInput`` (``threetears.evals.analysis.report_input``) is what the report planner reads, and
``docs/eval-report-kit/schema/report-input.schema.json`` is its documentation copy. Contracts held here:

- the model and the schema name the same fields, require the same ones, close the same objects and allow the same
  enum values, object by object;
- each fixture input (``tests/fixtures/report_plans/r1..r5/input.json``) validates as the model and against the
  schema, and r5's is the kit's real Kairo input, byte for byte;
- ``build_inputs.py`` reproduces the committed inputs byte for byte, so they never drift from the references or the
  model;
- each r1-r4 input carries the signals its page's shape is classified by (GENERATOR.md §1), and the figures its page
  prints, copied rather than recomputed;
- the model copies what it is given: the real input round-trips unchanged, and an unknown key or role is refused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from pydantic import ValidationError

from packages.evals.tests.fixtures.report_plans.build_inputs import REFERENCES, build_inputs
from packages.evals.tests.report_kit_support import KIT
from threetears.evals.analysis.report_input import ReportInput

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "report_plans"
SCHEMA = KIT / "schema" / "report-input.schema.json"
NAMES = ["r1", "r2", "r3", "r4", "r5"]


def _input(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((FIXTURES / name / "input.json").read_text(encoding="utf-8"))
    return loaded


def _docs_schema() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(SCHEMA.read_text(encoding="utf-8"))
    return loaded


# --- the model and its documentation schema -----------------------------------------------------------------------


def _shape(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every object and enum in ``schema``, by where it sits: ``.arms[]`` an array's items, ``.measures{}`` a map's
    values. An object gives its property names, its required set and whether it is closed; an enum its values (a
    nullable enum's ``None`` dropped, since nullability is a type, not a value); a const its value."""
    defs = schema.get("$defs", {})
    found: dict[str, dict[str, Any]] = {}

    def walk(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return
        if "$ref" in node:
            walk(defs[node["$ref"].removeprefix("#/$defs/")], path)
        for branch in node.get("anyOf", []):
            walk(branch, path)
        entry = found.setdefault(path, {})
        if "enum" in node:
            entry["enum"] = sorted(v for v in node["enum"] if v is not None)
        if "const" in node:
            entry["const"] = node["const"]
        if "properties" in node:
            entry["properties"] = sorted(node["properties"])
            entry["required"] = sorted(node.get("required", []))
            entry["closed"] = node.get("additionalProperties") is False
            for name, sub in node["properties"].items():
                walk(sub, f"{path}.{name}")
        if isinstance(node.get("items"), dict):
            walk(node["items"], f"{path}[]")
        if isinstance(node.get("additionalProperties"), dict):
            walk(node["additionalProperties"], f"{path}{{}}")

    walk(schema, "")
    return {path: entry for path, entry in found.items() if entry}


def test_the_docs_schema_and_the_model_agree() -> None:
    docs = _shape(_docs_schema())
    model = _shape(ReportInput.model_json_schema(by_alias=True))
    assert sorted(docs) == sorted(model), (
        f"only in the docs schema: {sorted(set(docs) - set(model))}; only in the model: {sorted(set(model) - set(docs))}"
    )
    differ = {path: {"docs": docs[path], "model": model[path]} for path in docs if docs[path] != model[path]}
    assert not differ, json.dumps(differ, indent=1)


def test_the_parity_check_sees_a_field_the_docs_schema_lacks() -> None:
    docs = _docs_schema()
    del docs["$defs"]["Task"]["properties"]["routable"]
    assert _shape(docs)[".tasks[]"] != _shape(ReportInput.model_json_schema(by_alias=True))[".tasks[]"]


def test_the_docs_schema_is_a_valid_schema() -> None:
    jsonschema.Draft202012Validator.check_schema(_docs_schema())


# --- the five fixture inputs --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_a_fixture_input_is_a_report_input(name: str) -> None:
    document = _input(name)
    ReportInput.model_validate(document)
    jsonschema.Draft202012Validator(_docs_schema()).validate(document)


@pytest.mark.parametrize("name", ["r1", "r2", "r3", "r4"])
def test_a_synthetic_input_says_it_is_synthetic(name: str) -> None:
    assert ReportInput.model_validate(_input(name)).meta.synthetic is True


def test_r5_is_the_kits_real_input_byte_for_byte() -> None:
    assert (FIXTURES / "r5" / "input.json").read_bytes() == (REFERENCES / "r5-kairo-round2.json").read_bytes()
    assert ReportInput.model_validate(_input("r5")).meta.synthetic is False


def test_building_the_inputs_reproduces_the_committed_files(tmp_path: Path) -> None:
    first = build_inputs(tmp_path / "first")
    second = build_inputs(tmp_path / "second")
    assert [p.relative_to(tmp_path / "first") for p in first] == [Path(n) / "input.json" for n in NAMES]
    for one, two in zip(first, second, strict=True):
        assert one.read_bytes() == two.read_bytes()
        committed = FIXTURES / one.relative_to(tmp_path / "first")
        assert one.read_bytes() == committed.read_bytes(), (
            f"{committed} is stale: rerun packages/evals/tests/fixtures/report_plans/build_inputs.py"
        )


# --- what each input carries --------------------------------------------------------------------------------------


def test_r1_carries_tasks_their_gates_and_the_cross_cutting_one() -> None:
    report = ReportInput.model_validate(_input("r1"))
    assert all(arm.role == "challenger" for arm in report.arms)
    assert [t.id for t in report.tasks if t.cross_cutting] == ["specul"]
    assert all(t.routable for t in report.tasks if not t.cross_cutting)
    gates = {c.id: c for c in report.criteria}
    assert gates["overall"].scope == "each_task" and gates["overall"].threshold == 85
    assert gates["p95"].statistic == "p95" and gates["p95"].direction == "lower_is_better"
    assert gates["specul.declines"].task == "specul" and gates["specul.declines"].threshold == 95
    trouble = next(t for t in report.tasks if t.id == "trouble")
    # the page's Watch: Atlas-M resolves 90% of troubleshooting issues, CI ±4.4
    resolves = trouble.per_arm["atlas-m"]["resolves"]
    assert (resolves.mean, resolves.half_width) == (90, 4.4)
    assert report.judged["overall"].scale == "ordinal" and report.judged["resolves"].scale == "pass_fail"


def test_r2_carries_a_full_factorial_grid() -> None:
    report = ReportInput.model_validate(_input("r2"))
    assert [f.id for f in report.factors] == ["model", "reasoning", "retry"]
    levels = {f.id: [level.id for level in f.levels] for f in report.factors}
    grid = {tuple(c.levels[f] for f in levels) for c in report.configs}
    assert len(grid) == len(report.arms) == len(levels["model"]) * len(levels["reasoning"]) * len(levels["retry"])
    assert {c.arm for c in report.configs} == {a.id for a in report.arms}
    assert report.measures["pass_rate"].grade == "pass_fail"
    assert report.measures["calls_per_task"].minimum == 1.38
    outcome = report.measures["outcome"]
    assert outcome.bins is not None
    assert all(len(stat.shares or []) == len(outcome.bins) for stat in outcome.per_arm.values())
    assert {e.stratum for e in report.examples} <= {t.id for t in report.tasks}


def test_r3_carries_the_target_matrix_its_cells_and_where_errors_go() -> None:
    report = ReportInput.model_validate(_input("r3"))
    matrix = report.target_matrix
    assert matrix is not None
    expected = {f"{r.id}|{c.id}" for r in matrix.rows.levels for c in matrix.columns.levels}
    assert set(report.cells) == expected
    arms = {a.id for a in report.arms}
    assert all(set(cell.per_arm) == arms for cell in report.cells.values())
    assert report.cells["complex|conv"].n_cases == 1960
    assert set(report.confusion) == {"complexity", "domain"}
    for table in report.confusion.values():
        assert set(table.per_arm) == arms
        assert all(
            len(m) == len(table.labels) and all(len(r) == len(table.labels) for r in m) for m in table.per_arm.values()
        )
    assert {c.scope for c in report.criteria} == {"overall", "each_cell"}
    assert report.measures["balanced_accuracy"].grade == "class_label"
    assert report.meta.volume_per_month == 30_000_000


def test_r4_carries_a_control_its_contrasts_guardrails_strata_and_human_preference() -> None:
    report = ReportInput.model_validate(_input("r4"))
    control = [a.id for a in report.arms if a.role == "control"]
    assert control == ["prod"]
    challengers = [a.id for a in report.arms if a.role == "challenger"]
    aspects = [a.id for a in report.aspects]
    assert {(c.arm, c.measure) for c in report.contrasts} == {(a, m) for a in challengers for m in aspects}
    persona_f64 = next(c for c in report.contrasts if (c.arm, c.measure) == ("f64", "persona"))
    assert (persona_f64.delta, persona_f64.interval, persona_f64.verdict) == (-6.1, (-7.3, -4.9), "regressed")
    facts_f64 = next(c for c in report.contrasts if (c.arm, c.measure) == ("f64", "facts"))
    assert facts_f64.verdict == "not_separated"
    assert [f.merit_axes for f in report.contrast_families] == [[]]
    assert {(c.measure, c.threshold, c.kind, c.standing) for c in report.criteria} == {
        ("facts", 90, "guardrail", True),
        ("specul", 95, "guardrail", True),
    }
    assert {s.dimension for s in report.strata} == {"depth", "scenario"}
    assert sum(s.share or 0 for s in report.strata if s.dimension == "depth") == pytest.approx(1.0)
    assert report.human_pref is not None and report.human_pref.control == "prod"
    assert report.primary_judged == "persona"
    # a challenger lists only the levers that differ from the control, which lists them all
    settings = {a.id: a.settings for a in report.arms}
    assert set(settings["prod"]) == {"model", "max_output_tokens", "reasoning"}
    assert settings["cor"] == {"model": "Corvid-3.8-flash", "reasoning": "high"}


# --- the model copies what it is given ----------------------------------------------------------------------------


def test_the_real_input_round_trips_unchanged() -> None:
    document = _input("r5")
    dumped = ReportInput.model_validate(document).model_dump(mode="json", by_alias=True, exclude_unset=True)
    assert dumped == document


def test_an_unknown_key_is_refused_not_dropped() -> None:
    document = _input("r5")
    document["arms"][0]["colour"] = "blue"
    with pytest.raises(ValidationError, match="colour"):
        ReportInput.model_validate(document)


def test_an_arm_role_outside_the_enum_is_refused() -> None:
    document = _input("r1")
    document["arms"][0]["role"] = "baseline"
    with pytest.raises(ValidationError, match="role"):
        ReportInput.model_validate(document)


def test_a_figure_keeps_the_type_it_was_given() -> None:
    report = ReportInput.model_validate(_input("r1"))
    assert type(report.criteria[0].threshold) is int
    assert type(next(c for c in report.criteria if c.id == "p95").threshold) is float
