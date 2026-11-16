"""The looks parts of the kit's report-input schema, checked against the model.

``docs/eval-report-kit/schema/report-input.schema.json`` is a documentation copy of ``ReportInput``'s schema, kept by
hand. What it tells a writer about looks and the reference rate must be the model's own words, so the descriptions of
the ``FalseSeparation`` and ``Looks`` fields and of the fields that carry them (:data:`MODEL_WORDED`) are checked
against the model's (:func:`looks_description_drift`). Where the model's words change, the file is edited to match.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from threetears.evals.analysis.report_input import ReportInput

#: The kit's report-input schema.
SCHEMA = Path(__file__).resolve().parents[3] / "docs" / "eval-report-kit" / "schema" / "report-input.schema.json"

#: The definitions whose every field's description is the model's.
MODEL_DEFINED = ("FalseSeparation", "Looks")

#: The fields, by definition, whose description is the model's.
MODEL_WORDED = (
    ("Aspect", "looks"),
    ("CaveatEntry", "looks"),
    ("CaveatEntry", "false_separation"),
    ("RemedyStep", "declared_look"),
)


def looks_description_drift() -> list[tuple[str, str]]:
    """Each looks field whose description in the kit's schema is not the model's, or that one side lacks.

    :return: ``(definition, field)`` pairs; empty where every one agrees.
    :rtype: list[tuple[str, str]]
    """
    docs: dict[str, Any] = json.loads(SCHEMA.read_text(encoding="utf-8"))["$defs"]
    model: dict[str, Any] = ReportInput.model_json_schema(by_alias=True)["$defs"]
    fields = [
        *MODEL_WORDED,
        *((name, field) for name in MODEL_DEFINED for field in {*model[name]["properties"], *docs[name]["properties"]}),
    ]
    return sorted(
        (name, field)
        for name, field in fields
        if docs[name]["properties"].get(field, {}).get("description")
        != model[name]["properties"].get(field, {}).get("description")
    )
