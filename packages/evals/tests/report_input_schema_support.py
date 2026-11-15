"""The parts of the kit's report-input schema that are the model's own words, written from the model.

``docs/eval-report-kit/schema/report-input.schema.json`` is a documentation copy of ``ReportInput``'s schema, laid out
by hand. The parts a caveat on looks is told in are not: the ``FalseSeparation`` and ``Looks`` definitions are the
model's schema for them, and the descriptions of the fields that carry them (:data:`MODEL_WORDED`) are the model's
descriptions, so what a writer reads of looks and of the reference rate cannot drift from what the model says.
``test_report_input.py`` holds the committed file to :func:`model_worded`.

To regenerate after a change to those models, from the repo root::

    uv run python -c "from packages.evals.tests.report_input_schema_support import write_model_worded; \
write_model_worded()"
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from threetears.evals.analysis.report_input import ReportInput

#: The kit's report-input schema.
SCHEMA = Path(__file__).resolve().parents[3] / "docs" / "eval-report-kit" / "schema" / "report-input.schema.json"

#: The definitions written whole from the model.
MODEL_DEFINED = ("FalseSeparation", "Looks")

#: The fields, by definition, whose description is the model's.
MODEL_WORDED = (
    ("Aspect", "looks"),
    ("CaveatEntry", "looks"),
    ("CaveatEntry", "false_separation"),
    ("RemedyStep", "declared_look"),
)

_DECODER = json.JSONDecoder()


def _model_defs() -> dict[str, Any]:
    defs: dict[str, Any] = ReportInput.model_json_schema(by_alias=True)["$defs"]
    return defs


def _definition(name: str) -> str:
    """The model's schema for one definition, laid out as the kit's file lays out the ones it writes."""
    schema = _model_defs()[name]
    properties = {}
    for field, shape in schema["properties"].items():
        shape = {key: value for key, value in shape.items() if key != "title"}
        if isinstance(shape.get("items"), dict):
            shape["items"] = {key: value for key, value in shape["items"].items() if key != "title"}
        properties[field] = {key: shape[key] for key in sorted(shape)}
    written: dict[str, Any] = {
        "additionalProperties": False,
        "description": schema.get("description", "").strip(),
        "properties": properties,
    }
    if schema.get("required"):
        written["required"] = schema["required"]
    written["type"] = "object"
    lines = json.dumps(written, indent=1, ensure_ascii=False).split("\n")
    return "\n".join(line if index == 0 else "  " + line for index, line in enumerate(lines))


def _span(text: str, name: str) -> tuple[int, int]:
    """Where one definition's object sits in the file: from its opening brace to its closing one."""
    start = text.index(f'\n  "{name}": {{\n') + len(f'\n  "{name}": ')
    close = re.compile(r"\n  \}").search(text, start)
    if close is None:
        raise ValueError(f"no end to the {name} definition")
    return start, close.end()


def model_worded(text: str) -> str:
    """``text`` (the kit's schema) with the looks parts written from the model.

    :param text: The file as it stands.
    :ptype text: str
    :return: The file with :data:`MODEL_DEFINED` written whole and :data:`MODEL_WORDED` described as the model does.
    :rtype: str
    """
    for name in MODEL_DEFINED:
        start, end = _span(text, name)
        text = text[:start] + _definition(name) + text[end:]
    defs = _model_defs()
    for name, field in MODEL_WORDED:
        start, end = _span(text, name)
        key = text.index(f'"{field}": ', text.index('"properties"', start), end)
        at = text.index('"description": ', key, end) + len('"description": ')
        _, stop = _DECODER.raw_decode(text, at)
        wording = json.dumps(defs[name]["properties"][field]["description"], ensure_ascii=False)
        text = text[:at] + wording + text[stop:]
    return text


def write_model_worded() -> None:
    """Write the looks parts of the kit's schema from the model."""
    SCHEMA.write_text(model_worded(SCHEMA.read_text(encoding="utf-8")), encoding="utf-8")
