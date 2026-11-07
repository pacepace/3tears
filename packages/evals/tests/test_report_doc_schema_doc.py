"""``docs/eval-report-kit/schema/report-doc.schema.json`` is the ``report-doc/1`` model's JSON schema, written out.

The file is what an analyst reads and writes against, so it is the model's own schema byte for byte, never a copy kept
by hand. Byte equality alone would hold for a schema that said nothing, so the file is also used as a schema: it
accepts every converted prototype document, and refuses what the model refuses for its bounds and vocabularies.

To regenerate it after a change to the model, from the repo root::

    uv run python -c "import json; from threetears.evals.kernel.report_doc import report_doc_json_schema; \
open('docs/eval-report-kit/schema/report-doc.schema.json', 'w').write(json.dumps(report_doc_json_schema(), \
indent=2, ensure_ascii=False) + '\\n')"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from packages.evals.tests.report_doc_support import LATER_DOCS, PROTOTYPE1_DOCS, converted_doc
from threetears.evals.kernel.report_doc import REPORT_DOC_SCHEMA, report_doc_json_schema

#: The documentation copy of the schema.
_DOCS_SCHEMA = Path(__file__).resolve().parents[3] / "docs" / "eval-report-kit" / "schema" / "report-doc.schema.json"


def _written() -> str:
    return json.dumps(report_doc_json_schema(), indent=2, ensure_ascii=False) + "\n"


def _docs() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(_DOCS_SCHEMA.read_text())
    return loaded


def test_the_docs_schema_is_the_model_s_schema_byte_for_byte() -> None:
    """A change to the model moves the file in the same commit (regenerate it as this module's docstring says)."""
    assert _DOCS_SCHEMA.read_text() == _written()


def test_the_docs_schema_is_a_valid_schema_that_names_its_id() -> None:
    """The file is a 2020-12 schema whose id is the schema id every document carries."""
    jsonschema.Draft202012Validator.check_schema(_docs())
    assert _docs()["$id"] == REPORT_DOC_SCHEMA


@pytest.mark.parametrize("name", PROTOTYPE1_DOCS + LATER_DOCS)
def test_the_docs_schema_accepts_each_converted_document(name: str) -> None:
    """What the model stores, the schema an analyst writes against accepts."""
    jsonschema.validate(converted_doc(name), _docs())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("decision", 0, "slot"), "verdict"),
        (("claims", 0, "evidence"), []),
        (("claims", 0, "weight"), "headline"),
        (("headline",), "x" * 241),
        (("confidence", "form"), "certain"),
        (("confidence", "caveats", 0, "checked"), "material"),
        (("claims", 0, "evidence", 0, "block"), "pie"),
        (("claims", 0, "evidence", 0, "query", "transform"), "log"),
        (("explore",), [{"control": "c", "binds": "b", "outcome": "o"}] * 4),
        (("verdict",), "keep"),
    ],
)
def test_the_docs_schema_refuses_what_the_model_bounds(path: tuple[str | int, ...], value: Any) -> None:
    """The file carries the model's vocabularies, bounds and closed objects, not only its field names."""
    doc = converted_doc("prototype3/kairo")
    node: Any = doc
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = value

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, _docs())
