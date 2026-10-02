"""Chat-model stand-ins that answer a structured-output call the way a real model does.

Every model call in this package goes through ``llm_retry.create_chat_model(...)
.with_structured_output(schema)``. A real model is handed the schema's JSON schema and answers
with JSON, which LangChain validates into an instance of *schema*. These fakes do the same: a
test states its answer as the JSON a model would send (a plain ``dict``), and the fake validates
it against whatever schema the code under test asked for. So a test never names the private
pydantic class a module forces its model call into -- it states the answer on the wire, which is
the contract the model actually has with the code.

:func:`answer_shape` names a requested schema by what it asks the model for, read from its JSON
schema the way a model reads it, so a test that holds answers for several different calls at
once (a page classifier, a candidate generator and a judge in one poll) can tell them apart
without depending on prompt text or a class name.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from pydantic import BaseModel

__all__ = [
    "CSS_CANDIDATES",
    "DISCOVERED_CANDIDATES",
    "DISCOVERED_ROW_CANDIDATES",
    "ENRICHMENT_NOTES",
    "JUDGE_VERDICT",
    "MULTI_ROW_JUDGE_VERDICT",
    "PAGE_VERDICT",
    "REGEX_CANDIDATES",
    "ROW_CANDIDATES",
    "answer_shape",
    "fake_structured_model",
    "models_answering_by_shape",
]

#: a candidate-generation answer whose candidates are ``{"selectors": {...}}``.
CSS_CANDIDATES = "css_candidates"
#: a candidate-generation answer whose candidates are ``{"row_selector", "field_selectors"}``.
ROW_CANDIDATES = "row_candidates"
#: a candidate-generation answer whose candidates are ``{"pattern": ...}``.
REGEX_CANDIDATES = "regex_candidates"
#: a schema-discovery answer whose candidates are ``{"fields": [...]}``.
DISCOVERED_CANDIDATES = "discovered_candidates"
#: a row schema-discovery answer whose candidates are ``{"row_selector", "fields": [...]}``.
DISCOVERED_ROW_CANDIDATES = "discovered_row_candidates"
#: a judge answer: ``{"winning_candidate_index": int | None, "reasoning": str}``.
JUDGE_VERDICT = "judge_verdict"
#: a multi-row judge answer: ``{"confirmed_record_indices": [...], "reasoning": str}``.
MULTI_ROW_JUDGE_VERDICT = "multi_row_judge_verdict"
#: a failed-page classifier answer: ``{"kind", "evidence", "confidence"}``.
PAGE_VERDICT = "page_verdict"
#: an enrichment answer: ``{"notes": {...}}``.
ENRICHMENT_NOTES = "enrichment_notes"


def _properties(json_schema: Mapping[str, Any], defs: Mapping[str, Any]) -> set[str]:
    """the property names of one JSON-schema object, following a ``$ref`` into *defs*.

    :param json_schema: an object schema, or a ``{"$ref": "#/$defs/<name>"}`` pointing at one
    :ptype json_schema: Mapping[str, Any]
    :param defs: the root schema's ``$defs``
    :ptype defs: Mapping[str, Any]
    :return: the object's property names
    :rtype: set[str]
    """
    ref = json_schema.get("$ref")
    resolved = defs[ref.rsplit("/", 1)[-1]] if isinstance(ref, str) else json_schema
    return set(resolved.get("properties", {}))


def answer_shape(schema: type[BaseModel]) -> str:
    """names what *schema* asks a model for, read from its JSON schema.

    :param schema: the response model the code under test passed to ``with_structured_output``
    :ptype schema: type[BaseModel]
    :return: one of this module's shape constants
    :rtype: str
    :raises AssertionError: when the schema is no shape this module knows, so a test that meets a
        new kind of call fails naming it rather than answering it with something else
    """
    json_schema = schema.model_json_schema()
    defs = json_schema.get("$defs", {})
    top = set(json_schema.get("properties", {}))
    shape: str | None = None
    if "winning_candidate_index" in top:
        shape = JUDGE_VERDICT
    elif "confirmed_record_indices" in top:
        shape = MULTI_ROW_JUDGE_VERDICT
    elif {"kind", "evidence", "confidence"} <= top:
        shape = PAGE_VERDICT
    elif top == {"notes"}:
        shape = ENRICHMENT_NOTES
    elif top == {"candidates"}:
        item = _properties(json_schema["properties"]["candidates"]["items"], defs)
        by_item = {
            frozenset({"selectors"}): CSS_CANDIDATES,
            frozenset({"row_selector", "field_selectors"}): ROW_CANDIDATES,
            frozenset({"pattern"}): REGEX_CANDIDATES,
            frozenset({"fields"}): DISCOVERED_CANDIDATES,
            frozenset({"row_selector", "fields"}): DISCOVERED_ROW_CANDIDATES,
        }
        shape = by_item.get(frozenset(item))
    if shape is None:
        raise AssertionError(f"no known answer shape for a schema with properties {sorted(top)}")
    return shape


def _as_answer(schema: type[BaseModel], answer: Any) -> Any:
    """validate a JSON-shaped *answer* into *schema*, as LangChain does with a model's reply.

    :param schema: the requested response model
    :ptype schema: type[BaseModel]
    :param answer: a ``dict`` of the model's JSON, or anything else, returned unchanged
    :ptype answer: Any
    :return: an instance of *schema* for a ``dict``; *answer* otherwise
    :rtype: Any
    """
    return schema.model_validate(answer) if isinstance(answer, dict) else answer


def fake_structured_model(result: Any = None, *, side_effect: Any = None) -> tuple[Any, AsyncMock]:
    """a chat model that answers every structured-output call with *result* (or *side_effect*).

    :param result: the model's JSON answer as a ``dict`` (validated into whatever schema is
        requested), or a ready instance of a public response model
    :ptype result: Any
    :param side_effect: as ``AsyncMock``'s: an exception to raise, or an iterable of answers and
        exceptions taken one per call
    :ptype side_effect: Any
    :return: the model (for patching ``llm_retry.create_chat_model``) and the mock every
        ``ainvoke`` is recorded on, with the prompt as its argument
    :rtype: tuple[Any, AsyncMock]
    """
    ainvoke_mock = AsyncMock(return_value=result, side_effect=side_effect)

    def _with_structured_output(schema: type[BaseModel], **_kwargs: Any) -> Any:
        async def _ainvoke(prompt: Any) -> Any:
            return _as_answer(schema, await ainvoke_mock(prompt))

        return SimpleNamespace(ainvoke=_ainvoke)

    return SimpleNamespace(with_structured_output=_with_structured_output), ainvoke_mock


def models_answering_by_shape(responses: Mapping[str, Any], requested: list[str] | None = None) -> Callable[..., Any]:
    """a ``create_chat_model`` replacement that answers by the shape of the schema asked for.

    :param responses: answer shape -> the model's JSON answer, a public response-model instance,
        or an exception to raise
    :ptype responses: Mapping[str, Any]
    :param requested: accumulates the shape of every structured call made, in order -- the record
        a test asserts on to prove a call was or was not made
    :ptype requested: list[str] | None
    :return: a side_effect for patching ``llm_retry.create_chat_model``
    :rtype: Callable[..., Any]
    """

    def _create(*_args: Any, **_kwargs: Any) -> Any:
        def _with_structured_output(schema: type[BaseModel], **_kw: Any) -> Any:
            shape = answer_shape(schema)
            if requested is not None:
                requested.append(shape)
            answer = responses.get(shape)
            if isinstance(answer, Exception):
                return SimpleNamespace(ainvoke=AsyncMock(side_effect=answer))
            return SimpleNamespace(ainvoke=AsyncMock(return_value=_as_answer(schema, answer)))

        return SimpleNamespace(with_structured_output=_with_structured_output)

    return _create
