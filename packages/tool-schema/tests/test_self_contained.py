"""a tool's JSON Schema made self-contained, and the one type a property declares.

The schema below is exactly what pydantic 2 renders for a storyboard tool's arguments -- nested
models as ``$ref`` into ``$defs``, optional fields as ``anyOf`` with ``null``, a two-model union, an
untyped field and a recursive outline -- written out so this dependency-free package is tested
without pydantic.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from threetears.tool_schema import declared_type, self_contained_input_schema

_STORYBOARD: dict[str, Any] = {
    "$defs": {
        "Close": {
            "description": "A close-up.",
            "properties": {"subject": {"title": "Subject", "type": "string"}},
            "required": ["subject"],
            "title": "Close",
            "type": "object",
        },
        "Lighting": {
            "description": "How a scene is lit.",
            "properties": {"key": {"description": "the key light", "title": "Key", "type": "string"}},
            "required": ["key"],
            "title": "Lighting",
            "type": "object",
        },
        "Node": {
            "description": "An outline node.",
            "properties": {
                "label": {"description": "the node's text", "title": "Label", "type": "string"},
                "children": {
                    "description": "the nodes under this one",
                    "items": {"$ref": "#/$defs/Node"},
                    "title": "Children",
                    "type": "array",
                },
            },
            "required": ["label"],
            "title": "Node",
            "type": "object",
        },
        "Scene": {
            "description": "Where a scene happens.",
            "properties": {
                "location": {"description": "the place", "title": "Location", "type": "string"},
                "lighting": {
                    "anyOf": [{"$ref": "#/$defs/Lighting"}, {"type": "null"}],
                    "default": None,
                    "description": "the lighting, when it matters",
                },
            },
            "required": ["location"],
            "title": "Scene",
            "type": "object",
        },
        "Shot": {
            "description": "One camera shot.",
            "properties": {
                "prompt": {"description": "what the shot shows", "title": "Prompt", "type": "string"},
                "seconds": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "default": None,
                    "description": "how long it runs",
                    "title": "Seconds",
                },
            },
            "required": ["prompt"],
            "title": "Shot",
            "type": "object",
        },
        "Wide": {
            "description": "A wide shot.",
            "properties": {"landscape": {"title": "Landscape", "type": "string"}},
            "required": ["landscape"],
            "title": "Wide",
            "type": "object",
        },
    },
    "properties": {
        "shots": {
            "description": "the shots, in order",
            "items": {"$ref": "#/$defs/Shot"},
            "title": "Shots",
            "type": "array",
        },
        "scene": {"$ref": "#/$defs/Scene", "description": "the scene they belong to"},
        "framing": {
            "anyOf": [{"$ref": "#/$defs/Close"}, {"$ref": "#/$defs/Wide"}],
            "description": "the framing",
            "title": "Framing",
        },
        "anything": {"default": None, "description": "any value at all", "title": "Anything"},
        "outline": {
            "anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}],
            "default": None,
            "description": "the outline, if there is one",
        },
    },
    "required": ["shots", "scene", "framing"],
    "title": "StoryboardInput",
    "type": "object",
}


_SHOT = {
    "type": "object",
    "description": "One camera shot.",
    "properties": {
        "prompt": {"type": "string", "description": "what the shot shows"},
        "seconds": {"type": "integer", "description": "how long it runs"},
    },
    "required": ["prompt"],
}


def _storyboard() -> dict[str, Any]:
    """the storyboard schema, self-contained.

    :return: the result
    :rtype: dict[str, Any]
    """
    return self_contained_input_schema(_STORYBOARD, tool_name="storyboard")


class TestSelfContainedInputSchema:
    def test_a_list_of_models_is_an_array_of_objects(self) -> None:
        """``list[Shot]`` is an array of Shot objects, each keeping its required list.

        :return: none
        :rtype: None
        """
        assert _storyboard()["properties"]["shots"] == {
            "type": "array",
            "description": "the shots, in order",
            "items": _SHOT,
        }

    def test_a_nested_sub_object_keeps_its_properties_required_list_and_field_description(self) -> None:
        """a sub-object is the object, and the field's description wins over the model's.

        :return: none
        :rtype: None
        """
        scene = _storyboard()["properties"]["scene"]
        assert scene["type"] == "object"
        assert scene["description"] == "the scene they belong to"
        assert scene["required"] == ["location"]
        assert scene["properties"]["location"] == {"type": "string", "description": "the place"}

    def test_an_optional_nested_model_is_unwrapped_to_the_object(self) -> None:
        """``Lighting | None`` collapses to the Lighting object, at any depth.

        :return: none
        :rtype: None
        """
        assert _storyboard()["properties"]["scene"]["properties"]["lighting"] == {
            "type": "object",
            "description": "the lighting, when it matters",
            "properties": {"key": {"type": "string", "description": "the key light"}},
            "required": ["key"],
        }

    def test_a_union_of_models_keeps_every_member(self) -> None:
        """``Close | Wide`` stays a union of both, never the first alone.

        :return: none
        :rtype: None
        """
        framing = _storyboard()["properties"]["framing"]
        assert framing["description"] == "the framing"
        assert [sorted(member["properties"]) for member in framing["anyOf"]] == [["subject"], ["landscape"]]

    def test_an_untyped_field_is_not_forced_to_a_string(self) -> None:
        """``Any`` stays untyped.

        :return: none
        :rtype: None
        """
        assert _storyboard()["properties"]["anything"] == {"default": None, "description": "any value at all"}

    def test_no_reference_is_left_dangling(self) -> None:
        """nothing in the result points anywhere.

        :return: none
        :rtype: None
        """
        result = _storyboard()
        assert "$ref" not in repr(result)
        assert "$defs" not in result

    def test_the_top_level_is_an_object_with_the_models_required_list(self) -> None:
        """``type``, ``properties`` and ``required`` are always there.

        :return: none
        :rtype: None
        """
        result = _storyboard()
        assert result["type"] == "object"
        assert result["required"] == ["shots", "scene", "framing"]
        assert "title" not in result
        assert "description" not in result

    def test_a_recursive_model_is_expanded_once_and_its_recursion_named(self) -> None:
        """the point of recursion keeps its type and says in words what it is.

        :return: none
        :rtype: None
        """
        outline = _storyboard()["properties"]["outline"]
        assert outline["type"] == "object"
        assert outline["description"] == "the outline, if there is one"
        children = outline["properties"]["children"]
        assert children["type"] == "array"
        assert children["description"] == "the nodes under this one"
        assert children["items"] == {
            "type": "object",
            "description": "A Node: the same shape as the Node that contains it.",
        }

    def test_the_draft_7_definitions_spelling_is_inlined_too(self) -> None:
        """pydantic 1 and draft-7 schemas keep definitions under ``definitions``.

        :return: none
        :rtype: None
        """
        schema = {
            "type": "object",
            "properties": {"shot": {"$ref": "#/definitions/Shot"}},
            "definitions": {"Shot": {"type": "object", "properties": {"prompt": {"type": "string"}}}},
        }
        assert self_contained_input_schema(schema, tool_name="t")["properties"]["shot"] == {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
        }

    def test_a_schema_with_no_properties_is_still_an_object(self) -> None:
        """an argument-less tool gets empty ``properties`` and ``required``.

        :return: none
        :rtype: None
        """
        assert self_contained_input_schema({}, tool_name="t") == {"type": "object", "properties": {}, "required": []}

    def test_the_input_is_not_mutated(self) -> None:
        """callers keep the schema they passed.

        :return: none
        :rtype: None
        """
        before = copy.deepcopy(_STORYBOARD)
        _storyboard()
        assert _STORYBOARD == before

    def test_a_reference_outside_the_schema_is_refused_naming_the_tool(self) -> None:
        """a foreign reference cannot be inlined, and a left one points at nothing.

        :return: none
        :rtype: None
        """
        schema = {"type": "object", "properties": {"shot": {"$ref": "https://example.com/shot.json"}}}
        with pytest.raises(ValueError, match=r"storyboard.*https://example.com/shot.json"):
            self_contained_input_schema(schema, tool_name="storyboard")

    def test_a_reference_to_a_missing_definition_is_refused_naming_the_tool(self) -> None:
        """a local reference to a definition the schema lacks is refused too.

        :return: none
        :rtype: None
        """
        schema = {"type": "object", "properties": {"shot": {"$ref": "#/$defs/Missing"}}}
        with pytest.raises(ValueError, match=r"storyboard.*#/\$defs/Missing"):
            self_contained_input_schema(schema, tool_name="storyboard")


class TestDeclaredType:
    def test_a_plain_type(self) -> None:
        """a property's own ``type`` answers.

        :return: none
        :rtype: None
        """
        assert declared_type(_STORYBOARD["properties"]["shots"], _STORYBOARD) == "array"

    def test_an_optional_field_answers_with_its_member(self) -> None:
        """``anyOf: [integer, null]`` declares integer.

        :return: none
        :rtype: None
        """
        shot = _STORYBOARD["$defs"]["Shot"]
        assert declared_type(shot["properties"]["seconds"], _STORYBOARD) == "integer"

    def test_a_nullable_type_list_answers_with_the_real_type(self) -> None:
        """``["array", "null"]`` declares array.

        :return: none
        :rtype: None
        """
        assert declared_type({"type": ["array", "null"]}, {}) == "array"

    def test_a_nested_model_answers_with_its_definition(self) -> None:
        """a ``$ref`` is followed into the definitions.

        :return: none
        :rtype: None
        """
        assert declared_type(_STORYBOARD["properties"]["scene"], _STORYBOARD) == "object"

    def test_an_optional_nested_model_answers_with_its_definition(self) -> None:
        """an optional ``$ref`` is followed too.

        :return: none
        :rtype: None
        """
        assert declared_type(_STORYBOARD["properties"]["outline"], _STORYBOARD) == "object"

    def test_a_union_of_real_types_declares_no_single_type(self) -> None:
        """``Close | Wide`` could be either.

        :return: none
        :rtype: None
        """
        assert declared_type(_STORYBOARD["properties"]["framing"], _STORYBOARD) is None

    def test_an_untyped_field_declares_nothing(self) -> None:
        """``Any`` declares no type.

        :return: none
        :rtype: None
        """
        assert declared_type(_STORYBOARD["properties"]["anything"], _STORYBOARD) is None

    def test_a_reference_it_cannot_follow_declares_nothing_and_does_not_raise(self) -> None:
        """a missing or foreign reference answers ``None``.

        :return: none
        :rtype: None
        """
        assert declared_type({"$ref": "#/$defs/Missing"}, {}) is None
        assert declared_type({"$ref": "https://example.com/x.json"}, {}) is None

    def test_a_reference_cycle_ends(self) -> None:
        """a definition that is only a reference to itself answers ``None`` rather than looping.

        :return: none
        :rtype: None
        """
        schema = {"$defs": {"Loop": {"$ref": "#/$defs/Loop"}}}
        assert declared_type({"$ref": "#/$defs/Loop"}, schema) is None


class TestPublicSurface:
    def test_every_exported_name_is_importable(self) -> None:
        """``__all__`` names what the package exports, and each resolves.

        :return: none
        :rtype: None
        """
        import threetears.tool_schema as tool_schema

        assert tool_schema.__all__
        for name in tool_schema.__all__:
            assert callable(getattr(tool_schema, name))
