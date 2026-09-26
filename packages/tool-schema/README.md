# 3tears-tool-schema

Dependency-free helpers for the JSON Schema a tool advertises for its arguments.

A tool's arguments reach a model as a JSON Schema: pydantic's `model_json_schema()`, an MCP
server's `inputSchema`, a LangChain tool's `args_schema`. Pydantic writes every nested model as a
`$ref` into the schema's `$defs`, and every optional field as `anyOf: [X, {"type": "null"}]`.
Anything that reads only a property's own `type` gets both wrong. A model is shown a string where
the tool wants a list of objects, and an input normaliser leaves `"[]"` as a string where the tool
wants a list.

This package has **no dependencies**, so any tool host can take it without inheriting a framework:
a LangGraph app, an MCP server, a model adapter, a validator.

## `self_contained_input_schema`

One schema per tool, with no references left in it.

```python
from pydantic import BaseModel
from threetears.tool_schema import self_contained_input_schema


class Shot(BaseModel):
    prompt: str
    seconds: int | None = None


class Storyboard(BaseModel):
    shots: list[Shot]


schema = self_contained_input_schema(Storyboard.model_json_schema(), tool_name="storyboard")
# {"type": "object",
#  "properties": {"shots": {"type": "array",
#                           "items": {"type": "object",
#                                     "properties": {"prompt": {"type": "string"},
#                                                    "seconds": {"type": "integer"}},
#                                     "required": ["prompt"]}}},
#  "required": ["shots"]}
```

What it does:

- Inlines every `$ref` into the schema's own `$defs` or `definitions`, through `items`, unions and
  nested properties. Each level keeps its `required` list and descriptions, and a field's own
  description wins over its model's.
- Collapses an optional union (`anyOf: [X, null]`) to `X` at every depth, dropping the
  `default: null` that would contradict `X`.
- Keeps a union of two or more real members whole, rather than choosing one.
- Leaves an untyped (`Any`) field untyped.
- Expands a recursive model until it recurs. The point of recursion keeps the definition's type and
  reads `"A Node: the same shape as the Node that contains it."`, rather than being cut to `{}`.
- Drops titles and the root description; the tool's description travels beside its schema.
- Always returns `type: "object"`, `properties` and `required`.

A `$ref` to anything but the schema's own definitions raises `ValueError` naming the tool and the
reference. A reference left in place would point at nothing the reader has.

### In an MCP server

List a tool whose arguments are a pydantic model, with an `inputSchema` any client can read:

```python
from mcp.types import Tool
from threetears.tool_schema import self_contained_input_schema

tool = Tool(
    name="storyboard",
    description="Plan the shots for a scene.",
    inputSchema=self_contained_input_schema(Storyboard.model_json_schema(), tool_name="storyboard"),
)
```

### In a LangGraph app

Hand a model provider that takes raw tool specs the same self-contained shape:

```python
from langchain_core.tools import StructuredTool
from threetears.tool_schema import self_contained_input_schema

tool = StructuredTool.from_function(plan, name="storyboard", args_schema=Storyboard)
spec = {
    "name": tool.name,
    "description": tool.description,
    "input_schema": self_contained_input_schema(Storyboard.model_json_schema(), tool_name=tool.name),
}
```

## `declared_type`

The one JSON type a property declares, read through the shapes pydantic writes. Use it to coerce
loose input toward the declared type.

```python
from threetears.tool_schema import declared_type

schema = Storyboard.model_json_schema()
declared_type(schema["properties"]["shots"], schema)  # "array"
```

It reads through:

- a nullable type list, such as `["array", "null"]`;
- an optional union;
- a `$ref` into the schema's definitions.

It answers `None` for a union of two or more real types, and for a reference it cannot follow. It
never raises, so code that must not fail on a schema it only half understands can call it.

In 3tears, `3tears-models` shows every bound tool to a Claude subscription model through
`self_contained_input_schema`, and `3tears-agent-tools` coerces tool input with `declared_type`.
