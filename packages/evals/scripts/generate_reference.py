"""Generate ``packages/evals/docs/reference.md`` from the package itself.

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is not installed.

    # Rewrite the page:
    uv run python packages/evals/scripts/generate_reference.py

    # Exit 1 when the committed page is stale, without writing it:
    uv run python packages/evals/scripts/generate_reference.py --check

    # List the public items that carry no description:
    uv run python packages/evals/scripts/generate_reference.py --gaps

Everything on the page is read from code: each public root's ``__all__`` (the roots are
``threetears.evals.PUBLIC_ROOTS``), first docstring paragraphs and ``#:`` comments, signatures, Pydantic field
descriptions, the goal-check language's module docstring and builtins, the action catalogue, the command line's
argparse definition and ``METRIC_DESCRIPTORS``. Nothing on it is written here except headings and the intro, so a
change to the package moves the page and ``tests/test_reference_doc.py`` fails until it is regenerated.

The output is deterministic: stable ordering, no timestamps, no object addresses, sets sorted.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import enum
import importlib
import inspect
import io
import os
import re
import sys
import textwrap
import types
import typing
from collections.abc import Callable, Iterable, Iterator
from functools import cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

#: The page this script writes.
REFERENCE_PATH = Path(__file__).resolve().parents[1] / "docs" / "reference.md"

#: The command that rewrites the page, as the page and the staleness test tell a reader.
REGENERATE = "uv run python packages/evals/scripts/generate_reference.py"

#: What a public item with no description reads as, so a gap is visible rather than papered over.
NO_DESCRIPTION = "*(no description)*"

#: A first paragraph longer than this is cut to its first sentence for a one-line description.
ONE_LINE_LIMIT = 160

#: The width ``--help`` output is wrapped at on the page.
HELP_WIDTH = 100

#: Gaps found while rendering: (where, name). Filled by :func:`render`.
GAPS: list[tuple[str, str]] = []


# --- text --------------------------------------------------------------------------------------------------------

_ROLE = re.compile(r":[a-z]+(?::[a-z]+)?:`(?:[^`<]*<)?~?([^`>]+)>?`")
_DOUBLE_TICK = re.compile(r"``(.+?)``")


def _role_target(match: re.Match[str]) -> str:
    """A Sphinx cross-reference as a code span: the last dotted part when it was ``~``-shortened."""
    whole = match.group(0)
    target = match.group(1)
    if "`~" in whole:
        target = target.rsplit(".", 1)[-1]
    return f"`{target}`"


def md_text(text: str) -> str:
    """Docstring reStructuredText as Markdown: roles and double backticks become code spans, ``<`` is escaped."""
    text = _ROLE.sub(_role_target, text)
    text = _DOUBLE_TICK.sub(lambda m: f"`{m.group(1)}`", text)
    parts = re.split(r"(`[^`]*`)", text)
    return "".join(p if p.startswith("`") else p.replace("<", "&lt;") for p in parts)


def cell(text: str) -> str:
    """Text safe inside a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def paragraphs(doc: str | None) -> list[str]:
    """A docstring's paragraphs, each joined onto one line."""
    if not doc:
        return []
    return [" ".join(line.strip() for line in p.splitlines()) for p in inspect.cleandoc(doc).split("\n\n") if p.strip()]


_SENTENCE_END = re.compile(r"(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)(?<!\betc)\.(?=\s+[A-Z`*(])")


def first_sentence(text: str) -> str:
    """The first sentence of ``text``, or all of it when it is one."""
    match = _SENTENCE_END.search(text)
    return text[: match.end()] if match else text


def one_line(doc: str | None) -> str:
    """A one-line description: the first paragraph, or its first sentence when that paragraph runs long."""
    paras = paragraphs(doc)
    if not paras:
        return ""
    first = paras[0]
    return first if len(first) <= ONE_LINE_LIMIT else first_sentence(first)


def stable_repr(value: Any) -> str:
    """A ``repr`` that is the same on every run: sets sorted, object addresses dropped."""
    if isinstance(value, (set, frozenset)):
        if not value:
            return f"{type(value).__name__}()"
        return "{" + ", ".join(sorted(stable_repr(v) for v in value)) + "}"
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    text = repr(value)
    return "…" if re.search(r" at 0x[0-9a-fA-F]+", text) else text


# --- source ------------------------------------------------------------------------------------------------------


@cache
def _module_ast(module_name: str) -> tuple[ast.Module, list[str]] | None:
    module = importlib.import_module(module_name)
    try:
        source = inspect.getsource(module)
    except OSError, TypeError:
        return None
    return ast.parse(source), source.splitlines()


def _assigned_names(node: ast.stmt) -> list[str]:
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    if isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name):
        return [node.name.id]
    return []


def _comment_doc(lines: list[str], lineno: int) -> str:
    """The ``#:`` comment block directly above line ``lineno`` (1-based), as a docstring."""
    found: list[str] = []
    i = lineno - 2
    while i >= 0 and lines[i].strip().startswith("#:"):
        found.append(lines[i].strip()[2:].removeprefix(" "))
        i -= 1
    return "\n".join(reversed(found))


def _statement_docs(body: list[ast.stmt], lines: list[str]) -> dict[str, tuple[str, ast.stmt]]:
    """Each name a statement list assigns → (its ``#:`` comment or attribute docstring, the statement)."""
    docs: dict[str, tuple[str, ast.stmt]] = {}
    for i, node in enumerate(body):
        for name in _assigned_names(node):
            doc = _comment_doc(lines, node.lineno)
            if not doc and i + 1 < len(body):
                nxt = body[i + 1]
                if (
                    isinstance(nxt, ast.Expr)
                    and isinstance(nxt.value, ast.Constant)
                    and isinstance(nxt.value.value, str)
                ):
                    doc = nxt.value.value
            docs[name] = (doc, node)
    return docs


def _resolve_relative(module_name: str, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    package = importlib.import_module(module_name)
    base = module_name if hasattr(package, "__path__") else module_name.rsplit(".", 1)[0]
    for _ in range(node.level - 1):
        base = base.rsplit(".", 1)[0]
    return f"{base}.{node.module}" if node.module else base


def defining_module(module_name: str, name: str, depth: int = 0) -> tuple[str, str] | None:
    """The module that assigns ``name`` and the name it is assigned under, following ``from … import`` re-exports."""
    parsed = _module_ast(module_name)
    if parsed is None or depth > 10:
        return None
    tree, lines = parsed
    if name in _statement_docs(tree.body, lines):
        return module_name, name
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    return defining_module(_resolve_relative(module_name, node), alias.name, depth + 1)
    return None


def assignment(module_name: str, name: str) -> tuple[str, ast.stmt | None, list[str]]:
    """A module-level name's docstring, its assigning statement and its module's lines."""
    found = defining_module(module_name, name)
    if found is None:
        return "", None, []
    where, real = found
    parsed = _module_ast(where)
    assert parsed is not None
    tree, lines = parsed
    doc, node = _statement_docs(tree.body, lines)[real]
    return doc, node, lines


def _source_of(node: ast.expr | None) -> str:
    return " ".join(ast.unparse(node).split()) if node is not None else ""


def class_field_sources(cls: type) -> dict[str, tuple[str, str]]:
    """Each annotated class attribute → (its annotation as written, its docstring), across the class's own MRO."""
    found: dict[str, tuple[str, str]] = {}
    for klass in reversed(cls.__mro__):
        if klass.__module__.split(".")[0] != "threetears":
            continue
        try:
            source = inspect.getsource(klass)
        except OSError, TypeError:
            continue
        tree = ast.parse(textwrap.dedent(source))
        node = tree.body[0]
        assert isinstance(node, ast.ClassDef)
        docs = _statement_docs(node.body, textwrap.dedent(source).splitlines())
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                found[stmt.target.id] = (_source_of(stmt.annotation), docs.get(stmt.target.id, ("", stmt))[0])
    return found


def attributes_section(doc: str | None) -> dict[str, str]:
    """The ``Attributes:`` section of a Google-style docstring: name → its description, joined onto one line."""
    if not doc:
        return {}
    lines = inspect.cleandoc(doc).splitlines()
    out: dict[str, str] = {}
    inside = False
    current: str | None = None
    for line in lines:
        if line.strip() == "Attributes:":
            inside = True
            continue
        if not inside:
            continue
        if line and not line.startswith(" "):
            break
        match = re.match(r"^ {4}(\w+)(?: \([^)]*\))?: ?(.*)$", line)
        if match:
            current = match.group(1)
            out[current] = match.group(2).strip()
        elif current and line.strip():
            out[current] += " " + line.strip()
    return out


# --- objects -----------------------------------------------------------------------------------------------------


def own_doc(obj: Any) -> str | None:
    """The docstring an object declares itself, never one it inherits or a dataclass invents."""
    if isinstance(obj, type):
        doc = obj.__dict__.get("__doc__")
        if not doc or doc.startswith(f"{obj.__name__}(") or doc == "An enumeration.":
            return None
        return typing.cast(str, doc)
    if inspect.isroutine(obj):
        return typing.cast("str | None", getattr(obj, "__doc__", None))
    return None


def is_alias(obj: Any) -> bool:
    """Whether ``obj`` is a type alias (a ``Literal``, a union, an ``Annotated`` or a parametrised generic)."""
    return typing.get_origin(obj) is not None or isinstance(obj, (types.UnionType, typing.TypeAliasType))


def class_kind(cls: type) -> str:
    if issubclass(cls, BaseException):
        return "exception"
    if issubclass(cls, enum.Enum):
        return "enum"
    if issubclass(cls, BaseModel):
        return "model"
    if typing.is_protocol(cls):
        return "protocol"
    if typing.is_typeddict(cls):
        return "typed dict"
    if dataclasses.is_dataclass(cls):
        return "dataclass"
    return "class"


def _format_annotation(annotation: Any) -> str:
    if isinstance(annotation, str):
        return annotation
    return inspect.formatannotation(annotation)


def signature(obj: Callable[..., Any]) -> str:
    """``name(params) -> return``, annotations as written, defaults rendered deterministically."""
    try:
        sig = inspect.signature(obj)
    except TypeError, ValueError:
        return ""
    parts: list[str] = []
    seen_kw_marker = False
    for param in sig.parameters.values():
        if param.kind is param.KEYWORD_ONLY and not seen_kw_marker:
            parts.append("*")
            seen_kw_marker = True
        prefix = {param.VAR_POSITIONAL: "*", param.VAR_KEYWORD: "**"}.get(param.kind, "")
        if param.kind is param.VAR_POSITIONAL:
            seen_kw_marker = True
        text = prefix + param.name
        if param.annotation is not param.empty:
            text += f": {_format_annotation(param.annotation)}"
        if param.default is not param.empty:
            text += (
                f" = {stable_repr(param.default)}"
                if param.annotation is not param.empty
                else f"={stable_repr(param.default)}"
            )
        parts.append(text)
    positional_only = [p for p in sig.parameters.values() if p.kind is p.POSITIONAL_ONLY]
    if positional_only:
        parts.insert(len(positional_only), "/")
    out = f"{obj.__name__}({', '.join(parts)})"
    if sig.return_annotation is not sig.empty:
        out += f" -> {_format_annotation(sig.return_annotation)}"
    return out


@dataclasses.dataclass(frozen=True)
class Item:
    """One public name, as the API section lists it."""

    name: str
    group: Literal["Functions", "Classes", "Types", "Constants"]
    kind: str
    description: str
    detail: str


def describe(root: str, name: str, obj: Any) -> Item:
    """How one public name reads on the page."""
    if isinstance(obj, type) and not is_alias(obj) and obj.__module__.startswith("threetears."):
        doc = own_doc(obj)
        detail = ""
        if issubclass(obj, enum.Enum):
            detail = "values: " + ", ".join(f"`{m.value!r}`" for m in obj)
        return Item(name, "Classes", class_kind(obj), one_line(doc), detail)
    if inspect.isroutine(obj):
        kind = "async function" if inspect.iscoroutinefunction(obj) else "function"
        return Item(name, "Functions", kind, one_line(own_doc(obj)), signature(obj))
    doc, node, _ = assignment(root, name)
    if is_alias(obj) or isinstance(obj, type):
        if typing.get_origin(obj) is Literal:
            detail = " | ".join(f"`{stable_repr(v)}`" for v in typing.get_args(obj))
            return Item(name, "Types", "literal", one_line(doc), detail)
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign, ast.TypeAlias)) else None
        return Item(name, "Types", "type alias", one_line(doc), f"`{_source_of(value)}`" if value is not None else "")
    if isinstance(obj, typing.TypeVar):
        return Item(name, "Types", "type variable", one_line(doc), "")
    simple = isinstance(obj, (str, int, float, bool, type(None), re.Pattern))
    text = stable_repr(obj.pattern if isinstance(obj, re.Pattern) else obj) if simple else ""
    detail = f"`= {text}`" if simple and len(text) <= 80 else ""
    return Item(name, "Constants", f"constant ({type(obj).__name__})", one_line(doc), detail)


# --- sections ----------------------------------------------------------------------------------------------------


def anchor(slug: str) -> str:
    return f'<a id="{slug}"></a>'


def root_slug(root: str) -> str:
    return "api-" + root.removeprefix("threetears.evals.").replace(".", "-")


def api_section() -> Iterator[str]:
    from threetears.evals import PUBLIC_ROOTS

    yield anchor("public-api")
    yield "## Public API"
    yield ""
    yield (
        "Import only from these roots, and only the names below: a module under a root is internal and may move. "
        "Within a root, names are grouped as functions, classes, types and constants, each sorted by name. A "
        "class's kind says what it is: a `model` is a Pydantic model, a `protocol` is something a host implements."
    )
    yield ""
    for root in PUBLIC_ROOTS:
        yield f"- [`{root}`](#{root_slug(root)})"
    yield ""
    first_root: dict[int, str] = {}
    for root in PUBLIC_ROOTS:
        module = importlib.import_module(root)
        yield anchor(root_slug(root))
        yield f"### `{root}`"
        yield ""
        intro = paragraphs(module.__doc__)
        if intro:
            yield md_text(intro[0])
            yield ""
        items: dict[str, list[Item]] = {}
        again: list[tuple[str, str]] = []
        for name in sorted(module.__all__, key=lambda n: (n.casefold(), n)):
            obj = getattr(module, name)
            if isinstance(obj, (type, types.FunctionType)) and id(obj) in first_root:
                again.append((name, first_root[id(obj)]))
                continue
            if isinstance(obj, (type, types.FunctionType)):
                first_root[id(obj)] = root
            item = describe(root, name, obj)
            if not item.description:
                GAPS.append((root, name))
            items.setdefault(item.group, []).append(item)
        for group in ("Functions", "Classes", "Types", "Constants"):
            if group not in items:
                continue
            yield f"**{group}**"
            yield ""
            for item in items[group]:
                description = md_text(item.description) if item.description else NO_DESCRIPTION
                yield f"- **`{item.name}`** · {item.kind} · {description}"
                if item.detail:
                    detail = item.detail if item.detail.startswith(("`", "values")) else f"`{item.detail}`"
                    yield f"  <br>{detail}"
            yield ""
        if again:
            yield "**Also exported here**"
            yield ""
            yield ", ".join(f"`{n}` ([`{r}`](#{root_slug(r)}))" for n, r in again)
            yield ""


def _field_rows(model: type, *, full: bool) -> Iterator[tuple[str, str, str, str]]:
    """(name, type, default, description) for each field of a Pydantic model or a dataclass, in declaration order."""
    sources = class_field_sources(model)
    from_attributes = attributes_section(model.__doc__)
    describe_text = (lambda d: paragraphs(d)[0] if paragraphs(d) else "") if full else one_line
    if issubclass(model, BaseModel):
        schema = model.model_json_schema(mode="serialization").get("properties", {})
        for name, info in model.model_fields.items():
            key = info.serialization_alias or info.alias or name
            text = info.description or schema.get(key, {}).get("description") or from_attributes.get(name, "")
            text = text or sources.get(name, ("", ""))[1]
            if info.is_required():
                default = "required"
            elif info.default_factory is not None:
                default = stable_repr(info.default_factory())  # type: ignore[call-arg]
            else:
                default = stable_repr(info.default) if info.default is not PydanticUndefined else "required"
            yield name, sources.get(name, (_format_annotation(info.annotation), ""))[0], default, describe_text(text)
        return
    assert dataclasses.is_dataclass(model)
    for f in dataclasses.fields(model):
        annotation, attr_doc = sources.get(f.name, (str(f.type), ""))
        text = attr_doc or from_attributes.get(f.name, "")
        if not f.init:
            default = "derived"
        elif f.default is not dataclasses.MISSING:
            default = stable_repr(f.default)
        elif f.default_factory is not dataclasses.MISSING:
            factory = f.default_factory
            default = f"{getattr(factory, '__name__', 'factory')}()"
        else:
            default = "required"
        yield f.name, annotation, default, describe_text(text)


def field_table(model: type, *, full: bool) -> Iterator[str]:
    yield "| Field | Type | Default | Description |"
    yield "|---|---|---|---|"
    for name, annotation, default, text in _field_rows(model, full=full):
        if not text:
            GAPS.append((f"{model.__name__} field", name))
        description = cell(md_text(text)) if text else NO_DESCRIPTION
        shown = default if default in ("required", "derived") else f"`{cell(default)}`"
        yield f"| `{name}` | `{cell(annotation)}` | {shown} | {description} |"
    yield ""


def configuration_section() -> Iterator[str]:
    from threetears.evals.contracts.host import HostProfile
    from threetears.evals.run import LaunchSettings

    yield anchor("configuration")
    yield "## Configuration"
    yield ""
    for slug, model in (("launch-settings", LaunchSettings), ("host-profile", HostProfile)):
        yield anchor(slug)
        yield f"### `{model.__name__}`"
        yield ""
        yield md_text(paragraphs(model.__doc__)[0])
        yield ""
        yield from field_table(model, full=True)


def documents_section() -> Iterator[str]:
    from threetears.evals.analysis import (
        AnalysisContextBundle,
        ChartBlock,
        DisclosureBlock,
        Report,
        ReportSource,
        TableBlock,
        TextBlock,
    )

    yield anchor("documents")
    yield "## The report and the analysis bundle"
    yield ""
    yield anchor("report")
    yield "### `Report`"
    yield ""
    yield md_text(paragraphs(Report.__doc__)[0])
    yield ""
    yield from field_table(Report, full=False)
    yield "Its `source`, and each kind of block in `blocks`:"
    yield ""
    for part in (ReportSource, TextBlock, TableBlock, ChartBlock, DisclosureBlock):
        yield f"#### `{part.__name__}`"
        yield ""
        summary = one_line(own_doc(part))
        if summary:
            yield md_text(summary)
            yield ""
        yield from field_table(part, full=False)
    yield anchor("analysis-bundle")
    yield "### `AnalysisContextBundle`"
    yield ""
    yield md_text(paragraphs(AnalysisContextBundle.__doc__)[0])
    yield ""
    yield "Its top-level fields, in declaration order; each one's type is described in the API section."
    yield ""
    yield from field_table(AnalysisContextBundle, full=False)


_SECTION_UNDERLINE = re.compile(r"^-{3,}$")


def rst_to_markdown(doc: str, heading: str) -> Iterator[str]:
    """A module docstring's body (after its summary line) as Markdown: sections, ``::`` blocks and roles."""
    lines = inspect.cleandoc(doc).splitlines()[1:]
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines) and _SECTION_UNDERLINE.match(lines[i + 1]) and line.strip():
            yield f"{heading} {line.strip()}"
            i += 2
            continue
        if line.rstrip().endswith("::"):
            yield md_text(line.rstrip()[:-1])
            yield ""
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            block: list[str] = []
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith("    ")):
                block.append(lines[i][4:])
                i += 1
            while block and not block[-1].strip():
                block.pop()
            yield "```python"
            yield from block
            yield "```"
            yield ""
            continue
        yield md_text(line)
        i += 1


def goal_language_section() -> Iterator[str]:
    from threetears.evals.contracts import dsl

    yield anchor("goal-checks")
    yield "## The goal-check language"
    yield ""
    yield md_text(paragraphs(dsl.__doc__)[0])
    yield ""
    yield from rst_to_markdown(typing.cast(str, dsl.__doc__), "####")
    yield ""
    yield anchor("goal-check-functions")
    yield "### Functions"
    yield ""
    yield "| Function | Arguments | Description |"
    yield "|---|---|---|"
    for name, arguments, description in goal_functions(inspect.getsource(dsl)):
        yield f"| `{name}` | `{arguments}` | {cell(md_text(description))} |"
    yield ""


#: The handler a goal-check function's description is read from, where it is not ``_builtin_<name>``.
_SHARED_HANDLERS = {
    "any": "_eval_any_all",
    "all": "_eval_any_all",
    "fired": "_builtin_fired",
    "fired_armed": "_builtin_fired",
}

#: Handler parameters the evaluator supplies, which an expression never writes.
_SUPPLIED = {"ledger", "ctx", "predicate", "call", "func_name", "node"}


def goal_functions(source: str) -> list[tuple[str, str, str]]:
    """(name, arguments, description) for each function the goal-check language admits, read from its source.

    Read from the module's text, as a documentation generator reads it, rather than by importing the module's
    private names: the function set is the frozenset the parser admits calls against, and each description is
    the first line of the handler that evaluates it. A function with no handler to read stops the generator.
    """
    tree = ast.parse(source)
    names: list[str] = []
    handlers: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            handlers[node.name] = node
        elif "_BUILTINS" in _assigned_names(node) and isinstance(node, ast.Assign):
            names = sorted(ast.literal_eval(node.value.args[0]))  # type: ignore[attr-defined]
    if not names:
        raise SystemExit("the goal-check language's function set (_BUILTINS in contracts/dsl.py) was not found")
    rows = []
    for name in names:
        handler = handlers.get(_SHARED_HANDLERS.get(name, f"_builtin_{name}"))
        if handler is None:
            raise SystemExit(f"goal-check function {name!r} has no handler in contracts/dsl.py to document it from")
        if name in ("any", "all"):
            arguments = "<body> for it in <path>"
        else:
            arguments = ", ".join(a.arg for a in handler.args.args if a.arg not in _SUPPLIED)
        rows.append((name, arguments, one_line(ast.get_docstring(handler))))
    return rows


def _json_type(schema: dict[str, Any]) -> str:
    if "enum" in schema:
        return " | ".join(f"`{v!r}`" for v in schema["enum"])
    if "anyOf" in schema:
        return " or ".join(_json_type(s) for s in schema["anyOf"])
    kind = schema.get("type", "any")
    if kind == "array":
        return f"array of {_json_type(schema.get('items', {}))}"
    return f"`{kind}`"


def actions_section() -> Iterator[str]:
    from threetears.evals.actions import PERMISSION_CLASSES, engine_actions, standard_tools

    actions = engine_actions()
    yield anchor("actions")
    yield "## The action catalogue (MCP)"
    yield ""
    yield (
        "Every engine action, as every transport mounts it (the FastMCP tools, a host's own CLI or REST mapping). "
        "A mounted tool takes `action` plus the union of its actions' parameters; `help` is generated. A "
        "parameter marked `?` is optional. `job` marks long work: it returns jobs to poll with `job_poll`. A "
        "host may contribute actions of its own, which are not listed here."
    )
    yield ""
    yield "| Tool (default prefix) | Classes it mounts | Description |"
    yield "|---|---|---|"
    for tool in standard_tools():
        classes = ", ".join(f"`{c}`" for c in PERMISSION_CLASSES if c in tool.permissions)
        yield f"| `{tool.name}` | {classes} | {cell(md_text(tool.description))} |"
    yield ""
    params: dict[tuple[str, str, str], list[str]] = {}
    workflows: dict[str, list[Any]] = {}
    for action in actions:
        workflows.setdefault(action.workflow, []).append(action)
    for workflow, members in workflows.items():
        yield f"### {workflow}"
        yield ""
        yield "| Action | Class | Parameters | Description |"
        yield "|---|---|---|---|"
        for action in members:
            schema = action.params.model_json_schema()
            names = []
            for pname, info in action.params.model_fields.items():
                names.append(f"`{pname}`" if info.is_required() else f"`{pname}?`")
                prop = schema["properties"][pname]
                params.setdefault((pname, _json_type(prop), info.description or ""), []).append(action.name)
            permission = f"`{action.permission}`" + (", job" if action.long_running else "")
            yield (f"| `{action.name}` | {permission} | {', '.join(names) or '—'} | {cell(md_text(action.summary))} |")
        yield ""
    yield anchor("action-parameters")
    yield "### Parameters"
    yield ""
    yield "| Parameter | Type | Description |"
    yield "|---|---|---|"
    for (pname, ptype, description), used_by in sorted(params.items()):
        shared = (pname for (other, _, _) in params if other == pname)
        note = f" (on {', '.join(f'`{a}`' for a in used_by)})" if sum(1 for _ in shared) > 1 else ""
        yield f"| `{pname}`{note} | {cell(ptype)} | {cell(md_text(description))} |"
    yield ""


def cli_section() -> Iterator[str]:
    from threetears.evals.quick import cli

    yield anchor("cli")
    yield "## The command line"
    yield ""
    yield f"`{cli.DEFAULT_PROG} <command> --host MODULE:FACTORY --scope SCOPE [options]`"
    yield ""
    yield (
        "A product mounting the commands under its own CLI (`run_cli(host_factory=...)`) drops `--host`. "
        "Every command takes `--scope`."
    )
    yield ""
    exit_codes = [p for p in paragraphs(cli.__doc__) if p.startswith("Exit codes")]
    if not exit_codes:
        raise SystemExit(
            "the CLI module docstring no longer states its exit codes in a paragraph starting 'Exit codes'"
        )
    yield md_text(exit_codes[0])
    yield ""
    yield "Each command's options, as its `--help` prints them:"
    yield ""
    for name in cli.ENGINE_COMMANDS:
        yield f"### `{name}`"
        yield ""
        yield "```text"
        yield command_help(cli.build_parser(cli.DEFAULT_PROG), name)
        yield "```"
        yield ""


def command_help(parser: argparse.ArgumentParser, command: str) -> str:
    """What ``<command> --help`` prints, at a fixed width and without colour, so it is the same on every machine."""
    saved = {key: os.environ.get(key) for key in ("COLUMNS", "NO_COLOR", "FORCE_COLOR", "PYTHON_COLORS")}
    os.environ.update({"COLUMNS": str(HELP_WIDTH), "NO_COLOR": "1"})
    for key in ("FORCE_COLOR", "PYTHON_COLORS"):
        os.environ.pop(key, None)
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(printed), contextlib.suppress(SystemExit):
            parser.parse_args([command, "--help"])
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return printed.getvalue().rstrip()


def measures_section() -> Iterator[str]:
    from threetears.evals.contracts import METRIC_DESCRIPTORS

    yield anchor("measures")
    yield "## Measures"
    yield ""
    yield (
        "The engine's own measures (`METRIC_DESCRIPTORS`), grouped by family in the order the engine declares "
        "them. Direction is which end is better (`—` for a coordinate, condition or raw count); scale says "
        "whether a relative change means anything (`ratio`) or only a difference does (`interval`), `—` when "
        "undeclared. A host "
        "declares its own measures on its `MeasureRegistry`, and a template mints one per rubric dimension and "
        "goal check; neither is listed here."
    )
    yield ""
    families: dict[str, list[Any]] = {}
    for descriptor in METRIC_DESCRIPTORS.values():
        families.setdefault(descriptor.family or "unclassified", []).append(descriptor)
    for family, members in families.items():
        yield f"### {family}"
        yield ""
        yield "| Measure | Direction | Scale | Unit | Description |"
        yield "|---|---|---|---|---|"
        for d in members:
            direction = {True: "higher", False: "lower", None: "—"}[d.higher_is_better]
            flags = [label for label, on in (("diagnostic", d.diagnostic), ("guardrail", d.guardrail)) if on]
            if flags:
                direction += f" ({', '.join(flags)})"
            yield (
                f"| `{d.name}` | {direction} | {d.scale or '—'} | {d.unit or '—'} | "
                f"{cell(md_text(first_sentence(d.description)))} |"
            )
        yield ""


SECTIONS: tuple[tuple[str, str, Callable[[], Iterable[str]]], ...] = (
    ("public-api", "Public API", api_section),
    ("configuration", "Configuration", configuration_section),
    ("documents", "The report and the analysis bundle", documents_section),
    ("goal-checks", "The goal-check language", goal_language_section),
    ("actions", "The action catalogue (MCP)", actions_section),
    ("cli", "The command line", cli_section),
    ("measures", "Measures", measures_section),
)


def render() -> str:
    """The whole page."""
    GAPS.clear()
    out = [
        "# Reference",
        "",
        "**For:** people who already know the engine and agents that need exact names. It answers: what does "
        "this root export, what does this setting or field mean, what can a goal check say, which actions and "
        "commands exist, and what does each measure measure. To learn the engine, start with the "
        "[tutorial](tutorial.md) instead.",
        "",
        f"**Generated** from the package by `packages/evals/scripts/generate_reference.py`. Do not edit it by "
        f"hand: change the docstring, field description or definition it came from, then run `{REGENERATE}`. "
        "`tests/test_reference_doc.py` fails while this page is stale.",
        "",
        "## Contents",
        "",
    ]
    out += [f"- [{title}](#{slug})" for slug, title, _ in SECTIONS]
    out.append("")
    for _, _, section in SECTIONS:
        out += list(section())
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).rstrip() + "\n"
    return text


def main(argv: list[str] | None = None) -> int:
    """Write the page, check it, or list its gaps."""
    parser = argparse.ArgumentParser(description=typing.cast(str, __doc__).splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="exit 1 when the committed page is stale; write nothing")
    mode.add_argument("--gaps", action="store_true", help="list the public items that carry no description")
    args = parser.parse_args(argv)
    page = render()
    if args.gaps:
        for where, name in GAPS:
            print(f"{where}: {name}")
        return 0
    if args.check:
        current = REFERENCE_PATH.read_text(encoding="utf-8") if REFERENCE_PATH.exists() else ""
        if current != page:
            print(f"{REFERENCE_PATH} is stale; run: {REGENERATE}", file=sys.stderr)
            return 1
        return 0
    REFERENCE_PATH.write_text(page, encoding="utf-8")
    print(f"wrote {REFERENCE_PATH} ({len(page.encode())} bytes, {page.count(chr(10))} lines, {len(GAPS)} gaps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
