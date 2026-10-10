"""The public surface of ``3tears-evals``, walked once for every tool that reads it.

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is not installed. Two generators
read it. ``generate_reference.py`` renders the walk as ``docs/reference.md`` for a reader;
``generate_api_snapshot.py`` records it as ``api/surface.json``, the checked-in copy ``tests/test_api_surface.py``
holds the installed package to. They share this one walk so the page a reader trusts and the snapshot a test
trusts never disagree about what a public name is, its kind or its signature.

The walk starts at each public root's ``__all__`` (the roots are ``threetears.evals.PUBLIC_ROOTS``) and reads
everything else from code: first docstring paragraphs and ``#:`` comments, signatures, Pydantic and dataclass
fields with their annotations as written. Everything it renders is deterministic: stable ordering, no
timestamps, no object addresses, sets sorted.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import importlib
import inspect
import re
import textwrap
import types
import typing
from collections.abc import Callable, Iterator
from functools import cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

#: The package directory: what a path in a rendered default is written relative to.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

#: A first paragraph longer than this is cut to its first sentence for a one-line description.
ONE_LINE_LIMIT = 160


# --- text --------------------------------------------------------------------------------------------------------


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
    """A ``repr`` that is the same on every run and checkout: sets sorted, object addresses dropped, and a path
    inside the package written relative to it, since an absolute one names the checkout the page was made in and
    fails the staleness test in every other."""
    if isinstance(value, Path) and value.is_absolute() and value.is_relative_to(PACKAGE_ROOT):
        return f"Path('<package>/{value.relative_to(PACKAGE_ROOT).as_posix()}')"
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


def assigned_names(node: ast.stmt) -> list[str]:
    """The names one module- or class-level statement assigns."""
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
        for name in assigned_names(node):
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


def source_of(node: ast.expr | None) -> str:
    """An expression as written, on one line."""
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
                found[stmt.target.id] = (source_of(stmt.annotation), docs.get(stmt.target.id, ("", stmt))[0])
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
    """What a public class is, in one word or two: what a reader and the snapshot both call it."""
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


#: A dotted path into the package, as an evaluated annotation prints a class: ``threetears.evals.schema.run.EvalRun``.
_ENGINE_PATH = re.compile(r"\bthreetears\.evals(?:\.\w+)*\.(\w+)")


def _format_annotation(annotation: Any) -> str:
    """An annotation as a reader names it: as written when it is a string, and otherwise with any engine class
    written by its bare name, because the module below a root that defines it is internal and may move."""
    text = annotation if isinstance(annotation, str) else inspect.formatannotation(annotation)
    return _ENGINE_PATH.sub(r"\1", text)


def signature(obj: Callable[..., Any], *, name: str | None = None) -> str:
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
    out = f"{name or obj.__name__}({', '.join(parts)})"
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
        return Item(name, "Types", "type alias", one_line(doc), f"`{source_of(value)}`" if value is not None else "")
    if isinstance(obj, typing.TypeVar):
        return Item(name, "Types", "type variable", one_line(doc), "")
    simple = isinstance(obj, (str, int, float, bool, type(None), re.Pattern))
    text = stable_repr(obj.pattern if isinstance(obj, re.Pattern) else obj) if simple else ""
    detail = f"`= {text}`" if simple and len(text) <= 80 else ""
    return Item(name, "Constants", f"constant ({type(obj).__name__})", one_line(doc), detail)


# --- fields ------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FieldRow:
    """One field of a Pydantic model or a dataclass, as declared.

    Attributes:
        name: The field's attribute name.
        annotation: Its annotation as written in the source.
        default: ``"required"``, ``"derived"`` (a dataclass field outside ``__init__``), or the default rendered.
        factory: The default factory, when the default is made rather than stated; ``default`` is then empty.
        description: Its description: the field's own, the schema's, the ``Attributes:`` entry or its docstring.
    """

    name: str
    annotation: str
    default: str
    factory: Callable[[], Any] | None
    description: str


def field_rows(model: type) -> Iterator[FieldRow]:
    """Each field of a Pydantic model or a dataclass, in declaration order."""
    sources = class_field_sources(model)
    from_attributes = attributes_section(model.__doc__)
    if issubclass(model, BaseModel):
        schema = model.model_json_schema(mode="serialization").get("properties", {})
        for name, info in model.model_fields.items():
            key = info.serialization_alias or info.alias or name
            text = info.description or schema.get(key, {}).get("description") or from_attributes.get(name, "")
            text = text or sources.get(name, ("", ""))[1]
            factory = typing.cast("Callable[[], Any] | None", info.default_factory)
            if info.is_required() or (factory is None and info.default is PydanticUndefined):
                default = "required"
            else:
                default = "" if factory is not None else stable_repr(info.default)
            annotation = sources.get(name, (_format_annotation(info.annotation), ""))[0]
            yield FieldRow(name, annotation, default, factory, text)
        return
    assert dataclasses.is_dataclass(model)
    for f in dataclasses.fields(model):
        annotation, attr_doc = sources.get(f.name, (str(f.type), ""))
        text = attr_doc or from_attributes.get(f.name, "")
        made: Callable[[], Any] | None = None
        if not f.init:
            default = "derived"
        elif f.default is not dataclasses.MISSING:
            default = stable_repr(f.default)
        elif f.default_factory is not dataclasses.MISSING:
            default, made = "", f.default_factory
        else:
            default = "required"
        yield FieldRow(f.name, annotation, default, made, text)


def factory_name(factory: Callable[[], Any]) -> str:
    """A default factory as a call, ``list()``: what is made, never a value one call happened to make."""
    return f"{getattr(factory, '__name__', 'factory')}()"


# --- the surface -------------------------------------------------------------------------------------------------


def public_names() -> Iterator[tuple[str, str, Any]]:
    """(root, name, object) for every name in every public root's ``__all__``: roots in ``PUBLIC_ROOTS`` order, names
    sorted as the reference lists them."""
    from threetears.evals import PUBLIC_ROOTS

    for root in PUBLIC_ROOTS:
        module = importlib.import_module(root)
        for name in sorted(module.__all__, key=lambda n: (n.casefold(), n)):
            yield root, name, getattr(module, name)


@dataclasses.dataclass(frozen=True)
class SurfaceEntry:
    """One public name as the snapshot records it: what it is and how it is called or built.

    Attributes:
        kind: The reference's kind: ``model``, ``protocol``, ``function``, ``literal``, ``constant (int)`` and so on.
        signature: How a caller uses it: a function's signature; a model's, dataclass's or typed dict's fields
            with their annotations as written and their defaults; a protocol's members; an enum's or a literal's
            values; a type alias's definition; a class's or exception's constructor. Empty for a constant, whose
            value is data rather than a contract (a schema version moves on every bump).
    """

    kind: str
    signature: str


def _fields_signature(cls: type) -> str:
    parts = []
    for row in field_rows(cls):
        text = f"{row.name}: {row.annotation}"
        if row.factory is not None:
            text += f" = {factory_name(row.factory)}"
        elif row.default not in ("required", "derived"):
            text += f" = {row.default}"
        elif row.default == "derived":
            text += " (derived)"
        parts.append(text)
    return f"{cls.__name__}({', '.join(parts)})"


def _typed_dict_signature(cls: type) -> str:
    sources = class_field_sources(cls)
    required = typing.cast("frozenset[str]", getattr(cls, "__required_keys__", frozenset()))
    parts = [f"{name}{'' if name in required else '?'}: {sources[name][0]}" for name in sources]
    return f"{cls.__name__}({', '.join(parts)})"


def _protocol_signature(cls: type) -> str:
    """A protocol's members, sorted: each method's signature, each attribute's annotation as written."""
    sources = class_field_sources(cls)
    members = []
    for name in sorted(typing.get_protocol_members(cls)):
        if name in sources:
            members.append(f"{name}: {sources[name][0]}")
            continue
        member = inspect.getattr_static(cls, name)
        if isinstance(member, (staticmethod, classmethod)):
            member = member.__func__
        if isinstance(member, property):
            members.append(f"{name}: {_format_annotation(inspect.signature(member.fget).return_annotation)}")
            continue
        members.append(signature(member, name=name) if callable(member) else name)
    return f"{cls.__name__}{{{'; '.join(members)}}}"


def surface_entry(root: str, name: str, obj: Any) -> SurfaceEntry:
    """What the snapshot records for one public name."""
    item = describe(root, name, obj)
    if item.group == "Classes":
        if item.kind in ("model", "dataclass"):
            return SurfaceEntry(item.kind, _fields_signature(obj))
        if item.kind == "typed dict":
            return SurfaceEntry(item.kind, _typed_dict_signature(obj))
        if item.kind == "protocol":
            return SurfaceEntry(item.kind, _protocol_signature(obj))
        if item.kind == "enum":
            return SurfaceEntry(item.kind, item.detail)
        # A constructor the engine wrote is a contract; one inherited from ``Exception`` or ``object`` is not.
        engine_built = any(
            "__init__" in k.__dict__ or "__new__" in k.__dict__
            for k in obj.__mro__
            if k.__module__.startswith("threetears.")
        )
        return SurfaceEntry(item.kind, signature(obj) if engine_built else "")
    if item.group == "Constants":
        return SurfaceEntry(item.kind, "")
    return SurfaceEntry(item.kind, item.detail.strip("`"))


def installed_surface() -> dict[str, SurfaceEntry]:
    """``root.name`` → its entry, for every public name the installed package exports."""
    return {f"{root}.{name}": surface_entry(root, name, obj) for root, name, obj in public_names()}


def exported_twice() -> dict[str, list[str]]:
    """Each name exported from more than one public root → those roots, in ``PUBLIC_ROOTS`` order."""
    roots: dict[str, list[str]] = {}
    for root, name, _ in public_names():
        roots.setdefault(name, []).append(root)
    return {name: where for name, where in sorted(roots.items()) if len(where) > 1}
