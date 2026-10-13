"""The public surface of ``3tears-evals``, walked once for every tool that reads it.

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is not installed. Two generators
read it. ``generate_reference.py`` renders the walk as ``docs/reference.md`` for a reader;
``generate_api_snapshot.py`` records it as ``api/surface.json``, the checked-in copy ``tests/test_api_surface.py``
holds the installed package to. They share this one walk so the page a reader trusts and the snapshot a test
trusts never disagree about what a public name is, its kind or its signature.

The walk starts at the package's own ``__all__`` and each public root's (the roots are
``threetears.evals.PUBLIC_ROOTS``) and reads everything else from code: first docstring paragraphs and ``#:`` comments, signatures, Pydantic and dataclass
fields with their annotations as written. Everything it renders is deterministic: stable ordering, no
timestamps, no object addresses, sets sorted.

Each name also has a stability tier, the one thing the walk does not read from code: :data:`HOST_CONTRACT` declares
the names a host binds to, and every other public name is ``closure``, public because a public signature reaches
it. The snapshot records each entry's tier, so a promotion or a demotion is a diff in review like any other.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import importlib
import inspect
import json
import re
import subprocess
import textwrap
import tomllib
import types
import typing
from collections.abc import Callable, Iterable, Iterator
from functools import cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
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
    # The installed version is read from the distribution's metadata, not written in the source: printing it would
    # make the page describe the checkout's install rather than the package, and go stale on every release.
    detail = f"`= {text}`" if simple and len(text) <= 80 and name != "__version__" else ""
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


#: The package itself. Its own ``__all__`` (the public roots, and the installed version) is public too and is walked
#: first: a consumer reads both from ``threetears.evals``, and no root holds either.
PACKAGE = "threetears.evals"


def walked_modules() -> tuple[str, ...]:
    """Every module whose ``__all__`` is public: the package itself, then each public root in ``PUBLIC_ROOTS``
    order."""
    from threetears.evals import PUBLIC_ROOTS

    return (PACKAGE, *PUBLIC_ROOTS)


def public_names() -> Iterator[tuple[str, str, Any]]:
    """(root, name, object) for every name in every walked module's ``__all__`` (:func:`walked_modules`): the
    package, then the roots in ``PUBLIC_ROOTS`` order, names sorted as the reference lists them."""
    for root in walked_modules():
        module = importlib.import_module(root)
        for name in sorted(module.__all__, key=lambda n: (n.casefold(), n)):
            yield root, name, getattr(module, name)


#: A public name's stability tier. ``host-contract`` is a name a host binds to (:data:`HOST_CONTRACT`): it moves
#: only with a ``MIGRATIONS.toml`` record and a deprecation cycle. ``closure`` is a name public because a public
#: signature reaches it: it may move in any release, recorded in ``MIGRATIONS.toml`` and never shimmed.
Tier = Literal["host-contract", "closure"]

#: The tiers, in the order the promise weakens.
TIERS: tuple[Tier, ...] = typing.get_args(Tier)


@dataclasses.dataclass(frozen=True)
class SurfaceEntry:
    """One public name as the snapshot records it: what it is, how it is called or built, and what it promises.

    Attributes:
        kind: The reference's kind: ``model``, ``protocol``, ``function``, ``literal``, ``constant (int)`` and so on.
        signature: How a caller uses it: a function's signature; a model's, dataclass's or typed dict's fields
            with their annotations as written and their defaults; a protocol's members; an enum's or a literal's
            values; a type alias's definition; a class's or exception's constructor. Empty for a constant, whose
            value is data rather than a contract (a schema version moves on every bump).
        tier: Its stability tier (:data:`Tier`): ``host-contract`` when :data:`HOST_CONTRACT` lists it, otherwise
            ``closure``, which is what a name nobody has promised is.
    """

    kind: str
    signature: str
    tier: Tier = "closure"


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


# --- tiers: what a host binds to ---------------------------------------------------------------------------------

#: The host contract: every public name a host binds to, by ``root.name``. Each moves only with a
#: ``MIGRATIONS.toml`` record and a deprecation cycle (the old name keeps working beside the new one for one minor
#: release), from 0.66.0 on. Every other public name is ``closure``.
#:
#: The ruling is what a host author is told to write against: ``docs/adopting-a-host.md``, ``docs/tutorial.md``, the
#: README and ``examples/`` (every name they import is here, which ``tests/test_api_tiers.py`` holds), and the
#: courier host (``tests/fixtures/courierhost/``), the least a product writes to be a host. A name a topic guide
#: alone imports, a field type a contract name's signature reaches, and a name a later section of the host guide
#: mentions for an optional feature are ``closure``: promoting one later is free, and demoting one is a break.
HOST_CONTRACT: frozenset[str] = frozenset(
    {
        # The host: the one value every entrypoint takes, and the services it is built from.
        "threetears.evals.kernel.host.EvalHost",
        "threetears.evals.kernel.host.HostProfile",
        "threetears.evals.kernel.host.CompletionClients",
        "threetears.evals.kernel.host.default_cell_timeout",
        "threetears.evals.kernel.EvalStorage",
        "threetears.evals.kernel.withhold_failure_detail",
        "threetears.evals.schema.ProviderFailureDescriber",
        "threetears.evals.schema.CompletionClient",
        "threetears.evals.schema.CompletionResult",
        "threetears.evals.schema.TraceSink",
        # The registries a profile holds: the host's vocabulary of levers, measures, bars and world.
        "threetears.evals.kernel.host.SHARED_CORE",
        "threetears.evals.kernel.host.Sweepable",
        "threetears.evals.kernel.host.SweepableRegistry",
        "threetears.evals.kernel.host.MeasureRegistry",
        "threetears.evals.kernel.MetricDescriptor",
        "threetears.evals.kernel.host.BarRegistry",
        "threetears.evals.kernel.host.Bar",
        "threetears.evals.kernel.host.PassThreshold",
        "threetears.evals.kernel.host.load_bar_registry",
        "threetears.evals.kernel.host.WorldRegistry",
        "threetears.evals.kernel.host.WorldDimension",
        "threetears.evals.kernel.host.SeedRefused",
        # The kind: the seam a host implements to turn a case into a call to its own code, and what crosses it.
        "threetears.evals.kernel.host.KindContract",
        "threetears.evals.kernel.CandidateKind",
        "threetears.evals.kernel.CandidateOutput",
        "threetears.evals.kernel.CandidateTelemetry",
        "threetears.evals.kernel.CandidatePreparationFailed",
        "threetears.evals.kernel.CandidateKindDefect",
        "threetears.evals.kernel.CellSink",
        "threetears.evals.kernel.CellCassettes",
        "threetears.evals.kernel.CellSpanWindow",
        "threetears.evals.kernel.VariantConfig",
        "threetears.evals.kernel.WorldSession",
        "threetears.evals.schema.SubjectSnapshot",
        "threetears.evals.schema.RoleUsage",
        "threetears.evals.schema.SweepableValue",
        "threetears.evals.schema.IntervalScale",
        "threetears.evals.schema.JudgedArtifact",
        "threetears.evals.schema.WorldSeed",
        # Launching: the launch host, a kind's launch entry and wiring, and the calls that start and run a run.
        "threetears.evals.run.LaunchHost",
        "threetears.evals.run.LaunchSettings",
        "threetears.evals.run.LaunchableKind",
        "threetears.evals.run.LaunchRequest",
        "threetears.evals.run.KindLauncher",
        "threetears.evals.run.KindWiring",
        "threetears.evals.run.KindFactory",
        "threetears.evals.run.SubjectResolver",
        "threetears.evals.run.UnknownSubject",
        "threetears.evals.run.launch_run",
        "threetears.evals.run.start_run",
        "threetears.evals.run.execute_run",
        "threetears.evals.run.RunnerOptions",
        "threetears.evals.run.price_subject_call_floor",
        "threetears.evals.run.run_blocking",
        "threetears.evals.kernel.default_job_timeout",
        "threetears.evals.kernel.EvalCampaign",
        # The stored documents a host's kind and launcher build and its own code reads back.
        "threetears.evals.schema.EvalRun",
        "threetears.evals.schema.EvalResult",
        "threetears.evals.schema.EvalTemplate",
        "threetears.evals.schema.EvalTestCase",
        "threetears.evals.schema.EvalTrace",
        # The store: the port a host implements, the conformance kits that prove it, and the engine's own stores.
        "threetears.evals.schema.DocumentStore",
        "threetears.evals.schema.AsyncDocumentStore",
        "threetears.evals.schema.StoreClock",
        "threetears.evals.schema.AsyncStoreClock",
        "threetears.evals.schema.StoreConflict",
        "threetears.evals.testing.STORE_CONFORMANCE_CASES",
        "threetears.evals.testing.StoreConformanceCase",
        "threetears.evals.testing.check_completion_conformance",
        "threetears.evals.testing.CompletionGuard",
        "threetears.evals.testing.JobsConformanceHost",
        "threetears.evals.testing.jobs_conformance_cases",
        "threetears.evals.storage.InMemoryDocumentStore",
        "threetears.evals.storage.SqliteDocumentStore",
        "threetears.evals.storage.RoutedDocumentStore",
        "threetears.evals.storage.DocTypeRoute",
        "threetears.evals.storage.bridge_async_store",
        # The refusals: one exception per refusal kind, which a host maps in its own routes, and the stored kind.
        "threetears.evals.kernel.EvalServiceError",
        "threetears.evals.kernel.NotFoundError",
        "threetears.evals.kernel.ConflictError",
        "threetears.evals.kernel.ValidationFailedError",
        "threetears.evals.kernel.StorageError",
        "threetears.evals.kernel.ProviderRefusedError",
        "threetears.evals.kernel.AdmissionRefusedError",
        "threetears.evals.kernel.UnreadableRunError",
        "threetears.evals.schema.RefusalKind",
        # The version constants a host checks a store or a stored document against, one per version space, and the
        # installed build's own version, which a host records beside what it stores since nothing stored carries it.
        "threetears.evals.__version__",
        "threetears.evals.schema.CORE_SCHEMA_VERSION",
        "threetears.evals.schema.CORE_SCHEMA_MINOR",
        "threetears.evals.schema.CORE_BASELINE_VERSION",
        "threetears.evals.schema.REGENERABLE_SCHEMA_VERSION",
        "threetears.evals.schema.CASSETTE_SCHEMA_VERSION",
        "threetears.evals.kernel.IDENTITY_VERSION",
        "threetears.evals.analysis.REPORT_VERSION",
        "threetears.evals.analysis.CELL_MODEL_VERSION",
        # The closed vocabularies a host writes and passes: each open stored vocabulary's values this build knows.
        "threetears.evals.schema.KnownUsageRole",
        "threetears.evals.schema.KnownSimulatorPurpose",
        "threetears.evals.schema.KnownCostCapOrigin",
        "threetears.evals.schema.KnownMeteredCallOrigin",
        "threetears.evals.schema.KnownCellTimeoutOrigin",
        "threetears.evals.schema.KnownModelRoleOrigin",
        "threetears.evals.schema.KnownRoleModelOrigin",
        "threetears.evals.schema.KnownReasoningEffort",
        "threetears.evals.schema.KnownOutOfRunPurpose",
        "threetears.evals.schema.KnownRefusalKind",
        # The same vocabularies' values as tuples, which a host reads instead of `get_args` on the open stored type.
        "threetears.evals.schema.USAGE_ROLES",
        "threetears.evals.schema.SIMULATOR_PURPOSES",
        "threetears.evals.schema.COST_CAP_ORIGINS",
        "threetears.evals.schema.METERED_CALL_ORIGINS",
        "threetears.evals.schema.CELL_TIMEOUT_ORIGINS",
        "threetears.evals.schema.MODEL_ROLE_ORIGINS",
        "threetears.evals.schema.ROLE_MODEL_ORIGINS",
        "threetears.evals.schema.REASONING_EFFORTS",
        "threetears.evals.schema.OUT_OF_RUN_PURPOSES",
        "threetears.evals.schema.REFUSAL_KINDS",
        # The quick path, what it returns, and the pieces the host guide hands a host of its own.
        "threetears.evals.quick.run_eval",
        "threetears.evals.quick.compare",
        "threetears.evals.quick.Comparison",
        "threetears.evals.quick.ArmKey",
        "threetears.evals.analysis.EvalSummary",
        "threetears.evals.analysis.Report",
        "threetears.evals.quick.callable_host",
        "threetears.evals.quick.CALLABLE_KIND",
        "threetears.evals.quick.Answer",
        "threetears.evals.quick.Judge",
        "threetears.evals.quick.Guardrail",
        "threetears.evals.quick.CandidateTools",
        "threetears.evals.quick.World",
        "threetears.evals.quick.WorldTool",
        "threetears.evals.quick.WorldTools",
        "threetears.evals.quick.Dimension",
        "threetears.evals.quick.measure",
        "threetears.evals.quick.callable_kind",
        "threetears.evals.quick.callable_kind_contracts",
        # Reading runs back, as the tutorial and the examples do.
        "threetears.evals.ops.scope_history",
        "threetears.evals.run.list_results",
        "threetears.evals.run.get_result_trace",
        # Analysing a campaign and rendering its report, as the examples do.
        "threetears.evals.analysis.inspect_campaign_bundle",
        "threetears.evals.analysis.generate_analysis",
        "threetears.evals.analysis.EVAL_ANALYSIS_GEN_DEFAULT",
        "threetears.evals.analysis.report_markdown",
        "threetears.evals.analysis.report_html",
        "threetears.evals.analysis.format_number",
        "threetears.evals.analysis.format_signed",
        "threetears.evals.analysis.ChartBlock",
        "threetears.evals.kernel.EvalAnalysis",
        "threetears.evals.vega.VegaRenderer",
        # The shipped provider adapter: a host that brings no client of its own binds to these.
        "threetears.evals.providers.models_client",
    }
)


def tier_of(qualified: str) -> Tier:
    """A public name's tier, by ``root.name``: ``host-contract`` when :data:`HOST_CONTRACT` lists it."""
    return "host-contract" if qualified in HOST_CONTRACT else "closure"


def installed_surface() -> dict[str, SurfaceEntry]:
    """``root.name`` → its entry, with its tier, for every public name the installed package exports."""
    surface = {}
    for root, name, obj in public_names():
        qualified = f"{root}.{name}"
        surface[qualified] = dataclasses.replace(surface_entry(root, name, obj), tier=tier_of(qualified))
    return surface


def exported_twice() -> dict[str, list[str]]:
    """Each name exported from more than one public root → those roots, in ``PUBLIC_ROOTS`` order."""
    roots: dict[str, list[str]] = {}
    for root, name, _ in public_names():
        roots.setdefault(name, []).append(root)
    return {name: where for name, where in sorted(roots.items()) if len(where) > 1}


def exports_by_name() -> dict[str, list[tuple[str, Any]]]:
    """Each public name → (root, object) for every root that exports it, in ``PUBLIC_ROOTS`` order."""
    found: dict[str, list[tuple[str, Any]]] = {}
    for root, name, obj in public_names():
        found.setdefault(name, []).append((root, obj))
    return found


def home_root(root: str, name: str) -> str:
    """The public root that holds the module defining ``root.name``: the root the name belongs to, wherever else it
    is exported from. The longest root the defining module sits under, so ``kernel.host`` wins over ``kernel``.

    A class or function says where it was defined; a constant or alias does not, and is followed through the
    ``from … import`` statements that re-export it to the module that assigns it.
    """
    from threetears.evals import PUBLIC_ROOTS

    obj = getattr(importlib.import_module(root), name)
    module = getattr(obj, "__module__", None) if isinstance(obj, type) or inspect.isroutine(obj) else None
    if not module or not module.startswith("threetears.evals."):
        found = defining_module(root, name)
        module = found[0] if found is not None else root
    under = [r for r in PUBLIC_ROOTS if module == r or module.startswith(f"{r}.")]
    return max(under, key=len) if under else root


# --- the record: what left the surface, and what is exported twice on purpose -------------------------------------

#: The checked-in snapshot of the public surface. Generated; never edited by hand.
SNAPSHOT_PATH = PACKAGE_ROOT / "api" / "surface.json"

#: The hand-written record of every name that left a public root: per release, its new home or ``removed``, and why.
MIGRATIONS_PATH = PACKAGE_ROOT / "MIGRATIONS.toml"

#: The hand-written declarations of the names exported from more than one root on purpose.
REEXPORTS_PATH = PACKAGE_ROOT / "api" / "reexports.toml"

#: The command that regenerates the snapshot, as every finding that regenerating fixes tells a reader.
REGENERATE_SNAPSHOT = "uv run python packages/evals/scripts/generate_api_snapshot.py"

#: What a change's ``to`` says when a name has no new home.
REMOVED = "removed"

#: The version of the section for the release being prepared, before it is named for its version.
UNRELEASED = "unreleased"

#: The snapshot file's own format version, so a reader of an older layout is refused rather than misread. 2 added
#: each entry's ``tier``.
SNAPSHOT_FORMAT = 2


class _Record(BaseModel):
    """Strict, as every stored shape in the package is: a misspelt key in a hand-written record is refused, never
    silently dropped, because a dropped ``why`` or ``names`` would let a removal pass unrecorded."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


def _qualified(text: str) -> str:
    if not text.startswith("threetears.evals."):
        raise ValueError(f"{text!r} is not a module under threetears.evals; write the root in full")
    return text


class Migration(_Record):
    """One name that left a public root, as release notes list it: ``old`` → ``new`` (or ``removed``), and why."""

    old: str
    new: str
    why: str


class NameChange(_Record):
    """One ``[[release.change]]``: names that left one root together, for one reason.

    ``names`` keep their name in ``to``; ``renamed`` maps an old name to its new one. ``to`` is a root, or
    ``"removed"`` when the names have no new home (and then nothing can be renamed).
    """

    source: str = Field(alias="from", description="The root the names left, in full: threetears.evals.<root>.")
    to: str = Field(description="The root they live in now, in full, or 'removed'.")
    names: list[str] = Field(default_factory=list, description="Names that keep their name in their new home.")
    renamed: dict[str, str] = Field(default_factory=dict, description="Old name → new name, for a name renamed.")
    why: str = Field(min_length=1, description="Why they left: what a porter needs to understand the change.")

    @model_validator(mode="after")
    def _coherent(self) -> NameChange:
        _qualified(self.source)
        if self.to != REMOVED:
            _qualified(self.to)
        if not self.names and not self.renamed:
            raise ValueError(f"a change from {self.source} names nothing: list `names` or `renamed`")
        if self.to == REMOVED and self.renamed:
            raise ValueError(f"a name removed from {self.source} has no new name: move it to `names`")
        if both := set(self.names) & set(self.renamed):
            raise ValueError(f"{sorted(both)} listed in both `names` and `renamed` of a change from {self.source}")
        if self.source == self.to and self.names:
            raise ValueError(f"{self.names} are listed as moving from {self.source} to itself; rename or drop them")
        return self

    def migrations(self) -> list[Migration]:
        """One per name, in the order written."""
        out = []
        for name in self.names:
            new = REMOVED if self.to == REMOVED else f"{self.to}.{name}"
            out.append(Migration(old=f"{self.source}.{name}", new=new, why=self.why))
        for old, new_name in self.renamed.items():
            out.append(Migration(old=f"{self.source}.{old}", new=f"{self.to}.{new_name}", why=self.why))
        return out


class Release(_Record):
    """One ``[[release]]``: what one version moved or removed from the public roots, and anything else a porter
    must change that is not a public name (a field, a submodule).

    A section is a final release's (``version = "0.66.0"``) or the one being prepared (``"unreleased"``), and its
    ``since`` is a final release's tag. A pre-release is neither: nothing it shipped is a promise, so it has no
    section and no section is measured from it.
    """

    version: str = Field(
        min_length=1, description="The version, as its tag reads without the leading 'v', or 'unreleased'."
    )
    since: str | None = Field(
        default=None,
        description=(
            "The tag of the final release before this one. Given, the section is checked against what actually "
            "left a root between that tag and this version's (or the working tree, before this version is tagged). "
            "A tag at which the package did not exist has an empty surface, so nothing can have left since it."
        ),
    )
    summary: str = Field(default="", description="What this section covers, in a paragraph.")
    changes: list[NameChange] = Field(default_factory=list, alias="change")
    notes: list[str] = Field(default_factory=list, description="Breaking changes that are not a public name.")

    @model_validator(mode="after")
    def _final_releases_only(self) -> Release:
        if self.version != UNRELEASED and _version_key(f"v{self.version}") is None:
            raise ValueError(
                f"section {self.version!r} is not a final release: a section's version is x.y.z or 'unreleased'"
            )
        if self.since is not None and _version_key(self.since) is None:
            raise ValueError(
                f"section {self.version} is since {self.since!r}, which is not a final release tag (v<x>.<y>.<z>)"
            )
        return self

    @model_validator(mode="after")
    def _each_name_once(self) -> Release:
        seen: set[str] = set()
        for migration in self.migrations():
            if migration.old in seen:
                raise ValueError(f"release {self.version} records {migration.old} twice")
            seen.add(migration.old)
        return self

    def migrations(self) -> list[Migration]:
        """Every name this release moved or removed, in the order written."""
        return [m for change in self.changes for m in change.migrations()]


class Migrations(_Record):
    """``MIGRATIONS.toml``: a release per section, newest first. The first section is the release being prepared,
    and is where a name leaving a root now must be recorded."""

    releases: list[Release] = Field(alias="release", min_length=1)

    @model_validator(mode="after")
    def _versions_unique(self) -> Migrations:
        versions = [r.version for r in self.releases]
        if dupes := sorted({v for v in versions if versions.count(v) > 1}):
            raise ValueError(f"MIGRATIONS.toml has more than one section for {dupes}")
        return self

    def release(self, version: str) -> Release | None:
        """The section for ``version`` (a leading ``v``, as a tag has, is ignored), if there is one."""
        wanted = version.removeprefix("v")
        return next((r for r in self.releases if r.version == wanted), None)


class ReExport(_Record):
    """One ``[[reexport]]``: names whose home is one root and which another root exports too, on purpose."""

    home: str = Field(description="The root holding the names' defining modules, in full.")
    also: list[str] = Field(min_length=1, description="Every other root exporting them, in full.")
    names: list[str] = Field(min_length=1)
    why: str = Field(min_length=1, description="Why the other roots need the names: what dropping them would break.")


class ReExports(_Record):
    """``api/reexports.toml``: every name exported from more than one root on purpose."""

    reexports: list[ReExport] = Field(default_factory=list, alias="reexport")

    def roots_of(self, name: str) -> tuple[str, frozenset[str]] | None:
        """(home, every root exporting it) as declared for ``name``, or ``None`` when it is not declared."""
        for entry in self.reexports:
            if name in entry.names:
                return entry.home, frozenset({entry.home, *entry.also})
        return None


def load_migrations(path: Path = MIGRATIONS_PATH) -> Migrations:
    """Read and validate ``MIGRATIONS.toml``."""
    return Migrations.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))


def load_reexports(path: Path = REEXPORTS_PATH) -> ReExports:
    """Read and validate ``api/reexports.toml``."""
    return ReExports.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, SurfaceEntry]:
    """The checked-in snapshot, or an empty one when there is none yet (every name then reads as added)."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != SNAPSHOT_FORMAT:
        raise SystemExit(f"{path} is format {data.get('format')!r}, not {SNAPSHOT_FORMAT}; regenerate it")
    for name, entry in data["names"].items():
        if entry.get("tier") not in TIERS:
            # An entry's tier is never defaulted on a read: a missing one would read as ``closure`` and drop a
            # promise without a diff saying so.
            raise SystemExit(
                f"{path} records {name} with tier {entry.get('tier')!r}, not one of {TIERS}; regenerate it"
            )
    return {name: SurfaceEntry(**entry) for name, entry in data["names"].items()}


def render_snapshot(surface: dict[str, SurfaceEntry]) -> str:
    """The snapshot file: one entry per ``root.name``, in walk order, so a change is a small, readable diff."""
    names = {name: {"kind": e.kind, "signature": e.signature, "tier": e.tier} for name, e in surface.items()}
    return json.dumps({"format": SNAPSHOT_FORMAT, "names": names}, indent=2, ensure_ascii=False) + "\n"


# --- findings ----------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Finding:
    """One way the installed surface and its record disagree.

    Attributes:
        problem: What is wrong and what to do about it.
        regenerating_fixes: Whether regenerating the snapshot is the whole remedy. A name added, or a kind,
            signature or tier changed, is: the diff the regeneration makes is the record. A name that left a root is not
            until ``MIGRATIONS.toml`` records it, and the generator refuses to write while any finding is not.
    """

    problem: str
    regenerating_fixes: bool


def snapshot_findings(
    installed: dict[str, SurfaceEntry],
    recorded: dict[str, SurfaceEntry],
    newest: Release,
    released: set[str] | None,
) -> list[Finding]:
    """How the installed surface differs from the snapshot, each difference with its remedy.

    Args:
        installed: The surface the package exports now.
        recorded: The checked-in snapshot.
        newest: ``MIGRATIONS.toml``'s first section, where a name leaving a root now is recorded.
        released: Every ``root.name`` the final release the newest section follows exported
            (:func:`released_names`), or ``None`` when that is not known, and then every name that left needs a
            record. A name outside it never shipped in a release: it may leave with only a regeneration, and
            ``MIGRATIONS.toml`` must not record it.

    Returns:
        The findings: removals in snapshot order, then additions and changes in walk order.
    """
    findings: list[Finding] = []
    moved = {m.old: m for m in newest.migrations()}
    for name in [n for n in recorded if n not in installed]:
        if name in moved:
            findings.append(
                Finding(f"{name} left the surface ({_destination(moved[name])}): run `{REGENERATE_SNAPSHOT}`.", True)
            )
        elif released is not None and name not in released:
            findings.append(
                Finding(
                    f"{name} left the surface, and no release shipped it ({newest.since} did not export it), so "
                    f"nothing records it: run `{REGENERATE_SNAPSHOT}`.",
                    True,
                )
            )
        else:
            remedy = (
                "It is host-contract: it keeps working beside its new home for one minor release before it leaves, "
                "and the record names both."
                if recorded[name].tier == "host-contract"
                else "No compatibility shim: a porter reads the release notes."
            )
            findings.append(
                Finding(
                    f"{name} left the public surface, and MIGRATIONS.toml's newest section ({newest.version}) does "
                    f"not record it. Add it to a [[release.change]] there with its new home (or 'removed') and why, "
                    f"then run `{REGENERATE_SNAPSHOT}`. {remedy}",
                    False,
                )
            )
    for name in [n for n in installed if n not in recorded]:
        findings.append(Finding(f"{name} is new on the public surface: run `{REGENERATE_SNAPSHOT}`.", True))
    for name in [n for n in installed if n in recorded]:
        now, was = installed[name], recorded[name]
        if (now.kind, now.signature) != (was.kind, was.signature):
            findings.append(
                Finding(
                    f"{name} changed: was {was.kind} `{was.signature}`, now {now.kind} `{now.signature}`. Run "
                    f"`{REGENERATE_SNAPSHOT}`; a breaking change that is not a name leaving goes in the newest "
                    f"section's `notes`.",
                    True,
                )
            )
        if now.tier != was.tier:
            findings.append(
                Finding(
                    f"{name} moved from tier {was.tier} to {now.tier} (HOST_CONTRACT in scripts/public_surface.py). "
                    f"Run `{REGENERATE_SNAPSHOT}`; once a release has shipped it as host-contract, leaving that tier "
                    f"is a breaking change for the newest section's `notes`.",
                    True,
                )
            )
    return findings


def tier_findings(installed: dict[str, SurfaceEntry]) -> list[Finding]:
    """Names :data:`HOST_CONTRACT` lists that no public root exports: a list naming a name that moved or left, which
    would promise nothing while reading as a promise."""
    return [
        Finding(
            f"HOST_CONTRACT in scripts/public_surface.py lists {name}, which no public root exports: name it at the "
            f"root that exports it now, or drop it once its record in MIGRATIONS.toml says it left.",
            False,
        )
        for name in sorted(HOST_CONTRACT - installed.keys())
    ]


def _destination(migration: Migration) -> str:
    return "removed" if migration.new == REMOVED else f"now {migration.new}"


def record_findings(installed: dict[str, SurfaceEntry], newest: Release) -> list[Finding]:
    """Where the newest section of ``MIGRATIONS.toml`` disagrees with the package: a name it says left a root is
    still exported there, or the new home it names exports no such name. Older sections are history, written
    against the surface of their own release, and are not checked against today's."""
    findings = []
    for migration in newest.migrations():
        if migration.old in installed:
            findings.append(
                Finding(
                    f"MIGRATIONS.toml ({newest.version}) records {migration.old} as gone, but it is still exported: "
                    f"drop the export or the record.",
                    False,
                )
            )
        if migration.new != REMOVED and migration.new not in installed:
            findings.append(
                Finding(
                    f"MIGRATIONS.toml ({newest.version}) sends {migration.old} to {migration.new}, which is not "
                    f"exported: name the root that exports it now.",
                    False,
                )
            )
    return findings


def duplicate_findings(
    exports: dict[str, list[tuple[str, Any]]], homes: dict[str, str], reexports: ReExports
) -> list[Finding]:
    """Names exported from more than one root that are not declared re-exports, and declarations that no longer
    match what is exported.

    A name has one home, so a reader and a porter know where it lives and a move is one record. A second root may
    export it only when it is the same object and ``api/reexports.toml`` says why that root needs it (``ops``
    re-exports the analysis results its operations hand back, because ``actions`` may reach analysis only
    through ``ops``). Two different objects under one name is never a re-export: rename one.

    Args:
        exports: Each public name → (root, object) for every root that exports it.
        homes: Each name exported twice → the root holding its defining module.
        reexports: The declarations.

    Returns:
        The findings, by name. None of them is fixed by regenerating the snapshot.
    """
    findings = []
    for name, where in sorted(exports.items()):
        if len(where) < 2:
            continue
        roots = [root for root, _ in where]
        if len({id(obj) for _, obj in where}) > 1:
            findings.append(
                Finding(
                    f"{name} names different objects in {', '.join(roots)}: a name means one thing on the "
                    f"surface. Rename one, and record the rename in MIGRATIONS.toml's newest section.",
                    False,
                )
            )
            continue
        declared = reexports.roots_of(name)
        home = homes.get(name, roots[0])
        if declared is None:
            findings.append(
                Finding(
                    f"{name} is exported from {', '.join(roots)}; its home is {home}. Drop it from the other "
                    f"roots and record each drop in MIGRATIONS.toml's newest section, or declare it in "
                    f"api/reexports.toml with why the other roots need it.",
                    False,
                )
            )
        elif declared != (home, frozenset(roots)):
            findings.append(
                Finding(
                    f"api/reexports.toml declares {name} at home in {declared[0]} and exported from "
                    f"{', '.join(sorted(declared[1]))}, but its home is {home} and it is exported from "
                    f"{', '.join(roots)}: correct the declaration.",
                    False,
                )
            )
    for entry in reexports.reexports:
        for name in entry.names:
            if len(exports.get(name, [])) < 2:
                findings.append(
                    Finding(
                        f"api/reexports.toml declares {name} a re-export, but it is exported from "
                        f"{len(exports.get(name, []))} root(s): drop the declaration.",
                        False,
                    )
                )
    return findings


def all_findings(
    installed: dict[str, SurfaceEntry],
    recorded: dict[str, SurfaceEntry],
    migrations: Migrations,
    reexports: ReExports,
) -> list[Finding]:
    """Every finding over the installed package: the snapshot, the newest record, the duplicates and the tiers."""
    exports = exports_by_name()
    homes = {name: home_root(where[0][0], name) for name, where in exports.items() if len(where) > 1}
    newest = migrations.releases[0]
    tag = newest_tag()
    return [
        *duplicate_findings(exports, homes, reexports),
        *tier_findings(installed),
        *record_findings(installed, newest),
        *snapshot_findings(installed, recorded, newest, released_names(newest)),
        *(f for release in migrations.releases if release.since for f in history_findings(release)),
        *labelling_findings(migrations, tag, shipped=tag is not None and package_at(tag)),
    ]


# --- history: a section against the two releases it sits between -------------------------------------------------

#: The package's source tree, as a path inside the repository: what a tagged tree is read under.
_SOURCE = "packages/evals/src"

#: The package's own module, as a path inside the repository: present in every tree the package is in.
_PACKAGE_INIT = f"{_SOURCE}/threetears/evals/__init__.py"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PACKAGE_ROOT, capture_output=True, text=True, check=False)


def tag_exists(tag: str) -> bool:
    """Whether this checkout has ``tag``."""
    return _git("rev-parse", "--verify", "--quiet", f"refs/tags/{tag}").returncode == 0


def package_at(tag: str) -> bool:
    """Whether the package is in the tree at ``tag``: a release from before it was added shipped none of its names.

    Raises:
        SystemExit: When the checkout lacks ``tag``, which says nothing about the package.
    """
    if not tag_exists(tag):
        raise SystemExit(f"{tag} is not a tag in this checkout: fetch the tags (git fetch --tags)")
    return _git("cat-file", "-e", f"{tag}:{_PACKAGE_INIT}").returncode == 0


def path_at(tag: str, path: str) -> bool:
    """Whether the tree at ``tag`` holds ``path`` (a file or a directory, as a path inside the repository): what a
    release shipped, read from git rather than from the working tree."""
    return _git("cat-file", "-e", f"{tag}:{path}").returncode == 0


def source_at(ref: str | None, path: str) -> str | None:
    """A file under the repository at ``ref``, or in the working tree when ``ref`` is ``None``; ``None`` when that
    tree holds no such file. The one way the dev tools read a released tree without installing it: the surface here,
    the released judge pins (``tests/test_judge_identity_pins.py``)."""
    if ref is None:
        local = PACKAGE_ROOT.parent.parent / path
        return local.read_text(encoding="utf-8") if local.exists() else None
    shown = _git("show", f"{ref}:{path}")
    return shown.stdout if shown.returncode == 0 else None


def _literal_assignment(source: str, name: str) -> Any:
    for node in ast.parse(source).body:
        if name in assigned_names(node) and isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            return ast.literal_eval(node.value)
    return None


def exports_at(ref: str | None) -> dict[str, set[str]]:
    """Each walked module at ``ref`` (a tag, or ``None`` for the working tree) → its ``__all__``: the package itself
    and each public root, read from the source as written there rather than imported, so a release's surface is read
    without installing it.

    A tag from before the package existed has no public root, so its surface is empty: every name is new since
    it, and none has left.
    """
    if ref is not None and not package_at(ref):
        return {}
    package = source_at(ref, _PACKAGE_INIT)
    if package is None:
        raise SystemExit(f"threetears.evals is not in the tree at {ref or 'the working tree'}")
    roots = _literal_assignment(package, "PUBLIC_ROOTS")
    own = _literal_assignment(package, "__all__")
    if own is None:
        raise SystemExit(f"{PACKAGE} has no literal __all__ at {ref or 'the working tree'}")
    found: dict[str, set[str]] = {PACKAGE: set(own)}
    for root in roots:
        base = f"{_SOURCE}/{root.replace('.', '/')}"
        source = source_at(ref, f"{base}/__init__.py") or source_at(ref, f"{base}.py")
        names = _literal_assignment(source, "__all__") if source is not None else None
        if names is None:
            raise SystemExit(f"{root} has no literal __all__ at {ref or 'the working tree'}")
        found[root] = set(names)
    return found


def departures(before: dict[str, set[str]], after: dict[str, set[str]]) -> set[str]:
    """Every ``root.name`` exported before and not from the same root after: what moved, was renamed or removed."""
    return {f"{root}.{name}" for root, names in before.items() for name in names - after.get(root, set())}


def released_names(release: Release) -> set[str] | None:
    """Every ``root.name`` exported by the final release ``release`` follows (its ``since``): the names a porter
    may import, and so the only ones whose leaving is recorded. ``None`` when that is not known (no ``since``, or a
    checkout without the tag, which :func:`history_findings` reports)."""
    if release.since is None or not tag_exists(release.since):
        return None
    return {f"{root}.{name}" for root, names in exports_at(release.since).items() for name in names}


def section_findings(
    release: Release, before: dict[str, set[str]], after: dict[str, set[str]], after_label: str
) -> list[Finding]:
    """Where a section disagrees with the surfaces of the two releases it sits between.

    It must record exactly the names that left a root between ``before`` and ``after`` (:func:`departures`), and
    each new home it names must export the name in ``after``. A name that was not in ``before`` never shipped,
    so it has not left whatever happened to it since, and recording it is refused.

    Args:
        release: The section, with its ``since``.
        before: The surface at ``since`` (:func:`exports_at`); empty when the package was not in that release.
        after: The surface at the section's own tag, or the working tree before it is tagged.
        after_label: How ``after`` is named in a finding.
    """
    left = departures(before, after)
    recorded = {m.old: m for m in release.migrations()}
    findings = [
        Finding(
            f"{name} left its root between {release.since} and {after_label}, and MIGRATIONS.toml's "
            f"{release.version} section does not record it.",
            False,
        )
        for name in sorted(left - recorded.keys())
    ]
    findings += [
        Finding(
            f"MIGRATIONS.toml's {release.version} section records {name}, which did not leave its root between "
            f"{release.since} and {after_label}.",
            False,
        )
        for name in sorted(recorded.keys() - left)
    ]
    for migration in recorded.values():
        if migration.new == REMOVED:
            continue
        root, name = migration.new.rsplit(".", 1)
        if name not in after.get(root, set()):
            findings.append(
                Finding(
                    f"MIGRATIONS.toml's {release.version} section sends {migration.old} to {migration.new}, "
                    f"which {after_label} does not export.",
                    False,
                )
            )
    return findings


def history_findings(release: Release) -> list[Finding]:
    """Where a section disagrees with the two releases it sits between (:func:`section_findings`), both read from
    git: ``since``, and the version's own tag, or the working tree before that tag exists. A checkout without the
    tags cannot check it and says so rather than passing.
    """
    assert release.since is not None
    tag = f"v{release.version}"
    if not tag_exists(release.since):
        return [
            Finding(
                f"MIGRATIONS.toml's {release.version} section is checked against {release.since}, a tag "
                f"this checkout lacks: fetch the tags (git fetch --tags, or fetch-depth: 0 in CI).",
                False,
            )
        ]
    after_ref = tag if tag_exists(tag) else None
    after_label = tag if after_ref else "the working tree"
    return section_findings(release, exports_at(release.since), exports_at(after_ref), after_label)


# --- labelling: a tagged release has its own section -------------------------------------------------------------


#: A release tag: ``v<major>.<minor>.<patch>`` and nothing more. Every package of the family releases at one
#: version, so every such tag is the package's too. A tag with any suffix is not a release: a semver pre-release
#: (``v0.66.0-rc.1``, ``v0.66.0-evaladopt.7``), a PEP 440 one (``v0.66.0rc1``, ``v0.66.0a1``, ``v0.66.0.dev1``),
#: or anything else. A pre-release promises nothing, so the records neither follow one nor are measured from one.
_RELEASE_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def _version_key(tag: str) -> tuple[int, int, int] | None:
    """``tag``'s place in release order, or ``None`` for a tag that is no final release's."""
    matched = _RELEASE_TAG.match(tag)
    if matched is None:
        return None
    major, minor, patch = matched.groups()
    return int(major), int(minor), int(patch)


def newest_release_tag(every: Iterable[str], version: str) -> str | None:
    """The release the working tree's records follow: the newest final release tag at or below ``version``.

    **Wherever the tag is.** A release is tagged on ``main``, and ``develop`` never merges ``main`` back, so a
    release's tag is never behind ``develop``'s HEAD: read by reachability, ``develop`` would hold its records
    against the release before, and fail the moment a real release is cut. A final release at or below the version
    the tree prepares has shipped, and the tree's records follow it. A pre-release tag is not a release
    (:data:`_RELEASE_TAG`) and is never followed, wherever it is.

    Args:
        every: Every tag in the checkout, on any ref.
        version: The version the tree prepares, as ``pyproject.toml`` states it.

    Returns:
        The newest such tag, or ``None`` when there is none.
    """
    prepared = _version_key(f"v{version}")
    finals = {
        tag: key for tag in every if (key := _version_key(tag)) is not None and (prepared is None or key <= prepared)
    }
    return max(finals, key=finals.__getitem__, default=None)


def package_version() -> str:
    """The version the package's ``pyproject.toml`` states: the release the working tree prepares."""
    project = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(project["project"]["version"])


def newest_tag() -> str | None:
    """The release tag the working tree's records follow (:func:`newest_release_tag`), or ``None`` without tags."""
    every = _git("tag", "--list", "v[0-9]*")
    if every.returncode != 0:
        return None
    return newest_release_tag(every.stdout.split(), package_version())


def labelling_findings(migrations: Migrations, tag: str | None, *, shipped: bool) -> list[Finding]:
    """Where the sections disagree with the newest release the working tree follows (:func:`newest_tag`).

    A release is tagged with its section already named for it, and the next release's work goes in a new
    ``unreleased`` section ``since`` that tag. A tag cut with its section still called ``unreleased`` leaves that
    section open: the next release's records pile into it, are checked against the working tree from the tag
    before, and ``release_notes.py v<tag>`` finds nothing. So once the tree follows a tag, that tag must have a
    section of its own, and an ``unreleased`` section must start at it.

    Args:
        migrations: The sections.
        tag: The newest release the tree follows, or ``None`` when the checkout has no release tag.
        shipped: Whether the package was in the tree at ``tag`` (:func:`package_at`). A release from before the
            package existed shipped nothing of it and has no section; the package's first section starts at it.
    """
    if tag is None:
        return [
            Finding(
                "MIGRATIONS.toml's sections are checked against the newest release the working tree follows, and "
                "this checkout has no release tag: fetch the tags (git fetch --tags, or fetch-depth: 0 in CI).",
                False,
            )
        ]
    version = tag.removeprefix("v")
    findings: list[Finding] = []
    if shipped and migrations.release(version) is None:
        findings.append(
            Finding(
                f"{tag} is tagged, and MIGRATIONS.toml has no section for it: the section it shipped is still "
                f'called "unreleased". Rename that section to version = "{version}" holding exactly what it held at '
                f"the tag (git show {tag}:packages/evals/MIGRATIONS.toml), and record what came after it in a new "
                f'section version = "unreleased", since = "{tag}". Name the section for its version before tagging.',
                False,
            )
        )
    newest = migrations.releases[0]
    if newest.version == UNRELEASED and newest.since != tag:
        findings.append(
            Finding(
                f"the unreleased section is since {newest.since}, but {tag} is the newest release this tree follows: "
                f'set since = "{tag}", and give {tag} its own section if it has none.',
                False,
            )
        )
    return findings
