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
the names a host binds to and :data:`STABLE` the rest of the declared surface, and every other exported name is
``provisional``: importable from its root, promised nothing. The snapshot records each entry's tier, so a promotion
or a demotion is a diff in review like any other.
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
#: only with a ``MIGRATIONS.toml`` record and a deprecation cycle. ``stable`` is the rest of the declared surface
#: (:data:`STABLE`): it may move in any release, recorded in ``MIGRATIONS.toml`` and never shimmed. ``provisional`` is
#: every other name a root exports, most because a declared signature reaches it, so a host can type its code without
#: reaching below a root: it is promised nothing, ``docs/reference.md`` does not list it, and the snapshot records
#: its name and kind but not its signature.
Tier = Literal["host-contract", "stable", "provisional"]

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
        tier: Its stability tier (:data:`Tier`): ``host-contract`` when :data:`HOST_CONTRACT` lists it, ``stable``
            when :data:`STABLE` does, otherwise ``provisional``, which is what a name nobody has promised is.
    """

    kind: str
    signature: str
    tier: Tier = "provisional"


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
#: release), from 0.69.0 on. :data:`STABLE` holds the rest of the declared surface; every other public name is
#: ``provisional``.
#:
#: The ruling is what a host author is told to write against: ``docs/adopting-a-host.md``, ``docs/ladder.md``, the
#: README and ``examples/`` (every name they import is here, which ``tests/test_api_tiers.py`` holds), and the
#: courier host (``tests/fixtures/courierhost/``), the least a product writes to be a host. A name a topic guide
#: alone imports, a field type a contract name's signature reaches, and a name a later section of the host guide
#: mentions for an optional feature are not: promoting one later is free, and demoting one is a break.
HOST_CONTRACT: frozenset[str] = frozenset(
    {
        # The host: the one value every entrypoint takes, and the services it is built from.
        "threetears.evals.kernel.host.EvalHost",
        "threetears.evals.kernel.host.HostProfile",
        "threetears.evals.kernel.host.CompletionClients",
        # The host's checks on a template, which create_template, update_template, the seed and a reset read off it.
        "threetears.evals.kernel.host.TemplateChecks",
        "threetears.evals.kernel.host.default_cell_timeout",
        "threetears.evals.kernel.EvalStorage",
        "threetears.evals.kernel.withhold_failure_detail",
        "threetears.evals.schema.ProviderFailureDescriber",
        "threetears.evals.schema.CompletionClient",
        "threetears.evals.schema.CompletionResult",
        # The five words a host's own client maps its provider's finish reason onto, and the role names that key
        # `CompletionClients` and the shipped factory's `defaults=`.
        "threetears.evals.schema.StopReason",
        "threetears.evals.kernel.host.CompletionRole",
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
        # An authored judge case in a host's boot seed, which a production seed builds in memory
        # (`SeedCorpus(judge_cases=...)`) as well as reads from `judge_cases/*.json`.
        "threetears.evals.run.SeedJudgeCase",
        # A template's own case in a host's boot seed (a persona, a bank input), built in memory
        # (`SeedCorpus(template_cases=...)`) as well as read from `template_cases/*.json`.
        "threetears.evals.run.SeedTemplateCase",
        # A named case set in a host's boot seed, which a production seed builds in memory
        # (`SeedCorpus(case_sets=...)`) as well as reads from `case_sets/*.json`.
        "threetears.evals.run.SeedCaseSet",
        # A release step brings seeded definitions up to date: the drift read over its corpus, the reset, and the two
        # catalogue actions a host mounts over them (Discodon runs the read and the reset as a release step).
        "threetears.evals.run.definition_drift",
        "threetears.evals.run.reset_definitions",
        # What the reset returns, and the digests a host computes over the corpus it seeded before `.30` (the guide
        # tells a host to write both).
        "threetears.evals.run.DefinitionReset",
        "threetears.evals.run.corpus_digests",
        "threetears.evals.actions.definition_drift_actions",
        # The production host imports these (`tests/fixtures/host_imports/discodon.txt`), so they are its contract:
        # the inner ceilings it derives its outer budgets from, the metering it folds, the template checks it runs and
        # the schema keys and prompt markers its own code matches. `Ceilings` carries the six ceilings as one value;
        # the six names stay beside it in `.32`.
        "threetears.evals.run.Ceilings",
        # The seam a host checks a proposed case name against, so it is never printed as `case N`.
        "threetears.evals.analysis.humanize",
        "threetears.evals.kernel.FIND_TEMPLATE",
        "threetears.evals.kernel.SEED_REMEDY",
        "threetears.evals.kernel.KIND_TEMPLATE",
        "threetears.evals.kernel.KIND_TEXT",
        "threetears.evals.schema.PROSE_SCHEMA_KEY",
        "threetears.evals.schema.nested_schemas",
        "threetears.evals.schema.RESERVED_DIM_IDS",
        "threetears.evals.schema.NON_TERMINAL_RUN_STATUSES",
        "threetears.evals.run.SCHEDULER_CALL_ATTEMPTS",
        "threetears.evals.run.SIMULATOR_MAX_TOKENS",
        "threetears.evals.run.SIMULATOR_ANSWER_BUDGET_TOKENS",
        "threetears.evals.run.MeteredCallTally",
        "threetears.evals.run.metered_cell_tally",
        "threetears.evals.run.fold_metered_cell",
        "threetears.evals.run.precondition_failure_text",
        "threetears.evals.run.refuse_ignored_template_fields",
        "threetears.evals.gen.PROPOSER_MAX_TOKENS",
        "threetears.evals.analysis.generation_ceiling_s",
        "threetears.evals.analysis.judge_phase_ceiling_s",
        "threetears.evals.ops.run_label",
        # The write that renames a stored case, which a host otherwise made by writing the case document itself.
        "threetears.evals.ops.case_rename",
        "threetears.evals.ops.CaseRenamed",
        # What the host guide tells a host to call, build, catch, type against or read in its prose: the cost-cap
        # wrapper and the error a refused reservation raises, the strict reply projection and the schema it refuses,
        # the reader of a boot seed directory, a measure's per-entry marker, the narrow store ports a function types
        # itself against, the read tier's row budget, the reporter kind's contract, the judge-wording sentence a
        # release note carries, the error a store raises on an event loop, and the renderer of started jobs.
        # `tests/test_guide_names_are_declared.py` holds the rule: a name the guide teaches is never provisional.
        "threetears.evals.kernel.CappedCompletion",
        "threetears.evals.kernel.CostCapReached",
        "threetears.evals.kernel.strict_response_format",
        "threetears.evals.kernel.UnportableSchema",
        "threetears.evals.kernel.host.MemberActsOn",
        "threetears.evals.run.load_seed_corpus",
        "threetears.evals.kernel.RunStore",
        "threetears.evals.kernel.ResultStore",
        "threetears.evals.kernel.RunRecordStore",
        "threetears.evals.kernel.DefinitionStore",
        "threetears.evals.kernel.CassetteStore",
        "threetears.evals.kernel.JobStore",
        "threetears.evals.analysis.READ_TIER_ROW_BUDGET",
        "threetears.evals.analysis.REPORTER_KIND_CONTRACT",
        "threetears.evals.schema.JUDGE_WORDING_CHANGE_SENTENCE",
        "threetears.evals.schema.StoreCalledOnEventLoop",
        "threetears.evals.actions.render_jobs_started",
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
        # The reader of a completion's optional `sent_settings`: what a host whose client reports what each request
        # carried tests its completion type against, beside the conformance check, and what it and every other reader
        # of a completion's members raise on a completion that breaks the protocol, which a host's suite catches.
        "threetears.evals.schema.sent_settings_of",
        "threetears.evals.schema.CompletionProtocolDefect",
        # Recorded completions in CI is a pattern a host uses: record its real client once, replay it in its suite.
        "threetears.evals.testing.RecordedCompletions",
        "threetears.evals.testing.RecordedPrompt",
        # What the recorder raises on a replay miss or a recording it cannot read, which a host's suite catches.
        "threetears.evals.testing.RecordingRefused",
        # The request settings a recording is matched on, required by `record` and `replay`, and the judge's own
        # value a host passes for its judge client: a host cannot call the recorder without naming one.
        "threetears.evals.schema.ClientRequestSettings",
        "threetears.evals.run.JUDGE_REQUEST_SETTINGS",
        # The simulator's own value, which a host's builder applies to its simulator client as the judge's to its judge,
        # and the analysis generator's, which it builds from its own answer budget, effort and headroom: what each role
        # is asked for, so a host's client sends what the run records as asked.
        "threetears.evals.run.SIMULATOR_REQUEST_SETTINGS",
        "threetears.evals.analysis.analysis_gen_request_settings_for",
        # The case floor below which no rate or difference is estimated, which a host's seed gate holds its templates to.
        "threetears.evals.analysis.MIN_CASES",
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
        # The public roots a consumer imports from, and which of them need an extra and its modules: what its own import
        # gate reads.
        "threetears.evals.PUBLIC_ROOTS",
        "threetears.evals.ROOT_EXTRAS",
        "threetears.evals.RootExtra",
        # And each public name's tier in the installed build, which a host's test gates its imports by.
        "threetears.evals.surface_tier",
        "threetears.evals.declared_names",
        "threetears.evals.schema.CORE_SCHEMA_VERSION",
        "threetears.evals.schema.CORE_SCHEMA_MINOR",
        "threetears.evals.schema.CORE_BASELINE_VERSION",
        "threetears.evals.schema.DURABLE_SCHEMA_VERSION",
        "threetears.evals.schema.DURABLE_SCHEMA_MINOR",
        "threetears.evals.schema.DURABLE_BASELINE_VERSION",
        "threetears.evals.schema.REGENERABLE_SCHEMA_VERSION",
        "threetears.evals.schema.CASSETTE_SCHEMA_VERSION",
        "threetears.evals.kernel.IDENTITY_VERSION",
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
        # The quick path at the package root, what it returns, and the host bridge a host adopting it builds.
        "threetears.evals.Eval",
        "threetears.evals.Result",
        "threetears.evals.Judge",
        "threetears.evals.Example",
        "threetears.evals.grader",
        "threetears.evals.World",
        "threetears.evals.Log",
        "threetears.evals.Model",
        "threetears.evals.Answer",
        "threetears.evals.quick.as_host",
        # Rung 4 serves its runs with one call over as_host(...) (examples/4-booking/serve.py).
        "threetears.evals.transports.fastmcp.mount_fastmcp",
        # A judge's rubric dimension, which states a scale other than pass/fail.
        "threetears.evals.schema.RubricDim",
        "threetears.evals.analysis.EvalSummary",
        "threetears.evals.analysis.Report",
        # What a summary says each apparatus role was asked for and actually sent, which a host's own surface reads to
        # show the gap: the reading, its state and its shares.
        "threetears.evals.analysis.RequestSettingsReading",
        "threetears.evals.analysis.SentSettingsState",
        "threetears.evals.analysis.SentSettingsShare",
        "threetears.evals.quick.measure",
        # Reading runs back, as the serving guide and the examples do.
        "threetears.evals.ops.scope_history",
        "threetears.evals.run.list_results",
        "threetears.evals.run.get_result_trace",
        # Analysing a campaign and rendering its report, as the examples do.
        "threetears.evals.analysis.inspect_campaign_bundle",
        "threetears.evals.analysis.generate_analysis",
        "threetears.evals.analysis.EVAL_ANALYSIS_GEN_DEFAULT",
        # The frozen bundle an analysis is written from, reloaded from its file (`from_json`) to write over it again.
        "threetears.evals.analysis.AnalysisContextBundle",
        "threetears.evals.analysis.report_markdown",
        "threetears.evals.analysis.report_html",
        "threetears.evals.analysis.format_number",
        "threetears.evals.analysis.format_signed",
        "threetears.evals.analysis.ChartBlock",
        "threetears.evals.kernel.EvalAnalysis",
        # The negotiation every completion client shares, the adapter's and a host's own: what a client of its own
        # calls to never fail a request for a setting a model lacks (the host guide's own-client section), the
        # values it reads and passes, the readers it hands a failure to, and the check its suite runs over it.
        "threetears.evals.negotiation.negotiate",
        "threetears.evals.negotiation.Negotiated",
        # The resend bound a host's wall-clock ceiling around a call is sized from.
        "threetears.evals.negotiation.MAX_RESENDS",
        "threetears.evals.negotiation.LearnedSettings",
        "threetears.evals.negotiation.WireSettings",
        "threetears.evals.negotiation.SettingFacts",
        "threetears.evals.negotiation.ProviderFamily",
        "threetears.evals.negotiation.SettingRefusal",
        "threetears.evals.negotiation.closest_settings",
        "threetears.evals.negotiation.facts_from_router_catalogue",
        "threetears.evals.negotiation.wire_settings_for",
        "threetears.evals.negotiation.read_router_refusal",
        "threetears.evals.negotiation.read_anthropic_refusal",
        "threetears.evals.negotiation.read_openai_refusal",
        "threetears.evals.testing.check_refusal_negotiation",
        "threetears.evals.testing.NegotiationConformanceFailure",
        "threetears.evals.testing.RouterTransport",
        # What a host's own Anthropic client builds its request's `thinking` from, and the budget floor it holds a
        # budget to; and the settings a refusal can name, which a host's own reader of its provider's refusals returns.
        "threetears.evals.negotiation.anthropic_thinking",
        "threetears.evals.negotiation.ANTHROPIC_MIN_THINKING_BUDGET",
        "threetears.evals.negotiation.RefusedSetting",
        # The shipped completion client: a host that brings no client of its own binds to these. `Model` is the
        # client (also exported from the package root), `Clients` the role factory, `describe_failure` its
        # failure describer, and `ProviderError` the one error a host catches; `Catalogue` is what models share.
        "threetears.evals.Model",
        "threetears.evals.providers.Model",
        "threetears.evals.providers.Clients",
        "threetears.evals.providers.Catalogue",
        # The timeout the catalogue fetches with on its own, which a host handing it an `http` client matches.
        "threetears.evals.providers.CATALOGUE_FETCH_TIMEOUT_S",
        "threetears.evals.providers.describe_failure",
        "threetears.evals.providers.ProviderError",
        # The words the client writes as a completion's `price_source`, which a host reading spend back compares with.
        "threetears.evals.providers.PriceSource",
        # What a price-catalogue read found, which a host writes its own pricing-outage line from.
        "threetears.evals.providers.PriceRefresh",
        # A declared question and its decision rule, typed: what a host writes to have the page settle keep or switch.
        "threetears.evals.kernel.Question",
        "threetears.evals.kernel.DecisionRule",
        "threetears.evals.kernel.RateDrop",
        "threetears.evals.kernel.CostLower",
    }
)


#: The rest of the declared surface: every public name beyond :data:`HOST_CONTRACT` that a reader is told about, by
#: ``root.name``. Each moves only with a ``MIGRATIONS.toml`` record, never shimmed. With the host contract it is what
#: ``docs/reference.md`` documents and ``docs/surface.md`` maps by rung, and it is held under :data:`DECLARED_CEILING`.
#:
#: The ruling is what someone is shown or already binds: every name a topic guide, the README, an example or the
#: courier host (``tests/fixtures/courierhost/``) imports (``tests/test_api_tiers.py`` holds it), the quick path's whole
#: root, and what Discodon's production code imports, the one production host. The toy host
#: (``tests/fixtures/toyhost/``) exercises the whole contract and reaches past it, so a name it alone imports is listed
#: here by choice, not held here: ``tests/test_api_tiers.py`` holds its imports to the public roots only. A name a
#: declared signature reaches but nobody imports is ``provisional``: still exported, so the surface stays closed
#: (``tests/test_public_surface_is_closed.py``), and promoted here when someone binds to it.
STABLE: frozenset[str] = frozenset(
    {
        # Rungs 0 and 1: the command line a TOML file or a host's own entry runs through, its exit codes, and the
        # quick path's seams beside the root's names: a grader, a measure, a world's runner and outcome, a calibration.
        "threetears.evals.quick.EXIT_FAILED",
        "threetears.evals.quick.EXIT_GATE_FAILED",
        "threetears.evals.quick.EXIT_OK",
        "threetears.evals.quick.EXIT_REFUSED",
        "threetears.evals.quick.EXIT_RUN_DID_NOT_COMPLETE",
        "threetears.evals.quick.Grader",
        "threetears.evals.quick.HostCommand",
        "threetears.evals.quick.HostFactory",
        "threetears.evals.quick.JudgeCalibration",
        "threetears.evals.quick.Measure",
        "threetears.evals.quick.WorldOutcome",
        "threetears.evals.quick.WorldRunner",
        "threetears.evals.quick.run_cli",
        # What a `Result` hands back when it is read case by case, verdict by verdict, or label by label.
        "threetears.evals.analysis.Verdict",
        # How a report speaks: the prose lint every rendered sentence and a memo's are held to, the one seam an
        # identifier passes on its way to a reader, and the glossary of every term of art a page links.
        "threetears.evals.analysis.GLOSSARY",
        "threetears.evals.analysis.lint_prose",
        # Rung 2: judges and calibration. Repeating, re-judging and rating a judge's results, and the agreement, drift,
        # evidence-tier and profile readings over them, as `docs/judges-and-calibration.md` runs them.
        "threetears.evals.analysis.inter_judge_agreement",
        "threetears.evals.analysis.judge_agreement",
        "threetears.evals.analysis.judge_drift",
        "threetears.evals.analysis.judge_evidence_tiers",
        "threetears.evals.analysis.judge_key",
        "threetears.evals.analysis.judge_kind_readings",
        "threetears.evals.analysis.judge_self_agreement",
        "threetears.evals.analysis.read_judge_temperatures",
        "threetears.evals.analysis.tier_sentence",
        "threetears.evals.kernel.AuthoredJudgeCase",
        "threetears.evals.kernel.AuthoredVerdict",
        "threetears.evals.kernel.EvalJudgeProfile",
        "threetears.evals.kernel.JUDGE_KIND",
        "threetears.evals.kernel.JobLeases",
        "threetears.evals.ops.JudgeProfilesRecord",
        "threetears.evals.ops.judge_profiles_record",
        "threetears.evals.run.JudgeRepeatReport",
        "threetears.evals.run.JudgeService",
        "threetears.evals.run.estimate_judge_repeat",
        "threetears.evals.run.freeze_judge_cases",
        "threetears.evals.run.judge_repeat_work",
        "threetears.evals.run.judge_temperature_work",
        "threetears.evals.run.rate_result",
        # The one door a stored verdict is graded again through: check dims from the trace, failed judged dims asked
        # again (the successor of the re-check `.27` removed; a production host's ruling binds it).
        "threetears.evals.run.regrade",
        "threetears.evals.run.run_job_work",
        "threetears.evals.run.second_judge_work",
        "threetears.evals.schema.CalibrationRating",
        "threetears.evals.schema.DEFAULT_JUDGE_TEMPERATURE",
        "threetears.evals.schema.RubricScore",
        "threetears.evals.schema.SecondJudge",
        "threetears.evals.schema.label_key_of",
        # Rung 3: comparing arms and reading the outcome. A campaign's design, its report bundle and frontier, the gate
        # a CI step reads, and the classifier measures `docs/building-an-inventory.md` builds on.
        "threetears.evals.analysis.GATE_TOKENS",
        "threetears.evals.analysis.GateResult",
        "threetears.evals.analysis.assemble_context_bundle",
        "threetears.evals.analysis.campaign_report",
        "threetears.evals.analysis.create_campaign",
        "threetears.evals.analysis.frontier",
        "threetears.evals.analysis.gate_verdicts",
        "threetears.evals.kernel.BarOverride",
        "threetears.evals.kernel.CONFUSION_CELL_MEASURE",
        "threetears.evals.kernel.CampaignDesign",
        "threetears.evals.kernel.ControlDeclaration",
        "threetears.evals.kernel.EvalInsight",
        "threetears.evals.kernel.MATCH_MEASURE",
        "threetears.evals.kernel.SweptAxis",
        "threetears.evals.kernel.confusion_cell",
        "threetears.evals.run.get_run",
        "threetears.evals.run.list_runs",
        "threetears.evals.schema.LatencyMetrics",
        # Rung 4: a world, a log and a host of one's own. What a kind, a launcher and a world are built from beyond
        # the host contract, as the agent guide and the example hosts under `tests/fixtures/` build them.
        "threetears.evals.kernel.MeasureFamily",
        "threetears.evals.kernel.host.ActsOn",
        "threetears.evals.kernel.host.ApparatusError",
        "threetears.evals.kernel.host.Interval",
        "threetears.evals.kernel.host.Ordinal",
        "threetears.evals.kernel.host.RolePins",
        "threetears.evals.kernel.host.StyleProfile",
        "threetears.evals.kernel.host.Triggered",
        "threetears.evals.kernel.host.WorldConformanceError",
        "threetears.evals.kernel.host.check_world_conformance",
        "threetears.evals.run.ArmPlan",
        "threetears.evals.run.ArmPrice",
        "threetears.evals.run.ArmQuote",
        "threetears.evals.run.CellExecutor",
        "threetears.evals.run.FidelityContract",
        "threetears.evals.run.GoalCheckUnevaluable",
        "threetears.evals.run.create_template",
        "threetears.evals.run.grade_goal_checks",
        "threetears.evals.run.require_candidate_model",
        "threetears.evals.schema.CallLedger",
        "threetears.evals.schema.BoundCompletionClient",
        "threetears.evals.schema.CellIdentity",
        "threetears.evals.schema.CellTrace",
        "threetears.evals.schema.ControlEndState",
        "threetears.evals.schema.GoalCheckControl",
        "threetears.evals.schema.GoalCheckControls",
        "threetears.evals.schema.GoalStateOutcome",
        "threetears.evals.schema.JudgeEvidence",
        "threetears.evals.schema.Precondition",
        "threetears.evals.schema.RecordedCall",
        "threetears.evals.schema.WorldPlacement",
        # Serving: the action catalogue a transport drives, the FastMCP mount, and the regression watch a
        # deployment runs.
        "threetears.evals.actions.Action",
        "threetears.evals.actions.ActionCatalogue",
        "threetears.evals.actions.ActionOutcome",
        "threetears.evals.actions.Caller",
        "threetears.evals.actions.ConformanceStage",
        "threetears.evals.actions.eval_catalogue",
        "threetears.evals.actions.standard_tools",
        "threetears.evals.ops.OpsHost",
        "threetears.evals.ops.RegressionWatch",
        "threetears.evals.transports.fastmcp.CallTimedOut",
        "threetears.evals.transports.fastmcp.CallerResolver",
        "threetears.evals.transports.fastmcp.HostResolution",
        "threetears.evals.transports.fastmcp.ToolCall",
        # The shipped client's reply, beside the host-contract client names. Its provider names (`Provider`, a
        # Literal `Model`'s signature reaches) are provisional: no host imports them.
        "threetears.evals.providers.Completion",
        # A production host's further bindings in `threetears.evals.analysis`: what Discodon's production code imports.
        "threetears.evals.analysis.AsRecordedReporterKind",
        "threetears.evals.analysis.REPORTER_KIND",
        "threetears.evals.analysis.ReporterKind",
        "threetears.evals.analysis.ReporterOverlays",
        "threetears.evals.analysis.dollars_text",
        "threetears.evals.analysis.refuse_undeliverable_reporter_template",
        "threetears.evals.analysis.reporter_cell_timeout_s",
        "threetears.evals.analysis.reporter_kind_contract",
        # A production host's further bindings in `threetears.evals.gen`: what Discodon's production code imports.
        "threetears.evals.gen.generate_variations",
        "threetears.evals.gen.plan_draft",
        "threetears.evals.gen.propose_draft",
        # A production host's further bindings in `threetears.evals.kernel`: what Discodon's production code imports.
        "threetears.evals.kernel.ActionSeam",
        "threetears.evals.kernel.AdmittedCall",
        "threetears.evals.kernel.DeliveryRecorder",
        "threetears.evals.kernel.DeliveryReplay",
        "threetears.evals.kernel.DeliverySeam",
        "threetears.evals.kernel.DeliveryTicket",
        "threetears.evals.kernel.EvalJobTimeout",
        "threetears.evals.kernel.ExternalRateTable",
        "threetears.evals.kernel.JUDGE_KIND_CONTRACT",
        "threetears.evals.kernel.JUDGE_KIND_MEASURES",
        "threetears.evals.kernel.MeritAxis",
        "threetears.evals.kernel.OutOfRunBudget",
        "threetears.evals.kernel.PlannedCall",
        "threetears.evals.kernel.PromptDrift",
        "threetears.evals.kernel.Recordable",
        "threetears.evals.kernel.ReplayedDelivery",
        "threetears.evals.kernel.RoleUsageLedger",
        "threetears.evals.kernel.SEED_PROMPTS",
        "threetears.evals.kernel.SeedPrompt",
        "threetears.evals.kernel.SeedSection",
        "threetears.evals.kernel.SeedTemplate",
        "threetears.evals.kernel.ToolWrap",
        "threetears.evals.kernel.extract_json",
        "threetears.evals.kernel.extract_json_array",
        "threetears.evals.kernel.fold_phase_timings",
        "threetears.evals.kernel.prompt_drift",
        "threetears.evals.kernel.seed_digest",
        # A production host's further bindings in `threetears.evals.kernel.host`: what Discodon's production code
        # imports.
        "threetears.evals.kernel.host.EvalCellTimeout",
        "threetears.evals.kernel.host.ResolvesInto",
        "threetears.evals.kernel.host.UnsupportedSchemaError",
        "threetears.evals.kernel.host.check_seed",
        # A production host's further bindings in `threetears.evals.ops`: what Discodon's production code imports.
        "threetears.evals.ops.AnalysisGeneration",
        "threetears.evals.ops.CampaignDefinition",
        "threetears.evals.ops.CampaignLine",
        "threetears.evals.ops.CampaignListing",
        "threetears.evals.ops.EvalSummary",
        "threetears.evals.ops.JobStatus",
        "threetears.evals.ops.JobsStarted",
        "threetears.evals.ops.LaunchArguments",
        "threetears.evals.ops.REPORTER_UNHONOURED_LAUNCH_ARGUMENTS",
        "threetears.evals.ops.ReporterAdmitted",
        "threetears.evals.ops.ReporterModelResolver",
        "threetears.evals.ops.ReporterModelRole",
        "threetears.evals.ops.RunLine",
        "threetears.evals.ops.RunListing",
        "threetears.evals.ops.TemplateListing",
        "threetears.evals.ops.admit_reporter_launch",
        "threetears.evals.ops.campaign_create",
        "threetears.evals.ops.campaigns_list",
        "threetears.evals.ops.history_launch_pricer",
        "threetears.evals.ops.job_cancel",
        "threetears.evals.ops.job_poll",
        "threetears.evals.ops.plan_reporter_arm",
        "threetears.evals.ops.price_largest_reporter_call",
        "threetears.evals.ops.report_read",
        "threetears.evals.ops.run_get",
        "threetears.evals.ops.run_launch",
        "threetears.evals.ops.runs_list",
        "threetears.evals.ops.start_action_job",
        "threetears.evals.ops.start_paid_call",
        "threetears.evals.ops.templates_list",
        # A production host's further bindings in `threetears.evals.run`: what Discodon's production code imports.
        "threetears.evals.run.AbandonedRunSweepReport",
        "threetears.evals.run.CallCeiling",
        "threetears.evals.run.CandidateTurn",
        "threetears.evals.run.CellContext",
        "threetears.evals.run.JUDGE_CALL_ATTEMPTS",
        "threetears.evals.run.JUDGE_MAX_TOKENS",
        "threetears.evals.run.MeteredCallLedger",
        "threetears.evals.run.SeedCorpus",
        "threetears.evals.run.SeedOutcome",
        "threetears.evals.run.SimulatorTurn",
        "threetears.evals.run.TemplateField",
        "threetears.evals.run.TurnDriver",
        "threetears.evals.run.assert_preconditions",
        "threetears.evals.run.build_judge_service",
        "threetears.evals.run.delete_run",
        "threetears.evals.run.drive_conversation",
        "threetears.evals.run.launchable_judge_kind",
        "threetears.evals.run.plan_judge",
        "threetears.evals.run.resolve_judge_pin",
        "threetears.evals.run.run_judge_llm",
        "threetears.evals.run.seed_eval_definitions",
        "threetears.evals.run.sweep_abandoned_runs",
        # A production host's further bindings in `threetears.evals.schema`: what Discodon's production code imports.
        "threetears.evals.schema.ActorPolicy",
        "threetears.evals.schema.AsyncDelivery",
        "threetears.evals.schema.AsyncDeliveryStatus",
        "threetears.evals.schema.AsyncExternalSpend",
        "threetears.evals.schema.CatalogRubricDim",
        "threetears.evals.schema.ConversationSpec",
        "threetears.evals.schema.ConversationStopCause",
        "threetears.evals.schema.DOC_TYPE_FACTS",
        "threetears.evals.schema.DURABLE_DOC_TYPES",
        "threetears.evals.schema.EVAL_DOC_TYPES",
        "threetears.evals.schema.EvalBaseModel",
        "threetears.evals.schema.ExternalSpend",
        "threetears.evals.schema.JudgeConfig",
        "threetears.evals.schema.OutOfRunSpend",
        "threetears.evals.schema.ProviderFailure",
        "threetears.evals.schema.RubricAxis",
        "threetears.evals.schema.RubricProposal",
        "threetears.evals.schema.VariationAxis",
        "threetears.evals.schema.VariationCounts",
        # A production host's further bindings in `threetears.evals.storage`: what Discodon's production code imports.
        "threetears.evals.storage.CoreReset",
        "threetears.evals.storage.MisplacedDocument",
        "threetears.evals.storage.ROUTING_RECORD_DOC_TYPE",
        "threetears.evals.storage.RouteChanged",
        # The rest of what Discodon's production code imports (`tests/fixtures/host_imports/discodon.txt`, held by
        # `tests/test_host_imports_are_declared.py`): nothing a host imports is provisional.
        "threetears.evals.actions.ConformanceSample",
        "threetears.evals.actions.Refusal",
        "threetears.evals.analysis.ABSENT",
        "threetears.evals.analysis.AS_RECORDED_MODEL",
        "threetears.evals.analysis.ReporterLabel",
        "threetears.evals.analysis.first_request",
        "threetears.evals.analysis.freeze_reporter_case",
        "threetears.evals.analysis.prompt_content_version",
        "threetears.evals.analysis.serialize_report",
        "threetears.evals.gen.EVAL_BOUNDARY_GEN_TEMPLATE_DEFAULT",
        "threetears.evals.gen.EVAL_PROPOSER_TEMPLATE_DEFAULT",
        "threetears.evals.kernel.ACCURACY_MEASURE",
        "threetears.evals.kernel.ENGINE_FAMILIES",
        "threetears.evals.kernel.FIND_CAMPAIGN",
        "threetears.evals.kernel.FIND_RUN",
        "threetears.evals.kernel.INCOMPLETE_STOP_REASONS",
        "threetears.evals.kernel.METRIC_DESCRIPTORS",
        "threetears.evals.kernel.MeasureSummary",
        "threetears.evals.kernel.TURN_BUDGET_ENDED_KEY",
        "threetears.evals.kernel.WorldSessionError",
        "threetears.evals.kernel.classifier_label_measure",
        "threetears.evals.kernel.count_substituted",
        "threetears.evals.kernel.count_substituted_deliveries",
        "threetears.evals.kernel.host.BarRegistrationError",
        "threetears.evals.kernel.host.CANDIDATE_MODEL_LEVER",
        "threetears.evals.kernel.host.MeasureRegistrationError",
        "threetears.evals.kernel.host.WorldConformanceReport",
        "threetears.evals.kernel.judge_case_of",
        "threetears.evals.kernel.production_replicating_cost",
        "threetears.evals.kernel.resolve_variant_identity",
        "threetears.evals.ops.CaseSetLine",
        "threetears.evals.ops.CaseSetMint",
        "threetears.evals.ops.JobHandle",
        "threetears.evals.ops.JudgeCaseAuthor",
        "threetears.evals.ops.await_job",
        "threetears.evals.ops.case_set_mint",
        "threetears.evals.ops.case_sets_list",
        "threetears.evals.ops.judge_case_author",
        "threetears.evals.ops.reporter_case_archive",
        "threetears.evals.ops.run_job_id",
        "threetears.evals.run.LaunchGroup",
        "threetears.evals.run.callers_missing_the_constructor",
        "threetears.evals.run.launch_as_group",
        "threetears.evals.run.mint_case_set",
        "threetears.evals.run.rejudge_result",
        "threetears.evals.run.resolve_constructor",
        "threetears.evals.schema.CORE_RESET_DOC_TYPES",
        "threetears.evals.schema.CaseSetRef",
        "threetears.evals.schema.utc_now_iso",
        "threetears.evals.testing.JobsConformanceCase",
        "threetears.evals.transports.fastmcp.CatalogueTool",
    }
)

#: The most names the declared surface (:data:`HOST_CONTRACT` and :data:`STABLE`) may hold: what a reader takes in in
#: one sitting. A name promoted past it means another leaves first; ``tests/test_api_tiers.py`` holds it.
DECLARED_CEILING = 525  # The host's imports set it on 2026-10-15; .33 adds the case rename a host calls.


def tier_of(qualified: str) -> Tier:
    """A public name's tier, by ``root.name``: ``host-contract`` when :data:`HOST_CONTRACT` lists it, ``stable`` when
    :data:`STABLE` does, otherwise ``provisional``."""
    if qualified in HOST_CONTRACT:
        return "host-contract"
    return "stable" if qualified in STABLE else "provisional"


def declared(tier: Tier) -> bool:
    """Whether a tier is part of the declared surface: promised, documented in ``docs/reference.md`` and held to its
    signature by the snapshot."""
    return tier != "provisional"


def installed_surface() -> dict[str, SurfaceEntry]:
    """``root.name`` → its entry, with its tier, for every public name the installed package exports."""
    surface = {}
    for root, name, obj in public_names():
        qualified = f"{root}.{name}"
        tier = tier_of(qualified)
        entry = surface_entry(root, name, obj)
        # A provisional name's signature is not a promise, so it is not recorded and a change to it is no diff here.
        surface[qualified] = dataclasses.replace(entry, tier=tier, signature=entry.signature if declared(tier) else "")
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

#: The tier map the wheel ships (``threetears.evals.surface_tier`` reads it): every public name and its tier, the
#: snapshot's tiers alone. Generated beside the snapshot, since ``api/`` is not installed and a host gates its imports
#: against the build it runs.
TIER_MAP_PATH = PACKAGE_ROOT / "src" / "threetears" / "evals" / "surface_tiers.json"

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
#: each entry's ``tier``; 3 split ``closure`` into ``stable`` and ``provisional``, whose entries carry no signature.
SNAPSHOT_FORMAT = 3


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
            # An entry's tier is never defaulted on a read: a missing one would read as ``provisional`` and drop a
            # promise without a diff saying so.
            raise SystemExit(
                f"{path} records {name} with tier {entry.get('tier')!r}, not one of {TIERS}; regenerate it"
            )
    return {name: SurfaceEntry(**entry) for name, entry in data["names"].items()}


def render_snapshot(surface: dict[str, SurfaceEntry]) -> str:
    """The snapshot file: one entry per ``root.name``, in walk order, so a change is a small, readable diff."""
    names = {name: {"kind": e.kind, "signature": e.signature, "tier": e.tier} for name, e in surface.items()}
    return json.dumps({"format": SNAPSHOT_FORMAT, "names": names}, indent=2, ensure_ascii=False) + "\n"


def render_tier_map(surface: dict[str, SurfaceEntry]) -> str:
    """The shipped tier map: each ``root.name`` and its tier, in the snapshot's order, one name a line."""
    tiers = {name: entry.tier for name, entry in surface.items()}
    return json.dumps({"format": 1, "tiers": tiers}, indent=0, ensure_ascii=False) + "\n"


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
                    f"{name} moved from tier {was.tier} to {now.tier} (HOST_CONTRACT and STABLE in "
                    f"scripts/public_surface.py). "
                    f"Run `{REGENERATE_SNAPSHOT}`; once a release has shipped it as host-contract, leaving that tier "
                    f"is a breaking change for the newest section's `notes`.",
                    True,
                )
            )
    return findings


def tier_findings(installed: dict[str, SurfaceEntry]) -> list[Finding]:
    """Names :data:`HOST_CONTRACT` or :data:`STABLE` lists that no public root exports, which would promise nothing
    while reading as a promise, and names both list, which would leave a name's tier to whichever is read first."""
    return [
        *(
            Finding(
                f"{label} in scripts/public_surface.py lists {name}, which no public root exports: name it at the "
                f"root that exports it now, or drop it once its record in MIGRATIONS.toml says it left.",
                False,
            )
            for label, listed in (("HOST_CONTRACT", HOST_CONTRACT), ("STABLE", STABLE))
            for name in sorted(listed - installed.keys())
        ),
        *(
            Finding(
                f"HOST_CONTRACT and STABLE in scripts/public_surface.py both list {name}: drop it from STABLE.", False
            )
            for name in sorted(HOST_CONTRACT & STABLE)
        ),
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
