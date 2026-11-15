"""What each declared public name of ``3tears-evals`` asks of its reader: its kind, and the budget each kind is held to.

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is not installed. The snapshot generator
(``generate_api_snapshot.py``) writes each declared name's kind into the tier map the wheel ships, beside its tier,
and ``generate_surface.py`` renders ``docs/surface.md`` and ``llms.txt`` from the same kinds.

A declared name (``host-contract`` or ``stable``) is exactly one of five kinds:

- ``typed``: a newcomer writes it in an eval file, a ladder entry point. Every one is in :data:`TYPED` with the rung
  that first teaches it.
- ``bound``: a host implements it or hands it to the engine: a protocol, port, registry, store, client, kind seam.
- ``invoked``: an operation called by name, from host code or by an agent through the action catalogue.
- ``received``: a value someone is handed and reads: a result or stored-document shape, a literal, a constant, an
  error.
- ``tests``: a conformance kit, a recorder or a guard a host runs in its own tests.

A kind is derived by rule wherever a rule is sound (:func:`kind_by_rule`), and the rest are recorded in
:data:`KIND_ENTRIES`. ``tests/test_api_tiers.py`` fails on a declared name with no kind, on an entry that a rule
already decides, and on an entry for a name that is not declared.
"""

from __future__ import annotations

import annotationlib
import ast
import dataclasses
import importlib
import inspect
import re
import sys
import typing
import warnings
from collections.abc import Iterator, Mapping
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from public_surface import PACKAGE_ROOT, SurfaceEntry, assignment, declared, is_alias, public_names
from pydantic import BaseModel

#: A declared name's kind: what it asks of the person or agent who meets it.
Kind = Literal["typed", "bound", "invoked", "received", "tests"]

#: The kinds, in the order a reader meets them.
KINDS: tuple[Kind, ...] = typing.get_args(Kind)

#: Every ``typed`` name, by ``root.name``, with the rung of the ladder (``docs/ladder.md``) that first teaches it.
#: Rungs 0 and 1 are a first eval. A name a ladder example imports is here at or below that example's rung
#: (:data:`LADDER_IMPORTS`); the rest are names the ladder's pages teach beside them.
TYPED: Mapping[str, int] = MappingProxyType(
    {
        # Rung 1: an eval, the client, what counts as right, and what a run hands back.
        "threetears.evals.Eval": 1,
        "threetears.evals.Model": 1,
        "threetears.evals.grader": 1,
        "threetears.evals.Answer": 1,
        "threetears.evals.Result": 1,
        "threetears.evals.quick.Grader": 1,
        "threetears.evals.quick.Measure": 1,
        "threetears.evals.quick.measure": 1,
        # Rung 2: a judge, its examples, and its calibration.
        "threetears.evals.Judge": 2,
        "threetears.evals.Example": 2,
        "threetears.evals.quick.JudgeCalibration": 2,
        # Rung 4: a world, a log, a host of one's own and serving it.
        "threetears.evals.World": 4,
        "threetears.evals.Log": 4,
        "threetears.evals.quick.WorldRunner": 4,
        "threetears.evals.quick.WorldOutcome": 4,
        "threetears.evals.quick.as_host": 4,
        "threetears.evals.quick.run_cli": 4,
        "threetears.evals.quick.HostCommand": 4,
        "threetears.evals.quick.HostFactory": 4,
        "threetears.evals.transports.fastmcp.mount_fastmcp": 4,
        # Rung 5: the report writer evaluated.
        "threetears.evals.harness.AnalystSkill": 5,
        "threetears.evals.harness.Writer": 5,
        "threetears.evals.harness.WriterEval": 5,
    }
)

#: The ladder's examples, each directory under ``examples/`` with its rung.
LADDER_EXAMPLES: Mapping[str, int] = MappingProxyType(
    {"1-triage": 1, "2-replies": 2, "3-invoices": 3, "4-booking": 4, "5-writer": 5}
)

#: What each ladder example imports from the package, read from its files: ``tests/test_api_tiers.py`` holds this to
#: the examples, and every name here to :data:`TYPED` at or below the example's rung.
LADDER_IMPORTS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "1-triage": frozenset({"threetears.evals.Eval", "threetears.evals.Model"}),
        "2-replies": frozenset(
            {"threetears.evals.Eval", "threetears.evals.Example", "threetears.evals.Judge", "threetears.evals.Model"}
        ),
        "3-invoices": frozenset({"threetears.evals.Eval", "threetears.evals.Model", "threetears.evals.grader"}),
        "4-booking": frozenset(
            {
                "threetears.evals.Eval",
                "threetears.evals.Model",
                "threetears.evals.World",
                "threetears.evals.quick.as_host",
                "threetears.evals.transports.fastmcp.mount_fastmcp",
            }
        ),
        "5-writer": frozenset(
            {
                "threetears.evals.Model",
                "threetears.evals.harness.AnalystSkill",
                "threetears.evals.harness.Writer",
                "threetears.evals.harness.WriterEval",
            }
        ),
    }
)

#: The kind of each declared name no rule decides (:func:`kind_by_rule` returns ``None`` for it): a model or a
#: dataclass that a declared callable takes as a parameter, or that no declared name names directly, so it may be a
#: shape someone is handed or a seam a host builds and hands the engine. An entry for a name a rule decides is refused,
#: so a rule and this list never disagree.
KIND_ENTRIES: Mapping[str, Kind] = MappingProxyType(
    {
        # The host and the seams it wires: what a host builds and hands every entrypoint, its routes, its own
        # actions and tools, and what its resolver answers.
        "threetears.evals.kernel.host.EvalHost": "bound",
        "threetears.evals.kernel.host.HostProfile": "bound",
        "threetears.evals.kernel.host.TemplateChecks": "bound",
        "threetears.evals.run.LaunchHost": "bound",
        "threetears.evals.run.KindWiring": "bound",
        "threetears.evals.ops.OpsHost": "bound",
        "threetears.evals.storage.DocTypeRoute": "bound",
        "threetears.evals.actions.Action": "bound",
        "threetears.evals.transports.fastmcp.CatalogueTool": "bound",
        "threetears.evals.transports.fastmcp.HostResolution": "bound",
        # Engine services a host builds and runs: the conversation scheduler, the regression watch, a role's ledger.
        "threetears.evals.run.TurnDriver": "bound",
        "threetears.evals.ops.RegressionWatch": "bound",
        "threetears.evals.kernel.RoleUsageLedger": "bound",
        # The markers a host writes on its own kind's level fields, which the engine reads off its model.
        "threetears.evals.kernel.host.ActsOn": "bound",
        "threetears.evals.kernel.host.Interval": "bound",
        "threetears.evals.kernel.host.MemberActsOn": "bound",
        "threetears.evals.kernel.host.Ordinal": "bound",
        "threetears.evals.kernel.host.ResolvesInto": "bound",
        "threetears.evals.kernel.host.Triggered": "bound",
        # Shapes: records, declarations and values a callable takes as well as hands back, or that only another
        # shape or a constant names. Each is read, and none is a seam.
        "threetears.evals.RootExtra": "received",
        "threetears.evals.schema.ActorPolicy": "received",
        "threetears.evals.schema.CalibrationRating": "received",
        "threetears.evals.schema.CallLedger": "received",
        "threetears.evals.schema.CaseSetRef": "received",
        "threetears.evals.schema.CatalogRubricDim": "received",
        "threetears.evals.schema.CellIdentity": "received",
        "threetears.evals.schema.ClientRequestSettings": "received",
        "threetears.evals.schema.EvalResult": "received",
        "threetears.evals.schema.EvalRun": "received",
        "threetears.evals.schema.EvalTemplate": "received",
        "threetears.evals.schema.EvalTestCase": "received",
        "threetears.evals.schema.EvalTrace": "received",
        "threetears.evals.schema.ExternalSpend": "received",
        "threetears.evals.schema.IntervalScale": "received",
        "threetears.evals.schema.JudgeConfig": "received",
        "threetears.evals.schema.JudgeEvidence": "received",
        "threetears.evals.schema.OutOfRunSpend": "received",
        "threetears.evals.schema.RecordedCall": "received",
        "threetears.evals.schema.RubricDim": "received",
        "threetears.evals.schema.RubricProposal": "received",
        "threetears.evals.schema.SecondJudge": "received",
        "threetears.evals.schema.SubjectSnapshot": "received",
        "threetears.evals.schema.ThinkingBinding": "received",
        "threetears.evals.schema.ToolResultClearing": "received",
        "threetears.evals.schema.TurnCeiling": "received",
        "threetears.evals.schema.TurnMessage": "received",
        "threetears.evals.schema.TurnReply": "received",
        "threetears.evals.schema.TurnToolSpec": "received",
        "threetears.evals.schema.WorldSeed": "received",
        "threetears.evals.kernel.AdmittedCall": "received",
        "threetears.evals.kernel.AuthoredJudgeCase": "received",
        "threetears.evals.kernel.CandidateOutput": "received",
        "threetears.evals.kernel.CandidateTelemetry": "received",
        "threetears.evals.kernel.CostLower": "received",
        "threetears.evals.kernel.EvalAnalysis": "received",
        "threetears.evals.kernel.EvalCampaign": "received",
        "threetears.evals.kernel.EvalInsight": "received",
        "threetears.evals.kernel.EvalJudgeProfile": "received",
        "threetears.evals.kernel.ExternalRateTable": "received",
        "threetears.evals.kernel.MeasureFamily": "received",
        "threetears.evals.kernel.MetricDescriptor": "received",
        "threetears.evals.kernel.OutOfRunBudget": "received",
        "threetears.evals.kernel.OverriddenBar": "received",
        "threetears.evals.kernel.PlannedCall": "received",
        "threetears.evals.kernel.RateDrop": "received",
        "threetears.evals.kernel.SeedPrompt": "received",
        "threetears.evals.kernel.SeedTemplate": "received",
        "threetears.evals.kernel.VariantConfig": "received",
        "threetears.evals.kernel.host.Bar": "received",
        "threetears.evals.kernel.host.PassThreshold": "received",
        "threetears.evals.kernel.host.RolePins": "received",
        "threetears.evals.kernel.host.Sweepable": "received",
        "threetears.evals.kernel.host.WorldDimension": "received",
        "threetears.evals.run.ArmQuote": "received",
        "threetears.evals.run.CandidateTurn": "received",
        "threetears.evals.run.Ceilings": "received",
        "threetears.evals.run.CellContext": "received",
        "threetears.evals.run.FidelityContract": "received",
        "threetears.evals.run.JudgeRepeatReport": "received",
        "threetears.evals.run.LaunchRequest": "received",
        "threetears.evals.run.LaunchSettings": "received",
        "threetears.evals.run.MeteredCallTally": "received",
        "threetears.evals.run.RunnerOptions": "received",
        "threetears.evals.run.SeedCorpus": "received",
        "threetears.evals.run.SimulatorTurn": "received",
        "threetears.evals.analysis.ChartBlock": "received",
        "threetears.evals.analysis.Verdict": "received",
        "threetears.evals.ops.CampaignDefinition": "received",
        "threetears.evals.ops.CaseSetMint": "received",
        "threetears.evals.ops.JobsStarted": "received",
        "threetears.evals.ops.JudgeCaseAuthor": "received",
        "threetears.evals.ops.JudgeProfilesRecord": "received",
        "threetears.evals.ops.LaunchArguments": "received",
        "threetears.evals.actions.ActionOutcome": "received",
        "threetears.evals.transports.fastmcp.ToolCall": "received",
        "threetears.evals.negotiation.SettingFacts": "received",
        "threetears.evals.negotiation.SettingRefusal": "received",
        "threetears.evals.negotiation.WireSettings": "received",
        "threetears.evals.providers.Completion": "received",
    }
)

#: The most declared names each kind may hold (information-architecture §8.2), which replaced one ceiling on the whole
#: declared surface. ``received`` has no total: its discipline is that every shape is reached from a ``typed``,
#: ``bound`` or ``invoked`` name (:func:`unreached_shapes`), never hidden to protect a count.
KIND_CEILINGS: Mapping[Kind, int | None] = MappingProxyType(
    {
        "typed": 25,
        "bound": 100,
        # Over the budget of 110 when kinds were first recorded (2026-11-15): every declared function outside the
        # ladder is an operation, and 129 are. This ceiling may only fall, to 110.
        "invoked": 129,
        "received": None,
        # A kit per port a host implements: today's count, which may only fall.
        "tests": 18,
    }
)

#: The received shapes no ``typed``, ``bound`` or ``invoked`` name reached when kinds were first recorded
#: (2026-11-15): each is handed to a host only through a private method, a job's untyped result or a docstring's
#: prose, so no declared signature names it. The list may only shrink: a shape newly unreached fails, and one a
#: signature comes to name must leave the list.
UNREACHED_SHAPES: frozenset[str] = frozenset(
    {
        "threetears.evals.schema.StoreCalledOnEventLoop",
        "threetears.evals.kernel.CandidateKindDefect",
        "threetears.evals.kernel.ProviderRefusedError",
        "threetears.evals.run.Ceilings",
        "threetears.evals.run.FidelityContract",
        "threetears.evals.run.JudgeRepeatReport",
        "threetears.evals.providers.Completion",
        "threetears.evals.providers.PriceSource",
    }
)

#: The most ``typed`` names a first eval (rungs 0 and 1) needs.
FIRST_EVAL_CEILING = 8

#: The most ``typed`` names rungs 0 to 3 need together.
THROUGH_RUNG_3_CEILING = 12

#: The root whose every declared name is a host's test kit.
TESTING_ROOT = "threetears.evals.testing"


# --- the walk: which engine types a public object names, and where ---------------------------------------------

#: Where an engine type sits in what names it.
Position = Literal["parameter", "return", "field", "base", "raises"]

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RAISES = re.compile(r"^:raises ~?([A-Za-z_][A-Za-z0-9_.]*):", re.M)


def _engine_class(value: object) -> bool:
    return inspect.isclass(value) and value.__module__.startswith("threetears.evals")


@cache
def _aliases() -> Mapping[int, object]:
    """Every type alias a public root exports, by ``id``: what an annotation token is matched against."""
    found: dict[int, object] = {}
    for _, _, obj in public_names():
        if is_alias(obj) and not inspect.isclass(obj):
            found[id(obj)] = obj
    return MappingProxyType(found)


@cache
def _plain_aliases() -> frozenset[str]:
    """Each exported type alias that is a bare builtin (``RefusalKind = str``), by name: an annotation names it only in
    its source text, since its value is the builtin."""
    from public_surface import describe

    return frozenset(
        name
        for root, name, obj in public_names()
        if inspect.isclass(obj) and not _engine_class(obj) and describe(root, name, obj).kind == "type alias"
    )


def _namespace(obj: object) -> dict[str, Any]:
    module = sys.modules.get(getattr(obj, "__module__", "") or "")
    return dict(vars(module)) if module else {}


def _types_in(annotation: object, namespace: Mapping[str, Any], _seen: set[int] | None = None) -> Iterator[object]:
    """Each engine class or exported type alias an annotation names, however it is written."""
    _seen = set() if _seen is None else _seen
    if isinstance(annotation, (list, tuple)):  # a ``Callable``'s parameter list
        for item in annotation:
            yield from _types_in(item, namespace, _seen)
        return
    if annotation is inspect.Parameter.empty or annotation is None:
        return
    if isinstance(annotation, str):
        for token in _IDENT.findall(annotation):
            value = namespace.get(token)
            if _engine_class(value) or id(value) in _aliases() or (token in _plain_aliases() and value is not None):
                yield value
            elif value is not None and is_alias(value) and id(value) not in _seen:
                # An alias no root exports (``Scale``) is read through to the types it unions.
                _seen.add(id(value))
                yield from _types_in(value, namespace, _seen)
        return
    if isinstance(annotation, annotationlib.ForwardRef):
        yield from _types_in(annotation.__forward_arg__, namespace, _seen)
        return
    if id(annotation) in _aliases():
        yield annotation
        return
    if _engine_class(annotation):
        yield annotation
        return
    if isinstance(annotation, typing.TypeAliasType):
        if id(annotation) not in _seen:
            _seen.add(id(annotation))
            yield from _types_in(annotation.__value__, namespace, _seen)
        return
    for arg in typing.get_args(annotation):
        yield from _types_in(arg, namespace, _seen)


#: An annotation is read twice: as its source text, whose words name an alias even when its value is a builtin
#: (``RefusalKind = str``), and as its value, which resolves a name imported only for type checking where it can.
_FORMATS = (annotationlib.Format.STRING, annotationlib.Format.FORWARDREF)


def _signatures(function: Any) -> Iterator[inspect.Signature]:
    for annotation_format in _FORMATS:
        try:
            yield inspect.signature(function, annotation_format=annotation_format)
        except TypeError, ValueError, NameError:
            continue


def _callable_edges(function: Any, namespace: Mapping[str, Any]) -> Iterator[tuple[Position, object]]:
    # A function's names resolve where it was written, whatever ``__module__`` its class was given for display.
    namespace = getattr(inspect.unwrap(function), "__globals__", None) or namespace
    for found in _signatures(function):
        for parameter in found.parameters.values():
            if parameter.name in ("self", "cls"):
                continue
            for value in _types_in(parameter.annotation, namespace):
                yield "parameter", value
        for value in _types_in(found.return_annotation, namespace):
            yield "return", value
    yield from _raises_edges(inspect.getdoc(function) or "", namespace)


@cache
def _exceptions_by_name() -> Mapping[str, tuple[type, ...]]:
    found: dict[str, list[type]] = {}
    for _, name, obj in public_names():
        if inspect.isclass(obj) and issubclass(obj, BaseException) and obj not in found.get(name, []):
            found.setdefault(name, []).append(obj)
    return MappingProxyType({name: tuple(each) for name, each in found.items()})


def _raises_edges(doc: str, namespace: Mapping[str, Any]) -> Iterator[tuple[Position, object]]:
    for entry in _RAISES.finditer(doc):
        name = entry.group(1).rsplit(".", 1)[-1]
        value = namespace.get(name)
        for candidate in (value,) if _engine_class(value) else _exceptions_by_name().get(name, ()):
            yield "raises", candidate


def _shape_class(value: type) -> bool:
    """A model, a dataclass or a typed dict: a class whose constructor is its fields."""
    return issubclass(value, BaseModel) or dataclasses.is_dataclass(value) or typing.is_typeddict(value)


def _class_namespace(klass: type) -> dict[str, Any]:
    """Where a class's annotations resolve: the module its ``__module__`` names when that module holds it, else the
    module one of its own methods was written in."""
    module = sys.modules.get(klass.__module__)
    if module is not None and getattr(module, klass.__name__, None) is klass:
        return dict(vars(module))
    for member_name, member in vars(klass).items():
        if inspect.isfunction(member) and not member_name.startswith("__"):
            return dict(member.__globals__)
    return _namespace(klass)


def _class_edges(value: type) -> Iterator[tuple[Position, object]]:
    for base in value.__mro__[1:]:
        if _engine_class(base):
            yield "base", base
    for klass in value.__mro__:
        if not klass.__module__.startswith("threetears.evals"):
            continue
        namespace = _class_namespace(klass)
        for annotation_format in _FORMATS:
            try:
                annotations = annotationlib.get_annotations(klass, format=annotation_format)
            except NameError, TypeError:
                continue
            for field_name, annotation in annotations.items():
                if not field_name.startswith("_"):
                    for found in _types_in(annotation, namespace):
                        yield "field", found
        for member_name, member in vars(klass).items():
            if member_name.startswith("_") and member_name not in ("__init__", "__call__"):
                continue
            if member_name == "__init__" and _shape_class(value):
                continue  # a shape's constructor is its fields, read above
            if isinstance(member, property) and member.fget is not None:
                yield from _callable_edges(member.fget, namespace)
            elif isinstance(member, (staticmethod, classmethod)):
                yield from _callable_edges(member.__func__, namespace)
            elif inspect.isfunction(member):
                yield from _callable_edges(member, namespace)
    yield from _raises_edges(inspect.getdoc(value) or "", _class_namespace(value))


def edges(value: object) -> tuple[tuple[Position, object], ...]:
    """Each engine type a public object names, with where: a parameter or return of it or of a method, a field, or an
    exception its docstring says it raises."""
    if inspect.isclass(value):
        return tuple(_class_edges(value))
    if id(value) in _aliases():
        namespace = _alias_namespace(value)
        parts = (value.__value__,) if isinstance(value, typing.TypeAliasType) else typing.get_args(value)
        return tuple(("field", found) for part in parts for found in _types_in(part, namespace) if found is not value)
    if callable(value):
        return tuple(_callable_edges(value, _namespace(value)))
    return ()


@cache
def _alias_homes() -> Mapping[int, str]:
    """Each exported type alias's defining module, by ``id``: an alias's own ``__module__`` may name its origin's."""
    from public_surface import defining_module

    found: dict[int, str] = {}
    for root, name, obj in public_names():
        if id(obj) in _aliases() and id(obj) not in found:
            home = defining_module(root, name)
            found[id(obj)] = home[0] if home is not None else root
    return MappingProxyType(found)


def _alias_namespace(value: object) -> dict[str, Any]:
    module = sys.modules.get(_alias_homes().get(id(value), ""))
    return dict(vars(module)) if module else {}


# --- kinds -------------------------------------------------------------------------------------------------------


def _objects(names: typing.Iterable[str]) -> dict[str, Any]:
    wanted = set(names)
    found: dict[str, Any] = {}
    for root, name, obj in public_names():
        if f"{root}.{name}" in wanted:
            found[f"{root}.{name}"] = obj
    return found


def constant_edges(qualified: str) -> tuple[tuple[Position, object], ...]:
    """Each engine type a public constant's annotation names: what holding it hands a reader (``ROOT_EXTRAS`` hands
    a ``RootExtra``)."""
    from public_surface import defining_module

    root, name = qualified.rsplit(".", 1)
    found = defining_module(root, name)
    _, node, _ = assignment(root, name)
    if found is None or not isinstance(node, ast.AnnAssign):
        return ()
    module = sys.modules.get(found[0])
    namespace = dict(vars(module)) if module else {}
    return tuple(("field", each) for each in _types_in(ast.unparse(node.annotation), namespace))


def _positions(objects: Mapping[str, Any]) -> dict[int, set[Position]]:
    """Every engine type the declared names name directly, by ``id``, with each position it is named in."""
    found: dict[int, set[Position]] = {}
    for obj in objects.values():
        for position, each in edges(obj):
            if each is not obj:  # a class's own method returning the class says nothing of who holds it
                found.setdefault(id(each), set()).add(position)
    return found


def _callable_alias(obj: object) -> bool:
    origin = typing.get_origin(obj.__value__ if isinstance(obj, typing.TypeAliasType) else obj)
    return origin is not None and getattr(origin, "__name__", "") == "Callable"


def kind_by_rule(qualified: str, obj: Any, entry: SurfaceEntry, positions: Mapping[int, set[Position]]) -> Kind | None:
    """A declared name's kind where a rule decides it, in this order, or ``None`` where none does.

    1. In :data:`TYPED`: ``typed``.
    2. Under :data:`TESTING_ROOT`, or a conformance check, its report, error, case or sample: ``tests``.
    3. A protocol: ``bound``.
    4. A function: ``invoked``.
    5. An exception, an enum, a literal or a constant: ``received``.
    6. A type alias over a ``Callable``: ``bound`` (a host hands the engine the callable); any other: ``received``.
    7. A model, dataclass or typed dict a declared name names as a field, a return or a base, and none takes as a
       parameter, so it is reached only as something handed back: ``received``.
    8. Any other class: ``bound`` (an engine implementation a host builds and hands the engine: a store, a registry,
       a client, a service).
    """
    if qualified in TYPED:
        return "typed"
    if qualified.startswith(f"{TESTING_ROOT}.") or "conformance" in qualified.rsplit(".", 1)[1].lower():
        return "tests"
    kind = entry.kind
    if kind == "protocol":
        return "bound"
    if kind in ("function", "async function"):
        return "invoked"
    if kind in ("exception", "enum", "literal") or kind.startswith("constant"):
        return "received"
    if kind == "type alias":
        return "bound" if _callable_alias(obj) else "received"
    if kind in ("model", "dataclass", "typed dict"):
        named = positions.get(id(obj), set())
        return "received" if named and "parameter" not in named else None
    if kind == "class":
        return "bound"
    return None


@dataclasses.dataclass(frozen=True)
class Kinds:
    """Every declared name's kind, and what is wrong with the record of them.

    :ivar of: ``root.name`` → its kind, for every declared name that has one, in the surface's order.
    :ivar problems: Each declared name with no kind, each :data:`KIND_ENTRIES` entry a rule already decides, and each
        entry for a name that is not declared.
    """

    of: Mapping[str, Kind]
    problems: tuple[str, ...]


def declared_kinds(surface: Mapping[str, SurfaceEntry]) -> Kinds:
    """Every declared name's kind: by rule (:func:`kind_by_rule`), else by :data:`KIND_ENTRIES`."""
    names = [name for name, entry in surface.items() if declared(entry.tier)]
    objects = _objects(names)
    positions = _positions(objects)
    of: dict[str, Kind] = {}
    problems: list[str] = []
    for name in names:
        ruled = kind_by_rule(name, objects.get(name), surface[name], positions)
        entered = KIND_ENTRIES.get(name)
        if ruled is not None and entered is not None:
            problems.append(f"{name} is {ruled} by rule and also entered as {entered}: drop it from KIND_ENTRIES")
        kind = ruled or entered
        if kind is None:
            problems.append(f"{name} has no kind: no rule decides it, so record it in KIND_ENTRIES")
        else:
            of[name] = kind
    problems += [f"{name} is in KIND_ENTRIES but is not declared" for name in sorted(set(KIND_ENTRIES) - set(names))]
    return Kinds(MappingProxyType(of), tuple(problems))


def counts(kinds: Mapping[str, Kind]) -> dict[Kind, int]:
    """How many declared names each kind holds."""
    return {kind: sum(1 for each in kinds.values() if each == kind) for kind in KINDS}


# --- shapes: what reaches each received name ---------------------------------------------------------------------


def _is_shape(entry: SurfaceEntry) -> bool:
    """A received name that is a type (a constant is a value, and nothing names it in a signature)."""
    return not entry.kind.startswith("constant")


@dataclasses.dataclass(frozen=True)
class Reach:
    """Which ``typed``, ``bound`` or ``invoked`` name reaches each received shape.

    :ivar by: ``root.name`` of a received shape → the declared name nearest it that reaches it: a parameter, return,
        field or raised error of that name, or of a type it reaches, names the shape. ``None`` when nothing does.
    """

    by: Mapping[str, str | None]


def shape_reach(surface: Mapping[str, SurfaceEntry], kinds: Mapping[str, Kind]) -> Reach:
    """Walk outward from every ``typed``, ``bound`` and ``invoked`` name, nearest first, through every engine type it
    names, and record for each received shape the starting name that reached it first."""
    starts = [name for kind in ("typed", "bound", "invoked") for name in kinds if kinds[name] == kind]
    held = [name for name, kind in kinds.items() if kind == "received" and not _is_shape(surface[name])]
    shapes = [name for name, kind in kinds.items() if kind == "received" and _is_shape(surface[name])]
    objects = _objects([*starts, *shapes])
    names_of: dict[int, list[str]] = {}
    for name in shapes:
        names_of.setdefault(id(objects[name]), []).append(name)
    origin: dict[int, str] = {}
    frontier: list[tuple[object, str]] = []
    for name in starts:
        obj = objects[name]
        if id(obj) not in origin:
            origin[id(obj)] = name
            frontier.append((obj, name))
    for name in held:
        for _, found in constant_edges(name):
            if id(found) not in origin:
                origin[id(found)] = name
                frontier.append((found, name))
    while frontier:
        following: list[tuple[object, str]] = []
        for obj, start in frontier:
            for _, found in edges(obj):
                if id(found) not in origin:
                    origin[id(found)] = start
                    following.append((found, start))
        frontier = following
    return Reach(MappingProxyType({name: origin.get(id(objects[name])) for name in shapes}))


def unreached_shapes(reach: Reach) -> list[str]:
    """Every received shape no ``typed``, ``bound`` or ``invoked`` name reaches."""
    return [name for name, by in reach.by.items() if by is None]


def near_duplicate_shapes(surface: Mapping[str, SurfaceEntry], kinds: Mapping[str, Kind]) -> list[tuple[str, ...]]:
    """Received models and dataclasses that declare the same set of fields under different names: each group, of two
    or more fields, in the surface's order. Reported, not refused: a near-duplicate is a finding for review."""
    by_fields: dict[frozenset[str], list[str]] = {}
    objects = _objects(name for name, kind in kinds.items() if kind == "received")
    seen: set[int] = set()
    for name, obj in objects.items():
        if not inspect.isclass(obj) or issubclass(obj, BaseException) or id(obj) in seen:
            continue
        seen.add(id(obj))
        if issubclass(obj, BaseModel):
            fields = frozenset(obj.model_fields)
        elif dataclasses.is_dataclass(obj):
            fields = frozenset(f.name for f in dataclasses.fields(obj))
        else:
            continue
        if len(fields) >= 2:
            by_fields.setdefault(fields, []).append(name)
    return [tuple(group) for group in by_fields.values() if len(group) > 1]


# --- the ladder --------------------------------------------------------------------------------------------------

#: The ladder's page, whose ``## Rung N: ...`` headings title each rung.
LADDER_PAGE = PACKAGE_ROOT / "docs" / "ladder.md"

_RUNG_HEADING = re.compile(r"^## Rung (\d)[a-z]?: (.+)$", re.M)


def rung_titles() -> dict[int, str]:
    """Each rung's title, from the first heading the ladder's page gives it."""
    titles: dict[int, str] = {}
    for number, title in _RUNG_HEADING.findall(LADDER_PAGE.read_text(encoding="utf-8")):
        titles.setdefault(int(number), title)
    return titles


def example_imports(directory: Path) -> frozenset[str]:
    """Every ``root.name`` an example's Python files import from the package."""
    found: set[str] = set()
    for path in sorted(directory.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("threetears.evals"):
                found |= {f"{node.module}.{alias.name}" for alias in node.names}
    return frozenset(found)


def ladder_imports() -> dict[str, frozenset[str]]:
    """What each ladder example imports, read from its files."""
    return {name: example_imports(PACKAGE_ROOT / "examples" / name) for name in LADDER_EXAMPLES}


def objects_of(names: typing.Iterable[str]) -> dict[str, Any]:
    """``root.name`` → the object its root exports, for each name given, in walk order. A one-minor shim's warning is
    not the reader's."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return _objects(names)


def module_of(qualified: str) -> Any:
    """The public root a ``root.name`` is exported from."""
    return importlib.import_module(qualified.rsplit(".", 1)[0])
