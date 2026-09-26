"""AST walkers for audit-details classification enforcement.

**what the walker reads.** Every call to a configured constructor
(``AuditEvent`` by default) or forwarder, bare or attribute-qualified, under the
configured source roots, and its ``details=`` keyword when that is:

- a dict literal, following nested dict literals (and lists or tuples of them)
  so a nested key is judged where the anonymization rule judges it;
- a ``dict(k=v)`` call;
- a local the enclosing function builds from those, plus literal-keyed
  ``name["k"] = ...``, ``name.update({...})`` / ``name.update(k=...)``,
  ``name.setdefault("k", ...)`` and ``name |= {...}``;
- inside a declared forwarder, the forwarder's own parameter -- passed through,
  spread (``{**details, "k": v}``), copied (``dict(details)``) or defaulted
  (``details or {}``) -- because the forwarder's call sites carry the keys.

**what it refuses.** A details argument it cannot read is reported as unreadable,
never passed as clean: a computed key, a ``**spread`` of anything but a forwarded
parameter, a details dict returned by a call, one passed in as the parameter of a
function that is not a declared forwarder, and ``AuditEvent(**fields)``. A wrapper
helper is therefore refused until the repo declares it a forwarder, at which point
its call sites are read instead.

**what it cannot see.** A details dict handed to some other function that adds
keys to it (a name passed out of its function is not followed); an event built by
``model_validate`` or ``model_copy`` rather than by a constructor call; and a
``details`` passed to a forwarder POSITIONALLY, since the walker does not map a
position to a parameter. A nested value that is not a literal (a variable, a call)
is a value, not keys, and is not read.

**which nested keys must be classified.** Only those whose every ancestor key is
safe for the event type: under an unsafe key the whole value is anonymized, keys
included, so nothing beneath it is ever consulted.

**which event types a site is credited for.** A family's safe keys count only for the
event types a site can be shown to publish, resolved statically by
:mod:`threetears.enforcement.audit_details.event_types`: a literal, a module constant
(local or imported from the scanned roots), an attribute on an imported module, a
conditional over those, a helper parameter resolved through every caller, or -- for a
forwarder call that names no event type -- the type the forwarder's own construction
publishes. A key is credited only when it is safe for EVERY resolved value; a site whose event type cannot be resolved gets the platform
set alone, and anything that needs a family is reported.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from threetears.enforcement.common import Violation, callee_names, iter_python_files, parse_python_file

from threetears.enforcement.audit_details.config import DEFAULT_AUDIT_CONSTRUCTORS, AuditDetailsConfig
from threetears.enforcement.audit_details.event_types import (
    EventTypeResolver,
    FunctionNode,
    SourceModule,
    module_name,
    parameter_names,
    source_module,
)

__all__ = [
    "AuditDetailsSite",
    "collect_audit_details_sites",
    "find_audit_details_violations",
    "read_audit_details_sites",
    "unclassified_detail_paths",
]

#: dict methods that add a key when called with a literal.
_ADDING_METHODS: frozenset[str] = frozenset({"update", "setdefault"})


@dataclass(frozen=True)
class AuditDetailsSite:
    """one audit construction or forwarder call, and what the walker could read of its details.

    :ivar lineno: line of the call
    :ivar event_types: every event type the ``event_type`` argument resolves to; empty when
        it is absent or cannot be resolved
    :ivar keys: every details key read, as its path from the top of ``details``
        (``("platform_repointed", "member_count")`` for a nested literal key)
    :ivar unreadable: ``(line, reason)`` for each part of the details argument the walker
        could not read
    """

    lineno: int
    event_types: frozenset[str]
    keys: frozenset[tuple[str, ...]]
    unreadable: tuple[tuple[int, str], ...]


@dataclass
class _Reading:
    """accumulates one site while its details argument is read.

    :ivar keys: key paths read so far
    :ivar unreadable: ``(line, reason)`` pairs recorded so far
    :ivar forwarded: parameter names of every enclosing declared forwarder
    """

    keys: set[tuple[str, ...]] = field(default_factory=set)
    unreadable: list[tuple[int, str]] = field(default_factory=list)
    forwarded: frozenset[str] = frozenset()


def read_audit_details_sites(
    tree: ast.Module,
    *,
    constructors: frozenset[str] = DEFAULT_AUDIT_CONSTRUCTORS,
    forwarders: frozenset[str] = frozenset(),
) -> list[AuditDetailsSite]:
    """read every audit construction and forwarder call in one module, resolving within it alone.

    :param tree: a parsed module
    :ptype tree: ast.Module
    :param constructors: callee names that build an audit event
    :ptype constructors: frozenset[str]
    :param forwarders: callee names of wrapper helpers that pass ``details=`` through
    :ptype forwarders: frozenset[str]
    :return: one site per call, in line order
    :rtype: list[AuditDetailsSite]
    """
    module = source_module(tree, name="")
    resolver = EventTypeResolver({module.name: module}, constructors=constructors, forwarders=forwarders)
    return _read_module(module, resolver, constructors, forwarders)


def _read_module(
    module: SourceModule,
    resolver: EventTypeResolver,
    constructors: frozenset[str],
    forwarders: frozenset[str],
) -> list[AuditDetailsSite]:
    """read every audit construction and forwarder call in one indexed module.

    :param module: the module
    :ptype module: SourceModule
    :param resolver: the event-type resolver over every scanned module
    :ptype resolver: EventTypeResolver
    :param constructors: callee names that build an audit event
    :ptype constructors: frozenset[str]
    :param forwarders: callee names of wrapper helpers that pass ``details=`` through
    :ptype forwarders: frozenset[str]
    :return: one site per call, in line order
    :rtype: list[AuditDetailsSite]
    """
    targets = constructors | forwarders
    sites = [
        _read_call(node, module, resolver, forwarders)
        for node in ast.walk(module.tree)
        if isinstance(node, ast.Call) and callee_names(node) & targets
    ]
    return sorted(sites, key=lambda site: site.lineno)


def unclassified_detail_paths(
    site: AuditDetailsSite,
    *,
    safe_keys_for: Callable[[str], frozenset[str]],
    personal_keys: frozenset[str],
) -> list[tuple[str, ...]]:
    """the key paths at one site that no classification names, where the rule consults them.

    a nested key is consulted only when every key above it is safe for the event type;
    beneath an unsafe key the whole value is anonymized. a site resolving to several event
    types is credited with the keys safe for every one of them; one resolving to none, with
    the platform set alone.

    :param site: a read site
    :ptype site: AuditDetailsSite
    :param safe_keys_for: the safe-key lookup for an event type
    :ptype safe_keys_for: Callable[[str], frozenset[str]]
    :param personal_keys: the keys classified as personal
    :ptype personal_keys: frozenset[str]
    :return: the unclassified paths, sorted
    :rtype: list[tuple[str, ...]]
    """
    safe = (
        frozenset.intersection(*(safe_keys_for(event_type) for event_type in site.event_types))
        if site.event_types
        else safe_keys_for("")
    )
    return sorted(
        path
        for path in site.keys
        if all(ancestor in safe for ancestor in path[:-1]) and path[-1] not in safe and path[-1] not in personal_keys
    )


def collect_audit_details_sites(config: AuditDetailsConfig) -> dict[Path, list[AuditDetailsSite]]:
    """every site under the configured source roots, event types resolved across all of them.

    :param config: per-repo enforcement config
    :ptype config: AuditDetailsConfig
    :return: file -> its sites, files with none omitted
    :rtype: dict[Path, list[AuditDetailsSite]]
    """
    modules: dict[str, SourceModule] = {}
    paths: dict[str, Path] = {}
    for root in config.src_roots:
        for path in iter_python_files(root):
            tree = parse_python_file(path)
            if tree is None:
                continue
            name, is_package = module_name(path, root)
            if name not in modules:
                modules[name] = source_module(tree, name=name, is_package=is_package)
                paths[name] = path
    resolver = EventTypeResolver(modules, constructors=config.constructors, forwarders=config.forwarders)
    by_file: dict[Path, list[AuditDetailsSite]] = {}
    for name, module in modules.items():
        sites = _read_module(module, resolver, config.constructors, config.forwarders)
        if sites:
            by_file[paths[name]] = sites
    return by_file


def find_audit_details_violations(config: AuditDetailsConfig) -> list[Violation]:
    """every unclassified key and every unreadable details argument under the source roots.

    :param config: per-repo enforcement config
    :ptype config: AuditDetailsConfig
    :return: violations, in file then line order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    if not config.src_roots:
        violations.append(
            Violation(
                category="audit_details.no_src_roots",
                file=config.repo_root,
                line=0,
                symbol="(config)",
                reason="no source roots configured; this shell is not enforcing anything",
            )
        )
    for path, sites in sorted(collect_audit_details_sites(config).items()):
        for site in sites:
            violations.extend(_site_violations(path, site, config))
    return violations


def _site_violations(path: Path, site: AuditDetailsSite, config: AuditDetailsConfig) -> list[Violation]:
    """the violations one site carries.

    :param path: the file holding the site
    :ptype path: Path
    :param site: the read site
    :ptype site: AuditDetailsSite
    :param config: per-repo enforcement config
    :ptype config: AuditDetailsConfig
    :return: its unclassified-key and unreadable violations
    :rtype: list[Violation]
    """
    event_type = ", ".join(sorted(site.event_types)) or "<unresolved event_type>"
    found = [
        Violation(
            category="audit_details.unclassified",
            file=path,
            line=site.lineno,
            symbol=".".join(key_path),
            reason=(
                f"[{event_type}] neither safe nor personal. classify it in threetears.agent.audit.anonymize: "
                "SAFE_DETAIL_KEYS if its value is structural for every producer (an id, a count, a status, "
                "a duration, a version), PERSONAL_DETAIL_KEYS if it can carry a name, email, address, IP, "
                "free or user-typed text, a path, exception text or a credential -- when unsure it is "
                "personal -- or a family declaration if it is structural in this event family only"
            ),
        )
        for key_path in unclassified_detail_paths(
            site, safe_keys_for=config.safe_keys_for, personal_keys=config.personal_keys
        )
    ]
    found.extend(
        Violation(
            category="audit_details.unreadable",
            file=path,
            line=line,
            symbol="details",
            reason=(
                f"{reason}. build details as a literal with string keys in the calling function, or declare "
                "a wrapper helper in AuditDetailsConfig.forwarders so its call sites are read instead: a key "
                "computed at runtime cannot be classified"
            ),
        )
        for line, reason in site.unreadable
    )
    return found


def _read_call(
    call: ast.Call,
    module: SourceModule,
    resolver: EventTypeResolver,
    forwarders: frozenset[str],
) -> AuditDetailsSite:
    """read one constructor or forwarder call.

    :param call: the call
    :ptype call: ast.Call
    :param module: the module holding it
    :ptype module: SourceModule
    :param resolver: the event-type resolver over every scanned module
    :ptype resolver: EventTypeResolver
    :param forwarders: callee names of declared wrapper helpers
    :ptype forwarders: frozenset[str]
    :return: the site
    :rtype: AuditDetailsSite
    """
    chain: tuple[FunctionNode, ...] = module.chains.get(id(call), ())
    forwarded: set[str] = set()
    for function in chain:
        if function.name in forwarders:
            forwarded |= parameter_names(function)
    reading = _Reading(forwarded=frozenset(forwarded))
    arguments: dict[str | None, ast.expr] = {}
    for keyword in call.keywords:
        if keyword.arg is None and not _is_forwarded(keyword.value, reading):
            reading.unreadable.append((call.lineno, "a **spread into the call may carry details"))
        arguments.setdefault(keyword.arg, keyword.value)
    details = arguments.get("details")
    scope: ast.AST = chain[-1] if chain else module.tree
    if isinstance(details, ast.Name):
        _read_local(details.id, scope, call.lineno, reading)
    elif details is not None:
        _read_literal(details, (), reading)
    return AuditDetailsSite(
        lineno=call.lineno,
        event_types=resolver.resolve_call(call, module),
        keys=frozenset(reading.keys),
        unreadable=tuple(reading.unreadable),
    )


def _is_empty_literal(node: ast.expr) -> bool:
    """whether ``node`` is ``{}``, ``dict()`` or ``None`` -- a default that adds no key.

    :param node: any expression
    :ptype node: ast.expr
    :return: ``True`` for an empty default
    :rtype: bool
    """
    empty_dict = isinstance(node, ast.Dict) and not node.keys
    empty_call = isinstance(node, ast.Call) and callee_names(node) == {"dict"} and not node.args and not node.keywords
    none = isinstance(node, ast.Constant) and node.value is None
    return empty_dict or empty_call or none


def _is_forwarded(node: ast.expr, reading: _Reading) -> bool:
    """whether ``node`` is a declared forwarder's own parameter, passed through, copied or defaulted.

    :param node: any expression
    :ptype node: ast.expr
    :param reading: the site being read, carrying the forwarded parameter names
    :ptype reading: _Reading
    :return: ``True`` when the expression adds no key the forwarder's call sites do not carry
    :rtype: bool
    """
    forwarded = False
    if isinstance(node, ast.Name):
        forwarded = node.id in reading.forwarded
    elif isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        forwarded = all(_is_forwarded(value, reading) or _is_empty_literal(value) for value in node.values)
    elif isinstance(node, ast.IfExp):
        forwarded = all(
            _is_forwarded(branch, reading) or _is_empty_literal(branch) for branch in (node.body, node.orelse)
        )
    elif isinstance(node, ast.Call) and callee_names(node) == {"dict"} and len(node.args) == 1 and not node.keywords:
        forwarded = _is_forwarded(node.args[0], reading)
    return forwarded


def _read_literal(node: ast.expr, path: tuple[str, ...], reading: _Reading) -> None:
    """add a literal's keys under ``path``, recording any part that cannot be read.

    :param node: a dict literal, a ``dict(...)`` call, or a forwarded parameter
    :ptype node: ast.expr
    :param path: the key path the literal sits under
    :ptype path: tuple[str, ...]
    :param reading: the site being read
    :ptype reading: _Reading
    :return: nothing
    :rtype: None
    """
    if _is_forwarded(node, reading) or _is_empty_literal(node):
        return
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None:
                if not _is_forwarded(value, reading):
                    reading.unreadable.append((value.lineno, "a **spread inside the details literal"))
            elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                reading.keys.add((*path, key.value))
                _read_nested(value, (*path, key.value), reading)
            else:
                reading.unreadable.append((key.lineno, "a computed key inside the details literal"))
    elif isinstance(node, ast.Call) and callee_names(node) == {"dict"} and len(node.args) <= 1:
        if node.args and not _is_forwarded(node.args[0], reading):
            reading.unreadable.append((node.lineno, "dict(...) copies a mapping the walker cannot read"))
        for keyword in node.keywords:
            if keyword.arg is None:
                if not _is_forwarded(keyword.value, reading):
                    reading.unreadable.append((node.lineno, "a **spread inside dict(...)"))
            else:
                reading.keys.add((*path, keyword.arg))
                _read_nested(keyword.value, (*path, keyword.arg), reading)
    else:
        reading.unreadable.append((node.lineno, f"details built from {type(node).__name__}, not a literal"))


def _read_nested(value: ast.expr, path: tuple[str, ...], reading: _Reading) -> None:
    """follow a details value into nested literals, whose keys the rule may also judge.

    :param value: the value under a details key
    :ptype value: ast.expr
    :param path: the key path of ``value``
    :ptype path: tuple[str, ...]
    :param reading: the site being read
    :ptype reading: _Reading
    :return: nothing
    :rtype: None
    """
    is_dict_call = isinstance(value, ast.Call) and callee_names(value) == {"dict"} and not value.args
    if isinstance(value, ast.Dict) or is_dict_call:
        _read_literal(value, path, reading)
    elif isinstance(value, ast.List | ast.Tuple | ast.Set):
        for element in value.elts:
            _read_nested(element, path, reading)


def _is_name(node: ast.expr, name: str) -> bool:
    """whether ``node`` is a bare reference to the local ``name``.

    :param node: any expression
    :ptype node: ast.expr
    :param name: the local being traced
    :ptype name: str
    :return: ``True`` for ``name`` itself
    :rtype: bool
    """
    return isinstance(node, ast.Name) and node.id == name


def _read_local(name: str, scope: ast.AST, lineno: int, reading: _Reading) -> None:
    """read the keys a function gives a local details dict before handing it over.

    :param name: the local passed as ``details=``
    :ptype name: str
    :param scope: the innermost enclosing function, or the module
    :ptype scope: ast.AST
    :param lineno: line of the call handing it over
    :ptype lineno: int
    :param reading: the site being read
    :ptype reading: _Reading
    :return: nothing
    :rtype: None
    """
    bound = name in reading.forwarded
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if _is_name(target, name) and node.value is not None:
                    bound = True
                    _read_literal(node.value, (), reading)
                elif isinstance(target, ast.Subscript) and _is_name(target.value, name):
                    _read_subscript(target, node, name, reading)
        elif isinstance(node, ast.AugAssign) and _is_name(node.target, name):
            _read_literal(node.value, (), reading)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and _is_name(node.func.value, name)
            and node.func.attr in _ADDING_METHODS
        ):
            _read_method_call(node, name, reading)
    if not bound:
        reading.unreadable.append(
            (lineno, f"details={name} is not assigned in this function (a parameter of an undeclared forwarder?)")
        )


def _read_subscript(target: ast.Subscript, node: ast.Assign | ast.AnnAssign, name: str, reading: _Reading) -> None:
    """read one ``name["k"] = value`` write.

    :param target: the subscript being assigned
    :ptype target: ast.Subscript
    :param node: the assignment
    :ptype node: ast.Assign | ast.AnnAssign
    :param name: the local being traced
    :ptype name: str
    :param reading: the site being read
    :ptype reading: _Reading
    :return: nothing
    :rtype: None
    """
    key = target.slice
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        reading.keys.add((key.value,))
        if node.value is not None:
            _read_nested(node.value, (key.value,), reading)
    else:
        reading.unreadable.append((node.lineno, f"{name}[<computed>] = ..."))


def _read_method_call(call: ast.Call, name: str, reading: _Reading) -> None:
    """read the keys one ``name.update(...)`` / ``name.setdefault(...)`` adds.

    :param call: the method call
    :ptype call: ast.Call
    :param name: the local being traced
    :ptype name: str
    :param reading: the site being read
    :ptype reading: _Reading
    :return: nothing
    :rtype: None
    """
    method = call.func.attr if isinstance(call.func, ast.Attribute) else ""
    first = call.args[0] if call.args else None
    if method == "setdefault" and isinstance(first, ast.Constant) and isinstance(first.value, str):
        reading.keys.add((first.value,))
    elif method == "setdefault":
        reading.unreadable.append((call.lineno, f"{name}.setdefault(<computed>, ...)"))
    else:
        for argument in call.args:
            _read_literal(argument, (), reading)
        for keyword in call.keywords:
            if keyword.arg is None and not _is_forwarded(keyword.value, reading):
                reading.unreadable.append((call.lineno, f"{name}.update(**...)"))
            elif keyword.arg is not None:
                reading.keys.add((keyword.arg,))
