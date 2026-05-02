"""walkers for the dict-state-detection enforcement domain.

two walkers in this module — :func:`find_dict_state_violations` for
detecting raw-dict persistent state assigned in ``__init__`` methods,
and :func:`find_stale_allowlist_entries` for catching allowlist /
known-violation entries that no longer match a real violation. a
helper, :func:`filter_against_allowlist`, splits a violation list into
the surviving (true-violation) and exempted subsets.

matched shapes for :func:`find_dict_state_violations`:

- ``self._x = {}`` — empty dict literal.
- ``self._x = dict()`` / ``self._x = OrderedDict()`` — empty constructor
  call (no positional or keyword args).
- ``self._x = something or {}`` — :class:`ast.BoolOp` ``Or`` with a
  trailing empty-dict literal fallback.
- ``self._x: dict[str, int] = ...`` — annotated assignment whose
  annotation is ``dict``, ``OrderedDict``, or a subscripted form; the
  value still has to be one of the bad shapes above (annotation alone
  is not enough — ``self._x: dict[str, int] = some_func()`` is fine).

intentionally NOT flagged:

- assignments outside ``__init__`` — module-level, ``setUp``, factory
  methods. only ``__init__`` is the persistent-state-construction site.
- non-private targets — ``self.x``. the walker filters on
  single-leading-underscore convention so public attributes that have
  been deliberately exposed for other reasons (e.g. unit-test access)
  are never re-flagged.
- non-self targets — ``local = {}``, ``some_other_obj.cache = {}``.
- non-dict shapes — ``self._x = []``, ``self._x = set()``,
  ``self._x = SomeOtherClass()``.

the walker reports class membership via the violation's ``reason``
field for human readability, but **matching is by (file, line, attr)**
— class name is intentionally not part of the key. line-based matching
is precise: a refactor that moves the assignment to a new line
surfaces the entry as stale, which is the correct prompt to revisit
the rationale.
"""

from __future__ import annotations

import ast
from pathlib import Path

from threetears.enforcement.common import (
    Violation,
    iter_python_files,
    parse_python_file,
    relative_posix_path,
)
from threetears.enforcement.dict_state_detection.config import (
    DictStateAllowlistEntry,
)

__all__ = [
    "filter_against_allowlist",
    "find_dict_state_violations",
    "find_stale_allowlist_entries",
]


_DETECTOR_CATEGORY = "dict_state_detection.dict_in_init"
_STALE_ALLOWLIST_CATEGORY = "dict_state_detection.stale_allowlist"
_STALE_KNOWN_VIOLATION_CATEGORY = (
    "dict_state_detection.stale_known_violation"
)


def _is_dict_literal(node: ast.expr) -> bool:
    """true iff ``node`` is an empty dict literal ``{}``.

    :param node: ast expression to test
    :ptype node: ast.expr
    :return: whether the node is a zero-element :class:`ast.Dict`
    :rtype: bool
    """
    return isinstance(node, ast.Dict) and len(node.keys) == 0


def _is_dict_call(node: ast.expr) -> bool:
    """true iff ``node`` is a ``dict()`` call with no args.

    a call with positional or keyword args (``dict(a=1)``) does not
    match — it constructs an initialised dict, which is a deliberate
    one-shot use, not a persistent-state slot.

    :param node: ast expression to test
    :ptype node: ast.expr
    :return: whether the node is the bare ``dict()`` constructor
    :rtype: bool
    """
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and len(node.args) == 0
        and len(node.keywords) == 0
    )


def _is_ordered_dict_call(node: ast.expr) -> bool:
    """true iff ``node`` is a no-arg ``OrderedDict()`` call.

    matches the bare ``OrderedDict()`` constructor; positional args
    (``OrderedDict([("a", 1)])``) do not match for the same reason
    that initialised :func:`_is_dict_call` does not.

    :param node: ast expression to test
    :ptype node: ast.expr
    :return: whether the node is the no-arg ``OrderedDict()`` call
    :rtype: bool
    """
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "OrderedDict"
        and len(node.args) == 0
    )


def _is_dict_or_fallback(node: ast.expr) -> bool:
    """true iff ``node`` is the ``something or {}`` fallback pattern.

    matches an :class:`ast.BoolOp` with :class:`ast.Or` whose values
    contain at least one empty-dict literal. covers both the canonical
    rightmost-fallback shape (``opts or {}``) and the rarer
    leftmost variant (``{} or opts``); both are equally invalid as a
    persistent-state init.

    :param node: ast expression to test
    :ptype node: ast.expr
    :return: whether the node is an ``Or`` BoolOp containing ``{}``
    :rtype: bool
    """
    if not isinstance(node, ast.BoolOp):
        return False
    if not isinstance(node.op, ast.Or):
        return False
    return any(_is_dict_literal(v) for v in node.values)


def _is_dict_type_annotation(annotation: ast.expr | None) -> bool:
    """true iff ``annotation`` resolves to ``dict`` or ``OrderedDict``.

    matches both the bare-name form (``self._x: dict = ...``) and the
    parameterised form (``self._x: dict[str, int] = ...``). nested or
    quoted annotations are not detected — the canonical's behaviour is
    preserved as-is.

    :param annotation: ast annotation expression, or ``None``
    :ptype annotation: ast.expr | None
    :return: whether the annotation references ``dict`` or
        ``OrderedDict``
    :rtype: bool
    """
    if annotation is None:
        return False
    if isinstance(annotation, ast.Name):
        return annotation.id in ("dict", "OrderedDict")
    if isinstance(annotation, ast.Subscript) and isinstance(
        annotation.value, ast.Name,
    ):
        return annotation.value.id in ("dict", "OrderedDict")
    return False


def _is_bad_value(node: ast.expr) -> bool:
    """true iff ``node`` is one of the four bad-shape value forms.

    the four shapes — empty dict literal, no-arg ``dict()``, no-arg
    ``OrderedDict()``, ``x or {}`` — are unified here so the
    annotation-vs-Assign branches in
    :func:`find_dict_state_violations` can use a single predicate.

    :param node: ast expression to test
    :ptype node: ast.expr
    :return: whether the node is any of the recognised bad shapes
    :rtype: bool
    """
    return (
        _is_dict_literal(node)
        or _is_dict_call(node)
        or _is_ordered_dict_call(node)
        or _is_dict_or_fallback(node)
    )


def _get_self_attr_name(target: ast.expr) -> str | None:
    """extract the attribute name from ``self._<x>``, or ``None`` otherwise.

    matches assignments to single-leading-underscore attributes on
    ``self`` only:

    - :class:`ast.Attribute` whose value is :class:`ast.Name` ``"self"``
    - the attribute name starts with ``"_"``

    public attributes (``self.foo``), nested receivers (``self.x.y``),
    and assignments to non-self objects (``other.cache``) are filtered
    out by returning ``None``.

    :param target: ast expression node from an :class:`ast.Assign` or
        :class:`ast.AnnAssign` target
    :ptype target: ast.expr
    :return: the attribute name (e.g. ``"_cache"``) or ``None``
    :rtype: str | None
    """
    if not isinstance(target, ast.Attribute):
        return None
    if not isinstance(target.value, ast.Name):
        return None
    if target.value.id != "self":
        return None
    if not target.attr.startswith("_"):
        return None
    return target.attr


def _shape_label(value: ast.expr, annotation: ast.expr | None) -> str:
    """human-readable label describing which bad-shape was detected.

    surfaced in the violation's ``reason`` field so the failure message
    is actionable without re-running the walker by hand.

    :param value: assignment value that triggered detection
    :ptype value: ast.expr
    :param annotation: ann-assignment annotation (``None`` for plain
        ``Assign`` nodes)
    :ptype annotation: ast.expr | None
    :return: short label such as ``"empty-dict literal"`` or
        ``"`dict()` call"``
    :rtype: str
    """
    if _is_dict_literal(value):
        if annotation is not None and _is_dict_type_annotation(annotation):
            return "annotated empty-dict literal"
        return "empty-dict literal"
    if _is_dict_call(value):
        return "`dict()` call"
    if _is_ordered_dict_call(value):
        return "`OrderedDict()` call"
    if _is_dict_or_fallback(value):
        return "`... or {}` fallback"
    return "raw dict shape"


def _scan_init_body(
    class_node: ast.ClassDef,
    init_node: ast.FunctionDef | ast.AsyncFunctionDef,
    module_path: Path,
    violations: list[Violation],
) -> None:
    """walk a single ``__init__`` body for raw-dict assignments.

    handles both :class:`ast.Assign` (with possibly multiple targets)
    and :class:`ast.AnnAssign` (with at most one target) statements.
    every recognised shape produces one :class:`Violation`; multiple
    targets in a single ``Assign`` produce one violation per offending
    target.

    :param class_node: enclosing :class:`ast.ClassDef` node — used to
        surface the class name in the violation reason
    :ptype class_node: ast.ClassDef
    :param init_node: the ``__init__`` function definition
    :ptype init_node: ast.FunctionDef | ast.AsyncFunctionDef
    :param module_path: absolute path of the module being scanned
    :ptype module_path: Path
    :param violations: accumulator, mutated in place
    :ptype violations: list[Violation]
    """
    for stmt in ast.walk(init_node):
        if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            attr_name = _get_self_attr_name(stmt.target)
            if attr_name is None:
                continue
            if not _is_bad_value(stmt.value):
                continue
            shape = _shape_label(stmt.value, stmt.annotation)
            violations.append(
                Violation(
                    category=_DETECTOR_CATEGORY,
                    file=module_path,
                    line=stmt.lineno,
                    symbol=attr_name,
                    reason=(
                        f"{class_node.name}.{attr_name} initialised as "
                        f"{shape}; use a 3tears L1 backend "
                        f"(SQLiteBackend) for pod-local cache, or NATS "
                        f"KV for cross-instance shared state. if "
                        f"genuinely ephemeral, add to the allowlist "
                        f"with a specific rationale."
                    ),
                )
            )
        elif isinstance(stmt, ast.Assign):
            if not _is_bad_value(stmt.value):
                continue
            shape = _shape_label(stmt.value, None)
            for target in stmt.targets:
                attr_name = _get_self_attr_name(target)
                if attr_name is None:
                    continue
                violations.append(
                    Violation(
                        category=_DETECTOR_CATEGORY,
                        file=module_path,
                        line=stmt.lineno,
                        symbol=attr_name,
                        reason=(
                            f"{class_node.name}.{attr_name} initialised "
                            f"as {shape}; use a 3tears L1 backend "
                            f"(SQLiteBackend) for pod-local cache, or "
                            f"NATS KV for cross-instance shared state. "
                            f"if genuinely ephemeral, add to the "
                            f"allowlist with a specific rationale."
                        ),
                    )
                )


def _scan_module(
    tree: ast.Module,
    module_path: Path,
    violations: list[Violation],
) -> None:
    """walk a parsed module, dispatching to :func:`_scan_init_body` per class.

    handles nested classes via :func:`ast.walk`, matching the canonical
    behaviour. each class's direct ``__init__`` (sync or async) is
    scanned independently — methods on inner classes count.

    :param tree: parsed ast module
    :ptype tree: ast.Module
    :param module_path: absolute path of the module being scanned
    :ptype module_path: Path
    :param violations: accumulator, mutated in place
    :ptype violations: list[Violation]
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for child in node.body:
            if not isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                continue
            if child.name != "__init__":
                continue
            _scan_init_body(node, child, module_path, violations)


def find_dict_state_violations(
    src_roots: tuple[Path, ...],
    repo_root: Path,
) -> list[Violation]:
    """walk ``src_roots`` for raw-dict persistent state in ``__init__``.

    one :class:`Violation` per offending assignment, with category
    :data:`_DETECTOR_CATEGORY` (``dict_state_detection.dict_in_init``).
    ``symbol`` is the attribute name (e.g. ``"_cache"``); ``line`` is
    the line of the assignment statement (the canonical also reports
    the assignment line, not the class-def line). matching is by
    ``(file, line, attr_name)`` — class name is included in the reason
    text for context but is not part of the matching key.

    :param src_roots: every src root the scanner should consider
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: repo root (retained for parity with sibling
        domains; relative-path rendering happens at report time)
    :ptype repo_root: Path
    :return: violations in source order (root-major, then file-major,
        then statement-major)
    :rtype: list[Violation]
    """
    _ = repo_root
    violations: list[Violation] = []
    for root in src_roots:
        for module_path in iter_python_files(root):
            tree = parse_python_file(module_path)
            if tree is None:
                continue
            _scan_module(tree, module_path, violations)
    return violations


def filter_against_allowlist(
    violations: list[Violation],
    allowlist: tuple[DictStateAllowlistEntry, ...],
    known_violations: tuple[DictStateAllowlistEntry, ...],
    repo_root: Path,
) -> tuple[list[Violation], list[Violation]]:
    """split violations into ``(true_violations, allowed_violations)``.

    every violation is matched against the union of ``allowlist`` and
    ``known_violations`` by the triple
    ``(relative_posix_path(file, repo_root), line, attr_name)`` — line
    matching is exact (not range-based). matched violations end up in
    the ``allowed_violations`` list; unmatched violations are returned
    in ``true_violations`` for the runner to fail on (or report).

    listing the same entry in both ``allowlist`` and
    ``known_violations`` is harmless — it just means the violation is
    suppressed twice. the stale-entry walker reports both as stale
    independently if no real violation matches, so the duplication
    surfaces eventually.

    :param violations: walker output to split
    :ptype violations: list[Violation]
    :param allowlist: legitimately-ephemeral entries, allowed forever
    :ptype allowlist: tuple[DictStateAllowlistEntry, ...]
    :param known_violations: tracked-for-migration entries
    :ptype known_violations: tuple[DictStateAllowlistEntry, ...]
    :param repo_root: repo root for relative-path comparison
    :ptype repo_root: Path
    :return: ``(true_violations, allowed_violations)`` — both preserve
        original order
    :rtype: tuple[list[Violation], list[Violation]]
    """
    keys: set[tuple[str, int, str]] = set()
    for entry in allowlist:
        keys.add((entry.file, entry.line, entry.attr_name))
    for entry in known_violations:
        keys.add((entry.file, entry.line, entry.attr_name))

    true_violations: list[Violation] = []
    allowed: list[Violation] = []
    for violation in violations:
        rel = relative_posix_path(violation.file, repo_root)
        if (rel, violation.line, violation.symbol) in keys:
            allowed.append(violation)
        else:
            true_violations.append(violation)
    return true_violations, allowed


def find_stale_allowlist_entries(
    src_roots: tuple[Path, ...],
    allowlist: tuple[DictStateAllowlistEntry, ...],
    known_violations: tuple[DictStateAllowlistEntry, ...],
    repo_root: Path,
) -> list[Violation]:
    """meta-walker: report allowlist / known-violation entries with no match.

    runs :func:`find_dict_state_violations` once, builds the
    ``(rel_path, line, attr_name)`` set of real violations, then emits
    one :class:`Violation` per allowlist or known-violation entry that
    is *not* in that set. the emitted violation's ``category`` is:

    - :data:`_STALE_ALLOWLIST_CATEGORY` for entries from ``allowlist``
    - :data:`_STALE_KNOWN_VIOLATION_CATEGORY` for entries from
      ``known_violations``

    so reports and exemption matching can keep the two surfaces
    distinct.

    the violation's ``file`` is reconstructed by joining ``repo_root``
    with the entry's ``file`` field. this is purely for report
    rendering; if the file no longer exists on disk, the absolute path
    is still stable and meaningful (``repo/relative.py: not found``-
    style messages are out of scope here).

    :param src_roots: every src root the scanner should consider
    :ptype src_roots: tuple[Path, ...]
    :param allowlist: legitimately-ephemeral entries to audit
    :ptype allowlist: tuple[DictStateAllowlistEntry, ...]
    :param known_violations: tracked-for-migration entries to audit
    :ptype known_violations: tuple[DictStateAllowlistEntry, ...]
    :param repo_root: repo root for relative-path comparison
    :ptype repo_root: Path
    :return: stale-entry violations, in the order
        (allowlist, then known_violations); within each list, original
        entry order is preserved
    :rtype: list[Violation]
    """
    real = find_dict_state_violations(src_roots, repo_root)
    real_keys: set[tuple[str, int, str]] = {
        (relative_posix_path(v.file, repo_root), v.line, v.symbol)
        for v in real
    }

    stale: list[Violation] = []
    for entry in allowlist:
        if (entry.file, entry.line, entry.attr_name) in real_keys:
            continue
        stale.append(_stale_violation(
            entry, repo_root, _STALE_ALLOWLIST_CATEGORY, "allowlist",
        ))
    for entry in known_violations:
        if (entry.file, entry.line, entry.attr_name) in real_keys:
            continue
        stale.append(_stale_violation(
            entry,
            repo_root,
            _STALE_KNOWN_VIOLATION_CATEGORY,
            "known_violations",
        ))
    return stale


def _stale_violation(
    entry: DictStateAllowlistEntry,
    repo_root: Path,
    category: str,
    list_label: str,
) -> Violation:
    """build a stale-entry violation for ``entry``.

    :param entry: the offending allowlist / known-violation entry
    :ptype entry: DictStateAllowlistEntry
    :param repo_root: repo root, used to reconstruct the absolute path
        of the entry's referenced file
    :ptype repo_root: Path
    :param category: violation category to emit
    :ptype category: str
    :param list_label: short human-readable list name
        (``"allowlist"`` / ``"known_violations"``) for the reason text
    :ptype list_label: str
    :return: the stale-entry violation record
    :rtype: Violation
    """
    return Violation(
        category=category,
        file=repo_root / entry.file,
        line=entry.line,
        symbol=entry.attr_name,
        reason=(
            f"{list_label} entry {entry.file}:{entry.line}:"
            f"{entry.attr_name} no longer matches a real violation; "
            f"the referenced code may have been refactored, removed, "
            f"or migrated. drop the entry."
        ),
    )
