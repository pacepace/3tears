"""ast-level transitive inheritance graph for cross-module subclass detection.

walkers like cache-primitive enforcement need to answer:
"does ``Foo`` ultimately inherit from ``BaseCollection``?" — even when
the inheritance chain crosses package boundaries (``Foo →
SchemaBackedCollection → BaseCollection``). a direct-only AST walk over
``cls.bases`` misses every intermediate base; this module builds a
class-name → direct-base-names graph by walking every ClassDef under a
given set of src roots, then offers a cycle-safe transitive query.

the graph is keyed by bare class name. namespace collisions (two
classes named ``Base`` in different modules) merge into a single
graph entry whose bases are the union of all instances. callers must
ensure ``src_roots`` includes every relevant inheritance chain (use
:func:`threetears.enforcement.common.pyproject_discovery.discover_src_roots`
to walk path deps).

each file's classes are kept in the gitignored scan cache by the file's bytes
(:func:`~threetears.enforcement.common.scan_cache.file_facts`), under the checkout
that holds the src root, so a process that walked the same bytes before parses none
of them again.
"""

from __future__ import annotations

import ast
from pathlib import Path

from threetears.enforcement.common.ast_helpers import iter_python_files
from threetears.enforcement.common.scan_cache import facts_root, file_facts

__all__ = [
    "ClassBaseGraph",
    "collect_class_base_graph",
    "extract_base_names",
    "transitively_subclasses_any",
]

ClassBaseGraph = dict[str, set[str]]
"""mapping from class name to the textual last-segment names of every
direct base. ``class Foo(bar.BaseCollection[T])`` contributes
``{"Foo": {"BaseCollection"}}`` because subscript wrappers and dotted
prefixes are stripped down to the final identifier."""


def extract_base_names(cls: ast.ClassDef) -> list[str]:
    """return the textual last-segment name of every base in ``cls.bases``.

    handles four base-expression shapes commonly seen in production
    code:

    - :class:`ast.Name` — bare name (``class X(Base):``)
    - :class:`ast.Attribute` — dotted form (``class X(pkg.Base):``);
      the last attribute segment wins
    - :class:`ast.Subscript` — generic form (``class X(Generic[T])`` /
      ``class X(BaseCollection[Row])``); the subscripted value is
      unwrapped recursively until a Name or Attribute is reached
    - nested combinations (``class X(pkg.BaseCollection[Row]):``)

    expressions that don't reduce to a textual name (call expressions,
    lambdas, etc.) are skipped — they cannot be subclasses for the
    purposes of textual graph lookup.

    bases are returned in source order, preserving multiple-inheritance
    iteration order for downstream walkers that care.

    :param cls: class definition to inspect
    :ptype cls: ast.ClassDef
    :return: list of last-segment base names in source order
    :rtype: list[str]
    """
    names: list[str] = []
    for base in cls.bases:
        name = _base_name(base)
        if name is not None:
            names.append(name)
    return names


def _base_name(node: ast.expr) -> str | None:
    """recursively unwrap ``node`` to its last-segment identifier or None.

    **Deliberately NOT** :func:`~threetears.enforcement.common.ast_helpers.dotted`,
    which walks the same shapes but returns the FULL dotted spelling. The class-base
    graph this feeds keys by bare name, so folding the two would change its keys. The
    consequence to know: on a dotted base the two disagree -- ``pkg.Base`` there,
    ``Base`` here -- so their outputs are not interchangeable and must not be compared.

    :param node: base expression
    :ptype node: ast.expr
    :return: bare last-segment name, or None if the expression is not
        reducible to a textual identifier
    :rtype: str | None
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _base_name(node.value)
    return None


def collect_class_base_graph(src_roots: tuple[Path, ...]) -> ClassBaseGraph:
    """build a class-name → direct-base-names graph by AST-scanning ``src_roots``.

    walks every ``.py`` file under each src root, parses it, and records
    one entry per :class:`ast.ClassDef`. namespace collisions across
    modules (two classes named ``Base`` in different files) merge into a
    single graph entry whose value is the **union** of every observed
    base set. this is intentional: the graph powers an
    "any-path-reaches" query, and merging keeps the query from missing
    a chain that goes through a collided name.

    callers requiring accurate cycle detection or per-namespace
    resolution must pass ``src_roots`` that include all sibling /
    upstream sources. the path-dep discovery helper exists for exactly
    this purpose.

    :param src_roots: src roots to scan; usually the union returned by
        :func:`threetears.enforcement.common.pyproject_discovery.discover_src_roots`
    :ptype src_roots: tuple[Path, ...]
    :return: class-name → direct-base-names graph
    :rtype: ClassBaseGraph
    """
    walked: list[tuple[Path, Path]] = []
    for root in src_roots:
        cache_root = facts_root(root)
        walked.extend((source, cache_root) for source in iter_python_files(root))
    classes: dict[Path, tuple[tuple[str, tuple[str, ...]], ...]] = {}
    for cache_root in dict.fromkeys(root for _, root in walked):
        files = [source for source, root in walked if root == cache_root]
        classes.update(file_facts(_class_bases, files, root=cache_root, unparsable="note"))
    graph: ClassBaseGraph = {}
    for source, _ in walked:
        for name, bases in classes.get(source, ()):
            bucket = graph.setdefault(name, set())
            bucket.update(bases)
    return graph


def _class_bases(path: Path, tree: ast.Module) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """one module's classes and their direct base names, in walk order, as the scan cache keeps them.

    :param path: the module's file
    :ptype path: Path
    :param tree: its parsed module
    :ptype tree: ast.Module
    :return: ``(class name, base names)`` per class definition
    :rtype: tuple[tuple[str, tuple[str, ...]], ...]
    """
    return tuple(
        (node.name, tuple(extract_base_names(node))) for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    )


def transitively_subclasses_any(
    class_name: str,
    target_names: frozenset[str],
    graph: ClassBaseGraph,
) -> bool:
    """true iff ``class_name`` reaches any name in ``target_names`` via ``graph``.

    visit-set guarded so a cyclic graph (``A → B → A``) terminates
    instead of recursing. the class itself counts as reaching the
    targets if its own name is one of them — useful for treating both
    direct and indirect subclasses uniformly.

    :param class_name: class to query
    :ptype class_name: str
    :param target_names: terminal class names that satisfy the query
    :ptype target_names: frozenset[str]
    :param graph: class-name → direct-base-names mapping
    :ptype graph: ClassBaseGraph
    :return: whether any path through ``graph`` from ``class_name``
        reaches a target name
    :rtype: bool
    """
    if class_name in target_names:
        return True
    visited: set[str] = set()
    stack: list[str] = [class_name]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        bases = graph.get(current, set())
        if bases & target_names:
            return True
        for base in bases:
            if base not in visited:
                stack.append(base)
    return False
