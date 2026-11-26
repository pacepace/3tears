"""walkers for the invalidation-listener enforcement domain.

Publishing a cross-pod invalidation is the easy half; CONSUMING it is the half that gets
forgotten, and for a long time no process in the hub repo did it -- a row written on
replica A was evicted nowhere, replica B served its own L1 copy until the pod restarted,
and nothing bounded it.

Two walkers, because there are two ways to get it wrong:

- :func:`find_unlistened_registries` -- an L2-live registry that no start call names.
  That is the original defect: the cache is incoherent across pods, silently.
- :func:`find_starts_without_stops` -- a module that starts a listener and never releases
  one. The subscription then outlives the connection it was made on, which is how a
  teardown ends up leaking or a restart ends up double-subscribed.

**The unit is the REGISTRY, not the process.** Subscription state lives on the registry
instance and nothing is inherited across instances, so a process holding two L2-live
registries must subscribe both -- the registry server does exactly that. What is forbidden
is two listeners on ONE registry: every broadcast handled twice, and a teardown that
releases only one.

**Module-scoped, with the blind spot that implies.** A registry constructed in one module
and given its client in another is invisible to this gate and to the L2-scope gate alike --
see :mod:`threetears.enforcement.common.collection_registry`. Keep construction and wiring
in one module.

Registry detection is shared with the L2-scope domain via
:mod:`threetears.enforcement.common.collection_registry`, deliberately: both domains ask
"which registries are L2-live", and two copies of that answer drift the moment a new
wiring shape appears.

The three scans read one per-file fact (:func:`_listener_facts`), kept in the gitignored scan
cache by the file's bytes (:func:`~threetears.enforcement.common.scan_cache.file_facts`) under the
checkout holding the src root, so a process that walked the same bytes before parses none again.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from threetears.enforcement.common import (
    Violation,
    argument_spellings,
    callee_names,
    constructed_registry_lines,
    iter_python_files,
    l2_live_registries,
    receiver,
)
from threetears.enforcement.common.scan_cache import facts_root, file_facts

__all__ = [
    "START_NAMES",
    "STOP_NAMES",
    "calls_any",
    "count_l2_live_registries",
    "find_starts_without_stops",
    "find_unlistened_registries",
    "started_registries",
    "starts_without_stopping",
    "unlistened_registries",
]

_UNLISTENED_CATEGORY: Final[str] = "invalidation_listener.live_registry_without_a_listener"
_UNPAIRED_CATEGORY: Final[str] = "invalidation_listener.start_without_stop"

#: calls that subscribe the stream. Both shapes: the registry method, and the
#: helper form a consumer may wrap it in (which typically adds the no-L2 warning).
START_NAMES: Final[frozenset[str]] = frozenset(
    {"start_invalidation_listener", "start_collection_invalidation_listener"},
)

#: calls that release it again.
STOP_NAMES: Final[frozenset[str]] = frozenset(
    {"stop_invalidation_listener", "stop_collection_invalidation_listener"},
)


def started_registries(tree: ast.AST) -> frozenset[str]:
    """return every registry spelling a start call names.

    Both call shapes: ``registry.start_invalidation_listener(client)`` and the wrapper
    form ``start_collection_invalidation_listener(registry, client, ...)``, where the
    registry arrives as an ARGUMENT rather than a receiver.

    :param tree: parsed module
    :ptype tree: ast.AST
    :return: spellings named by a start call
    :rtype: frozenset[str]
    """
    started: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not callee_names(node) & START_NAMES:
            continue
        called_on = receiver(node)
        if called_on is not None:
            started.add(called_on)
        started.update(argument_spellings(node))
    return frozenset(started)


def unlistened_registries(tree: ast.AST) -> list[str]:
    """return every L2-live registry in a module that no start call names.

    :param tree: parsed module
    :ptype tree: ast.AST
    :return: offending registry spellings, sorted
    :rtype: list[str]
    """
    return sorted(l2_live_registries(tree) - started_registries(tree))


def calls_any(tree: ast.AST, names: frozenset[str]) -> bool:
    """report whether the module invokes any of the named symbols.

    :param tree: parsed module
    :ptype tree: ast.AST
    :param names: symbol names to match
    :ptype names: frozenset[str]
    :return: ``True`` when at least one such call exists
    :rtype: bool
    """
    return any(isinstance(node, ast.Call) and callee_names(node) & names for node in ast.walk(tree))


def starts_without_stopping(tree: ast.AST) -> bool:
    """report whether a module starts a listener and never releases one.

    Module-scoped rather than per-registry on purpose: the stop routinely lives in a
    different function from the start (a lifespan's two halves, a class's ``connect`` and
    ``close``), so pairing them per-registry would flag every correct teardown.

    :param tree: parsed module
    :ptype tree: ast.AST
    :return: ``True`` when a start has no matching stop in the same module
    :rtype: bool
    """
    return calls_any(tree, START_NAMES) and not calls_any(tree, STOP_NAMES)


def count_l2_live_registries(
    src_roots: tuple[Path, ...],
    skip_basenames: frozenset[str] = frozenset(),
) -> int:
    """count the L2-live registries the scan can actually see.

    The non-vacuity floor. A reader that stopped matching reports a clean tree, which is
    indistinguishable from a correct one -- so the live count is asserted from BELOW too.

    :param src_roots: src roots to scan
    :ptype src_roots: tuple[Path, ...]
    :param skip_basenames: file basenames to skip
    :ptype skip_basenames: frozenset[str]
    :return: how many L2-live registries were found
    :rtype: int
    """
    return sum(facts.live_registries for _, facts in _listener_facts_under(src_roots, skip_basenames))


def find_unlistened_registries(
    src_roots: tuple[Path, ...],
    skip_basenames: frozenset[str] = frozenset(),
) -> list[Violation]:
    """flag every L2-live registry that subscribes no invalidation listener.

    :param src_roots: src roots to scan
    :ptype src_roots: tuple[Path, ...]
    :param skip_basenames: file basenames to skip
    :ptype skip_basenames: frozenset[str]
    :return: violations in source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for source, facts in _listener_facts_under(src_roots, skip_basenames):
        violations.extend(
            Violation(
                category=_UNLISTENED_CATEGORY,
                file=source,
                line=line,
                symbol=name,
                reason=(
                    f"'{name}' holds an L2-live collection and subscribes no invalidation "
                    f"listener, so a row a peer replica has already replaced is served from "
                    f"this pod's L1 until the process restarts, and nothing bounds it. an L1 "
                    f"TTL is the wrong remedy -- it bounds the staleness by accident and hides "
                    f"the missing listener. call start_invalidation_listener(nats_client) once "
                    f"the client is bound, and stop it in this process's shutdown path"
                ),
            )
            for name, line in facts.unlistened
        )
    return violations


def find_starts_without_stops(
    src_roots: tuple[Path, ...],
    skip_basenames: frozenset[str] = frozenset(),
) -> list[Violation]:
    """flag every module that starts a listener and never releases one.

    :param src_roots: src roots to scan
    :ptype src_roots: tuple[Path, ...]
    :param skip_basenames: file basenames to skip
    :ptype skip_basenames: frozenset[str]
    :return: violations in source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for source, facts in _listener_facts_under(src_roots, skip_basenames):
        if not facts.starts_without_stopping:
            continue
        violations.append(
            Violation(
                category=_UNPAIRED_CATEGORY,
                file=source,
                line=1,
                # the module STEM, not its path: `common.exemptions` constrains a symbol
                # to an identifier, so a path-shaped one cannot be exempted at all -- the
                # entry raises `ExemptionError` and takes the whole domain down with it.
                # `Violation.file` already carries the path the report renders.
                symbol=source.stem,
                reason=(
                    "this module starts an invalidation listener and never releases one, so "
                    "the subscription outlives the connection it was made on. stop it in the "
                    "same lifecycle that started it -- before the transport goes away"
                ),
            )
        )
    return violations


@dataclass(frozen=True)
class _ListenerFacts:
    """what the three scans read of one module.

    :ivar live_registries: how many L2-live registries it holds
    :ivar unlistened: ``(registry, construction line)`` per L2-live registry no start call
        names, sorted by registry; line 1 when the construction line is unknown
    :ivar starts_without_stopping: whether it starts a listener and never releases one
    """

    live_registries: int
    unlistened: tuple[tuple[str, int], ...]
    starts_without_stopping: bool


def _listener_facts(path: Path, tree: ast.Module) -> _ListenerFacts:
    """one module's facts for the three scans, as the scan cache keeps them.

    :param path: the module's file
    :ptype path: Path
    :param tree: its parsed module
    :ptype tree: ast.Module
    :return: the module's facts
    :rtype: _ListenerFacts
    """
    offenders = unlistened_registries(tree)
    lines = constructed_registry_lines(tree) if offenders else {}
    return _ListenerFacts(
        live_registries=len(l2_live_registries(tree)),
        unlistened=tuple((name, lines.get(name, 1)) for name in offenders),
        starts_without_stopping=starts_without_stopping(tree),
    )


def _listener_facts_under(
    src_roots: tuple[Path, ...],
    skip_basenames: frozenset[str],
) -> list[tuple[Path, _ListenerFacts]]:
    """each walked module's facts, from the scan cache, in walk order.

    files are walked as :func:`iter_python_files` yields them, root by root, skipping
    ``skip_basenames``; a file that does not parse is noted and left out, as
    :func:`~threetears.enforcement.common.ast_helpers.parse_python_file` skipped it. facts are
    kept under the checkout holding each src root, or under the root itself outside one.

    :param src_roots: src roots to scan
    :ptype src_roots: tuple[Path, ...]
    :param skip_basenames: file basenames to skip
    :ptype skip_basenames: frozenset[str]
    :return: ``(file, facts)`` per parsed module
    :rtype: list[tuple[Path, _ListenerFacts]]
    """
    walked: list[tuple[Path, Path]] = []
    for root in src_roots:
        cache_root = facts_root(root)
        walked.extend((source, cache_root) for source in iter_python_files(root) if source.name not in skip_basenames)
    facts: dict[Path, _ListenerFacts] = {}
    for cache_root in dict.fromkeys(root for _, root in walked):
        files = [source for source, root in walked if root == cache_root]
        facts.update(file_facts(_listener_facts, files, root=cache_root, unparsable="note"))
    return [(source, facts[source]) for source, _ in walked if source in facts]
