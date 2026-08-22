"""shared AST vocabulary for reasoning about ``CollectionRegistry`` wiring.

Two enforcement domains ask questions about the same three facts -- which names are bound
to a ``CollectionRegistry()``, which of those are L2-live, and what a call was handed --
and until this module existed each carried its own copy of the AST helpers that answer
them. The copies were near-identical: same ``_dotted`` / ``_callee_names`` / ``receiver``
/ ``argument_spellings``, same ``_REGISTRY_CTOR`` / ``_CLIENT_KEYWORDS`` /
``_CLIENT_SPELLINGS`` constants, differing only in which private name they hid behind.

That is the drift shape these gates exist to catch, one level up: a new wiring form
teaches one walker to see it and leaves the other blind, and the blind one keeps
reporting a clean tree. The client-spelling set is the sharp edge -- it is a heuristic
list of identifiers, so it grows, and it must grow in one place.

**Both gates built on this are MODULE-SCOPED, and that is a real blind spot.** A registry
constructed in one module and handed its client in another is invisible to both: this
module sees the construction with no client, the other sees a client reaching a name it
never saw constructed. Neither reports anything, which reads exactly like clean wiring.
Cross-module dataflow is out of reach for an AST gate that does not resolve imports, so
the answer is not to widen these functions -- it is to keep the construction and the
wiring in one module, which every current call site does.

The functions here are PUBLIC by intent. A walker in another package reaching a private
helper would be a Shape-A underscore violation, which is exactly the pressure that
produced the second copy.
"""

from __future__ import annotations

import ast
from typing import Final

__all__ = [
    "CLIENT_KEYWORDS",
    "CLIENT_SPELLINGS",
    "L2_BINDER_METHODS",
    "REGISTRY_CTOR",
    "argument_spellings",
    "callee_names",
    "constructed_registries",
    "constructed_registry_lines",
    "dotted",
    "l2_live_registries",
    "names_a_live_client",
    "receiver",
]

#: the constructor whose bound names every walker here tracks.
REGISTRY_CTOR: Final[str] = "CollectionRegistry"

#: the registry methods that can attach an L2 client to a table or to the registry default.
#: ``register`` is here because it accepts the same override keywords, not because it is a
#: common wiring path.
L2_BINDER_METHODS: Final[frozenset[str]] = frozenset({"configure", "register", "bind_table"})

#: keyword names that carry a live L2 client. ``nats_client`` is the per-collection
#: constructor form, which WINS over the registry default -- a walker that watched only
#: ``l2_client`` would miss every collection wired that way.
CLIENT_KEYWORDS: Final[frozenset[str]] = frozenset({"l2_client", "nats_client"})

#: identifier spellings that name a live client when passed POSITIONALLY, where there is no
#: keyword to key on. A heuristic list, and the reason this module exists: it grows, and a
#: second copy of it goes stale silently rather than loudly.
CLIENT_SPELLINGS: Final[frozenset[str]] = frozenset({"nc", "nats_client", "l2_client", "_nc", "_nats_client"})


def dotted(node: ast.expr) -> str | None:
    """return the dotted spelling of a Name-rooted expression.

    ``registry`` and ``self._registry`` resolve; ``build()[0]`` does not, because there is
    no static name for it.

    :param node: expression to spell
    :ptype node: ast.expr
    :return: dotted spelling, or ``None`` when the expression is not name-rooted
    :rtype: str | None
    """
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    spelled: str | None = None
    if isinstance(current, ast.Name):
        parts.append(current.id)
        spelled = ".".join(reversed(parts))
    return spelled


def callee_names(call: ast.Call) -> frozenset[str]:
    """return the bare and attribute spellings of a call's callee.

    :param call: call to inspect
    :ptype call: ast.Call
    :return: callee names
    :rtype: frozenset[str]
    """
    callee = call.func
    names: set[str] = set()
    if isinstance(callee, ast.Name):
        names.add(callee.id)
    elif isinstance(callee, ast.Attribute):
        names.add(callee.attr)
    return frozenset(names)


def receiver(call: ast.Call) -> str | None:
    """return the dotted spelling of a method call's receiver.

    :param call: call to inspect
    :ptype call: ast.Call
    :return: receiver spelling, or ``None`` when the call is not a method call
    :rtype: str | None
    """
    callee = call.func
    spelled: str | None = None
    if isinstance(callee, ast.Attribute):
        spelled = dotted(callee.value)
    return spelled


def argument_spellings(call: ast.Call) -> frozenset[str]:
    """return the dotted spelling of every positional and keyword argument value.

    :param call: call to inspect
    :ptype call: ast.Call
    :return: argument spellings
    :rtype: frozenset[str]
    """
    spellings: set[str] = set()
    for argument in [*call.args, *(keyword.value for keyword in call.keywords)]:
        spelled = dotted(argument)
        if spelled is not None:
            spellings.add(spelled)
    return frozenset(spellings)


def names_a_live_client(call: ast.Call) -> bool:
    """report whether a call is handed a live NATS/L2 client.

    Two shapes: an ``l2_client=`` / ``nats_client=`` keyword whose value is not the ``None``
    literal, and an argument -- positional or keyword -- whose dotted spelling ends in a
    known client name. The second is what covers the positional wiring shape, which a
    keyword-only check misses entirely.

    An explicit ``None`` is DISQUALIFYING rather than ignored: it is how a caller turns L2
    off for one collection, and reading it as live would flag the deliberate opt-out.

    :param call: call to inspect
    :ptype call: ast.Call
    :return: whether a live client reaches the call
    :rtype: bool
    """
    live = False
    for keyword in call.keywords:
        if keyword.arg not in CLIENT_KEYWORDS:
            continue
        explicit_none = isinstance(keyword.value, ast.Constant) and keyword.value.value is None
        if not explicit_none:
            live = True
    if not live:
        live = any(spelled.rsplit(".", 1)[-1] in CLIENT_SPELLINGS for spelled in argument_spellings(call))
    return live


def constructed_registry_lines(tree: ast.AST) -> dict[str, int]:
    """return every name bound to a ``CollectionRegistry()``, mapped to its construction line.

    The construction site is what a violation is reported against: it is the one line that
    stays put while the ``configure()`` call moves, which matters because an exemption entry
    keyed on a line silently stops matching when the line above it changes.

    :param tree: parsed module
    :ptype tree: ast.AST
    :return: registry spelling -> first construction line
    :rtype: dict[str, int]
    """
    lines: dict[str, int] = {}
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if node.value is None or not isinstance(node.value, ast.Call):
            continue
        if REGISTRY_CTOR not in callee_names(node.value):
            continue
        for target in targets:
            spelled = dotted(target)
            if spelled is not None:
                lines.setdefault(spelled, node.lineno)
    return lines


def constructed_registries(tree: ast.AST) -> frozenset[str]:
    """return the spelling of every name bound to a ``CollectionRegistry()``.

    :param tree: parsed module
    :ptype tree: ast.AST
    :return: registry spellings, e.g. ``registry`` / ``self._registry``
    :rtype: frozenset[str]
    """
    return frozenset(constructed_registry_lines(tree))


def l2_live_registries(tree: ast.AST) -> frozenset[str]:
    """return every constructed registry that holds an L2-live collection.

    All the wiring shapes: the registry-default binders in :data:`L2_BINDER_METHODS`, the
    per-collection ``nats_client=`` keyword, and a positionally-passed client. The
    per-collection form is the one a ``configure(l2_client=)`` sweep misses -- an explicit
    client WINS over the registry default, so a registry can be L2-live without
    ``l2_client`` appearing anywhere near it.

    :param tree: parsed module
    :ptype tree: ast.AST
    :return: spellings of the L2-live registries
    :rtype: frozenset[str]
    """
    constructed = constructed_registries(tree)
    live: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_receiver = receiver(node)
        names = callee_names(node)
        if call_receiver in constructed and names & L2_BINDER_METHODS and names_a_live_client(node):
            live.add(call_receiver)
            continue
        if not names_a_live_client(node):
            continue
        live.update(spelled for spelled in argument_spellings(node) if spelled in constructed)
    return frozenset(live)
