"""Fidelity contracts — proving an eval invokes what production invokes.

**The problem this exists for.** An eval is only a measurement of the system if
it constructs the same call the system constructs. Whether it does cannot be
inferred from structure, read off a diagram, or asserted in a document: the two
paths can look alike, name the same helpers, and still put different bytes on
the wire. It is discovered the expensive way — by reporting a number that turns
out to measure the apparatus. ``classifier_run`` measured a hardcoded generic
prompt for the whole life of the classifier eval; every figure it ever
produced characterised that stub, including one below the chance floor that read
as a model problem.

**The pattern.** A measured behavior is *fidelity-proven* when four things hold.

1. **One constructor.** Exactly one function turns *(subject, stimulus)* into
   the payload the provider receives. Production calls it, the eval calls it,
   and nothing else assembles that payload anywhere in the tree.
2. **One subject constructor.** The subject the eval measures is built by the
   same function that builds production's, so the eval cannot be handed a
   differently-shaped subject. Where the eval reads the subject across a process
   boundary, the *serialised subject* crosses — never a re-derivation of it, and
   never a field-by-field remapping, which is a second derivation wearing a
   plausible name.
3. **A registered contract.** The behavior appears in the host's contract
   registry with its constructor and the callers that must reach it, and a
   canary checks each caller's *source* still references the constructor.
   Source rather than runtime, because a divergence hidden behind an unexecuted
   branch is still a divergence.
4. **A boundary-equality test.** A test drives the real production caller and
   the real eval caller with one subject and one stimulus, captures what each
   hands the provider, and asserts the two payloads are identical. This is the
   load-bearing half. (1)–(3) prove the eval *reaches* the shared path; only (4)
   proves it reaches it with the same arguments.

**What this does not catch, stated so nobody over-trusts it.** A new eval
surface that never registers a contract. A registry is a checklist, not a
detector — it makes an undeclared path visible to a reviewer, and makes a
declared-but-unproven one fail. The check becomes structural only when
evaluability is derived from the host profile rather than declared beside it.

**Host-agnostic by construction, and checked rather than claimed.** This module
holds the mechanism only: a contract is four strings, and the reachability check
is an AST walk over dotted names. Nothing here knows what a classifier is, or
what host it is running in. The registry that names real modules belongs to the
host, because a registry of dotted
paths is host coupling that no import canary can see: the strings resolve at
call time, so a copy of this package carried to a second host would ship a
registry whose every entry fails to import, and that host's first experience of
the fidelity gate would be a red canary about modules it has never had. This
package holds no registry; ``tests/test_hostneutral_fidelity.py`` proves the
mechanism on a package the test writes and no host owns.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "FidelityContract",
    "callers_missing_the_constructor",
    "contract_for",
    "resolve_constructor",
]


@dataclass(frozen=True)
class FidelityContract:
    """One shared construction path and the callers required to reach it.

    Attributes:
        behavior: Unique name for the path. A behavior with more than one shared
            constructor registers one contract per constructor, dotted
            (``"relevance_classifier.call"``).
        constructor: Dotted path to the single function that builds the payload.
        callers: Dotted module paths that must reach ``constructor``. Production
            first, by convention, so a reader sees which side is the reference.
        why: One line naming what breaks when they diverge. Not decoration —
            it is what tells a later reader whether a proposed refactor is
            allowed to split the path.
    """

    behavior: str
    constructor: str
    callers: tuple[str, ...]
    why: str


def contract_for(contracts: tuple[FidelityContract, ...], behavior: str) -> FidelityContract:
    """Look up a contract by behavior name.

    The registry is a parameter rather than a module global: it is the host's
    data, and this module is the part a second host inherits unchanged.

    Args:
        contracts: The host's registry.
        behavior: The contract's ``behavior``.

    Returns:
        The contract.

    Raises:
        KeyError: When no contract carries that name.
    """
    for contract in contracts:
        if contract.behavior == behavior:
            return contract
    raise KeyError(f"no fidelity contract named {behavior!r}")


def resolve_constructor(contract: FidelityContract) -> object:
    """Import a contract's constructor.

    The dotted path may name a function on a module or a method on a class in one
    (``pkg.mod.Class.method``), because a host's shared construction path is as often a
    classmethod or staticmethod as a bare function. The module boundary is found by
    importing the longest prefix that imports, then walking the rest as attributes — a
    prefix split would have to guess, and guessing wrong reports a missing module for a
    constructor that is present.

    Args:
        contract: The contract.

    Returns:
        The constructor object.

    Raises:
        ImportError: When no prefix of the path imports as a module, or when a module on the
            path exists and fails to import.
        AttributeError: When the remaining segments are not attributes of it.
    """
    module, attributes = _split_at_module_boundary(contract.constructor)
    resolved: object = module
    for attribute in attributes:
        resolved = getattr(resolved, attribute)
    return resolved


def _split_at_module_boundary(constructor: str) -> tuple[object, list[str]]:
    """Find where a dotted constructor path stops being a module.

    The ONE derivation of that boundary, shared by :func:`resolve_constructor` and
    :func:`callers_missing_the_constructor`. They ask the same question — is this a
    module-level function, or a method on a class? — and a second answer is how the two
    come to disagree about what a caller must reference.

    Args:
        constructor: The contract's dotted constructor path.

    Returns:
        ``(module, attribute_parts)``: the deepest importable module on the path, and the
        segments below it. One part is a module-level function; two is a method on a class.

    Raises:
        ImportError: When no prefix of the path imports as a module, or when a module on the
            path exists and fails to import.
    """
    parts = constructor.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        candidate = ".".join(parts[:cut])
        try:
            module = importlib.import_module(candidate)
        except ModuleNotFoundError as exc:
            # Only "this prefix is not a module" may be swallowed — that is the search. A
            # module that EXISTS and fails to import (a broken import inside it, a missing
            # dependency) raises ModuleNotFoundError too, naming a DIFFERENT module, and
            # continuing past it would report the constructor as absent from a module that
            # is right there. The fidelity canary's whole job is telling "this path is
            # gone" from "this path is broken".
            #
            # The test is on the prefix and not on equality: importing ``a.b`` when ``a``
            # does not exist reports ``a``, so the ordinary search reports a name SHORTER
            # than the candidate. Equality re-raised on exactly the case this loop exists
            # to walk past.
            if exc.name is None or not (candidate == exc.name or candidate.startswith(f"{exc.name}.")):
                raise
            continue
        return module, parts[cut:]
    raise ImportError(f"no importable module prefix in {constructor!r}")


def _referenced_names(source: str) -> set[str]:
    """Collect every identifier a module's source USES — reads, as a call or as a value.

    Covers the two ways a caller can use a function: a bare name (``build(...)``, or
    ``factory=build`` handing it on) and an attribute access on the module or class
    (``product.build(...)``). An import is deliberately not a use: a caller that still imports
    the constructor and builds its own object instead is exactly the drift a fidelity contract
    exists to catch, and counting the import would leave it green. Names in store context
    (``build = ...``) are not uses either. An AST walk rather than a substring search, so a
    name inside a comment or a docstring does not count as reaching it.

    Args:
        source: Python source text.

    Returns:
        The identifiers referenced.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            names.add(node.attr)
            qualified = _attribute_chain(node)
            if qualified is not None:
                names.add(qualified)
    return names


def _attribute_chain(node: ast.Attribute) -> str | None:
    """Render an attribute access as its dotted source text, when it is a plain chain.

    Anything whose head is a bare NAME renders, however long the chain:
    ``SearchTool.resolve_model_source`` does, and so does ``self._tool.resolve_model_source``
    — ``self`` is a name, so that one renders to ``self._tool.resolve_model_source`` and simply
    fails to match a contract naming ``SearchTool``. What returns ``None`` is a head that is
    not a name at all: ``registry[key].resolve_model_source`` (a subscript) or
    ``build().resolve_model_source`` (a call). The distinction is worth stating precisely
    because the plausible-sounding version of it — "attribute access on self does not render" —
    is false, and a test written against it passes for a reason other than the one it claims.
    Returning ``None`` for a non-name head is the conservative direction: the bare
    attribute is added regardless, so a caller reaching the constructor through an
    expression is still matched by a contract that names a module-level function, and only
    a METHOD contract — which is asking a stricter question on purpose — declines it.

    Args:
        node: The attribute node.

    Returns:
        The dotted chain, or ``None`` when the head is not a bare name.
    """
    parts = [node.attr]
    current: ast.expr = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def callers_missing_the_constructor(contract: FidelityContract) -> list[str]:
    """Name the declared callers whose source no longer reaches the constructor.

    **What "reaches" means depends on where the module boundary falls**, which is why that
    boundary has one derivation (:func:`_split_at_module_boundary`) rather than an
    ``rpartition`` here. A module-level constructor is matched by its bare name, which is
    all a caller can write. A constructor that is a METHOD is matched by
    ``Class.method`` — matching the bare ``method`` would accept an attribute of that name
    on any object at all, and a method name collides far more readily than a module-level
    function name does. A caller that reaches the method on a receiver this cannot identify
    — ``self._tool.resolve_model_source`` renders as that chain and does not MATCH one
    naming the class; ``registry[key].resolve_model_source`` renders nothing at all —
    therefore reads as missing. That is the conservative direction, and a contract whose
    callers genuinely reach it that way should name the module-level seam instead, since a
    fidelity contract is a claim about ONE shared construction path and a path whose
    receiver the walk cannot identify is not one an AST canary can vouch for.

    Args:
        contract: The contract to check.

    Returns:
        The dotted module paths that fail, in declaration order. Empty when
        every caller still references the constructor. A caller whose source
        cannot be located fails too — an unlocatable caller is not a proven one.

    Raises:
        ImportError: When the constructor's own module path does not import — a contract
            whose constructor is gone is a broken contract, not a set of missing callers.
    """
    module, attributes = _split_at_module_boundary(contract.constructor)
    defining_module = getattr(module, "__name__", "")
    qualified = ".".join(attributes[-2:])
    bare = attributes[-1]
    missing: list[str] = []
    for caller in contract.callers:
        spec = importlib.util.find_spec(caller)
        origin = spec.origin if spec is not None else None
        if origin is None:
            missing.append(caller)
            continue
        referenced = _referenced_names(Path(origin).read_text(encoding="utf-8"))
        # Inside the module that DEFINES the class, the bare name is the honest bar: the
        # class's own methods reach each other as ``cls.x`` / ``self.x``, which name the
        # class as surely as ``Class.x`` does while rendering to a chain that cannot match
        # it. Everywhere else the chain is required, because a bare method name matches an
        # attribute of that name on any object at all.
        accepted = {bare} if caller == defining_module else {qualified}
        if not accepted & referenced:
            missing.append(caller)
    return missing
