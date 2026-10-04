"""Resolve an ``ast.ImportFrom`` to the absolute module it actually names.

Every structural canary over this tree walks import statements, and each of them has to
answer the same sub-question before it can judge anything: *what module does this import
name?* For an absolute import the answer is on the node. For a relative one it is arithmetic
over the importing file's own package, and getting it wrong fails in the direction nothing
notices — a mis-resolution names no host module, which every caller reads as "clean".

**Why this is shared when the canaries deliberately are not.** ``test_extraction_import_boundary.py``
and ``test_no_host_names_in_shared_contract.py`` keep their *subjects* independent on purpose:
one asks about host imports and the other about host nouns, and a shared subject is how two
canaries start agreeing with each other instead of with the code. That argument protects a
*judgement*. It says nothing about machinery with a single right answer, which is what this is:
Python's own import semantics decide what ``from ..config import x`` names, no canary has a
position on it, and the walks that each held their own copy did not agree. One skipped relative
imports outright. One read the dotted name off the node unresolved, so ``from ..tools.base``
came out as ``tools.base`` and matched no prefix. One hard-coded the single level its own tree
happens to use. One truncated to a wrong package when the level walked above the root. One was
correct. A copy of an answer nobody disputes is one more chance to hold a different one.

It names nothing but the standard library, so a host's own conformance suite can ask the same
structural questions of its tree with the same resolver.
"""

from __future__ import annotations

import ast
from pathlib import Path


def absolute_module(path: Path, node: ast.ImportFrom, *, root: Path) -> str:
    """Resolve one ``from ... import`` to the absolute module it actually names.

    A walk that skipped relative imports held a rationale true for one level and no more:
    ``from .sibling import x`` inside ``threetears/evals/`` does resolve under ``threetears.evals``,
    so the rule already permits it -- but ``from ..config import app_config`` in the same file
    resolves to ``threetears.config``, and ``from ...models.base import ...`` in ``viz/``
    resolves outside the package too. Both are crossings, and no register, ceiling or
    anti-rot check would have seen them.

    Latent rather than live -- the package writes no relative imports -- and that is the
    condition under which such a hole stays open: the gate goes green either way, so nothing
    ever asks. Resolving is barely more code than skipping and removes the question.

    Args:
        path: The file the import was written in, used to locate its package.
        node: The ``ImportFrom`` node; ``level`` is 0 for an absolute import.
        root: The directory the dotted name is rooted at -- the package's ``src`` here, and
            whatever the importable tree hangs off in a consumer that vendors this file.
            Explicit rather than derived from ``__file__`` so the resolver does not carry an
            assumption about its own depth into a package that will place it elsewhere.

    Returns:
        The absolute dotted module name, or the empty string when the level walks above
        ``root`` — which no importable module can do, so nothing downstream needs to
        distinguish that from "names no host module".
    """
    if not node.level:
        return node.module or ""

    # Level 1 is the containing package, so drop the module's own name first. For an
    # ``__init__.py`` that name IS ``__init__``, and dropping it lands on the same package.
    package = path.relative_to(root).with_suffix("").as_posix().split("/")[:-1]
    upward = node.level - 1
    if upward:
        if upward >= len(package):
            return ""
        package = package[:-upward]

    base = ".".join(package)
    if not node.module:
        return base
    return f"{base}.{node.module}" if base else node.module
