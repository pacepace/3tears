"""The shared relative-import resolver's own arithmetic.

Every structural canary over this tree resolves relative imports through
:func:`~packages.evals.tests.import_resolution.absolute_module`, and its ``level > 0`` branch
executes against nothing in this package: no module under ``threetears.evals`` writes a relative
import today. That is the condition under which a resolver bug stays invisible — a mis-resolution
returns the empty string, which every caller reads as "names no host module", so the failure
mode is a GREEN gate over a real crossing rather than a red one. These cases are the only
thing that exercises the arithmetic before it matters.

They are checked against Python's own import semantics rather than against any canary's
verdict: what a dotted name resolves to is not a question this repo has a position on. The
canaries' own compositions — that a resolved crossing is *judged* as a reach — stay with the
canaries, in ``test_extraction_import_boundary.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from packages.evals.tests.import_resolution import absolute_module


#: The root the synthetic paths below hang off: the package source directory.
_REPO_ROOT = Path(__file__).resolve().parents[1] / "src"


@pytest.mark.parametrize(
    ("source", "level", "module", "expected"),
    [
        # Synthetic paths and names (``leaf``, ``sub``, ``sibling``): the resolver is arithmetic over the
        # path, and a real module here would be rewritten by every package move, which would change the
        # arithmetic being asserted.
        # Level 1 is the containing package: the module's own name drops, nothing else.
        ("threetears/evals/leaf.py", 1, "sibling", "threetears.evals.sibling"),
        ("threetears/evals/leaf.py", 1, None, "threetears.evals"),
        # Level 2 walks one package up -- the crossing the old skip could not see.
        ("threetears/evals/leaf.py", 2, "config", "threetears.config"),
        ("threetears/evals/sub/models.py", 3, "models.base", "threetears.models.base"),
        # An __init__.py's own name IS "__init__", so level 1 lands on its own package
        # rather than on the parent.
        ("threetears/evals/__init__.py", 1, "sibling", "threetears.evals.sibling"),
        ("threetears/evals/sub/__init__.py", 2, "sibling", "threetears.evals.sibling"),
        # Absolute imports are returned unchanged, including the bare-module form.
        ("threetears/evals/leaf.py", 0, "acme.config", "acme.config"),
        # Walking above the root is not something an importable module can do; the empty
        # string is the documented answer. One level past the root and far past it are
        # separate cases: an implementation that slices instead of guarding gets the first
        # right and silently returns a TRUNCATED package for the second.
        ("threetears/evals/leaf.py", 4, "config", ""),
        ("threetears/evals/leaf.py", 9, "config", ""),
    ],
)
def test_a_relative_import_resolves_to_what_it_actually_names(
    source: str, level: int, module: str | None, expected: str
):
    """The resolver's arithmetic, checked against Python's import semantics.

    Args:
        source: Source-root-relative path of the file the import is written in.
        level: The ``ImportFrom`` level; 0 is an absolute import.
        module: The dotted name after the dots, or ``None`` for ``from . import x``.
        expected: The absolute module the import names.
    """
    node = ast.ImportFrom(module=module, names=[ast.alias(name="x")], level=level)
    assert absolute_module(_REPO_ROOT / source, node, root=_REPO_ROOT) == expected


def test_the_root_is_taken_from_the_caller_rather_than_from_this_files_own_depth():
    """The resolver carries no assumption about where it sits in the tree.

    The property that lets it travel: a consumer vendoring this file places it at whatever
    depth its own suite uses, and the dotted names it must produce are rooted at that
    consumer's tree rather than at this repository. Deriving the root from ``__file__``
    would work here and be wrong there, silently — every name would come out with the
    wrong prefix and no crossing would match.
    """
    node = ast.ImportFrom(module="widgets", names=[ast.alias(name="x")], level=2)
    elsewhere = Path("/somewhere/else")

    assert absolute_module(elsewhere / "acme/eval/thing.py", node, root=elsewhere) == "acme.widgets"
