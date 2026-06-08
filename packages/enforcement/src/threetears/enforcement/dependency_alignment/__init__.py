"""declared-vs-actual dependency alignment enforcement.

the uv workspace installs every package together, so a package can
import a sibling it never declared (or declare one it never imports)
and tests still pass. the declarations stop describing reality, and
every drift is a latent ``ImportError`` for a standalone
``pip install`` of that package.

this domain walks each workspace package's ``src/`` tree, extracts
``threetears.*`` imports with their import context, and compares
against the package's declared dependencies:

- ``dependency.missing`` -- an unguarded module-top import of another
  workspace distribution that is neither a declared dependency nor
  covered by any optional extra.
- ``dependency.stale`` -- a declared ``3tears*`` dependency that no
  module in the package imports through any path (including guarded,
  deferred, and ``TYPE_CHECKING`` imports).

imports inside ``if TYPE_CHECKING:`` blocks, ``try/except
ImportError`` guards, or function bodies are deliberately NOT flagged
as missing -- those are the sanctioned shapes for optional and
deferred dependencies (see ``channels``' webhook extra) -- but they DO
count as usage when deciding staleness.
"""

from threetears.enforcement.dependency_alignment.config import DependencyAlignmentConfig
from threetears.enforcement.dependency_alignment.runner import run_dependency_alignment_enforcement
from threetears.enforcement.dependency_alignment.walkers import dependency_alignment_violations

__all__ = [
    "DependencyAlignmentConfig",
    "dependency_alignment_violations",
    "run_dependency_alignment_enforcement",
]
