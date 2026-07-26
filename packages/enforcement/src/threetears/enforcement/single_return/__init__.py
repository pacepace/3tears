"""single-return enforcement domain -- one AST walker.

a function's business logic returns at most once. leading guard
clauses (``if x: return y`` before any other statement) are exempt
and unlimited, because they are the shape that makes a single
business-logic return achievable; the comparison / truthiness / repr
dunders are exempt because the language pushes them toward
branch-returns.

nested ``def`` / ``lambda`` scopes are charged to themselves rather
than to the enclosing function. that is the one place a hand-rolled
version of this walker reliably goes wrong, and the reason this lives
in the shared package: the fix had to be applied twice, in two
verbatim copies, before it moved here.

per-repo configuration goes through :class:`SingleReturnConfig`;
:func:`run_single_return_enforcement` is the pytest-friendly entry
point that orchestrates the walker, applies exemptions, emits the
report, and fails in strict mode.
"""

from threetears.enforcement.single_return.config import SingleReturnConfig
from threetears.enforcement.single_return.runner import run_single_return_enforcement
from threetears.enforcement.single_return.walkers import find_multiple_business_returns

__all__ = [
    "SingleReturnConfig",
    "find_multiple_business_returns",
    "run_single_return_enforcement",
]
