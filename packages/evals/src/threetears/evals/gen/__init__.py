"""The engine's generation package: the prompts and expanders that author test material.

Of the engine it imports only itself and :mod:`threetears.evals.contracts`.

**This module is the package's public root.** A host imports from here and from no module below
it, and only the names in ``__all__``; ``tests/test_package_matrix.py`` holds that, and
``tests/test_public_surface_is_closed.py`` holds every engine type a public signature hands a host to
being exported from a public root. A ``# debt:`` comment on an export names what retires it. Code
inside the package imports its own modules directly.
"""

from __future__ import annotations

from threetears.evals.gen.prompts.boundary_gen import EVAL_BOUNDARY_GEN_TEMPLATE_DEFAULT
from threetears.evals.gen.prompts.proposer import EVAL_PROPOSER_TEMPLATE_DEFAULT
from threetears.evals.gen.proposers import PROPOSER_MAX_TOKENS, ProposedDraft, propose_draft
from threetears.evals.gen.variation_gen import generate_variations, price_variations
from threetears.evals.gen.variation_gen import EvalTestCaseStore, GeneratedVariations


__all__ = [
    "EVAL_BOUNDARY_GEN_TEMPLATE_DEFAULT",
    "EVAL_PROPOSER_TEMPLATE_DEFAULT",
    "PROPOSER_MAX_TOKENS",
    "EvalTestCaseStore",
    "GeneratedVariations",
    "ProposedDraft",
    "generate_variations",
    "price_variations",
    "propose_draft",
]
