"""The toy host's fidelity registry: the construction paths its eval is held to sharing with production.

A registry of dotted paths is the HOST's data — the engine ships the mechanism
(:class:`~threetears.evals.run.FidelityContract`, :func:`~threetears.evals.run.callers_missing_the_constructor`,
:func:`~threetears.evals.run.resolve_constructor`) and no registry, because a dotted path into one
product's tree means nothing in another's. A host writes this module and one source canary over it
(``tests/test_fidelity_adoption.py`` is the toy host's).
"""

from __future__ import annotations

from threetears.evals.run import FidelityContract

_TOYHOST = "packages.evals.tests.fixtures.toyhost"

#: Every contract the toy host declares. Production first in each ``callers``, by convention.
TOYHOST_FIDELITY_CONTRACTS: tuple[FidelityContract, ...] = (
    FidelityContract(
        behavior="invoice_extraction.request",
        constructor=f"{_TOYHOST}.product.extraction_request",
        callers=(f"{_TOYHOST}.product", f"{_TOYHOST}.kind"),
        why="an extractor eval that builds its own prompt measures that prompt, not the product's",
    ),
)

__all__ = ["TOYHOST_FIDELITY_CONTRACTS"]
