"""The toy host — a complete eval host for invoice field extraction, as reference code.

A product adopts the engine by declaring a vocabulary (a :class:`~threetears.evals.contracts.host.HostProfile`),
writing the candidate kinds that drive what it evaluates, and handing the engine one
:class:`~threetears.evals.contracts.host.EvalHost`. This package is all three for one product, built
against the engine's public roots alone.

**Domain: invoice field extraction.** No conversation and no simulated user: scalar
levers, an observational corpus beside a commissioned run, a code grader beside a model judge, and
one world dimension in every registrable quadrant. Each element exercises one property of the host
contract; ``README.md`` maps them.
"""

from __future__ import annotations

from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_ID, toyhost_profile

__all__ = ["TOYHOST_ID", "toyhost_profile"]
