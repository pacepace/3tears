"""Claims grading's engine-internal steps: listing the claims a candidate's text states, and checking quotes verbatim.

A host declares claims grading on its template and hands each cell's text over as
:class:`~threetears.evals.schema.claims.Segment` values; the engine runs these steps in its judge phase. Nothing here is
a host's to import: the steps are reached through that declaration, so this package exports nothing.
"""

from __future__ import annotations

__all__: list[str] = []
