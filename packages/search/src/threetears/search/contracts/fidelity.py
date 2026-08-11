"""Fidelity vocabulary -- what the consumer needs, what it got (SR-B6).

The consumer states the fidelity it needs on the request; every candidate
reports what fidelity is available from the provider and what was actually
achieved. Extraction a consumer will not read is money and latency spent
for nothing; extraction a consumer needed and did not get is a silent
partial answer -- the two fields exist so neither can happen silently.

Open vocabulary with named well-known values, matching the criteria
discipline: an adapter or a consumer may introduce finer grades, and a
reader that does not recognise one treats it as unknown rather than
failing.
"""

from __future__ import annotations

from typing import Final

__all__ = ["FIDELITY_BYTES", "FIDELITY_CONTENT", "FIDELITY_SNIPPET"]

#: listing-grade: identity, title, provider snippet -- no page content.
FIDELITY_SNIPPET: Final[str] = "snippet"

#: content-grade: the information itself, as extracted text (whether it
#: arrived with the search response or from a later fetch -- SR-A2).
FIDELITY_CONTENT: Final[str] = "content"

#: byte-grade: the raw carrier bytes are fetchable (image file, PDF).
FIDELITY_BYTES: Final[str] = "bytes"
