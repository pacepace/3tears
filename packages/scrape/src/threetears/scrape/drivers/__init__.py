"""ScrapeDriver backend implementations -- every real dependency is constructor-injected."""

from __future__ import annotations

#: Deliberately empty, and documented in `docs/adoption/scrape.md` as such. Each driver is
#: imported from its own module (`from .nodriver_sidecar import NodriverSidecarDriver`), because
#: re-exporting them here would import every backend's dependencies to reach any one of them --
#: camoufox pulls Playwright, the document driver pulls the parsers, and a caller wanting the plain
#: JSON driver should pay for none of it. Populating this would be a behaviour change, not tidying.
__all__: list[str] = []
