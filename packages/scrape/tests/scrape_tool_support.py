"""Shared ways of asking a :class:`ScrapeTool` what it would do, through its own front door.

Imported by its repo-root name (``packages.scrape.tests.scrape_tool_support``), like the other
support modules here: not a ``test_*`` module, so pytest does not collect it, and public because
its sibling test modules share it.
"""

from __future__ import annotations

from typing import Any

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection
from threetears.scrape.driver import NavStep, RenderedPage
from threetears.scrape.robots import RobotsGate
from threetears.scrape.tool import ScrapeTool

__all__ = ["NeverRendersDriver", "derived_target_id"]


async def _disallow_everything(_url: str) -> tuple[int, str]:
    return 200, "User-agent: *\nDisallow: /\n"


# parity-with: threetears.scrape.driver.ScrapeDriver
class NeverRendersDriver:
    """A driver whose render is a failure: the probe below must stop before any fetch."""

    @property
    def name(self) -> str:
        return "never-renders"

    async def render(
        self,
        url: str,
        *,
        timeout: float = 30.0,
        wait_for: str | None = None,
        capture_network: bool = False,
        nav_steps: list[NavStep] | None = None,
        session_state: dict[str, object] | None = None,
    ) -> RenderedPage:
        raise AssertionError(f"the target-id probe fetched {url}; it must stop at the robots gate")


async def derived_target_id(url: str, field_schema: dict[str, Any]) -> str:
    """The ``target_id`` ScrapeTool derives for an ad-hoc call with no caller-supplied one.

    Read off the tool's own answer rather than recomputed: a robots file disallowing every
    path stops the call before any fetch, and the ``needs_human`` escalation it returns
    names the target it was deciding about. A test that seeds a recipe, a health row or a
    stored solve for an ad-hoc target keys it the way the tool will, without binding to how
    the tool computes the key.

    :param url: the url the ad-hoc call scrapes
    :ptype url: str
    :param field_schema: the call's field schema, as the tool's input takes it
    :ptype field_schema: dict[str, Any]
    :return: the derived target id
    :rtype: str
    """
    registry, config = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
    tool = ScrapeTool(
        recipe_collection=ScrapeRecipeCollection(registry, config, nats_client=None),
        extraction_collection=ScrapeExtractionCollection(registry, config, nats_client=None),
        drivers={"nodriver": NeverRendersDriver()},
        api_key="unused-the-call-stops-at-robots",
        robots=RobotsGate(fetch=_disallow_everything),
        block_private_hosts=False,
    )
    result = await tool.execute(url=url, field_schema=field_schema)
    assert result.metadata["validation_status"] == "needs_human", (
        f"the target-id probe did not stop at the robots gate: {result.error!r}"
    )
    target_id: str = result.metadata["target_id"]
    return target_id
