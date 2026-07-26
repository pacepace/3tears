"""robots.txt: honour the rate, escalate the refusal, and never let the file stop the work.

Most of these are about what happens when the file is unhelpful rather than when it is
correct. A missing, broken, slow or nonsense robots.txt must all end in "allowed", because
treating an unreachable text file as a wall lets one 500 stop a scrape the site never
objected to -- and that failure is silent, since nothing looks wrong except that the target
stopped producing data.

Every assertion is on observed behaviour, never on a config flag. A politeness setting that
is "on" while nothing waits is worse than one that is off, because it is believed.
"""

from __future__ import annotations

import pytest
from threetears.scrape.robots import DEFAULT_USER_AGENT, RobotsGate, RobotsPolicy

_ROBOTS_DISALLOW = "User-agent: *\nDisallow: /private\n"
_ROBOTS_DELAY = "User-agent: *\nCrawl-delay: 10\n"
_ROBOTS_BOTH = "User-agent: *\nCrawl-delay: 5\nDisallow: /private\n"


def _fetcher(body: str, status: int = 200):
    calls: list[str] = []

    async def _fetch(url: str) -> tuple[int, str]:
        calls.append(url)
        return status, body

    _fetch.calls = calls  # type: ignore[attr-defined]
    return _fetch


async def test_a_disallowed_path_is_not_fetched_and_asks_for_a_human() -> None:
    """Neither silently obeyed nor silently ignored.

    Obeying makes the target permanently invisible with no way to say "we have an agreement
    with this site"; ignoring is what gives crawlers their reputation. Escalating is the third
    option, and it closes because a person working the page over VNC is not an automated agent.
    """
    gate = RobotsGate(fetch=_fetcher(_ROBOTS_DISALLOW))
    decision = await gate.check("https://example.gov/private/list")

    assert decision.allowed is False
    assert decision.needs_human is True
    assert "disallow" in decision.reason.lower()


async def test_an_allowed_path_on_the_same_site_is_untouched() -> None:
    """A Disallow is per path. Treating one rule as a site-wide ban would lose most of a site."""
    gate = RobotsGate(fetch=_fetcher(_ROBOTS_DISALLOW))
    decision = await gate.check("https://example.gov/public/list")
    assert decision.allowed is True
    assert decision.needs_human is False


async def test_a_crawl_delay_is_actually_waited_between_fetches() -> None:
    """Asserted as an observed wait, not as a stored setting.

    The failure this excludes is the one that matters: a flag that reads "on" while every
    fetch goes out immediately. Nothing about that looks wrong from the inside.
    """
    gate = RobotsGate(fetch=_fetcher(_ROBOTS_DELAY))

    first = await gate.check("https://example.gov/a", now=1000.0)
    assert first.wait_seconds == 0.0, "nothing is owed before the first fetch"
    gate.note_fetched("https://example.gov/a", now=1000.0)

    soon = await gate.check("https://example.gov/b", now=1004.0)
    assert soon.allowed is True
    assert soon.wait_seconds == pytest.approx(6.0), "the site asked for 10s and 4 had passed"

    later = await gate.check("https://example.gov/b", now=1010.0)
    assert later.wait_seconds == 0.0


async def test_the_clock_starts_on_a_fetch_not_on_a_check() -> None:
    """A check that did not lead to a fetch must not consume the site's patience.

    The circuit can suppress a fetch after robots has been consulted, and a caller can change
    its mind. Folding the two would make a rejected check count as a visit.
    """
    gate = RobotsGate(fetch=_fetcher(_ROBOTS_DELAY))
    await gate.check("https://example.gov/a", now=1000.0)
    await gate.check("https://example.gov/a", now=1001.0)

    assert (await gate.check("https://example.gov/a", now=1002.0)).wait_seconds == 0.0


async def test_the_delay_is_per_origin() -> None:
    """One site's patience is not another's, and scheme and port are part of an origin."""
    gate = RobotsGate(fetch=_fetcher(_ROBOTS_DELAY))
    gate.note_fetched("https://a.example/x", now=1000.0)

    assert (await gate.check("https://b.example/x", now=1001.0)).wait_seconds == 0.0
    assert (await gate.check("https://a.example/y", now=1001.0)).wait_seconds > 0.0


async def test_an_absurd_crawl_delay_is_capped() -> None:
    """A file asking for a day is asking us not to crawl, and a scheduler asleep for a day
    is indistinguishable from one that has hung."""
    gate = RobotsGate(RobotsPolicy(max_crawl_delay_seconds=60.0), fetch=_fetcher("User-agent: *\nCrawl-delay: 86400\n"))
    gate.note_fetched("https://example.gov/a", now=1000.0)
    assert (await gate.check("https://example.gov/a", now=1000.0)).wait_seconds == pytest.approx(60.0)


async def test_a_malformed_crawl_delay_is_ignored_without_losing_the_rest_of_the_file() -> None:
    """The site failed to express a rate, not a ban. The Disallow it DID express still holds."""
    gate = RobotsGate(fetch=_fetcher("User-agent: *\nCrawl-delay: soon\nDisallow: /private\n"))
    gate.note_fetched("https://example.gov/a", now=1000.0)

    assert (await gate.check("https://example.gov/a", now=1000.0)).wait_seconds == 0.0
    assert (await gate.check("https://example.gov/private", now=1000.0)).needs_human is True


@pytest.mark.parametrize(
    ("status", "body"),
    [(404, ""), (500, "oops"), (200, ""), (200, "\x00\xff not a robots file")],
    ids=["missing", "server-error", "empty", "garbage"],
)
async def test_an_unusable_robots_file_never_blocks_the_work(status: int, body: str) -> None:
    """Every unusable outcome means the same thing: the site has not told us anything.

    Distinguishing them would produce four ways of saying "allowed", and treating any of them
    as a refusal lets one bad response to a text file stop a scrape silently.
    """
    gate = RobotsGate(fetch=_fetcher(body, status=status))
    decision = await gate.check("https://example.gov/anything")
    assert decision.allowed is True
    assert decision.needs_human is False


async def test_a_fetcher_that_raises_does_not_raise_out_of_check() -> None:
    async def _explode(_url: str) -> tuple[int, str]:
        raise RuntimeError("dns is down")

    gate = RobotsGate(fetch=_explode)
    assert (await gate.check("https://example.gov/x")).allowed is True


async def test_no_fetcher_at_all_is_permissive() -> None:
    """A caller that never wired one gets today's behaviour rather than a silent stop."""
    assert (await RobotsGate().check("https://example.gov/x")).allowed is True


async def test_robots_is_read_once_per_origin_not_once_per_target() -> None:
    """Several targets commonly share an origin. Re-fetching per target would make politeness
    cost more requests than it saves."""
    fetch = _fetcher(_ROBOTS_BOTH)
    gate = RobotsGate(fetch=fetch)

    for path in ("/a", "/b", "/c"):
        await gate.check(f"https://example.gov{path}", now=1000.0)

    assert fetch.calls == ["https://example.gov/robots.txt"]  # type: ignore[attr-defined]


async def test_both_behaviours_can_be_turned_off_independently() -> None:
    """They are separate settings because they are separate decisions.

    A deployment may have a written agreement covering access while still wanting to be
    polite about rate -- or the reverse.
    """
    fetch_off = RobotsGate(RobotsPolicy(flag_disallowed=False), fetch=_fetcher(_ROBOTS_BOTH))
    assert (await fetch_off.check("https://example.gov/private")).allowed is True

    delay_off = RobotsGate(RobotsPolicy(respect_crawl_delay=False), fetch=_fetcher(_ROBOTS_BOTH))
    delay_off.note_fetched("https://example.gov/a", now=1000.0)
    assert (await delay_off.check("https://example.gov/a", now=1000.0)).wait_seconds == 0.0


async def test_both_behaviours_are_on_without_configuration() -> None:
    """The default is the whole point. Opt-in politeness is absent wherever nobody configured it."""
    policy = RobotsPolicy()
    assert policy.respect_crawl_delay is True
    assert policy.flag_disallowed is True
    assert policy.user_agent == DEFAULT_USER_AGENT

    gate = RobotsGate(fetch=_fetcher(_ROBOTS_BOTH))
    gate.note_fetched("https://example.gov/a", now=1000.0)
    assert (await gate.check("https://example.gov/private")).needs_human is True
    assert (await gate.check("https://example.gov/a", now=1001.0)).wait_seconds > 0.0


async def test_an_explicit_origin_override_skips_the_file_entirely() -> None:
    """For a deployment with a written agreement with one site.

    Per origin and recorded, deliberately: "we have permission for this one" and "ignore
    robots everywhere" must not be the same setting.
    """
    gate = RobotsGate(
        RobotsPolicy(overrides=frozenset({"https://example.gov"})),
        fetch=_fetcher(_ROBOTS_DISALLOW),
    )
    decision = await gate.check("https://example.gov/private")
    assert decision.allowed is True
    assert "override" in decision.reason


async def test_a_non_http_url_has_no_robots_to_consult() -> None:
    assert (await RobotsGate(fetch=_fetcher(_ROBOTS_DISALLOW)).check("file:///etc/passwd")).allowed is True


# ---------------------------------------------------------------------------
# The wiring. These are the tests whose absence let chunk 09 be marked done
# while nothing consulted a robots.txt -- the same defect as chunk 06, against
# a rule this plan had already recorded: any chunk that widens a contract names
# the caller that closes it.
# ---------------------------------------------------------------------------


class _RecordingDriver:
    """# parity-with: threetears.scrape.driver.ScrapeDriver"""

    def __init__(self) -> None:
        self.fetched: list[str] = []

    @property
    def name(self) -> str:
        return "recording"

    async def render(
        self,
        url: str,
        *,
        timeout: float = 30.0,
        wait_for: str | None = None,
        capture_network: bool = False,
        nav_steps: list | None = None,
        session_state: dict | None = None,
    ):
        from threetears.scrape.driver import RenderedPage

        self.fetched.append(url)
        return RenderedPage(
            html="<html><body><table><tr><td>Acme Corp</td><td>42</td></tr></table></body></html>",
            status=200,
            final_url=url,
            timing_ms=1.0,
        )


async def _tool_with_robots(driver, gate, *, target_id: str):
    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import DefaultCoreConfig
    from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection
    from threetears.scrape.tool import ScrapeTool

    reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
    recipes = ScrapeRecipeCollection(reg, cfg, nats_client=None)
    await recipes.save_entity(
        recipes.create(
            {
                "target_id": target_id,
                "extraction_strategy": {"employer": "td:nth-child(1)", "affected_count": "td:nth-child(2)"},
                "won_at": None,
                "last_validated_at": None,
                "consecutive_validation_failures": 0,
            }
        )
    )
    return ScrapeTool(
        recipe_collection=recipes,
        extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
        drivers={"nodriver": driver},
        robots=gate,
        api_key="k",
    )


async def test_a_disallowed_target_is_never_fetched_and_asks_for_a_human() -> None:
    """The whole chunk, at the only boundary that can prove it.

    Every piece of `robots.py` was built and tested and NOTHING consulted it, so no crawl
    delay was ever waited and no Disallow ever escalated -- and nothing in a log or a column
    would have shown it. The driver's call list is the assertion, because "we did not fetch"
    is the claim.
    """
    from threetears.scrape.tool import _derive_target_id

    url = "https://example.gov/private/list"
    schema = {"employer": "str", "affected_count": "int"}
    driver = _RecordingDriver()
    gate = RobotsGate(fetch=_fetcher(_ROBOTS_DISALLOW))
    tool = await _tool_with_robots(driver, gate, target_id=_derive_target_id(url, schema))

    result = await tool.execute(url=url, field_schema=schema)

    assert driver.fetched == [], "a disallowed target was fetched anyway"
    assert result.success is False
    assert result.metadata["needs_human"] is True
    assert result.metadata["reason"] == "robots_disallow"


async def test_an_allowed_target_is_fetched_normally() -> None:
    """A Disallow on one path must not stop the rest of a site."""
    from threetears.scrape.tool import _derive_target_id

    url = "https://example.gov/public/list"
    schema = {"employer": "str", "affected_count": "int"}
    driver = _RecordingDriver()
    gate = RobotsGate(fetch=_fetcher(_ROBOTS_DISALLOW))
    tool = await _tool_with_robots(driver, gate, target_id=_derive_target_id(url, schema))

    await tool.execute(url=url, field_schema=schema)

    assert driver.fetched == [url]


async def test_a_crawl_delay_actually_delays_a_second_fetch() -> None:
    """Observed timing, not a stored setting.

    A politeness flag that reads "on" while every fetch goes out immediately is worse than one
    that is off, because it is believed. The sleep is patched so the test asserts the wait was
    REQUESTED and for how long, without spending it.
    """
    from unittest.mock import AsyncMock, patch

    from threetears.scrape.tool import _derive_target_id

    url = "https://example.gov/list"
    schema = {"employer": "str", "affected_count": "int"}
    driver = _RecordingDriver()
    gate = RobotsGate(fetch=_fetcher(_ROBOTS_DELAY))
    tool = await _tool_with_robots(driver, gate, target_id=_derive_target_id(url, schema))

    with patch("threetears.scrape.tool.asyncio.sleep", new=AsyncMock()) as slept:
        await tool.execute(url=url, field_schema=schema)
        assert slept.await_count == 0, "nothing is owed before the first fetch"

        await tool.execute(url=url, field_schema=schema)
        assert slept.await_count == 1, "the site asked for 10s between requests and none was waited"
        # The docstring claimed this asserted "for how long" while asserting only "> 0".
        # Ten seconds requested, one fetch ago, so the remainder is the whole delay.
        assert slept.await_args[0][0] == pytest.approx(10.0, abs=1.0)

    assert driver.fetched == [url, url]


async def test_passing_none_explicitly_consults_nothing() -> None:
    """The opt-OUT, which now has to be asked for.

    Previously this was what every caller got by omission, so a documented on-by-default
    setting was off in every deployment while the configuration looked correct.
    """
    from threetears.scrape.tool import _derive_target_id

    url = "https://example.gov/private/list"
    schema = {"employer": "str", "affected_count": "int"}
    driver = _RecordingDriver()
    tool = await _tool_with_robots(driver, None, target_id=_derive_target_id(url, schema))

    await tool.execute(url=url, field_schema=schema)

    assert driver.fetched == [url]


async def test_a_tool_built_without_mentioning_robots_still_gets_a_gate() -> None:
    """ "On by default" has to be true of a tool nobody configured, or it is not a default.

    This is the chunk's own "Watch for" one layer up from where it was first fixed: a config
    that is on by default while reading a value nothing sets is on in name and off in fact.
    """
    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import DefaultCoreConfig
    from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection
    from threetears.scrape.robots import RobotsGate
    from threetears.scrape.tool import ScrapeTool

    reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
    tool = ScrapeTool(
        recipe_collection=ScrapeRecipeCollection(reg, cfg, nats_client=None),
        extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
        drivers={},
        api_key="k",
    )

    assert isinstance(tool._robots, RobotsGate), "a tool that never mentioned robots consults none"  # noqa: SLF001


async def test_a_gate_with_no_arguments_actually_reads_a_file() -> None:
    """The other half of the same failure: a default gate with no fetcher reads nothing.

    "Both behaviours on by default" would then be true of a policy object and false of every
    deployment, with no log line anywhere saying so.
    """
    gate = RobotsGate()
    assert gate._fetch is not None  # noqa: SLF001


async def test_a_suppressed_fetch_does_not_pay_the_crawl_delay() -> None:
    """Politeness is owed for a request that happens, not for one the circuit refuses.

    Waiting before the circuit check would block a caller for up to the delay ceiling only to
    be told the fetch was suppressed -- a politeness cost paid to a site that receives nothing.
    Both gates are still honoured; only the order changed.
    """
    from unittest.mock import AsyncMock, patch

    from threetears.scrape.circuit import BackoffPolicy, TargetCircuit
    from threetears.scrape.health import ScrapeTargetHealthCollection
    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import DefaultCoreConfig
    from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection
    from threetears.scrape.tool import ScrapeTool, _derive_target_id

    url = "https://example.gov/list"
    schema = {"employer": "str", "affected_count": "int"}
    target_id = _derive_target_id(url, schema)

    reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
    health = ScrapeTargetHealthCollection(reg, cfg, nats_client=None)
    circuit = TargetCircuit(health, policy=BackoffPolicy(failure_threshold=1))
    await circuit.record_blocked(target_id)

    gate = RobotsGate(fetch=_fetcher(_ROBOTS_DELAY))
    gate.note_fetched(url)

    driver = _RecordingDriver()
    tool = ScrapeTool(
        recipe_collection=ScrapeRecipeCollection(reg, cfg, nats_client=None),
        extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
        health_collection=health,
        circuit=circuit,
        drivers={"nodriver": driver},
        robots=gate,
        api_key="k",
    )

    with patch("threetears.scrape.tool.asyncio.sleep", new=AsyncMock()) as slept:
        result = await tool.execute(url=url, field_schema=schema)

    assert driver.fetched == [], "the circuit was supposed to suppress this fetch"
    assert slept.await_count == 0, "a suppressed fetch still paid the crawl delay"
    assert "backing off" in (result.error or "")
