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

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from _pacer_fakes import _FakeDelayPacer
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


async def test_a_tool_built_without_mentioning_robots_still_consults_one() -> None:
    """The default has to be provable without the test supplying the thing it is proving.

    My first version of this assigned `tool._robots` before asserting, so reverting the
    constructor default to `None` left it green -- it tested the gate the test had just
    installed. The inline comment claimed it substituted only the FETCHER; the line replaced
    the whole gate.

    Patching the default FETCHER BUILDER instead leaves the tool's own wiring untouched: if
    `ScrapeTool` stops building a gate, nothing consults the stub and the disallowed path is
    fetched.
    """
    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import DefaultCoreConfig
    from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection
    from threetears.scrape.tool import ScrapeTool, _derive_target_id

    url = "https://example.gov/private/list"
    schema = {"employer": "str", "affected_count": "int"}
    driver = _RecordingDriver()

    def _disallowing(_egress=None):
        async def _fetch(_url: str) -> tuple[int, str]:
            return 200, _ROBOTS_DISALLOW

        return _fetch

    reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
    recipes = ScrapeRecipeCollection(reg, cfg, nats_client=None)
    await recipes.save_entity(
        recipes.create(
            {
                "target_id": _derive_target_id(url, schema),
                "extraction_strategy": {"employer": "td:nth-child(1)", "affected_count": "td:nth-child(2)"},
                "won_at": None,
                "last_validated_at": None,
                "consecutive_validation_failures": 0,
            }
        )
    )

    with patch("threetears.scrape.robots._default_fetch_via", _disallowing):
        # Built AFTER the patch and never touched again: the gate under test is the one the
        # constructor decides to make, which is the whole claim.
        tool = ScrapeTool(
            recipe_collection=recipes,
            extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
            drivers={"nodriver": driver},
            api_key="k",
        )
        result = await tool.execute(url=url, field_schema=schema)

    assert driver.fetched == [], "a tool that never mentioned robots fetched a disallowed path"
    assert result.metadata["needs_human"] is True


async def test_a_gate_with_no_arguments_actually_reaches_for_a_file() -> None:
    """The other half of the same failure, also asserted behaviourally.

    A default gate with no fetcher reads nothing, so "both behaviours on by default" would be
    true of a policy object and false of every deployment. The conftest replaces the default
    fetcher with one that raises, so what this proves is that a bare gate CALLS it -- reaching
    the "site told us nothing" path rather than skipping the read entirely.
    """
    reached: list[str] = []

    def _offline(_egress=None):
        async def _fetch(url: str) -> tuple[int, str]:
            reached.append(url)
            raise RuntimeError("no network")

        return _fetch

    with patch("threetears.scrape.robots._default_fetch_via", _offline):
        decision = await RobotsGate().check("https://example.gov/x")

    assert reached == ["https://example.gov/robots.txt"], "a bare gate never went looking for a file"
    assert decision.allowed is True, "an unreachable file must not block the work"


@pytest.mark.real_robots_fetch
async def test_the_default_fetcher_leaves_by_the_configured_exit() -> None:
    """The branch's one security fix, guarded by a test that fails when it regresses.

    My first version asserted `gate._egress is not None` and then inspected the EGRESS
    DRIVER's transport -- never the fetcher. Removing `transport=` from `_default_fetch_via`
    left it green, so the fix protecting against disclosing the container's real address had
    no protection of its own. The autouse conftest made it worse: it patched the real builder
    suite-wide, so nothing anywhere executed it.

    This runs the REAL builder (hence the marker) and intercepts the client it constructs, so
    the assertion is on the transport the robots request would actually use.
    """
    import httpx

    from threetears.core.egress import ProxyEgress
    from threetears.scrape import robots as robots_mod

    captured: dict[str, Any] = {}
    real_client = httpx.AsyncClient

    class _Recording(real_client):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["transport"] = kwargs.get("transport")
            super().__init__(
                *args, **{**kwargs, "transport": httpx.MockTransport(lambda _r: httpx.Response(200, text=""))}
            )

    fetch = robots_mod._default_fetch_via(ProxyEgress("tor", "socks5://127.0.0.1:9050"))
    with patch.object(httpx, "AsyncClient", _Recording):
        await fetch("https://example.gov/robots.txt")

    transport = captured["transport"]
    assert transport is not None, "the robots read went out on the container's own route"
    pool = transport._pool
    assert "9050" in str(getattr(pool, "_proxy_url", "")), "the robots read left by the wrong exit"


@pytest.mark.real_robots_fetch
async def test_the_default_fetcher_with_no_exit_binds_no_transport() -> None:
    """A deployment with no egress configured gets httpx's own default, not a broken one."""
    import httpx

    from threetears.scrape import robots as robots_mod

    captured: dict[str, Any] = {}
    real_client = httpx.AsyncClient

    class _Recording(real_client):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["transport"] = kwargs.get("transport")
            super().__init__(
                *args, **{**kwargs, "transport": httpx.MockTransport(lambda _r: httpx.Response(200, text=""))}
            )

    fetch = robots_mod._default_fetch_via(None)
    with patch.object(httpx, "AsyncClient", _Recording):
        await fetch("https://example.gov/robots.txt")

    assert captured["transport"] is None


async def _seed(tool, url: str, schema: dict) -> None:
    """Give the tool a winning recipe so no test here reaches a model."""
    from threetears.scrape.tool import _derive_target_id

    recipes = tool._recipe_collection
    await recipes.save_entity(
        recipes.create(
            {
                "target_id": _derive_target_id(url, schema),
                "extraction_strategy": {"employer": "td:nth-child(1)", "affected_count": "td:nth-child(2)"},
                "won_at": None,
                "last_validated_at": None,
                "consecutive_validation_failures": 0,
            }
        )
    )


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


async def test_a_fetch_of_an_unblocked_target_writes_nothing() -> None:
    """The clear-down runs on every allowed fetch, so it must not cost a write on every one.

    Unconditional, it created a health row for targets that had never had one and put the
    optimistic-lock fence on the hot path of every poll, to correct a state the overwhelming
    majority of targets have never been in.
    """
    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import DefaultCoreConfig
    from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection
    from threetears.scrape.health import ScrapeTargetHealthCollection
    from threetears.scrape.tool import ScrapeTool, _derive_target_id

    url = "https://example.gov/public/list"
    schema = {"employer": "str", "affected_count": "int"}
    target_id = _derive_target_id(url, schema)

    reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
    health = ScrapeTargetHealthCollection(reg, cfg, nats_client=None)
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
    driver = _RecordingDriver()
    tool = ScrapeTool(
        recipe_collection=recipes,
        extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
        health_collection=health,
        drivers={"nodriver": driver},
        robots=RobotsGate(fetch=_fetcher(_ROBOTS_DISALLOW)),
        api_key="k",
    )

    # Asserted on the WRITE, not on the resulting row. The row-shape version could not fail:
    # nothing on this path ever sets `robots_blocked_at`, so restoring the unconditional clear
    # -- which creates a row with `robots_blocked_at=None` -- satisfied both disjuncts and the
    # test stayed green against the exact regression its own docstring named.
    from unittest.mock import AsyncMock

    clear = AsyncMock()
    with (
        patch("threetears.scrape.eval_loop.classify_failed_page", return_value=None),
        patch("threetears.scrape.tool.clear_robots_block", new=clear),
    ):
        await tool.execute(url=url, field_schema=schema)

    clear.assert_not_awaited()


async def test_a_health_store_failure_does_not_escape_the_clear_down() -> None:
    """The clear-down sits outside every try in `execute`, so it has to hold its own.

    A housekeeping write must never turn a page the caller already paid for into a failed
    ToolResult -- the same posture `circuit.py` takes for its identical call.

    Asserted on the method rather than through `execute`: driving it end to end means a real
    extraction, a health collection and the classifier behind it, so a failure there would be
    attributed to this and a pass would prove less than it appears to.
    """
    from unittest.mock import AsyncMock

    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import DefaultCoreConfig
    from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection
    from threetears.scrape.health import ScrapeTargetHealthCollection, record_robots_block
    from threetears.scrape.tool import ScrapeTool

    reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
    health = ScrapeTargetHealthCollection(reg, cfg, nats_client=None)
    await record_robots_block(health, target_id="t", reason="was disallowed")

    tool = ScrapeTool(
        recipe_collection=ScrapeRecipeCollection(reg, cfg, nats_client=None),
        extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
        health_collection=health,
        drivers={},
        api_key="k",
    )

    clear = AsyncMock(side_effect=RuntimeError("health store is gone"))
    with patch("threetears.scrape.tool.clear_robots_block", new=clear):
        # Returns rather than raises: the caller's fetch is already done and paid for.
        await tool._clear_robots_block_if_any("t")

    # And it REACHED the clear. Asserting only "no exception" would pass against a method that
    # returns unconditionally -- so inverting the guard would leave every test here green
    # while a blocked target could never be released.
    clear.assert_awaited_once()


# ---------------------------------------------------------------------------
# The fleet pacer, and what it may not override. Previously the module's one
# untested branch, which is why it could quietly return the wrong answer: the
# pacer is a RATE primitive with no idea what any particular site asked for.
# ---------------------------------------------------------------------------


async def test_a_granted_fleet_claim_does_not_cancel_the_site_s_own_crawl_delay() -> None:
    """The bug this excludes returns 0.0 the moment a pacer is injected.

    The pacer coordinates the fleet against a bucket configured in deployment; it is never
    told what this origin's file said. So treating a granted claim as the whole answer
    discards the parsed Crawl-delay outright -- and does it only in the fleet deployments
    where several pods make that delay matter most, which is the worst possible place for it
    to happen and the least likely place for anyone to notice.
    """
    pacer = _FakeDelayPacer(claimed=True)
    gate = RobotsGate(fetch=_fetcher(_ROBOTS_DELAY), delay_pacer=pacer)

    gate.note_fetched("https://example.gov/a", now=1000.0)
    decision = await gate.check("https://example.gov/b", now=1004.0)
    fleet = await gate.claim_fleet_turn("https://example.gov/b")

    assert pacer.keys == ["https://example.gov"], "the pacer is keyed per origin"
    # Each half asserted on its own. Writing `max(...)` here would move the combination into
    # the test body -- the test would then compute what production is supposed to compute, and
    # deleting the `max` in `ScrapeTool` would leave it green. That combination is asserted
    # where it lives, in test_tool.py.
    assert decision.wait_seconds == pytest.approx(6.0), "the site asked 10s and 4 had passed"
    assert fleet == 0.0, "a granted claim owes nothing, and must not shorten the site's own delay"


async def test_the_longer_of_the_two_constraints_wins_when_the_fleet_is_slower() -> None:
    """Both bind, so the pacer can only ever make the wait longer, never shorter."""
    pacer = _FakeDelayPacer(claimed=False, retry_after_seconds=25.0)
    gate = RobotsGate(fetch=_fetcher(_ROBOTS_DELAY), delay_pacer=pacer)

    gate.note_fetched("https://example.gov/a", now=1000.0)
    decision = await gate.check("https://example.gov/b", now=1004.0)
    fleet = await gate.claim_fleet_turn("https://example.gov/b")

    assert decision.wait_seconds == pytest.approx(6.0), "this pod's own clock still owes 6s"
    assert fleet == pytest.approx(25.0), "and the fleet owes 25s; the caller takes the longer"


async def test_a_pacer_outage_falls_back_to_this_pod_s_own_clock() -> None:
    """A KV outage costs fleet-wide precision, never politeness and never the scrape."""

    class _BrokenPacer(_FakeDelayPacer):
        async def claim(self, key: str = "default", *, tokens: float = 1.0, max_wait_seconds: float = 0.0) -> Any:
            raise RuntimeError("kv unavailable")

    gate = RobotsGate(fetch=_fetcher(_ROBOTS_DELAY), delay_pacer=_BrokenPacer())
    gate.note_fetched("https://example.gov/a", now=1000.0)

    decision = await gate.check("https://example.gov/b", now=1004.0)
    fleet = await gate.claim_fleet_turn("https://example.gov/b")
    assert decision.allowed is True, "an outage must not stop the scrape"
    assert fleet == 0.0, "an outage yields no fleet wait rather than raising"
    assert decision.wait_seconds == pytest.approx(6.0), "and must not stop the politeness either"


# ---------------------------------------------------------------------------
# Bounded retention: both per-origin stores are caches, so they self-bound.
# ---------------------------------------------------------------------------


async def test_the_per_origin_stores_do_not_grow_without_bound() -> None:
    """A long-lived process scraping a wide set of sites otherwise holds every origin forever.

    Asserted by observing the store after more origins than the cap, rather than by reading
    the cap back off the object -- a bound that is configured but never enforced is exactly
    the failure worth excluding.
    """
    gate = RobotsGate(fetch=_fetcher(_ROBOTS_DELAY), max_origins=3)

    for i in range(10):
        await gate.check(f"https://s{i}.example/a", now=1000.0 + i)
        gate.note_fetched(f"https://s{i}.example/a", now=1000.0 + i)

    assert len(gate._cache) == 3, "the parsed files are capped"
    assert len(gate._last_fetch_at) == 3, "and so are the fetch clocks"
    assert "https://s9.example" in gate._cache, "the most recent origin survives"
    assert "https://s0.example" not in gate._cache, "the oldest is the one evicted"


async def test_forget_drops_one_origin_and_leaves_the_rest() -> None:
    """For a caller retiring a site, mirroring the circuit's own forget_target lever."""
    gate = RobotsGate(fetch=_fetcher(_ROBOTS_DELAY))
    await gate.check("https://a.example/x", now=1000.0)
    await gate.check("https://b.example/x", now=1000.0)
    gate.note_fetched("https://a.example/x", now=1000.0)

    gate.forget("https://a.example/anything")

    assert "https://a.example" not in gate._cache
    assert "https://a.example" not in gate._last_fetch_at
    assert "https://b.example" in gate._cache, "forgetting one origin leaves the others"


async def test_a_forget_during_a_fetch_is_not_undone_by_it() -> None:
    """The window between dropping the lock to fetch and retaking it to write.

    `forget` exists for "this site's file just changed". If the completing write can resurrect
    the pre-forget parser, the method does the opposite of what it advertises precisely when a
    caller reaches for it -- and does so only under concurrency, so nothing looks wrong.
    """
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_fetch(url: str) -> tuple[int, str]:
        started.set()
        await release.wait()
        return 200, _ROBOTS_DISALLOW

    gate = RobotsGate(fetch=_slow_fetch)
    task = asyncio.create_task(gate.check("https://example.gov/private/x", now=1000.0))
    await asyncio.wait_for(started.wait(), timeout=1.0)

    gate.forget("https://example.gov")
    release.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert "https://example.gov" not in gate._cache, (
        "the in-flight fetch wrote back over the forget, so the next check reuses the file that was discarded"
    )


async def test_the_declared_timeout_covers_the_longest_wait_the_gate_can_ask_for() -> None:
    """Read off the declared budget, not a literal, which is the point.

    The robots wait happens BEFORE the render, so a budget derived from the render alone
    describes a shorter call than the one that runs. An executor obeying it cancels mid-sleep,
    and that cancellation is what stranded the circuit's probe -- the deadline manufactured the
    failure it was supposed to bound.
    """
    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import DefaultCoreConfig
    from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection
    from threetears.scrape.tool import ScrapeTool

    reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
    gate = RobotsGate(RobotsPolicy(max_crawl_delay_seconds=240.0), fetch=_fetcher(_ROBOTS_DELAY))
    tool = ScrapeTool(
        recipe_collection=ScrapeRecipeCollection(reg, cfg, nats_client=None),
        extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
        drivers={},
        api_key="k",
        robots=gate,
    )

    declared = tool.mcp_schema().timeout_seconds
    assert declared is not None
    assert declared >= gate.max_wait_seconds, (
        f"the tool advertises {declared}s while its own robots gate can sleep for "
        f"{gate.max_wait_seconds}s before the render even starts"
    )


async def test_turning_the_crawl_delay_off_stops_charging_the_budget_for_it() -> None:
    """A ceiling that is never waited must not inflate every declared deadline."""
    gate = RobotsGate(RobotsPolicy(respect_crawl_delay=False, max_crawl_delay_seconds=300.0))
    assert gate.max_wait_seconds == 0.0


async def test_a_health_store_failure_does_not_escape_the_escalation() -> None:
    """This tool returns a ToolResult; it does not raise. The queue write is not exempt.

    `record_robots_block` was the one health write in `execute` with no guard, so a store
    outage turned "this target needs a human" into an exception out of a tool whose contract
    is that it never raises -- losing the escalation entirely rather than just the queue entry.
    """
    from unittest.mock import patch as _patch

    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import DefaultCoreConfig
    from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection
    from threetears.scrape.health import ScrapeTargetHealthCollection
    from threetears.scrape.tool import ScrapeTool

    reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
    health = ScrapeTargetHealthCollection(reg, cfg, nats_client=None)
    tool = ScrapeTool(
        recipe_collection=ScrapeRecipeCollection(reg, cfg, nats_client=None),
        extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
        drivers={"nodriver": _RecordingDriver()},
        api_key="k",
        health_collection=health,
        robots=RobotsGate(fetch=_fetcher(_ROBOTS_DISALLOW)),
    )

    with _patch("threetears.scrape.tool.record_robots_block", side_effect=RuntimeError("store down")):
        result = await tool.execute(url="https://example.gov/private/x", field_schema={"a": "str"})

    assert result.success is False
    assert "needs a human" in (result.error or ""), (
        "a store outage swallowed the escalation instead of costing only the queue entry"
    )


async def test_a_circuit_probe_is_not_exempt_from_the_crawl_delay() -> None:
    """One of chunk 09's own named traps, and the rule both module docstrings state.

    The two gates are different kinds: `Crawl-delay` is a FLOOR on politeness that applies to
    a target working perfectly, while the circuit's `blocked_until` is a CEILING on cost that
    applies to one that is not. Neither may weaken the other. Exempting a probe is the
    tempting shortcut -- probes are rare and someone wants to know -- and it breaks the
    politeness contract precisely when the target is already unhappy with us.

    The setup drives the DURABLE row into a probe, not just an in-process breaker: `check`
    answers from the health row, so a test that only opened a local breaker got a plain closed
    circuit and asserted nothing about probes at all.
    """
    from datetime import datetime, timedelta, timezone

    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import DefaultCoreConfig
    from threetears.models.circuit_breaker import CircuitState
    from threetears.scrape.circuit import TargetCircuit
    from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection
    from threetears.scrape.health import ScrapeTargetHealthCollection, record_circuit_state
    from threetears.scrape.tool import ScrapeTool, _derive_target_id

    reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
    health = ScrapeTargetHealthCollection(reg, cfg, nats_client=None)
    url, schema = "https://example.gov/probed", {"a": "str"}
    target_id = _derive_target_id(url, schema)

    # An OPEN circuit whose window has already elapsed: the next check promotes it and admits
    # the single probe. That is the state the rule is about.
    now = datetime.now(timezone.utc)
    await record_circuit_state(
        health,
        target_id=target_id,
        circuit_state=CircuitState.OPEN.value,
        consecutive_fetch_failures=3,
        blocked_until=now - timedelta(seconds=1),
    )
    # Spied rather than pre-checked: a verifying `check()` here would CONSUME the single
    # probe, after which the tool's own check is suppressed and the delay is legitimately
    # skipped -- the test would then fail for a reason that has nothing to do with the rule.
    decisions: list[Any] = []

    class _SpyCircuit(TargetCircuit):
        async def check(self, target: str, *, now: Any = None) -> Any:
            decision = await super().check(target, now=now)
            decisions.append(decision)
            return decision

    gate = RobotsGate(fetch=_fetcher(_ROBOTS_DELAY))
    gate.note_fetched(url)
    tool = ScrapeTool(
        recipe_collection=ScrapeRecipeCollection(reg, cfg, nats_client=None),
        extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
        drivers={"nodriver": _RecordingDriver()},
        api_key="k",
        health_collection=health,
        circuit=_SpyCircuit(health),
        robots=gate,
    )

    slept: list[float] = []

    async def _record_sleep(seconds: float) -> None:
        slept.append(seconds)

    with patch("asyncio.sleep", _record_sleep):
        await tool.execute(url=url, field_schema=schema)

    # The FIRST wait, and its value. `max(slept) > 0` looked like it asserted this and did
    # not: the eval loop's retry backoffs (2s, 4s, 8s...) satisfy it whether or not the crawl
    # delay was waited, so the exemption this test forbids passed straight through it.
    assert decisions and decisions[0].is_probe, (
        f"the tool's circuit did not admit a probe, so this asserts nothing about the rule: {decisions}"
    )
    assert slept, "nothing was waited at all"
    assert slept[0] == pytest.approx(10.0, abs=0.5), (
        f"the first wait was {slept[0]}s, not the ~10s this site asked for: a circuit probe "
        f"skipped the crawl delay. All waits: {slept}"
    )


async def test_a_check_that_never_fetches_does_not_spend_the_sites_fleet_budget() -> None:
    """`TokenBucket.claim` CONSUMES, so asking during `check` charges for a visit that may
    never happen.

    This is the same rule `note_fetched` enforces for the local clock -- the site pays when we
    actually visit it -- and it matters more here, because the token is shared across pods.
    `ScrapeTool` consults robots before the circuit, so a walled target polled inside its
    backoff was draining a token per poll and delaying every SIBLING target on that origin: a
    target that is behaving perfectly, slowed down by one that is not.
    """
    pacer = _FakeDelayPacer(claimed=True)
    gate = RobotsGate(fetch=_fetcher(_ROBOTS_DELAY), delay_pacer=pacer)

    for _ in range(5):
        await gate.check("https://example.gov/a", now=1000.0)

    assert pacer.keys == [], (
        f"checking spent {len(pacer.keys)} fleet token(s) without fetching; a suppressed poll "
        "now delays every other target on that site"
    )

    await gate.claim_fleet_turn("https://example.gov/a")
    assert pacer.keys == ["https://example.gov"], "committing to a fetch does take the turn"


async def test_claiming_a_turn_without_a_pacer_is_a_no_op() -> None:
    """So a caller never has to branch on whether fleet coordination is configured."""
    gate = RobotsGate(fetch=_fetcher(_ROBOTS_DELAY))
    assert await gate.claim_fleet_turn("https://example.gov/a") == 0.0


async def test_an_origin_that_asked_for_nothing_is_not_paced() -> None:
    """The preconditions the local clock applies, applied to the fleet claim too.

    Moving the claim out of `check` dropped them: an origin with a written agreement, one
    serving no robots.txt, or one declaring no `Crawl-delay` -- which is most sites -- was
    paced fleet-wide by a bucket the local clock would never have consulted. Throttling sites
    that asked for nothing is not politeness, and it is invisible because the pacer's own tests
    all use a file that DOES declare a delay.
    """
    no_delay = RobotsGate(fetch=_fetcher("User-agent: *\nDisallow: /nope\n"), delay_pacer=_FakeDelayPacer())
    assert await no_delay.claim_fleet_turn("https://example.gov/a") == 0.0
    assert no_delay._delay_pacer.keys == []  # noqa: SLF001

    unreachable = RobotsGate(fetch=_fetcher("", status=500), delay_pacer=_FakeDelayPacer())
    assert await unreachable.claim_fleet_turn("https://example.gov/a") == 0.0
    assert unreachable._delay_pacer.keys == []  # noqa: SLF001

    agreed = RobotsGate(
        RobotsPolicy(overrides=frozenset({"https://example.gov"})),
        fetch=_fetcher(_ROBOTS_DELAY),
        delay_pacer=_FakeDelayPacer(),
    )
    assert await agreed.claim_fleet_turn("https://example.gov/a") == 0.0
    assert agreed._delay_pacer.keys == [], "an origin we have an agreement with was still throttled"  # noqa: SLF001


async def test_a_fleet_wait_is_capped_like_a_declared_delay() -> None:
    """`retry_after_seconds` is the bucket's own number and otherwise unbounded.

    `max_wait_seconds` sizes `ScrapeTool`'s advertised deadline, so an uncapped fleet wait puts
    the call back outside the very budget the tool was taught to declare -- the deadline then
    manufactures the cancellation, which is the bug that stranded a probe.
    """
    gate = RobotsGate(
        RobotsPolicy(max_crawl_delay_seconds=30.0),
        fetch=_fetcher(_ROBOTS_DELAY),
        delay_pacer=_FakeDelayPacer(claimed=False, retry_after_seconds=9999.0),
    )

    assert await gate.claim_fleet_turn("https://example.gov/a") == pytest.approx(30.0)
    assert gate.max_wait_seconds == pytest.approx(30.0), "the declared ceiling still bounds it"


async def test_a_malformed_delay_is_reported_once_per_poll_not_twice(caplog) -> None:
    """The other `announce` branch, reached the only way it can be.

    `urllib.robotparser` validates `Crawl-delay` itself and returns None for anything
    non-integer, so no real file reaches this branch -- it guards the parser `_parser_for`
    hands over, and a future stdlib that returns the raw token. A stub parser is therefore not
    a shortcut here; it is the only caller that exists.
    """

    class _RawDelayParser:
        """Returns what a stricter stdlib would have filtered."""

        def crawl_delay(self, _agent: str) -> str:
            return "soon"

        def can_fetch(self, _agent: str, _url: str) -> bool:
            return True

    gate = RobotsGate(fetch=_fetcher(_ROBOTS_DELAY))

    with caplog.at_level("INFO", logger="threetears.scrape.robots"):
        first = gate._capped_delay("https://example.gov", _RawDelayParser())  # noqa: SLF001
        second = gate._capped_delay("https://example.gov", _RawDelayParser(), announce=False)  # noqa: SLF001

    assert first is None and second is None, "an unparseable delay is ignored, not honoured"
    said = [r for r in caplog.records if "unparseable crawl delay" in r.getMessage()]
    assert len(said) == 1, f"announced {len(said)} times; the second caller must stay quiet"


async def test_a_capped_delay_is_reported_once_per_poll_not_twice(caplog) -> None:
    """`_capped_delay` runs twice per fetch now -- once for the local wait, once to decide
    whether the origin is paced at all -- and both of its branches log.

    A note that stutters is a note people learn to skim, and the duplication says nothing new.
    """
    gate = RobotsGate(
        RobotsPolicy(max_crawl_delay_seconds=5.0),
        fetch=_fetcher("User-agent: *\nCrawl-delay: 900\n"),
        delay_pacer=_FakeDelayPacer(),
    )

    with caplog.at_level("INFO", logger="threetears.scrape.robots"):
        await gate.check("https://example.gov/a")
        await gate.claim_fleet_turn("https://example.gov/a")

    capped = [r for r in caplog.records if "capped at" in r.getMessage()]
    assert len(capped) == 1, f"the cap was announced {len(capped)} times for one poll"
