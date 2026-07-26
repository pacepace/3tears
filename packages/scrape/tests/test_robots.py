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
