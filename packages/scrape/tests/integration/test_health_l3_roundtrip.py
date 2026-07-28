"""Real-Postgres round-trip for ``scrape_target_health``, through the real migrations.

This package's first integration test, and it exists because of a specific failure.
``ScrapeTarget.link_selector`` shipped as a persisted field with no DDL column and every
test in the package stayed green, because ``ScrapeCollection`` falls back to an in-memory
dict when no L3 pool is configured, and a dict has no schema to violate. The bug only
appeared against a real database, which nothing here ever touched.

``test_migrations_drift.py`` closes that gap offline by reading column names back out of
the migration SQL, which is fast and runs everywhere. It is still a check against a
*parse* of the DDL rather than against a database that actually accepted it. This suite
is the other half: apply the real migrations to a real Postgres, then write and read a
real row through the same collection production uses.

Guarded by ``@pytest.mark.integration``. The full sweep (``./scripts/test.sh`` with no
package) passes ``-m "not integration"`` and deselects it outright; ``./scripts/test.sh
scrape`` does collect it, and it then skips on the ``db_container`` fixture when docker is
absent. Either way nobody needs docker to run the package's tests, and neither path is
what proves this suite ran -- select it explicitly with ``-m integration``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import asyncpg
import pytest
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.scrape.health import (
    ScrapeTargetHealthCollection,
    clear_robots_block,
    content_fingerprint,
    record_circuit_state,
    record_classification,
    record_robots_block,
    record_validated_fetch,
)
from threetears.scrape.migrations import apply_migrations

pytestmark = pytest.mark.integration


@pytest.fixture
async def pg_pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    """A plain asyncpg pool with every 3tears-scrape migration applied.

    Deliberately the real ``apply_migrations`` rather than hand-written DDL: the thing
    under test is whether the migrations this package ships actually provision what its
    entities read, so writing the schema by hand here would test a copy of the answer.
    """
    pool: asyncpg.Pool = await asyncpg.create_pool(db_container, min_size=1, max_size=4)
    try:
        await apply_migrations(pool)
        yield pool
    finally:
        await pool.close()


@pytest.fixture
def health(pg_pool: asyncpg.Pool) -> ScrapeTargetHealthCollection:
    """L3 only, no L1 backend wired.

    Deliberate: an L1 SQLite backend has to be initialized per table (normally by
    ``DataStore.create_table``, which a collection built directly never calls), and
    caching is not what this suite is testing. With no L1, every read goes to the real
    Postgres, which is precisely the path that needs proving.
    """
    registry = CollectionRegistry()
    registry.configure(l3_pool=pg_pool)
    return ScrapeTargetHealthCollection(registry, DefaultCoreConfig(collection_flush="ALWAYS"), nats_client=None)


def _target(name: str) -> str:
    """Unique target id per test: the container is session-scoped and rows persist."""
    return f"{name}_{uuid.uuid4().hex[:8]}"


async def test_every_health_field_round_trips_through_real_postgres(
    health: ScrapeTargetHealthCollection,
) -> None:
    """Write every column, evict L1, read it back from Postgres.

    Writing ALL fields at once is the point rather than a convenience: a single field
    with no column raises ``asyncpg.UndefinedColumnError`` on the upsert, so this fails
    loudly for exactly the bug class that motivated the suite. The cache invalidation
    keeps the read honest even if a caching tier is wired in later.
    """
    target_id = _target("warn_oh")
    blocked_at = datetime(2026, 7, 25, 3, 30, tzinfo=UTC)
    entity = health.create(
        {
            "target_id": target_id,
            "content_fingerprint": "a" * 64,
            "fingerprint_updated_at": blocked_at,
            "consecutive_fetch_failures": 3,
            "circuit_state": "open",
            "blocked_until": blocked_at + timedelta(hours=6),
            "last_blocked_at": blocked_at,
            "last_block_kind": "interstitial",
            "classified_fingerprint": "b" * 64,
            "classified_verdict": "blocked",
            "classified_evidence": "the page asks the visitor to verify a browser",
            "session_state_sealed": "sealed-ciphertext-token",
            "session_state_expires_at": blocked_at + timedelta(days=1),
        }
    )
    await entity.save()

    await health.invalidate_cache(target_id)
    stored = await health.get(target_id)

    assert stored is not None
    assert stored.content_fingerprint == "a" * 64
    assert stored.consecutive_fetch_failures == 3
    assert stored.circuit_state == "open"
    assert stored.last_block_kind == "interstitial"
    assert stored.classified_fingerprint == "b" * 64
    assert stored.classified_verdict == "blocked"
    assert stored.classified_evidence == "the page asks the visitor to verify a browser"
    assert stored.session_state_sealed == "sealed-ciphertext-token"
    assert stored.last_blocked_at == blocked_at
    assert stored.blocked_until == blocked_at + timedelta(hours=6)
    assert stored.session_state_expires_at == blocked_at + timedelta(days=1)


async def test_a_health_row_needs_only_a_target_id(health: ScrapeTargetHealthCollection) -> None:
    """Every column is nullable or defaulted, so a first observation can be minimal.

    The blocked path writes health for a target that may never have succeeded, so it must
    be able to create a row without inventing values for columns it knows nothing about.
    """
    target_id = _target("warn_new")
    entity = health.create({"target_id": target_id})
    await entity.save()

    await health.invalidate_cache(target_id)
    stored = await health.get(target_id)

    assert stored is not None
    assert stored.content_fingerprint is None
    assert stored.consecutive_fetch_failures == 0
    assert stored.circuit_state == "closed"


async def test_record_validated_fetch_works_against_a_real_schema(health: ScrapeTargetHealthCollection) -> None:
    """The production write path, not a hand-built entity, against real Postgres."""
    target_id = _target("warn_md")
    page = "<html><body><table><tr><td>Acme Corp</td></tr></table></body></html>"

    await record_validated_fetch(health, target_id=target_id, html=page)

    await health.invalidate_cache(target_id)
    stored = await health.get(target_id)
    assert stored is not None
    assert stored.content_fingerprint == content_fingerprint(page)
    assert stored.fingerprint_updated_at is not None


async def test_a_second_success_merges_rather_than_replacing(health: ScrapeTargetHealthCollection) -> None:
    """The read-then-update path, against a real schema.

    Distinct from the fresh-row write above: the second call takes the branch that loads
    the existing row and saves it back as an existing entity, which is where a real
    database can disagree with an in-memory dict (column types, the compare-and-swap
    fence, and the creation timestamp that must not be reset by a later success).
    """
    target_id = _target("warn_merge")
    seeded = health.create(
        {
            "target_id": target_id,
            "consecutive_fetch_failures": 2,
            "circuit_state": "half_open",
            "last_block_kind": "interstitial",
        }
    )
    await seeded.save()

    await health.invalidate_cache(target_id)
    before = await health.get(target_id)
    assert before is not None
    created_at = before.date_created

    page = "<html><body><table><tr><td>Acme Corp</td></tr></table></body></html>"
    await record_validated_fetch(health, target_id=target_id, html=page)

    await health.invalidate_cache(target_id)
    stored = await health.get(target_id)
    assert stored is not None
    assert stored.content_fingerprint == content_fingerprint(page)
    # Unrelated health survives the success that proves the target recovered.
    assert stored.consecutive_fetch_failures == 2
    assert stored.circuit_state == "half_open"
    assert stored.last_block_kind == "interstitial"
    # And the row's creation time is not reset by a later fetch.
    assert stored.date_created == created_at


async def test_a_blocked_verdict_and_a_later_success_coexist_on_one_row(
    health: ScrapeTargetHealthCollection,
) -> None:
    """The two writers touch the same row through the same merge, against a real schema.

    Worth its own integration test rather than trusting the in-memory one: these are two
    separate read-modify-writes against a row that already exists, so both take the fenced
    update path where a real database can disagree with a dict. It also pins that the
    verdict cache and the reference fingerprint are genuinely different columns -- if they
    ever collapsed into one, a recovery would erase the block record or a block would
    poison the comparison page, and both would pass an in-memory check that only reads
    back what it wrote.
    """
    target_id = _target("warn_wall")
    page = "<html><body><h1>Checking your browser</h1></body></html>"

    await record_classification(
        health,
        target_id=target_id,
        fingerprint=content_fingerprint(page),
        kind="blocked",
        evidence="the page asks the visitor to verify a browser",
    )

    await health.invalidate_cache(target_id)
    blocked = await health.get(target_id)
    assert blocked is not None
    assert blocked.classified_fingerprint == content_fingerprint(page)
    assert blocked.classified_verdict == "blocked"
    assert blocked.last_blocked_at is not None
    assert blocked.content_fingerprint is None, "a wall must never become the reference page"

    recovered_page = "<html><body><table><tr><td>Acme Corp</td></tr></table></body></html>"
    await record_validated_fetch(health, target_id=target_id, html=recovered_page)

    await health.invalidate_cache(target_id)
    stored = await health.get(target_id)
    assert stored is not None
    assert stored.content_fingerprint == content_fingerprint(recovered_page)
    # The block record survives the fetch that proves the target recovered, which is the
    # moment that history is most worth having.
    assert stored.classified_verdict == "blocked"
    assert stored.last_blocked_at is not None


async def test_the_circuit_writer_round_trips_a_full_trip_and_recovery(
    health: ScrapeTargetHealthCollection,
) -> None:
    """The circuit's own writer against a real schema, not only a hand-built row.

    The columns themselves are already covered above, but they were covered by a row this
    suite assembled. ``record_circuit_state`` is what production actually calls, it writes a
    timestamp and an integer into columns an in-memory dict would accept in any shape, and
    the recovery half writes SQL NULL into a ``TIMESTAMPTZ`` -- the case where "the dict took
    it" says least about whether Postgres will.
    """
    target_id = _target("circuit")
    blocked_at = datetime.now(UTC).replace(microsecond=0)

    await record_circuit_state(
        health,
        target_id=target_id,
        circuit_state="open",
        consecutive_fetch_failures=4,
        blocked_until=blocked_at + timedelta(hours=2),
        blocked_at=blocked_at,
    )
    tripped = await health.get(target_id)
    assert tripped is not None
    assert tripped.circuit_state == "open"
    assert tripped.consecutive_fetch_failures == 4
    assert tripped.blocked_until == blocked_at + timedelta(hours=2)
    assert tripped.last_blocked_at == blocked_at

    await record_circuit_state(
        health,
        target_id=target_id,
        circuit_state="closed",
        consecutive_fetch_failures=0,
        blocked_until=None,
    )
    recovered = await health.get(target_id)
    assert recovered is not None
    assert recovered.circuit_state == "closed"
    assert recovered.consecutive_fetch_failures == 0
    assert recovered.blocked_until is None
    # The block history survives the recovery: a merge that replaced the row would erase
    # the evidence that this target was ever walled, which is the one thing an operator
    # looking at a recovered target wants to see.
    assert recovered.last_blocked_at == blocked_at


async def test_list_walled_finds_only_targets_a_human_can_actually_help(
    health: ScrapeTargetHealthCollection,
) -> None:
    """The query the circuit needed and never had, and the filter that makes it useful.

    Without this a caller with fifty targets and four walls had to fetch all fifty to find
    the four -- which is exactly the cost the circuit exists to avoid, so its absence made the
    circuit argue against itself.

    The filter is the substance. The circuit opens on repeated TRANSPORT failures too, and a
    human sent to a host that stopped answering has nothing to do when they arrive. Only a
    bot-wall verdict stamps ``last_blocked_at``, so that column is the discriminator, and this
    asserts an unreachable target is absent rather than merely that a walled one is present.

    Real Postgres, necessarily: ``list_walled`` returns an empty list without an L3 pool, so
    a unit test cannot reach a single line of its behaviour.
    """
    walled = _target("walled")
    unreachable = _target("unreachable")
    healthy = _target("healthy")
    blocked_at = datetime.now(UTC).replace(microsecond=0)

    # A wall: circuit open AND last_blocked_at stamped.
    await record_circuit_state(
        health,
        target_id=walled,
        circuit_state="open",
        consecutive_fetch_failures=3,
        blocked_until=blocked_at + timedelta(hours=1),
        blocked_at=blocked_at,
    )
    # A host that stopped answering: circuit open, no block stamp. Same suppression, nothing
    # for a person to do.
    await record_circuit_state(
        health,
        target_id=unreachable,
        circuit_state="open",
        consecutive_fetch_failures=3,
        blocked_until=blocked_at + timedelta(hours=1),
    )
    # A target that has only ever worked.
    await record_validated_fetch(health, target_id=healthy, html="<html><body>fine</body></html>")

    found = {row.target_id for row in await health.list_walled()}

    assert walled in found
    assert unreachable not in found, (
        "a target whose host stopped answering was queued for a human, who will arrive with nothing to clear"
    )
    assert healthy not in found


async def test_list_walled_keeps_a_target_whose_backoff_has_elapsed(
    health: ScrapeTargetHealthCollection,
) -> None:
    """An expired window means the next poll will probe, not that the wall is gone.

    Dropping a target from this list when its backoff elapses would make a human queue empty
    itself on a timer, while every target in it is still walled.
    """
    target_id = _target("elapsed")
    long_ago = datetime.now(UTC).replace(microsecond=0) - timedelta(days=2)

    await record_circuit_state(
        health,
        target_id=target_id,
        circuit_state="open",
        consecutive_fetch_failures=5,
        blocked_until=long_ago + timedelta(hours=1),
        blocked_at=long_ago,
    )

    assert target_id in {row.target_id for row in await health.list_walled()}


async def test_a_recovered_target_leaves_the_queue(health: ScrapeTargetHealthCollection) -> None:
    """The other half: a human cleared it, so it must stop being asked about.

    ``last_blocked_at`` deliberately survives a recovery as evidence, so a filter on that
    column alone would keep every target that was EVER walled in the queue forever. The
    circuit state is what makes it current.
    """
    target_id = _target("recovered")
    blocked_at = datetime.now(UTC).replace(microsecond=0)

    await record_circuit_state(
        health,
        target_id=target_id,
        circuit_state="open",
        consecutive_fetch_failures=3,
        blocked_until=blocked_at + timedelta(hours=1),
        blocked_at=blocked_at,
    )
    assert target_id in {row.target_id for row in await health.list_walled()}

    await record_circuit_state(
        health,
        target_id=target_id,
        circuit_state="closed",
        consecutive_fetch_failures=0,
        blocked_until=None,
    )

    found = await health.list_walled()
    assert target_id not in {row.target_id for row in found}
    # And the evidence is still on the row, just not in the queue.
    recovered = await health.get(target_id)
    assert recovered is not None
    assert recovered.last_blocked_at == blocked_at


async def test_list_walled_carries_what_an_operator_needs(health: ScrapeTargetHealthCollection) -> None:
    """A queue item has to be actionable, not just an id.

    Whoever picks this up needs to know what the page said and when it is due a probe;
    otherwise they open a session against a target and discover the reason for themselves.
    """
    target_id = _target("actionable")
    blocked_at = datetime.now(UTC).replace(microsecond=0)

    await record_circuit_state(
        health,
        target_id=target_id,
        circuit_state="open",
        consecutive_fetch_failures=3,
        blocked_until=blocked_at + timedelta(hours=6),
        blocked_at=blocked_at,
    )
    await record_classification(
        health,
        target_id=target_id,
        fingerprint="deadbeef",
        kind="blocked",
        evidence="the page asks the visitor to verify a browser",
    )

    (row,) = [r for r in await health.list_walled() if r.target_id == target_id]
    assert row.classified_evidence == "the page asks the visitor to verify a browser"
    assert row.blocked_until == blocked_at + timedelta(hours=6)
    assert row.consecutive_fetch_failures == 3


async def test_the_egress_column_records_which_exit_an_observation_came_from(
    health: ScrapeTargetHealthCollection,
) -> None:
    """v011, and the reason it is a separate migration rather than another v010 column.

    With more than one exit configured, "this target is walled" stops being the useful fact
    and "walled FROM THIS EXIT" starts being it -- otherwise a target blocked through one route
    looks permanently walled, its circuit backs it off, and a working alternative is never
    tried, the backoff having learned a lesson about the exit rather than the target.

    Against real Postgres because this is an ALTER on a shipped table: v010 is on develop, so
    a database that applied it records version 10 as done and would never pick up an edit to
    it. The thing worth proving is that the migration runner actually adds the column.
    """
    target_id = _target("egress")
    blocked_at = datetime.now(UTC).replace(microsecond=0)

    await record_circuit_state(
        health,
        target_id=target_id,
        circuit_state="open",
        consecutive_fetch_failures=3,
        blocked_until=blocked_at + timedelta(hours=1),
        blocked_at=blocked_at,
        egress="tor",
    )

    row = await health.get(target_id)
    assert row is not None
    assert row.last_egress == "tor"
    # And it survives into the queue, which is where a caller decides whether to try another.
    (queued,) = [r for r in await health.list_walled() if r.target_id == target_id]
    assert queued.last_egress == "tor"


async def test_an_unstamped_row_reports_no_egress_rather_than_direct(
    health: ScrapeTargetHealthCollection,
) -> None:
    """ "We did not record an exit" is a real state, not a gap.

    Every row written before this column existed, and every deployment that configures no
    egress at all. A default of ``'direct'`` would assert something about rows nobody stamped,
    and a query later could not tell the assertion from an observation.
    """
    target_id = _target("noegress")
    await record_circuit_state(
        health,
        target_id=target_id,
        circuit_state="open",
        consecutive_fetch_failures=3,
        blocked_until=datetime.now(UTC),
        blocked_at=datetime.now(UTC),
    )
    row = await health.get(target_id)
    assert row is not None
    assert row.last_egress is None


async def test_a_robots_disallowed_target_reaches_the_human_queue(
    health: ScrapeTargetHealthCollection,
) -> None:
    """The decision has to land on a row, or nobody can be sent to it.

    `list_walled` answers from the health table. A robots block recorded only in a ToolResult
    is visible to whichever caller happened to run and to nobody else, so a target the scraper
    itself decided needs a human would never be findable by the platform whose job it is to
    send one.

    It carries no circuit state on purpose: a policy decision is not a fetch failure, and
    counting it as one would open the circuit and back off a site that works perfectly.
    """
    target_id = _target("robots")
    await record_robots_block(health, target_id=target_id, reason="example.gov/robots.txt disallows 3tears-scrape")

    row = await health.get(target_id)
    assert row is not None
    assert row.robots_blocked_at is not None
    assert "disallows" in (row.robots_blocked_reason or "")
    assert row.circuit_state == "closed", "a robots block was counted as a fetch failure"
    assert row.consecutive_fetch_failures == 0

    queued = {r.target_id for r in await health.list_walled()}
    assert target_id in queued, "a disallowed target never reached the queue a human works"


async def test_a_robots_block_can_leave_the_queue_again(health: ScrapeTargetHealthCollection) -> None:
    """The escalation has to close, or the queue only ever grows.

    A blocked target entered `list_walled` and nothing could take it out: the circuit's
    clear-down touched only circuit columns. And because the queue is ordered by block time and
    bounded by a limit, a row re-stamped on every poll would climb to the top and stay there,
    pushing genuinely walled targets off the end of a list somebody is working through.
    """
    target_id = _target("robotsclear")
    await record_robots_block(health, target_id=target_id, reason="disallowed")
    assert target_id in {r.target_id for r in await health.list_walled()}

    await clear_robots_block(health, target_id=target_id)

    row = await health.get(target_id)
    assert row is not None
    assert row.robots_blocked_at is None
    assert row.robots_blocked_reason is None
    assert target_id not in {r.target_id for r in await health.list_walled()}


async def test_re_blocking_the_same_target_does_not_refresh_its_position(
    health: ScrapeTargetHealthCollection,
) -> None:
    """Stamped once, not once per poll.

    The queue is time-ordered and limited, so a row refreshed every poll outranks every real
    wall and the list a human works becomes one target repeated.
    """
    target_id = _target("robotsstamp")
    await record_robots_block(health, target_id=target_id, reason="disallowed")
    first = await health.get(target_id)
    assert first is not None
    original = first.robots_blocked_at

    await record_robots_block(health, target_id=target_id, reason="disallowed")

    again = await health.get(target_id)
    assert again is not None
    assert again.robots_blocked_at == original, "an unchanged block was re-stamped and jumped the queue"


async def test_a_human_clearing_a_target_also_clears_its_robots_block(
    health: ScrapeTargetHealthCollection,
) -> None:
    """A person who cleared this target cleared it whichever way it got into the queue."""
    from threetears.scrape.circuit import TargetCircuit

    target_id = _target("bothways")
    await record_robots_block(health, target_id=target_id, reason="disallowed")

    await TargetCircuit(health).record_human_cleared(target_id)

    row = await health.get(target_id)
    assert row is not None
    assert row.robots_blocked_at is None
    assert target_id not in {r.target_id for r in await health.list_walled()}
