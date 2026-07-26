"""Per-target fetch health: what happened when we last tried to fetch a target.

The eval loop already remembers one thing per target -- the extraction strategy that
won (``ScrapeRecipe``). It remembers nothing about the *fetch*: whether the page came
back at all, whether it looked like the page we learned against, or whether we were
handed a bot wall instead of the content. That gap is why three genuinely different
failures are currently indistinguishable and share a single response:

- the site was redesigned, so the stored selectors no longer match
- the stored selectors are simply wrong, against a page that has not changed
- we were blocked, and the "page" is a challenge interstitial

Today all three increment ``ScrapeRecipe.consecutive_validation_failures`` identically,
so a blocked target burns through the failure threshold and spends a full LLM candidate
round learning to extract data from a challenge page, discarding a recipe that was never
broken.

:class:`ScrapeTargetHealth` is the missing memory. It is deliberately a separate entity
from ``ScrapeRecipe`` rather than more columns on it: health exists for targets that have
never had a recipe at all (blocked before they ever extracted successfully), and folding
the two together would mean writing a strategy-less recipe row to hold that state, then
adding a guard so the reuse path never mistakes that empty strategy for a real one. A
separate row makes the situation simply not arise -- health with no recipe is a health row
and no recipe row, which is an honest description of what is true.

This module writes the fingerprint of a validated page, the cached verdict about a page that
failed, and where the target's fetch circuit stands. The sealed-session columns are declared
here, and their table created in one migration, because the shape is already designed and a
single DDL beats several ALTERs against the same young table; the code that populates them
lands with the human-in-the-loop work that needs them.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any, ClassVar

from threetears.core.entities.base import BaseEntity

from .collections import ScrapeCollection, _parse_dt
from .extraction import html_to_text

__all__ = [
    "ScrapeTargetHealth",
    "ScrapeTargetHealthCollection",
    "content_fingerprint",
    "record_circuit_state",
    "record_classification",
    "record_robots_block",
    "record_validated_fetch",
]


def content_fingerprint(html: str) -> str:
    """A stable digest of *html*'s readable text, for "is this the same page as last time".

    Fingerprints the page's TEXT, not its markup: a template that re-orders its own
    attributes, re-indents, or changes a wrapper class has not changed what the page
    says, and a fingerprint that flipped on those would report "the site changed" on
    every deploy the site makes. Whitespace is collapsed for the same reason.

    Not a security primitive and not a cache key -- sha256 is used because it is the
    obvious durable digest, not because collision resistance is load-bearing here. The
    only question ever asked of it is equality against the previous value.

    :param html: the fetched page, exactly as the driver returned it
    :ptype html: str
    :return: hex sha256 of the page's whitespace-collapsed readable text
    :rtype: str
    """
    normalized = " ".join(html_to_text(html).split())
    return hashlib.sha256(normalized.encode()).hexdigest()


class ScrapeTargetHealth(BaseEntity):
    """One row per target: what happened when we last tried to fetch it.

    Every field defaults to its never-observed value, so a target that has only ever
    succeeded reads as healthy without anything having to write a row's worth of zeroes
    on its behalf.
    """

    primary_key_field: str = "target_id"

    @property
    def target_id(self) -> str:
        """The target this health record belongs to."""
        return str(self._get_raw("target_id", ""))

    @property
    def content_fingerprint(self) -> str | None:
        """Digest of the page as it looked on the last validated fetch; ``None`` until one happens.

        The comparison value for telling a redesigned page apart from an unchanged one
        when extraction later fails. See :func:`content_fingerprint`.
        """
        result: str | None = self._get_raw("content_fingerprint", None)
        return result

    @property
    def fingerprint_updated_at(self) -> datetime | None:
        """When :attr:`content_fingerprint` was last stamped."""
        return _parse_dt(self._get_raw("fingerprint_updated_at"))

    @property
    def consecutive_fetch_failures(self) -> int:
        """Fetch-stage failures in a row: blocked, transport, timeout.

        Deliberately distinct from ``ScrapeRecipe.consecutive_validation_failures``, which
        counts a different thing (the stored strategy not matching a page we did receive).
        Conflating them is the bug this entity exists to make impossible.

        Written by :mod:`threetears.scrape.circuit`: raised on a blocked or unreachable
        fetch, reset to zero the moment a fetch reaches real content again. It is the input
        to the circuit's failure threshold.

        "Consecutive" describes what this column counts on its own, and is exact for a single
        pod. Where a fleet-wide ``WindowedCounter`` is injected, a blocked fetch stores the
        greater of this count and the fleet's windowed count, and that window deliberately
        survives a success -- so a target that recovers and is walled again inside the window
        re-trips faster than a first-time block. That is the point of the window rather than a
        leak in it: the per-row count is memoryless by design, and one pod's success is not
        evidence that the other pods' blocks did not happen.
        """
        return int(self._get_raw("consecutive_fetch_failures", 0))

    @property
    def circuit_state(self) -> str:
        """``"closed"`` | ``"open"`` | ``"half_open"``, defaulting to ``"closed"``.

        The three-state vocabulary of ``threetears.models.circuit_breaker.CircuitState``,
        stored durably here rather than held in that class's own process-local instance,
        because a target blocked on one pod is blocked on all of them. The transitions
        between the three are still that class's, driven through its ``restore()`` seam by
        :mod:`threetears.scrape.circuit`; only the storage lives here.
        """
        return str(self._get_raw("circuit_state", "closed"))

    @property
    def blocked_until(self) -> datetime | None:
        """When the next fetch attempt is permitted; ``None`` means no backoff is in force.

        This gates the FETCH, which is what bounds a walled target's cost. A target inside
        this window is not fetched, so it reaches neither candidate generation nor the page
        classifier -- the classifier's own verdict cache cannot bound it, because a real
        interstitial renders a per-request id into the very text the cache keys on.
        """
        return _parse_dt(self._get_raw("blocked_until"))

    @property
    def last_blocked_at(self) -> datetime | None:
        """When this target was last observed to be behind a wall."""
        return _parse_dt(self._get_raw("last_blocked_at"))

    @property
    def last_egress(self) -> str | None:
        """Which exit the last recorded observation left by; ``None`` when none was recorded.

        With more than one exit configured, this is what separates "this target is walled"
        from "this target is walled FROM THIS EXIT". Without it a target blocked through one
        route looks permanently walled, its circuit backs it off, and a working alternative is
        never tried -- the backoff learning a lesson about the exit rather than the target.

        ``None`` is a real state, not a gap: every row written before this column existed, and
        every deployment that configures no egress at all. See
        :data:`threetears.core.egress.DirectEgress` for why "direct" is a named exit rather
        than the absence of one -- a caller that stamps it is saying something, and a caller
        that stamps nothing is not.
        """
        result: str | None = self._get_raw("last_egress", None)
        return result

    @property
    def robots_blocked_at(self) -> datetime | None:
        """When ``robots.txt`` last held this target back; ``None`` if it never has.

        Deliberately NOT the circuit's columns. A robots block is a policy decision, not a
        fetch failure: counting it as one would open the circuit, start a backoff and mark a
        working site unhealthy over a rule that says nothing about whether it works.
        """
        return _parse_dt(self._get_raw("robots_blocked_at"))

    @property
    def robots_blocked_reason(self) -> str | None:
        """What the file said, in words an operator reads before deciding."""
        result: str | None = self._get_raw("robots_blocked_reason", None)
        return result

    @property
    def last_block_kind(self) -> str | None:
        """What kind of wall was last observed, as evidence for an operator; ``None`` if never.

        **Nothing writes this yet**, deliberately: the classifier reports THAT a page is a wall,
        not which vendor's wall it is, and filling this with the literal string ``"blocked"``
        would populate a column meant to distinguish walls with a value that never distinguishes
        anything. It waits for a real taxonomy. :attr:`last_blocked_at` IS written on a blocked
        verdict; this is not.
        """
        result: str | None = self._get_raw("last_block_kind", None)
        return result

    @property
    def classified_fingerprint(self) -> str | None:
        """Digest of the last page a classification was asked about; ``None`` until one is.

        Deliberately NOT :attr:`content_fingerprint`, which answers a different question.
        That one is the page as it looked when extraction last *succeeded*, and is the
        reference for "has the site changed". A classification is only ever asked about a
        page that just *failed*, so storing it in the same column would overwrite the only
        reference the comparison has, with a page that is by definition not it.
        """
        result: str | None = self._get_raw("classified_fingerprint", None)
        return result

    @property
    def classified_verdict(self) -> str | None:
        """What the page at :attr:`classified_fingerprint` was judged to be; ``None`` if never.

        One of ``challenge.PageVerdictKind``'s values. Read back when the same page is seen
        again, which is what keeps a target walled for a week costing one classification
        rather than one per poll. It also records that we have already acted on that exact
        page, so a repeated "changed" verdict stops regenerating against a page we have
        demonstrably already failed to learn.
        """
        result: str | None = self._get_raw("classified_verdict", None)
        return result

    @property
    def classified_evidence(self) -> str | None:
        """Why that verdict was reached, in the classifier's own words; ``None`` if never.

        Carried forward onto every extraction the cached verdict produces, so an operator
        looking at a blocked row a week later sees what the page actually said rather than
        just a label.
        """
        result: str | None = self._get_raw("classified_evidence", None)
        return result

    @property
    def session_state_sealed(self) -> str | None:
        """A human-cleared browser session's cookies and storage, sealed at rest; ``None`` if none.

        Ciphertext only. Sealed via ``threetears.core.security.encryption`` under an
        operator-supplied master key, because these are live session credentials: never
        stored in the clear, never logged, never included in a debug dump.

        Written by :func:`threetears.scrape.session_state.record_session_state` when a human
        clears a target in a HITL session and their exported browser state is sealed for reuse.
        Absent means no human has cleared this target, or the stored solve has been cleared.
        """
        result: str | None = self._get_raw("session_state_sealed", None)
        return result

    @property
    def session_state_expires_at(self) -> datetime | None:
        """When the sealed session state stops being trusted.

        Treated as advisory: past this point the state is ignored and a human is needed
        again, which degrades to "ask for help", never to bad data.

        Written together with :attr:`session_state_sealed`, always. A token with no expiry is a
        credential of unknown lifetime, so a missing one here is read as expired rather than as
        eternal -- see :func:`threetears.scrape.session_state.usable_session_state`.
        """
        return _parse_dt(self._get_raw("session_state_expires_at"))


class ScrapeTargetHealthCollection(ScrapeCollection[ScrapeTargetHealth]):
    """Collection of per-target fetch health, keyed by ``target_id``."""

    primary_key_column = "target_id"
    datetime_columns: ClassVar[frozenset[str]] = ScrapeCollection.datetime_columns | {
        "fingerprint_updated_at",
        "blocked_until",
        "last_blocked_at",
        "session_state_expires_at",
        "robots_blocked_at",
    }

    @property
    def table_name(self) -> str:
        """Return the L3 table name for this collection."""
        return "scrape_target_health"

    @property
    def entity_class(self) -> type[ScrapeTargetHealth]:
        """Return the entity type this collection manages."""
        return ScrapeTargetHealth

    async def list_walled(self, *, now: datetime | None = None, limit: int = 200) -> list[ScrapeTargetHealth]:
        """Targets currently suppressed because a human has to clear them.

        The one question about this table that is not a primary-key lookup, and until now the
        only way to answer it was to scrape every target and read the result. A caller with
        fifty targets and four walls had to do fifty fetches to find the four -- which is
        precisely what the circuit exists to avoid, so the absence of this made the circuit
        argue against itself.

        **Two ways a target lands here**, and both need a person: a bot wall the scraper
        cannot pass, and a ``robots.txt`` that disallows us. The second has no circuit state at
        all -- a policy decision is not a fetch failure -- so filtering on the circuit alone
        would answer "who is stuck" while omitting every target the scraper itself decided
        needs a human.

        **Walled, not merely failing.** The filter is ``last_blocked_at IS NOT NULL``, because
        the circuit opens on repeated transport failures too and those are nobody's to clear:
        a human sent to a host that stopped answering has nothing to do when they arrive. Only
        a bot-wall verdict stamps ``last_blocked_at`` (``record_circuit_state`` leaves it alone
        for an unreachable fetch), so it is the discriminator that already exists rather than
        one invented here.

        Rows whose backoff has elapsed are included. An expired window means the next poll
        will probe, not that the wall is gone -- the target is still walled until something
        proves otherwise, and dropping it from this list would make a queue empty itself on a
        timer.

        Served by the partial index ``scrape_target_health_circuit_state``, which was created
        for this query in ``v010`` and has had nothing to serve since.

        :param now: current time; injected by tests, defaults to now. Unused by the predicate
            today and taken anyway, so adding a freshness bound later is not a signature change
        :ptype now: datetime | None
        :param limit: cap on rows returned, newest block first
        :ptype limit: int
        :return: health rows for targets a human needs to look at
        :rtype: list[ScrapeTargetHealth]
        """
        del now
        if self.l3_pool is None:
            # No durable store means the in-memory fallback, which has no query surface at
            # all. Returning empty rather than raising matches every other read in this
            # package: a caller without L3 gets "nothing is walled", which is true of a
            # process that cannot remember anything between restarts anyway.
            return []
        # cache-bypass: a multi-row scan by circuit state is not pk-addressable, so the L1
        # row cache cannot serve it.
        rows = await self.l3_pool.fetch(
            "SELECT target_id, content_fingerprint, fingerprint_updated_at, "
            "consecutive_fetch_failures, circuit_state, blocked_until, last_blocked_at, "
            "last_block_kind, last_egress, robots_blocked_at, robots_blocked_reason, "
            "classified_fingerprint, classified_verdict, classified_evidence, "
            "session_state_sealed, session_state_expires_at, date_created, date_updated "
            "FROM scrape_target_health "
            "WHERE (circuit_state <> 'closed' AND last_blocked_at IS NOT NULL) "
            "   OR robots_blocked_at IS NOT NULL "
            "ORDER BY COALESCE(last_blocked_at, robots_blocked_at) DESC LIMIT $1",
            limit,
        )
        return [ScrapeTargetHealth(dict(row), is_new=False, collection=self) for row in rows]


async def _merge_health(
    health_collection: ScrapeTargetHealthCollection,
    *,
    target_id: str,
    changes: dict[str, Any],
) -> ScrapeTargetHealth:
    """Apply *changes* onto *target_id*'s existing health row, creating one if there is none.

    The one read-modify-write in this module, shared by every writer, because getting it
    wrong is subtle in two ways that a second copy would eventually get wrong differently.

    Merging rather than replacing: each writer owns a few columns and knows nothing about
    the rest, so a write that replaced the row would have a success erase the block history
    that proves the target recovered.

    Building an existing row as NOT new, rather than through ``create()``:
    ``BaseCollection.save_entity`` stamps ``date_created`` only for a new entity and fences
    the write with the entity's own ``original_date_updated``. Re-creating an existing row
    would both reset its creation time on every write and skip the compare-and-swap that
    keeps two pods from clobbering each other.

    :param health_collection: this target's health store
    :ptype health_collection: ScrapeTargetHealthCollection
    :param target_id: the target whose health is being updated
    :ptype target_id: str
    :param changes: column -> new value, applied over whatever is already stored
    :ptype changes: dict[str, Any]
    :return: the persisted health row
    :rtype: ScrapeTargetHealth
    """
    existing = await health_collection.get(target_id)
    data: dict[str, Any] = dict(existing.to_dict()) if existing is not None else {"target_id": target_id}
    data.update(changes)

    entity = (
        health_collection.create(data)
        if existing is None
        else ScrapeTargetHealth(data, is_new=False, collection=health_collection)
    )
    await health_collection.save_entity(entity)
    return entity


async def record_validated_fetch(
    health_collection: ScrapeTargetHealthCollection,
    *,
    target_id: str,
    html: str,
) -> ScrapeTargetHealth:
    """Stamp *target_id*'s fingerprint from a fetch that just validated.

    Called only on success, and only from the eval loop's two entry points, so there is
    exactly one place a fingerprint is born no matter which of the eight reuse or
    regeneration paths produced the extraction. A fingerprint recorded on any other
    outcome would be a fingerprint of a page we did not successfully read, which is the
    opposite of the comparison value this is for.

    :param health_collection: this target's health store
    :ptype health_collection: ScrapeTargetHealthCollection
    :param target_id: the target that just validated
    :ptype target_id: str
    :param html: the page it validated against
    :ptype html: str
    :return: the persisted health row
    :rtype: ScrapeTargetHealth
    """
    return await _merge_health(
        health_collection,
        target_id=target_id,
        changes={
            "content_fingerprint": content_fingerprint(html),
            "fingerprint_updated_at": datetime.now(UTC),
        },
    )


async def record_circuit_state(
    health_collection: ScrapeTargetHealthCollection,
    *,
    target_id: str,
    circuit_state: str,
    consecutive_fetch_failures: int,
    blocked_until: datetime | None,
    blocked_at: datetime | None = None,
    egress: str | None = None,
) -> ScrapeTargetHealth:
    """Persist where *target_id*'s fetch circuit now stands.

    One writer for both directions, because the two are exact mirrors and a pair of them
    would eventually stop being mirrors: the columns a trip writes are the columns a
    recovery has to clear, and a recovery that cleared three of four would leave a target
    reading as closed while still carrying a future ``blocked_until`` that gates it.

    This function decides nothing. What the new state IS comes from
    ``threetears.models.circuit_breaker.CircuitBreaker``, whose transition rules are
    driven and then written here; see :mod:`threetears.scrape.circuit`.

    :param health_collection: this target's health store
    :ptype health_collection: ScrapeTargetHealthCollection
    :param target_id: the target whose circuit moved
    :ptype target_id: str
    :param circuit_state: the new state, a ``CircuitState`` value
    :ptype circuit_state: str
    :param consecutive_fetch_failures: the new consecutive fetch-failure count
    :ptype consecutive_fetch_failures: int
    :param blocked_until: when the next fetch is permitted, or ``None`` to clear the window
    :ptype blocked_until: datetime | None
    :param blocked_at: when this block was observed; omitted leaves the previous value
    :ptype blocked_at: datetime | None
    :param egress: which exit this observation left by, e.g. ``"tor"``; omitted leaves the
        previous value, so a caller with no egress configured never stamps one
    :ptype egress: str | None
    :return: the persisted health row
    :rtype: ScrapeTargetHealth
    """
    changes: dict[str, Any] = {
        "circuit_state": circuit_state,
        "consecutive_fetch_failures": consecutive_fetch_failures,
        "blocked_until": blocked_until,
    }
    if blocked_at is not None:
        changes["last_blocked_at"] = blocked_at
    if egress is not None:
        # Only written when the caller actually knows: an unstamped row means "no exit was
        # recorded", which is different from and more honest than asserting "direct".
        changes["last_egress"] = egress
    return await _merge_health(health_collection, target_id=target_id, changes=changes)


async def record_robots_block(
    health_collection: ScrapeTargetHealthCollection,
    *,
    target_id: str,
    reason: str,
    now: datetime | None = None,
) -> ScrapeTargetHealth:
    """Record that ``robots.txt`` is holding *target_id* back, so a human can be sent to it.

    This is what puts a disallowed target in front of a person. Without it the decision lives
    only in the ToolResult of whichever caller happened to run, and a target the scraper
    itself decided needs a human reaches no queue at all.

    Writes no circuit column. A robots block is not evidence the site is failing, and treating
    it as a fetch failure would back off a target that works perfectly.

    :param health_collection: where the durable state lives
    :ptype health_collection: ScrapeTargetHealthCollection
    :param target_id: the target being held back
    :ptype target_id: str
    :param reason: the file's own words, for the operator
    :ptype reason: str
    :param now: current time; injected by tests
    :ptype now: datetime | None
    :return: the persisted row
    :rtype: ScrapeTargetHealth
    """
    return await _merge_health(
        health_collection,
        target_id=target_id,
        changes={
            "robots_blocked_at": now or datetime.now(UTC),
            "robots_blocked_reason": reason,
        },
    )


async def record_classification(
    health_collection: ScrapeTargetHealthCollection,
    *,
    target_id: str,
    fingerprint: str,
    kind: str,
    evidence: str,
) -> ScrapeTargetHealth:
    """Remember that the page digesting to *fingerprint* was judged to be *kind*.

    Two jobs in one write. It is the verdict cache, so seeing the same page again costs a
    row read instead of a model call. It is also the record that we have already ACTED on
    that page, which is what stops a ``"changed"`` verdict regenerating on every poll after
    a regeneration that did not stick.

    A ``"blocked"`` verdict additionally stamps :attr:`ScrapeTargetHealth.last_blocked_at`.
    It deliberately leaves :attr:`ScrapeTargetHealth.last_block_kind` alone: that column
    means which KIND of wall was seen, and this classifier deliberately has no vendor
    taxonomy to put in it -- writing the literal string ``"blocked"`` there would fill a
    column meant to distinguish walls with a value that never distinguishes anything. The
    classifier's own account of the page goes to ``classified_evidence`` instead, where it
    is read back.

    Nothing here touches ``consecutive_fetch_failures`` or ``circuit_state``: those are a
    counter and a state machine that need a recovery rule as much as a failure rule, and
    half of that pair landing here would leave a counter that only ever climbs.

    :param health_collection: this target's health store
    :ptype health_collection: ScrapeTargetHealthCollection
    :param target_id: the target whose page was classified
    :ptype target_id: str
    :param fingerprint: digest of the page that was classified, from :func:`content_fingerprint`
    :ptype fingerprint: str
    :param kind: the verdict, one of ``challenge.PageVerdictKind``'s values
    :ptype kind: str
    :param evidence: why that verdict was reached, in the classifier's own words
    :ptype evidence: str
    :return: the persisted health row
    :rtype: ScrapeTargetHealth
    """
    changes: dict[str, Any] = {
        "classified_fingerprint": fingerprint,
        "classified_verdict": kind,
        "classified_evidence": evidence,
    }
    if kind == "blocked":
        changes["last_blocked_at"] = datetime.now(UTC)
    return await _merge_health(health_collection, target_id=target_id, changes=changes)
