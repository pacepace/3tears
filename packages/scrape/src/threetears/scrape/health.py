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

This module writes only :attr:`ScrapeTargetHealth.content_fingerprint`, on a validated
fetch. The remaining columns are declared here, and their table created in one migration,
because the shape is already designed and a single DDL beats three ALTERs against the same
young table; the code that populates them lands with the failure-classification and
backoff work that needs them.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from threetears.core.entities.base import BaseEntity
from threetears.observe import get_logger

from .collections import ScrapeCollection, _parse_dt
from .extraction import html_to_text

__all__ = [
    "ScrapeTargetHealth",
    "ScrapeTargetHealthCollection",
    "content_fingerprint",
    "record_validated_fetch",
]

log = get_logger(__name__)


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
        """
        return int(self._get_raw("consecutive_fetch_failures", 0))

    @property
    def circuit_state(self) -> str:
        """``"closed"`` | ``"open"`` | ``"half_open"``, defaulting to ``"closed"``.

        The three-state vocabulary of ``threetears.models.circuit_breaker.CircuitState``,
        stored durably here rather than held in that class's own process-local instance,
        because a target blocked on one pod is blocked on all of them.
        """
        return str(self._get_raw("circuit_state", "closed"))

    @property
    def blocked_until(self) -> datetime | None:
        """When the next fetch attempt is permitted; ``None`` means no backoff is in force."""
        return _parse_dt(self._get_raw("blocked_until"))

    @property
    def last_blocked_at(self) -> datetime | None:
        """When this target was last observed to be behind a wall."""
        return _parse_dt(self._get_raw("last_blocked_at"))

    @property
    def last_block_kind(self) -> str | None:
        """What kind of wall was last observed, as evidence for an operator; ``None`` if never."""
        result: str | None = self._get_raw("last_block_kind", None)
        return result

    @property
    def session_state_sealed(self) -> str | None:
        """A human-cleared browser session's cookies and storage, sealed at rest; ``None`` if none.

        Ciphertext only. Sealed via ``threetears.core.security.encryption`` under an
        operator-supplied master key, because these are live session credentials: never
        stored in the clear, never logged, never included in a debug dump.
        """
        result: str | None = self._get_raw("session_state_sealed", None)
        return result

    @property
    def session_state_expires_at(self) -> datetime | None:
        """When the sealed session state stops being trusted.

        Treated as advisory: past this point the state is ignored and a human is needed
        again, which degrades to "ask for help", never to bad data.
        """
        return _parse_dt(self._get_raw("session_state_expires_at"))


class ScrapeTargetHealthCollection(ScrapeCollection[ScrapeTargetHealth]):
    """Collection of per-target fetch health, keyed by ``target_id``."""

    primary_key_column = "target_id"

    @property
    def table_name(self) -> str:
        """Return the L3 table name for this collection."""
        return "scrape_target_health"

    @property
    def entity_class(self) -> type[ScrapeTargetHealth]:
        """Return the entity type this collection manages."""
        return ScrapeTargetHealth


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

    Merges onto whatever health the target already has rather than replacing it: the
    failure and circuit columns describe a different concern and must survive a success.

    :param health_collection: this target's health store
    :ptype health_collection: ScrapeTargetHealthCollection
    :param target_id: the target that just validated
    :ptype target_id: str
    :param html: the page it validated against
    :ptype html: str
    :return: the persisted health row
    :rtype: ScrapeTargetHealth
    """
    existing = await health_collection.get(target_id)
    data: dict[str, Any] = dict(existing.to_dict()) if existing is not None else {"target_id": target_id}
    data["content_fingerprint"] = content_fingerprint(html)
    data["fingerprint_updated_at"] = datetime.now(UTC)

    if existing is None:
        entity = health_collection.create(data)
    else:
        # Built as NOT new, rather than through create(): BaseCollection.save_entity stamps
        # date_created only for a new entity and fences the write with the entity's own
        # original_date_updated, so re-creating an existing row would both reset its
        # creation time on every successful fetch and skip the compare-and-swap that keeps
        # concurrent pods from clobbering each other.
        entity = ScrapeTargetHealth(data, is_new=False, collection=health_collection)
    await health_collection.save_entity(entity)
    return entity
