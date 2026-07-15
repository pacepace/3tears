"""Pluggable sources for scrape target configuration.

The eval loop / driver / plugin layers only ever consume ``ScrapeTarget``
objects (each carrying its own ``field_schema`` -- see ``collections.py``);
they never care where those objects came from. This module is the seam that
keeps the core deliberately unopinionated about that: a Python literal, a
YAML file, a database-backed ``ScrapeTargetCollection``, or a single
hand-built ``ScrapeTarget`` for a one-off ad-hoc scrape are all just
``TargetSource`` implementations, chosen by the caller.

Zero faidh imports (see ``scrape/__init__.py``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml
from threetears.observe import get_logger

from .collections import ScrapeTarget, ScrapeTargetCollection

__all__ = [
    "CollectionTargetSource",
    "StaticTargetSource",
    "TargetSource",
    "YamlTargetSource",
    "bootstrap_targets",
    "read_yaml_targets",
]

log = get_logger(__name__)


class TargetSource(ABC):
    """Read-only: returns the targets a source currently knows about.

    Never mutates anything -- seeding a database-backed source from another
    source is a distinct operation (:func:`bootstrap_targets`), not part of
    this interface, so a plain read never has a surprising write side effect.
    """

    @abstractmethod
    async def load(self) -> dict[str, ScrapeTarget]:
        """Return target_id -> ``ScrapeTarget`` for every target this source knows about."""
        ...


class StaticTargetSource(TargetSource):
    """Wraps an in-memory ``dict[str, ScrapeTarget]``.

    Covers two cases at once: targets defined as a Python literal (the
    original ``WARN_ACT_TARGETS``-style shape), and a genuinely ad-hoc
    one-off scrape (construct a single ``ScrapeTarget`` inline, wrap it in
    a one-entry ``StaticTargetSource``, no file or database involved at all).
    """

    def __init__(self, targets: dict[str, ScrapeTarget]) -> None:
        """
        :param targets: target_id -> ``ScrapeTarget``, held by reference (not copied on init)
        :ptype targets: dict[str, ScrapeTarget]
        """
        self._targets = targets

    async def load(self) -> dict[str, ScrapeTarget]:
        """Return a shallow copy of the wrapped dict."""
        return dict(self._targets)


class YamlTargetSource(TargetSource):
    """Reads target definitions from a YAML file.

    Expected shape -- a mapping of target_id to the same fields
    ``ScrapeTarget`` exposes (``url``, ``driver_backend``, ``rate_limit_key``,
    ``cadence``, ``multi_row``, ``wait_for``, ``field_schema``)::

        warn_act_md:
          url: "https://www.dllr.state.md.us/employment/warn.shtml"
          driver_backend: nodriver
          rate_limit_key: warn_act_state_sites
          cadence: "86400"
          multi_row: true
          field_schema:
            employer: str
            notice_date: str
            affected_count: int

    ``field_schema`` values are the plain type-name strings ``ScrapeTarget``
    already stores internally (see ``collections.decode_field_schema``) --
    YAML has no way to express a live Python ``type`` object, so this is the
    natural on-disk shape, not an extra encoding step.
    """

    def __init__(self, path: str | Path) -> None:
        """
        :param path: path to the YAML file
        :ptype path: str | Path
        """
        self._path = Path(path)

    async def load(self) -> dict[str, ScrapeTarget]:
        """Parse the YAML file and return target_id -> ``ScrapeTarget``.

        :raises FileNotFoundError: if the configured path does not exist
        :raises yaml.YAMLError: if the file is not valid YAML
        """
        return read_yaml_targets(self._path)


def read_yaml_targets(path: str | Path) -> dict[str, ScrapeTarget]:
    """Synchronous YAML parse -- plain local file I/O, no genuine async need.

    :func:`YamlTargetSource.load` wraps this for the async ``TargetSource``
    interface; callers that need a target dict at import time or in a plain
    sync context (a module-level constant, a script) can call this directly
    instead of spinning up an event loop for what is just reading a file.

    :param path: path to the YAML file
    :ptype path: str | Path
    :return: target_id -> ``ScrapeTarget``
    :rtype: dict[str, ScrapeTarget]
    :raises FileNotFoundError: if *path* does not exist
    :raises yaml.YAMLError: if the file is not valid YAML
    """
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    targets: dict[str, ScrapeTarget] = {}
    for target_id, fields in raw.items():
        data = dict(fields)
        data["target_id"] = target_id
        targets[target_id] = ScrapeTarget(data)
    return targets


class CollectionTargetSource(TargetSource):
    """Reads target definitions from a database-backed ``ScrapeTargetCollection``.

    Works identically whether the collection's L3 is a real asyncpg pool or
    the in-memory fallback (Chunk 08's ``ScrapeCollection`` abstracts that
    difference away already) -- this source never needs to know which.
    """

    def __init__(self, collection: ScrapeTargetCollection) -> None:
        """
        :param collection: the collection to read from
        :ptype collection: ScrapeTargetCollection
        """
        self._collection = collection

    async def load(self) -> dict[str, ScrapeTarget]:
        """Return every target currently stored in the collection."""
        entities = await self._collection.list_all()
        return {entity.target_id: entity for entity in entities}


async def bootstrap_targets(source: TargetSource, collection: ScrapeTargetCollection) -> int:
    """Seed *collection* from *source* for every target not already present.

    Never overwrites an existing row in *collection* -- a target added or
    edited directly through the database (not via the seed source) is left
    alone, so this is safe to call unconditionally on every startup, not
    just once on first deploy. This is the "recreate what's in the database"
    half of the design: a git-tracked YAML source stays the durable,
    reviewable record even once the database is the thing actually queried
    at runtime.

    :param source: where the seed data comes from (typically a :class:`YamlTargetSource`)
    :ptype source: TargetSource
    :param collection: the database-backed collection to seed
    :ptype collection: ScrapeTargetCollection
    :return: number of targets newly written (0 if *collection* already had every one)
    :rtype: int
    """
    seed_targets = await source.load()
    seeded = 0
    for target_id, target in seed_targets.items():
        existing = await collection.get(target_id)
        if existing is not None:
            continue
        entity = collection.create(target.to_dict())
        await entity.save()
        seeded += 1
    if seeded:
        log.info("scrape: bootstrapped %d target(s) into %s from seed source", seeded, collection.table_name)
    return seeded
