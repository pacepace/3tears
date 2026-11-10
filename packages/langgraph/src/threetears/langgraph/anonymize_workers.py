"""checkpoint anonymization's CPU half in worker processes, so a person's erasure never stalls the event loop.

**Why.** Anonymizing a stored checkpoint is CPU work: decode the blob with the saver's serializer (which
revives every message through langchain and pydantic constructors), apply the rule, encode it again. A thread
is no way out -- it holds the interpreter lock and the loop waits behind it -- so the work runs in worker
processes, and the loop does only the database I/O around it
(:meth:`~threetears.langgraph.ThreeTierCheckpointSaver.aanonymize_threads`).

**Plain data across.** A job is handed plain data and hands plain data back, never a decoded object: per blob
its serialization type tag, its stored bytes and whether it is a metadata-channel write; and the turn-metadata
classification the CALLING process holds (:func:`~threetears.langgraph.anonymize.metadata_key_classification`,
two sets of key names), applied explicitly in the worker, since a declaration is visible only in the process
that makes it. It answers, per blob, the new bytes or None (unchanged), the type name of a serializer failure,
and the type tag a re-encoding would have changed to; and the turn-metadata keys no classification names. The
serializer crosses by name (its class, which pickles as a reference) and is made once per worker.

**One shared pool.** Every saver in the process shares one pool of :data:`ANONYMIZE_WORKERS` spawned processes
(spawned, never forked: a serving process runs threads, whose locks a fork copies mid-hold), made at the first
anonymization and kept: an erasure walks one saver per agent, and a pool per saver would spawn per agent. Each
worker starts below the serving process's priority. A worker that dies fails the jobs it held
(``BrokenProcessPool``, which fails that erasure; rerunning it is the remedy) and the pool is replaced at the next
job. :func:`close_anonymize_workers` shuts it down; at interpreter exit ``concurrent.futures`` joins it anyway.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import threading
from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Final

from threetears.langgraph.anonymize import (
    MetadataKeyClassification,
    anonymize_checkpoint_value,
    unclassified_metadata_keys,
)
from threetears.observe import get_logger

__all__ = [
    "ANONYMIZE_WORKERS",
    "BlobOutcome",
    "StoredBlob",
    "anonymize_blobs",
    "anonymize_in_workers",
    "close_anonymize_workers",
]

log = get_logger(__name__)

#: the worker processes, side by side. An erasure is rare and never latency-bound; two halve a large thread's
#: wall time without holding more than two interpreters' memory once the first erasure has run.
ANONYMIZE_WORKERS: Final = 2

#: one stored blob, as a job is handed it: its serde type tag, its bytes, and whether it is a write to the
#: metadata channel (so its whole value is turn metadata)
type StoredBlob = tuple[str, bytes, bool]

#: one blob's outcome: its new bytes (None when unchanged or not rewritable), the type name of the serializer
#: failure that stopped it (empty when none), and the type tag its re-encoding would have changed to (empty when
#: it would not)
type BlobOutcome = tuple[bytes | None, str, str]

#: each worker's serializers, made once per class
_serdes: dict[type[Any], Any] = {}

#: the pool, and whether a dead worker broke it; changed under the lock, on a thread, never the loop
_lock = threading.Lock()
_pool: ProcessPoolExecutor | None = None
_broken = False


def anonymize_blobs(
    serde_type: type[Any], identifying: frozenset[str], kept: frozenset[str], blobs: Sequence[StoredBlob]
) -> tuple[list[BlobOutcome], frozenset[str]]:
    """decode, anonymize and re-encode stored blobs: the job a worker runs, also callable in-process.

    only the serializer is guarded. a serializer's protocol names no errors: the default raises ``ValueError``,
    ``TypeError`` or ``NotImplementedError`` on bytes it cannot handle, and revives stored types through
    langchain and pydantic constructors that raise what they raise. a failure there is one blob that fails the
    same way on every run, so its type is answered and the rest go on. the rule runs outside that guard: a bug in
    it is not an unreadable blob, and it raises.

    :param serde_type: the saver's serializer class, made here with no arguments
    :ptype serde_type: type[Any]
    :param identifying: the turn-metadata keys whose values are anonymized, as the caller classifies them
    :ptype identifying: frozenset[str]
    :param kept: the keys ruled not to identify a person, as the caller classifies them
    :ptype kept: frozenset[str]
    :param blobs: the stored blobs
    :ptype blobs: Sequence[StoredBlob]
    :return: each blob's outcome, in order, and every turn-metadata key the classification does not name
    :rtype: tuple[list[BlobOutcome], frozenset[str]]
    """
    serde = _serdes.get(serde_type)
    if serde is None:
        serde = _serdes[serde_type] = serde_type()
    classification = MetadataKeyClassification(identifying=identifying, kept=kept)
    outcomes: list[BlobOutcome] = []
    unclassified: set[str] = set()
    for type_tag, blob, is_metadata in blobs:
        try:
            decoded = serde.loads_typed((type_tag, blob))
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- the serde protocol has no error contract; answered as this blob's failure and reported by the caller, never silent
            outcomes.append((None, type(exc).__name__, ""))
            continue
        unclassified |= unclassified_metadata_keys(decoded, is_metadata=is_metadata, classification=classification)
        rewritten = anonymize_checkpoint_value(decoded, classification=classification)
        if rewritten is decoded:
            outcomes.append((None, "", ""))
            continue
        try:
            new_type, new_blob = serde.dumps_typed(rewritten)
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- the serde protocol has no error contract; answered as this blob's failure and reported by the caller, never silent
            outcomes.append((None, type(exc).__name__, ""))
            continue
        # rewriting it under another tag would leave a row the graph reads with the wrong decoder
        outcomes.append((None, ValueError.__name__, new_type) if new_type != type_tag else (new_blob, "", ""))
    return outcomes, frozenset(unclassified)


def _worker_started() -> None:
    """a worker's start: scheduled below the serving process, so it stays responsive.

    :return: nothing
    :rtype: None
    """
    os.nice(5)


def _submitted(
    serde_type: type[Any], classification: MetadataKeyClassification, chunks: list[list[StoredBlob]]
) -> list[Future[tuple[list[BlobOutcome], frozenset[str]]]]:
    """submit each chunk to the pool, made (or replaced, after a worker died) first: on a thread, since a fresh
    pool spawns its workers on submission.

    :param serde_type: the serializer class
    :ptype serde_type: type[Any]
    :param classification: the caller's classification
    :ptype classification: MetadataKeyClassification
    :param chunks: the blobs, a job each
    :ptype chunks: list[list[StoredBlob]]
    :return: each job's future
    :rtype: list[Future[tuple[list[BlobOutcome], frozenset[str]]]]
    """
    global _pool, _broken
    with _lock:
        if _pool is not None and _broken:
            _pool.shutdown(wait=False)
            _pool = None
        if _pool is None:
            _pool = ProcessPoolExecutor(
                max_workers=ANONYMIZE_WORKERS,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_worker_started,
            )
            _broken = False
        pool = _pool
        return [
            pool.submit(anonymize_blobs, serde_type, classification.identifying, classification.kept, chunk)
            for chunk in chunks
        ]


async def anonymize_in_workers(
    serde_type: type[Any], classification: MetadataKeyClassification, blobs: Sequence[StoredBlob]
) -> tuple[list[BlobOutcome], frozenset[str]]:
    """:func:`anonymize_blobs` over ``blobs`` in the worker processes, split among them; the loop only waits.

    :param serde_type: the saver's serializer class: importable by name and made with no arguments
    :ptype serde_type: type[Any]
    :param classification: the classification the calling process holds
    :ptype classification: MetadataKeyClassification
    :param blobs: the stored blobs
    :ptype blobs: Sequence[StoredBlob]
    :return: each blob's outcome, in order, and every turn-metadata key the classification does not name
    :rtype: tuple[list[BlobOutcome], frozenset[str]]
    :raises BrokenProcessPool: when a worker died (the pool is replaced at the next job)
    :raises Exception: what the rule raised in the worker
    """
    if not blobs:
        return [], frozenset()
    size = -(-len(blobs) // ANONYMIZE_WORKERS)
    chunks = [list(blobs[start : start + size]) for start in range(0, len(blobs), size)]
    futures = await asyncio.to_thread(_submitted, serde_type, classification, chunks)
    try:
        answers = await asyncio.gather(*(asyncio.wrap_future(future) for future in futures))
    except BrokenProcessPool:
        _mark_broken()
        raise
    outcomes: list[BlobOutcome] = []
    unclassified: set[str] = set()
    for chunk_outcomes, chunk_unclassified in answers:
        outcomes.extend(chunk_outcomes)
        unclassified |= chunk_unclassified
    return outcomes, frozenset(unclassified)


def _mark_broken() -> None:
    """note that a worker died, so the next job replaces the pool.

    :return: nothing
    :rtype: None
    """
    global _broken
    if not _broken:
        log.warning("checkpoint anonymization worker died; its pool is replaced at the next job")
    _broken = True


async def close_anonymize_workers() -> None:
    """shut the worker processes down, waiting for them; the next anonymization starts a fresh pool.

    :return: nothing
    :rtype: None
    """
    await asyncio.to_thread(_close)


def _close() -> None:
    """:func:`close_anonymize_workers`, on a thread.

    :return: nothing
    :rtype: None
    """
    global _pool
    with _lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=True)
