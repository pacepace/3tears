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

**One shared pool**, through :class:`threetears.core.process_workers.ProcessWorkers` (spawned, made at the first
anonymization, plain data checked both ways, a dead worker replaced): every saver in the process shares its
:data:`ANONYMIZE_WORKERS` processes, since an erasure walks one saver per agent and a pool per saver would spawn
per agent. A worker that dies fails that erasure (``BrokenProcessPool``; rerunning it is the remedy).
:func:`close_anonymize_workers` shuts it down; at interpreter exit ``concurrent.futures`` joins it anyway.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from threetears.core.process_workers import ProcessWorkers
from threetears.observe import BuildOnce, contain, get_logger
from threetears.langgraph.anonymize import (
    MetadataKeyClassification,
    anonymize_checkpoint_value,
    unclassified_metadata_keys,
)

__all__ = ["BlobOutcome", "StoredBlob", "anonymize_in_workers"]

#: the worker processes, side by side. An erasure is rare and never latency-bound; two halve a large thread's
#: wall time without holding more than two interpreters' memory (about 105 MB each) once the first erasure has run.
ANONYMIZE_WORKERS: Final = 2

#: one stored blob, as a job is handed it: its serde type tag, its bytes, and whether it is a write to the
#: metadata channel (so its whole value is turn metadata)
type StoredBlob = tuple[str, bytes, bool]

#: one blob's outcome: its new bytes (None when unchanged or not rewritable), the type name of the serializer
#: failure that stopped it (empty when none), and the type tag its re-encoding would have changed to (empty when
#: it would not)
type BlobOutcome = tuple[bytes | None, str, str]

#: each worker's serializers, made once per class
_serdes: BuildOnce[type[Any], Any] = BuildOnce()

#: what a worker logs for a blob its serializer could not decode or re-encode; the caller reports the row
_SERDE_FAILED: Final = (
    "a stored checkpoint blob's serializer failed in an anonymize worker; the blob is answered as failed and the "
    "calling process reports its row"
)

log = get_logger(__name__)

#: the processes every saver in this process shares (:class:`~threetears.core.process_workers.ProcessWorkers`)
_workers = ProcessWorkers(workers=ANONYMIZE_WORKERS, name="checkpoint-anonymize")


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
    serde = _serdes.get(serde_type, serde_type)
    classification = MetadataKeyClassification(identifying=identifying, kept=kept)
    outcomes: list[BlobOutcome] = []
    unclassified: set[str] = set()
    for type_tag, blob, is_metadata in blobs:
        try:
            decoded = serde.loads_typed((type_tag, blob))
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- the serde protocol has no error contract; answered as this blob's failure and reported by the caller, never silent
            contain(exc, log=log, what=_SERDE_FAILED)
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
            contain(exc, log=log, what=_SERDE_FAILED)
            outcomes.append((None, type(exc).__name__, ""))
            continue
        # rewriting it under another tag would leave a row the graph reads with the wrong decoder
        outcomes.append((None, ValueError.__name__, new_type) if new_type != type_tag else (new_blob, "", ""))
    return outcomes, frozenset(unclassified)


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
    answers = await _workers.run_each(
        anonymize_blobs,
        [
            (serde_type, classification.identifying, classification.kept, list(blobs[start : start + size]))
            for start in range(0, len(blobs), size)
        ],
    )
    outcomes: list[BlobOutcome] = []
    unclassified: set[str] = set()
    for chunk_outcomes, chunk_unclassified in answers:
        outcomes.extend(chunk_outcomes)
        unclassified |= chunk_unclassified
    return outcomes, frozenset(unclassified)


async def close_anonymize_workers() -> None:
    """shut the worker processes down, waiting for them; the next anonymization starts a fresh pool.

    :return: nothing
    :rtype: None
    """
    await _workers.close()
