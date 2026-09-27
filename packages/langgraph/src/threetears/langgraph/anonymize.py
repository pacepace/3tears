"""which stored checkpoint fields identify a person, and the rewrite that anonymizes them.

person erasure keeps the conversation. every checkpoint and pending write stays, every id
stays (thread, checkpoint, message, task), and the text of every message stays for
analysis. what changes is each stored field that IDENTIFIES the person, in place, to
:data:`threetears.observe.erasure.ANONYMIZED_MARKER` -- the platform's one erasure marker.

**the rule, in one place.**

- a human message's ``name``: the sender's chat display name, which the agent runtime sets
  from the channel's sender (``HumanMessage(content=..., name=external_user_name)``).
  every other message's ``name`` is not a person -- an AI message's names the agent, a
  tool message's names the tool -- and is kept.
- a value under any key in :data:`IDENTIFYING_METADATA_KEYS`, or declared through
  :func:`declare_identifying_metadata_keys`, at any depth of any stored value: the turn
  metadata the channel router sends with each message, which the graph keeps in its
  ``metadata`` channel, its ``__start__`` input and its pending writes.

**an unknown key is kept, and this is a deliberate inversion of the audit rule.** audit
``details`` are a record nothing reads back, so masking a key nobody classified costs
only that field. a checkpoint is working state the graph reloads and resumes from: the
``metadata`` channel also carries the injectors' ledgers (``surfaced_memory_ids``,
``governed_knowledge_block`` and the rest), and masking them would change what the agent
does on its next turn. so this rule names what identifies the person and changes exactly
that; a producer that puts a new identifying key into graph state must classify it -- here,
or through the declarations below.

**an unknown key is kept, but never silently.** the producers of turn metadata (the channel
router, the agent runtime, the injectors) mostly live outside this package, so nothing here
can know every key they write. :data:`KEPT_METADATA_KEYS` records the keys ruled NOT to
identify a person; :func:`unclassified_metadata_keys` names every turn-metadata key in a
stored value that no classification names, and the saver reports them on
:attr:`CheckpointAnonymization.unclassified_metadata_keys`. a run that met one cannot vouch
that the erasure is complete: if that key identifies a person, its values are still stored.

**a consumer classifies its own keys.** the keys a consumer's router or runtime writes are
decided by that consumer, not here: :func:`declare_identifying_metadata_keys` adds keys whose
value identifies or describes the person (anonymized exactly like
:data:`IDENTIFYING_METADATA_KEYS`), and :func:`declare_kept_metadata_keys` adds keys ruled not
to (kept exactly like :data:`KEPT_METADATA_KEYS`). :func:`metadata_key_classification` is the
one lookup the rule consults: the built-in sets plus every declaration. a declaration is
visible only inside the process that makes it -- the same caveat as
:func:`threetears.agent.audit.declare_safe_detail_keys`. it must therefore be made in the
process that RUNS the anonymization (the one calling
:meth:`~threetears.langgraph.ThreeTierCheckpointSaver.aanonymize_threads`), which is often not
the process that wrote the checkpoint: a declaration made only in the agent pod that produced
a key leaves the eraser reporting it as unclassified.

**what it does not reach.** a value held by an object that is not a mapping, a list, a
tuple or a message (an interrupt payload, a custom state class) is kept as it is, and so
is identity written into free text -- the message content is kept by ruling.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from langchain_core.messages import BaseMessage, HumanMessage

from threetears.observe import get_logger
from threetears.observe.erasure import ANONYMIZED_MARKER

__all__ = [
    "IDENTIFYING_METADATA_KEYS",
    "KEPT_METADATA_KEYS",
    "METADATA_CHANNEL",
    "CheckpointAnonymization",
    "MetadataKeyClassification",
    "UnreadableCheckpointBlob",
    "anonymize_checkpoint_value",
    "declare_identifying_metadata_keys",
    "declare_kept_metadata_keys",
    "metadata_key_classification",
    "unclassified_metadata_keys",
]

log = get_logger(__name__)


#: metadata keys whose value identifies the person who sent a turn. the channel router
#: writes both into every channel message's metadata (the sender's display name and the
#: chat platform's id for them); the agent runtime carries that metadata into graph state
#: unchanged. ``channel_ref`` / ``workspace_ref`` name a channel and a workspace, not a
#: person, and the platform's own ``user_id`` is an id, which erasure never changes.
IDENTIFYING_METADATA_KEYS: Final[frozenset[str]] = frozenset({"external_user_name", "external_user_id"})

#: turn-metadata keys ruled NOT to identify a person, so kept as they are. the router's
#: ``channel_ref`` / ``workspace_ref`` name a channel and a workspace, and ``user_id`` is the
#: platform's own id, which erasure never changes; the rest are the ledgers 3tears' own
#: injectors keep in the metadata channel (memory, knowledge and schema injection), which the
#: agent reads back on its next turn. a key in neither this set nor
#: :data:`IDENTIFYING_METADATA_KEYS` is reported by :func:`unclassified_metadata_keys`.
KEPT_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "channel_ref",
        "workspace_ref",
        "user_id",
        "surfaced_memory_ids",
        "documented_schema_block",
        "governed_knowledge_block",
        "knowledge_shadow_disclosures",
        "knowledge_concept_shadow_disclosures",
        "knowledge_injected_concepts",
        "knowledge_injected_entries",
    }
)

#: the state channel that carries turn metadata, and the key the turn's input carries it
#: under -- the name :func:`threetears.langgraph.merge_metadata` reduces.
METADATA_CHANNEL: Final[str] = "metadata"


@dataclass(frozen=True)
class MetadataKeyClassification:
    """every turn-metadata key someone decided about, at one moment, in this process.

    :ivar identifying: keys whose value identifies or describes the person, anonymized at
        any depth: :data:`IDENTIFYING_METADATA_KEYS` plus every
        :func:`declare_identifying_metadata_keys` declaration
    :ivar kept: keys ruled not to, kept as they are: :data:`KEPT_METADATA_KEYS` plus every
        :func:`declare_kept_metadata_keys` declaration
    """

    identifying: frozenset[str]
    kept: frozenset[str]

    def is_classified(self, key: str) -> bool:
        """
        whether someone decided about ``key``, one way or the other.

        :param key: a turn-metadata key
        :ptype key: str
        :return: ``True`` when the key is identifying or kept
        :rtype: bool
        """
        return key in self.identifying or key in self.kept


#: serializes declarations. readers take the current classification without the lock: it is
#: replaced wholesale, never mutated in place.
_declare_lock = threading.Lock()
_classification = MetadataKeyClassification(identifying=IDENTIFYING_METADATA_KEYS, kept=KEPT_METADATA_KEYS)


def metadata_key_classification() -> MetadataKeyClassification:
    """
    the complete classification of turn-metadata keys in this process.

    the single lookup behind :func:`anonymize_checkpoint_value` and
    :func:`unclassified_metadata_keys`: the built-in sets plus every declaration this process
    has made. a consumer's enforcement test asks it whether every key it writes is classified.

    :return: the identifying and kept keys as they stand now
    :rtype: MetadataKeyClassification
    """
    return _classification


def declare_identifying_metadata_keys(keys: Iterable[str]) -> None:
    """
    declare turn-metadata keys whose value identifies or describes the person, in this process.

    the value under a declared key is anonymized at any depth of every stored value, exactly
    like :data:`IDENTIFYING_METADATA_KEYS`: a name, an email, a locale, a timezone, an external
    account id, text the person typed. when unsure whether a key is personal, declare it here:
    a kept personal value is a failed erasure, a masked routing value costs one field.
    declarations accumulate and repeating one is harmless.

    the declaration is visible only in the calling process. make it in the process that runs
    the anonymization (the one calling
    :meth:`~threetears.langgraph.ThreeTierCheckpointSaver.aanonymize_threads`), not only in the
    one that writes the key.

    :param keys: the turn-metadata keys to anonymize
    :ptype keys: Iterable[str]
    :return: nothing
    :rtype: None
    :raises TypeError: if ``keys`` is a bare string, which would declare its characters
    :raises ValueError: if a key is blank, is :data:`METADATA_CHANNEL` itself, or is already
        classified as kept -- a key cannot be both, and the batch declares nothing
    """
    _declare(keys, identifying=True)


def declare_kept_metadata_keys(keys: Iterable[str]) -> None:
    """
    declare turn-metadata keys ruled NOT to identify or describe a person, in this process.

    the value under a declared key is kept, exactly like :data:`KEPT_METADATA_KEYS`: platform
    routing and scoping data with no personal content, or a ledger the agent reads back on
    its next turn. declarations accumulate and repeating one is harmless.

    the declaration is visible only in the calling process. make it in the process that runs
    the anonymization (the one calling
    :meth:`~threetears.langgraph.ThreeTierCheckpointSaver.aanonymize_threads`), not only in the
    one that writes the key.

    :param keys: the turn-metadata keys to keep
    :ptype keys: Iterable[str]
    :return: nothing
    :rtype: None
    :raises TypeError: if ``keys`` is a bare string, which would declare its characters
    :raises ValueError: if a key is blank, is :data:`METADATA_CHANNEL` itself, or is already
        classified as identifying -- a key cannot be both, and the batch declares nothing
    """
    _declare(keys, identifying=False)


def _declare(keys: Iterable[str], *, identifying: bool) -> None:
    """
    add keys to one side of the classification, refusing the whole batch on any bad key.

    :param keys: the turn-metadata keys to declare
    :ptype keys: Iterable[str]
    :param identifying: ``True`` to declare them identifying, ``False`` to declare them kept
    :ptype identifying: bool
    :return: nothing
    :rtype: None
    :raises TypeError: if ``keys`` is a bare string
    :raises ValueError: if a key is blank, is the metadata channel's own name, or is already on
        the other side
    """
    if isinstance(keys, str):
        raise TypeError(f"keys must be a collection of key names, not the bare string {keys!r}")
    declared = frozenset(keys)
    blank = sorted(key for key in declared if not key.strip())
    if blank:
        raise ValueError(f"turn-metadata keys must be non-blank; received {blank!r}")
    if METADATA_CHANNEL in declared:
        raise ValueError(
            f"{METADATA_CHANNEL!r} names the turn-metadata channel itself, not a key in it; declaring it would "
            "classify every stored channel value under that name at once"
        )
    global _classification
    with _declare_lock:
        _classification = _with_declared(_classification, declared, identifying=identifying)
    log.info(
        "turn-metadata keys classified for checkpoint anonymization",
        extra={"extra_data": {"classification": "identifying" if identifying else "kept", "keys": sorted(declared)}},
    )


def _with_declared(
    current: MetadataKeyClassification, declared: frozenset[str], *, identifying: bool
) -> MetadataKeyClassification:
    """
    the classification with keys added to one side, refused when any is on the other.

    :param current: the classification as it stands
    :ptype current: MetadataKeyClassification
    :param declared: the non-blank keys being declared
    :ptype declared: frozenset[str]
    :param identifying: ``True`` to add them to the identifying side, ``False`` to the kept side
    :ptype identifying: bool
    :return: a new classification; ``current`` is never changed
    :rtype: MetadataKeyClassification
    :raises ValueError: if a key is already classified the other way
    """
    contradicted = sorted(declared & (current.kept if identifying else current.identifying))
    if contradicted:
        already = "kept" if identifying else "identifying"
        raise ValueError(f"turn-metadata keys already classified as {already} cannot be both: {contradicted!r}")
    return (
        MetadataKeyClassification(identifying=current.identifying | declared, kept=current.kept)
        if identifying
        else MetadataKeyClassification(identifying=current.identifying, kept=current.kept | declared)
    )


@dataclass(frozen=True)
class UnreadableCheckpointBlob:
    """one stored blob the rule could not be applied to, named so an operator can find it.

    a blob the saver's serializer cannot decode -- or whose re-encoding would change its
    serialization type -- cannot be rewritten, and its bytes may still hold the person's
    data. it is reported rather than skipped silently, and every other row is still
    rewritten. the graph cannot load a blob it cannot decode either, so deleting the row
    loses nothing the graph could use.

    :ivar thread_id: the thread as the caller named it
    :ivar table: ``checkpoints`` or ``checkpoint_writes``
    :ivar column: the blob's column (``checkpoint``, ``metadata_`` or ``blob``)
    :ivar checkpoint_ns: the row's checkpoint namespace
    :ivar checkpoint_id: the row's checkpoint id
    :ivar task_id: the pending write's task id; ``None`` for a checkpoint row
    :ivar idx: the pending write's index; ``None`` for a checkpoint row
    :ivar error_type: the class of the error that stopped the rewrite (its text is left
        out: a decoder's message can echo the bytes it could not read)
    """

    thread_id: str
    table: str
    column: str
    checkpoint_ns: str
    checkpoint_id: str
    task_id: str | None
    idx: int | None
    error_type: str


@dataclass(frozen=True)
class CheckpointAnonymization:
    """what one anonymization run rewrote, and what it could not.

    :ivar threads: threads processed
    :ivar checkpoints_rewritten: checkpoint rows whose stored blobs changed
    :ivar writes_rewritten: pending-write rows whose stored blob changed
    :ivar l2_prefix_swept: whether every namespaced L2 bundle was swept -- ``None`` when the
        saver has no L2, ``False`` when its L2 cannot sweep by prefix (only root-namespace
        bundles were evicted)
    :ivar unreadable: every stored blob the rule could not be applied to; empty when the
        erasure reached every row. a non-empty value means the erasure is NOT complete for
        those rows, however many times the run is repeated
    :ivar unclassified_metadata_keys: every turn-metadata key the run found that
        :func:`metadata_key_classification` does not name, sorted. their values were kept. a
        non-empty value means the run cannot vouch that the erasure is complete: a key a
        producer added that identifies a person is still stored, until it is classified and
        the run repeated
    """

    threads: int
    checkpoints_rewritten: int
    writes_rewritten: int
    l2_prefix_swept: bool | None
    unreadable: tuple[UnreadableCheckpointBlob, ...] = ()
    unclassified_metadata_keys: tuple[str, ...] = ()


def anonymize_checkpoint_value(value: Any) -> Any:
    """a stored checkpoint value with every identifying field anonymized.

    walks mappings, lists, tuples and langchain messages. keeps every key, every id,
    every message's content and every container's type; returns the same object when
    nothing in it changed, so a caller can tell an already-anonymized value by identity or
    equality. pure: the input is never mutated. idempotent: the marker is replaced by
    itself.

    the identifying keys are :func:`metadata_key_classification`'s, read once per call.

    :param value: a deserialized checkpoint, checkpoint metadata, or pending-write value
    :ptype value: Any
    :return: the value with every identifying field set to the marker
    :rtype: Any
    """
    return _anonymized(value, metadata_key_classification().identifying)


def _anonymized(value: Any, identifying: frozenset[str]) -> Any:
    """
    :func:`anonymize_checkpoint_value` against one fixed set of identifying keys.

    :param value: a deserialized checkpoint, checkpoint metadata, or pending-write value
    :ptype value: Any
    :param identifying: the keys whose values are masked, at any depth
    :ptype identifying: frozenset[str]
    :return: the value with every identifying field set to the marker
    :rtype: Any
    """
    result: Any = value
    if isinstance(value, BaseMessage):
        result = _anonymize_message(value, identifying)
    elif isinstance(value, Mapping):
        rewritten = {
            key: _mask(child) if key in identifying else _anonymized(child, identifying) for key, child in value.items()
        }
        if any(rewritten[key] is not child for key, child in value.items()):
            result = rewritten
    elif isinstance(value, list | tuple):
        children = [_anonymized(child, identifying) for child in value]
        if any(new is not old for new, old in zip(children, value, strict=True)):
            result = children if isinstance(value, list) else tuple(children)
    return result


def unclassified_metadata_keys(value: Any, *, is_metadata: bool = False) -> frozenset[str]:
    """every turn-metadata key in a stored value that no classification names.

    turn metadata is a mapping stored under the :data:`METADATA_CHANNEL` key, at any depth --
    the checkpoint's channel values, the turn's ``__start__`` input, a checkpoint's recorded
    writes -- or a pending write's whole value when it was written to that channel, which the
    caller says with *is_metadata*. its top-level keys are the ones producers write and the
    ones classified; what sits beneath a key is that key's own value. the walk reaches the
    containers :func:`anonymize_checkpoint_value` reaches, and errs toward reporting: a
    mapping stored under a ``metadata`` key that is not turn metadata has its keys reported
    too, which costs a line in a report rather than a key missed.

    a key is classified when :func:`metadata_key_classification` names it, read once per call:
    the built-in sets plus every declaration this process has made.

    :param value: a deserialized checkpoint, checkpoint metadata, or pending-write value
    :ptype value: Any
    :param is_metadata: whether *value* itself is the metadata channel's value
    :ptype is_metadata: bool
    :return: the keys no classification names
    :rtype: frozenset[str]
    """
    return _unclassified(value, metadata_key_classification(), is_metadata=is_metadata)


def _unclassified(value: Any, classification: MetadataKeyClassification, *, is_metadata: bool) -> frozenset[str]:
    """
    :func:`unclassified_metadata_keys` against one fixed classification.

    :param value: a deserialized checkpoint, checkpoint metadata, or pending-write value
    :ptype value: Any
    :param classification: the keys someone decided about
    :ptype classification: MetadataKeyClassification
    :param is_metadata: whether *value* itself is the metadata channel's value
    :ptype is_metadata: bool
    :return: the keys the classification does not name
    :rtype: frozenset[str]
    """
    found: set[str] = set()
    if isinstance(value, BaseMessage):
        for field_name in ("additional_kwargs", "response_metadata"):
            found |= _unclassified(getattr(value, field_name), classification, is_metadata=False)
    elif isinstance(value, Mapping):
        if is_metadata:
            found |= {key for key in value if isinstance(key, str) and not classification.is_classified(key)}
        for key, child in value.items():
            found |= _unclassified(child, classification, is_metadata=key == METADATA_CHANNEL)
    elif isinstance(value, list | tuple):
        for child in value:
            found |= _unclassified(child, classification, is_metadata=False)
    return frozenset(found)


def _mask(value: Any) -> Any:
    """the marker for a present identifying value; ``None`` and the marker itself unchanged.

    :param value: a value under an identifying key
    :ptype value: Any
    :return: ``value`` when it is ``None`` or already the marker, else the marker
    :rtype: Any
    """
    return value if value is None or value == ANONYMIZED_MARKER else ANONYMIZED_MARKER


def _anonymize_message(message: BaseMessage, identifying: frozenset[str]) -> BaseMessage:
    """a message with a human sender's name masked and its kwargs walked.

    :param message: a langchain message
    :ptype message: BaseMessage
    :param identifying: the keys whose values are masked in the message's kwargs
    :ptype identifying: frozenset[str]
    :return: the same message when nothing changed, else a copy with the same id and content
    :rtype: BaseMessage
    """
    update: dict[str, Any] = {}
    if isinstance(message, HumanMessage) and message.name is not None and message.name != ANONYMIZED_MARKER:
        update["name"] = ANONYMIZED_MARKER
    for field_name in ("additional_kwargs", "response_metadata"):
        current = getattr(message, field_name)
        rewritten = _anonymized(current, identifying)
        if rewritten is not current:
            update[field_name] = rewritten
    return message.model_copy(update=update) if update else message
