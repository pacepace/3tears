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
- a value under any key in :data:`IDENTIFYING_METADATA_KEYS`, at any depth of any stored
  value: the turn metadata the channel router sends with each message, which the graph
  keeps in its ``metadata`` channel, its ``__start__`` input and its pending writes.

**an unknown key is kept, and this is a deliberate inversion of the audit rule.** audit
``details`` are a record nothing reads back, so masking a key nobody classified costs
only that field. a checkpoint is working state the graph reloads and resumes from: the
``metadata`` channel also carries the injectors' ledgers (``surfaced_memory_ids``,
``governed_knowledge_block`` and the rest), and masking them would change what the agent
does on its next turn. so this rule names what identifies the person and changes exactly
that; a producer that puts a new identifying key into graph state must add it here.

**what it does not reach.** a value held by an object that is not a mapping, a list, a
tuple or a message (an interrupt payload, a custom state class) is kept as it is, and so
is identity written into free text -- the message content is kept by ruling.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from langchain_core.messages import BaseMessage, HumanMessage

from threetears.observe.erasure import ANONYMIZED_MARKER

__all__ = [
    "IDENTIFYING_METADATA_KEYS",
    "CheckpointAnonymization",
    "UnreadableCheckpointBlob",
    "anonymize_checkpoint_value",
]


#: metadata keys whose value identifies the person who sent a turn. the channel router
#: writes both into every channel message's metadata (the sender's display name and the
#: chat platform's id for them); the agent runtime carries that metadata into graph state
#: unchanged. ``channel_ref`` / ``workspace_ref`` name a channel and a workspace, not a
#: person, and the platform's own ``user_id`` is an id, which erasure never changes.
IDENTIFYING_METADATA_KEYS: Final[frozenset[str]] = frozenset({"external_user_name", "external_user_id"})


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
    """

    threads: int
    checkpoints_rewritten: int
    writes_rewritten: int
    l2_prefix_swept: bool | None
    unreadable: tuple[UnreadableCheckpointBlob, ...] = ()


def anonymize_checkpoint_value(value: Any) -> Any:
    """a stored checkpoint value with every identifying field anonymized.

    walks mappings, lists, tuples and langchain messages. keeps every key, every id,
    every message's content and every container's type; returns the same object when
    nothing in it changed, so a caller can tell an already-anonymized value by identity or
    equality. pure: the input is never mutated. idempotent: the marker is replaced by
    itself.

    :param value: a deserialized checkpoint, checkpoint metadata, or pending-write value
    :ptype value: Any
    :return: the value with every identifying field set to the marker
    :rtype: Any
    """
    result: Any = value
    if isinstance(value, BaseMessage):
        result = _anonymize_message(value)
    elif isinstance(value, Mapping):
        rewritten = {
            key: _mask(child) if key in IDENTIFYING_METADATA_KEYS else anonymize_checkpoint_value(child)
            for key, child in value.items()
        }
        if any(rewritten[key] is not child for key, child in value.items()):
            result = rewritten
    elif isinstance(value, list | tuple):
        children = [anonymize_checkpoint_value(child) for child in value]
        if any(new is not old for new, old in zip(children, value, strict=True)):
            result = children if isinstance(value, list) else tuple(children)
    return result


def _mask(value: Any) -> Any:
    """the marker for a present identifying value; ``None`` and the marker itself unchanged.

    :param value: a value under an identifying key
    :ptype value: Any
    :return: ``value`` when it is ``None`` or already the marker, else the marker
    :rtype: Any
    """
    return value if value is None or value == ANONYMIZED_MARKER else ANONYMIZED_MARKER


def _anonymize_message(message: BaseMessage) -> BaseMessage:
    """a message with a human sender's name masked and its kwargs walked.

    :param message: a langchain message
    :ptype message: BaseMessage
    :return: the same message when nothing changed, else a copy with the same id and content
    :rtype: BaseMessage
    """
    update: dict[str, Any] = {}
    if isinstance(message, HumanMessage) and message.name is not None and message.name != ANONYMIZED_MARKER:
        update["name"] = ANONYMIZED_MARKER
    for field_name in ("additional_kwargs", "response_metadata"):
        current = getattr(message, field_name)
        rewritten = anonymize_checkpoint_value(current)
        if rewritten is not current:
            update[field_name] = rewritten
    return message.model_copy(update=update) if update else message
