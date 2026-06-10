"""shared wire models for correction-harvest drafts (knowledge-task-06).

correction harvesting (KNW-50..54) detects when a user corrects the
agent in conversation and drafts a USER-scoped playbook entry or
concept from the correction. the agent pod cannot write
``platform.*`` directly — the broker's read-only carve-out admits the
agent only SELECT traffic against the knowledge tables, and the hub is
the SOLE writer of platform-scoped rows (mirroring the
``WorkspaceCreateEvent`` -> ``WorkspaceNamespaceEmitter`` seam). so the
write travels as an EVENT: the agent publishes one of these models on
``{ns}.knowledge.draft`` and a hub-side emitter materializes / mutates
the row through the hub's knowledge Collections.

two model families ride that subject:

- :class:`KnowledgeDraftEvent` — a NEW draft minted from a detected
  correction (KNW-51). carries the draft body, the target entity type,
  the authoring identity (customer + user, so the row is born
  user-scoped, D1), the anchoring datasource, and the turn provenance
  (conversation / source message / turn count) so an audit can walk
  from any knowledge row back to the exact turn that spawned it.
- :class:`KnowledgeDraftCommand` — a lifecycle op on the AUTHOR's OWN
  existing draft (KNW-54): ``confirm`` flips ``status`` draft -> active,
  ``edit`` rewrites the draft body in place, ``discard`` deletes it. the
  command always carries the acting ``user_id`` so the hub emitter can
  enforce that a caller only mutates their OWN drafts (a draft is
  pre-publication and private to its author, never visible to peers or
  admins).

these are pure wire models (no hub / SDK coupling) so both repos import
the SAME definition — the drift hazard the cross-repo schema-parity law
exists to prevent. they live in ``threetears.knowledge`` because both
the hub emitter and the SDK harvester already depend on this submodule
for the scope vocabulary; the merge in this submodule is NOT touched.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel

__all__ = [
    "DraftCommandOp",
    "DraftTarget",
    "KnowledgeDraftCommand",
    "KnowledgeDraftEvent",
    "KnowledgeDraftMessage",
]

#: which knowledge entity a correction drafts into. a correction about a
#: PROCEDURE (which table to start from, what to exclude) drafts an
#: ``entry``; a correction about a TERM's meaning / binding drafts a
#: ``concept``.
DraftTarget = Literal["entry", "concept"]

#: lifecycle op on the author's own draft (KNW-54). ``confirm`` flips
#: draft -> active; ``edit`` rewrites the draft body; ``discard`` deletes.
DraftCommandOp = Literal["confirm", "edit", "discard"]


class KnowledgeDraftEvent(BaseModel):
    """new correction-harvest draft to materialize hub-side (KNW-51).

    published on ``{ns}.knowledge.draft`` by the agent-side correction
    harvester after a turn produces a high-confidence correction. the
    hub-side emitter materializes ONE ``status='draft'`` row in
    ``platform.playbook_entries`` (when ``target == 'entry'``) or
    ``platform.concepts`` (when ``target == 'concept'``), born at USER
    scope from ``(customer_id, user_id)`` (D1). the row carries the turn
    provenance verbatim so an audit walks knowledge -> turn.

    a draft is born WITHOUT an ``origin_*_id`` link even when it
    contradicts a governed ancestor: a draft must not participate in a
    shadow chain (an unconfirmed, possibly-misread draft must not steer
    the merge). the contradicted entity is recorded as a CANDIDATE in
    ``contradicts_entity_id`` so the eventual confirm / promotion review
    is a one-glance diff; the hub sets the real origin link only at
    confirm / promotion time.

    the ``draft_id`` is minted agent-side and reused as the row primary
    key so the hub upsert is idempotent under at-least-once delivery
    (retried publish writes the SAME id via ``ON CONFLICT (id) DO
    UPDATE``).

    :param draft_id: deterministic draft UUID minted agent-side; reused
        as the knowledge row primary key for idempotent upsert
    :ptype draft_id: UUID
    :param target: which entity the draft materializes into
    :ptype target: DraftTarget
    :param customer_id: authoring user's customer (user-scope rows are
        always customer-bound, D1); the row is born at user scope
    :ptype customer_id: UUID
    :param user_id: authoring user; the draft is private to this user
        until confirmed
    :ptype user_id: UUID
    :param datasource_id: the anchoring datasource the correction is
        about (the routing / RBAC anchor, knowledge-task-08)
    :ptype datasource_id: UUID
    :param title: draft entry title (entry target) or concept name
        (concept target)
    :ptype title: str
    :param body: draft entry body (entry target) or concept definition
        (concept target)
    :ptype body: str
    :param related_domain: free-text domain hint the classifier
        produced (recorded for review; not a routing key)
    :ptype related_domain: str
    :param conversation_id: conversation the correction occurred in
    :ptype conversation_id: UUID
    :param message_id_source: source message id for the turn (mirrors
        memory-extraction indexing)
    :ptype message_id_source: UUID
    :param turn_count: turn number within the conversation
    :ptype turn_count: int
    :param contradicts_entity_id: governed ancestor the correction
        contradicts, recorded as a CANDIDATE origin; ``None`` when the
        correction does not contradict a governed row. NEVER set as the
        real origin link on a draft
    :ptype contradicts_entity_id: UUID | None
    """

    draft_id: UUID
    target: DraftTarget
    customer_id: UUID
    user_id: UUID
    datasource_id: UUID
    title: str
    body: str
    related_domain: str = ""
    conversation_id: UUID
    message_id_source: UUID
    turn_count: int
    contradicts_entity_id: UUID | None = None


class KnowledgeDraftCommand(BaseModel):
    """lifecycle op on the author's own existing draft (KNW-54).

    published on ``{ns}.knowledge.draft`` by the ``knowledge_drafts``
    conversational tool when the user confirms / edits / discards one of
    their drafts. the hub-side emitter looks the row up, enforces that
    ``user_id`` OWNS it AND it is still ``status='draft'`` (the only
    rows this command may touch), then applies the op:

    - ``confirm`` flips ``status`` draft -> active (the draft becomes an
      ordinary user-scope row, eligible for promotion, KNW-52 / KNW-53);
    - ``edit`` rewrites ``title`` / ``body`` in place (still a draft);
    - ``discard`` deletes the row.

    a command naming a row the caller does not own, or a row that is no
    longer a draft, is a no-op on the hub side (the emitter never
    mutates another user's row and never re-drafts an active row).

    :param draft_id: the draft row to act on
    :ptype draft_id: UUID
    :param target: which entity table the draft lives in
    :ptype target: DraftTarget
    :param op: the lifecycle operation
    :ptype op: DraftCommandOp
    :param user_id: acting user; MUST own the draft for the op to apply
    :ptype user_id: UUID
    :param title: new title for an ``edit`` op; ignored otherwise
    :ptype title: str | None
    :param body: new body for an ``edit`` op; ignored otherwise
    :ptype body: str | None
    """

    draft_id: UUID
    target: DraftTarget
    op: DraftCommandOp
    user_id: UUID
    title: str | None = None
    body: str | None = None


class KnowledgeDraftMessage(BaseModel):
    """single typed envelope carried on ``{ns}.knowledge.draft``.

    both the new-draft :class:`KnowledgeDraftEvent` and the lifecycle
    :class:`KnowledgeDraftCommand` ride ONE subject, so the wire carries
    a discriminated envelope rather than a bare model: the ``kind`` field
    tells the hub-side emitter which payload to act on WITHOUT raw
    ``json.loads`` peeking (every NATS handler validates with
    ``Model.model_validate_json`` per the wire-format law). exactly one
    of ``event`` / ``command`` is populated, matching ``kind``.

    :param kind: which payload this envelope carries
    :ptype kind: Literal["event", "command"]
    :param event: the new-draft event when ``kind == 'event'``
    :ptype event: KnowledgeDraftEvent | None
    :param command: the lifecycle command when ``kind == 'command'``
    :ptype command: KnowledgeDraftCommand | None
    """

    kind: Literal["event", "command"]
    event: KnowledgeDraftEvent | None = None
    command: KnowledgeDraftCommand | None = None

    @classmethod
    def for_event(cls, event: KnowledgeDraftEvent) -> KnowledgeDraftMessage:
        """wrap a new-draft event in an envelope.

        :param event: the new-draft event to carry
        :ptype event: KnowledgeDraftEvent
        :return: an envelope with ``kind='event'``
        :rtype: KnowledgeDraftMessage
        """
        return cls(kind="event", event=event)

    @classmethod
    def for_command(
        cls, command: KnowledgeDraftCommand,
    ) -> KnowledgeDraftMessage:
        """wrap a lifecycle command in an envelope.

        :param command: the lifecycle command to carry
        :ptype command: KnowledgeDraftCommand
        :return: an envelope with ``kind='command'``
        :rtype: KnowledgeDraftMessage
        """
        return cls(kind="command", command=command)
