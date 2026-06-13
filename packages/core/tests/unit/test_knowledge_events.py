"""unit tests for the correction-harvest wire models (knowledge-task-06).

these models cross the agent -> hub seam: the SDK serializes a
:class:`KnowledgeDraftEvent` / :class:`KnowledgeDraftCommand` to JSON,
publishes it on ``{ns}.knowledge.draft``, and the hub deserializes it
back. the tests exercise the real round-trip (``model_dump_json`` ->
``model_validate_json``) and the field contracts both sides depend on,
not just construction.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from threetears.knowledge import (
    KnowledgeDraftCommand,
    KnowledgeDraftEvent,
    KnowledgeDraftMessage,
)


def test_draft_event_roundtrips_through_json() -> None:
    """a draft event survives serialize -> wire -> deserialize unchanged."""
    event = KnowledgeDraftEvent(
        draft_id=uuid4(),
        target="entry",
        customer_id=uuid4(),
        user_id=uuid4(),
        datasource_id=uuid4(),
        title="exclude fraud accounts from active-user counts",
        body="active_users excludes rows where fraud_flag = true",
        related_domain="active users",
        conversation_id=uuid4(),
        message_id_source=uuid4(),
        turn_count=3,
        contradicts_entity_id=uuid4(),
    )
    wire = event.model_dump_json()
    restored = KnowledgeDraftEvent.model_validate_json(wire)
    assert restored == event
    # provenance survives the round-trip verbatim (audit walks on it).
    assert restored.conversation_id == event.conversation_id
    assert restored.message_id_source == event.message_id_source
    assert restored.turn_count == 3
    assert restored.contradicts_entity_id == event.contradicts_entity_id


def test_draft_event_contradicts_defaults_none() -> None:
    """a correction that contradicts nothing leaves the candidate link unset."""
    event = KnowledgeDraftEvent(
        draft_id=uuid4(),
        target="concept",
        customer_id=uuid4(),
        user_id=uuid4(),
        datasource_id=uuid4(),
        title="early vote",
        body="early_vote excludes UOCAVA ballots until certification",
        conversation_id=uuid4(),
        message_id_source=uuid4(),
        turn_count=1,
    )
    assert event.contradicts_entity_id is None
    assert event.related_domain == ""


def test_draft_event_rejects_unknown_target() -> None:
    """only 'entry' / 'concept' are valid draft targets."""
    with pytest.raises(ValidationError):
        KnowledgeDraftEvent(
            draft_id=uuid4(),
            target="memory",  # type: ignore[arg-type]
            customer_id=uuid4(),
            user_id=uuid4(),
            datasource_id=uuid4(),
            title="t",
            body="b",
            conversation_id=uuid4(),
            message_id_source=uuid4(),
            turn_count=1,
        )


def test_draft_command_roundtrips_and_carries_actor() -> None:
    """a lifecycle command round-trips and always names the acting user."""
    user_id = uuid4()
    draft_id = uuid4()
    command = KnowledgeDraftCommand(
        draft_id=draft_id,
        target="entry",
        op="confirm",
        user_id=user_id,
    )
    restored = KnowledgeDraftCommand.model_validate_json(
        command.model_dump_json(),
    )
    assert restored == command
    # the acting user is load-bearing: the hub enforces own-draft-only.
    assert restored.user_id == user_id
    assert restored.draft_id == draft_id


def test_draft_command_edit_carries_new_body() -> None:
    """an edit command carries the replacement title + body."""
    command = KnowledgeDraftCommand(
        draft_id=uuid4(),
        target="concept",
        op="edit",
        user_id=uuid4(),
        title="active users",
        body="active_users = MAU excluding internal + fraud",
    )
    restored = KnowledgeDraftCommand.model_validate_json(
        command.model_dump_json(),
    )
    assert restored.op == "edit"
    assert restored.title == "active users"
    assert restored.body == "active_users = MAU excluding internal + fraud"


def test_draft_command_rejects_unknown_op() -> None:
    """only confirm / edit / discard are valid lifecycle ops."""
    with pytest.raises(ValidationError):
        KnowledgeDraftCommand(
            draft_id=uuid4(),
            target="entry",
            op="promote",  # type: ignore[arg-type]
            user_id=uuid4(),
        )


def test_envelope_wraps_event_and_roundtrips() -> None:
    """the envelope carries an event under kind='event' and round-trips."""
    event = KnowledgeDraftEvent(
        draft_id=uuid4(),
        target="entry",
        customer_id=uuid4(),
        user_id=uuid4(),
        datasource_id=uuid4(),
        title="t",
        body="b",
        conversation_id=uuid4(),
        message_id_source=uuid4(),
        turn_count=1,
    )
    envelope = KnowledgeDraftMessage.for_event(event)
    restored = KnowledgeDraftMessage.model_validate_json(
        envelope.model_dump_json(),
    )
    assert restored.kind == "event"
    assert restored.event == event
    assert restored.command is None


def test_envelope_wraps_command_and_roundtrips() -> None:
    """the envelope carries a command under kind='command' and round-trips."""
    command = KnowledgeDraftCommand(
        draft_id=uuid4(),
        target="concept",
        op="discard",
        user_id=uuid4(),
    )
    envelope = KnowledgeDraftMessage.for_command(command)
    restored = KnowledgeDraftMessage.model_validate_json(
        envelope.model_dump_json(),
    )
    assert restored.kind == "command"
    assert restored.command == command
    assert restored.event is None


def test_envelope_kind_discriminates_without_json_loads() -> None:
    """the kind field lets a consumer route without peeking at raw json.

    this is the property the hub-side emitter depends on (wire-format
    law: validate with model_validate_json, dispatch on kind).
    """
    event_env = KnowledgeDraftMessage.for_event(
        KnowledgeDraftEvent(
            draft_id=uuid4(),
            target="entry",
            customer_id=uuid4(),
            user_id=uuid4(),
            datasource_id=uuid4(),
            title="t",
            body="b",
            conversation_id=uuid4(),
            message_id_source=uuid4(),
            turn_count=1,
        ),
    )
    command_env = KnowledgeDraftMessage.for_command(
        KnowledgeDraftCommand(
            draft_id=uuid4(),
            target="entry",
            op="confirm",
            user_id=uuid4(),
        ),
    )
    assert event_env.kind == "event"
    assert command_env.kind == "command"
