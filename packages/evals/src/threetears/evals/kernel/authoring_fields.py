"""The write seam every eval authoring surface shares: a field the target model does not declare, and an edit of a
stored document that keeps what a newer build wrote on it.

Templates, rubric dimensions, judge configs and campaigns are authored by the same two surfaces
(MCP and REST) through functions in two packages that may not import each other, so the guards
they share live here rather than in either.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from pydantic import BaseModel

from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.schema.base import CoreDocumentModel


def reject_unknown_authoring_fields(
    action: str, fields: dict[str, Any], model: type[BaseModel], server_fields: tuple[str, ...]
) -> None:
    """Refuse caller field names the target model does not declare.

    **The write seam.** Constructing one of these models with an undeclared field raises
    anyway — the base is ``extra="forbid"`` — but this guard refuses the caller before the
    merge reaches a model at all, which is what lets it name the action and the field in an
    operator-facing message rather than surfacing a pydantic error. Before it, a misspelled
    field name could be dropped in silence and then reported as changed, because the MCP
    handlers list the keys the caller SENT. For a judge config the no-op was
    worse than cosmetic, since ``update_judge_config`` archives the record and mints a new
    id — so a ``judge_config_ids`` pin came to name an archived version while nothing the
    operator asked for applied.

    Checked here rather than at each surface so MCP and REST inherit one contract
    structurally; a second guard at the handlers is one that drifts. ``update_campaign``
    already did this with an explicit allowlist and is the shape followed.

    Server-owned fields stay IGNORED rather than refused. They are documented that way, and
    a caller echoing back a record it just read — ``id``, ``created_at`` — is doing
    something reasonable that a refusal would break.

    Args:
        action: The action name, for the message.
        fields: The caller's partial field map.
        model: The model the merge will construct — the authority on what is declared.
        server_fields: Names ignored without error.

    Raises:
        ValidationFailedError: ``fields`` names something the model does not declare.
    """
    declared = set(model.model_fields)
    if unknown := sorted(set(fields) - declared - set(server_fields)):
        accepted = ", ".join(sorted(declared - set(server_fields)))
        raise ValidationFailedError(
            f"{action}: not a field of {model.__name__}: {', '.join(unknown)}",
            remedy=(f"send only {accepted}; the server-owned {', '.join(sorted(server_fields))} are ignored if sent"),
        )


def edited_core_document[M: CoreDocumentModel](
    model: type[M], existing: M, changes: Mapping[str, Any], *, drop: Collection[str] = ()
) -> M:
    """``existing`` with ``changes`` set on it: what the caller sent validated strictly, what a newer build wrote kept.

    An edit of a stored core document is two questions, answered by two validations. **Is what the caller sent
    valid?** The stored document as THIS build declares it (``model_dump``, which carries no field a newer minor's
    read kept, at any depth), with ``changes`` on top, at this build's minor, is validated as strictly as any
    payload — so a value the caller sent, at any depth, is refused exactly as it is on a document this build wrote,
    and a caller can never reach the stored read's tolerance (an unknown field NAME is refused before this, by
    :func:`reject_unknown_authoring_fields`, and ``schema_minor`` is a server field every authoring surface
    ignores). **What is written?** ``existing`` itself with each change set on it (``model_copy``), written through
    its ``to_dict`` and read back through ``from_dict``, the stored read: so every field a newer build wrote on the
    document, and on every shape the edit carries over, is kept. A change given as a model instance (a stored
    design the surface re-stamped) is set as that instance, kept fields and all; a change given as plain data (what
    a caller sent) is set as the strict check validated it, carrying nothing a newer build wrote, because nothing a
    caller sends is. Only the second validation's result is returned; the first only refuses.

    Args:
        model: The core document's model.
        existing: The document as its stored read returned it.
        changes: The fields to set, each to its new value: the caller's (already checked by name) and the
            surface's own (``updated_at``, a re-stamped design).
        drop: Fields to leave out of the result, so their defaults are taken (a new version's fresh ``id``).

    Returns:
        The edited document, ready to write.

    Raises:
        ValidationError: ``changes`` makes a document this build refuses.
    """
    as_declared = {
        key: value.model_dump(mode="json") if isinstance(value, BaseModel) else value for key, value in changes.items()
    }
    declared = {**existing.model_dump(mode="json"), **as_declared, "schema_minor": model.version_space.minor}
    for key in drop:
        declared.pop(key, None)
    checked = model.model_validate(declared)
    update = {key: value if isinstance(value, BaseModel) else getattr(checked, key) for key, value in changes.items()}
    written = existing.model_copy(update=update).to_dict()
    for key in drop:
        written.pop(key, None)
    return model.from_dict(written)


__all__ = [
    "edited_core_document",
    "reject_unknown_authoring_fields",
]
