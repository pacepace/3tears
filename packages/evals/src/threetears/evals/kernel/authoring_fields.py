"""The write-seam refusal every eval authoring surface shares: a field the target model does not declare.

Templates, rubric dimensions, judge configs and campaigns are authored by the same two surfaces
(MCP and REST) through functions in two packages that may not import each other, so the one guard
they share lives here rather than in either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from threetears.evals.kernel.errors import ValidationFailedError

if TYPE_CHECKING:
    from pydantic import BaseModel


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
            f"{action}: not a field of {model.__name__}: {', '.join(unknown)}. Accepted: {accepted}. "
            f"Ignored if sent (server-owned): {', '.join(sorted(server_fields))}"
        )


__all__ = [
    "reject_unknown_authoring_fields",
]
