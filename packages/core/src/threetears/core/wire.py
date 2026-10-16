"""the tolerant reader for wire models that cross between separately deployed pieces.

a model a receiver validates from another deployable's bytes -- a pod's request at the hub, the
registry's call at a tool pod -- must not refuse the whole message because its sender is newer.
with ``extra="forbid"`` an additive optional field refuses every call from whichever side upgraded
first, which is a lockstep release in disguise. :class:`TolerantWireModel` ignores fields it does not
know, so senders and receivers upgrade on their own schedules (expand / migrate / contract, owner
ruling 2026-10-07).

ignoring is not always safe. a field that would let a caller name its own authority -- one retired
for that reason (a body ``user_id`` the door used to trust), or one that was never the sender's to
send (a ``customer_id`` beside a token that already says it) -- must still be refused: a stale or
hostile client sending it alongside a valid token would otherwise have it silently dropped and be
granted broader authority than it asked for, with no error anywhere. such fields are named in
:attr:`refused_fields` and refused at the border.

config models, internal models and anything inside one deployable stay ``extra="forbid"``: a typo
there is a bug to surface, not a newer peer to tolerate.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, model_validator

from threetears.observe import get_logger

__all__ = ["TolerantWireModel"]

log = get_logger(__name__)


class TolerantWireModel(BaseModel):
    """base for a wire model read across a deployment boundary: unknown fields ignored, named ones refused.

    :cvar refused_fields: field names a sender may not send because they would steer authority
    """

    model_config = ConfigDict(extra="ignore")

    refused_fields: ClassVar[frozenset[str]] = frozenset()

    @model_validator(mode="before")
    @classmethod
    def refuse_authority_fields(cls, data: Any) -> Any:  # noqa: ANN401 -- pydantic hands a before-validator the raw input
        """
        refuse message carrying field this model refuses.

        :param data: raw input before field validation
        :ptype data: Any
        :return: input unchanged when it names no refused field
        :rtype: Any
        :raises ValueError: when input names a refused field
        """
        if isinstance(data, dict) and cls.refused_fields:
            named = sorted(cls.refused_fields.intersection(data))
            if named:
                log.info(
                    "wire message refused for a field that would steer authority",
                    extra={"extra_data": {"model": cls.__name__, "refused_fields": named}},
                )
                raise ValueError(f"field(s) {named} may not be sent to {cls.__name__}")
        return data
