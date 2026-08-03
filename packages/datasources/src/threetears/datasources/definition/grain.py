"""entity grain of a dataset definition, and its optional delivered alias.

One shipped deliverable renames the grain in its delivered projection --
``aud.voterbase_id AS individual_id`` -- and the rename propagates into
the physical layout (``DISTKEY(individual_id)``, ``SORTKEY(individual_id)``),
so the alias is part of the delivered contract rather than a cosmetic
label.

What this spec does NOT carry is the source-record column (``list_id``).
The long artifact's grain is ``(unit, source record, entity)``, but that
is a property of the artifact rather than of the definition's entity
grain, and it lands with ``ArtifactSpec`` in ``dsm-task-01d``.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["GrainSpec"]


class GrainSpec(BaseModel):
    """entity column of a definition, plus an optional delivered alias.

    :ivar entity_column: column identifying one entity, e.g. ``voterbase_id``
    :ivar delivered_alias: name that column takes in delivered artifacts,
        or ``None`` when it is delivered under its own name
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_column: str = Field(min_length=1)
    delivered_alias: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _alias_renames_something(self) -> Self:
        """reject an alias identical to the entity column.

        A self-alias emits no rename yet reads as a deliberate one, so it
        is refused rather than silently ignored.

        :returns: validated grain
        :rtype: GrainSpec
        :raises ValueError: alias equals entity column
        """
        if self.delivered_alias == self.entity_column:
            raise ValueError(
                f"delivered_alias {self.delivered_alias!r} equals entity_column, which renames nothing; omit it instead"
            )
        return self

    @property
    def delivered_column(self) -> str:
        """name this grain carries in delivered artifacts.

        :returns: delivered alias when set, else entity column
        :rtype: str
        """
        return self.delivered_alias or self.entity_column
