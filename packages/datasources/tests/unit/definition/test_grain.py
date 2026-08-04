"""unit tests for :class:`GrainSpec`.

one real deliverable renames the grain in its delivered projection
(``voterbase_id AS individual_id``), and the rename propagates into the
physical layout, so the alias is part of the contract rather than a
cosmetic label.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.datasources.definition import GrainSpec


class TestGrainSpec:
    """entity column plus an optional delivered alias."""

    def test_entity_column_only(self) -> None:
        grain = GrainSpec(entity_column="voterbase_id")
        assert grain.delivered_alias is None
        assert grain.delivered_column == "voterbase_id"

    def test_delivered_alias(self) -> None:
        grain = GrainSpec(entity_column="voterbase_id", delivered_alias="individual_id")
        assert grain.delivered_column == "individual_id"

    def test_rejects_empty_entity_column(self) -> None:
        with pytest.raises(ValidationError):
            GrainSpec(entity_column="")

    def test_rejects_alias_equal_to_the_entity_column(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            GrainSpec(entity_column="voterbase_id", delivered_alias="voterbase_id")
        assert "voterbase_id" in str(excinfo.value)

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            GrainSpec.model_validate({"entity_column": "voterbase_id", "record_column": "list_id"})

    def test_round_trips(self) -> None:
        payload = {"entity_column": "voterbase_id", "delivered_alias": "individual_id"}
        assert GrainSpec.model_validate(payload).model_dump() == payload
