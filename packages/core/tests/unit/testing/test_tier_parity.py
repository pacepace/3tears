"""the tier-parity gate names the collection, column and tier of every value a tier hands back changed.

What it must catch, each pinned with a collection that has the fault: a column an L1 cannot hold
exactly (refused, never kept approximately), and an L2 codec that changes a value's type. And what
it must pass: a collection whose every tier hands back what was written.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from threetears.core.collections.schema_backed import (
    BOOL_TYPE,
    BYTES_TYPE,
    DATETIMETZ_TYPE,
    JSONB_TYPE,
    NUMERIC_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.tier_parity import tier_parity_findings


class _Row(BaseEntity):
    primary_key_field = "id"


def _schema(name: str, precision: int) -> TableSchema:
    return TableSchema(
        name=name,
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("label", STRING_TYPE),
            Column("amount", NUMERIC_TYPE, precision=precision, scale=2),
            Column("seen", DATETIMETZ_TYPE),
            Column("doc", JSONB_TYPE, nullable=True),
            Column("flag", BOOL_TYPE),
            Column("blob", BYTES_TYPE, nullable=True),
        ],
    )


class _Exact(SchemaBackedCollection[_Row]):
    """every tier hands back what was written."""

    schema: ClassVar[TableSchema] = _schema("tier_exact", 12)
    entity_class = _Row  # type: ignore[assignment]

    @property
    def table_name(self) -> str:
        return self.schema.name


class _TooWide(_Exact):
    """a NUMERIC wider than a 64-bit integer holds at scale."""

    schema: ClassVar[TableSchema] = _schema("tier_too_wide", 20)


class _UntypedCodec(_Exact):
    """an L2 codec that hands back JSON's own types: a UUID as text, a Decimal as text."""

    schema: ClassVar[TableSchema] = _schema("tier_untyped", 12)

    def deserialize(self, data: bytes) -> dict[str, Any]:
        decoded: dict[str, Any] = json.loads(data)
        return decoded


def test_a_collection_whose_every_tier_is_exact_passes() -> None:
    assert tier_parity_findings([_Exact]) == []


def test_a_numeric_the_sqlite_l1_cannot_hold_is_named_with_its_collection_column_and_tier() -> None:
    findings = tier_parity_findings([_TooWide])
    assert len(findings) == 1, findings
    assert findings[0].startswith("_TooWide: SQLite L1: tier_too_wide.amount: NUMERIC(20, 2)")


def test_an_l2_codec_that_changes_a_type_is_named_per_column() -> None:
    findings = tier_parity_findings([_UntypedCodec])
    tiers = {finding.split(": ")[0]: finding.split(": ")[1] for finding in findings}
    assert tiers == {"_UntypedCodec.id": "L2", "_UntypedCodec.amount": "L2", "_UntypedCodec.blob": "L2"}, findings
