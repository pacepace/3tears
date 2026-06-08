"""column, index, foreign key, and table definition models for agent data layer."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator, model_validator

from threetears.observe import get_logger

__all__ = [
    "ColumnDef",
    "ForeignKeyDef",
    "IndexDef",
    "TableDef",
]

log = get_logger(__name__)

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

_ALLOWED_COLUMN_TYPES = frozenset(
    {
        "text",
        "integer",
        "bigint",
        "boolean",
        "timestamp",
        "uuid",
        "jsonb",
        "decimal",
        "bytea",
        "vector",
    }
)

_ALLOWED_REFERENTIAL_ACTIONS = frozenset(
    {
        "CASCADE",
        "SET NULL",
        "RESTRICT",
        "NO ACTION",
    }
)


def _validate_identifier(value: str, label: str) -> str:
    """validate SQL identifier against safe pattern.

    :param value: identifier string to validate
    :ptype value: str
    :param label: human-readable label for error messages
    :ptype label: str
    :return: validated identifier string
    :rtype: str
    :raises ValueError: if identifier does not match allowed pattern
    """
    if not _IDENTIFIER_PATTERN.match(value):
        msg = f"{label} must match ^[a-z][a-z0-9_]*$, got: {value!r}"
        raise ValueError(msg)
    return value


class ColumnDef(BaseModel):
    """column definition for agent table creation.

    :param name: column identifier matching ^[a-z][a-z0-9_]*$
    :ptype name: str
    :param column_type: PostgreSQL column type from allowed set
    :ptype column_type: str
    :param nullable: whether column allows NULL values
    :ptype nullable: bool
    :param default: SQL default expression, None for no default
    :ptype default: str | None
    :param primary_key: whether column is part of primary key
    :ptype primary_key: bool
    :param vector_dim: pgvector dimension; required for (and only valid
        with) ``column_type == "vector"``
    :ptype vector_dim: int | None
    """

    name: str
    column_type: str
    nullable: bool = True
    default: str | None = None
    primary_key: bool = False
    vector_dim: int | None = Field(default=None, gt=0)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """validate column name against safe identifier pattern.

        :param value: column name to validate
        :ptype value: str
        :return: validated column name
        :rtype: str
        :raises ValueError: if name does not match allowed pattern
        """
        result = _validate_identifier(value, "column name")
        return result

    @field_validator("column_type")
    @classmethod
    def validate_column_type(cls, value: str) -> str:
        """validate column_type against allowed set.

        :param value: column type string to validate
        :ptype value: str
        :return: validated column type string
        :rtype: str
        :raises ValueError: if column_type not in allowed set
        """
        if value not in _ALLOWED_COLUMN_TYPES:
            msg = f"column_type must be one of {sorted(_ALLOWED_COLUMN_TYPES)}, got: {value!r}"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def validate_vector_dim(self) -> ColumnDef:
        """cross-validate vector_dim against column_type.

        :return: validated column definition
        :rtype: ColumnDef
        :raises ValueError: if vector_dim is missing for a vector column
            or present on a non-vector column
        """
        if self.column_type == "vector" and self.vector_dim is None:
            msg = f"column {self.name!r}: column_type 'vector' requires vector_dim to be set"
            raise ValueError(msg)
        if self.column_type != "vector" and self.vector_dim is not None:
            msg = f"column {self.name!r}: vector_dim is only valid with column_type 'vector'"
            raise ValueError(msg)
        return self


class IndexDef(BaseModel):
    """index definition for agent table.

    :param name: index identifier matching ^[a-z][a-z0-9_]*$
    :ptype name: str
    :param columns: list of column names included in index
    :ptype columns: list[str]
    :param unique: whether index enforces uniqueness
    :ptype unique: bool
    """

    name: str
    columns: list[str]
    unique: bool = False

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """validate index name against safe identifier pattern.

        :param value: index name to validate
        :ptype value: str
        :return: validated index name
        :rtype: str
        :raises ValueError: if name does not match allowed pattern
        """
        result = _validate_identifier(value, "index name")
        return result

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, value: list[str]) -> list[str]:
        """validate all column names in index definition.

        :param value: list of column names to validate
        :ptype value: list[str]
        :return: validated list of column names
        :rtype: list[str]
        :raises ValueError: if any column name does not match allowed pattern
        """
        for col in value:
            _validate_identifier(col, "column name")
        return value


class ForeignKeyDef(BaseModel):
    """foreign key constraint definition.

    :param name: constraint identifier matching ^[a-z][a-z0-9_]*$
    :ptype name: str
    :param columns: list of source column names
    :ptype columns: list[str]
    :param references_table: target table name (must be in same agent schema)
    :ptype references_table: str
    :param references_columns: list of target column names
    :ptype references_columns: list[str]
    :param on_delete: referential action on delete (CASCADE, SET NULL, RESTRICT, NO ACTION)
    :ptype on_delete: str
    :param on_update: referential action on update (CASCADE, SET NULL, RESTRICT, NO ACTION)
    :ptype on_update: str
    """

    name: str
    columns: list[str]
    references_table: str
    references_columns: list[str]
    on_delete: str = "CASCADE"
    on_update: str = "NO ACTION"

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """validate foreign key name against safe identifier pattern.

        :param value: foreign key name to validate
        :ptype value: str
        :return: validated foreign key name
        :rtype: str
        :raises ValueError: if name does not match allowed pattern
        """
        result = _validate_identifier(value, "foreign key name")
        return result

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, value: list[str]) -> list[str]:
        """validate all source column names in foreign key definition.

        :param value: list of column names to validate
        :ptype value: list[str]
        :return: validated list of column names
        :rtype: list[str]
        :raises ValueError: if any column name does not match allowed pattern
        """
        for col in value:
            _validate_identifier(col, "column name")
        return value

    @field_validator("references_table")
    @classmethod
    def validate_references_table(cls, value: str) -> str:
        """validate references_table against safe identifier pattern.

        :param value: referenced table name to validate
        :ptype value: str
        :return: validated table name
        :rtype: str
        :raises ValueError: if table name does not match allowed pattern
        """
        result = _validate_identifier(value, "references_table")
        return result

    @field_validator("references_columns")
    @classmethod
    def validate_references_columns(cls, value: list[str]) -> list[str]:
        """validate all target column names in foreign key definition.

        :param value: list of referenced column names to validate
        :ptype value: list[str]
        :return: validated list of column names
        :rtype: list[str]
        :raises ValueError: if any column name does not match allowed pattern
        """
        for col in value:
            _validate_identifier(col, "references column name")
        return value

    @field_validator("on_delete")
    @classmethod
    def validate_on_delete(cls, value: str) -> str:
        """validate on_delete against allowed referential actions.

        :param value: referential action string to validate
        :ptype value: str
        :return: validated referential action string
        :rtype: str
        :raises ValueError: if action not in allowed set
        """
        if value not in _ALLOWED_REFERENTIAL_ACTIONS:
            msg = f"on_delete must be one of {sorted(_ALLOWED_REFERENTIAL_ACTIONS)}, got: {value!r}"
            raise ValueError(msg)
        return value

    @field_validator("on_update")
    @classmethod
    def validate_on_update(cls, value: str) -> str:
        """validate on_update against allowed referential actions.

        :param value: referential action string to validate
        :ptype value: str
        :return: validated referential action string
        :rtype: str
        :raises ValueError: if action not in allowed set
        """
        if value not in _ALLOWED_REFERENTIAL_ACTIONS:
            msg = f"on_update must be one of {sorted(_ALLOWED_REFERENTIAL_ACTIONS)}, got: {value!r}"
            raise ValueError(msg)
        return value


class TableDef(BaseModel):
    """complete table definition for agent schema creation.

    :param name: table identifier matching ^[a-z][a-z0-9_]*$
    :ptype name: str
    :param columns: list of column definitions
    :ptype columns: list[ColumnDef]
    :param indexes: list of index definitions
    :ptype indexes: list[IndexDef]
    :param foreign_keys: list of foreign key constraint definitions
    :ptype foreign_keys: list[ForeignKeyDef]
    """

    name: str
    columns: list[ColumnDef]
    indexes: list[IndexDef] = Field(default_factory=list)
    foreign_keys: list[ForeignKeyDef] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """validate table name against safe identifier pattern.

        :param value: table name to validate
        :ptype value: str
        :return: validated table name
        :rtype: str
        :raises ValueError: if name does not match allowed pattern
        """
        result = _validate_identifier(value, "table name")
        return result
