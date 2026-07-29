"""what a definition emits, how it is laid out, and what is delivered.

Three properties here are decisions rather than shapes.

**Which artifacts a definition emits is not universal, and neither is any
artifact's column set.** One shipped audience's long table has no
``list_id`` where the template's has it -- and an expansion edge joins on
it -- while another carries a rollup label in the same position. So
:class:`ArtifactSpec` declares both, per artifact.

**Physical layout attaches to ANY materialized artifact, not only the
delivered one.** One committed audience carries a ``DISTKEY`` and
``SORTKEY`` on its *qualified* table, hand-edited into the tool's own
generated output -- which is the tell that the tool should have owned
them.

**THE ``ENCODE`` STRATEGY, DECIDED.** Redshift's ``CREATE TABLE AS``
grammar accepts ``DISTSTYLE``, ``DISTKEY``, ``SORTKEY`` and ``BACKUP``
and has **no column-level ``ENCODE`` clause**, so an artifact declaring
per-column encodings cannot be produced by the single CTAS the rest of
the design assumes. The choice is delegated here and it is made:

    an artifact declaring encodings is materialized by an explicit
    ``CREATE TABLE ... ENCODE`` followed by ``INSERT INTO ... SELECT``;
    an artifact declaring none is materialized by one CTAS.

Not a follow-up ``ALTER TABLE ... ALTER COLUMN ... ENCODE``: support for
it is version- and encoding-dependent, it rewrites the table anyway, and
a partially-encoded table between the CTAS and the alter is a state the
reaper's inventory has no name for. :attr:`ArtifactSpec.materialization_strategy`
states which of the two a given artifact needs, so ``dsc-task-04`` emits
it and ``parity-task-01`` diffs it rather than each re-deriving the rule.

**``DerivedColumn.expression`` stays OPEN, and that is considered rather
than lazy.** A closed union of kinds -- aggregate, flag, ranked category
-- cannot express "concatenate two labels, trim the result, then classify
on the exact value of that concatenation", which is one real shipped
deliverable. Do not tighten it.

**Grants and physical layout are NOT hashed** (D1, design section 7).
They are policy and tuning rather than semantics and they live on the
dataset record beside the definition, so revoking a group's read access
or changing a ``SORTKEY`` must not mint a version and must not fire
``definition_drift`` over every historical run.

:class:`GrantSpec` is a WAREHOUSE grant and nothing else. Per D21 a
definition carries no ``visibility`` and no dataset grant: those are
platform state, the platform row governs who may REFERENCE a dataset, and
the warehouse grant is what actually permits ``SELECT``. The two are
halves of one decision and must not drift, but only one of them is
definition content.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from threetears.datasources.definition.expression import Expression
from threetears.datasources.definition.namespace import Reference
from threetears.datasources.definition.source import ArtifactKind, ArtifactRef

__all__ = [
    "ArtifactSpec",
    "ColumnEncoding",
    "DeliverySpec",
    "DerivedColumn",
    "DistStyle",
    "GrantSpec",
    "GranteeKind",
    "MaterializationStrategy",
    "PhysicalLayout",
    "WarehousePrivilege",
]


class DistStyle(StrEnum):
    """Redshift distribution style.

    :cvar EVEN: round-robin; carries no distribution key
    :cvar KEY: distributed on one column, which must be declared
    :cvar ALL: full copy on every node
    :cvar AUTO: left to the warehouse
    """

    EVEN = "even"
    KEY = "key"
    ALL = "all"
    AUTO = "auto"


class ColumnEncoding(StrEnum):
    """Redshift per-column compression encoding.

    The four the corpus's one fully-specified table uses are ``raw``,
    ``lzo`` twice, and ``az64``; the rest are carried because an encoding
    the model cannot name forces the whole layout into a raw string.

    :cvar RAW: no compression
    :cvar AZ64: proprietary encoding for numeric and date types
    :cvar BYTEDICT: dictionary of up to 256 values
    :cvar DELTA: differences between consecutive values
    :cvar DELTA32K: wider delta
    :cvar LZO: general-purpose, favoured for varchar
    :cvar MOSTLY8: narrow values in a wide column
    :cvar MOSTLY16: narrow values in a wide column
    :cvar MOSTLY32: narrow values in a wide column
    :cvar RUNLENGTH: runs of one repeated value
    :cvar TEXT255: dictionary of the 245 most common words
    :cvar TEXT32K: dictionary of common words up to 32K
    :cvar ZSTD: general-purpose, high ratio
    """

    RAW = "raw"
    AZ64 = "az64"
    BYTEDICT = "bytedict"
    DELTA = "delta"
    DELTA32K = "delta32k"
    LZO = "lzo"
    MOSTLY8 = "mostly8"
    MOSTLY16 = "mostly16"
    MOSTLY32 = "mostly32"
    RUNLENGTH = "runlength"
    TEXT255 = "text255"
    TEXT32K = "text32k"
    ZSTD = "zstd"


class MaterializationStrategy(StrEnum):
    """how one artifact is materialized, given its declared layout.

    Decided by DSM-01D-12 rather than by the emitter, because
    ``dsc-task-04`` emits it and ``parity-task-01`` diffs it and an
    undecided field blocks both.

    :cvar CTAS: one ``CREATE TABLE AS SELECT``, carrying ``DISTSTYLE`` /
        ``DISTKEY`` / ``SORTKEY`` / ``BACKUP``
    :cvar CREATE_THEN_INSERT: ``CREATE TABLE ... ENCODE`` then ``INSERT
        INTO ... SELECT``, because CTAS has no column-level ``ENCODE``
    """

    CTAS = "ctas"
    CREATE_THEN_INSERT = "create_then_insert"


class GranteeKind(StrEnum):
    """what a warehouse grant is issued to.

    Both members are required: the corpus grants to a GROUP at five
    template sites, and a per-user grant is what a hand-delivered audience
    is read through.

    :cvar USER: one warehouse user
    :cvar GROUP: a warehouse group
    """

    USER = "user"
    GROUP = "group"


class WarehousePrivilege(StrEnum):
    """privilege one grant carries.

    :cvar SELECT: read
    :cvar INSERT: append
    :cvar UPDATE: modify
    :cvar DELETE: remove
    :cvar ALL: every privilege on the object
    """

    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    ALL = "all"


class PhysicalLayout(BaseModel):
    """distribution, sort, and per-column encodings for one table.

    :ivar diststyle: distribution style
    :ivar distkey: distribution column, required by and only by
        :attr:`DistStyle.KEY`
    :ivar sortkey: sort columns, in declared order
    :ivar encodings: per-column compression; declaring any of these
        chooses :attr:`MaterializationStrategy.CREATE_THEN_INSERT`
    :ivar backup: whether the table is included in cluster snapshots
    """

    model_config = ConfigDict(extra="forbid")

    diststyle: DistStyle | None = None
    distkey: str | None = Field(default=None, min_length=1)
    sortkey: list[str] = Field(default_factory=list)
    encodings: dict[str, ColumnEncoding] = Field(default_factory=dict)
    backup: bool = True

    @model_validator(mode="after")
    def _layout_is_coherent(self) -> Self:
        """reject a distribution style and key that disagree.

        :returns: validated layout
        :rtype: PhysicalLayout
        :raises ValueError: nothing is declared, a key style carries no
            key, a non-key style carries one, or a sort column repeats
        """
        if not (self.diststyle or self.distkey or self.sortkey or self.encodings or not self.backup):
            raise ValueError("layout declares nothing; omit it rather than authoring an empty one")
        if self.diststyle is DistStyle.KEY and self.distkey is None:
            raise ValueError("diststyle 'key' distributes on a column; declare distkey")
        if self.diststyle is not None and self.diststyle is not DistStyle.KEY and self.distkey is not None:
            raise ValueError(f"diststyle {self.diststyle.value!r} distributes on no column; drop distkey")
        if len(self.sortkey) != len(set(self.sortkey)):
            raise ValueError(f"sortkey names a column twice: {self.sortkey!r}")
        if any(not column.strip() for column in self.sortkey):
            raise ValueError("sortkey carries a blank column name")
        return self

    @property
    def materialization_strategy(self) -> MaterializationStrategy:
        """how a table under this layout is created.

        :returns: two statements when encodings are declared, else one CTAS
        :rtype: MaterializationStrategy
        """
        return MaterializationStrategy.CREATE_THEN_INSERT if self.encodings else MaterializationStrategy.CTAS

    @property
    def columns_named(self) -> tuple[str, ...]:
        """every column this layout names.

        :returns: distribution, sort, and encoded columns, de-duplicated
        :rtype: tuple[str, ...]
        """
        named: dict[str, None] = {}
        if self.distkey is not None:
            named[self.distkey] = None
        for column in (*self.sortkey, *self.encodings):
            named[column] = None
        return tuple(named)


class ArtifactSpec(BaseModel):
    """one artifact this definition emits, and its own column set.

    :ivar artifact: which artifact
    :ivar columns: this artifact's columns, in projection order
    :ivar grain: columns one row is unique on, or empty when undeclared
    :ivar delivered: whether this is the artifact handed to the client
    :ivar layout: physical layout; NOT hashed
    """

    model_config = ConfigDict(extra="forbid")

    hash_excluded_fields: ClassVar[frozenset[str]] = frozenset({"layout"})

    artifact: ArtifactKind
    columns: list[str] = Field(min_length=1)
    grain: list[str] = Field(default_factory=list)
    delivered: bool = False
    layout: PhysicalLayout | None = None

    @model_validator(mode="after")
    def _columns_grain_and_layout_agree(self) -> Self:
        """reject a repeated column and a grain or layout naming an absent one.

        :returns: validated artifact
        :rtype: ArtifactSpec
        :raises ValueError: a column is blank or repeats, or a grain key
            or layout column is outside the declared column set
        """
        if any(not column.strip() for column in self.columns):
            raise ValueError(f"{self.artifact.value}: columns carries a blank name")
        if len(self.columns) != len(set(self.columns)):
            raise ValueError(f"{self.artifact.value}: columns names a column twice")
        declared = set(self.columns)
        outside = [key for key in self.grain if key not in declared]
        if outside:
            raise ValueError(f"{self.artifact.value}: grain names {outside}, which this artifact does not project")
        if len(self.grain) != len(set(self.grain)):
            raise ValueError(f"{self.artifact.value}: grain repeats a key")
        if self.layout is not None:
            unlaid = [column for column in self.layout.columns_named if column not in declared]
            if unlaid:
                raise ValueError(
                    f"{self.artifact.value}: layout names {unlaid}, which this artifact does not "
                    "project. a DISTKEY on an absent column fails at create time, after the build "
                    "has already run"
                )
        return self

    @property
    def materialization_strategy(self) -> MaterializationStrategy:
        """how this artifact's table is created.

        :returns: the declared layout's strategy, or one CTAS when no
            layout is declared
        :rtype: MaterializationStrategy
        """
        return MaterializationStrategy.CTAS if self.layout is None else self.layout.materialization_strategy


class GrantSpec(BaseModel):
    """one WAREHOUSE grant issued over this definition's artifacts.

    Not a platform visibility or ``dataset_customers`` row (D21): this
    grant is what actually permits ``SELECT`` in the warehouse, and the
    platform row governing who may reference the dataset lives on the
    dataset record beside the definition.

    :ivar privilege: privilege granted
    :ivar grantee_kind: whether the grantee is a user or a group
    :ivar grantee: warehouse user or group name
    :ivar artifacts: artifacts covered; empty covers every emitted one,
        which is what the corpus's grant at every stage means
    """

    model_config = ConfigDict(extra="forbid")

    privilege: WarehousePrivilege = WarehousePrivilege.SELECT
    grantee_kind: GranteeKind
    grantee: str = Field(min_length=1)
    artifacts: list[ArtifactKind] = Field(default_factory=list)

    @model_validator(mode="after")
    def _grantee_and_scope_are_coherent(self) -> Self:
        """reject a blank grantee and a repeated artifact.

        :returns: validated grant
        :rtype: GrantSpec
        :raises ValueError: the grantee is blank or an artifact repeats
        """
        if not self.grantee.strip():
            raise ValueError("grantee carries no text")
        if len(self.artifacts) != len(set(self.artifacts)):
            raise ValueError(f"{self.grantee}: artifacts names an artifact twice")
        return self

    def covers(self, artifact: ArtifactKind) -> bool:
        """whether this grant applies to one artifact.

        :param artifact: artifact being checked
        :ptype artifact: ArtifactKind
        :returns: True when the grant names it, or names none at all
        :rtype: bool
        """
        return not self.artifacts or artifact in self.artifacts


class DerivedColumn(BaseModel):
    """one delivered column computed over an artifact.

    :attr:`expression` is deliberately OPEN rather than a closed union of
    aggregate / flag / ranked-category kinds, because one real deliverable
    concatenates two labels, trims the result, and a second classifies on
    the exact value of that concatenation. Tightening this makes that
    deliverable inexpressible and pushes it into ``raw:``.

    :ivar name: delivered column name
    :ivar sql_type: declared width; part of the delivered contract, since
        one shipped table sizes two columns to their exact category
        strings
    :ivar expression: what the column carries
    :ivar over: artifact it is computed against, or ``None`` for the
        delivered one
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    sql_type: str | None = Field(default=None, min_length=1)
    expression: Expression
    over: ArtifactRef | None = None

    @model_validator(mode="after")
    def _name_is_not_blank(self) -> Self:
        """reject a name that is only whitespace.

        :returns: validated column
        :rtype: DerivedColumn
        :raises ValueError: name carries no text
        """
        if not self.name.strip():
            raise ValueError("derived column name carries no text")
        return self

    @property
    def references(self) -> tuple[Reference, ...]:
        """references reachable from this column's expression.

        :returns: references in appearance order
        :rtype: tuple[Reference, ...]
        """
        return (self.expression,) if isinstance(self.expression, Reference) else self.expression.references


class DeliverySpec(BaseModel):
    """derived columns, warehouse grants, and the delivered table's layout.

    :attr:`grants` and :attr:`layout` are declared hash-excluded, and the
    exclusion is applied BEFORE serialization rather than filtered out of
    a digest afterwards. A nested occurrence missed by a filter shows up
    months later as spurious drift over every historical run.

    :ivar columns: delivered derived columns, in authored order
    :ivar grants: warehouse grants; NOT hashed
    :ivar layout: physical layout of the delivered artifact; NOT hashed
    """

    model_config = ConfigDict(extra="forbid")

    hash_excluded_fields: ClassVar[frozenset[str]] = frozenset({"grants", "layout"})

    columns: list[DerivedColumn] = Field(default_factory=list)
    grants: list[GrantSpec] = Field(default_factory=list)
    layout: PhysicalLayout | None = None

    @model_validator(mode="after")
    def _names_are_distinct(self) -> Self:
        """reject a repeated delivered column or a repeated grantee.

        :returns: validated delivery
        :rtype: DeliverySpec
        :raises ValueError: a delivered column name or a grantee repeats
        """
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("delivery names a delivered column twice")
        grantees = [(grant.grantee_kind, grant.grantee, grant.privilege) for grant in self.grants]
        if len(grantees) != len(set(grantees)):
            raise ValueError("delivery grants the same privilege to one grantee twice")
        return self
