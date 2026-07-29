"""expression, comparison, and predicate algebra for a dataset definition.

Three shapes an authored operand can take, and the encoding is
deliberately unambiguous rather than convenient:

- a bare string is ALWAYS a namespaced reference. ``"entity.vb_voterbase_age"``
  is a reference; ``"x"`` is a validation error, not a string literal.
- a string LITERAL is written ``{"literal": "x"}``. Guessing between the
  two on shape would reproduce the one-name-two-semantics trap the
  namespace exists to remove, on operands instead of on aliases.
- an arithmetic operand is written ``{"arith": "70 + param.run_year -
  resolved.record_year"}``. The text is stored verbatim; the compiler
  parses it, and this package only SCANS it for namespaced references so
  the stage guard can see them.

Comparisons carry expressions on BOTH sides, which the working-age filter
requires::

    entity.vb_voterbase_age > 20
    AND entity.vb_voterbase_age < (70 + param.run_year - resolved.record_year)

The operator vocabulary includes ``ILIKE`` and POSIX ``~`` / ``~*``,
because production uses ``ILIKE`` 28 times in one file and regex in three
audiences. The prototype prompt's "never ``ILIKE`` or regex" rule is
deleted rather than carried forward: it was safety theatre over a
string-paste emitter, and the corpus does not obey it.

The operator KEYWORD is a closed vocabulary and is canonicalised on
input, so the corpus's two lowercase ``ilike`` spellings normalise. The
operand LITERAL round-trips byte-exactly, which is what a lossy
re-emission would actually corrupt -- the corpus carries single AND
doubled backslash classes inside one regex literal, and a doubled single
quote inside a string literal.

``Predicate`` carries ``all_of`` / ``any_of`` / ``negate`` / ``compare``
and, from ``dsm-task-01d``, ``membership`` / ``concept`` / ``raw``.
:class:`~threetears.datasources.definition.source.Membership` is defined
in :mod:`~threetears.datasources.definition.source` because it points at
a ``SourceRef``, and that module closes the forward reference. Importing
the ``definition`` package always imports it, so the model is complete by
the time anything can use it.

**THE LITERAL CARRIES ITS TYPE, AND THAT IS A CORRECTNESS PROPERTY.**
Before ``dsm-task-01d`` a literal stored the scalar alone, and JSON
erases the difference between a text ``'5'`` and a decimal ``5``::

    LiteralExpression(literal=Decimal('5'))  ->  {"literal": "5"}  ->  '5'

Precision was never at risk -- ``float`` is excluded from
:data:`ScalarValue` and coerced to ``Decimal``, so a fraction never
passes through a binary float. What was lost is the TYPE, and losing it
costs two things the content hash cannot afford: a definition carrying a
decimal literal comes back structurally different after a store-and-load
cycle, and a text ``'1.0'`` compared against a numeric column -- which a
real committed definition does -- hashes identically to the numeric
``1.0``, so the edit between them mints no version. So
:attr:`LiteralExpression.literal_type` is a stored field, inferred from
the authored scalar when it is not written, and it round-trips.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Self, TypeAlias

from pydantic import BaseModel, BeforeValidator, ConfigDict, model_validator

from threetears.datasources.definition.namespace import Namespace, Reference

if TYPE_CHECKING:
    from threetears.datasources.definition.source import Membership

__all__ = [
    "ArithmeticExpression",
    "Comparison",
    "ComparisonOperator",
    "Expression",
    "LiteralExpression",
    "LiteralType",
    "Predicate",
    "ScalarValue",
    "scan_references",
]

ScalarValue: TypeAlias = bool | int | Decimal | str | None
"""scalar an authored literal may hold.

``Decimal`` rather than ``float`` throughout, so a threshold authored as
``1.5`` survives serialization without binary-float rounding. ``bool``
precedes ``int`` so ``True`` stays a boolean.

Every OTHER field carrying a ``ScalarValue`` -- a parameter default, an
enumeration member, a sentinel value, a swept value -- is declared beside
a :class:`~threetears.datasources.definition.parameters.ParameterType`,
so the type is already stated there and no second tag is needed. The
literal operand is the one place a scalar stands alone, which is why it
carries :class:`LiteralType`.
"""

_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
_TWO_SEGMENT_NAMESPACES = "|".join(member.value for member in Namespace if member is not Namespace.REL)
_REFERENCE_SCANNER = re.compile(
    rf"\brel\.{_IDENTIFIER}\.{_IDENTIFIER}\b|\b(?:{_TWO_SEGMENT_NAMESPACES})\.{_IDENTIFIER}\b"
)


class ComparisonOperator(StrEnum):
    """operator vocabulary a comparison may use.

    Pattern and regex members are present because production uses them:
    ``ILIKE`` 28 times in one file, POSIX ``~`` / ``~*`` in three
    audiences. ``NOT_EQUAL`` and ``NOT_EQUAL_ANSI`` stay distinct members
    so re-emission is faithful to what was authored -- the corpus writes
    both spellings.

    :cvar EQUAL: ``=``
    :cvar NOT_EQUAL: ``!=``
    :cvar NOT_EQUAL_ANSI: ``<>``
    :cvar LESS_THAN: ``<``
    :cvar LESS_OR_EQUAL: ``<=``
    :cvar GREATER_THAN: ``>``
    :cvar GREATER_OR_EQUAL: ``>=``
    :cvar LIKE: ``LIKE``
    :cvar NOT_LIKE: ``NOT LIKE``
    :cvar ILIKE: ``ILIKE``
    :cvar NOT_ILIKE: ``NOT ILIKE``
    :cvar REGEX: POSIX ``~``
    :cvar REGEX_CASE_INSENSITIVE: POSIX ``~*``
    :cvar NOT_REGEX: POSIX ``!~``
    :cvar NOT_REGEX_CASE_INSENSITIVE: POSIX ``!~*``
    :cvar IS_NULL: ``IS NULL``; unary, takes no right operand
    :cvar IS_NOT_NULL: ``IS NOT NULL``; unary, takes no right operand
    """

    EQUAL = "="
    NOT_EQUAL = "!="
    NOT_EQUAL_ANSI = "<>"
    LESS_THAN = "<"
    LESS_OR_EQUAL = "<="
    GREATER_THAN = ">"
    GREATER_OR_EQUAL = ">="
    LIKE = "LIKE"
    NOT_LIKE = "NOT LIKE"
    ILIKE = "ILIKE"
    NOT_ILIKE = "NOT ILIKE"
    REGEX = "~"
    REGEX_CASE_INSENSITIVE = "~*"
    NOT_REGEX = "!~"
    NOT_REGEX_CASE_INSENSITIVE = "!~*"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"


_UNARY_OPERATORS = frozenset({ComparisonOperator.IS_NULL, ComparisonOperator.IS_NOT_NULL})


def scan_references(text: str) -> tuple[Reference, ...]:
    """namespaced references scanned out of an authored SQL-ish string.

    Used by arithmetic operands and by ``raw:`` fragments, so a fragment
    binding ``source.*`` at the qualification stage is refused exactly as
    a typed operand is. The scan over-detects rather than under-detects --
    a namespaced-looking token inside a quoted substring is treated as a
    reference -- because over-detection fails loudly and under-detection
    produces a wrong audience silently.

    :param text: authored text
    :ptype text: str
    :returns: references in first-appearance order, de-duplicated
    :rtype: tuple[Reference, ...]
    """
    seen: dict[str, Reference] = {}
    for match in _REFERENCE_SCANNER.finditer(text):
        found = match.group(0)
        if found not in seen:
            seen[found] = Reference(ref=found)
    return tuple(seen.values())


def _canonicalise_operator(value: object) -> object:
    """upper-case and whitespace-collapse a word operator before validation.

    The corpus authors ``ilike`` and ``ILIKE`` in one file, and writes
    ``IS NULL`` with varying spacing. Symbol operators pass through
    untouched.

    :param value: raw authored operator
    :ptype value: object
    :returns: canonical spelling when recognisable, else value unchanged
    :rtype: object
    """
    canonical: object = value
    if isinstance(value, str):
        collapsed = " ".join(value.split()).upper()
        if collapsed in {member.value for member in ComparisonOperator}:
            canonical = collapsed
    return canonical


class LiteralType(StrEnum):
    """declared type of an authored literal operand.

    Stored rather than inferred at read time, because JSON has no way to
    tell a text ``"5"` from a decimal ``5`` once both are written as the
    same scalar, and the two are different SQL.

    :cvar NULL: SQL ``NULL``
    :cvar BOOLEAN: ``true`` / ``false``
    :cvar INTEGER: whole number
    :cvar DECIMAL: exact decimal; never a binary float
    :cvar TEXT: string literal
    """

    NULL = "null"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    DECIMAL = "decimal"
    TEXT = "text"


def _infer_literal_type(value: object) -> LiteralType:
    """derive the type tag from an authored Python scalar.

    :param value: authored scalar
    :ptype value: object
    :returns: inferred type tag
    :rtype: LiteralType
    :raises ValueError: value is not a scalar a literal may hold
    """
    inferred: LiteralType
    if value is None:
        inferred = LiteralType.NULL
    elif isinstance(value, bool):
        inferred = LiteralType.BOOLEAN
    elif isinstance(value, int):
        inferred = LiteralType.INTEGER
    elif isinstance(value, Decimal | float):
        inferred = LiteralType.DECIMAL
    elif isinstance(value, str):
        inferred = LiteralType.TEXT
    else:
        raise ValueError(f"literal {value!r} is not a scalar; a literal holds null, boolean, integer, decimal, or text")
    return inferred


def _coerce_to_literal_type(value: object, literal_type: LiteralType) -> ScalarValue:
    """coerce an authored scalar onto its declared type tag.

    Reading a stored literal back is the case that matters: JSON hands
    over ``"5"`` for a decimal and for text alike, and only the tag says
    which was meant.

    :param value: authored scalar, possibly the JSON spelling of another type
    :ptype value: object
    :param literal_type: declared type tag
    :ptype literal_type: LiteralType
    :returns: scalar of the declared type
    :rtype: ScalarValue
    :raises ValueError: value cannot carry the declared type
    """
    coerced: ScalarValue
    if literal_type is LiteralType.NULL:
        if value is not None:
            raise ValueError(f"literal_type 'null' carries no value; got {value!r}")
        coerced = None
    elif literal_type is LiteralType.BOOLEAN:
        if not isinstance(value, bool):
            raise ValueError(f"literal_type 'boolean' requires true or false; got {value!r}")
        coerced = value
    elif literal_type is LiteralType.INTEGER:
        if isinstance(value, bool) or not isinstance(value, int | str):
            raise ValueError(f"literal_type 'integer' requires a whole number; got {value!r}")
        coerced = int(value)
    elif literal_type is LiteralType.DECIMAL:
        if isinstance(value, bool) or not isinstance(value, int | float | str | Decimal):
            raise ValueError(f"literal_type 'decimal' requires a number; got {value!r}")
        try:
            coerced = Decimal(str(value))
        except InvalidOperation as error:
            raise ValueError(f"literal_type 'decimal' cannot carry {value!r}") from error
    else:
        if not isinstance(value, str):
            raise ValueError(f"literal_type 'text' requires a string; got {value!r}")
        coerced = value
    return coerced


class LiteralExpression(BaseModel):
    """authored constant operand, carrying its own type.

    Written explicitly so a bare string is never ambiguous between a
    column reference and a data value, and typed explicitly so a stored
    literal is never ambiguous between text and a number.

    :ivar literal: constant value; ``None`` is SQL ``NULL``
    :ivar literal_type: declared type; inferred from :attr:`literal` when
        the authored form omits it, and always written when serialized
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    literal: ScalarValue
    literal_type: LiteralType

    @model_validator(mode="before")
    @classmethod
    def _literal_carries_its_type(cls, value: object) -> object:
        """infer the type tag, or coerce the scalar onto the declared one.

        :param value: raw authored payload
        :ptype value: object
        :returns: payload carrying a scalar and a matching type tag
        :rtype: object
        :raises ValueError: the scalar and the declared type disagree
        """
        payload: object = value
        if isinstance(value, dict) and "literal" in value:
            scalar = value["literal"]
            declared = value.get("literal_type")
            if declared is None:
                payload = {**value, "literal_type": _infer_literal_type(scalar)}
                if isinstance(scalar, float):
                    payload = {**payload, "literal": Decimal(str(scalar))}
            else:
                literal_type = LiteralType(declared)
                payload = {
                    **value,
                    "literal": _coerce_to_literal_type(scalar, literal_type),
                    "literal_type": literal_type,
                }
        return payload

    @property
    def references(self) -> tuple[Reference, ...]:
        """references reachable from this operand.

        :returns: empty tuple; a literal binds nothing
        :rtype: tuple[Reference, ...]
        """
        return ()


class ArithmeticExpression(BaseModel):
    """authored arithmetic operand, carried verbatim.

    This package deliberately does NOT parse the text: ``sqlglot`` is a
    dependency of no 3tears package and the compiler that owns parsing
    lives Hub-side. What happens here is a SCAN for namespaced
    references, so the stage guard can refuse ``source.*`` inside a
    qualification predicate's arithmetic. The scan over-detects rather
    than under-detects -- a namespaced-looking token inside a quoted
    substring is treated as a reference -- because over-detection fails
    loudly and under-detection produces a wrong audience silently.

    :ivar arith: authored arithmetic text, re-emitted byte-exactly
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    arith: str

    @model_validator(mode="after")
    def _arith_is_not_blank(self) -> Self:
        """reject empty arithmetic text.

        :returns: validated expression
        :rtype: ArithmeticExpression
        :raises ValueError: text is blank
        """
        if not self.arith.strip():
            raise ValueError("arith carries no text")
        return self

    @property
    def references(self) -> tuple[Reference, ...]:
        """namespaced references scanned out of the arithmetic text.

        :returns: references in first-appearance order, de-duplicated
        :rtype: tuple[Reference, ...]
        """
        return scan_references(self.arith)


def _coerce_expression(value: object) -> object:
    """map an authored shorthand onto one of three expression members.

    A bare string becomes a reference; a bare scalar becomes a literal;
    a mapping is routed by its own key.

    :param value: raw authored operand
    :ptype value: object
    :returns: value in a shape the expression union can validate
    :rtype: object
    """
    coerced: object = value
    if isinstance(value, str):
        coerced = {"ref": value}
    elif isinstance(value, bool | int | float | Decimal):
        coerced = {"literal": value}
    return coerced


Expression: TypeAlias = Annotated[
    Reference | LiteralExpression | ArithmeticExpression,
    BeforeValidator(_coerce_expression),
]
"""one operand of a comparison: reference, literal, or arithmetic."""


def _expression_references(expression: Expression) -> tuple[Reference, ...]:
    """references reachable from one operand.

    :param expression: operand to walk
    :ptype expression: Expression
    :returns: references in appearance order
    :rtype: tuple[Reference, ...]
    """
    return (expression,) if isinstance(expression, Reference) else expression.references


class Comparison(BaseModel):
    """one comparison, with expressions on both sides.

    Both sides are expressions because the working-age filter requires
    it: ``entity.vb_voterbase_age < (70 + param.run_year -
    resolved.record_year)``.

    :ivar left: left-hand operand
    :ivar right: right-hand operand; ``None`` for unary operators only
    :ivar op: comparison operator
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    left: Expression
    op: Annotated[ComparisonOperator, BeforeValidator(_canonicalise_operator)]
    right: Expression | None = None

    @model_validator(mode="after")
    def _arity_matches_the_operator(self) -> Self:
        """require a right operand for binary operators and forbid one for unary.

        :returns: validated comparison
        :rtype: Comparison
        :raises ValueError: arity does not match operator
        """
        is_unary = self.op in _UNARY_OPERATORS
        if is_unary and self.right is not None:
            raise ValueError(f"{self.op.value!r} is unary and takes no right operand")
        if not is_unary and self.right is None:
            raise ValueError(f"{self.op.value!r} requires a right operand")
        return self

    @property
    def references(self) -> tuple[Reference, ...]:
        """references reachable from both operands.

        :returns: references in left-then-right order
        :rtype: tuple[Reference, ...]
        """
        right = () if self.right is None else _expression_references(self.right)
        return _expression_references(self.left) + right


_PREDICATE_FORMS: tuple[str, ...] = ("all_of", "any_of", "negate", "compare", "membership", "concept", "raw")


class Predicate(BaseModel):
    """recursive boolean expression; exactly one field is set.

    Authored order inside :attr:`all_of` and :attr:`any_of` is preserved,
    because the corpus's emitted SQL preserves it byte-for-byte and parity
    diffs against that.

    :attr:`concept` and :attr:`raw` are the two STRING inputs, and both
    are parsed into an AST and re-emitted rather than pasted. They are
    scanned for namespaced references here so the stage guard sees them --
    a ``raw:`` fragment naming ``source.*`` inside a qualification
    predicate is refused exactly as a typed one is.

    :ivar all_of: conjunction, in authored order
    :ivar any_of: disjunction, in authored order
    :ivar negate: negation of one predicate
    :ivar compare: one comparison
    :ivar membership: ``IN`` / ``NOT IN`` a value list or a source
    :ivar concept: governed concept name, resolving a ``sql_fragment``;
        an audience-local YAML anchor is this made first-class
    :ivar raw: boolean SQL fragment, an escape hatch ``parity-task-03``
        scores
    """

    model_config = ConfigDict(extra="forbid")

    all_of: list[Predicate] | None = None
    any_of: list[Predicate] | None = None
    negate: Predicate | None = None
    compare: Comparison | None = None
    membership: Membership | None = None
    concept: str | None = None
    raw: str | None = None

    @model_validator(mode="after")
    def _exactly_one_form(self) -> Self:
        """require exactly one form, and reject an empty branch or fragment.

        :returns: validated predicate
        :rtype: Predicate
        :raises ValueError: no form or more than one form is set, a branch
            list is empty, or a string form carries no text
        """
        set_fields = [name for name in _PREDICATE_FORMS if getattr(self, name) is not None]
        if len(set_fields) != 1:
            raise ValueError(
                f"predicate carries exactly one of {' / '.join(_PREDICATE_FORMS)}; got {set_fields or 'none'}"
            )
        for name in ("all_of", "any_of"):
            branches = getattr(self, name)
            if branches is not None and not branches:
                raise ValueError(f"{name} carries no branches")
        for name in ("concept", "raw"):
            text = getattr(self, name)
            if text is not None and not text.strip():
                raise ValueError(f"{name} carries no text")
        return self

    @property
    def is_escape_hatch(self) -> bool:
        """whether a ``raw:`` fragment is reachable from this tree.

        ``concept`` is NOT an escape hatch: a governed concept resolves a
        reviewed fragment through the knowledge layer, where a ``raw:``
        string is authored inline and reviewed by nobody.

        :returns: True when any branch carries ``raw``
        :rtype: bool
        """
        nested = [*(self.all_of or ()), *(self.any_of or ())]
        if self.negate is not None:
            nested.append(self.negate)
        return self.raw is not None or any(branch.is_escape_hatch for branch in nested)

    @property
    def references(self) -> tuple[Reference, ...]:
        """every reference reachable from this predicate tree.

        :returns: references in depth-first authored order
        :rtype: tuple[Reference, ...]
        """
        collected: tuple[Reference, ...] = ()
        for branches in (self.all_of, self.any_of):
            for branch in branches or ():
                collected = collected + branch.references
        if self.negate is not None:
            collected = collected + self.negate.references
        if self.compare is not None:
            collected = collected + self.compare.references
        if self.membership is not None:
            collected = collected + self.membership.references
        if self.raw is not None:
            collected = collected + scan_references(self.raw)
        return collected
