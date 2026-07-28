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

``Predicate`` here carries ``all_of`` / ``any_of`` / ``negate`` /
``compare``. ``membership``, ``concept``, and ``raw`` need
``ArtifactRef`` and the concept layer and land in ``dsm-task-01d``;
``extra="forbid"`` means an early attempt to author one fails loudly
instead of being dropped.
"""

from __future__ import annotations

import re
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Self, TypeAlias

from pydantic import BaseModel, BeforeValidator, ConfigDict, model_validator

from threetears.datasources.definition.namespace import Namespace, Reference

__all__ = [
    "ArithmeticExpression",
    "Comparison",
    "ComparisonOperator",
    "Expression",
    "LiteralExpression",
    "Predicate",
    "ScalarValue",
]

ScalarValue: TypeAlias = bool | int | Decimal | str | None
"""scalar an authored literal may hold.

``Decimal`` rather than ``float`` throughout, so a threshold authored as
``1.5`` survives serialization without binary-float rounding. ``bool``
precedes ``int`` so ``True`` stays a boolean.
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


class LiteralExpression(BaseModel):
    """authored constant operand.

    Written explicitly so a bare string is never ambiguous between a
    column reference and a data value.

    :ivar literal: constant value; ``None`` is SQL ``NULL``
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    literal: ScalarValue

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
        seen: dict[str, Reference] = {}
        for match in _REFERENCE_SCANNER.finditer(self.arith):
            text = match.group(0)
            if text not in seen:
                seen[text] = Reference(ref=text)
        return tuple(seen.values())


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


class Predicate(BaseModel):
    """recursive boolean expression; exactly one field is set.

    Authored order inside :attr:`all_of` and :attr:`any_of` is preserved,
    because the corpus's emitted SQL preserves it byte-for-byte and parity
    diffs against that.

    ``membership``, ``concept``, and ``raw`` are named by design section 7
    and land in ``dsm-task-01d`` with ``ArtifactRef`` and the concept
    layer. Until then ``extra="forbid"`` refuses them loudly.

    :ivar all_of: conjunction, in authored order
    :ivar any_of: disjunction, in authored order
    :ivar negate: negation of one predicate
    :ivar compare: one comparison
    """

    model_config = ConfigDict(extra="forbid")

    all_of: list[Predicate] | None = None
    any_of: list[Predicate] | None = None
    negate: Predicate | None = None
    compare: Comparison | None = None

    @model_validator(mode="after")
    def _exactly_one_form(self) -> Self:
        """require exactly one form, and reject an empty branch list.

        :returns: validated predicate
        :rtype: Predicate
        :raises ValueError: no form or more than one form is set, or a
            branch list is empty
        """
        set_fields = [name for name in ("all_of", "any_of", "negate", "compare") if getattr(self, name) is not None]
        if len(set_fields) != 1:
            raise ValueError(
                f"predicate carries exactly one of all_of / any_of / negate / compare; got {set_fields or 'none'}"
            )
        for name in ("all_of", "any_of"):
            branches = getattr(self, name)
            if branches is not None and not branches:
                raise ValueError(f"{name} carries no branches")
        return self

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
        return collected
