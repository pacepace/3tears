"""Mapping identity-provider claims onto local group memberships.

A tenant federates through their own identity provider and wants membership
in a directory group to grant membership in a local one. That mapping is a
pure function over an already-verified claim set: rules in, group references
out, no I/O.

**Default-deny is structural, not a branch.** :func:`resolve_claim_grants`
starts from the empty set and only ever ADDS references for bindings that
positively match. There is no ``else`` that could widen a grant, so a missing
rule, a rule with no bindings, and a claim set missing every mapped claim all
produce the same empty result by construction rather than by three separate
correct decisions. An asserted claim never auto-grants anything.

**The claims must already be verified.** This function trusts its input
completely. Handing it an unverified ``id_token``'s claims hands an attacker
direct control over group membership.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["ClaimBinding", "ClaimMappingRule", "resolve_claim_grants"]


@dataclass(frozen=True, slots=True)
class ClaimBinding:
    """One claim value granting one or more group references.

    :ivar claim: the claim name to read, e.g. ``"groups"`` or ``"hd"``.
    :ivar value: the exact value that must be asserted. Compared exactly -- no prefix, no
        wildcard, no case folding. A pattern here would be a way to accidentally grant far
        more than intended, and the failure would be silent.
    :ivar group_refs: the local group references this binding grants.
    """

    claim: str
    value: str
    group_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ClaimMappingRule:
    """A tenant's full set of bindings.

    :ivar bindings: evaluated independently. Order does not matter -- the result is a union,
        so two bindings granting the same reference is not a conflict.
    """

    bindings: tuple[ClaimBinding, ...]


def resolve_claim_grants(rule: ClaimMappingRule | None, claims: Mapping[str, Any]) -> frozenset[str]:
    """Evaluate ``rule`` against a verified login's asserted claims.

    :param rule: the tenant's rule, or ``None`` when the connection has no mapping. ``None``
        is default-deny, not an error: no mapping configured means no grant, exactly like an
        unmatched value.
    :ptype rule: ClaimMappingRule | None
    :param claims: the VERIFIED upstream claim set (module docstring).
    :ptype claims: Mapping[str, Any]
    :return: the group references this login resolves to. Empty when nothing matches; never
        raises.
    :rtype: frozenset[str]
    """
    if rule is None:
        return frozenset()
    resolved: set[str] = set()
    for binding in rule.bindings:
        if binding.value in _asserted_values(claims.get(binding.claim)):
            resolved.update(binding.group_refs)
    return frozenset(resolved)


def _asserted_values(raw: Any) -> frozenset[str]:
    """Normalize a claim's value to a set of strings.

    Most claims are scalars; some -- an Entra ``groups`` claim, say -- assert a list.
    Treating a scalar as a one-element set lets one matching path handle both instead of a
    branch per claim shape, which is where a "handles lists" bug would eventually live.

    A ``str`` is deliberately checked BEFORE the sequence case: a string IS a sequence, and
    iterating one would match a single CHARACTER against a binding value.
    """
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        return frozenset({raw})
    if isinstance(raw, Sequence | set | frozenset):
        return frozenset(str(value) for value in raw)
    return frozenset({str(raw)})
