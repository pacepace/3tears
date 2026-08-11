"""Shared pydantic base for every wire-crossing contract type.

One config, stated once: contract instances are frozen (a result is a fact,
not a workspace -- later stages derive new instances with ``model_copy``),
and unknown fields are rejected so a shape mismatch fails at the border
instead of surviving as silently-dropped data. Openness where the spec
requires it (criteria vocabulary, facets, scores) is carried by explicitly
open *fields*, never by tolerating unknown *fields on the type* (SR-B1,
SR-C1 vs D13's additive-within-a-minor rule -- the family versions in
lockstep, so both readers of a payload share a minor).
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

__all__ = ["ContractModel"]


class ContractModel(BaseModel):
    """Base class for all ``threetears.search.contracts`` payload types.

    Every subclass JSON round-trips (SR-L4): no callables, open files, or
    port objects may appear in any field -- ports are parameters, never
    payload.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: field names whose element order is canonically irrelevant -- consumed
    #: by :func:`threetears.search.contracts.canonicalize` so that two
    #: requests differing only in, e.g., criteria order share one canonical
    #: form (D26, SR-F1).
    CANONICAL_ORDER_INSENSITIVE: ClassVar[frozenset[str]] = frozenset()

    #: field names that never participate in canonical serialization --
    #: consumed by :func:`threetears.search.contracts.canonicalize`.
    #: Operational fields say how one invocation was wired (whether it
    #: recorded, which budget scopes it debited), not which search it is;
    #: keying them into the digest would give every recording a key no
    #: later replay can derive (SR-F7) and every eval run a unique identity
    #: (SR-F1). Only the semantic parameters key (Gate A, 2026-08-10).
    CANONICAL_EXCLUDED: ClassVar[frozenset[str]] = frozenset()
