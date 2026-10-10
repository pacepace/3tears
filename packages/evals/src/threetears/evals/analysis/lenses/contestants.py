"""The contestant key the frontier and the history share, and the identity-version disclosures that go with it.

A contestant is one (variant, identity version): :func:`_contestant_key` keys a placed result on it, and
:func:`_identity_version_span` / :func:`_identity_span_disclosure` say when a lens pooled several identity versions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from threetears.evals.kernel.identity import IDENTITY_VERSION

if TYPE_CHECKING:
    from threetears.evals.schema.models import (
        EvalResult,
    )


#: What both lenses group on. The predicate version is IN the key rather than a filter
#: applied beside it, because the join this gate protects *is* the grouping dict — a gate
#: keyed on a projection of the group key admits exactly the class it exists to refuse,
#: and the projected-away field is the gap.
ContestantKey = tuple[str, int]


def _contestant_key(result: EvalResult) -> ContestantKey:
    """Key a result to its contestant, and to the identity version stamped beside that key.

    One key, both lenses. ``frontier`` and ``history`` rank and trend the same
    contestants, so a grouping policy that lived in each of them separately would be
    two authorings of one rule — which is how the version gate came to exist on the
    bundle path and not here.

    The comparison this key performs is **equality, and there is no ``>=`` reading**.
    Keys stamped at different identity versions are incomparable in both directions: an
    older key is not "good enough" for a newer reader, and a newer key is not admissible
    to an older series. That is what makes this a partition rather than a compatibility
    check, and it is why a version mismatch is never resolved in one side's favour.

    **The stamp is one counter over two predicates, so it is a conservative
    discriminator rather than a precise one.** ``IDENTITY_VERSION`` backs the variant and
    context predicates together, so a bump on the CONTEXT side moves the number stamped
    beside a ``variant_key`` whose own predicate did not change — v9 and v10 are both that
    case. Such a pair carries a byte-identical digest under two stamps and is partitioned
    here anyway. That is deliberate and errs the safe way: the stamp cannot say WHICH
    predicate moved, and merging on a digest whose provenance is unverifiable is the wrong
    direction, since nothing downstream undoes a wrong merge. What it costs is a split that
    is sometimes narrower than necessary, which the disclosures state honestly rather than
    dressing up as a predicate change.

    Args:
        result: The observation to place.

    Returns:
        ``(variant_key, identity_version)`` — the key the runner stamped and the predicate
        version stamped beside it.
    """
    return (result.variant_key, result.identity_version)


def _identity_version_disclosure(version: int) -> str | None:
    """Say so when a contestant's key was minted by a predicate this build does not use.

    Derived from the same equality :func:`_contestant_key` groups on, never from a
    correlate: a sentence that agreed with the grouping on the observations we happen
    to have and disagreed on the one it exists for would be worse than none.

    Args:
        version: The predicate version stamped on the contestant's key.

    Returns:
        The sentence, or ``None`` when the key is this build's.
    """
    if version == IDENTITY_VERSION:
        return None
    return (
        f"stamped at identity version v{version}, not this build's v{IDENTITY_VERSION} — "
        "ranked separately from contestants stamped at a different version rather than pooled with them. "
        "The stamp does not say which predicate moved, so this may be a narrower split than the change warranted"
    )


def _identity_version_span(results: Iterable[EvalResult]) -> list[int]:
    """The distinct identity versions the KEYED observations in an answer were stamped at.

    Args:
        results: Every observation the answer is built over. Pass the placed rows
            rather than the assembled points: an answer whose subjects were all
            filtered out still spanned what it spanned, and a span derived downstream
            goes missing exactly when the reader most needs it.

    Returns:
        The versions, ascending.
    """
    return sorted({r.identity_version for r in results})


def _identity_span_disclosure(versions: Sequence[int]) -> str | None:
    """Explain a corpus that spans identity versions, so a doubled row is not a mystery.

    Partitioning stops two stampings of one stack being ranked as rival
    contestants; it does not stop them APPEARING as two rows. Without this sentence a
    reader meets one model twice, under two digests, with nothing saying why — which is
    the second half of the same defect and the one a per-row flag cannot answer.

    Args:
        versions: The span from :func:`_identity_version_span`.

    Returns:
        The sentence, or ``None`` when the answer rests on a single predicate — where
        there is no split to explain.
    """
    if len(versions) < 2:
        return None
    named = ", ".join(f"v{v}" for v in versions)
    return (
        f"spans identity versions {named}. Two keys stamped at different versions cannot be shown FROM THE "
        "STAMP ALONE to describe the same contestant, so they are listed separately — one model appearing "
        "more than once here is that split, not two rival configurations. The stamp covers both the variant "
        "and the context predicate, so some of these splits are narrower than the change that caused them."
    )


__all__ = [
    "ContestantKey",
]
