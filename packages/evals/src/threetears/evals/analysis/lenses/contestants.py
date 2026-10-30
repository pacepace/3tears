"""The contestant key the frontier and the history share, and the identity-version disclosures that go with it.

A contestant is one (variant, identity version): :func:`contestant_key` keys a placed result on it, and
:func:`identity_version_span` / :func:`identity_span_disclosure` say when a lens spans several identity versions. The
frontier and the history both import these, so they are this module's API.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from threetears.evals.kernel.identity import IDENTITY_VERSION, compute_variant_key

if TYPE_CHECKING:
    from threetears.evals.kernel.result_reading import ResultView
    from threetears.evals.schema.models import (
        EvalRun,
    )


#: What both lenses group on. The predicate version is IN the key rather than a filter
#: applied beside it, because the join this gate protects *is* the grouping dict — a gate
#: keyed on a projection of the group key admits exactly the class it exists to refuse,
#: and the projected-away field is the gap.
ContestantKey = tuple[str, int]


def contestant_key(result: ResultView, run: EvalRun) -> ContestantKey:
    """Key a result to its contestant, and to the identity version the contestant is placed at.

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
    beside a ``variant_key`` whose own predicate did not change. Such a pair carries a byte-identical digest under two
    stamps and is partitioned here anyway. That is deliberate and errs the safe way: the stamp cannot say WHICH
    predicate moved, and merging on a digest whose provenance is unverifiable is the wrong
    direction, since nothing downstream undoes a wrong merge. What it costs is a split that
    is sometimes narrower than necessary, which the disclosures state honestly rather than
    dressing up as a predicate change.

    **Except where the run's own record verifies the key under this build.** A run records the
    lever map its results' key was digested from (``EvalRun.variant_levers``). Where today's
    predicate over that map reproduces the stamped key, the key is this build's key whatever
    number was stamped beside it, so the result is placed at this build's version and pools
    with results stamped under it. A bump on the context side therefore splits no contestant
    whose run recorded its map; only a key nothing here can reproduce keeps its stamped version.

    :param result: The observation to place.
    :ptype result: ResultView
    :param run: The run the result belongs to, whose recorded lever map verifies its key.
    :ptype run: EvalRun
    :return: ``(variant_key, identity_version)`` — the key the runner stamped, and this build's version where
        the run's map reproduces it, else the version stamped beside it.
    :rtype: ContestantKey
    """
    if (
        result.identity_version != IDENTITY_VERSION
        and run.variant_levers is not None
        and compute_variant_key(run.variant_levers) == result.variant_key
    ):
        return (result.variant_key, IDENTITY_VERSION)
    return (result.variant_key, result.identity_version)


def identity_version_disclosure(version: int) -> str | None:
    """Say so when a contestant is placed at an identity version other than this build's.

    Derived from the same equality :func:`contestant_key` groups on, never from a
    correlate: a sentence that agreed with the grouping on the observations we happen
    to have and disagreed on the one it exists for would be worse than none.

    :param version: The identity version the contestant is placed at (:func:`contestant_key`).
    :ptype version: int
    :return: The sentence, or ``None`` when the key is this build's.
    :rtype: str | None
    """
    if version == IDENTITY_VERSION:
        return None
    return (
        f"stamped at identity version v{version}, not this build's v{IDENTITY_VERSION} — "
        "ranked separately from contestants stamped at a different version rather than pooled with them. "
        "The stamp does not say which predicate moved, so this may be a narrower split than the change warranted"
    )


def identity_version_span(keys: Iterable[ContestantKey]) -> list[int]:
    """The distinct identity versions the KEYED observations in an answer were placed at.

    :param keys: The contestant key of every observation the answer is built over
        (:func:`contestant_key`). Pass the placed rows' keys rather than the assembled
        points: an answer whose subjects were all filtered out still spanned what it
        spanned, and a span derived downstream goes missing exactly when the reader
        most needs it.
    :ptype keys: Iterable[ContestantKey]
    :return: The versions, ascending.
    :rtype: list[int]
    """
    return sorted({version for _, version in keys})


def identity_span_disclosure(versions: Sequence[int]) -> str | None:
    """Explain a corpus that spans identity versions, so a doubled row is not a mystery.

    Partitioning stops two stampings of one stack being ranked as rival
    contestants; it does not stop them APPEARING as two rows. Without this sentence a
    reader meets one model twice, under two digests, with nothing saying why — which is
    the second half of the same defect and the one a per-row flag cannot answer.

    :param versions: The span from :func:`identity_version_span`.
    :ptype versions: Sequence[int]
    :return: The sentence, or ``None`` when the answer rests on a single predicate — where
        there is no split to explain.
    :rtype: str | None
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
    "contestant_key",
    "identity_span_disclosure",
    "identity_version_disclosure",
    "identity_version_span",
]
