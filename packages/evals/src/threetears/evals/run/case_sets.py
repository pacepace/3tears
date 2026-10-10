"""Named, versioned case sets: minting the next version, and resolving one a launch targets.

A :class:`~threetears.evals.contracts.models.CaseSet` is a name, a version and a frozen, ordered list of one
template's test cases. It is append-only: :func:`mint_case_set` writes the next version, and storing a version
that exists is refused, so ``smoke v1`` names the same cases however the template has been edited since. A launch
names a set (:class:`~threetears.evals.contracts.models.CaseSetRef`); :func:`resolve_case_set` turns it into the
cases the launch runs, refusing a set whose cases no longer all resolve, and the run records the set beside the
ids it froze. The set is a label on that frozen identity, never a second one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from threetears.evals.contracts.errors import NotFoundError, ValidationFailedError
from threetears.evals.contracts.models import CaseSet, CaseSetRef

if TYPE_CHECKING:
    from threetears.evals.contracts.models import EvalTemplate, EvalTestCase
    from threetears.evals.contracts.storage import CaseSetStore


def _cases_of(storage: CaseSetStore, test_case_ids: Sequence[str], scope_id: str) -> dict[str, EvalTestCase]:
    """The named cases that resolve in the scope, by id."""
    return {case.id: case for case in storage.load_test_cases_by_ids(list(test_case_ids), scope_id)}


def _refuse_cases_outside(
    cases: dict[str, EvalTestCase], test_case_ids: Sequence[str], *, template_id: str, what: str
) -> None:
    """Refuse ids that resolve to no case in the scope, or to a case of another template.

    Args:
        cases: The ids that resolved, with their cases.
        test_case_ids: Every id the set names.
        template_id: The template the set belongs to.
        what: The set, as a refusal names it.

    Raises:
        ValidationFailedError: Naming every missing id, or every case of another template.
    """
    if missing := [case_id for case_id in test_case_ids if case_id not in cases]:
        raise ValidationFailedError(
            f"{what} names case(s) that no longer resolve in this scope: {', '.join(missing)}; a set runs exactly its "
            "cases, so mint a new version without them"
        )
    if strays := [case_id for case_id in test_case_ids if cases[case_id].template_id != template_id]:
        raise ValidationFailedError(
            f"{what} names case(s) of another template than {template_id!r}: {', '.join(strays)}; a set holds one "
            "template's cases, because a run's cases are its template's"
        )


def mint_case_set(
    storage: CaseSetStore,
    *,
    scope_id: str,
    name: str,
    template_id: str,
    test_case_ids: Sequence[str],
    tracked: bool = True,
) -> CaseSet:
    """Store the next version of the case set ``name``: version 1 for a new name, else one past the latest.

    Args:
        storage: Where sets and their cases live.
        scope_id: The scope the set lives in.
        name: The set's name.
        template_id: The template whose cases it lists; every version of one name shares it.
        test_case_ids: The cases, in order — each a stored case of the template in the scope.
        tracked: Whether the set is followed over time or made for one launch (metadata only).

    Returns:
        The stored version.

    Raises:
        ValidationFailedError: The list is empty or repeats a case, a case does not resolve or is another
            template's, the name belongs to another template, or the list is the latest version's unchanged.
        ConflictError: Another writer stored this version first.
    """
    versions = storage.query_case_sets(scope_id, name=name)
    latest = versions[0] if versions else None
    if latest is not None and latest.template_id != template_id:
        raise ValidationFailedError(
            f"case set {name!r} lists template {latest.template_id!r}'s cases, not {template_id!r}'s; name a new set"
        )
    if latest is not None and list(test_case_ids) == latest.test_case_ids:
        raise ValidationFailedError(
            f"case set {latest.ref.label!r} already lists exactly these cases; a new version records a change"
        )
    what = f"case set {name!r}"
    _refuse_cases_outside(
        _cases_of(storage, test_case_ids, scope_id), test_case_ids, template_id=template_id, what=what
    )
    try:
        minted = CaseSet(
            scope_id=scope_id,
            name=name,
            version=1 if latest is None else latest.version + 1,
            template_id=template_id,
            test_case_ids=list(test_case_ids),
            tracked=tracked,
        )
    except ValueError as e:
        raise ValidationFailedError(f"invalid case set: {e}") from e
    storage.save_case_set(minted)
    return minted


def resolve_case_set(
    storage: CaseSetStore, ref: CaseSetRef, *, template: EvalTemplate, scope_id: str
) -> tuple[CaseSet, list[EvalTestCase]]:
    """The set a launch names, and its cases in its order — or the refusal.

    Args:
        storage: Where sets and their cases live.
        ref: The set the launch names.
        template: The template the launch runs.
        scope_id: The scope the launch runs in.

    Returns:
        The stored set, and its cases in the set's order.

    Raises:
        NotFoundError: No such version of the set in the scope.
        ValidationFailedError: The set lists another template's cases, or a case of it no longer resolves (every
            missing id named).
    """
    case_set = storage.load_case_set(ref.name, ref.version, scope_id)
    if case_set is None:
        raise NotFoundError("case set", ref.label)
    what = f"case set {ref.label!r}"
    if case_set.template_id != template.id:
        raise ValidationFailedError(
            f"{what} lists template {case_set.template_id!r}'s cases, and this launch runs {template.id!r}"
        )
    cases = _cases_of(storage, case_set.test_case_ids, scope_id)
    _refuse_cases_outside(cases, case_set.test_case_ids, template_id=template.id, what=what)
    return case_set, [cases[case_id] for case_id in case_set.test_case_ids]


__all__ = ["mint_case_set", "resolve_case_set"]
