"""A stored case's curation: the name its author gave it, changed in place.

A case's name (:attr:`~threetears.evals.schema.models.EvalTestCase.name`) is in no digest and no identity key, so a
case renamed keeps its id, its content hash and every result measured against it; the report names it by the new name
from its next read. :func:`case_rename` is the write a surface makes for it, through the store's own write path, so no
host writes the case document directly.
"""

from __future__ import annotations

from pydantic import Field

from threetears.evals.analysis.style import CASE_LABEL_CHARS, humanize
from threetears.evals.kernel.errors import ConflictError, NotFoundError, ValidationFailedError
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.remedies import FIND_TEST_CASE
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.logs import get_logger
from threetears.evals.schema.models import EvalTestCase

log = get_logger(__name__)


class CaseRenamed(EvalBaseModel):
    """A renamed case: which, and its name before and after."""

    test_case_id: str = Field(description="The case's id, unchanged by the rename.")
    old_name: str | None = Field(description="The name it carried before; None when it had none.")
    new_name: str = Field(description="The name it carries now, which the report calls it by.")


def _siblings(host: EvalHost, case: EvalTestCase, scope_id: str) -> dict[str, str]:
    """Every other case a reader tells this one apart from by name, and where it sits beside it.

    The cases of its template, and the cases of each case set listing it (a set may list cases of another template).

    Returns:
        Each such case's id, by the place it shares with this one (``template '<id>'`` or ``case set <name> vN``).
    """
    storage = host.storage
    beside: dict[str, str] = {}
    if case.template_id is not None:
        for stored in storage.query_test_cases(scope_id, template_id=case.template_id):
            beside.setdefault(stored.id, f"template {case.template_id!r}")
    for case_set in storage.query_case_sets(scope_id):
        if case.id in case_set.test_case_ids:
            for listed in case_set.test_case_ids:
                beside.setdefault(listed, f"case set {case_set.name} v{case_set.version}")
    beside.pop(case.id, None)
    return beside


def case_rename(host: EvalHost, scope_id: str, case_id: str, name: str) -> CaseRenamed:
    """Rename a stored case: write its ``name`` and nothing else.

    The name is curation, as a run's labels are: no digest or identity key reads it, so the case keeps its id, its
    ``content_hash``, its ``seed_name`` and ``seed_digest`` (a renamed seeded case still reads stale or customized by
    its content alone) and every result measured against it. The write goes through the store's own path for a case
    (``save_test_case``), so the host's routing and its write rules hold. The name already carried writes nothing.

    Args:
        host: The host whose store holds the case.
        scope_id: The scope it lives in.
        case_id: The case, by its test case id.
        name: Its new name, as the report should call it: one token such as ``double-charge``, surrounding whitespace
            dropped.

    Returns:
        The case's id, and its name before and after.

    Raises:
        ValidationFailedError: The report could not call the case by the name: a blank name, one with whitespace
            inside, a bare number, a generated id, or one longer than a case label runs (``humanize("case", name)``
            is empty), so the page would name it ``case N``.
        NotFoundError: No case with that id in the scope.
        ConflictError: Another case of its template, or of a case set listing it, already carries the name.
        StorageError: The write did not land.
    """
    given = name.strip()
    if not given or humanize("case", given) != given:
        raise ValidationFailedError(
            f"the report would not call a case {name!r} by that name, so it would read as 'case N': a case name is one "
            f"token (no spaces), not a bare number or a generated id, and at most {CASE_LABEL_CHARS} characters",
            remedy="name it the way its author would, such as double-charge",
        )
    storage = host.storage
    case = storage.load_test_case(case_id, scope_id)
    if case is None:
        raise NotFoundError("test case", case_id, remedy=FIND_TEST_CASE)
    if case.name == given:
        return CaseRenamed(test_case_id=case.id, old_name=case.name, new_name=given)
    beside = _siblings(host, case, scope_id)
    if clashes := sorted(
        (beside[other.id], other.id)
        for other in storage.load_test_cases_by_ids(sorted(beside), scope_id)
        if other.name == given
    ):
        raise ConflictError(
            f"case {case.id!r} cannot be named {given!r}: "
            + "; ".join(f"case {other!r} of its {where} already is" for where, other in clashes)
            + ", and a report names each case once",
            remedy="choose a name no other case of its template or case sets carries, or rename that case first",
        )
    data = case.to_dict()
    data["name"] = given
    storage.save_test_case(EvalTestCase.from_dict(data))
    log.info("eval.case_rename case=%s scope=%s old=%s new=%s", case.id, scope_id, case.name, given)
    return CaseRenamed(test_case_id=case.id, old_name=case.name, new_name=given)


__all__ = ["CaseRenamed", "case_rename"]
