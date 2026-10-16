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
from threetears.evals.kernel.judge_cases import JUDGE_KIND
from threetears.evals.kernel.remedies import FIND_TEST_CASE
from threetears.evals.run.authoring import admit_template_case
from threetears.evals.run.definition_seed import SeedCorpus, SeedTemplateCase
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

    :return: Each such case's id, by the place it shares with this one (``template '<id>'`` or ``case set <name> vN``).
    :rtype: dict[str, str]
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


def _found_by_name(host: EvalHost, case: EvalTestCase, scope_id: str, corpus: SeedCorpus | None) -> bool:
    """Whether definition drift finds ``case`` as a corpus case by its name, as it finds a template case stored before
    seeds recorded ``seed_name``: live on its template's current revision, recording no seed name, named as a corpus
    case of that template, and no live case beside it recording that name."""
    storage = host.storage
    if corpus is None or case.seed_name is not None or case.name is None or case.archived or case.template_id is None:
        return False
    template = storage.load_template(case.template_id, scope_id)
    if template is None or template.candidate_kind == JUDGE_KIND:
        return False
    current = storage.load_template_by_name(template.name, scope_id)
    if current is None or current.id != template.id:
        return False
    if not any(own.template == template.name and own.name == case.name for own in corpus.template_cases):
        return False
    return not any(
        other.seed_name == case.name and not other.archived
        for other in storage.query_test_cases(scope_id, template_id=template.id)
    )


def _admit_renamed(host: EvalHost, case: EvalTestCase, name: str, scope_id: str) -> None:
    """Hold a template's own case, under the name it would carry, to the host's check over a template case.

    A case of no template, or of a judge template, is not a template case and is not asked. A host supplying no
    template checks is not asked either: a rename admits no template.
    """
    checks = host.template_checks
    if checks is None or case.template_id is None:
        return
    template = host.storage.load_template(case.template_id, scope_id)
    if template is None or template.candidate_kind == JUDGE_KIND:
        return
    admit_template_case(
        SeedTemplateCase(
            template=template.name,
            name=name,
            variation_params=case.variation_params,
            host_payload=case.host_payload,
            stratum=case.stratum,
        ),
        checks=checks,
    )


def case_rename(
    host: EvalHost, scope_id: str, case_id: str, name: str, *, corpus: SeedCorpus | None = None
) -> CaseRenamed:
    """Rename a stored case: write its ``name``, and nothing else but the seed name drift would have found it by.

    The name is curation, as a run's labels are: no digest or identity key reads it, so the case keeps its id, its
    ``content_hash``, its ``seed_name`` and ``seed_digest`` and every result measured against it. A rename never
    changes how definition drift finds the case: drift finds a template case stored before seeds recorded
    ``seed_name`` by its name, so when ``corpus`` names it so, the same write records its old name as ``seed_name``;
    a case drift would not find by name gets none. The write goes through the store's own path for a case
    (``save_test_case``), so the host's routing and its write rules hold, conditionally on the token read with the
    case: a write that landed between the read and the save (a retirement, an edit) refuses the rename with
    :class:`ConflictError` rather than being undone. The name already carried writes nothing. Two renames of sibling
    cases to one name at once can both pass the duplicate check, since the check reads other documents than the one
    each writes.

    :param host: The host whose store holds the case.
    :ptype host: EvalHost
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param case_id: The case, by its test case id.
    :ptype case_id: str
    :param name: Its new name, as the report should call it: one token such as ``double-charge``, surrounding whitespace
        dropped.
    :ptype name: str
    :param corpus: The host's seed corpus for the scope (:func:`~threetears.evals.run.load_seed_corpus`), which says
        whether drift finds the case by its name; ``None`` for a host that seeds no template cases.
    :ptype corpus: SeedCorpus | None
    :return: The case's id, and its name before and after.
    :rtype: CaseRenamed
    :raises ValidationFailedError: The report could not call the case by the name: a blank name, one with whitespace
        inside, a bare number, a generated id, or one longer than a case label runs (``humanize("case", name)``
        is empty), so the page would name it ``case N``.
    :raises ValidationFailedError: (also) The case is a template's own, and the host's check over a template case
        (``TemplateChecks.refuse_inadmissible_template_case``) refuses it under the new name: its reason is given.
    :raises NotFoundError: No case with that id in the scope.
    :raises ConflictError: Another case of its template, or of a case set listing it, already carries the name; or the
        case was written since it was read.
    :raises StorageError: The write did not land.
    """
    given = name.strip()
    if not given or humanize("case", given) != given:
        raise ValidationFailedError(
            f"the report would not call a case {name!r} by that name, so it would read as 'case N': a case name is one "
            f"token (no spaces), not a bare number or a generated id, and at most {CASE_LABEL_CHARS} characters",
            remedy="name it the way its author would, such as double-charge",
        )
    storage = host.storage
    case, etag = storage.load_test_case_with_etag(case_id, scope_id)
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
    _admit_renamed(host, case, given, scope_id)
    data = case.to_dict()
    data["name"] = given
    if _found_by_name(host, case, scope_id, corpus):
        data["seed_name"] = case.name
    storage.save_test_case(EvalTestCase.from_dict(data), if_match=etag)
    log.info("eval.case_rename case=%s scope=%s old=%s new=%s", case.id, scope_id, case.name, given)
    return CaseRenamed(test_case_id=case.id, old_name=case.name, new_name=given)


__all__ = ["CaseRenamed", "case_rename"]
