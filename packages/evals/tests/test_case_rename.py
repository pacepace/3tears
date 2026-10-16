"""``case_rename``: a stored case's name changes in place, and nothing else about it does.

A case's name is in no digest and no identity key, so a rename keeps the case's id, its content hash, its seed record
and every result measured against it, and the report calls it by the new name. Before this write a host could rename a
case only by writing its document itself, past the store's write path. The rename refuses a name the report would not
print, a name another case beside it carries, and an unknown case, as the ``cases_rename`` action does.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals import Eval
from threetears.evals.actions import Caller, MountedTool, eval_catalogue, standard_tools
from threetears.evals.analysis import assemble_context_bundle, build_code_only_report, get_campaign, report_markdown
from threetears.evals.kernel import ConflictError, NotFoundError, ValidationFailedError
from threetears.evals.kernel.host import EvalHost
from threetears.evals.ops import CaseRenamed, case_rename, reporter_case_archive
from threetears.evals.schema import CaseSet, EvalTestCase, case_set_doc_id
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, ops_fixture
from packages.evals.tests.test_reporter_case_actions import _fixture as reporter_fixture, _freeze as freeze

_SCOPE = TOYHOST_SCOPE


def _case(case_id: str, name: str | None, template_id: str | None = None) -> EvalTestCase:
    return EvalTestCase(
        id=case_id,
        scope_id=_SCOPE,
        template_id=template_id or toyhost_template().id,
        variation_params={"ticket": case_id},
        name=name,
        content_hash=f"hash-{case_id}",
        seed_name=name,
        seed_digest=f"seeded-{case_id}",
        created_at="2026-10-01T00:00:00+00:00",
    )


def _host(*cases: EvalTestCase) -> EvalHost:
    host = toyhost_host()
    for case in cases:
        host.storage.save_test_case(case)
    return host


def test_a_rename_writes_the_name_and_keeps_the_id_the_content_and_the_seed_record() -> None:
    before = _case("c1", "double-charge")
    host = _host(before)

    renamed = case_rename(host, _SCOPE, "c1", "  charged-twice ")

    assert renamed == CaseRenamed(test_case_id="c1", old_name="double-charge", new_name="charged-twice")
    after = host.storage.load_test_case("c1", _SCOPE)
    assert after is not None and after.name == "charged-twice"
    assert after.model_dump(exclude={"name"}) == before.model_dump(exclude={"name"})


def test_a_case_with_no_name_takes_one_and_the_name_it_carries_writes_nothing() -> None:
    host = _host(_case("c1", None))
    assert case_rename(host, _SCOPE, "c1", "refund") == CaseRenamed(test_case_id="c1", old_name=None, new_name="refund")
    assert case_rename(host, _SCOPE, "c1", "refund").old_name == "refund"


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("   ", id="blank"),
        pytest.param("charged twice", id="two-words"),
        pytest.param("7", id="a-bare-number"),
        pytest.param("eval-6dad555b3db40e35-0", id="a-generated-id"),
        pytest.param("x" * 41, id="too-long"),
    ],
)
def test_a_name_the_report_would_print_as_case_n_is_refused(name: str) -> None:
    host = _host(_case("c1", "double-charge"))
    with pytest.raises(ValidationFailedError, match="would read as 'case N'") as refused:
        case_rename(host, _SCOPE, "c1", name)
    assert refused.value.remedy
    stored = host.storage.load_test_case("c1", _SCOPE)
    assert stored is not None and stored.name == "double-charge"


def test_a_name_another_case_of_the_template_carries_is_refused() -> None:
    host = _host(_case("c1", "double-charge"), _case("c2", "refund"), _case("c3", "crash", template_id="other"))
    with pytest.raises(ConflictError, match="case 'c2' of its template") as refused:
        case_rename(host, _SCOPE, "c1", "refund")
    assert refused.value.status_code == 409
    # A case of another template, in no set beside it, does not clash.
    assert case_rename(host, _SCOPE, "c1", "crash").new_name == "crash"


def test_a_name_another_case_of_a_case_set_listing_it_carries_is_refused() -> None:
    host = _host(_case("c1", "double-charge"), _case("c3", "crash", template_id="other"))
    host.storage.save_case_set(
        CaseSet(
            id=case_set_doc_id("smoke", 1),
            scope_id=_SCOPE,
            name="smoke",
            version=1,
            template_id=toyhost_template().id,
            test_case_ids=["c1", "c3"],
        )
    )
    with pytest.raises(ConflictError, match="case 'c3' of its case set smoke v1"):
        case_rename(host, _SCOPE, "c1", "crash")


def test_an_unknown_case_is_refused() -> None:
    with pytest.raises(NotFoundError, match="test case 'nope' not found"):
        case_rename(_host(), _SCOPE, "nope", "refund")


def test_a_write_landing_between_the_read_and_the_save_refuses_the_rename_rather_than_undoing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = reporter_fixture()
    host = fixture.host.eval_host
    case_id = freeze(fixture).test_case_id
    read = host.storage.load_test_case_with_etag

    def retired_meanwhile(test_case_id: str, scope_id: str) -> tuple[EvalTestCase | None, str | None]:
        loaded = read(test_case_id, scope_id)
        reporter_case_archive(host, test_case_id, scope_id, archived=True, reason="orphaned")
        return loaded

    monkeypatch.setattr(host.storage, "load_test_case_with_etag", retired_meanwhile)
    with pytest.raises(ConflictError, match="changed since it was read"):
        case_rename(host, _SCOPE, case_id, "orphaned-memo")
    stored = host.storage.load_test_case(case_id, _SCOPE)
    assert stored is not None and stored.archived and stored.archived_reason == "orphaned"
    assert stored.name != "orphaned-memo"


# --- the action -----------------------------------------------------------------------------------------------------


def _tool() -> MountedTool:
    return next(tool for tool in eval_catalogue().mount_all(standard_tools()) if tool.action("cases_rename"))


async def _rename(host: Any, case_id: str, name: str, caller: Caller = CALLER) -> Any:
    return await _tool().call(
        {"action": "cases_rename", "test_case_id": case_id, "case_name": name}, host=host, caller=caller
    )


async def test_the_action_renames_and_refuses_as_the_operation_does() -> None:
    host = off_loop(ops_fixture).host
    for case in (_case("c1", "double-charge"), _case("c2", "refund")):
        off_loop(host.eval_host.storage.save_test_case, case)

    outcome = await _rename(host, "c1", "charged-twice")
    assert not outcome.is_error, outcome.text
    assert CaseRenamed.model_validate(outcome.structured).new_name == "charged-twice"

    for case_id, name, said in [
        ("c1", "charged twice", "would read as 'case N'"),
        ("c1", "refund", "case 'c2' of its template"),
        ("nope", "refund", "test case 'nope' not found"),
    ]:
        refused = await _rename(host, case_id, name)
        assert refused.is_error and refused.text.startswith("refused: cases_rename:"), refused.text
        assert said in refused.text
    elsewhere = await _rename(host, "c1", "refund-again", Caller(scope_id="another-scope", identity="agent:elsewhere"))
    assert elsewhere.is_error and "test case 'c1' not found" in elsewhere.text
    action = _tool().action("cases_rename")
    assert action is not None and action.permission == "write"


# --- the page -------------------------------------------------------------------------------------------------------


async def _answer(case: Mapping[str, Any]) -> str:
    return "billing" if "refund" in str(case["text"]) else "bug"


async def test_the_page_names_a_renamed_case_by_its_new_name_and_keeps_its_results() -> None:
    cases = [
        {"id": "double-charge", "text": "refund me twice", "label": "billing"},
        {"id": "crash", "text": "the app crashes", "label": "bug"},
    ]
    result = await Eval(cases, _answer, ["label"], k=1, name="Renamed cases").run(store=":memory:")
    storage, scope = result.host.storage, result.scope_id
    (renamed,) = [c for c in off_loop(storage.query_test_cases, scope) if c.name == "double-charge"]
    results_before = {r.id: r.test_case_id for r in off_loop(storage.query_eval_results_by_run, result.run_id, scope)}

    off_loop(case_rename, result.host, scope, renamed.id, "charged-twice")

    results_after = {r.id: r.test_case_id for r in off_loop(storage.query_eval_results_by_run, result.run_id, scope)}
    assert results_after == results_before and renamed.id in results_after.values()
    campaign = off_loop(get_campaign, storage, result.campaign_id, scope)
    bundle = off_loop(assemble_context_bundle, campaign, storage=storage, profile=result.host.profile)
    page = report_markdown(build_code_only_report(bundle, assembled_at="2026-10-15T00:00:00+00:00"))
    assert "charged-twice" in page and "double-charge" not in page
