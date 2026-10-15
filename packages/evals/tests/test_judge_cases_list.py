"""``judge_cases_list``: which cases a judge template holds, so an operator can mint a case set over all of them.

A case set lists cases by id, and a template carries no list of its own, so without a read nobody can say which ids
a template holds (a set of 12 over a template of 38 is the failure it answers). The action reads only: it calls no
model and writes nothing.
"""

from __future__ import annotations

import json
from typing import Any

from threetears.evals.actions import Caller, MountedTool, eval_catalogue, read_only_tools, standard_tools
from threetears.evals.kernel import JUDGE_KIND
from threetears.evals.ops import JudgeCaseAuthor, judge_case_author
from threetears.evals.ops.judge_cases import JudgeCaseListing
from threetears.evals.schema.models import EvalTemplate, EvalTestCase
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_INSTANT
from packages.evals.tests.fixtures.toyhost.judge import FAITHFULNESS_DIM, toyhost_judged_template
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, ops_fixture

_AUTHOR = "cy@example.com"


def _tool() -> MountedTool:
    return next(tool for tool in eval_catalogue().mount_all(standard_tools()) if tool.action("judge_cases_list"))


async def _list(host: Any, template_id: str, caller: Caller = CALLER) -> Any:
    return await _tool().call({"action": "judge_cases_list", "template_id": template_id}, host=host, caller=caller)


def _judge_template() -> EvalTemplate:
    return EvalTemplate(
        id="listed-judge",
        scope_id=TOYHOST_SCOPE,
        name="Listed judge",
        intent="Score how faithfully an extraction reproduces its invoice.",
        candidate_kind=JUDGE_KIND,
        created_at=TOYHOST_INSTANT,
        updated_at=TOYHOST_INSTANT,
    )


def _author(host: Any, expected: int, output: str) -> str:
    report = judge_case_author(
        host.eval_host,
        JudgeCaseAuthor(
            template_id=_judge_template().id,
            criterion_template_id=toyhost_judged_template().id,
            criterion=FAITHFULNESS_DIM,
            output=json.dumps({"invoice_number": output}),
            case_material="Invoice doc-42",
            expected=expected,
            note="because",
        ),
        TOYHOST_SCOPE,
        author=_AUTHOR,
    )
    return report.cases[0].test_case_id


def _host() -> Any:
    host = off_loop(ops_fixture).host
    storage = host.eval_host.storage
    off_loop(storage.save_template, toyhost_judged_template())
    off_loop(storage.save_template, _judge_template())
    return host


async def test_it_lists_every_stored_case_in_stored_order_with_its_author_verdict_and_liveness() -> None:
    host = _host()
    first = off_loop(_author, host, 1, "INV-1")
    second = off_loop(_author, host, 3, "INV-2")
    storage = host.eval_host.storage
    stored = off_loop(storage.load_test_case, second, TOYHOST_SCOPE)
    assert stored is not None
    off_loop(storage.save_test_case, stored.model_copy(update={"archived": True}))

    outcome = await _list(host, _judge_template().id)

    assert not outcome.is_error, outcome.text
    listing = JudgeCaseListing.model_validate(outcome.structured)
    assert [case.test_case_id for case in listing.cases] == [
        case.id for case in off_loop(storage.query_test_cases, TOYHOST_SCOPE, template_id=_judge_template().id)
    ]
    by_id = {case.test_case_id: case for case in listing.cases}
    assert set(by_id) == {first, second} and listing.unreadable == []
    assert (by_id[first].criterion, by_id[first].authored_by, by_id[first].expected, by_id[first].live) == (
        FAITHFULNESS_DIM,
        _AUTHOR,
        1,
        True,
    )
    assert (by_id[second].expected, by_id[second].live, by_id[second].labels) == (3, False, 1)
    assert first in outcome.text and second in outcome.text


async def test_a_template_with_no_cases_lists_none() -> None:
    host = _host()
    outcome = await _list(host, _judge_template().id)
    assert not outcome.is_error, outcome.text
    listing = JudgeCaseListing.model_validate(outcome.structured)
    assert (listing.template_id, listing.cases, listing.unreadable) == (_judge_template().id, [], [])


async def test_a_case_that_is_not_a_readable_judge_case_is_listed_apart() -> None:
    host = _host()
    off_loop(
        host.eval_host.storage.save_test_case,
        EvalTestCase(id="plain", scope_id=TOYHOST_SCOPE, template_id=_judge_template().id),
    )
    outcome = await _list(host, _judge_template().id)
    listing = JudgeCaseListing.model_validate(outcome.structured)
    assert listing.cases == [] and [(u.test_case_id, u.reason) for u in listing.unreadable] == [
        ("plain", "it carries no judge case")
    ]


async def test_an_unknown_template_is_refused_with_the_usual_remedy() -> None:
    outcome = await _list(_host(), "no-such-template")
    assert outcome.is_error and outcome.text.startswith("refused: judge_cases_list:")
    assert "template 'no-such-template' not found" in outcome.text and "templates_list" in outcome.text


async def test_a_template_of_another_kind_is_refused() -> None:
    outcome = await _list(_host(), toyhost_judged_template().id)
    assert outcome.is_error and f"not '{JUDGE_KIND}'" in outcome.text


async def test_a_scope_the_template_is_not_in_does_not_resolve() -> None:
    host = _host()
    off_loop(_author, host, 1, "INV-1")
    outcome = await _list(host, _judge_template().id, Caller(scope_id="another-scope", identity="agent:elsewhere"))
    assert outcome.is_error and f"template '{_judge_template().id}' not found" in outcome.text


def test_it_is_a_read_and_the_read_only_tool_carries_it() -> None:
    action = _tool().action("judge_cases_list")
    assert action is not None and action.permission == "read"
    (reading,) = eval_catalogue().mount_all(read_only_tools())
    assert reading.action("judge_cases_list") is not None
