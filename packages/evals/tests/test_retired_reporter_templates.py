"""A stored analysis reporter template and its cases read as data, and their launch is refused by name.

The analysis reporter kind is retired: no host registers ``analysis_reporter`` now, and the package ships no kind
for it. A host that seeded reporter cases still stores them, as an ``eval_template`` declaring the kind and
``eval_test_case`` documents carrying a frozen reporter case in their ``host_payload``. Nothing stored changes shape
or version, so pinned here:

- **The listings read them.** ``templates_list`` lists the template with the kind it declares, the store reads both
  cases back with their payload byte for byte, and a case set lists them; none raises.
- **A launch is refused, naming the kind.** ``run_launch`` and ``launch_estimate`` refuse the template through the
  ordinary path for a kind the host has no launcher for, before anything is priced or spent.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.actions import MountedTool, eval_catalogue, standard_tools
from threetears.evals.ops import CaseSetListing, TemplateListing
from threetears.evals.schema.models import EvalTemplate, EvalTestCase, RubricDim
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_INSTANT, TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, ops_fixture

#: The kind a stored reporter template declares, as the store holds it.
_REPORTER_KIND = "analysis_reporter"

#: The stored reporter template's id.
_TEMPLATE_ID = "memo-reporter"


def _template() -> EvalTemplate:
    """The reporter template as a host stored it. Built when asked, so importing this module reads no clock."""
    return EvalTemplate(
        id=_TEMPLATE_ID,
        scope_id=TOYHOST_SCOPE,
        name="Memo reporter",
        intent="Judge the memo each campaign got.",
        candidate_kind=_REPORTER_KIND,
        rubric=[
            RubricDim(
                name="reporter.grounded_in_the_bundle",
                description="Every claim rests on a number the bundle holds.",
                scale="ordinal",
            )
        ],
        created_at=TOYHOST_INSTANT,
        updated_at=TOYHOST_INSTANT,
    )


def _reporter_payload(campaign_id: str, memo: str) -> dict[str, Any]:
    """A reporter case as a host froze one: the bundle, the recorded memo, its writer's message and a reader's label."""
    return {
        "reporter_case": {
            "bundle": {"campaign_id": campaign_id, "schema_version": 60},
            "bundle_fingerprint": f"fp-{campaign_id}",
            "bundle_assembled_at": "2026-10-22T15:53:06+00:00",
            "recorded_analysis_id": f"analysis-{campaign_id}",
            "recorded_memo": memo,
            "writer_message": "the campaign's evidence, as the memo writer read it",
            "recorded_writer_message_digest": f"digest-{campaign_id}",
            "labels": [
                {
                    "dimension": "reporter.grounded_in_the_bundle",
                    "direction": "high",
                    "quote": "every number checks out",
                    "criterion": {"description": "as stated", "scale": "ordinal", "scoring_guide": {}},
                }
            ],
        }
    }


def _cases() -> list[EvalTestCase]:
    """The template's two stored cases. Built when asked, so importing this module reads no clock."""
    return [
        EvalTestCase(
            id=f"reporter-case-{n}",
            scope_id=TOYHOST_SCOPE,
            template_id=_TEMPLATE_ID,
            host_payload=_reporter_payload(f"c-{n}", memo),
        )
        for n, memo in (
            (1, "## Summary\n\nThe candidate is the better arm."),
            (2, "## Summary\n\nNothing separated the arms."),
        )
    ]


def _evals() -> MountedTool:
    return next(tool for tool in eval_catalogue().mount_all(standard_tools()) if tool.action("templates_list"))


def _host() -> Any:
    """The toy host, its store holding the reporter template and its two cases."""
    host = off_loop(ops_fixture).host
    storage = host.eval_host.storage
    off_loop(storage.save_template, _template())
    for case in _cases():
        off_loop(storage.save_test_case, case)
    return host


async def test_the_template_lists_with_the_kind_it_declares() -> None:
    host = _host()
    checks = host.eval_host.template_checks
    assert checks is not None and _REPORTER_KIND not in checks.kinds, "the toy host registers no reporter kind"

    outcome = await _evals().call({"action": "templates_list"}, host=host, caller=CALLER)

    assert not outcome.is_error, outcome.text
    listed = {line.id: line for line in TemplateListing.model_validate(outcome.structured).templates}
    assert listed[_TEMPLATE_ID].candidate_kind == _REPORTER_KIND
    assert _TEMPLATE_ID in outcome.text


async def test_the_store_reads_both_cases_back_with_their_payloads_unchanged() -> None:
    host = _host()

    stored = off_loop(host.eval_host.storage.query_test_cases, TOYHOST_SCOPE, template_id=_TEMPLATE_ID)

    assert sorted((case.id, case.host_payload) for case in stored) == [
        (case.id, case.host_payload) for case in _cases()
    ]


async def test_a_case_set_lists_the_two_cases() -> None:
    host = _host()
    ids = [case.id for case in _cases()]

    minted = await _evals().call(
        {"action": "case_set_mint", "case_set": "reporter-seed", "template_id": _TEMPLATE_ID, "test_case_ids": ids},
        host=host,
        caller=CALLER,
    )
    assert not minted.is_error, minted.text
    listed = await _evals().call(
        {"action": "case_sets_list", "case_set_filter": "reporter-seed"}, host=host, caller=CALLER
    )

    assert not listed.is_error, listed.text
    (line,) = CaseSetListing.model_validate(listed.structured).case_sets
    assert (line.template_id, line.test_case_ids) == (_TEMPLATE_ID, ids)


async def test_a_launch_and_its_estimate_are_refused_naming_the_kind() -> None:
    host = _host()
    arguments = {"template_id": _TEMPLATE_ID, "subject_id": TOYHOST_SUBJECT.subject_id, "models": ["as-recorded"]}

    for action in ("run_launch", "launch_estimate"):
        outcome = await _evals().call({"action": action, **arguments}, host=host, caller=CALLER)

        assert outcome.is_error, f"{action} launched or priced a retired kind: {outcome.text}"
        assert outcome.text.startswith(f"refused: {action}:"), outcome.text
        assert f"candidate_kind={_REPORTER_KIND!r}, which this host has no launcher for" in outcome.text
    stored = off_loop(host.eval_host.storage.query_test_cases, TOYHOST_SCOPE, template_id=_TEMPLATE_ID)
    assert len(stored) == len(_cases()), "a refused launch leaves the cases as they were"
