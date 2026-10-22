"""A reporter case's labels are readers' verdicts, so the host attests who relays whose.

A reporter case's labels are read as people's verdicts on a memo: the judge's agreement with readers. So
``reporter_case_freeze`` asks the host's own check (``EvalHost.attests_person``) before it reads anything, exactly as
``result_rate_person`` and ``judge_case_author`` do: the caller (``collected_by``) relays the verdict of each reader a
label names (``reader``), or its own when a label names none. Pinned here:

- a host that supplies no check, or one that refuses any reader, refuses the freeze (``refused``, 403), logs a
  warning, and nothing is written, before the template is read; a freeze with no labels asks nothing;
- every distinct reader is asked once, in order; each label is stored under the host's name for its reader, and
  records the caller that relayed it; the receipt shows both;
- a check that answers other than a name, or raises, propagates as the host's defect; a label naming its own relay,
  or a blank reader, is refused; nothing is written;
- who gave or relayed a verdict is not part of the case: a re-freeze of the same verdicts by another caller returns
  the stored case, whose record stands; a case stored before readers were recorded (a frozen v16.0 document) is
  returned by an attested re-freeze of its verdicts, without ``supersedes``;
- a label naming neither is stored without either key, as the build before stored it; a relayed one carries both.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``reporter_case_freeze``: not asking the host; asking after the template is read; asking for a freeze with no
  labels; storing the caller's spelling of a reader; recording no ``collected_by``; accepting a caller-sent
  ``collected_by``.
- ``freeze_reporter_case``: comparing ``reader`` or ``collected_by`` when deciding a re-freeze is the same case.
- ``reporter_case_payload``: writing ``reader: null`` or ``collected_by: null``, or dropping a set one.
- the action: recording a fixed identity as ``collected_by`` rather than the call's caller.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.actions import Caller, MountedTool, eval_catalogue, standard_tools
from threetears.evals.analysis.reporter_kind import reporter_case_of
from threetears.evals.kernel import EvalServiceError, ValidationFailedError
from threetears.evals.ops import (
    FrozenReporterCase,
    OpsHost,
    ReporterCaseFreeze,
    analysis_generate,
    reporter_case_freeze,
)
from threetears.evals.schema import EvalTestCase
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture, settled
from packages.evals.tests.reporter_case_support import GROUNDED, USEFUL, reporter_fixture, reporter_template

PERSON = "ana@example.com"


@dataclass
class _Attests:
    """A host's check that vouches for one caller relaying the readers it names, and records every question asked.

    ``names`` maps each spelling the host accepts to its own name for that person.
    """

    caller: str
    names: dict[str, str]
    asked: list[dict[str, str]] = field(default_factory=list)

    def __call__(self, *, scope_id: str, caller: str, person: str) -> str | None:
        self.asked.append({"scope_id": scope_id, "caller": caller, "person": person})
        return self.names.get(person) if caller == self.caller else None


def _host(fixture: OpsFixture, attests: Any) -> OpsHost:
    """The fixture's host, with ``attests`` as its person check (``None`` for a host that supplies none)."""
    launch = fixture.host.launch
    return replace(fixture.host, launch=replace(launch, eval_host=replace(launch.eval_host, attests_person=attests)))


async def _analysis_id(fixture: OpsFixture) -> str:
    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    analysis_id = (await settled(fixture.host, job.job_id)).analysis_id
    assert analysis_id is not None
    return analysis_id


def _label(**changes: Any) -> dict[str, Any]:
    return {"dimension": GROUNDED, "direction": "high", "quote": "every number traces to a run", **changes}


def _freeze(
    fixture: OpsFixture, analysis_id: str | None, labels: list[dict[str, Any]], **fields: Any
) -> dict[str, Any]:
    return {
        "action": "reporter_case_freeze",
        "template_id": reporter_template().id,
        "campaign_id": fixture.campaign.id,
        **({"recorded_analysis_id": analysis_id} if analysis_id is not None else {}),
        **({"labels": labels} if labels else {}),
        **fields,
    }


def _tool() -> MountedTool:
    return {tool.name: tool for tool in eval_catalogue().mount_all(standard_tools())}["evals"]


def _stored_cases(fixture: OpsFixture) -> list[EvalTestCase]:
    return off_loop(fixture.host.eval_host.storage.query_test_cases, TOYHOST_SCOPE, template_id=reporter_template().id)


# --- refused: nothing is written ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("attests", "says"),
    [
        (_Attests(caller="agent:someone-else", names={PERSON: PERSON}), "does not attest 'agent:test' relaying"),
        (_Attests(caller=CALLER.identity, names={"bo@example.com": "bo@example.com"}), "does not attest 'agent:test'"),
        (None, "vouches for no person"),
    ],
    ids=["another-caller", "another-reader", "no-check"],
)
async def test_a_verdict_the_host_does_not_attest_is_refused_and_nothing_is_written(
    attests: Any, says: str, caplog: pytest.LogCaptureFixture
) -> None:
    fixture = off_loop(reporter_fixture)
    analysis_id = await _analysis_id(fixture)
    host = _host(fixture, attests)

    with caplog.at_level(logging.WARNING, logger="threetears.evals"):
        outcome = await _tool().call(_freeze(fixture, analysis_id, [_label(reader=PERSON)]), host=host, caller=CALLER)
    unknown = await _tool().call(
        _freeze(fixture, analysis_id, [_label(reader=PERSON)], template_id="no-such-reporter"), host=host, caller=CALLER
    )

    assert outcome.is_error and outcome.refusal is not None
    assert (outcome.refusal.kind, outcome.refusal.status) == ("refused", 403)
    assert says in outcome.text and "reader verdict" in outcome.text
    assert not _stored_cases(fixture)
    assert unknown.refusal is not None and unknown.refusal.status == 403, "asked before the template is read"
    assert any(
        record.levelno == logging.WARNING
        and record.getMessage().startswith(
            f"eval.reporter_case_freeze refused caller={CALLER.identity} person={PERSON} scope={TOYHOST_SCOPE}:"
        )
        for record in caplog.records
    ), "an operator sees the refused attestation"


async def test_one_refused_reader_refuses_the_whole_freeze() -> None:
    fixture = off_loop(reporter_fixture)
    analysis_id = await _analysis_id(fixture)
    attests = _Attests(caller=CALLER.identity, names={PERSON: PERSON})

    outcome = await _tool().call(
        _freeze(fixture, analysis_id, [_label(reader=PERSON), _label(dimension=USEFUL, reader="bo@example.com")]),
        host=_host(fixture, attests),
        caller=CALLER,
    )

    assert outcome.refusal is not None and outcome.refusal.status == 403
    assert not _stored_cases(fixture)


async def test_a_freeze_with_no_labels_carries_no_verdict_and_asks_nothing() -> None:
    fixture = off_loop(reporter_fixture)
    attests = _Attests(caller="agent:someone-else", names={})

    outcome = await _tool().call(_freeze(fixture, None, []), host=_host(fixture, attests), caller=CALLER)

    assert not outcome.is_error, outcome.text
    assert attests.asked == []
    assert len(_stored_cases(fixture)) == 1


@pytest.mark.parametrize("answer", [True, "", "   "], ids=["yes", "empty", "blank"])
def test_a_host_check_that_answers_other_than_a_name_is_a_defect_and_nothing_is_written(answer: Any) -> None:
    fixture = reporter_fixture()

    def answers(*, scope_id: str, caller: str, person: str) -> Any:
        return answer

    with pytest.raises(TypeError, match=f"attests_person returned {answer!r}"):
        reporter_case_freeze(
            _host(fixture, answers).eval_host,
            ReporterCaseFreeze(template_id="no-such-reporter", campaign_id=fixture.campaign.id, labels=[_label()]),
            TOYHOST_SCOPE,
            collected_by=CALLER.identity,
        )
    assert not fixture.host.eval_host.storage.query_test_cases(TOYHOST_SCOPE, template_id=reporter_template().id)


def test_a_host_with_no_check_refuses_the_operation_itself() -> None:
    """The operation, not only the action, refuses: no surface records a reader's verdict the host never attested."""
    fixture = reporter_fixture()

    with pytest.raises(EvalServiceError, match="vouches for no person") as refused:
        reporter_case_freeze(
            _host(fixture, None).eval_host,
            ReporterCaseFreeze(template_id=reporter_template().id, campaign_id=fixture.campaign.id, labels=[_label()]),
            TOYHOST_SCOPE,
            collected_by=CALLER.identity,
        )
    assert refused.value.status_code == 403


@pytest.mark.parametrize(
    ("label", "says"),
    [
        (_label(collected_by="agent:forged"), "name collected_by"),
        (_label(reader="  "), "a label's reader is"),
        (_label(reader=7), "a label's reader is"),
    ],
    ids=["relay-named", "blank-reader", "not-a-name"],
)
def test_a_label_naming_its_own_relay_or_no_reader_is_refused(label: dict[str, Any], says: str) -> None:
    fixture = reporter_fixture()
    attests = _Attests(caller=CALLER.identity, names={PERSON: PERSON, CALLER.identity: CALLER.identity})

    with pytest.raises(ValidationFailedError, match=says):
        reporter_case_freeze(
            _host(fixture, attests).eval_host,
            ReporterCaseFreeze(template_id=reporter_template().id, campaign_id=fixture.campaign.id, labels=[label]),
            TOYHOST_SCOPE,
            collected_by=CALLER.identity,
        )
    assert attests.asked == [], "refused before the host is asked"


# --- attested: each reader's verdict, and who relayed it ----------------------------------------------------------


async def test_each_reader_is_asked_once_and_their_verdicts_are_stored_under_the_hosts_names() -> None:
    """Labels naming one reader in two spellings, and one naming none (the caller's own), ask about two people."""
    fixture = off_loop(reporter_fixture)
    analysis_id = await _analysis_id(fixture)
    attests = _Attests(caller=PERSON, names={"Ana@Example.com": PERSON, PERSON: PERSON})
    caller = Caller(scope_id=TOYHOST_SCOPE, identity=PERSON)
    labels = [
        _label(reader="Ana@Example.com"),
        _label(dimension=USEFUL, direction="low", quote="nothing to act on"),
        _label(dimension=USEFUL, direction="mid", quote="some of it", reader=PERSON),
    ]

    outcome = await _tool().call(_freeze(fixture, analysis_id, labels), host=_host(fixture, attests), caller=caller)

    assert not outcome.is_error, outcome.text
    assert attests.asked == [
        {"scope_id": TOYHOST_SCOPE, "caller": PERSON, "person": "Ana@Example.com"},
        {"scope_id": TOYHOST_SCOPE, "caller": PERSON, "person": PERSON},
    ]
    (stored,) = _stored_cases(fixture)
    case = reporter_case_of(stored)
    assert case is not None
    assert [(label.reader, label.collected_by) for label in case.labels] == [(PERSON, PERSON)] * 3
    receipt = FrozenReporterCase.model_validate(outcome.structured)
    assert [(label.reader, label.collected_by) for label in receipt.labels] == [(PERSON, PERSON)] * 3


async def test_the_action_records_the_host_resolved_caller_as_the_relay() -> None:
    """An agent names the reader whose verdict it relays; ``collected_by`` is the caller the host resolved."""
    other = Caller(scope_id=TOYHOST_SCOPE, identity="agent:other")
    fixture = off_loop(reporter_fixture)
    analysis_id = await _analysis_id(fixture)
    attests = _Attests(caller=other.identity, names={PERSON: PERSON})

    outcome = await _tool().call(
        _freeze(fixture, analysis_id, [_label(reader=PERSON)]), host=_host(fixture, attests), caller=other
    )

    assert not outcome.is_error, outcome.text
    assert attests.asked == [{"scope_id": TOYHOST_SCOPE, "caller": other.identity, "person": PERSON}]
    (stored,) = _stored_cases(fixture)
    case = reporter_case_of(stored)
    assert case is not None
    ((reader, collected_by),) = [(label.reader, label.collected_by) for label in case.labels]
    assert (reader, collected_by) == (PERSON, other.identity)
    assert f"collected_by={other.identity}" in outcome.text, "the receipt names the relay"
    (label,) = stored.host_payload["reporter_case"]["labels"]
    assert (label["reader"], label["collected_by"]) == (PERSON, other.identity), "a relayed label carries both keys"


async def test_the_same_verdicts_relayed_again_by_another_caller_return_the_stored_case() -> None:
    """Who gave or relayed a verdict is provenance, not the case: no supersedes, and the stored record stands."""
    fixture = off_loop(reporter_fixture)
    analysis_id = await _analysis_id(fixture)
    first = _Attests(caller=CALLER.identity, names={PERSON: PERSON})
    second = Caller(scope_id=TOYHOST_SCOPE, identity="agent:other")
    again = _Attests(caller=second.identity, names={"bo@example.com": "bo@example.com"})

    stored_first = await _tool().call(
        _freeze(fixture, analysis_id, [_label(reader=PERSON)]), host=_host(fixture, first), caller=CALLER
    )
    repeated = await _tool().call(
        _freeze(fixture, analysis_id, [_label(reader="bo@example.com")]), host=_host(fixture, again), caller=second
    )

    assert not repeated.is_error, repeated.text
    assert (
        FrozenReporterCase.model_validate(repeated.structured).test_case_id
        == FrozenReporterCase.model_validate(stored_first.structured).test_case_id
    )
    (stored,) = _stored_cases(fixture)
    case = reporter_case_of(stored)
    assert case is not None
    ((reader, collected_by),) = [(label.reader, label.collected_by) for label in case.labels]
    assert (reader, collected_by) == (PERSON, CALLER.identity), "the stored record stands"


# --- the stored case: a case stored before readers were recorded -----------------------------------------------------

#: A labelled reporter case as the build before readers were recorded stored it (71c9f5f52: its labels carry no
#: ``reader`` or ``collected_by`` key), with the analysis whose memo it pins, both as that build stored them over the
#: toy host's corpus campaign. Never edit it: it is what production stores hold.
_FROZEN_V16_0 = Path(__file__).resolve().parent / "fixtures" / "reporter_cases" / "labelled_case_v16_0.json"


#: The fields of a reporter case that hold the evidence a freeze re-assembles, rather than its labels.
_EVIDENCE = ("bundle", "bundle_fingerprint", "writer_message", "limits")


def _frozen_store() -> tuple[OpsFixture, EvalTestCase, str]:
    """The toy host's operations fixture over a store holding the frozen analysis and case.

    The case's labels, the shape under test, are kept exactly as stored. Its evidence is taken from this machine's own
    freeze of the same campaign and memo: a bundle's fingerprint is not reproducible across machines (the freezing
    machine and the test pool re-assemble this corpus to two fingerprints), and a re-freeze whose evidence moved is
    refused for that reason alone, which would hide what this file pins.
    """
    from threetears.evals.analysis import freeze_reporter_case
    from threetears.evals.run import get_template

    frozen = json.loads(_FROZEN_V16_0.read_text(encoding="utf-8"))
    local_store = InMemoryDocumentStore()
    local = ops_fixture(store=local_store)
    local.host.eval_host.storage.save_template(reporter_template())
    local_store.upsert(frozen["analysis"])
    here = freeze_reporter_case(
        local.host.eval_host,
        template_id=reporter_template().id,
        campaign_id=local.campaign.id,
        scope_id=TOYHOST_SCOPE,
        analysis_id=frozen["analysis"]["id"],
        load_template=lambda template_id: get_template(local.host.eval_host, template_id, TOYHOST_SCOPE),
    )
    document = json.loads(json.dumps(frozen["reporter_case"]))
    for name in _EVIDENCE:
        document["host_payload"]["reporter_case"][name] = here.host_payload["reporter_case"][name]

    store = InMemoryDocumentStore()
    fixture = ops_fixture(store=store)
    fixture.host.eval_host.storage.save_template(reporter_template())
    store.upsert(frozen["analysis"])
    store.upsert(document)
    return fixture, EvalTestCase.from_dict(document), frozen["analysis"]["id"]


def test_a_case_stored_before_readers_were_recorded_reads_with_none() -> None:
    _, stored, _ = _frozen_store()
    (label,) = stored.host_payload["reporter_case"]["labels"]
    assert not {"reader", "collected_by"} & set(label), "the fixture is the shape stored before"

    case = reporter_case_of(stored)

    assert case is not None and [(label.reader, label.collected_by) for label in case.labels] == [(None, None)]


def test_an_attested_re_freeze_of_its_verdicts_returns_the_stored_case_without_supersedes() -> None:
    fixture, stored, analysis_id = _frozen_store()
    attests = _Attests(caller="agent:relay", names={PERSON: PERSON})

    receipt = reporter_case_freeze(
        _host(fixture, attests).eval_host,
        ReporterCaseFreeze(
            template_id=reporter_template().id,
            campaign_id=fixture.campaign.id,
            recorded_analysis_id=analysis_id,
            labels=[_label(reader=PERSON)],
        ),
        TOYHOST_SCOPE,
        collected_by="agent:relay",
    )

    assert receipt.test_case_id == stored.id
    (only,) = fixture.host.eval_host.storage.query_test_cases(TOYHOST_SCOPE, template_id=reporter_template().id)
    assert only.host_payload == stored.host_payload, "the stored case is never rewritten"


async def test_a_label_naming_no_reader_is_stored_as_the_build_before_stored_it() -> None:
    """The host's own door, naming no reader, writes each label without either key, which a v16.0 build reads."""
    from threetears.evals.analysis import freeze_reporter_case
    from threetears.evals.run import get_template

    fixture = off_loop(reporter_fixture)
    analysis_id = await _analysis_id(fixture)
    eval_host = _host(fixture, None).eval_host

    stored = off_loop(
        freeze_reporter_case,
        eval_host,
        template_id=reporter_template().id,
        campaign_id=fixture.campaign.id,
        scope_id=TOYHOST_SCOPE,
        analysis_id=analysis_id,
        labels=[_label()],
        load_template=lambda template_id: get_template(eval_host, template_id, TOYHOST_SCOPE),
    )

    (label,) = stored.host_payload["reporter_case"]["labels"]
    assert not {"reader", "collected_by"} & set(label)
    frozen = json.loads(_FROZEN_V16_0.read_text(encoding="utf-8"))
    (before,) = frozen["reporter_case"]["host_payload"]["reporter_case"]["labels"]
    assert set(label) == set(before), "the same keys the build before wrote"
