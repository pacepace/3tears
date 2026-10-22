"""An authored judge case is a person's verdict, so the host attests who may author one, and for whom.

Every judge case label is read downstream as a person's rating: a judge campaign's agreement with people, and the
judge profile it records. An authored case's verdict is its one label, by its author, so ``judge_case_author`` asks
the host's own check (``EvalHost.attests_person``) before it writes, exactly as ``result_rate_person`` does: the caller
(``collected_by``) authors for the person it names (``person``), or for itself when it names none. Pinned here:

- a host that supplies no check, or one that refuses, refuses the call (``refused``, 403), logs a warning, and nothing
  is written, before either template is read;
- an attested case is stored under the host's name for the person, as its author and its label's rater, and records
  the caller that relayed it;
- omitting ``person`` asks the host about the caller authoring as itself;
- a check that answers other than a name, or raises, propagates as the host's defect, and nothing is written;
- the action records the caller the host resolved for the call, and lets no call name the relay; the receipt and
  the listing name it;
- a relay is provenance, never content: a case stored before relays were recorded (a frozen v16.0 document) reads,
  digests to its stored ``content_hash`` and is named again, relayed or not, by its stored id; an unrelayed case is
  stored byte for byte as that build stored it (no ``collected_by`` key), and only a relayed one carries the key.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``judge_case_author``: not asking the host; asking after the templates are read; storing the caller's spelling of
  the person; recording no ``collected_by``; asking about another person than the one named, or the caller.
- ``attested_person``: recording when the host supplies no check; reading a non-name answer as a name or a refusal;
  logging no warning on a refusal.
- the action: recording a fixed identity as ``collected_by`` rather than the call's caller.
- ``JudgeCase.content_digest``: keeping ``collected_by`` in the pre-image. ``judge_case_payload``: writing
  ``collected_by: null`` on an unrelayed case, or dropping a relay.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from threetears.evals.actions import Caller, MountedTool, eval_catalogue, standard_tools
from threetears.evals.kernel import (
    JUDGE_KIND,
    AuthoredJudgeCase,
    EvalServiceError,
    EvalStorage,
    JudgeCase,
    judge_case_of,
    judge_case_payload,
)
from threetears.evals.ops import JudgeCaseAuthor, OpsHost, judge_case_author
from threetears.evals.ops.judge_cases import judge_cases_list
from threetears.evals.run import freeze_judge_cases
from threetears.evals.schema import EvalTemplate, EvalTestCase, RubricDim
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_INSTANT
from packages.evals.tests.fixtures.toyhost.judge import FAITHFULNESS_DIM, toyhost_judged_template
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture

PERSON = "ana@example.com"


@dataclass
class _Attests:
    """A host's check that vouches for one caller authoring for the people it names, and records every question asked.

    ``names`` maps each spelling the host accepts to its own name for that person.
    """

    caller: str
    names: dict[str, str]
    asked: list[dict[str, str]] = field(default_factory=list)

    def __call__(self, *, scope_id: str, caller: str, person: str) -> str | None:
        self.asked.append({"scope_id": scope_id, "caller": caller, "person": person})
        return self.names.get(person) if caller == self.caller else None


def _judge_template() -> EvalTemplate:
    return EvalTemplate(
        id="attested-judge",
        scope_id=TOYHOST_SCOPE,
        name="Attested judge",
        intent="Score how faithfully an extraction reproduces its invoice.",
        candidate_kind=JUDGE_KIND,
        created_at=TOYHOST_INSTANT,
        updated_at=TOYHOST_INSTANT,
    )


def _fixture() -> OpsFixture:
    fixture = off_loop(ops_fixture)
    storage = fixture.host.eval_host.storage
    off_loop(storage.save_template, toyhost_judged_template())
    off_loop(storage.save_template, _judge_template())
    return fixture


def _host(fixture: OpsFixture, attests: Any) -> OpsHost:
    """The fixture's host, with ``attests`` as its person check (``None`` for a host that supplies none)."""
    launch = fixture.host.launch
    return replace(fixture.host, launch=replace(launch, eval_host=replace(launch.eval_host, attests_person=attests)))


def _case(**changes: Any) -> dict[str, Any]:
    """The action's arguments for one hard negative on the toy criterion."""
    return {
        "template_id": _judge_template().id,
        "criterion_template_id": toyhost_judged_template().id,
        "criterion": FAITHFULNESS_DIM,
        "output": json.dumps({"invoice_number": "INV-0042"}),
        "case_material": "Invoice doc-42",
        "expected": 1,
        "note": "three of four fields missing is not faithful",
        **changes,
    }


def _tool() -> MountedTool:
    return next(tool for tool in eval_catalogue().mount_all(standard_tools()) if tool.action("judge_case_author"))


def _stored_cases(fixture: OpsFixture) -> list[Any]:
    return off_loop(fixture.host.eval_host.storage.query_test_cases, TOYHOST_SCOPE, template_id=_judge_template().id)


# --- refused: nothing is written ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("attests", "says"),
    [
        (_Attests(caller="agent:someone-else", names={PERSON: PERSON}), "does not attest 'agent:test' relaying"),
        (_Attests(caller=CALLER.identity, names={"bo@example.com": "bo@example.com"}), "does not attest 'agent:test'"),
        (None, "vouches for no person"),
    ],
    ids=["another-caller", "another-person", "no-check"],
)
async def test_a_case_the_host_does_not_attest_is_refused_and_nothing_is_written(
    attests: Any, says: str, caplog: pytest.LogCaptureFixture
) -> None:
    fixture = _fixture()
    host = _host(fixture, attests)

    with caplog.at_level(logging.WARNING, logger="threetears.evals"):
        outcome = await _tool().call({"action": "judge_case_author", **_case(person=PERSON)}, host=host, caller=CALLER)
    unknown = await _tool().call(
        {
            "action": "judge_case_author",
            **_case(person=PERSON, template_id="no-such-judge", criterion_template_id="no-such-subject"),
        },
        host=host,
        caller=CALLER,
    )

    assert outcome.is_error and outcome.refusal is not None
    assert (outcome.refusal.kind, outcome.refusal.status) == ("refused", 403)
    assert says in outcome.text and "judge case" in outcome.text
    assert not _stored_cases(fixture)
    assert unknown.refusal is not None and unknown.refusal.status == 403, "asked before either template is read"
    assert any(
        record.levelno == logging.WARNING
        and record.getMessage().startswith(
            f"eval.judge_case_author refused caller={CALLER.identity} person={PERSON} scope={TOYHOST_SCOPE}:"
        )
        for record in caplog.records
    ), "an operator sees the refused attestation"


def test_a_host_with_no_check_refuses_the_operation_itself() -> None:
    """The operation, not only the action, refuses: no surface writes a person's verdict the host never attested."""
    fixture = _fixture()

    with pytest.raises(EvalServiceError, match="vouches for no person") as refused:
        judge_case_author(
            _host(fixture, None).eval_host,
            JudgeCaseAuthor.model_validate(_case()),
            TOYHOST_SCOPE,
            collected_by=CALLER.identity,
        )
    assert refused.value.status_code == 403
    assert not _stored_cases(fixture)


@pytest.mark.parametrize("answer", [True, "", "   "], ids=["yes", "empty", "blank"])
def test_a_host_check_that_answers_other_than_a_name_is_a_defect_and_nothing_is_written(answer: Any) -> None:
    """A bare yes, or a blank name, is the host's defect, raised before a template is read, never the caller's."""
    fixture = _fixture()

    def answers(*, scope_id: str, caller: str, person: str) -> Any:
        return answer

    with pytest.raises(TypeError, match=f"attests_person returned {answer!r}"):
        judge_case_author(
            _host(fixture, answers).eval_host,
            JudgeCaseAuthor.model_validate(_case(template_id="no-such-judge")),
            TOYHOST_SCOPE,
            collected_by=CALLER.identity,
        )
    assert not _stored_cases(fixture)


async def test_a_host_check_that_raises_propagates_and_nothing_is_written() -> None:
    fixture = _fixture()

    def broken(*, scope_id: str, caller: str, person: str) -> str | None:
        raise RuntimeError("the host's account directory is down")

    with pytest.raises(RuntimeError, match="account directory is down"):
        await _tool().call({"action": "judge_case_author", **_case()}, host=_host(fixture, broken), caller=CALLER)
    assert not _stored_cases(fixture)


# --- attested: the person's verdict, and who relayed it -------------------------------------------------------------


def test_an_attested_case_is_the_persons_under_the_hosts_name_and_records_its_relay() -> None:
    """The caller's spelling of the person is not stored: the host's name for them is the author and the rater."""
    fixture = _fixture()
    attests = _Attests(caller=CALLER.identity, names={"Ana@Example.com": PERSON})

    report = judge_case_author(
        _host(fixture, attests).eval_host,
        JudgeCaseAuthor.model_validate(_case(person="Ana@Example.com")),
        TOYHOST_SCOPE,
        collected_by=CALLER.identity,
    )

    assert attests.asked == [{"scope_id": TOYHOST_SCOPE, "caller": CALLER.identity, "person": "Ana@Example.com"}]
    (frozen,) = report.cases
    assert frozen.created and (frozen.authored_by, frozen.collected_by) == (PERSON, CALLER.identity)
    (stored,) = _stored_cases(fixture)
    case = judge_case_of(stored)
    assert case is not None and case.source.authored is not None
    (label,) = case.labels
    assert (label.rater, label.rating_id) == (PERSON, None)
    assert (case.source.authored.author, case.source.authored.collected_by) == (PERSON, CALLER.identity)


def test_omitting_the_person_asks_whether_the_caller_authors_as_itself() -> None:
    """A person's own account authoring a case names no one else: the host is asked about the caller as the person."""
    fixture = _fixture()
    attests = _Attests(caller=PERSON, names={PERSON: PERSON})

    report = judge_case_author(
        _host(fixture, attests).eval_host, JudgeCaseAuthor.model_validate(_case()), TOYHOST_SCOPE, collected_by=PERSON
    )

    assert attests.asked == [{"scope_id": TOYHOST_SCOPE, "caller": PERSON, "person": PERSON}]
    (frozen,) = report.cases
    assert (frozen.authored_by, frozen.collected_by) == (PERSON, PERSON)


def test_the_same_case_relayed_by_another_caller_is_the_stored_case() -> None:
    """Who relayed a case is provenance, not content: a second relay of it names the stored case and leaves it as is."""
    fixture = _fixture()
    attests = _Attests(caller=CALLER.identity, names={PERSON: PERSON})
    other = _Attests(caller="agent:other", names={PERSON: PERSON})
    case = JudgeCaseAuthor.model_validate(_case(person=PERSON))

    first = judge_case_author(_host(fixture, attests).eval_host, case, TOYHOST_SCOPE, collected_by=CALLER.identity)
    again = judge_case_author(_host(fixture, other).eval_host, case, TOYHOST_SCOPE, collected_by="agent:other")

    assert [c.test_case_id for c in again.cases] == [c.test_case_id for c in first.cases]
    assert not again.cases[0].created
    assert again.cases[0].collected_by is None, "a repeat recorded no relay: the stored one stands"
    (stored,) = _stored_cases(fixture)
    kept = judge_case_of(stored)
    assert kept is not None and kept.source.authored is not None
    assert kept.source.authored.collected_by == CALLER.identity, "the stored case is never rewritten"
    listing = judge_cases_list(fixture.host.eval_host, _judge_template().id, TOYHOST_SCOPE)
    (entry,) = listing.cases
    assert (entry.authored_by, entry.collected_by) == (PERSON, CALLER.identity), "the listing names who relayed it"


# --- the action ------------------------------------------------------------------------------------------------


async def test_the_action_records_the_host_resolved_caller_as_the_relay() -> None:
    """``collected_by`` is the caller the host resolved for the call: another caller is asked about, and recorded."""
    other = Caller(scope_id=TOYHOST_SCOPE, identity="agent:other")
    fixture = _fixture()
    attests = _Attests(caller=other.identity, names={PERSON: PERSON})

    outcome = await _tool().call(
        {"action": "judge_case_author", **_case(person=PERSON)}, host=_host(fixture, attests), caller=other
    )

    assert not outcome.is_error, outcome.text
    assert attests.asked == [{"scope_id": TOYHOST_SCOPE, "caller": other.identity, "person": PERSON}]
    (line,) = [line for line in outcome.text.splitlines() if line.startswith("- test_case_id=")]
    assert f"authored_by={PERSON}" in line and f"collected_by={other.identity}" in line, "the receipt names the relay"
    (stored,) = _stored_cases(fixture)
    case = judge_case_of(stored)
    assert case is not None and case.source.authored is not None
    assert (case.source.authored.author, case.source.authored.collected_by) == (PERSON, other.identity)


def test_the_action_lets_no_call_name_the_relay_or_the_author() -> None:
    action = _tool().action("judge_case_author")
    assert action is not None
    assert not {"collected_by", "author"} & set(action.params.model_fields)
    assert "person" in action.params.model_fields


# --- the stored case: a relay is provenance, never content --------------------------------------------------------

#: An authored judge case as the build before relays were recorded stored it (durable v16.0's judge case shape, written
#: by ``freeze_judge_cases(authored=)`` at 71c9f5f52), with its id and ``content_hash``. Never edit it: it is what
#: production stores hold.
_FROZEN_V16_0 = Path(__file__).resolve().parent / "fixtures" / "judge_cases" / "authored_case_v16_0.json"


def _frozen_v16_0() -> EvalTestCase:
    return EvalTestCase.from_dict(json.loads(_FROZEN_V16_0.read_text(encoding="utf-8")))


def _frozen_v16_0_authored(**changes: Any) -> AuthoredJudgeCase:
    """The case the frozen document holds, as its author wrote it."""
    return AuthoredJudgeCase(
        criterion=RubricDim(
            name="extraction.faithful",
            description="Every field matches the invoice.",
            scale="ordinal",
            scoring_guide={"1": "nothing matches", "5": "every field matches"},
        ),
        output='{"invoice_number": "INV-0042"}',
        case_material="Invoice INV-0042 from Acme, total 120.00 EUR.",
        intent="Extract the invoice's fields.",
        expected=1,
        author="reviewer@example.com",
        note="three of four fields missing is not faithful",
        **changes,
    )


def _frozen_v16_0_template() -> EvalTemplate:
    stored = _frozen_v16_0()
    return EvalTemplate(
        id=stored.template_id,
        scope_id=stored.scope_id,
        name="Frozen judge",
        intent="Score how faithfully an extraction reproduces its invoice.",
        candidate_kind=JUDGE_KIND,
    )


def test_a_case_stored_before_relays_were_recorded_reads_and_keeps_its_id_and_hash() -> None:
    stored = _frozen_v16_0()

    case = judge_case_of(stored)

    assert case is not None and case.source.authored is not None and case.source.authored.collected_by is None
    assert case.content_digest() == stored.content_hash


@pytest.mark.parametrize("collected_by", [None, "agent:relay"], ids=["unrelayed", "relayed"])
def test_the_same_case_written_now_names_the_stored_one_relayed_or_not(collected_by: str | None) -> None:
    """A reseed, or a relay of the same case, finds the stored case by its id rather than writing a second beside it."""
    stored = _frozen_v16_0()
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_test_case(stored)

    report = freeze_judge_cases(
        storage,
        template=_frozen_v16_0_template(),
        scope_id=stored.scope_id,
        authored=[_frozen_v16_0_authored(collected_by=collected_by)],
    )

    (frozen,) = report.cases
    assert (frozen.test_case_id, frozen.created) == (stored.id, False)


def test_an_unrelayed_case_is_stored_exactly_as_a_v16_0_build_stored_it() -> None:
    """No ``collected_by`` key on an unrelayed case, so a build at durable v16.0, which refuses a key its judge case does
    not declare, still reads every seeded, reset, quick-path or host-frozen case."""
    payload = judge_case_payload(JudgeCase.from_authored(_frozen_v16_0_authored()))

    assert payload == _frozen_v16_0().host_payload
    assert "collected_by" not in payload["judge_case"]["source"]["authored"]


def test_a_relayed_case_carries_who_relayed_it() -> None:
    payload = judge_case_payload(JudgeCase.from_authored(_frozen_v16_0_authored(collected_by="agent:relay")))

    assert payload["judge_case"]["source"]["authored"]["collected_by"] == "agent:relay"
    case = JudgeCase.model_validate(payload["judge_case"])
    assert case.source.authored is not None and case.source.authored.collected_by == "agent:relay"
