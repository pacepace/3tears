"""A person's rating relayed by a caller: recorded as the person's only when the host attests the caller for them.

A person's verdicts (a blind read, a review session) reach the store through someone else, often an agent acting for
an account. ``result_rate`` records whatever its caller says as an ``agent`` rating; ``result_rate_person`` records
it as the person's, naming its ``source`` and the caller that relayed it (``collected_by``), and only when the host's
own check (``EvalHost.attests_person``) vouches for that caller and that person. Pinned here:

- an attested rating is stored as the person's, with its source and relaying caller, and the host was asked about
  exactly that caller and person in that scope;
- it calibrates the judge as a person's, and follows its output by label key into a judge case's labels;
- the rating is stored under the host's name for the person, so two spellings it knows as one are one rater;
- a host that declines, or that supplies no check, refuses the call (``refused``, 403), logs a warning, and nothing
  is written, before the result is even read; a check that raises, or answers other than a name, propagates as the
  host's defect, and nothing is written;
- a relay replaces the person's earlier rating, direct or relayed, and says what it replaced; it dates the rating
  when the person gave it;
- an agent's rating through ``result_rate`` is unchanged, and carries no relay record;
- the document holds a relay record whole, on a person's rating only, and reads one stored before the field as none.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``result_rate_person``: ignoring the host's answer; recording when the host supplies no check; recording as an
  ``agent``, with or without the relay record; storing the caller's spelling of the person; reading a raising check
  as a refusal; accepting an answer that is not a name; logging no warning on a refusal; reporting no replacement,
  or not its source; logging no replacement.
- ``rate_result``: not storing ``source`` and ``collected_by``; not copying the score's label key; ignoring
  ``rated_at``; accepting a future one.
- ``CalibrationRating``: removing the relay-record check.
- the action: recording a fixed identity as ``collected_by`` rather than the call's caller.
- ``result_rate``: recording the agent as a ``person``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.actions import Caller, MountedTool, eval_catalogue, standard_tools
from threetears.evals.analysis import judge_agreement
from threetears.evals.ops import OpsHost, result_rate_person
from threetears.evals.kernel import ValidationFailedError
from threetears.evals.run import judge_case_labels, rate_result
from threetears.evals.schema import CalibrationRating, EvalResult, JudgeEvidence, RubricDim, RubricScore, label_key_of
from packages.evals.tests.factories import make_calibration_rating, make_eval_result, memory_storage
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_JUDGED_DIMENSION
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture
from packages.evals.tests.off_loop import off_loop

PERSON = "ana@example.com"
SOURCE = "blind_read:2026-10-17"
TONE = "conversation.tone"
KEY = label_key_of(
    JudgeEvidence(subject="a support agent", case_material="the queue holds 2 tickets", artifact="Agent: two."),
    RubricDim(name=TONE, description="Warm and on topic.", scale="ordinal", scoring_guide={"5": "warm"}),
)


@dataclass
class _Attests:
    """A host's check that vouches for one caller relaying the people it names, and records every question asked.

    ``names`` maps each spelling the host accepts to its own name for that person.
    """

    caller: str
    names: dict[str, str]
    asked: list[dict[str, str]] = field(default_factory=list)

    def __call__(self, *, scope_id: str, caller: str, person: str) -> str | None:
        self.asked.append({"scope_id": scope_id, "caller": caller, "person": person})
        return self.names.get(person) if caller == self.caller else None


def _attests(caller: str = CALLER.identity, person: str = PERSON) -> _Attests:
    """The check that vouches for ``caller`` relaying ``person``, under the spelling given."""
    return _Attests(caller=caller, names={person: person})


def _host(fixture: OpsFixture, attests: Any) -> OpsHost:
    """The fixture's host, with ``attests`` as its person check (``None`` for a host that supplies none)."""
    launch = fixture.host.launch
    return replace(fixture.host, launch=replace(launch, eval_host=replace(launch.eval_host, attests_person=attests)))


def _judged(fixture: OpsFixture, run_index: int = 0) -> EvalResult:
    storage = fixture.host.eval_host.storage
    result = off_loop(storage.query_eval_results_by_run, fixture.campaign.run_ids[run_index], TOYHOST_SCOPE)[0]
    assert result.judge_score(TOYHOST_JUDGED_DIMENSION) is not None, "the toy result is judged on that dimension"
    return result


def _evals() -> MountedTool:
    return {tool.name: tool for tool in eval_catalogue().mount_all(standard_tools())}["evals"]


def _relay(
    result_id: str, *, person: str = PERSON, score: int = 4, source: str = SOURCE, **extra: Any
) -> dict[str, Any]:
    return {
        "action": "result_rate_person",
        "result_id": result_id,
        "rubric_dim": TOYHOST_JUDGED_DIMENSION,
        "score": score,
        "rating_reason": "faithful layout",
        "person": person,
        "rating_source": source,
        **extra,
    }


# --- the action -------------------------------------------------------------------------------------------------


async def test_an_attested_persons_rating_is_recorded_as_theirs_with_its_source_and_its_relay() -> None:
    fixture = off_loop(ops_fixture)
    attests = _attests()
    judged = _judged(fixture)

    outcome = await _evals().call(_relay(judged.id), host=_host(fixture, attests), caller=CALLER)

    assert not outcome.is_error, outcome.text
    assert attests.asked == [{"scope_id": TOYHOST_SCOPE, "caller": CALLER.identity, "person": PERSON}]
    (rating,) = off_loop(fixture.host.eval_host.storage.query_calibration_ratings, TOYHOST_SCOPE, result_id=judged.id)
    assert (rating.rater, rating.rater_kind, rating.score) == (PERSON, "person", 4)
    assert (rating.source, rating.collected_by) == (SOURCE, CALLER.identity)
    assert outcome.structured is not None
    assert {key: outcome.structured[key] for key in ("rater_kind", "source", "collected_by", "replaced")} == {
        "rater_kind": "person",
        "source": SOURCE,
        "collected_by": CALLER.identity,
        "replaced": False,
    }


async def test_an_attested_persons_rating_calibrates_the_judge_as_a_persons() -> None:
    fixture = off_loop(ops_fixture)
    judged = _judged(fixture)

    outcome = await _evals().call(_relay(judged.id), host=_host(fixture, _attests()), caller=CALLER)

    assert not outcome.is_error, outcome.text
    ratings = off_loop(fixture.host.eval_host.storage.query_calibration_ratings, TOYHOST_SCOPE, result_id=judged.id)
    agreement = judge_agreement(ratings, [judged])
    assert not agreement.unpaired
    assert [(dim.rubric_dim, dim.n, dim.raters) for dim in agreement.dimensions] == [
        (TOYHOST_JUDGED_DIMENSION, 1, [PERSON])
    ]


def test_an_attested_persons_rating_follows_its_output_to_another_result_by_label_key() -> None:
    """Rated on one result, the label is found for a second result whose judge read the byte-identical output."""
    storage, _ = memory_storage()
    rated, twin = (
        make_eval_result(
            id=result_id,
            rubric_scores=[
                RubricScore(
                    dim=TONE,
                    score=3,
                    scale="ordinal",
                    served_model="judge-a",
                    output_fingerprint=KEY.output_fingerprint,
                    criterion_fingerprint=KEY.criterion_fingerprint,
                    axis="capability",
                    judge_temperature=0.0,
                )
            ],
        )
        for result_id in ("r-rated", "r-twin")
    )
    for result in (rated, twin):
        storage.save_eval_result(result)
    host = replace(toyhost_host(storage=storage), attests_person=_attests("agent:relay"))

    written = result_rate_person(
        host,
        rated.id,
        rated.scope_id,
        rubric_dim=TONE,
        score=2,
        reason="cold",
        person=PERSON,
        source=SOURCE,
        collected_by="agent:relay",
    )

    assert (written.output_fingerprint, written.criterion_fingerprint) == KEY
    labels = judge_case_labels(storage, twin.scope_id, twin, TONE, scale="ordinal", label_key=KEY)
    assert [(label.id, label.rater_kind, label.source) for label in labels] == [(written.rating_id, "person", SOURCE)]


@pytest.mark.parametrize(
    ("attests", "says"),
    [
        (_attests("agent:someone-else"), "does not attest 'agent:test' relaying"),
        (_attests(person="bo@example.com"), "does not attest 'agent:test' relaying"),
        (None, "vouches for no person"),
    ],
    ids=["another-caller", "another-person", "no-check"],
)
async def test_a_rating_the_host_does_not_attest_is_refused_and_nothing_is_written(
    attests: Any, says: str, caplog: pytest.LogCaptureFixture
) -> None:
    fixture = off_loop(ops_fixture)
    storage = fixture.host.eval_host.storage
    judged = _judged(fixture)
    host = _host(fixture, attests)

    with caplog.at_level(logging.WARNING, logger="threetears.evals"):
        outcome = await _evals().call(_relay(judged.id), host=host, caller=CALLER)
    unknown = await _evals().call(_relay("no-such-result"), host=host, caller=CALLER)

    assert outcome.is_error and outcome.refusal is not None
    assert (outcome.refusal.kind, outcome.refusal.status) == ("refused", 403)
    assert says in outcome.text and "result_rate" in outcome.text, "the refusal names the agent's own door"
    assert not off_loop(storage.query_calibration_ratings, TOYHOST_SCOPE)
    assert unknown.refusal is not None and unknown.refusal.status == 403, "asked before the result is read"
    assert any(
        record.levelno == logging.WARNING
        and record.getMessage().startswith(
            f"eval.result_rate_person refused caller={CALLER.identity} person={PERSON} scope={TOYHOST_SCOPE}:"
        )
        for record in caplog.records
    ), "an operator sees the refused attestation"


async def test_a_host_check_that_raises_propagates_and_nothing_is_written() -> None:
    """A check that raises is the host's defect: it reaches the caller as itself, never read as an attestation."""
    fixture = off_loop(ops_fixture)
    judged = _judged(fixture)

    def broken(*, scope_id: str, caller: str, person: str) -> str | None:
        raise RuntimeError("the host's account directory is down")

    with pytest.raises(RuntimeError, match="account directory is down"):
        await _evals().call(_relay(judged.id), host=_host(fixture, broken), caller=CALLER)
    assert not off_loop(fixture.host.eval_host.storage.query_calibration_ratings, TOYHOST_SCOPE)


async def test_a_host_check_that_answers_yes_rather_than_a_name_is_a_defect_and_nothing_is_written() -> None:
    fixture = off_loop(ops_fixture)
    judged = _judged(fixture)

    def yes(*, scope_id: str, caller: str, person: str) -> Any:
        return True

    with pytest.raises(TypeError, match="returned bool"):
        await _evals().call(_relay(judged.id), host=_host(fixture, yes), caller=CALLER)
    assert not off_loop(fixture.host.eval_host.storage.query_calibration_ratings, TOYHOST_SCOPE)


async def test_two_spellings_the_host_knows_as_one_person_are_one_rater() -> None:
    """The rating is stored under the host's name for the person, never the caller's spelling of them."""
    fixture = off_loop(ops_fixture)
    judged = _judged(fixture)
    attests = _Attests(caller=CALLER.identity, names={"Ana@Example.com": PERSON, PERSON: PERSON})
    host = _host(fixture, attests)
    tool = _evals()

    first = await tool.call(_relay(judged.id, person="Ana@Example.com", score=2), host=host, caller=CALLER)
    second = await tool.call(_relay(judged.id, person=PERSON, score=4), host=host, caller=CALLER)

    assert not first.is_error and not second.is_error, (first.text, second.text)
    ratings = off_loop(fixture.host.eval_host.storage.query_calibration_ratings, TOYHOST_SCOPE, result_id=judged.id)
    assert [(rating.rater, rating.score) for rating in ratings] == [(PERSON, 4)]
    assert second.structured is not None and second.structured["replaced"] is True


async def test_a_relay_replaces_the_persons_earlier_rating_and_says_what_it_replaced(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One person holds one rating of one dimension of one result: their direct rating, then a relay from one blind
    read, then one from another, each replacing the last, and each relay naming what it replaced."""
    fixture = off_loop(ops_fixture)
    storage = fixture.host.eval_host.storage
    judged = _judged(fixture)
    host = _host(fixture, _attests())
    tool = _evals()
    off_loop(
        rate_result,
        storage,
        result_id=judged.id,
        scope_id=TOYHOST_SCOPE,
        rubric_dim=TOYHOST_JUDGED_DIMENSION,
        rater=PERSON,
        rater_kind="person",
        score=3,
        reason="rated it myself",
    )

    with caplog.at_level(logging.INFO, logger="threetears.evals"):
        over_direct = await tool.call(_relay(judged.id, score=2, source="blind_read:a"), host=host, caller=CALLER)
        over_relay = await tool.call(_relay(judged.id, score=5, source="blind_read:b"), host=host, caller=CALLER)

    assert over_direct.structured is not None and over_relay.structured is not None
    assert (over_direct.structured["replaced"], over_direct.structured["replaced_source"]) == (True, None)
    assert (over_relay.structured["replaced"], over_relay.structured["replaced_source"]) == (True, "blind_read:a")
    (rating,) = off_loop(storage.query_calibration_ratings, TOYHOST_SCOPE, result_id=judged.id)
    assert (rating.score, rating.source, rating.collected_by) == (5, "blind_read:b", CALLER.identity)
    replaced = [r.getMessage() for r in caplog.records if r.getMessage().startswith("eval.rate_result replaced ")]
    assert len(replaced) == 2 and "replaced_source=blind_read:a" in replaced[1]


async def test_a_relay_dates_the_rating_when_the_person_gave_it() -> None:
    fixture = off_loop(ops_fixture)
    judged = _judged(fixture)
    host = _host(fixture, _attests())

    outcome = await _evals().call(_relay(judged.id, rated_at="2026-10-17T10:30:00-07:00"), host=host, caller=CALLER)

    assert not outcome.is_error, outcome.text
    (rating,) = off_loop(fixture.host.eval_host.storage.query_calibration_ratings, TOYHOST_SCOPE, result_id=judged.id)
    assert rating.rated_at == "2026-10-17T17:30:00+00:00"
    assert outcome.structured is not None and outcome.structured["rated_at"] == rating.rated_at


@pytest.mark.parametrize(
    ("rated_at", "says"),
    [("2026-10-17T10:30:00", "with an offset"), ("2999-01-01T00:00:00+00:00", "later than now"), ("17 Oct", "offset")],
    ids=["no-offset", "future", "not-iso"],
)
async def test_a_relay_time_that_is_not_a_past_dated_instant_is_refused(rated_at: str, says: str) -> None:
    fixture = off_loop(ops_fixture)
    judged = _judged(fixture)

    outcome = await _evals().call(_relay(judged.id, rated_at=rated_at), host=_host(fixture, _attests()), caller=CALLER)

    assert outcome.refusal is not None and outcome.refusal.kind == "validation_failed" and says in outcome.text
    assert not off_loop(fixture.host.eval_host.storage.query_calibration_ratings, TOYHOST_SCOPE)


async def test_an_agents_rating_is_still_the_agents_and_carries_no_relay() -> None:
    fixture = off_loop(ops_fixture)
    judged = _judged(fixture)
    host = _host(fixture, _attests())

    outcome = await _evals().call(
        {
            "action": "result_rate",
            "result_id": judged.id,
            "rubric_dim": TOYHOST_JUDGED_DIMENSION,
            "score": 4,
            "rating_reason": "faithful layout",
        },
        host=host,
        caller=CALLER,
    )

    assert not outcome.is_error, outcome.text
    (rating,) = off_loop(fixture.host.eval_host.storage.query_calibration_ratings, TOYHOST_SCOPE, result_id=judged.id)
    assert (rating.rater, rating.rater_kind, rating.source, rating.collected_by) == (
        CALLER.identity,
        "agent",
        None,
        None,
    )
    assert [u.reason for u in judge_agreement([rating], [judged]).unpaired] == ["rated_by_an_agent"]


async def test_the_person_and_the_agent_rating_the_same_result_stand_side_by_side() -> None:
    """The relay names the person as the rater, so the agent's own opinion of the same result is a second rating."""
    fixture = off_loop(ops_fixture)
    judged = _judged(fixture)
    host = _host(fixture, _attests())
    tool = _evals()

    await tool.call(_relay(judged.id, score=2), host=host, caller=CALLER)
    await tool.call(
        {
            "action": "result_rate",
            "result_id": judged.id,
            "rubric_dim": TOYHOST_JUDGED_DIMENSION,
            "score": 5,
            "rating_reason": "mine",
        },
        host=host,
        caller=CALLER,
    )

    ratings = off_loop(fixture.host.eval_host.storage.query_calibration_ratings, TOYHOST_SCOPE, result_id=judged.id)
    assert sorted((r.rater_kind, r.rater, r.score) for r in ratings) == [
        ("agent", CALLER.identity, 5),
        ("person", PERSON, 2),
    ]


def test_neither_rating_action_lets_the_caller_name_the_rater_kind_or_the_relay() -> None:
    tool = _evals()
    for name in ("result_rate", "result_rate_person"):
        action = tool.action(name)
        assert action is not None
        assert not {"rater_kind", "collected_by"} & set(action.params.model_fields), name


async def test_the_relay_is_the_host_resolved_caller_never_a_name_the_call_gives() -> None:
    """``collected_by`` is the caller the host resolved for the call: another caller is recorded as itself."""
    other = Caller(scope_id=TOYHOST_SCOPE, identity="agent:other")
    fixture = off_loop(ops_fixture)
    judged = _judged(fixture)
    host = _host(fixture, _attests(other.identity))

    outcome = await _evals().call(_relay(judged.id), host=host, caller=other)

    assert not outcome.is_error, outcome.text
    (rating,) = off_loop(fixture.host.eval_host.storage.query_calibration_ratings, TOYHOST_SCOPE, result_id=judged.id)
    assert rating.collected_by == other.identity


# --- the write and the document ---------------------------------------------------------------------------------


def test_rate_result_refuses_a_relay_record_on_an_agents_rating_as_a_validation_failure() -> None:
    storage, _ = memory_storage()
    result = make_eval_result(
        id="r-1",
        rubric_scores=[RubricScore(dim=TONE, score=3, scale="ordinal", axis="capability", judge_temperature=0.0)],
    )
    storage.save_eval_result(result)

    with pytest.raises(ValidationFailedError, match="an agent's rating"):
        rate_result(
            storage,
            result_id=result.id,
            scope_id=result.scope_id,
            rubric_dim=TONE,
            rater="agent:test",
            rater_kind="agent",
            score=3,
            reason="mine",
            source=SOURCE,
            collected_by="agent:test",
        )
    assert not storage.query_calibration_ratings(result.scope_id)


@pytest.mark.parametrize(
    ("fields", "says"),
    [
        ({"rater_kind": "person", "source": SOURCE}, "sets only source"),
        ({"rater_kind": "person", "collected_by": "agent:test"}, "sets only collected_by"),
        ({"rater_kind": "agent", "source": SOURCE, "collected_by": "agent:test"}, "an agent's rating"),
    ],
    ids=["source-alone", "collected-by-alone", "on-an-agent"],
)
def test_a_relay_record_is_whole_and_on_a_persons_rating_only(fields: dict[str, Any], says: str) -> None:
    document = make_calibration_rating().to_dict()
    del document["id"]  # derived again from the fields, the kind among them

    with pytest.raises(ValidationError, match=says):
        CalibrationRating.model_validate({**document, **fields})


def test_a_relayed_rating_round_trips_and_one_stored_before_the_field_reads_with_none() -> None:
    relayed = make_calibration_rating(rater_kind="person", source=SOURCE, collected_by="agent:test")
    assert CalibrationRating.from_dict(relayed.to_dict()) == relayed

    before = {
        key: value
        for key, value in make_calibration_rating().to_dict().items()
        if key not in ("source", "collected_by")
    }
    before["schema_minor"] = 0

    read = CalibrationRating.from_dict(before)

    assert (read.source, read.collected_by) == (None, None)
