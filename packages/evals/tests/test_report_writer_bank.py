"""The writer case bank: a host mints writer cases from its own campaigns into a dev set and a held-out set.

These tests seed the bank from the package fixtures (:mod:`packages.evals.tests.writer_case_support`), as a host
mints from its own snapshots, and mint from the toy host's campaign as a host mints from a campaign. They hold that
both sets resolve and no case sits in both; that a group's cases sit on one side, its recorded side winning over the
digest rule; that minting again returns the same case in the same set; that a mint differing from the source's case,
a withheld note the shipped skill lacks, and a case listed on the other side are each refused; and that a launch over
a named set runs exactly its cases, each withholding its own note, with nothing chosen at launch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, OpsFixture
from packages.evals.tests.report_writer_support import MODEL, FakeToolTurnClient, reply
from packages.evals.tests.writer_case_support import SEED, WRITER_TEMPLATE, seed_writer_cases, writer_host
from threetears.evals.actions.report_writing import report_writer_kind
from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.host.kinds import freeze
from threetears.evals.ops import OpsHost
from threetears.evals.ops.report_writer.tally import ReportBudget
from threetears.evals.ops.report_writer.writer_bank import (
    MintedWriterCase,
    WriterCaseMint,
    WriterSplit,
    writer_case_mint,
    writer_split,
)
from threetears.evals.ops.report_writer.writer_contract import (
    WRITER_KIND,
    WRITER_KIND_CONTRACT,
    EffortWord,
    WriterOverlays,
)
from threetears.evals.ops.report_writer.writer_kind import writer_case_of, writer_subject_components
from threetears.evals.run.case_sets import mint_case_set
from threetears.evals.run.runner import RunnerOptions, execute_run
from threetears.evals.schema.models import EvalTestCase
from threetears.evals.schema.subject import SubjectSnapshot
from threetears.evals.schema.turns import ToolTurnClient, TurnText

#: The default split's sets.
_SPLIT = WriterSplit()


async def _mint(host: OpsHost, **named: object) -> MintedWriterCase:
    """Mint into the writer template as :data:`CALLER`."""
    return await writer_case_mint(
        host,
        WriterCaseMint.model_validate({"template_id": WRITER_TEMPLATE.id, **named}),
        CALLER.scope_id,
        created_by=CALLER.identity,
    )


def _stored(host: OpsHost) -> list[EvalTestCase]:
    """Every case stored under the writer template."""
    return off_loop(host.eval_host.storage.query_test_cases, CALLER.scope_id, template_id=WRITER_TEMPLATE.id)


async def test_the_seed_mints_every_fixture_into_two_sets_that_resolve_and_share_no_case() -> None:
    host = off_loop(writer_host).host

    minted = await seed_writer_cases(host)

    sets = off_loop(writer_split, host, WRITER_TEMPLATE, CALLER.scope_id, split=_SPLIT)
    dev_ids = [case.id for case in sets.dev[1]]
    held_out_ids = [case.id for case in sets.held_out[1]]
    assert dev_ids and held_out_ids, "both sides hold cases"
    assert not set(dev_ids) & set(held_out_ids)
    assert sorted(dev_ids + held_out_ids) == sorted(receipt.test_case_id for receipt in minted.values())
    # Each set grew one version per case it gained, in the order the cases were minted.
    assert sets.dev[0].version == len(dev_ids) and sets.held_out[0].version == len(held_out_ids)
    for name, receipt in minted.items():
        group, withhold = SEED[name]
        listed = held_out_ids if receipt.side == "held_out" else dev_ids
        assert receipt.test_case_id in listed, name
        assert receipt.group == group and receipt.withhold == list(withhold) and not receipt.reused
    # A misfit is minted in its reference's group, and a group never straddles the split.
    sides: dict[str, set[str]] = {}
    for receipt in minted.values():
        sides.setdefault(receipt.group, set()).add(receipt.side)
    assert all(len(side) == 1 for side in sides.values()), sides
    # Every stored case records where it came from, what it withholds and where its mint put it.
    for stored in _stored(host):
        case = writer_case_of(stored)
        assert case is not None and case.origin is not None and case.case_version == 1
        assert case.origin.source == "snapshot" and case.origin.host_id == host.eval_host.profile.host_id
        assert case.origin.minted_by == CALLER.identity and case.origin.skill == load_skill().digest
        assert case.origin.case_set == _SPLIT.set_of(case.origin.side)


async def test_minting_a_case_again_returns_it_in_its_set_and_mints_no_version() -> None:
    host = off_loop(writer_host).host
    first = await seed_writer_cases(host)

    again = await seed_writer_cases(host)

    for name, receipt in again.items():
        assert receipt.reused, name
        assert (receipt.test_case_id, receipt.side) == (first[name].test_case_id, first[name].side)
        assert receipt.case_set.name == first[name].case_set.name
    assert len(_stored(host)) == len(SEED)
    sets = off_loop(writer_split, host, WRITER_TEMPLATE, CALLER.scope_id, split=_SPLIT)
    assert {receipt.case_set for receipt in again.values()} == {sets.dev[0].ref, sets.held_out[0].ref}


async def test_a_mint_differing_from_its_sources_case_is_refused_and_stores_nothing() -> None:
    host = off_loop(writer_host).host
    kairo = (await seed_writer_cases(host))["kairo round 2"]

    with pytest.raises(ValidationFailedError, match="this mint differs: withholding"):
        await _mint(host, snapshot_id=kairo.snapshot_id, withhold=[])

    assert len(_stored(host)) == len(SEED)


async def test_a_withheld_note_the_shipped_skill_lacks_is_refused_before_anything_is_frozen() -> None:
    fixture: OpsFixture = off_loop(writer_host)

    with pytest.raises(ValidationFailedError, match="no exemplar is named nowhere.md"):
        await _mint(fixture.host, campaign_id=fixture.campaign.id, withhold=["nowhere.md"])

    assert _stored(fixture.host) == []
    assert off_loop(fixture.host.eval_host.storage.query_case_sets, CALLER.scope_id) == []


async def test_a_campaign_is_minted_through_its_freeze_and_a_placed_group_keeps_its_side() -> None:
    fixture: OpsFixture = off_loop(writer_host)
    host = fixture.host
    seeded = await seed_writer_cases(host)
    held_out_group = next(receipt.group for receipt in seeded.values() if receipt.side == "held_out")

    # A share this small would put a new group on the dev side; the group's recorded side wins.
    minted = await _mint(
        host,
        campaign_id=fixture.campaign.id,
        group=held_out_group,
        split=WriterSplit(held_out_fraction=0.000001).model_dump(),
    )
    again = await _mint(host, campaign_id=fixture.campaign.id, group=held_out_group)

    assert minted.side == "held_out" and minted.source_key == f"campaign:{fixture.campaign.id}"
    assert again.reused and again.test_case_id == minted.test_case_id and again.case_set == minted.case_set
    (stored,) = [case for case in _stored(host) if case.id == minted.test_case_id]
    case = writer_case_of(stored)
    assert case is not None and case.origin is not None
    assert (case.origin.source, case.origin.campaign_id) == ("campaign", fixture.campaign.id)


async def test_a_case_listed_on_the_other_side_is_refused_at_mint_and_when_the_split_resolves() -> None:
    host = off_loop(writer_host).host
    seeded = await seed_writer_cases(host)
    kairo = seeded["kairo round 2"]
    other = _SPLIT.set_of("dev" if kairo.side == "held_out" else "held_out")
    storage = host.eval_host.storage
    (latest, *_) = off_loop(storage.query_case_sets, CALLER.scope_id, name=other)
    off_loop(
        lambda: mint_case_set(
            storage,
            scope_id=CALLER.scope_id,
            name=other,
            template_id=WRITER_TEMPLATE.id,
            test_case_ids=[*latest.test_case_ids, kairo.test_case_id],
        )
    )

    with pytest.raises(ValidationFailedError, match=f"set {other!r} lists it too"):
        await _mint(host, snapshot_id=kairo.snapshot_id, withhold=["kairo-round2.md"], group="kairo")
    with pytest.raises(ValidationFailedError, match=f"{kairo.test_case_id} are listed by"):
        off_loop(writer_split, host, WRITER_TEMPLATE, CALLER.scope_id, split=_SPLIT)


async def test_minting_into_a_template_of_another_kind_is_refused() -> None:
    fixture: OpsFixture = off_loop(writer_host)
    other = toyhost_template()

    with pytest.raises(ValidationFailedError, match=f"not {WRITER_KIND!r}"):
        await writer_case_mint(
            fixture.host,
            WriterCaseMint(template_id=other.id, campaign_id=fixture.campaign.id),
            CALLER.scope_id,
            created_by=CALLER.identity,
        )


def test_a_mint_names_exactly_one_source_and_a_split_two_sets() -> None:
    with pytest.raises(ValueError, match="none was named"):
        WriterCaseMint(template_id=WRITER_TEMPLATE.id)
    with pytest.raises(ValueError, match="campaign_id and snapshot_id were named"):
        WriterCaseMint(template_id=WRITER_TEMPLATE.id, campaign_id="c", snapshot_id="s")
    with pytest.raises(ValueError, match="two names"):
        WriterSplit(dev="one", held_out="one")


# parity-with: threetears.evals.ops.report_writer.loop.TurnModel
@dataclass
class _FakeRefusingModel:
    """One cell's model: an analyst that never submits a document, its system text kept by the snapshot it read."""

    systems: dict[str, list[str]]
    model_name: str = MODEL
    sessions: dict[str, FakeToolTurnClient] = field(default_factory=dict)

    async def turn_pricing_refusal(self) -> str | None:
        return None

    def turn_session(self, name: str, *, snapshot: str) -> ToolTurnClient:
        client = FakeToolTurnClient(*(reply(TurnText("I have no report yet.")) for _ in range(4)))
        self.sessions[name] = client
        self.systems.setdefault(snapshot, []).append(name)
        return client


async def test_a_launch_over_a_named_set_runs_exactly_its_cases_each_withholding_its_own_note() -> None:
    host = off_loop(writer_host).host
    seeded = await seed_writer_cases(host)
    sets = off_loop(writer_split, host, WRITER_TEMPLATE, CALLER.scope_id, split=_SPLIT)
    case_set, cases = sets.held_out
    built: list[_FakeRefusingModel] = []

    def models(effort: EffortWord | None) -> _FakeRefusingModel:
        model = _FakeRefusingModel(systems={})
        built.append(model)
        return model

    overlays = WRITER_KIND_CONTRACT.validate_overlays({"revisions": 1})
    run = make_eval_run(
        scope_id=CALLER.scope_id,
        template_id=WRITER_TEMPLATE.id,
        candidate_kind=WRITER_KIND,
        candidate_model=MODEL,
        overlays=freeze(overlays),
        subject_snapshot=SubjectSnapshot(
            subject_id="report writer",
            subject_label="report writer",
            components=writer_subject_components(load_skill()),
            state={},
        ),
        k_runs=1,
        test_case_ids=case_set.test_case_ids,
    )
    kind = report_writer_kind(
        host,
        CALLER,
        overlays=WriterOverlays.model_validate(run.overlays),
        models=models,
        budget=ReportBudget(usd=5.0),
    )

    await execute_run(
        host.eval_host,
        run=run,
        template=WRITER_TEMPLATE,
        test_cases=cases,
        judge_service=None,
        options=RunnerOptions(candidate_kinds={WRITER_KIND: lambda _cell: kind}),
    )

    results = off_loop(host.eval_host.storage.query_eval_results_by_run, run.id, run.scope_id)
    assert sorted(result.test_case_id for result in results) == sorted(case_set.test_case_ids)
    assert all(result.infra_error is None for result in results), [result.infra_error for result in results]
    # Each cell's analyst was not given its own case's note, and was given every other: the case chose, not the launch.
    withheld = {receipt.snapshot_id: receipt.withhold for receipt in seeded.values()}
    analysts = [(snapshot, model.sessions["analyst"]) for model in built for snapshot in model.systems]
    assert len(analysts) == len(case_set.test_case_ids)
    notes = load_skill().analyst().exemplars
    for snapshot, client in analysts:
        (system,) = {request["system"] for request in client.requests}
        for note in notes:
            given = f'<file path="exemplars/{note}">' in system
            assert given is (note not in withheld[snapshot]), (snapshot, note)
