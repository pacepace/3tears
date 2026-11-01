"""The writer case bank's first seed: a writer case of every package fixture, minted as a host mints its own.

A host mints writer cases from its own campaigns (:func:`~threetears.evals.ops.report_writer.writer_bank.writer_case_mint`);
the package's fixtures are the first cases a bank holds. Each fixture's snapshot (:func:`fixture_snapshots`: r1 to
r5, the six misfits, and Kairo round 2) is restored into the host's scope, as a host's own frozen snapshot would be
there, and minted by its id, withholding the exemplar note reasoned on its own campaign: the note each recorded run
of ``fixtures/report_docs/prototype5`` withheld. A misfit is a reference input reshaped, so it is minted in its
reference's split group and sits on its reference's side: a change is never tuned on one and decided on the other.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from packages.evals.tests.analyst_support import KAIRO, fixture_snapshots
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.ops_support import CALLER, OpsFixture, ops_fixture
from threetears.evals.kernel.host.measures import MeasureRegistry
from threetears.evals.ops import OpsHost
from threetears.evals.ops.analyst import report_snapshot_restore
from threetears.evals.ops.report_writer.writer_bank import (
    MintedWriterCase,
    WriterCaseMint,
    WriterSplit,
    writer_case_mint,
)
from threetears.evals.ops.report_writer.writer_contract import WRITER_KIND, WRITER_KIND_MEASURES, writer_kind_contract
from threetears.evals.run import LaunchSettings
from threetears.evals.schema.models import EvalTemplate

#: The writer template every seeded case is minted into.
WRITER_TEMPLATE: Final = EvalTemplate(
    id="writer-template",
    scope_id=CALLER.scope_id,
    name="Report writer",
    intent="Write the report.",
    candidate_kind=WRITER_KIND,
)

#: Each fixture's split group and the notes its case withholds: its reference's note, Kairo's own, and none for r5,
#: which no note was reasoned on.
SEED: Final[dict[str, tuple[str, tuple[str, ...]]]] = {
    KAIRO: ("kairo", ("kairo-round2.md",)),
    "r1": ("r1", ("r1-model-selection.md",)),
    "r1 without its gates": ("r1", ("r1-model-selection.md",)),
    "r1 reading only bars": ("r1", ("r1-model-selection.md",)),
    "r2": ("r2", ("r2-config-sweep.md",)),
    "r2 without its gates": ("r2", ("r2-config-sweep.md",)),
    "r2 with its gates read in each task": ("r2", ("r2-config-sweep.md",)),
    "r2 with one gate": ("r2", ("r2-config-sweep.md",)),
    "r3": ("r3", ("r3-classifier.md",)),
    "r3 without its configs": ("r3", ("r3-classifier.md",)),
    "r4": ("r4", ("r4-tradeoff.md",)),
    "r5": ("r5", ()),
}


def writer_host(*, settings: LaunchSettings = TOYHOST_LAUNCH_SETTINGS) -> OpsFixture:
    """The toy host with the writer's contract on its kinds and its measures on its registry, the writer template
    saved.

    :param settings: The host's launch settings.
    :ptype settings: LaunchSettings
    :return: The fixture.
    :rtype: OpsFixture
    """
    profile = toyhost_profile()
    profile = replace(
        profile,
        kinds=(*profile.kinds, writer_kind_contract(host_seats=("reviewer_pool",))),
        measures=MeasureRegistry((*TOYHOST_MEASURES, *WRITER_KIND_MEASURES), families=profile.measures.families),
    )
    fixture = ops_fixture(generation=False, profile=profile, settings=settings)
    fixture.host.eval_host.storage.save_template(WRITER_TEMPLATE)
    return fixture


async def seed_writer_cases(host: OpsHost, *, split: WriterSplit | None = None) -> dict[str, MintedWriterCase]:
    """Mint a writer case of every fixture into :data:`WRITER_TEMPLATE`, in :data:`SEED`'s order.

    :param host: The host, its writer template saved.
    :ptype host: OpsHost
    :param split: The sets the cases join; ``None`` for the default names and share.
    :ptype split: WriterSplit | None
    :return: Each fixture's receipt, by name.
    :rtype: dict[str, MintedWriterCase]
    """
    snapshots = fixture_snapshots(CALLER.scope_id)
    minted: dict[str, MintedWriterCase] = {}
    for name, (group, withhold) in SEED.items():
        restored = await report_snapshot_restore(host, CALLER.scope_id, snapshots[name])
        minted[name] = await writer_case_mint(
            host,
            WriterCaseMint(
                template_id=WRITER_TEMPLATE.id,
                snapshot_id=restored.snapshot_id,
                withhold=list(withhold),
                group=group,
                split=split or WriterSplit(),
            ),
            CALLER.scope_id,
            created_by=CALLER.identity,
        )
    return minted
