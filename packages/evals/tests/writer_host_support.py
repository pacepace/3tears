"""The toy host as the report writer's tests read it: the caller, the writer template and an operations host.

Shared by every test on the writer's path (rung 5's, the writer bank's, the writer harness's, the recorded writer's)
and by the export of rung 5's case files. The host is the toy host's own operations host
(:func:`~packages.evals.tests.fixtures.toyhost.ops.toyhost_ops_host`): the writer's kind contract and measures on its
profile, its report budget and its reading of a model's family, its template saved and its corpus campaign stored.
Its completion clients are the caller's: the writer's own models are handed in by the call that runs them (a turn model
to ``write_report``, a model factory to the writer's launchable kind), so a test passes clients only for a launch,
whose judge the host builds. Nothing here reaches the report planner, the memo generator or their fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS
from packages.evals.tests.fixtures.toyhost.ops import toyhost_ops_host
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from threetears.evals.actions import Caller
from threetears.evals.kernel import EvalCampaign, EvalStorage
from threetears.evals.kernel.host import CompletionClients
from threetears.evals.ops import OpsHost
from threetears.evals.ops.report_writer.writer_contract import WRITER_KIND
from threetears.evals.run import LaunchSettings
from threetears.evals.schema.models import EvalTemplate
from threetears.evals.storage import InMemoryDocumentStore

__all__ = ["CALLER", "WRITER_TEMPLATE", "WriterHost", "writer_host"]

#: Who the tests call as: the toy host's scope, under a name a written campaign records.
CALLER: Final = Caller(scope_id=TOYHOST_SCOPE, identity="agent:test")

#: The writer template every seeded case is minted into.
WRITER_TEMPLATE: Final = EvalTemplate(
    id="writer-template",
    scope_id=CALLER.scope_id,
    name="Report writer",
    intent="Write the report.",
    candidate_kind=WRITER_KIND,
    # Stamped, not read from the clock: a constant built at import reads no clock.
    created_at="2026-08-01T00:00:00+00:00",
    updated_at="2026-08-01T00:00:00+00:00",
)


@dataclass(frozen=True)
class WriterHost:
    """The operations host, and the corpus campaign stored in it.

    :ivar host: The host.
    :ivar campaign: The toy host's corpus campaign, which a case can be minted from.
    """

    host: OpsHost
    campaign: EvalCampaign


def writer_host(
    *, settings: LaunchSettings = TOYHOST_LAUNCH_SETTINGS, clients: CompletionClients | None = None
) -> WriterHost:
    """The toy host's operations host, its template and the writer template saved and its corpus campaign stored.

    :param settings: The host's launch settings.
    :ptype settings: LaunchSettings
    :param clients: The host's completion clients, the judge of a writer launch among them; ``None`` for none.
    :ptype clients: CompletionClients | None
    :return: The host and its campaign.
    :rtype: WriterHost
    """
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_template(toyhost_template())
    campaign, corpus = toyhost_campaign()
    for run in corpus.load_eval_runs(campaign.run_ids, campaign.scope_id):
        storage.save_eval_run(run)
        for result in corpus.query_eval_results_by_run(run.id, campaign.scope_id):
            storage.save_eval_result(result)
    storage.save_campaign(campaign)
    storage.save_template(WRITER_TEMPLATE)
    return WriterHost(toyhost_ops_host(storage=storage, clients=clients, settings=lambda: settings), campaign)
