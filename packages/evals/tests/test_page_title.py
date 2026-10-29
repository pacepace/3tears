"""A campaign's title heads it wherever a page heads it: the campaign listing, the analysis listing, and the first
line of the report page in every format.

Contracts held here (``threetears.evals.analysis.campaign_title``, ``threetears.evals.ops.analyses_list``,
``threetears.evals.ops.report_read``):

- **one title**: the title of a frozen bundle's arms equals the title of the campaign declaring them;
- **the report's head** is the title the analysis is listed by: the page's `<title>`, which a document's page prints
  (its first line is its headline, the title line of the campaign page is covered by test_campaign_first_screen).
"""

from __future__ import annotations

from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE, generate, ops_fixture, settled
from threetears.evals.analysis import assemble_campaign_bundle
from threetears.evals.analysis.campaign_title import campaign_title, declared_title
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.ops import analyses_list, report_read


def test_a_frozen_bundle_s_arms_title_a_campaign_as_the_campaign_declaring_them_is_titled() -> None:
    fixture = ops_fixture()
    host = fixture.host.eval_host
    stored = host.storage.load_campaign(fixture.campaign.id, TOYHOST_SCOPE)
    bundle = assemble_campaign_bundle(host, fixture.campaign.id, TOYHOST_SCOPE)
    assert stored is not None and len(bundle.arms) >= 2
    declared = stored.model_copy(update={"arms": list(bundle.arms)})

    assert declared_title(
        stored.name, bundle.arms, [(aspect.aspect, aspect.short_name) for aspect in bundle.aspects]
    ) == campaign_title(declared)


async def test_a_stored_analysis_is_listed_and_its_report_headed_by_its_campaign_s_title() -> None:
    fixture = off_loop(ops_fixture)
    (job,) = (await generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    await settled(fixture.host, job.job_id)
    host = fixture.host.eval_host

    def read() -> None:
        (listed,) = analyses_list(host, fixture.campaign.id, TOYHOST_SCOPE).analyses
        assert listed.title.endswith(f"({fixture.campaign.name})")
        html = report_read(host, fixture.campaign.id, TOYHOST_SCOPE, format="html").body
        assert f"<title>{listed.title}</title>" in html

    await run_blocking(host.blocking_executor, read)
