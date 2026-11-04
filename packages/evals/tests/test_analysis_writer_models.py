"""A host's allowed analysis writers are checked before the first provider request (#644).

A model measured unable to write the strict analysis contract used to fail its first call, buy the one
repair, fail again and be refused: two billed calls for nothing. A host now declares the writers it allows
(``HostProfile.analysis_writer_models``), and a writer outside the list is refused with nothing spent:
no provider request, and for a requested model, no client built. An empty list is today's behaviour.
"""

from __future__ import annotations

import dataclasses

import pytest

from threetears.evals.kernel import ValidationFailedError
from threetears.evals.kernel.host import HostProfile
from threetears.evals.ops import analysis_estimate
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.ops_support import TOYHOST_SCOPE, generate, ops_fixture, settled
from packages.evals.tests.fixtured_client import MODEL
from packages.evals.tests.off_loop import off_loop

#: A writer the host has not measured, so does not allow.
UNLISTED = "vendor/unlisted-writer"


def _allowing(*models: str) -> HostProfile:
    return dataclasses.replace(toyhost_profile(), analysis_writer_models=models)


class TestAnalysisGenerate:
    async def test_a_requested_writer_the_host_does_not_allow_is_refused_with_no_request(self) -> None:
        fixture = off_loop(ops_fixture, profile=_allowing(MODEL))

        with pytest.raises(ValidationFailedError, match=f"allows '{MODEL}'"):
            await generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE, model=UNLISTED)

        assert fixture.writers == fixture.analysts == [], "no client was built, so no provider request could be made"

    async def test_an_allowed_writer_proceeds(self) -> None:
        fixture = off_loop(ops_fixture, profile=_allowing(MODEL))

        (job,) = (await generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE, model=MODEL)).jobs

        assert (await settled(fixture.host, job.job_id)).state == "completed"
        (analyst,) = fixture.analysts
        assert analyst.model_name == MODEL and set(analyst.sessions) == {"analyst", "editor-1"}

    async def test_the_hosts_default_writer_is_held_to_the_list_too(self) -> None:
        """Checked in the job, once its client names the default writer: the job ends refused, before any call."""
        fixture = off_loop(ops_fixture, profile=_allowing("vendor/some-other-writer"))

        (job,) = (await generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
        status = await settled(fixture.host, job.job_id)

        assert status.state == "failed" and status.detail is not None
        assert status.detail.startswith("refused before any call: ") and "default writer" in status.detail
        (analyst,) = fixture.analysts
        assert analyst.sessions == {}, "nothing was sent"
        assert analyst.released, "the client it built was released"

    async def test_an_estimate_gives_the_same_answer(self) -> None:
        fixture = off_loop(ops_fixture, profile=_allowing(MODEL))

        with pytest.raises(ValidationFailedError, match="analysis_writer_models"):
            await analysis_estimate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE, model=UNLISTED)

    async def test_a_host_with_no_list_allows_any_writer(self) -> None:
        fixture = off_loop(ops_fixture)

        (job,) = (await generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE, model=UNLISTED)).jobs

        assert (await settled(fixture.host, job.job_id)).state == "completed"


class TestTheHostsWriterList:
    """The one check every door asks: the host's own answer for a writer it does or does not allow."""

    def test_a_writer_the_host_does_not_allow_is_refused_before_the_first_call(self) -> None:
        refusal = _allowing(MODEL).analysis_writer_refusal(UNLISTED)

        assert refusal is not None and "nothing was spent" in refusal

    def test_an_allowed_writer_is_sent(self) -> None:
        assert _allowing(MODEL).analysis_writer_refusal(MODEL) is None
