"""A host's allowed analysis writers are checked before the first provider request (#644).

A model measured unable to write the strict analysis contract used to fail its first call, buy the one
repair, fail again and be refused: two billed calls for nothing. A host now declares the writers it allows
(``HostProfile.analysis_writer_models``), and a writer outside the list is refused with nothing spent:
no provider request, and for a requested model, no client built. An empty list is today's behaviour.
A reporter run measuring writers is exempt, since it is how a host learns which writers to list.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from threetears.evals.analysis.errors import GenerationError
from threetears.evals.analysis.generator import generate_analysis
from threetears.evals.contracts import ValidationFailedError
from threetears.evals.contracts.host import HostProfile
from threetears.evals.contracts.models import utc_now_iso
from threetears.evals.ops import analysis_estimate, analysis_generate
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.ops_support import TOYHOST_SCOPE, ops_fixture, settled
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient, memo_payload

#: A writer the host has not measured, so does not allow.
UNLISTED = "vendor/unlisted-writer"


def _allowing(*models: str) -> HostProfile:
    return dataclasses.replace(toyhost_profile(), analysis_writer_models=models)


class TestAnalysisGenerate:
    async def test_a_requested_writer_the_host_does_not_allow_is_refused_with_no_request(self) -> None:
        fixture = ops_fixture(profile=_allowing(MODEL))

        with pytest.raises(ValidationFailedError, match=f"allows '{MODEL}'"):
            await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE, model=UNLISTED)

        assert fixture.writers == [], "no client was built, so no provider request could be made"

    async def test_an_allowed_writer_proceeds(self) -> None:
        fixture = ops_fixture(profile=_allowing(MODEL))

        (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE, model=MODEL)).jobs

        assert (await settled(fixture.host, job.job_id)).state == "completed"
        assert len(fixture.writers[0].calls) == 1

    async def test_the_hosts_default_writer_is_held_to_the_list_too(self) -> None:
        fixture = ops_fixture(profile=_allowing("vendor/some-other-writer"))

        with pytest.raises(ValidationFailedError, match="default writer"):
            await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)

        (writer,) = fixture.writers
        assert writer.calls == [] and writer.priced == [], "nothing was sent or priced"
        assert writer.closed == 1, "the client it built was released"

    async def test_an_estimate_gives_the_same_answer(self) -> None:
        fixture = ops_fixture(profile=_allowing(MODEL))

        with pytest.raises(ValidationFailedError, match="analysis_writer_models"):
            await analysis_estimate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE, model=UNLISTED)

    async def test_a_host_with_no_list_allows_any_writer(self) -> None:
        fixture = ops_fixture()

        (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE, model=UNLISTED)).jobs

        assert (await settled(fixture.host, job.job_id)).state == "completed"


class TestGenerateAnalysis:
    async def _generate(self, profile: HostProfile, *, model: str, measuring_writers: bool = False) -> FixturedClient:
        bundle = toyhost_bundle(profile=profile)
        client = FixturedClient(json.dumps(memo_payload(bundle)))
        await generate_analysis(
            bundle,
            prompt=PROMPT,
            model=model,
            client=client,
            prompt_id=PROMPT_ID,
            bundle_assembled_at=utc_now_iso(),
            profile=profile,
            measuring_writers=measuring_writers,
        )
        return client

    async def test_a_writer_the_host_does_not_allow_is_refused_before_the_first_call(self) -> None:
        bundle = toyhost_bundle(profile=_allowing(MODEL))
        client = FixturedClient(json.dumps(memo_payload(bundle)))

        with pytest.raises(GenerationError, match="nothing was spent"):
            await generate_analysis(
                bundle,
                prompt=PROMPT,
                model=UNLISTED,
                client=client,
                prompt_id=PROMPT_ID,
                bundle_assembled_at=utc_now_iso(),
                profile=_allowing(MODEL),
            )

        assert client.calls == []

    async def test_an_allowed_writer_is_sent(self) -> None:
        assert len((await self._generate(_allowing(MODEL), model=MODEL)).calls) == 1

    async def test_a_reporter_run_measuring_writers_is_not_held_to_the_list(self) -> None:
        client = await self._generate(_allowing(MODEL), model=UNLISTED, measuring_writers=True)
        assert len(client.calls) == 1
