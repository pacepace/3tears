"""The report writer smoke's dry mode: the live smoke's path, end to end, on scripted models that write nothing.

``fixtures/writer_cases/writer_smoke.py`` is run live by hand (#1009 chunk 06). Its dry mode walks the same launcher,
campaign and promotion read, so this test keeps that path from rotting between live runs: two arms over three held-out
cases, a campaign with the shipped skill as its control, and a promotion read that answers, undecided.
"""

from __future__ import annotations

from packages.evals.tests.fixtures.writer_cases.writer_smoke import CASES, smoke


async def test_the_dry_smoke_runs_both_arms_over_every_case_and_the_promotion_read_answers_undecided() -> None:
    found = await smoke(dry=True, cap_usd=5.0, out=None)

    assert set(found["arms"]) == {"shipped", "inert variant"}
    for arm in found["arms"].values():
        assert len(arm["results"]) == len(CASES)
        assert {result["termination"] for result in arm["results"]} == {"completed"}
        assert all("report_writer.ending" in result["host_measures"] for result in arm["results"])
    assert found["promotion"]["decision"] == "undecided"
    assert found["promotion"]["cases"] == len(CASES)
