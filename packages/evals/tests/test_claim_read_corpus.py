"""A claim read on real labels: a host's broadcast claims, labelled by one person and drawn by that label for a second.

The fixture (``fixtures/claim_corpus``, CC-BY 4.0, Discodon / Kairo (Radio Discodon)) holds 954 claims an AI DJ made
in production eval runs, each with a first label, and a second label on 180 drawn by the first: every claim the first
called contradicted, every one it could not settle, and 60 of the 834 it supported. That is the claim read's own
design, stratified by the first label, so the first label stands as the instrument and the second as people, and the
reading's weighting turns the 180 back into the 954. People agree with each other well past the calibration bar, and
on 34 cells of eight runs, the floor of twenty cells is met.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.analysis.claim_read import read_claim_calibration
from threetears.evals.schema.claim_read import (
    STRATA_OF,
    ClaimAnswer,
    ClaimDraw,
    ClaimItem,
    ClaimItemView,
    ClaimRead,
    ClaimReadArm,
    ClaimStratumDraw,
    ClaimVerdict,
    item_view_digest,
)

_CORPUS = Path(__file__).resolve().parent / "fixtures" / "claim_corpus" / "claimset"

#: What the reading gives on the corpus, as ``read_claim_calibration`` read it when this test was written.
KAPPA = 0.7771
RECALL = 0.7792
PRECISION = 0.8108


def _lines(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (_CORPUS / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def corpus_read() -> ClaimRead:
    """The corpus as a claim read: each claim's stratum its first label, the second label's 180 as the drawn items, each
    drawn at its stratum's rate, and the second label as one person's answer to ``true``."""
    claims = _lines("claims.jsonl")
    second = {line["claim_id"]: line for line in _lines("second_labels.jsonl")}
    by_id = {claim["claim_id"]: claim for claim in claims}
    population = Counter(claim["label"] for claim in claims)
    by_arm: dict[str, Counter[str]] = {stratum: Counter() for stratum in STRATA_OF["truth"]}
    cells: dict[str, set[str]] = {stratum: set() for stratum in STRATA_OF["truth"]}
    for claim in claims:
        by_arm[claim["label"]][claim["run_id"]] += 1
        cells[claim["label"]].add(claim["result_id"])
    drawn = Counter(by_id[claim_id]["label"] for claim_id in second)
    items = []
    answers = []
    for claim_id, label in second.items():
        claim = by_id[claim_id]
        excerpt = claim["utterance_excerpt"]
        view = ClaimItemView(claim=claim["claim"], excerpt=excerpt, start=0, end=len(excerpt), segment=excerpt)
        items.append(
            ClaimItem(
                item=claim_id,
                stratum=claim["label"],
                arm=claim["run_id"],
                run_id=claim["run_id"],
                result_id=claim["result_id"],
                case_id=claim["case"],
                claim=0,
                verdict=claim["label"],
                inclusion=drawn[claim["label"]] / population[claim["label"]],
                view_digest=item_view_digest(view),
                view=view,
            )
        )
        answers.append(
            ClaimAnswer(
                item=claim_id,
                rater="second-labeller",
                verdicts=[ClaimVerdict(question="true", score=0 if label["label"] == "contradicted" else 1)],
            )
        )
    return ClaimRead(
        id="claimread-corpus",
        scope_id="corpus",
        name="broadcast claims, labelled twice",
        created_by="corpus",
        criterion="broadcast.claims",
        criterion_fingerprint="0" * 64,
        questions=["true"],
        arms=[ClaimReadArm(key=run, run_ids=[run]) for run in sorted({claim["run_id"] for claim in claims})],
        draw=ClaimDraw(
            seed=0,
            stratify_by="truth",
            # The corpus's second label took every claim of a stratum it read, up to eleven of one cell.
            per_cell=max(Counter(by_id[claim_id]["result_id"] for claim_id in second).values()),
            population_digest="corpus",
            strata=[
                ClaimStratumDraw(
                    stratum=stratum,
                    population=population[stratum],
                    cells=len(cells[stratum]),
                    planned=drawn[stratum],
                    drawn=drawn[stratum],
                    by_arm=dict(sorted(by_arm[stratum].items())),
                )
                for stratum in STRATA_OF["truth"]
            ],
        ),
        items=items,
        answers=answers,
    )


def test_two_people_agree_on_real_broadcast_claims_past_the_calibration_bar() -> None:
    (reading,) = read_claim_calibration(corpus_read()).questions

    assert (reading.items, reading.cells) == (180, 34)
    assert reading.kappa.value == pytest.approx(KAPPA, abs=0.005)
    assert reading.recall.value == pytest.approx(RECALL, abs=0.005)
    assert reading.precision.value == pytest.approx(PRECISION, abs=0.005)
    assert reading.kappa.lower is not None and reading.kappa.lower >= 0.6
    assert reading.check is not None and reading.check.result == "met"
    # The second labeller supported every one of the 60 claims the first supported: a unanimous stratum, which still
    # carries its spread into kappa's interval.
    supported = {s.stratum: s for s in reading.strata}["supported"]
    assert (supported.items, supported.agreed) == (60, 1.0)
    assert reading.kappa.upper is not None and reading.kappa.upper < 1.0
